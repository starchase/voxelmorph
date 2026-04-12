import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialTransformer, IntegrateVelocityField
from .adaptive_fda import AdaptiveFDA3D

class ConvBlock(nn.Module):
    """
    A specific convolutional block for UNet.
    """
    def __init__(self, ndim, in_channels, out_channels, stride=1):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))

class SharedEncoder(nn.Module):
    """
    Shared dual-stream encoder for Siamese Network.
    Extracts features identically for both Source and Target.
    """
    def __init__(self, in_channels=1, enc_nf=[16, 32, 32, 32], ndim=3):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        prev_channels = in_channels
        
        for nf in enc_nf:
            self.enc_blocks.append(ConvBlock(ndim, prev_channels, nf, stride=2))
            prev_channels = nf

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        return features

class SiameseUNetBaseline(nn.Module):
    """
    Vanilla Siamese U-Net Baseline with optional Adaptive FDA.

    Architecture overview
    ---------------------
    - Shared Encoder (Siamese)
    - Standard UNet Decoder (concatenation-based skip connections)
    - Optional Adaptive FDA module at the input stage (Ablation 1)

    Parameters
    ----------
    inshape : tuple
        Spatial shape of the input volume, e.g. ``(D, H, W)``.
    in_channels : int
        Number of input image channels.
    enc_nf : list of int
        Number of feature maps at each encoder level.
    dec_nf : list of int
        Number of feature maps at each decoder level.
    ndim : int
        Number of spatial dimensions (3 for 3-D volumes).
    int_steps : int
        Number of integration steps for diffeomorphic registration.
        ``0`` disables diffeomorphic integration (SVF).
    use_fda : bool
        Whether to prepend an :class:`~voxelmorph.nn.adaptive_fda.AdaptiveFDA3D`
        module before the encoder.  When ``True``, the source image is adapted
        towards the target's low-frequency appearance before feature extraction.
    fda_beta_init : float
        Initial beta bandwidth for the FDA module (only used when
        ``use_fda=True``).
    fda_mode : str
        FDA operating mode: ``'fixed'``, ``'learnable'``, or ``'adaptive'``.
        See :class:`~voxelmorph.nn.adaptive_fda.AdaptiveFDA3D` for details.
    """

    def __init__(
        self,
        inshape,
        in_channels=1,
        enc_nf=[16, 32, 32, 32],
        dec_nf=[32, 32, 32, 16],
        ndim=3,
        int_steps=0,
        use_fda: bool = False,
        fda_beta_init: float = 0.1,
        fda_mode: str = "learnable",
    ):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps

        # --- [Ablation 1: Adaptive FDA] ---
        # Decoupled as a stand-alone module so it can be removed cleanly for
        # ablation experiments without touching the rest of the network.
        self.fda = (
            AdaptiveFDA3D(
                beta_init=fda_beta_init,
                mode=fda_mode,
                in_channels=in_channels,
            )
            if use_fda
            else None
        )

        # 1. Shared Encoder
        self.encoder = SharedEncoder(in_channels, enc_nf, ndim)
        
        # 2. Standard Decoder
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        prev_channels = enc_nf[-1] * 2  # The very bottom layer merges source and target
        
        for i, nf in enumerate(dec_nf):
            # For ablation extensibility, we keep the decode path modular
            # Normal skip connection includes: Upsampled features + Source Skip + Target Skip
            skip_idx = len(enc_nf) - 2 - i
            skip_channels = enc_nf[skip_idx] * 2 if skip_idx >= 0 else 0
            
            in_ch = prev_channels + skip_channels
            
            self.dec_blocks.append(ConvBlock(ndim, in_ch, nf, stride=1))
            prev_channels = nf
            
        # 3. Final Flow Prediction (Only at full resolution)
        Conv = getattr(nn, f'Conv{ndim}d')
        # We might need a couple of extra convolutions to reach native resolution 
        # because the encoder downsamples 4 times but decoder currently processes upsampled features.
        self.flow_conv = Conv(dec_nf[-1], ndim, kernel_size=3, padding=1)
        
        # Initialize flow weights to very small values
        self.flow_conv.weight.data.normal_(0, 1e-5)
        self.flow_conv.bias.data.zero_()

        # 4. Spatial Transformer
        self.spatial_transform = SpatialTransformer()
        if self.int_steps > 0:
            self.integrate = IntegrateVelocityField(steps=self.int_steps)
        else:
            self.integrate = None

    def forward(self, source, target, return_warped_source=True, return_field_type='displacement'):
        # --- [Ablation 1: Adaptive FDA] ---
        # Adapt source appearance towards target in the Fourier domain.
        # When self.fda is None (use_fda=False) this is a no-op, giving the
        # vanilla baseline.  The original source tensor is kept for warping
        # at the end so that the output image is in the original source space.
        if self.fda is not None:
            source_input = self.fda(source, target)
        else:
            source_input = source
        target_input = target
        
        # 1. Feature Extraction (Shared)
        feat_s = self.encoder(source_input)
        feat_t = self.encoder(target_input)
        
        # 2. Decoding (Standard U-Net Upsampling)
        # Start from the bottom-most features
        x = torch.cat([feat_s[-1], feat_t[-1]], dim=1)
        
        for i, block in enumerate(self.dec_blocks):
            # Upsample
            mode = 'trilinear' if self.ndim == 3 else 'bilinear'
            x = F.interpolate(x, scale_factor=2.0, mode=mode, align_corners=False)
            
            # Skip connections
            skip_idx = len(feat_s) - 2 - i
            if skip_idx >= 0:
                s_skip = feat_s[skip_idx]
                t_skip = feat_t[skip_idx]
                # --- [Ablation 2 Hook: Diff-Aware Skip will go here] ---
                skip_concat = torch.cat([s_skip, t_skip], dim=1) 
                x = torch.cat([x, skip_concat], dim=1)
                
            x = block(x)

        # 3. Flow prediction (Only at full resolution)
        # --- [Ablation 3 Hook: Coarse-to-fine FPN handling will replace this] ---
        velocity = self.flow_conv(x)
        
        if self.integrate is not None:
            displacement = self.integrate(velocity)
        else:
            displacement = velocity
            
        outputs = []
        if return_field_type == 'displacement':
            outputs.append(displacement)
        else:
            outputs.append(velocity)
            
        if return_warped_source:
             outputs.append(self.spatial_transform(source, displacement))
             
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
