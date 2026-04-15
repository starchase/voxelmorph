import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialTransformer, IntegrateVelocityField

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


class DecoupledEncoder(nn.Module):
    """
    Dual-stream encoder for Siamese Network with Appearance Decoupling.
    Allows specifying the number of decoupled layers at the beginning.
    """
    def __init__(self, in_channels=1, enc_nf=[16, 32, 32, 32], ndim=3, decouple_layers=2):
        super().__init__()
        self.decouple_layers = decouple_layers
        
        self.enc_blocks_source = nn.ModuleList()
        self.enc_blocks_target = nn.ModuleList()
        self.shared_blocks = nn.ModuleList()
        
        prev_channels = in_channels
        
        for i, nf in enumerate(enc_nf):
            if i < decouple_layers:
                self.enc_blocks_source.append(ConvBlock(ndim, prev_channels, nf, stride=2))
                self.enc_blocks_target.append(ConvBlock(ndim, prev_channels, nf, stride=2))
            else:
                self.shared_blocks.append(ConvBlock(ndim, prev_channels, nf, stride=2))
            prev_channels = nf

    def forward(self, source, target):
        feat_s, feat_t = [], []
        x_s, x_t = source, target
        
        # 1. Decoupled forward pass
        for i in range(len(self.enc_blocks_source)):
            x_s = self.enc_blocks_source[i](x_s)
            x_t = self.enc_blocks_target[i](x_t)
            feat_s.append(x_s)
            feat_t.append(x_t)
            
        # 2. Shared forward pass
        for i in range(len(self.shared_blocks)):
            x_s = self.shared_blocks[i](x_s)
            x_t = self.shared_blocks[i](x_t)
            feat_s.append(x_s)
            feat_t.append(x_t)
            
        return feat_s, feat_t

class SiameseUNetBaseline(nn.Module):
    """
    Vanilla Siamese U-Net Baseline.
    - Shared Encoder
    - Standard Unet Decoder (No Coarse-to-fine sub-flows yet)
    - Concatenation-based Skip Connections (No Diff-Aware yet)
    - No Frequency Domain Alignment yet
    """
    def __init__(self, inshape, in_channels=1, enc_nf=[16, 32, 32, 32], dec_nf=[32, 32, 32, 16], ndim=3, int_steps=0, decouple_layers=2, use_daps=False):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps
        self.use_daps = use_daps

        # 1. Shared Encoder
        self.encoder = DecoupledEncoder(in_channels, enc_nf, ndim, decouple_layers=decouple_layers)
        
        # 2. Standard Decoder
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        if self.use_daps:
            self.coarse_flow_convs = nn.ModuleList()
            self.daps_stn = SpatialTransformer()
            Conv = getattr(nn, f'Conv{ndim}d')

        prev_channels = enc_nf[-1] * 2  # The very bottom layer merges source and target
        
        for i, nf in enumerate(dec_nf):
            # For ablation extensibility, we keep the decode path modular
            # Normal skip connection includes: Upsampled features + Source Skip + Target Skip
            skip_idx = len(enc_nf) - 2 - i
            
            if self.use_daps:
                skip_channels = enc_nf[skip_idx] * 3 if skip_idx >= 0 else 0
                coarse_conv_layer = Conv(prev_channels, ndim, kernel_size=3, padding=1)
                coarse_conv_layer.weight.data.normal_(0, 1e-5)
                coarse_conv_layer.bias.data.zero_()
                self.coarse_flow_convs.append(coarse_conv_layer)
            else:
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

    def forward(self, source, target, return_warped_source=True, return_field_type='displacement', return_coarse_flows=False):
        # --- [Ablation 1 Hook: FDA will go here] ---
        source_input = source
        target_input = target
        
        # 1. Feature Extraction (Decoupled/Shared)
        feat_s, feat_t = self.encoder(source_input, target_input)
        
        coarse_flows = []
        
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
                
                # --- [Ablation 2: DAPS (Deformation-Aware Progressive Skip)] ---
                if getattr(self, 'use_daps', False):
                    # a) Predict intermediate coarse flow
                    coarse_flow = self.coarse_flow_convs[i](x)
                    coarse_flows.append(coarse_flow)
                    if coarse_flow.shape[2:] != s_skip.shape[2:]:
                        coarse_flow = F.interpolate(coarse_flow, size=s_skip.shape[2:], mode=mode, align_corners=False)
                        
                    # b) Warp the source skip feature
                    s_skip_warped = self.daps_stn(s_skip, coarse_flow)
                    
                    # c) Calculate explicit absolute difference (Residual Error)
                    diff = torch.abs(s_skip_warped - t_skip)
                    
                    # d) Concat: [warped_source, target, difference]
                    skip_concat = torch.cat([s_skip_warped, t_skip, diff], dim=1)
                else:
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
             
        if return_coarse_flows:
             outputs.append(coarse_flows)
             
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
