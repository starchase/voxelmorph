import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialTransformer, IntegrateVelocityField

class ConvBlock(nn.Module):
    """
    A specific convolutional block for UNet.
    """
    def __init__(self, ndim, in_channels, out_channels, stride=1, use_norm=False):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        
        self.use_norm = use_norm
        if self.use_norm:
            Norm = getattr(nn, f'InstanceNorm{ndim}d')
            self.norm = Norm(out_channels)

    def forward(self, x):
        x = self.main(x)
        if self.use_norm:
            x = self.norm(x)
        return self.activation(x)

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
    def __init__(self, in_channels=1, enc_nf=[16, 32, 32, 32], ndim=3, decouple_layers=2, use_dsin=False):
        super().__init__()
        self.decouple_layers = decouple_layers
        self.use_dsin = use_dsin
        
        self.enc_blocks_source = nn.ModuleList()
        self.enc_blocks_target = nn.ModuleList()
        self.shared_blocks = nn.ModuleList()
        
        prev_channels = in_channels
        
        for i, nf in enumerate(enc_nf):
            # DSIN: Apply norm only to the decoupled shallow layers (or conditionally all)
            apply_norm = self.use_dsin and (i < decouple_layers)
            
            if i < decouple_layers:
                # Decoupled convolution weights + Decoupled (Domain-Specific) Instance Norms
                self.enc_blocks_source.append(ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm))
                self.enc_blocks_target.append(ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm))
            else:
                # Shared convolution weights, usually no norm here for cross-modal interactive consistency
                self.shared_blocks.append(ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=False))
            prev_channels = nf

    def forward(self, source, target):
        feat_s, feat_t = [], []
        x_s, x_t = source, target
        
        # 1. Decoupled forward pass (with DSIN if enabled)
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

class DAPS_PLR_Block(nn.Module):
    """
    Deformation-Aware Progressive Skip with Pyramid-Level Regularization (DAPS-PLR)
    This module predicts a local deformation sub-flow, warps the source skip feature,
    calculates the residual difference, and returns the concatenated feature along with
    the predicted sub-flow for pyramid-level regularization (PLR).
    """
    def __init__(self, ndim, in_channels):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        self.flow_conv = Conv(in_channels, ndim, kernel_size=3, padding=1)
        # Initialize sub-flow to identity (close to zero)
        self.flow_conv.weight.data.normal_(0, 1e-5)
        self.flow_conv.bias.data.zero_()
        self.stn = SpatialTransformer()
        self.ndim = ndim

    def forward(self, x, s_skip, t_skip):
        # a) Predict intermediate coarse flow
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        coarse_flow = self.flow_conv(x)
        
        # Keep original sub-flow shape for PLR Return, but interpolate for warping if needed
        warp_flow = coarse_flow
        if warp_flow.shape[2:] != s_skip.shape[2:]:
            warp_flow = F.interpolate(warp_flow, size=s_skip.shape[2:], mode=mode, align_corners=False)
            
        # b) Warp the source skip feature
        s_skip_warped = self.stn(s_skip, warp_flow)
        
        # c) Calculate explicit absolute difference (Residual Error Map)
        diff = torch.abs(s_skip_warped - t_skip)
        
        # d) Concat: [warped_source, target, difference]
        skip_concat = torch.cat([s_skip_warped, t_skip, diff], dim=1)
        
        return skip_concat, coarse_flow


class SiameseUNetBaseline(nn.Module):
    """
    Vanilla Siamese U-Net Baseline.
    - Shared Encoder
    - Standard Unet Decoder (No Coarse-to-fine sub-flows yet)
    - Concatenation-based Skip Connections (No Diff-Aware yet)
    - No Frequency Domain Alignment yet
    """
    def __init__(self, inshape, in_channels=1, enc_nf=[16, 32, 32, 32], dec_nf=[32, 32, 32, 16], ndim=3, int_steps=0, decouple_layers=2, use_daps=False, use_dsin=False, use_cmim=False):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps
        self.use_daps = use_daps
        self.use_dsin = use_dsin
        self.use_cmim = use_cmim

        # 1. Shared Encoder with pluggable DSIN support
        self.encoder = DecoupledEncoder(
            in_channels, enc_nf, ndim, 
            decouple_layers=decouple_layers, 
            use_dsin=use_dsin
        )
        
        # 1.5 Cross-Modal Interaction Module (CMIM) at 1/8 and 1/16 scales
        self.cmim_blocks = nn.ModuleList()
        if self.use_cmim:
            # Last layer (1/16 scale)
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-1]))
            # Second to last layer (1/8 scale)
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-2]))

        
        # 2. Standard Decoder
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        if self.use_daps:
            self.daps_plr_blocks = nn.ModuleList()

        prev_channels = enc_nf[-1] * 2  # The very bottom layer merges source and target
        
        for i, nf in enumerate(dec_nf):
            # For ablation extensibility, we keep the decode path modular
            # Normal skip connection includes: Upsampled features + Source Skip + Target Skip
            skip_idx = len(enc_nf) - 2 - i
            
            if self.use_daps:
                skip_channels = enc_nf[skip_idx] * 3 if skip_idx >= 0 else 0
                if skip_idx >= 0:
                    daps_block = DAPS_PLR_Block(ndim, prev_channels)
                    self.daps_plr_blocks.append(daps_block)
                else:
                    self.daps_plr_blocks.append(None)
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
        # Start from the bottom-most features (1/16 scale)
        if self.use_cmim:
            feat_s[-1] = self.cmim_blocks[0](feat_s[-1], feat_t[-1])
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
                
                # --- [CMIM at 1/8 Scale] ---
                if self.use_cmim and skip_idx == len(feat_s) - 2:
                    s_skip = self.cmim_blocks[1](s_skip, t_skip)
                
                # --- [Ablation 2: DAPS-PLR (Deformation-Aware Progressive Skip w/ Pyramid-Level Regularization)] ---
                if getattr(self, 'use_daps', False):
                    # Process through the DAPS-PLR block
                    skip_concat, coarse_flow = self.daps_plr_blocks[i](x, s_skip, t_skip)
                    coarse_flows.append(coarse_flow)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalInteractionModule(nn.Module):
    """
    Cross-Modal Interaction Module (CMIM)
    Uses 3D Multi-Head Cross-Attention to dynamically align features
    between Source and Target at deep layers (e.g., 1/8 and 1/16 scales)
    where the spatial dimensions are small enough to avoid memory explosion.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Linear projections for Query, Key, Value
        self.q_conv = nn.Conv3d(channels, channels, 1)
        self.k_conv = nn.Conv3d(channels, channels, 1)
        self.v_conv = nn.Conv3d(channels, channels, 1)
        
        self.out_conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        
    def forward(self, source, target):
        B, C, D, H, W = source.shape
        N = D * H * W
        
        # Target acts as Query (where should target look in source?)
        # Source acts as Key and Value
        # Reshape to (B, heads, N, C/heads)
        q = self.q_conv(target).view(B, self.num_heads, C // self.num_heads, N).transpose(-1, -2)
        k = self.k_conv(source).view(B, self.num_heads, C // self.num_heads, N)
        v = self.v_conv(source).view(B, self.num_heads, C // self.num_heads, N).transpose(-1, -2)
        
        # Scaled Dot-Product Attention: (B, heads, N, N)
        attn = torch.matmul(q, k) / (C // self.num_heads) ** 0.5
        attn = F.softmax(attn, dim=-1)
        
        # Output: (B, heads, N, C/heads) -> (B, C, D, H, W)
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        
        # Residual connection + norm. 
        # Since it's Target Querying Source, the output is aligned to Target geometry. 
        # We add it to Source to create a "Target-Aware Source Feature"
        out = self.out_conv(out)
        return self.norm(source + out)

