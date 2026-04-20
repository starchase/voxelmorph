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
    def __init__(self, inshape, in_channels=1, enc_nf=[16, 32, 32, 32], dec_nf=[32, 32, 32, 16], ndim=3, int_steps=0, decouple_layers=2, use_daps=False, use_pdaps=False, use_dsin=False, use_cmim=False, use_wmca=False, use_pyramid=False):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps
        self.use_daps = use_daps
        self.use_pdaps = use_pdaps
        self.use_dsin = use_dsin
        self.use_cmim = use_cmim
        self.use_wmca = use_wmca
        self.use_pyramid = use_pyramid

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
            
        # 1.6 Window Cross-Attention (W-MCA) at 1/2 and 1/4 scales
        self.wmca_blocks = nn.ModuleDict()
        if self.use_wmca:
            # 只在 1/4 (skip_idx=1) 添加，1/2 尺寸实在太大导致 OOM
            self.wmca_blocks["1"] = WindowCrossAttention3D(enc_nf[1], window_size=7)

        
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
            elif getattr(self, 'use_pdaps', False):
                # P-DAPS provides [s_skip_warped, t_skip, diff]
                skip_channels = enc_nf[skip_idx] * 3 if skip_idx >= 0 else 0
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

        # --- [Pyramid Coarse-to-fine Flows] ---
        if self.use_pyramid:
            self.pyramid_flows = nn.ModuleList()
            for nf in dec_nf:
                p_flow_conv = Conv(nf, ndim, kernel_size=3, padding=1)
                p_flow_conv.weight.data.normal_(0, 1e-5)
                p_flow_conv.bias.data.zero_()
                self.pyramid_flows.append(p_flow_conv)

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
        pyramid_acc_flow = None
        
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
                
                # --- [W-MCA at 1/4 and 1/2 Scales] ---
                if getattr(self, 'use_wmca', False) and str(skip_idx) in self.wmca_blocks:
                    s_skip = self.wmca_blocks[str(skip_idx)](t_skip, s_skip)
                
                # --- [Ablation 2: DAPS-PLR / P-DAPS (Deformation-Aware Progressive Skip)] ---
                if getattr(self, 'use_pdaps', False) and getattr(self, 'use_pyramid', False):
                    # P-DAPS: Use the explicit pyramid flow from the previous layer to warp s_skip
                    if pyramid_acc_flow is not None:
                        # Upsample previous flow to current skip connection resolution
                        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
                        flow_up = F.interpolate(pyramid_acc_flow, size=s_skip.shape[2:], mode=mode, align_corners=False)
                        flow_up = flow_up * 2.0  # Scale magnitude since resolution doubled
                        s_skip_warped = self.spatial_transform(s_skip, flow_up)
                        diff = torch.abs(s_skip_warped - t_skip)
                        skip_concat = torch.cat([s_skip_warped, t_skip, diff], dim=1)
                    else:
                        # Top-most layer (1/16 scale doesn't have a previous flow)
                        diff = torch.abs(s_skip - t_skip)
                        skip_concat = torch.cat([s_skip, t_skip, diff], dim=1)
                elif getattr(self, 'use_daps', False):
                    # Original DAPS-PLR block
                    skip_concat, coarse_flow = self.daps_plr_blocks[i](x, s_skip, t_skip)
                    coarse_flows.append(coarse_flow)
                else:
                    skip_concat = torch.cat([s_skip, t_skip], dim=1)
                    
                x = torch.cat([x, skip_concat], dim=1)
                
            x = block(x)

            # --- [Pyramid Coarse-to-fine generation & Deep Supervision] ---
            if getattr(self, 'use_pyramid', False):
                sub_flow = self.pyramid_flows[i](x)
                if pyramid_acc_flow is None:
                    pyramid_acc_flow = sub_flow
                else:
                    mode = 'trilinear' if self.ndim == 3 else 'bilinear'
                    up_flow = F.interpolate(pyramid_acc_flow, size=sub_flow.shape[2:], mode=mode, align_corners=False)
                    # Rescale magnitude since resolution doubled
                    up_flow = up_flow * 2.0
                    pyramid_acc_flow = up_flow + sub_flow
                
                # Exclude the final full resolution layer from coarse_flows list
                if i < len(self.dec_blocks) - 1:
                    coarse_flows.append(pyramid_acc_flow)

        # 3. Flow prediction (Only at full resolution)
        # --- [Ablation 3 Hook: Coarse-to-fine FPN handling will replace this] ---
        if getattr(self, 'use_pyramid', False):
            velocity = pyramid_acc_flow
        else:
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

def window_partition_3d(x, window_size):
    """
    Partition 3D feature map into non-overlapping windows.
    x: (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape
    x = x.view(B, C, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, window_size**3, C)
    return windows

def window_reverse_3d(windows, window_size, D, H, W):
    """
    Reverse the partition of 3D feature map from windows.
    """
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    C = windows.shape[-1]
    x = windows.view(B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, C, D, H, W)
    return x

class WindowCrossAttention3D(nn.Module):
    """
    Window-based Cross-Attention for high resolution feature alignment.
    Decoupled: Query comes from Fixed/Target, Key/Value comes from Moving/Source.
    """
    def __init__(self, dim, window_size=7, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.InstanceNorm3d(dim)

    def forward(self, x_fixed, x_moving):
        orig_moving = x_moving
        B, C, D, H, W = x_fixed.shape
        
        pad_d = (self.window_size - D % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_fixed = F.pad(x_fixed, (0, pad_w, 0, pad_h, 0, pad_d))
            x_moving = F.pad(x_moving, (0, pad_w, 0, pad_h, 0, pad_d))
            _, _, D_pad, H_pad, W_pad = x_fixed.shape
        else:
            D_pad, H_pad, W_pad = D, H, W

        fixed_windows = window_partition_3d(x_fixed, self.window_size) 
        moving_windows = window_partition_3d(x_moving, self.window_size)
        
        N_w = fixed_windows.shape[0] 
        q = self.q(fixed_windows).reshape(N_w, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(moving_windows).reshape(N_w, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N_w, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = window_reverse_3d(x, self.window_size, D_pad, H_pad, W_pad)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :, :D, :H, :W]
            
        # Residual connection to keep Original texture
        x = self.norm(orig_moving + x)
            
        return x
