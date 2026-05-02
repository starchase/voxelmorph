import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialTransformer, IntegrateVelocityField
from .cross_mamba import CrossMambaModule
from .vss_mamba import VSSBlock3D

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

class ResidualMambaBlock(nn.Module):
    """
    Scientific Residual + Norm stabilized Mamba block.
    VSSBlock3D natively contains a residual connection, so we just
    stack them directly without applying norms over the residual sum,
    which would otherwise destroy the identity mapping.
    """
    def __init__(self, channels, num_blocks=1, scan_axes=('d',), gamma_init=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock3D(channels, scan_axes=scan_axes, gamma_init=gamma_init)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

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
    def __init__(self, in_channels=1, enc_nf=[16, 32, 32, 32], ndim=3, decouple_layers=2, use_dsin=False, encoder_type='cnn'):
        super().__init__()
        self.decouple_layers = decouple_layers
        self.use_dsin = use_dsin
        self.encoder_type = encoder_type
        
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
                if self.encoder_type == 'mamba' and i >= 2:
                    # Thick Deep Bottleneck: 1 block at 1/8 scale, 3 blocks at 1/16 scale
                    num_mamba = 3 if i == len(enc_nf) - 1 else 1
                    # Both 1/8 and 1/16 scales now use true 3D scanning
                    scan_axes = ('d', 'h', 'w')
                    self.shared_blocks.append(nn.Sequential(
                        ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=False),
                        ResidualMambaBlock(nf, num_blocks=num_mamba, scan_axes=scan_axes, gamma_init=0.2)
                    ))
                else:
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
        self.flow_conv.weight.data.normal_(0, 1e-6)
        self.flow_conv.bias.data.zero_()
        self.stn = SpatialTransformer()
        self.ndim = ndim

    def forward(self, x, s_skip, t_skip):
        # a) Predict intermediate coarse flow
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        coarse_flow = 20.0 * torch.tanh(self.flow_conv(x) / 20.0)
        
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



class StructureAwareWindowCostVolume3D(nn.Module):
    """
    Structure-Aware Window Cost Volume (S-WCV).
    Extracts high-frequency structural features (gradients/edges) to compute 
    correlation, making it robust to modality/intensity shifts.
    """
    def __init__(self, dim, window_size=7, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Fixed 3D difference kernels for spatial gradients
        # [Channels, 1, KD, KH, KW] for depth-wise convolution
        kernel_d = torch.tensor([[[-1., 0., 1.]]]).view(1, 1, 3, 1, 1).expand(dim, 1, 3, 1, 1)
        kernel_h = torch.tensor([[[-1.], [0.], [1.]]]).view(1, 1, 1, 3, 1).expand(dim, 1, 1, 3, 1)
        kernel_w = torch.tensor([[[-1., 0., 1.]]]).view(1, 1, 1, 1, 3).expand(dim, 1, 1, 1, 3)
        
        self.register_buffer('kernel_d', kernel_d)
        self.register_buffer('kernel_h', kernel_h)
        self.register_buffer('kernel_w', kernel_w)

        # Q and K now take original feature + structural gradient (dim * 2)
        self.q = nn.Linear(dim * 2, dim, bias=qkv_bias)
        self.k = nn.Linear(dim * 2, dim, bias=qkv_bias)
        
        self.cv_proj = nn.Sequential(
            nn.Conv3d(dim + 2 * num_heads, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        )
        self.norm = nn.InstanceNorm3d(dim)

    def extract_structure(self, x):
        # Apply 3D spatial gradients (finite differences)
        grad_d = F.conv3d(x, self.kernel_d, padding=(1, 0, 0), groups=self.dim)
        grad_h = F.conv3d(x, self.kernel_h, padding=(0, 1, 0), groups=self.dim)
        grad_w = F.conv3d(x, self.kernel_w, padding=(0, 0, 1), groups=self.dim)
        # Structural edge magnitude
        grad_mag = torch.sqrt(grad_d**2 + grad_h**2 + grad_w**2 + 1e-5)
        # Concat origin features with structure features -> (B, 2C, D, H, W)
        return torch.cat([x, grad_mag], dim=1)

    def forward(self, x_fixed, x_moving):
        orig_moving = x_moving
        
        # 1. Structural Extraction
        feat_fixed = self.extract_structure(x_fixed)
        feat_moving = self.extract_structure(x_moving)
        
        # 2. Geometry matching logic (same Windowing as S-WCV)
        B, C_feat, D, H, W = feat_moving.shape
        C = self.dim
        
        pad_d = (self.window_size - D % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            feat_fixed = F.pad(feat_fixed, (0, pad_w, 0, pad_h, 0, pad_d))
            feat_moving = F.pad(feat_moving, (0, pad_w, 0, pad_h, 0, pad_d))
            
        _, _, D_pad, H_pad, W_pad = feat_moving.shape
        
        fixed_windows = window_partition_3d(feat_fixed, self.window_size)
        moving_windows = window_partition_3d(feat_moving, self.window_size)
        
        fixed_windows = fixed_windows.view(-1, self.window_size**3, C_feat)
        moving_windows = moving_windows.view(-1, self.window_size**3, C_feat)
        
        N_w = fixed_windows.shape[0]
        
        # Structural Query and Key
        q = self.q(moving_windows).reshape(N_w, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(fixed_windows).reshape(N_w, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q_norm = torch.clamp(q.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        q = q / q_norm
        k_norm = torch.clamp(k.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        k = k / k_norm

        corr = (q @ k.transpose(-2, -1)) 
        
        cv_max, _ = corr.max(dim=-1)
        cv_mean = corr.mean(dim=-1)
        
        cv_feat = torch.cat([cv_max, cv_mean], dim=1)
        cv_feat = cv_feat.transpose(1, 2)
        
        cv_img = window_reverse_3d(cv_feat, self.window_size, D_pad, H_pad, W_pad)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            cv_img = cv_img[:, :, :D, :H, :W]
            
        fused = torch.cat([orig_moving, cv_img], dim=1)
        fused = self.cv_proj(fused)
            
        return self.norm(orig_moving + fused)

class SiameseUNetBaseline(nn.Module):
    """
    Vanilla Siamese U-Net Baseline.
    - Shared Encoder
    - Standard Unet Decoder (No Coarse-to-fine sub-flows yet)
    - Concatenation-based Skip Connections (No Diff-Aware yet)
    - No Frequency Domain Alignment yet
    """
    def __init__(self, inshape, in_channels=1, enc_nf=[16, 32, 32, 32], dec_nf=[32, 32, 32, 16], ndim=3, int_steps=0, decouple_layers=2, use_daps=False, use_pdaps=False, use_dsin=False, use_cmim=False, use_cross_mamba=False, use_wcv=False,
                 use_swcv=False, use_gcv=False, encoder_type='cnn'):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps
        self.use_daps = use_daps
        self.use_pdaps = use_pdaps
        self.use_dsin = use_dsin
        self.use_cmim = use_cmim
        self.use_cross_mamba = use_cross_mamba
        self.use_wcv = use_wcv
        self.use_swcv = use_swcv
        self.use_gcv = use_gcv
        self.encoder_type = encoder_type
        
        # --- [Architectural Refactoring] ---
        # Note: P-DAPS natively encapsulates coarse-to-fine deformation (previously isolated as 'pyramid').

        # 1. Shared Encoder with pluggable DSIN support
        self.encoder = DecoupledEncoder(
            in_channels, enc_nf, ndim, 
            decouple_layers=decouple_layers, 
            use_dsin=use_dsin,
            encoder_type=encoder_type
        )
        
        # 1.5 Cross-Modal Interaction Module
        # CMIM: apply at 1/8 and 1/16 scales
        # Cross-Mamba: apply to resolutions 1/16(idx=1) and 1/8(idx=0)
        self.cmim_blocks = nn.ModuleList()
        if self.use_cmim:
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-1]))
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-2]))
        elif self.use_cross_mamba:
            self.cmim_blocks.append(CrossMambaModule(enc_nf[-1])) # 1/16 bottleneck
            self.cmim_blocks.append(CrossMambaModule(enc_nf[-2])) # 1/8 scale
            
        # 1.6 Window Cost Volume (WCV) at shallow scales
        
        # 1.7 Global Cost Volume (GCV) at deep scales
        self.gcv_blocks = nn.ModuleDict()
        if self.use_gcv:
            # 1/16 scale (bottleneck)
            self.gcv_blocks["bottleneck"] = GlobalCostVolume3D(enc_nf[-1])
            # 1/8 scale, skip_idx=2
            self.gcv_blocks["2"] = GlobalCostVolume3D(enc_nf[-2])

        self.wcv_blocks = nn.ModuleDict()
        if self.use_wcv:
            # Deep Bottleneck (1/16 scale)
            self.wcv_blocks["bottleneck"] = WindowCostVolume3D(enc_nf[-1], window_size=7)
            # Deep skip connection (1/8 scale, skip_idx=2)
            self.wcv_blocks["2"] = WindowCostVolume3D(enc_nf[-2], window_size=7)

        self.swcv_blocks = nn.ModuleDict()
        if self.use_swcv:
            # Deep Bottleneck (1/16 scale)
            self.swcv_blocks["bottleneck"] = StructureAwareWindowCostVolume3D(enc_nf[-1], window_size=7)
            # Deep skip connection (1/8 scale, skip_idx=2)
            self.swcv_blocks["2"] = StructureAwareWindowCostVolume3D(enc_nf[-2], window_size=7)

        
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
            
            # Asymmetric Decoder: Inject 1 Mamba block at the very first decoder stage (1/8 scale)
            # to smoothly transition global topology into the CNN reconstruction pipeline.
            if self.encoder_type == 'mamba' and i == 0:
                self.dec_blocks.append(nn.Sequential(
                    ConvBlock(ndim, in_ch, nf, stride=1),
                    ResidualMambaBlock(nf, num_blocks=1, scan_axes=('d', 'h', 'w'), gamma_init=0.2)
                ))
            else:
                self.dec_blocks.append(ConvBlock(ndim, in_ch, nf, stride=1))
                
            prev_channels = nf
            
        # 3. Final Flow Prediction (Only at full resolution)
        Conv = getattr(nn, f'Conv{ndim}d')
        # We might need a couple of extra convolutions to reach native resolution 
        # because the encoder downsamples 4 times but decoder currently processes upsampled features.
        self.flow_conv = Conv(dec_nf[-1], ndim, kernel_size=3, padding=1)
        
        # Initialize flow weights to very small values
        self.flow_conv.weight.data.normal_(0, 1e-6)
        self.flow_conv.bias.data.zero_()

        # --- [P-DAPS Coarse-to-fine Flows] ---
        if self.use_pdaps:
            self.pyramid_flows = nn.ModuleList()
            pdaps_limits = []
            for nf in dec_nf:
                p_flow_conv = Conv(nf, ndim, kernel_size=3, padding=1)
                # Extremely small initialization is required to start with an identity transform
                p_flow_conv.weight.data.normal_(0, 1e-7)
                p_flow_conv.bias.data.zero_()
                self.pyramid_flows.append(p_flow_conv)
            for i in range(len(dec_nf)):
                pdaps_limits.append(2.0 / (2 ** (len(dec_nf) - i - 1)))
            self.register_buffer('pdaps_flow_limits', torch.tensor(pdaps_limits, dtype=torch.float32), persistent=False)

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
        if self.use_cmim or getattr(self, 'use_cross_mamba', False):
            feat_s[-1] = self.cmim_blocks[0](feat_s[-1], feat_t[-1])
        if getattr(self, 'use_wcv', False) and "bottleneck" in self.wcv_blocks:
            feat_s[-1] = self.wcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])
        if getattr(self, 'use_swcv', False) and "bottleneck" in self.swcv_blocks:
            feat_s[-1] = self.swcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])
        if getattr(self, 'use_gcv', False) and "bottleneck" in self.gcv_blocks:
            feat_s[-1] = self.gcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])

            
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
                
                # --- [Cross-Mamba at multi scales] ---
                if getattr(self, 'use_cross_mamba', False):
                    if skip_idx == len(feat_s) - 2:   # 1/8 scale
                        s_skip = self.cmim_blocks[1](s_skip, t_skip)
                
                # --- [GCV at Deep Scales] ---
                if getattr(self, 'use_gcv', False) and str(skip_idx) in self.gcv_blocks:
                    s_skip = self.gcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)
                
                # --- [Ablation 2: P-DAPS (Pyramid Deformation-Aware Progressive Skip) + Structure Matching] ---
                # NOTE: P-DAPS essentially absorbs the traditional feature pyramid.
                if getattr(self, 'use_pdaps', False):
                    # P-DAPS: Use the explicit pyramid flow from the previous layer to warp s_skip
                    if pyramid_acc_flow is not None:
                        # Upsample previous flow to current skip connection resolution
                        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
                        flow_up = F.interpolate(pyramid_acc_flow, size=s_skip.shape[2:], mode=mode, align_corners=False)
                        flow_up = flow_up * 2.0  # Scale magnitude since resolution doubled
                        
                        # Fix: integrate velocity field to displacement field before warping features
                        if self.integrate is not None:
                            disp_up = self.integrate(flow_up)
                        else:
                            disp_up = flow_up
                            
                        s_skip_warped = self.spatial_transform(s_skip, disp_up)
                        
                        # --- [Warp-then-Match: Apply S-WCV / WCV on aligned features] ---
                        if getattr(self, 'use_wcv', False) and str(skip_idx) in self.wcv_blocks:
                            s_skip_warped = self.wcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip_warped)
                        if getattr(self, 'use_swcv', False) and str(skip_idx) in self.swcv_blocks:
                            s_skip_warped = self.swcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip_warped)
                            
                        # Calculate diff and concat mapping explicit error
                        diff = torch.abs(s_skip_warped - t_skip)
                        skip_concat = torch.cat([s_skip_warped, t_skip, diff], dim=1)
                    else:
                        # Top-most layer (1/16 scale doesn't have a previous flow)
                        if getattr(self, 'use_wcv', False) and str(skip_idx) in self.wcv_blocks:
                            s_skip = self.wcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)
                        if getattr(self, 'use_swcv', False) and str(skip_idx) in self.swcv_blocks:
                            s_skip = self.swcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)
                            
                        diff = torch.abs(s_skip - t_skip)
                        skip_concat = torch.cat([s_skip, t_skip, diff], dim=1)
                        
                elif getattr(self, 'use_daps', False):
                    # Original DAPS-PLR block
                    skip_concat, coarse_flow = self.daps_plr_blocks[i](x, s_skip, t_skip)
                    coarse_flows.append(coarse_flow)
                else:
                    # --- [Independent Match (No P-DAPS)] ---
                    if getattr(self, 'use_wcv', False) and str(skip_idx) in self.wcv_blocks:
                        s_skip = self.wcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)
                    if getattr(self, 'use_swcv', False) and str(skip_idx) in self.swcv_blocks:
                        s_skip = self.swcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)
                        
                    skip_concat = torch.cat([s_skip, t_skip], dim=1)
                    
                x = torch.cat([x, skip_concat], dim=1)
                
            x = block(x)

            # --- [P-DAPS Coarse-to-fine Flow generation & Deep Supervision] ---
            if getattr(self, 'use_pdaps', False):
                sub_flow_limit = self.pdaps_flow_limits[i].to(dtype=x.dtype)
                raw_sub_flow = self.pyramid_flows[i](x)
                sub_flow = sub_flow_limit * torch.tanh(raw_sub_flow / sub_flow_limit)
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
        if getattr(self, 'use_pdaps', False):
            velocity = pyramid_acc_flow
        else:
            velocity = 20.0 * torch.tanh(self.flow_conv(x) / 20.0)
        
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
        self.fuse_conv = nn.Conv3d(channels * 2, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        
    def forward(self, source, target):
        B, C, D, H, W = source.shape
        N = D * H * W
        
        # [CRITICAL GEOMETRY FIX]
        # For the Source feature branch, we MUST maintain the Source coordinate grid.
        # Query = Source (Look outwards from the Source grid)
        # Key/Value = Target (Search for matching patterns in the Target grid)
        # This returns Target features aligned to the Source grid!
        q = self.q_conv(source).view(B, self.num_heads, C // self.num_heads, N).transpose(-1, -2)
        k = self.k_conv(target).view(B, self.num_heads, C // self.num_heads, N)
        v = self.v_conv(target).view(B, self.num_heads, C // self.num_heads, N).transpose(-1, -2)
        
        # Scaled Dot-Product Attention: (B, heads, N, N)
        # attn matrix: Source_pixels -> Target_pixels
        attn = torch.matmul(q, k) / (C // self.num_heads) ** 0.5
        attn = torch.nn.functional.softmax(attn, dim=-1)
        
        # Output geometry = Source! (Target features mapped to Source grid)
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        
        # Now out and source are strictly in the SAME spatial coordinates (Source)
        # We safely fuse them. Target is NOT concatenated here because it lives in a different spatial grid.
        out = self.out_conv(out)
        fused = self.fuse_conv(torch.cat([source, out], dim=1))
        
        return self.norm(fused + source)

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


class GlobalCostVolume3D(nn.Module):
    """
    Global Cost Volume: Computes explicit All-to-All correlation map.
    Returns matched features based on maximum and mean correlation across the entire image.
    """
    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.cv_proj = nn.Sequential(
            nn.Conv3d(dim + 2 * num_heads, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        )
        self.norm = nn.InstanceNorm3d(dim)

    def forward(self, x_fixed, x_moving):
        orig_moving = x_moving
        B, C, D, H, W = x_moving.shape
        N = D * H * W
        
        q = self.q(x_moving.view(B, C, N).transpose(1, 2)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(x_fixed.view(B, C, N).transpose(1, 2)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # Cosine Similarity Context: mathematically bound the correlation map exactly to [-1, 1] to prevent ANY numerical cascade from Mamba
        q_norm = torch.clamp(q.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        q = q / q_norm
        k_norm = torch.clamp(k.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        k = k / k_norm
        
        # Explicit Correlation Matrix (B, num_heads, N, N)
        corr = (q @ k.transpose(-2, -1)) # No scale needed for cosine similarity
        
        cv_max, _ = corr.max(dim=-1) # (B, num_heads, N)
        cv_mean = corr.mean(dim=-1)  # (B, num_heads, N)
        
        cv_feat = torch.cat([cv_max, cv_mean], dim=1) # (B, 2*num_heads, N)
        cv_img = cv_feat.view(B, 2*self.num_heads, D, H, W)
        
        fused = torch.cat([orig_moving, cv_img], dim=1)
        fused = self.cv_proj(fused)
            
        return self.norm(orig_moving + fused)

class WindowCostVolume3D(nn.Module):
    """
    State-Space Cost Volume (SSM-CV): Explicit Correlation Volume.
    Instead of multiplying by V (which causes feature teleportation), 
    we aggregate the correlation matrix (Q @ K^T) into a dense mismatch heatmap.
    """
    def __init__(self, dim, window_size=7, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Compress the max and mean similarities into original channel dimension
        self.cv_proj = nn.Sequential(
            nn.Conv3d(dim + 2 * num_heads, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        )
        self.norm = nn.InstanceNorm3d(dim)

    def forward(self, x_fixed, x_moving):
        orig_moving = x_moving
        B, C, D, H, W = x_moving.shape
        
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
        
        N_w = moving_windows.shape[0] 
        
        # Source/Moving is Query
        q = self.q(moving_windows).reshape(N_w, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # Target/Fixed is Key
        k = self.k(fixed_windows).reshape(N_w, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Cosine Similarity Bounds for Window Cost Volume
        q_norm = torch.clamp(q.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        q = q / q_norm
        k_norm = torch.clamp(k.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        k = k / k_norm

        # Explicit Correlation Matrix (NO softmax, NO V multiplication)
        # Shape: [N_w, num_heads, W^3, W^3]
        corr = (q @ k.transpose(-2, -1)) # No scale needed for cosine
        
        # Aggregate mismatch heatmap statistics (Max correlation and Mean correlation)
        # Max shows the best matching point, Mean shows global contextual confidence
        cv_max, _ = corr.max(dim=-1) # [N_w, num_heads, W^3]
        cv_mean = corr.mean(dim=-1)  # [N_w, num_heads, W^3]
        
        cv_feat = torch.cat([cv_max, cv_mean], dim=1) # [N_w, 2*num_heads, W^3]
        cv_feat = cv_feat.transpose(1, 2) # [N_w, W^3, 2*num_heads]
        
        cv_img = window_reverse_3d(cv_feat, self.window_size, D_pad, H_pad, W_pad)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            cv_img = cv_img[:, :, :D, :H, :W]
            
        # Concatenate Cost Volume Heatmap with original Source Features
        fused = torch.cat([orig_moving, cv_img], dim=1)
        fused = self.cv_proj(fused)
            
        return self.norm(orig_moving + fused)
