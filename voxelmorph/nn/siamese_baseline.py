import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialTransformer, IntegrateVelocityField, FixedGradientMagnitude3D
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


class ModalityInvariantSparseCostVolume3D(nn.Module):
    """Local cosine cost volume summarized as offset and confidence cues."""

    def __init__(self, feature_channels, projection_channels=8, search_radius=2, temperature=0.1):
        super().__init__()
        if search_radius < 1:
            raise ValueError('search_radius must be at least 1')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.search_radius = int(search_radius)
        self.temperature = float(temperature)
        self.projection = nn.Sequential(
            nn.Conv3d(feature_channels, projection_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(projection_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
        offsets = [
            (d, h, w)
            for d in range(-self.search_radius, self.search_radius + 1)
            for h in range(-self.search_radius, self.search_radius + 1)
            for w in range(-self.search_radius, self.search_radius + 1)
        ]
        self.offset_tuples = offsets
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.float32), persistent=False)

    @staticmethod
    def _shift(feature, offset):
        d, h, w = offset
        pd, ph, pw = abs(d), abs(h), abs(w)
        padded = F.pad(feature, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)
        d0, h0, w0 = pd + d, ph + h, pw + w
        return padded[:, :, d0:d0 + feature.shape[2], h0:h0 + feature.shape[3], w0:w0 + feature.shape[4]]

    def forward(self, source_feature, target_feature):
        source = F.normalize(self.projection(source_feature).float(), dim=1, eps=1e-6)
        target = F.normalize(self.projection(target_feature).float(), dim=1, eps=1e-6)
        correlations = torch.cat([
            (source * self._shift(target, offset)).sum(dim=1, keepdim=True)
            for offset in self.offset_tuples
        ], dim=1)
        probabilities = torch.softmax(correlations / self.temperature, dim=1)
        offsets = self.offsets.to(device=probabilities.device, dtype=probabilities.dtype)
        expected_offset = torch.einsum('bndhw,nc->bcdhw', probabilities, offsets)
        confidence, _ = probabilities.max(dim=1, keepdim=True)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / max(float(torch.log(torch.tensor(probabilities.shape[1]))), 1e-6)
        peak_correlation = correlations.max(dim=1, keepdim=True).values
        return torch.cat([expected_offset, confidence, entropy, peak_correlation], dim=1).to(source_feature.dtype)


class UncertaintyAwareSelfSimilarityCorrespondence3D(nn.Module):
    """Modality-robust local correspondence from self-similarity descriptors."""

    def __init__(
        self,
        channels,
        feature_channels=None,
        search_radius=2,
        temperature=0.1,
        guidance_strength=0.5,
    ):
        super().__init__()
        feature_channels = int(feature_channels or channels)
        projection_channels = min(8, feature_channels)
        hidden_channels = max(channels // 2, 8)
        self.search_radius = int(search_radius)
        self.temperature = float(temperature)
        self.guidance_strength = float(guidance_strength)
        if self.search_radius < 1:
            raise ValueError('search_radius must be at least 1')
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        if self.guidance_strength < 0:
            raise ValueError('guidance_strength must be non-negative')

        self.shared_projection = nn.Sequential(
            nn.Conv3d(feature_channels, projection_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(projection_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
        self.guidance = nn.Sequential(
            nn.Conv3d(6, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
        )
        self.decoder_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
        )
        self.guidance[-1].weight.data.zero_()
        self.guidance[-1].bias.data.zero_()
        self.decoder_gate[-1].weight.data.zero_()
        self.decoder_gate[-1].bias.data.zero_()

    @staticmethod
    def _shift(feature, offset):
        dd, dh, dw = offset
        padded = F.pad(
            feature,
            (
                max(-dw, 0), max(dw, 0),
                max(-dh, 0), max(dh, 0),
                max(-dd, 0), max(dd, 0),
            ),
            mode='replicate',
        )
        d0, h0, w0 = max(dd, 0), max(dh, 0), max(dw, 0)
        depth, height, width = feature.shape[-3:]
        return padded[..., d0:d0 + depth, h0:h0 + height, w0:w0 + width]

    def _self_similarity(self, feature):
        feature = self.shared_projection(feature.float()).float()
        offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        descriptors = [
            (feature * self._shift(feature, offset)).mean(dim=1, keepdim=True)
            for offset in offsets
        ]
        descriptor = torch.cat(descriptors, dim=1)
        descriptor = descriptor - descriptor.mean(dim=1, keepdim=True)
        return F.normalize(descriptor, dim=1, eps=1e-6)

    def _correspondence_guidance(self, source_feature, target_feature):
        source = self._self_similarity(source_feature)
        target = self._self_similarity(target_feature)
        radius = self.search_radius
        offsets = [
            (dd, dh, dw)
            for dd in range(-radius, radius + 1)
            for dh in range(-radius, radius + 1)
            for dw in range(-radius, radius + 1)
        ]
        logits = torch.cat(
            [(source * self._shift(target, offset)).sum(dim=1, keepdim=True) for offset in offsets],
            dim=1,
        )
        probabilities = torch.softmax(logits / self.temperature, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / probabilities.new_tensor(float(len(offsets))).log()
        confidence = 1.0 - entropy
        peak = probabilities.amax(dim=1, keepdim=True)
        offset_tensor = source.new_tensor(offsets).transpose(0, 1).view(1, 3, -1, 1, 1, 1)
        expected_offset = (probabilities.unsqueeze(1) * offset_tensor).sum(dim=2) / float(radius)
        expected_offset = expected_offset * confidence
        match_score = (probabilities * logits).sum(dim=1, keepdim=True) * confidence
        return torch.cat([confidence, peak, match_score, expected_offset], dim=1)

    def forward(self, x, source_feature, target_feature):
        input_dtype = x.dtype
        # CUDA replication_pad3d and the correspondence softmax are not robust
        # in BF16/FP16. Keep USSC matching in FP32 under the surrounding AMP
        # context, then cast the residual back to the decoder dtype.
        device_type = source_feature.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            guidance = self._correspondence_guidance(source_feature.float(), target_feature.float())
            if guidance.shape[-3:] != x.shape[-3:]:
                guidance = F.interpolate(guidance, size=x.shape[-3:], mode='trilinear', align_corners=False)
            update = self.guidance(guidance.float())
            gate = torch.sigmoid(self.decoder_gate(x.float()))
        return (x.float() + self.guidance_strength * gate * update).to(dtype=input_dtype)


# Backward-compatible class alias for old checkpoints and imports.
CrossImageFrequencyConsistencyModulation3D = UncertaintyAwareSelfSimilarityCorrespondence3D


class StructureErrorMambaRefiner(nn.Module):
    """Low-resolution residual velocity refiner driven by structural error maps."""

    def __init__(self, ndim, in_channels, hidden_channels=16, scan_axes=('d',), flow_limit=1.0):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        Norm = getattr(nn, f'InstanceNorm{ndim}d')
        self.flow_limit = float(flow_limit)
        self.stem = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=3, padding=1),
            Norm(hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.context = ResidualMambaBlock(hidden_channels, num_blocks=1, scan_axes=scan_axes, gamma_init=0.1)
        self.local = nn.Sequential(
            Conv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            Norm(hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.delta = Conv(hidden_channels, ndim, kernel_size=3, padding=1)
        self.delta.weight.data.normal_(0, 1e-6)
        self.delta.bias.data.zero_()

    def forward(self, x):
        features = self.stem(x)
        features = self.context(features)
        features = self.local(features)
        delta = self.delta(features)
        if self.flow_limit > 0:
            limit = torch.as_tensor(self.flow_limit, dtype=delta.dtype, device=delta.device).clamp(min=1e-3)
            delta = limit * torch.tanh(delta / limit)
        return delta


class DeformationReliabilityFieldCalibration3D(nn.Module):
    """Calibrate unreliable local velocity residuals without image matching."""

    def __init__(self, ndim=3, hidden_channels=16, strength=0.5):
        super().__init__()
        if ndim != 3:
            raise ValueError('DeformationReliabilityFieldCalibration3D requires ndim=3')
        if strength < 0:
            raise ValueError('strength must be non-negative')
        self.strength = float(strength)
        descriptor_channels = 2 * ndim + 4
        self.reliability = nn.Sequential(
            nn.Conv3d(descriptor_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        self.reliability[-1].weight.data.zero_()
        self.reliability[-1].bias.data.zero_()

    @staticmethod
    def _diff(value, dim):
        difference = value.diff(dim=dim)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim) + 1] = 1
        return F.pad(difference, tuple(pad), mode='replicate')

    def forward(self, velocity):
        input_dtype = velocity.dtype
        velocity_float = velocity.float()
        smooth = F.avg_pool3d(velocity_float, kernel_size=3, stride=1, padding=1)
        residual = velocity_float - smooth

        du_d, du_h, du_w = [self._diff(velocity_float[:, 0:1], dim) for dim in (2, 3, 4)]
        dv_d, dv_h, dv_w = [self._diff(velocity_float[:, 1:2], dim) for dim in (2, 3, 4)]
        dw_d, dw_h, dw_w = [self._diff(velocity_float[:, 2:3], dim) for dim in (2, 3, 4)]
        divergence = du_d + dv_h + dw_w
        curl = torch.cat([dw_h - dv_w, du_w - dw_d, dv_d - du_h], dim=1)
        strain_magnitude = torch.sqrt(
            du_d.square() + dv_h.square() + dw_w.square()
            + 0.5 * (du_h + dv_d).square()
            + 0.5 * (du_w + dw_d).square()
            + 0.5 * (dv_w + dw_h).square()
            + 1e-6
        )
        curl_magnitude = torch.sqrt(curl.square().sum(dim=1, keepdim=True) + 1e-6)
        residual_magnitude = torch.sqrt(residual.square().sum(dim=1, keepdim=True) + 1e-6)
        descriptor = torch.cat(
            [
                velocity_float,
                residual,
                strain_magnitude,
                divergence.abs(),
                curl_magnitude,
                residual_magnitude,
            ],
            dim=1,
        )
        suppression = torch.tanh(self.reliability(descriptor))
        calibrated = velocity_float - self.strength * suppression * residual
        return calibrated.to(dtype=input_dtype)


class DecoderAdaptiveSkipRouting3D(nn.Module):
    """Conservatively route dual-stream skips using modality-robust structure."""

    def __init__(self, decoder_channels, skip_channels, reduction=4, strength=0.2):
        super().__init__()
        if reduction < 1:
            raise ValueError('DASR reduction must be positive')
        if strength < 0:
            raise ValueError('DASR strength must be non-negative')
        hidden_channels = max(skip_channels // reduction, 8)
        self.strength = float(strength)
        self.decoder_projection = nn.Conv3d(decoder_channels, skip_channels, kernel_size=1, bias=False)
        self.channel_router = nn.Sequential(
            nn.Conv3d(4, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, 2, kernel_size=1),
        )
        self.spatial_router = nn.Sequential(
            nn.Conv3d(5, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_channels, 2, kernel_size=1),
        )
        self.channel_router[-1].weight.data.zero_()
        self.channel_router[-1].bias.data.zero_()
        self.spatial_router[-1].weight.data.zero_()
        self.spatial_router[-1].bias.data.zero_()

    @staticmethod
    def _normalize(feature):
        return F.instance_norm(feature.float(), eps=1e-5)

    @staticmethod
    def _gradient_magnitude(feature):
        gradients = []
        for dim in (2, 3, 4):
            difference = feature.diff(dim=dim)
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - dim) + 1] = 1
            gradients.append(F.pad(difference, tuple(pad), mode='replicate'))
        return torch.sqrt(sum(gradient.square() for gradient in gradients) + 1e-6)

    def _compute_gates(self, decoder, source_skip, target_skip):
        decoder = self.decoder_projection(decoder.float())
        source_structure = self._gradient_magnitude(self._normalize(source_skip))
        target_structure = self._gradient_magnitude(self._normalize(target_skip))
        decoder_structure = self._gradient_magnitude(self._normalize(decoder))
        structure_disagreement = torch.abs(source_structure - target_structure)

        pooled = torch.cat(
            [
                F.adaptive_avg_pool3d(source_structure.mean(dim=1, keepdim=True), 1),
                F.adaptive_avg_pool3d(target_structure.mean(dim=1, keepdim=True), 1),
                F.adaptive_avg_pool3d(decoder_structure.mean(dim=1, keepdim=True), 1),
                F.adaptive_avg_pool3d(structure_disagreement.mean(dim=1, keepdim=True), 1),
            ],
            dim=1,
        )
        source_channel, target_channel = self.channel_router(pooled).chunk(2, dim=1)

        spatial_descriptor = torch.cat(
            [
                source_structure.mean(dim=1, keepdim=True),
                target_structure.mean(dim=1, keepdim=True),
                decoder_structure.mean(dim=1, keepdim=True),
                structure_disagreement.mean(dim=1, keepdim=True),
                structure_disagreement.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        source_spatial, target_spatial = self.spatial_router(spatial_descriptor).chunk(2, dim=1)

        route_logits = torch.stack(
            [source_channel + source_spatial, target_channel + target_spatial],
            dim=1,
        )
        route_weights = 2.0 * torch.softmax(route_logits, dim=1)
        source_gate = 1.0 + self.strength * (route_weights[:, 0] - 1.0)
        target_gate = 1.0 + self.strength * (route_weights[:, 1] - 1.0)
        return source_gate, target_gate

    def forward(self, decoder, source_skip, target_skip):
        input_dtype = source_skip.dtype
        with torch.autocast(device_type=source_skip.device.type, enabled=False):
            source_gate, target_gate = self._compute_gates(
                decoder.float(),
                source_skip.float(),
                target_skip.float(),
            )
        return (
            (source_skip.float() * source_gate).to(dtype=input_dtype),
            (target_skip.float() * target_gate).to(dtype=target_skip.dtype),
        )


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


class ParallelLocalGlobalBlock(nn.Module):
    """
    轻量级并联双轨模块：通道切割 (Channel Split)
    - 局部高频分支：3D Convolution
    - 全局低频分支：Mamba
    """
    def __init__(self, in_channels, num_blocks=1, scan_axes=('d', 'h', 'w'), gamma_init=0.2):
        super().__init__()
        # 为了极度节省显存，通道对半切
        self.half_c = in_channels // 2
        
        # 1. 局部分支 (Local CNN): 捕捉小器官或血管边缘的突变纹理
        self.local_branch = nn.Sequential(
            nn.Conv3d(self.half_c, self.half_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.half_c),
            nn.LeakyReLU(0.2)
        )
        
        # 2. 全局分支 (Global Mamba): 长序列拓扑提取和大位移估计
        self.global_branch = ResidualMambaBlock(
            self.half_c, 
            num_blocks=num_blocks, 
            scan_axes=scan_axes, 
            gamma_init=gamma_init
        )
        
        # 3. 融合层 (Fusion): 让 3x3 空间和序列空间特征深度重组
        self.fusion_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 按特征通道切分 (B, C, D, H, W) => 变成两个 (B, C//2, D, H, W)
        x_local, x_global = torch.split(x, self.half_c, dim=1)
        
        # 并联运算
        out_local = self.local_branch(x_local)
        out_global = self.global_branch(x_global)
        
        # 拼接还原回全通道
        out_cat = torch.cat([out_local, out_global], dim=1)
        
        # 残差连接
        return x + self.fusion_conv(out_cat)


class DecoupledEncoder(nn.Module):
    """
    Dual-stream encoder for Siamese Network with Appearance Decoupling.
    Allows specifying the number of decoupled layers at the beginning.
    """
    def __init__(self, in_channels=1, enc_nf=[16, 32, 32, 32], ndim=3, decouple_layers=2, use_dsin=False, encoder_type='cnn', mamba_shallow_multi=False, mamba_enc_shallow_multi=None, mamba_quarter_scale=False, mamba_parallel_block=False):
        super().__init__()
        self.decouple_layers = decouple_layers
        self.use_dsin = use_dsin
        self.encoder_type = encoder_type
        if mamba_enc_shallow_multi is None:
            mamba_enc_shallow_multi = mamba_shallow_multi
        self.mamba_enc_shallow_multi = mamba_enc_shallow_multi
        self.mamba_quarter_scale = mamba_quarter_scale
        self.mamba_parallel_block = mamba_parallel_block
        
        self.enc_blocks_source = nn.ModuleList()
        self.enc_blocks_target = nn.ModuleList()
        self.shared_blocks = nn.ModuleList()
        
        prev_channels = in_channels
        
        for i, nf in enumerate(enc_nf):
            # DSIN: Apply norm only to the decoupled shallow layers (or conditionally all)
            apply_norm = self.use_dsin and (i < decouple_layers)
            
            use_mamba_here = (self.encoder_type == 'mamba') and (
                (i >= 2) or (i == 1 and self.mamba_quarter_scale)
            )
            
            if i < decouple_layers:
                # Decoupled convolution weights + Decoupled (Domain-Specific) Instance Norms
                if use_mamba_here:
                    # Dynamically choose between pure ResidualMamba or Parallel Local-Global Mamba
                    MambaClass = ParallelLocalGlobalBlock if self.mamba_parallel_block else ResidualMambaBlock
                    self.enc_blocks_source.append(nn.Sequential(
                        ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm),
                        MambaClass(nf, num_blocks=1, scan_axes=('d',), gamma_init=0.2)
                    ))
                    self.enc_blocks_target.append(nn.Sequential(
                        ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm),
                        MambaClass(nf, num_blocks=1, scan_axes=('d',), gamma_init=0.2)
                    ))
                else:
                    self.enc_blocks_source.append(ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm))
                    self.enc_blocks_target.append(ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=apply_norm))
            else:
                # Shared convolution weights, usually no norm here for cross-modal interactive consistency
                if use_mamba_here:
                    # Thick Deep Bottleneck: 1 block at 1/8 scale, 3 blocks at 1/16 scale
                    num_mamba = 3 if i == len(enc_nf) - 1 else 1
                    if i == len(enc_nf) - 1:
                        # 1/16 deep bottleneck always uses full 3D scanning
                        scan_axes = ('d', 'h', 'w')
                    elif i == 2:
                        # 1/8 shallow block uses multi-axis only if configured
                        scan_axes = ('d', 'h', 'w') if self.mamba_enc_shallow_multi else ('d',)
                    else:
                        # 1/4 block uses single-axis only to save memory
                        scan_axes = ('d',)
                        
                    MambaClass = ParallelLocalGlobalBlock if self.mamba_parallel_block else ResidualMambaBlock
                    self.shared_blocks.append(nn.Sequential(
                        ConvBlock(ndim, prev_channels, nf, stride=2, use_norm=False),
                        MambaClass(nf, num_blocks=num_mamba, scan_axes=scan_axes, gamma_init=0.2)
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


class SharedSpatialCoordinateCalibration(nn.Module):
    """Calibrate both streams in a shared continuous spatial frame."""

    def __init__(self, channels, hidden_channels=16, strength=0.2):
        super().__init__()
        hidden_channels = max(4, min(int(hidden_channels), channels))
        self.strength = float(strength)
        self.coordinate_mlp = nn.Sequential(
            nn.Conv3d(9, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden_channels, 2 * channels, kernel_size=1),
        )
        nn.init.zeros_(self.coordinate_mlp[-1].weight)
        nn.init.zeros_(self.coordinate_mlp[-1].bias)

    @staticmethod
    def _coordinate_basis(feature):
        depth, height, width = feature.shape[2:]
        z = torch.linspace(-1.0, 1.0, depth, dtype=feature.dtype, device=feature.device)
        y = torch.linspace(-1.0, 1.0, height, dtype=feature.dtype, device=feature.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=feature.dtype, device=feature.device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        basis = torch.stack([
            zz, yy, xx,
            zz.square(), yy.square(), xx.square(),
            torch.sin(torch.pi * zz), torch.sin(torch.pi * yy), torch.sin(torch.pi * xx),
        ], dim=0)
        return basis.unsqueeze(0)

    def forward(self, feature):
        scale, bias = self.coordinate_mlp(self._coordinate_basis(feature)).chunk(2, dim=1)
        strength = feature.new_tensor(self.strength)
        return feature * (1.0 + strength * torch.tanh(scale)) + strength * torch.tanh(bias)


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


class BoundaryGuidanceBlock(nn.Module):
    """
    Lightweight boundary-aware feature modulation block.
    """

    def __init__(self, ndim, channels, hidden_channels=None):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        hidden_channels = hidden_channels or max(channels // 4, 8)
        self.gate = nn.Sequential(
            Conv(2, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            Conv(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, feature, primary_boundary, secondary_boundary, strength=0.5):
        boundary_context = torch.cat([primary_boundary, secondary_boundary], dim=1).to(feature.dtype)
        boundary_gate = torch.sigmoid(self.gate(boundary_context))
        return feature * (1.0 + strength * boundary_gate)



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
                 cross_mamba_scales='1/16,1/8',
                 use_swcv=False, use_gcv=False, encoder_type='cnn', mamba_shallow_multi=False, mamba_enc_shallow_multi=None, mamba_dec_shallow_multi=None, mamba_quarter_scale=False, mamba_dec_quarter_scale=False,  mamba_parallel_block=False, fusion_method='compress_concat', window_size=9, pdaps_flow_limit=20.0, use_residual_flow_pyramid=False, residual_flow_limit=4.0, use_error_guided_residual=False, error_guided_metric='feature_ncc',cross_mamba_offset_limit=0.0, cross_mamba_offset_smooth_kernel=1, use_boundary_branch=False,
                 boundary_branch_scales='deep', boundary_branch_strength=0.5, boundary_kernel='sobel', boundary_smooth_kernel=3,
                 use_cross_frequency_modulation=False, cross_frequency_scales='1/8,1/4', frequency_low_ratio=0.25, cross_frequency_structure_calibration=False,
                 spectral_window_size=4, spectral_temperature=0.1, spectral_guidance_strength=0.5,
                 use_ussc=False, ussc_scales='1/8', ussc_search_radius=2, ussc_temperature=0.1, ussc_guidance_strength=0.5,
                 use_dasr=False, dasr_scales='1/2', dasr_reduction=4, dasr_strength=0.2,
                 use_drfc=False, drfc_scale=0.25, drfc_hidden_channels=16, drfc_strength=0.5,
                 use_sdmr=False, sdmr_use_mind=True, sdmr_scale=0.125, sdmr_hidden_channels=16, sdmr_flow_limit=1.0, sdmr_alpha=0.5,
                 use_dpfc=False, dpfc_flow_limit=8.0, use_miscv=False, miscv_scales='1/8,1/4',
                 use_sscc=False, sscc_scales='1/16,1/8', sscc_hidden_channels=16, sscc_strength=0.2,
                 use_cagr=False, cagr_strength=0.5,
                 miscv_projection_channels=8, miscv_search_radius=2, miscv_temperature=0.1):
        super().__init__()
        self.inshape = inshape
        self.ndim = ndim
        self.int_steps = int_steps
        self.use_pdaps = use_pdaps
        self.fusion_method = fusion_method
        self.use_pdaps = use_pdaps
        self.use_residual_flow_pyramid = use_residual_flow_pyramid
        self.use_dpfc = bool(use_dpfc)
        self.dpfc_flow_limit = float(dpfc_flow_limit)
        self.use_miscv = bool(use_miscv)
        self.miscv_scales = self._parse_decoder_scales(miscv_scales)
        self.use_sscc = bool(use_sscc)
        self.sscc_scales = self._parse_cross_mamba_scales(sscc_scales)
        self.use_cagr = bool(use_cagr)
        self.cagr_strength = float(cagr_strength)
        self.use_error_guided_residual = use_error_guided_residual
        self.error_guided_metric = error_guided_metric
        self.use_sdmr = use_sdmr
        self.sdmr_use_mind = sdmr_use_mind
        self.sdmr_scale = float(sdmr_scale)
        self.sdmr_alpha = float(sdmr_alpha)
        self.use_drfc = bool(use_drfc)
        self.drfc_scale = float(drfc_scale)
        
        if (self.use_error_guided_residual and self.error_guided_metric == 'mind') or (self.use_sdmr and self.sdmr_use_mind):
            from .losses import MINDLoss
            self.mind_extractor = MINDLoss()
        self.residual_flow_limit = residual_flow_limit
        self.use_dsin = use_dsin
        self.use_cmim = use_cmim
        self.use_daps = use_daps
        self.use_cross_mamba = use_cross_mamba
        self.use_wcv = use_wcv
        self.use_swcv = use_swcv
        self.use_gcv = use_gcv
        self.encoder_type = encoder_type
        self.use_boundary_branch = use_boundary_branch
        self.use_dasr = bool(use_dasr)
        self.dasr_scales = self._parse_decoder_scales(dasr_scales)
        self.use_ussc = bool(use_ussc or use_cross_frequency_modulation)
        selected_ussc_scales = ussc_scales if use_ussc else cross_frequency_scales
        self.ussc_scales = self._parse_decoder_scales(selected_ussc_scales)
        self.use_cross_frequency_modulation = self.use_ussc
        self.cross_frequency_scales = self.ussc_scales
        self.cross_frequency_structure_calibration = cross_frequency_structure_calibration
        self.boundary_branch_scales = boundary_branch_scales
        self.boundary_branch_strength = boundary_branch_strength
        self.cross_mamba_scales = self._parse_cross_mamba_scales(cross_mamba_scales)
        self.cross_mamba_offset_limit = float(cross_mamba_offset_limit)
        self.cross_mamba_offset_smooth_kernel = int(cross_mamba_offset_smooth_kernel)
        if mamba_enc_shallow_multi is None:
            mamba_enc_shallow_multi = mamba_shallow_multi
        if mamba_dec_shallow_multi is None:
            mamba_dec_shallow_multi = mamba_shallow_multi
        self.mamba_enc_shallow_multi = mamba_enc_shallow_multi
        self.mamba_dec_shallow_multi = mamba_dec_shallow_multi
        self.mamba_dec_quarter_scale = mamba_dec_quarter_scale

        if self.use_pdaps and self.use_residual_flow_pyramid:
            raise ValueError('use_pdaps and use_residual_flow_pyramid cannot be enabled at the same time')
        if self.use_dpfc and (self.use_pdaps or self.use_residual_flow_pyramid):
            raise ValueError('use_dpfc is mutually exclusive with use_pdaps and use_residual_flow_pyramid')
        if self.use_dpfc and self.int_steps <= 0:
            raise ValueError('use_dpfc requires int_steps > 0 for per-scale diffeomorphic integration')
        if self.use_dpfc and self.dpfc_flow_limit <= 0:
            raise ValueError('dpfc_flow_limit must be positive')
        if self.use_error_guided_residual and not self.use_residual_flow_pyramid:
            raise ValueError('use_error_guided_residual requires use_residual_flow_pyramid to be enabled')
        
        # --- [Architectural Refactoring] ---
        # Note: P-DAPS natively encapsulates coarse-to-fine deformation (previously isolated as 'pyramid').

        # 1. Shared Encoder with pluggable DSIN support
        self.encoder = DecoupledEncoder(
            in_channels, enc_nf, ndim, 
            decouple_layers=decouple_layers, 
            use_dsin=use_dsin,
            encoder_type=encoder_type,
            mamba_shallow_multi=mamba_shallow_multi,
            mamba_enc_shallow_multi=mamba_enc_shallow_multi,
            mamba_quarter_scale=mamba_quarter_scale,
            mamba_parallel_block=mamba_parallel_block
        )
        self.sscc_blocks = nn.ModuleDict()
        if self.use_sscc:
            scale_to_index = {'1/2': -4, '1/4': -3, '1/8': -2, '1/16': -1}
            for scale_name in sorted(self.sscc_scales):
                feature_index = scale_to_index[scale_name]
                self.sscc_blocks[scale_name.replace('/', '_')] = SharedSpatialCoordinateCalibration(
                    enc_nf[feature_index],
                    hidden_channels=sscc_hidden_channels,
                    strength=sscc_strength,
                )

        self.boundary_feature_indices = []
        self.boundary_guidance = nn.ModuleDict()
        if self.use_boundary_branch:
            if boundary_branch_scales == 'all':
                self.boundary_feature_indices = list(range(len(enc_nf)))
            else:
                self.boundary_feature_indices = list(range(max(0, len(enc_nf) - 2), len(enc_nf)))

            self.boundary_extractor = FixedGradientMagnitude3D(
                operator=boundary_kernel,
                smooth_kernel_size=boundary_smooth_kernel,
                normalize=True,
            )
            for idx in self.boundary_feature_indices:
                self.boundary_guidance[str(idx)] = BoundaryGuidanceBlock(ndim, enc_nf[idx])
        
        # 1.5 Cross-Modal Interaction Module
        # CMIM: apply at 1/8 and 1/16 scales
        # Cross-Mamba: configurable deep interaction scales, defaulting to 1/16 + 1/8.
        self.cmim_blocks = nn.ModuleList()
        self.cross_mamba_blocks = nn.ModuleDict()
        if self.use_cmim:
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-1]))
            self.cmim_blocks.append(CrossModalInteractionModule(enc_nf[-2]))
        elif self.use_cross_mamba:
            if '1/16' in self.cross_mamba_scales:
                self.cross_mamba_blocks['1_16'] = CrossMambaModule(
                    enc_nf[-1],
                    offset_limit=self.cross_mamba_offset_limit,
                    offset_smooth_kernel=self.cross_mamba_offset_smooth_kernel,
                )
            if '1/8' in self.cross_mamba_scales:
                self.cross_mamba_blocks['1_8'] = CrossMambaModule(
                    enc_nf[-2],
                    offset_limit=self.cross_mamba_offset_limit,
                    offset_smooth_kernel=self.cross_mamba_offset_smooth_kernel,
                )
            if '1/4' in self.cross_mamba_scales:
                self.cross_mamba_blocks['1_4'] = CrossMambaModule(
                    enc_nf[-3],
                    offset_limit=self.cross_mamba_offset_limit,
                    offset_smooth_kernel=self.cross_mamba_offset_smooth_kernel,
                )
            if '1/2' in self.cross_mamba_scales:
                self.cross_mamba_blocks['1_2'] = CrossMambaModule(
                    enc_nf[-4],
                    offset_limit=self.cross_mamba_offset_limit,
                    offset_smooth_kernel=self.cross_mamba_offset_smooth_kernel,
                )
            
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
            self.wcv_blocks["bottleneck"] = WindowCostVolume3D(enc_nf[-1], window_size=window_size)
            # Deep skip connection (1/8 scale, skip_idx=2)
            self.wcv_blocks["2"] = WindowCostVolume3D(enc_nf[-2], window_size=window_size)

        self.swcv_blocks = nn.ModuleDict()
        if self.use_swcv:
            # Deep Bottleneck (1/16 scale)
            self.swcv_blocks["bottleneck"] = StructureAwareWindowCostVolume3D(enc_nf[-1], window_size=window_size)
            # Deep skip connection (1/8 scale, skip_idx=2)
            self.swcv_blocks["2"] = StructureAwareWindowCostVolume3D(enc_nf[-2], window_size=window_size)

        
        # 2. Standard Decoder
        self.dec_blocks = nn.ModuleList()
        self.ussc_blocks = nn.ModuleDict()
        self.dasr_blocks = nn.ModuleDict()
        self.miscv_blocks = nn.ModuleDict()
        self.miscv_guidance = nn.ModuleDict()
        self.up_blocks = nn.ModuleList()
        
        if self.use_daps:
            self.daps_plr_blocks = nn.ModuleList()

        if self.fusion_method == 'add':
            prev_channels = enc_nf[-1]
        else:
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
            
            # Asymmetric Decoder: inject Mamba at 1/8 scale, and optionally at 1/4 scale.
            if self.encoder_type == 'mamba' and (i == 0 or (i == 1 and self.mamba_dec_quarter_scale)):
                if i == 0:
                    dec_scan_axes = ('d', 'h', 'w') if self.mamba_dec_shallow_multi else ('d',)
                else:
                    dec_scan_axes = ('d',)
                self.dec_blocks.append(nn.Sequential(
                    ConvBlock(ndim, in_ch, nf, stride=1),
                    ResidualMambaBlock(nf, num_blocks=1, scan_axes=dec_scan_axes, gamma_init=0.2)
                ))
            else:
                self.dec_blocks.append(ConvBlock(ndim, in_ch, nf, stride=1))

            decoder_scale = f'1/{2 ** (len(enc_nf) - i - 1)}'
            if self.use_dasr and skip_idx >= 0 and decoder_scale in self.dasr_scales:
                if ndim != 3:
                    raise ValueError('DecoderAdaptiveSkipRouting3D requires ndim=3')
                # Keep subsequent base-model initialization identical for fair
                # same-seed DASR ablations.
                with torch.random.fork_rng(devices=[]):
                    self.dasr_blocks[str(i)] = DecoderAdaptiveSkipRouting3D(
                        decoder_channels=prev_channels,
                        skip_channels=enc_nf[skip_idx],
                        reduction=dasr_reduction,
                        strength=dasr_strength,
                    )
            if self.use_ussc and decoder_scale in self.ussc_scales:
                if ndim != 3:
                    raise ValueError('UncertaintyAwareSelfSimilarityCorrespondence3D requires ndim=3')
                self.ussc_blocks[str(i)] = UncertaintyAwareSelfSimilarityCorrespondence3D(
                    nf,
                    feature_channels=enc_nf[skip_idx],
                    search_radius=ussc_search_radius if use_ussc else max(1, spectral_window_size // 2),
                    temperature=ussc_temperature if use_ussc else spectral_temperature,
                    guidance_strength=ussc_guidance_strength if use_ussc else spectral_guidance_strength,
                )
            if self.use_miscv and skip_idx >= 0 and decoder_scale in self.miscv_scales:
                radius = miscv_search_radius if decoder_scale == '1/8' else 1
                self.miscv_blocks[str(i)] = ModalityInvariantSparseCostVolume3D(
                    feature_channels=enc_nf[skip_idx],
                    projection_channels=miscv_projection_channels,
                    search_radius=radius,
                    temperature=miscv_temperature,
                )
                guidance = getattr(nn, f'Conv{ndim}d')(6, nf, kernel_size=1)
                guidance.weight.data.zero_()
                guidance.bias.data.zero_()
                self.miscv_guidance[str(i)] = guidance
                
            prev_channels = nf
            
        # 3. Final Flow Prediction (Only at full resolution)
        Conv = getattr(nn, f'Conv{ndim}d')
        # We might need a couple of extra convolutions to reach native resolution 
        # because the encoder downsamples 4 times but decoder currently processes upsampled features.
        self.flow_conv = Conv(dec_nf[-1], ndim, kernel_size=3, padding=1)
        
        # Initialize flow weights to very small values
        self.flow_conv.weight.data.normal_(0, 1e-6)
        self.flow_conv.bias.data.zero_()

        if self.use_residual_flow_pyramid:
            self.residual_flow_heads = nn.ModuleList()
            for nf in dec_nf:
                flow_head = Conv(nf, ndim, kernel_size=3, padding=1)
                flow_head.weight.data.normal_(0, 1e-6)
                flow_head.bias.data.zero_()
                self.residual_flow_heads.append(flow_head)

        if self.use_dpfc:
            self.dpfc_residual_blocks = nn.ModuleList()
            self.dpfc_velocity_heads = nn.ModuleList()
            if self.use_cagr:
                self.cagr_confidence_heads = nn.ModuleList()
            for i, nf in enumerate(dec_nf):
                miscv_channels = 6 if str(i) in self.miscv_blocks else 0
                residual_block = ConvBlock(ndim, nf + ndim + 2 + miscv_channels, nf, stride=1)
                velocity_head = Conv(nf, ndim, kernel_size=3, padding=1)
                velocity_head.weight.data.normal_(0, 1e-6)
                velocity_head.bias.data.zero_()
                self.dpfc_residual_blocks.append(residual_block)
                self.dpfc_velocity_heads.append(velocity_head)
                if self.use_cagr:
                    confidence_head = Conv(nf, 1, kernel_size=3, padding=1)
                    confidence_head.weight.data.zero_()
                    confidence_head.bias.data.zero_()
                    self.cagr_confidence_heads.append(confidence_head)
            full_resolution_limits = [
                self.dpfc_flow_limit / (2 ** i)
                for i in range(len(dec_nf))
            ]
            self.register_buffer(
                'dpfc_full_resolution_limits',
                torch.tensor(full_resolution_limits, dtype=torch.float32),
                persistent=True,
            )

        self.error_guided_residual_gates = nn.ModuleDict()
        self.error_guided_residual_stage_indices = set()
        self.latest_error_guided_residual_maps = []
        if self.use_error_guided_residual:
            gate_stage_count = max(0, min(2, len(dec_nf) - 1))
            self.error_guided_residual_stage_indices = set(range(gate_stage_count))
            for stage_idx in self.error_guided_residual_stage_indices:
                gate = nn.Sequential(
                    Conv(1, 8, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    Conv(8, 1, kernel_size=1),
                )
                gate[-1].weight.data.zero_()
                gate[-1].bias.data.zero_()
                self.error_guided_residual_gates[str(stage_idx)] = gate

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
                # 统一使用按层衰减分配物理形变上限，防止深层轻微波动放大后撕裂结构
                pdaps_limits.append(pdaps_flow_limit / (2 ** (len(dec_nf) - i - 1)))
            
            # 改为 nn.Parameter(requires_grad=False)，修复在自动探索大形变时因 NCC 目标直接导致位移限度无限扩展而撕裂网络的问题
            self.pdaps_flow_limits = nn.Parameter(torch.tensor(pdaps_limits, dtype=torch.float32), requires_grad=False)

        if self.use_sdmr:
            if not (0 < self.sdmr_scale <= 1.0):
                raise ValueError(f'sdmr_scale must be in (0, 1], got {self.sdmr_scale}')
            sdmr_in_channels = ndim + 4 + (1 if self.sdmr_use_mind else 0)
            self.sdmr_boundary_extractor = FixedGradientMagnitude3D(
                operator=boundary_kernel,
                smooth_kernel_size=boundary_smooth_kernel,
                normalize=True,
            )
            self.sdmr_refiner = StructureErrorMambaRefiner(
                ndim=ndim,
                in_channels=sdmr_in_channels,
                hidden_channels=sdmr_hidden_channels,
                scan_axes=('d',),
                flow_limit=sdmr_flow_limit,
            )
        if self.use_drfc:
            if not (0 < self.drfc_scale <= 1.0):
                raise ValueError(f'drfc_scale must be in (0, 1], got {self.drfc_scale}')
            self.drfc = DeformationReliabilityFieldCalibration3D(
                ndim=ndim,
                hidden_channels=drfc_hidden_channels,
                strength=drfc_strength,
            )

        # 4. Spatial Transformer
        self.spatial_transform = SpatialTransformer()
        if self.int_steps > 0:
            self.integrate = IntegrateVelocityField(steps=self.int_steps)
        else:
            self.integrate = None

    def _resize_boundary(self, boundary_map, feature):
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        return F.interpolate(boundary_map, size=feature.shape[2:], mode=mode, align_corners=False)

    def _resize_displacement_to_feature(self, displacement, feature):
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        feature_shape = feature.shape[2:]
        displacement_shape = displacement.shape[2:]
        if displacement_shape != feature_shape:
            displacement = F.interpolate(displacement, size=feature_shape, mode=mode, align_corners=True)
            scale = displacement.new_tensor([
                (feature_shape[axis] - 1) / max(displacement_shape[axis] - 1, 1)
                for axis in range(self.ndim)
            ]).view(1, self.ndim, *([1] * self.ndim))
            displacement = displacement * scale
        return displacement

    @staticmethod
    def _dpfc_structure_residual(source_feature, target_feature):
        source_float = source_feature.float()
        target_float = target_feature.float()
        source_norm = F.normalize(source_float, dim=1, eps=1e-6)
        target_norm = F.normalize(target_float, dim=1, eps=1e-6)
        absolute_residual = (source_norm - target_norm).abs().mean(dim=1, keepdim=True)
        cosine_residual = 1.0 - (source_norm * target_norm).sum(dim=1, keepdim=True)
        return torch.cat([absolute_residual, cosine_residual], dim=1).to(source_feature.dtype)

    def _resize_flow(self, flow, target_shape):
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        source_shape = flow.shape[2:]
        if source_shape == target_shape:
            return flow
        resized = F.interpolate(flow, size=target_shape, mode=mode, align_corners=False)
        scale = flow.new_tensor([
            target_shape[axis] / source_shape[axis]
            for axis in range(self.ndim)
        ]).view(1, self.ndim, *([1] * self.ndim))
        return resized * scale

    def _sdmr_low_shape(self, spatial_shape):
        return tuple(max(2, int(round(dim * self.sdmr_scale))) for dim in spatial_shape)

    def _compute_drfc_velocity(self, velocity):
        full_shape = velocity.shape[2:]
        low_shape = tuple(max(4, int(round(dim * self.drfc_scale))) for dim in full_shape)
        velocity_low = self._resize_flow(velocity.float(), low_shape)
        with torch.autocast(device_type=velocity.device.type, enabled=False):
            calibrated_low = self.drfc(velocity_low.float())
        correction_low = calibrated_low - velocity_low
        correction_full = self._resize_flow(correction_low, full_shape)
        return (velocity.float() + correction_full).to(dtype=velocity.dtype)

    def _compute_sdmr_delta_velocity(self, source, target, velocity, displacement):
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        full_shape = velocity.shape[2:]
        low_shape = self._sdmr_low_shape(full_shape)

        with torch.no_grad():
            warped_source = self.spatial_transform(source.float(), displacement.float())
            source_low = F.interpolate(source.float(), size=low_shape, mode=mode, align_corners=False)
            target_low = F.interpolate(target.float(), size=low_shape, mode=mode, align_corners=False)
            warped_low = F.interpolate(warped_source, size=low_shape, mode=mode, align_corners=False)
            intensity_error = torch.abs(warped_low - target_low)
            edge_error = torch.abs(
                self.sdmr_boundary_extractor(warped_low) - self.sdmr_boundary_extractor(target_low)
            )
            error_inputs = [warped_low, target_low, intensity_error, edge_error]
            if self.sdmr_use_mind:
                mind_warped = self.mind_extractor._mind_ssc(warped_low)
                mind_target = self.mind_extractor._mind_ssc(target_low)
                mind_error = torch.abs(mind_warped - mind_target).mean(dim=1, keepdim=True)
                error_inputs.append(mind_error)

        velocity_low = self._resize_flow(velocity.float(), low_shape)
        refiner_input = torch.cat([velocity_low] + [item.to(dtype=velocity_low.dtype) for item in error_inputs], dim=1)
        delta_low = self.sdmr_refiner(refiner_input)
        delta_full = self._resize_flow(delta_low, full_shape)
        return delta_full.to(dtype=velocity.dtype)

    def _feature_gradient_magnitude(self, feature):
        gradients = []
        for axis in range(self.ndim):
            diff = feature.diff(dim=axis + 2)
            pad = [0, 0] * self.ndim
            pad_index = 2 * (self.ndim - axis - 1) + 1
            pad[pad_index] = 1
            gradients.append(F.pad(diff, tuple(pad)))

        grad_sq = sum(grad ** 2 for grad in gradients)
        return torch.sqrt(grad_sq + 1e-6)

    def _feature_edge_loss(self, source_features, target_features, displacement, feature_edge_indices):
        feature_edge_loss = displacement.new_tensor(0.0)
        valid_scales = 0
        for idx in feature_edge_indices:
            if idx < 0:
                idx = len(source_features) + idx
            if idx < 0 or idx >= len(source_features):
                continue

            source_feature = source_features[idx].float().detach()
            target_feature = target_features[idx].float().detach()
            feature_displacement = self._resize_displacement_to_feature(displacement, source_feature)
            warped_source_feature = self.spatial_transform(source_feature, feature_displacement)
            source_grad = self._feature_gradient_magnitude(warped_source_feature)
            target_grad = self._feature_gradient_magnitude(target_feature)
            feature_edge_loss = feature_edge_loss + torch.mean(torch.abs(source_grad - target_grad))
            valid_scales += 1

        if valid_scales == 0:
            return feature_edge_loss
        return feature_edge_loss / valid_scales

    def _compute_error_guided_residual_gate(self, stage_idx, source_feature, target_feature, accumulated_flow):
        stage_key = str(stage_idx)
        gate_module = self.error_guided_residual_gates[stage_key] if stage_key in self.error_guided_residual_gates else None
        if gate_module is None or source_feature is None or target_feature is None:
            return None, None

        source_feature_float = source_feature.float()
        target_feature_float = target_feature.float()
        warped_source = source_feature_float

        if accumulated_flow is not None:
            mode = 'trilinear' if self.ndim == 3 else 'bilinear'
            flow_up = F.interpolate(accumulated_flow, size=source_feature.shape[2:], mode=mode, align_corners=False)
            flow_up = flow_up * 2.0
            if self.integrate is not None:
                warp_displacement = self.integrate(flow_up)
            else:
                warp_displacement = flow_up
            warped_source = self.spatial_transform(source_feature_float, warp_displacement.float())

        # CRITICAL FIX: Error map computation depends on the metric mode
        if getattr(self, 'use_error_guided_residual', False) and getattr(self, 'error_guided_metric', 'feature_ncc') == 'mind' and hasattr(self, '_mind_source'):
            size = source_feature.shape[2:]
            mode = 'trilinear' if self.ndim == 3 else 'bilinear'
            mind_s = F.interpolate(self._mind_source.float(), size=size, mode=mode, align_corners=False)
            mind_t = F.interpolate(self._mind_target.float(), size=size, mode=mode, align_corners=False)
            if accumulated_flow is not None:
                warped_mind_s = self.spatial_transform(mind_s, warp_displacement.float())
            else:
                warped_mind_s = mind_s
            # MIND is structurally invariant, direct L1 difference measures structural misalignment
            error_map = torch.abs(warped_mind_s - mind_t).mean(dim=1, keepdim=True).detach()
        else:
            # Fallback to feature-space NCC
            # (CT/MR have different intensity distributions, so aligned features don't numerically match).
            # We compute Pearson Correlation across the channel dimension and take 1 - abs(NCC)
            # to measure true structural misalignment invariant to modality intensity gaps.
            s_mean = warped_source.mean(dim=1, keepdim=True)
            t_mean = target_feature_float.mean(dim=1, keepdim=True)
            s_dev = warped_source - s_mean
            t_dev = target_feature_float - t_mean
            
            covar = torch.sum(s_dev * t_dev, dim=1, keepdim=True)
            s_var = torch.sum(s_dev * s_dev, dim=1, keepdim=True)
            t_var = torch.sum(t_dev * t_dev, dim=1, keepdim=True)
            
            ncc = covar / (torch.sqrt(s_var * t_var) + 1e-5)
            # abs(ncc) handles inverse polarity (e.g. CT bright bone vs MR dark bone).
            # 1.0 - abs(ncc) represents structural discrepancy (0 when perfectly aligned/inversely-aligned, 1 when misaligned)
            error_map = (1.0 - torch.abs(ncc)).detach()

        gate_logits = gate_module(error_map)
        gate = 0.5 + torch.sigmoid(gate_logits)
        return gate.to(dtype=source_feature.dtype), error_map.to(dtype=source_feature.dtype)

    def _parse_cross_mamba_scales(self, cross_mamba_scales):
        if cross_mamba_scales is None:
            return {'1/16', '1/8'}

        if isinstance(cross_mamba_scales, str):
            items = [item.strip() for item in cross_mamba_scales.split(',') if item.strip()]
        else:
            items = [str(item).strip() for item in cross_mamba_scales if str(item).strip()]

        if not items:
            return {'1/16', '1/8'}

        valid = {'1/16', '1/8', '1/4', '1/2'}
        invalid = sorted(set(items) - valid)
        if invalid:
            raise ValueError(f'Unsupported cross_mamba_scales: {invalid}. Valid values are {sorted(valid)}')

        return set(items)

    def _parse_decoder_scales(self, scales):
        if scales is None:
            return set()
        if isinstance(scales, str):
            items = [item.strip() for item in scales.split(',') if item.strip()]
        else:
            items = [str(item).strip() for item in scales if str(item).strip()]
        valid = {'1/8', '1/4', '1/2'}
        invalid = sorted(set(items) - valid)
        if invalid:
            raise ValueError(f'Unsupported USSC scales: {invalid}. Valid values are {sorted(valid)}')
        return set(items)

    def forward(self, source, target, return_warped_source=True, return_field_type='displacement', return_coarse_flows=False, return_residual_flows=False, return_feature_edge_loss=False, feature_edge_indices=(0, 1), swap_encoder_branches=False):
        # --- [Ablation 1 Hook: FDA will go here] ---
        source_input = source
        target_input = target
        
        if getattr(self, 'use_error_guided_residual', False) and getattr(self, 'error_guided_metric', 'feature_ncc') == 'mind':
            self._mind_source = self.mind_extractor._mind_ssc(source_input)
            self._mind_target = self.mind_extractor._mind_ssc(target_input)
        
        # 1. Feature Extraction (Decoupled/Shared)
        if swap_encoder_branches:
            # Preserve modality-specific shallow encoders when reversing the
            # registration direction (e.g. MR->CT after training CT->MR).
            feat_t, feat_s = self.encoder(target_input, source_input)
        else:
            feat_s, feat_t = self.encoder(source_input, target_input)

        if self.use_sscc:
            scale_to_index = {'1/2': -4, '1/4': -3, '1/8': -2, '1/16': -1}
            for scale_name in self.sscc_scales:
                block = self.sscc_blocks[scale_name.replace('/', '_')]
                feature_index = scale_to_index[scale_name]
                feat_s[feature_index] = block(feat_s[feature_index])
                feat_t[feature_index] = block(feat_t[feature_index])

        if self.use_boundary_branch:
            source_boundary = self.boundary_extractor(source_input)
            target_boundary = self.boundary_extractor(target_input)
            for idx in self.boundary_feature_indices:
                source_boundary_scale = self._resize_boundary(source_boundary, feat_s[idx]).to(feat_s[idx].dtype)
                target_boundary_scale = self._resize_boundary(target_boundary, feat_t[idx]).to(feat_t[idx].dtype)
                boundary_block = self.boundary_guidance[str(idx)]
                feat_s[idx] = boundary_block(
                    feat_s[idx],
                    source_boundary_scale,
                    target_boundary_scale,
                    strength=self.boundary_branch_strength,
                )
                feat_t[idx] = boundary_block(
                    feat_t[idx],
                    target_boundary_scale,
                    source_boundary_scale,
                    strength=self.boundary_branch_strength,
                )
        
        coarse_flows = []
        residual_flows = []
        pyramid_acc_flow = None
        residual_acc_flow = None
        dpfc_displacement = None
        self.latest_error_guided_residual_maps = []
        
        # 2. Decoding (Standard U-Net Upsampling)
        # Start from the bottom-most features (1/16 scale)
        if self.use_cmim:
            feat_s[-1] = self.cmim_blocks[0](feat_s[-1], feat_t[-1])
        elif getattr(self, 'use_cross_mamba', False) and '1_16' in self.cross_mamba_blocks:
            feat_s[-1] = self.cross_mamba_blocks['1_16'](feat_s[-1], feat_t[-1])
        if getattr(self, 'use_wcv', False) and "bottleneck" in self.wcv_blocks:
            feat_s[-1] = self.wcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])
        if getattr(self, 'use_swcv', False) and "bottleneck" in self.swcv_blocks:
            feat_s[-1] = self.swcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])
        if getattr(self, 'use_gcv', False) and "bottleneck" in self.gcv_blocks:
            feat_s[-1] = self.gcv_blocks["bottleneck"](x_fixed=feat_t[-1], x_moving=feat_s[-1])

            
        # --- Fusion Method Applier ---
        if self.fusion_method == 'add':
            x = feat_s[-1] + feat_t[-1] # Element-wise sum logic to preserve Mamba space
        else:
            x = torch.cat([feat_s[-1], feat_t[-1]], dim=1)
        
        for i, block in enumerate(self.dec_blocks):
            # Upsample
            mode = 'trilinear' if self.ndim == 3 else 'bilinear'
            x = F.interpolate(x, scale_factor=2.0, mode=mode, align_corners=False)
            
            # Skip connections
            skip_idx = len(feat_s) - 2 - i
            s_skip_raw = None
            t_skip_raw = None
            ussc_source_feature = None
            ussc_target_feature = None
            miscv_descriptor = None
            if skip_idx >= 0:
                s_skip = feat_s[skip_idx]
                t_skip = feat_t[skip_idx]
                
                s_skip_raw = s_skip
                t_skip_raw = t_skip
                
                # --- [CMIM at 1/8 Scale] ---
                if self.use_cmim and skip_idx == len(feat_s) - 2:
                    s_skip = self.cmim_blocks[1](s_skip, t_skip)
                
                # --- [Cross-Mamba at multi scales] ---
                if getattr(self, 'use_cross_mamba', False):
                    if skip_idx == len(feat_s) - 2 and '1_8' in self.cross_mamba_blocks:   # 1/8 scale
                        s_skip = self.cross_mamba_blocks['1_8'](s_skip, t_skip)
                    elif skip_idx == len(feat_s) - 3 and '1_4' in self.cross_mamba_blocks: # 1/4 scale
                        s_skip = self.cross_mamba_blocks['1_4'](s_skip, t_skip)
                    elif skip_idx == len(feat_s) - 4 and '1_2' in self.cross_mamba_blocks: # 1/2 scale
                        s_skip = self.cross_mamba_blocks['1_2'](s_skip, t_skip)

                if self.use_dasr and str(i) in self.dasr_blocks:
                    s_skip, t_skip = self.dasr_blocks[str(i)](x, s_skip, t_skip)
                
                # --- [GCV at Deep Scales] ---
                if getattr(self, 'use_gcv', False) and str(skip_idx) in self.gcv_blocks:
                    s_skip = self.gcv_blocks[str(skip_idx)](x_fixed=t_skip, x_moving=s_skip)

                # USSC compares symmetric pre-interaction features. Cross-Mamba
                # updates source only, which would make self-similarity spaces
                # asymmetric and corrupt local correspondence probabilities.
                ussc_source_feature = s_skip_raw
                ussc_target_feature = t_skip_raw
                
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
            if self.use_ussc and str(i) in self.ussc_blocks:
                x = self.ussc_blocks[str(i)](
                    x,
                    ussc_source_feature,
                    ussc_target_feature,
                )
            if self.use_miscv and str(i) in self.miscv_blocks:
                miscv_source_feature = s_skip_raw
                if self.use_dpfc and dpfc_displacement is not None:
                    miscv_coarse_displacement = self._resize_displacement_to_feature(dpfc_displacement, s_skip_raw)
                    miscv_source_feature = self.spatial_transform(s_skip_raw, miscv_coarse_displacement)
                miscv_descriptor = self.miscv_blocks[str(i)](miscv_source_feature, t_skip_raw)
                x = x + self.miscv_guidance[str(i)](miscv_descriptor)

            if self.use_residual_flow_pyramid:
                sub_flow_limit = torch.as_tensor(self.residual_flow_limit, dtype=x.dtype, device=x.device).clamp(min=1e-3)
                residual_gate = None
                error_map = None
                if self.use_error_guided_residual and i in self.error_guided_residual_stage_indices:
                    residual_gate, error_map = self._compute_error_guided_residual_gate(i, s_skip_raw, t_skip_raw, residual_acc_flow)

                raw_residual_flow = self.residual_flow_heads[i](x)
                if residual_gate is not None:
                    if residual_gate.shape[2:] != raw_residual_flow.shape[2:]:
                        residual_gate = F.interpolate(residual_gate, size=raw_residual_flow.shape[2:], mode=mode, align_corners=False)
                    raw_residual_flow = raw_residual_flow * residual_gate
                residual_flow = sub_flow_limit * torch.tanh(raw_residual_flow / sub_flow_limit)
                
                if residual_gate is not None:
                    self.latest_error_guided_residual_maps.append({
                        'stage_idx': i,
                        'skip_idx': skip_idx,
                        'error_map': error_map.detach(),
                        'gate_map': residual_gate.detach(),
                    })
                residual_flows.append(residual_flow)

                if residual_acc_flow is None:
                    residual_acc_flow = residual_flow
                else:
                    mode = 'trilinear' if self.ndim == 3 else 'bilinear'
                    up_flow = F.interpolate(residual_acc_flow, size=residual_flow.shape[2:], mode=mode, align_corners=False)
                    up_flow = up_flow * 2.0
                    residual_acc_flow = up_flow + residual_flow

                if return_coarse_flows and i < len(self.dec_blocks) - 1:
                    if self.integrate is not None:
                        coarse_flows.append(self.integrate(residual_acc_flow))
                    else:
                        coarse_flows.append(residual_acc_flow)

            if self.use_dpfc:
                full_limit = self.dpfc_full_resolution_limits[i].to(dtype=x.dtype, device=x.device)
                scale = x.new_tensor([
                    x.shape[2 + axis] / self.inshape[axis]
                    for axis in range(self.ndim)
                ]).view(1, self.ndim, *([1] * self.ndim))
                local_limit = (full_limit * scale).clamp(min=1e-3)
                if dpfc_displacement is None:
                    coarse_displacement = x.new_zeros(x.shape[0], self.ndim, *x.shape[2:])
                else:
                    coarse_displacement = self._resize_displacement_to_feature(dpfc_displacement, x)

                if s_skip_raw is not None and t_skip_raw is not None:
                    warped_source_feature = self.spatial_transform(s_skip_raw, coarse_displacement)
                    structure_residual = self._dpfc_structure_residual(warped_source_feature, t_skip_raw)
                else:
                    source_scale = F.interpolate(source, size=x.shape[2:], mode=mode, align_corners=True)
                    target_scale = F.interpolate(target, size=x.shape[2:], mode=mode, align_corners=True)
                    warped_source_scale = self.spatial_transform(source_scale, coarse_displacement)
                    source_gradient = F.avg_pool3d(warped_source_scale.float(), kernel_size=3, stride=1, padding=1)
                    target_gradient = F.avg_pool3d(target_scale.float(), kernel_size=3, stride=1, padding=1)
                    structure_residual = torch.cat([
                        (warped_source_scale.float() - source_gradient).abs(),
                        (target_scale.float() - target_gradient).abs(),
                    ], dim=1).to(x.dtype)

                normalized_coarse = coarse_displacement / local_limit
                residual_inputs = [x, structure_residual, normalized_coarse]
                if miscv_descriptor is not None:
                    residual_inputs.append(miscv_descriptor)
                residual_feature = self.dpfc_residual_blocks[i](
                    torch.cat(residual_inputs, dim=1)
                )
                raw_local_velocity = self.dpfc_velocity_heads[i](residual_feature)
                local_velocity = local_limit * torch.tanh(raw_local_velocity / local_limit)
                if self.use_cagr:
                    confidence_logits = self.cagr_confidence_heads[i](residual_feature)
                    confidence_scale = 1.0 + self.cagr_strength * (2.0 * torch.sigmoid(confidence_logits) - 1.0)
                    local_velocity = local_velocity * confidence_scale
                residual_flows.append(local_velocity)
                local_displacement = self.integrate(local_velocity)

                if dpfc_displacement is None:
                    dpfc_displacement = local_displacement
                else:
                    # Apply the accumulated coarse transform first, then the
                    # current local transform using exact displacement composition.
                    dpfc_displacement = local_displacement + self.spatial_transform(
                        coarse_displacement,
                        local_displacement,
                    )

                if return_coarse_flows and i < len(self.dec_blocks) - 1:
                    coarse_flows.append(dpfc_displacement)

            # --- [P-DAPS Coarse-to-fine Flow generation & Deep Supervision] ---
            if getattr(self, 'use_pdaps', False):
                sub_flow_limit = torch.clamp(self.pdaps_flow_limits[i], min=1e-3).to(dtype=x.dtype)
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
        elif self.use_residual_flow_pyramid:
            velocity = residual_acc_flow
        elif self.use_dpfc:
            displacement = dpfc_displacement
            # A composition of independently integrated velocities has no
            # single stationary-velocity equivalent.
            velocity = displacement
        else:
            velocity = 20.0 * torch.tanh(self.flow_conv(x) / 20.0)

        if self.use_drfc and not self.use_dpfc:
            velocity = self._compute_drfc_velocity(velocity)
        
        if not self.use_dpfc:
            if self.integrate is not None:
                displacement = self.integrate(velocity)
            else:
                displacement = velocity

        if self.use_sdmr and not self.use_dpfc:
            delta_velocity = self._compute_sdmr_delta_velocity(source, target, velocity, displacement)
            velocity = velocity + self.sdmr_alpha * delta_velocity
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

        if return_residual_flows:
            outputs.append(residual_flows)

        if return_feature_edge_loss:
            outputs.append(self._feature_edge_loss(feat_s, feat_t, displacement, feature_edge_indices))

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
