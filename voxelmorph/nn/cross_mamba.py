import torch
import torch.nn as nn
from mamba_ssm import Mamba

class CrossMambaModule(nn.Module):
    """
    Improved 3D Cross-Mamba Module with Multi-Directional Scanning.
    Inspired by VMamba/SegMamba's Cross-Scan Module to preserve spatial locality.
    """
    def __init__(self, channels, d_state=16, d_conv=4, expand=2, img_size=(10, 12, 10), use_resampling=True):
        super().__init__()
        self.channels = channels
        self.use_resampling = use_resampling
        
        # 3D learnable positional embedding (optional but highly recommended for 1D scanning)
        # 用一个极小的晶格尺寸 (10x12x10) 作为连续位置编码的种子，大大降低参数量
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, *img_size))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # 共享一个 Mamba 权重以减少参数量，同时处理多个方向展开
        self.mamba = Mamba(
            d_model=channels, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
        )
        self.seq_norm = nn.LayerNorm(channels)
        
        # 因为提取了 6 个方向（3个轴 x 2个正反向）的特征图，使用 1x1x1 卷积做降维综合
        self.direction_fuse = nn.Conv3d(channels * 6, channels, 1)
        
        # 🌟 方案一：显式局部重采样 (Deformable Feature Resampling)
        # 用 Mamba 的全局上下文输出预测 3D 局部形变偏移量 (dx, dy, dz) 和调制掩码
        self.offset_conv = nn.Conv3d(channels, 3, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        # 扩大感受野：加入 3x3x3 局部空间卷积补偿，缝合重采样后的局部几何信息
        self.fuse_local = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)
        
        # 将偏移量初始化为零，以便一开始是个纯残差网络，避免早期崩溃
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
        # 掩码初始化略给一点偏置(1.0即可让sigmoid起始为0.73)，避免一开始重采样特征被折半
        self.mask_conv.weight.data.zero_()
        self.mask_conv.bias.data.fill_(1.0)
        
        self.act = nn.GELU()
        self.norm = nn.InstanceNorm3d(channels)
        
    def forward(self, source, target):
        # 加上 3D 选择性位置编码
        if self.pos_embed.shape[2:] != source.shape[2:]:
            # If the feature map size changes (e.g. from downsampling), interpolate the positional embedding
            pos_embed = nn.functional.interpolate(self.pos_embed, size=source.shape[2:], mode='trilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed

        source = source + pos_embed
        target = target + pos_embed

        # 防雷机制：强制转成 FP32 运算，防止 Mamba 在半精度下的指数运算爆炸
        B, C, D, H, W = source.shape
        src_fp32 = source.float()
        tgt_fp32 = target.float()
        
        out_features = []
        
        # ====== Direction 1: D(Z) axis ======
        # (B, C, D, H, W) -> (B, C, L) -> (B, L, C)
        seq_s_z = src_fp32.view(B, C, -1).transpose(1, 2)
        seq_t_z = tgt_fp32.view(B, C, -1).transpose(1, 2)
        out_features.append(self._scan_and_extract(seq_s_z, seq_t_z, D, H, W, order='z'))
        out_features.append(self._scan_and_extract(seq_s_z, seq_t_z, D, H, W, order='z', reverse=True))

        # ====== Direction 2: H(Y) axis ======
        # permute to B, C, H, D, W
        seq_s_y = src_fp32.permute(0, 1, 3, 2, 4).reshape(B, C, -1).transpose(1, 2)
        seq_t_y = tgt_fp32.permute(0, 1, 3, 2, 4).reshape(B, C, -1).transpose(1, 2)
        out_features.append(self._scan_and_extract(seq_s_y, seq_t_y, D, H, W, order='y'))
        out_features.append(self._scan_and_extract(seq_s_y, seq_t_y, D, H, W, order='y', reverse=True))

        # ====== Direction 3: W(X) axis ======
        # permute to B, C, W, D, H
        seq_s_x = src_fp32.permute(0, 1, 4, 2, 3).reshape(B, C, -1).transpose(1, 2)
        seq_t_x = tgt_fp32.permute(0, 1, 4, 2, 3).reshape(B, C, -1).transpose(1, 2)
        out_features.append(self._scan_and_extract(seq_s_x, seq_t_x, D, H, W, order='x'))
        out_features.append(self._scan_and_extract(seq_s_x, seq_t_x, D, H, W, order='x', reverse=True))
        
        # 合并 6 个空间扫描维度的结果表征 (B, 6*C, D, H, W) -> (B, C, D, H, W)
        multi_dir_concat = torch.cat(out_features, dim=1) 
        mamba_guidance = self.direction_fuse(multi_dir_concat)
        mamba_guidance = self.act(mamba_guidance)
        mamba_guidance = mamba_guidance.to(source.dtype)
        
        if self.use_resampling:
            # 🌟 方案一实施：利用 Mamba 获取的长距离指导，执行显式局部特征重采样
            # 1. 预测特征级的微观偏移量 (Offset) 和调制权重 (Mask)
            offset = self.offset_conv(mamba_guidance)  # (B, 3, D, H, W)
            mask = torch.sigmoid(self.mask_conv(mamba_guidance))  # (B, C, D, H, W)
            
            # 2. 构建特征级的仿射网格 (Feature Grid)
            device, dtype = source.device, source.dtype
            vectors = [torch.arange(0, s, device=device, dtype=dtype) for s in (D, H, W)]
            grids = torch.meshgrid(vectors, indexing='ij')
            base_grid = torch.stack(grids).unsqueeze(0).expand(B, -1, -1, -1, -1) # (B, 3, D, H, W)
            
            # 将偏移量应用于 base grid （这里 offset 是基于像素尺度的局部位移）
            deformed_grid = base_grid + offset
            
            # 将 grid 归一化到 [-1, 1] 才能被 F.grid_sample 使用
            normalized_grid = torch.zeros_like(deformed_grid)
            for i, s in enumerate((D, H, W)):
                normalized_grid[:, i, ...] = 2.0 * (deformed_grid[:, i, ...] / (s - 1.0)) - 1.0
                
            # grid_sample 要求的座标系是在最后一维，且在3D中顺序是 (W, H, D) -> 即 (x, y, z)
            normalized_grid = normalized_grid.permute(0, 2, 3, 4, 1) # -> (B, D, H, W, 3)
            normalized_grid = normalized_grid[..., [2, 1, 0]] # -> flip to (x, y, z)
            
            # 3. 对 Source 进行双线性可变形重采样。修改 padding_mode 为 'border' 防止边界黑边伪影
            resampled_source = nn.functional.grid_sample(
                source, normalized_grid, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            
            # 4. 施加 Mamba 预测的门控掩码，并经过 3x3x3 空间卷积进行平滑雕花
            modulated_source = resampled_source * mask
            
            # 【核心修复】将重采样后的 source 与 Mamba 本身提取的全局语义特征拼接融合！如果不拼，Mamba 就变成了一个纯 STN，丧失特征提取意义。
            cat_feat = torch.cat([modulated_source, mamba_guidance], dim=1)
            refined_fused = self.fuse_local(cat_feat)
            
            return self.norm(self.act(refined_fused) + source)
        else:
            # 退回原始的特征直接加法融合逻辑，取消重采样模块的介入
            cat_feat = torch.cat([source, mamba_guidance], dim=1)
            refined_fused = self.fuse_local(cat_feat)
            return self.norm(self.act(refined_fused) + source)
        
    def _scan_and_extract(self, seq_s, seq_t, D, H, W, order='z', reverse=False):
        B, L, C = seq_s.shape
        
        if reverse:
            # 解决 Logic Bug：必须分别翻转各自的维度以保证目标始终是在前半截作为历史预输入！
            seq_s = torch.flip(seq_s, dims=[1])
            seq_t = torch.flip(seq_t, dims=[1])

        seq_s = self.seq_norm(seq_s)
        seq_t = self.seq_norm(seq_t)
            
        # Target 和 Source 物理级交织（Interleaved）重组。这是配准长距离匹配的灵魂。
        # [T1, S1, T2, S2, ..., TL, SL]
        stacked = torch.stack([seq_t, seq_s], dim=2) 
        seq_concat = stacked.view(B, 2 * L, C)
        
        scan_out = self.mamba(seq_concat)
        
        # 提取 Source，由于是交织排列 (T,S,T,S)，Source 全在奇数位 1, 3, 5...
        out_s = scan_out[:, 1::2, :]
        
        if reverse:
            out_s = torch.flip(out_s, dims=[1])
            
        out_s = out_s.transpose(1, 2) # (B, C, L)
        
        # 由于我们是从不同的主序平展开的，现在要安全地 reshape 回原来的三维结构尺度
        if order == 'z':
            out_3d = out_s.reshape(B, C, D, H, W)
        elif order == 'y':     
            out_3d = out_s.reshape(B, C, H, D, W).permute(0, 1, 3, 2, 4)
        elif order == 'x':
            out_3d = out_s.reshape(B, C, W, D, H).permute(0, 1, 3, 4, 2)
            
        return out_3d
