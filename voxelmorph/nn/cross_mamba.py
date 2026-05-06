import torch
import torch.nn as nn
from mamba_ssm import Mamba

class CrossMambaModule(nn.Module):
    """
    Improved 3D Cross-Mamba Module with Multi-Directional Scanning.
    Inspired by VMamba/SegMamba's Cross-Scan Module to preserve spatial locality.
    """
    def __init__(self, channels, d_state=16, d_conv=4, expand=2, img_size=(10, 12, 10)):
        super().__init__()
        self.channels = channels
        
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
        self.cross_gate = nn.Parameter(torch.tensor(-2.0))
        # 扩大感受野：加入 3x3x3 局部空间卷积补偿，缝合 Mamba 序列化可能丢失的局部几何信息
        self.fuse_local = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)
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
        cross_out = self.direction_fuse(multi_dir_concat)
        gate = torch.sigmoid(self.cross_gate).to(dtype=source.dtype)
        gated_cross_out = cross_out.to(source.dtype) * gate
        
        # 结果降维转回对应精度，并与原本的 source 进行形变场纠正引导
        cat_feat = torch.cat([source, gated_cross_out], dim=1)
        fused = self.fuse_local(cat_feat)
        return self.norm(self.act(fused) + source)
        
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
