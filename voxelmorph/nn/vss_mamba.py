import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VSSBlock3D(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # Adding a local spatial compensator (DWConv)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x.shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        shortcut = x
        
        # Local Spatial Positional Encoding
        x = self.dwconv(x)
        x = self.act(x)
        
        # Flatten for Mamba
        x = x.view(B, C, -1).permute(0, 2, 1) # (B, L, C)
        x = self.ln(x)
        
        # Multi-scan approx (simplified for now with bidirectional or forward)
        x_fwd = self.mamba(x)
        x_rev = self.mamba(torch.flip(x, dims=[1]))
        x = x_fwd + torch.flip(x_rev, dims=[1])
        
        # Reshape back
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        return x + shortcut
