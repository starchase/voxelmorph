import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VSSBlock3D(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2, gamma_init=0.2, scan_axes=('d',)):
        super().__init__()
        self.channels = channels
        self.scan_axes = tuple(scan_axes)
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
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), gamma_init))
        if len(self.scan_axes) > 1:
            self.axis_logits = nn.Parameter(torch.zeros(len(self.scan_axes)))
        else:
            self.register_parameter('axis_logits', None)

    def _scan_sequence(self, x_seq):
        x_seq = self.ln(x_seq)
        x_fwd = self.mamba(x_seq)
        x_rev = self.mamba(torch.flip(x_seq, dims=[1]))
        return x_fwd + torch.flip(x_rev, dims=[1])

    def _flatten_axis(self, x, axis):
        if axis == 'd':
            return x.flatten(2).transpose(1, 2), None
        if axis == 'h':
            return x.permute(0, 1, 3, 2, 4).contiguous().flatten(2).transpose(1, 2), 'h'
        if axis == 'w':
            return x.permute(0, 1, 4, 2, 3).contiguous().flatten(2).transpose(1, 2), 'w'
        raise ValueError(f'Unsupported scan axis: {axis}')

    def _restore_axis(self, x_seq, B, C, D, H, W, axis_tag):
        x = x_seq.transpose(1, 2).contiguous()
        if axis_tag is None:
            return x.view(B, C, D, H, W)
        if axis_tag == 'h':
            return x.view(B, C, H, D, W).permute(0, 1, 3, 2, 4).contiguous()
        if axis_tag == 'w':
            return x.view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous()
        raise ValueError(f'Unsupported restore axis: {axis_tag}')

    def forward(self, x):
        # x.shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        shortcut = x
        
        # Local Spatial Positional Encoding
        x = self.dwconv(x)
        x = self.act(x)
        
        axis_outputs = []
        for axis in self.scan_axes:
            x_seq, axis_tag = self._flatten_axis(x, axis)
            x_seq = self._scan_sequence(x_seq)
            axis_outputs.append(self._restore_axis(x_seq, B, C, D, H, W, axis_tag))

        if len(axis_outputs) == 1:
            x = axis_outputs[0]
        else:
            axis_weights = torch.softmax(self.axis_logits, dim=0).view(-1, 1, 1, 1, 1, 1)
            x = (torch.stack(axis_outputs, dim=0) * axis_weights).sum(dim=0)
        return shortcut + self.gamma * x
