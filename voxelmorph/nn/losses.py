"""
Loss functions for image registration.
"""

# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import FixedGradientMagnitude3D


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        raise NotImplementedError(
            'voxelmorph.nn.losses.NCC is deprecated. Use neurite.nn.modules.NCC instead.'
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            'voxelmorph.nn.losses.NCC is deprecated. Use neurite.nn.modules.NCC instead.'
        )


class MSE:
    """
    Deprecated. Use neurite.nn.modules.MSE instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "voxelmorph.nn.losses.MSE is deprecated. Use neurite.nn.modules.MSE instead."
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.MSE is deprecated. Use neurite.nn.modules.MSE instead."
        )


class Dice:
    """
    Deprecated. Use neurite.nn.modules.Dice instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Dice is deprecated. Use neurite.nn.modules.Dice instead."
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Dice is deprecated. Use neurite.nn.modules.Dice instead."
        )


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. Use neurite.nn.modules.Grad instead."
        )

    def _diffs(self, y):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. Use neurite.nn.modules.Grad instead."
        )

    def loss(self, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. Use neurite.nn.modules.Grad instead."
        )

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, num_bin)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.register_buffer('vol_bin_centers', vol_bin_centers)

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o)

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, num_bin)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.register_buffer('vol_bin_centers', vol_bin_centers)
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)
        
        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o)
        
        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)
        
        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size,
                                            (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size,
                                            (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5, 6)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                            (x + x_r) // self.patch_size, self.patch_size,
                                            (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5, 6)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))
        
        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)
        
        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self,y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

class MINDLoss(nn.Module):
    """
    Standard 3D MIND-SSC (Modality Independent Neighbourhood Descriptor -
    Self-Similarity Context) loss.

    This implementation follows the common 12-channel formulation used in
    deformable multimodal registration. It builds self-similarity descriptors
    from pairwise patch SSDs in a 6-neighbourhood and compares descriptors with
    a voxel-wise squared error.
    """
    def __init__(self, radius=2, dilation=2, eps=1e-8):
        super(MINDLoss, self).__init__()
        self.radius = radius
        self.dilation = dilation
        self.eps = eps

        neighbourhood = torch.tensor([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 2],
            [2, 1, 1],
            [1, 2, 1],
        ], dtype=torch.long)
        pairwise_dist = torch.sum(
            (neighbourhood[:, None, :] - neighbourhood[None, :, :]) ** 2,
            dim=-1,
        )
        pair_indices = torch.nonzero(
            torch.triu((pairwise_dist == 2), diagonal=1),
            as_tuple=False,
        )

        kernel_1 = torch.zeros((pair_indices.shape[0], 1, 3, 3, 3), dtype=torch.float32)
        kernel_2 = torch.zeros((pair_indices.shape[0], 1, 3, 3, 3), dtype=torch.float32)
        for idx, (first, second) in enumerate(pair_indices):
            kernel_1[idx, 0, neighbourhood[first, 0], neighbourhood[first, 1], neighbourhood[first, 2]] = 1.0
            kernel_2[idx, 0, neighbourhood[second, 0], neighbourhood[second, 1], neighbourhood[second, 2]] = 1.0

        self.register_buffer('kernel_1', kernel_1)
        self.register_buffer('kernel_2', kernel_2)

    def _mind_ssc(self, image):
        if image.dim() != 5:
            raise ValueError(f'MINDLoss expects 5D input [B, C, D, H, W], got shape {tuple(image.shape)}')
        if image.shape[1] != 1:
            raise ValueError(f'MINDLoss expects single-channel input, got {image.shape[1]} channels')

        kernel_1 = self.kernel_1.to(dtype=image.dtype)
        kernel_2 = self.kernel_2.to(dtype=image.dtype)

        pad_size = self.dilation
        image_pad = F.pad(image, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='replicate')

        shifted_1 = F.conv3d(image_pad, kernel_1, dilation=self.dilation)
        shifted_2 = F.conv3d(image_pad, kernel_2, dilation=self.dilation)
        patch_ssd = (shifted_1 - shifted_2).pow(2)
        patch_ssd = F.avg_pool3d(
            patch_ssd,
            kernel_size=2 * self.radius + 1,
            stride=1,
            padding=self.radius,
        )

        patch_ssd = patch_ssd - patch_ssd.amin(dim=1, keepdim=True)
        mind_var = patch_ssd.mean(dim=1, keepdim=True)
        mind_var = torch.clamp(mind_var, min=self.eps)
        mind_var = torch.clamp(mind_var, min=mind_var.detach().mean() * 1e-3, max=mind_var.detach().mean() * 1e3)

        descriptor = torch.exp(-patch_ssd / mind_var)
        descriptor = descriptor / (descriptor.sum(dim=1, keepdim=True) + self.eps)
        return descriptor

    def forward(self, y_pred, y_true, mask=None):
        mind_pred = self._mind_ssc(y_pred)
        mind_true = self._mind_ssc(y_true)
        mse = (mind_pred - mind_true) ** 2
        
        if mask is None:
            # 自动生成前景 Mask，过滤掉医疗图像中大面积全黑背景区域的异常平方差运算
            bg_val = y_true.amin()  # 自动推断背景值 (一般是 0 或 -1)
            mask = (y_true > bg_val + 1e-3) | (y_pred > bg_val + 1e-3)
            
        mse = mse * mask
        # 归一化仅针对前景区域
        return torch.sum(mse) / (mask.sum() * mse.shape[1] + 1e-8)


class JointMIMINDLoss(nn.Module):
    """
    Combines Mutual Information (global alignment) and MIND-SSC (local fine-grained alignment).
    MI returns positive scalar to maximize -> Negated here for minimization.
    MIND returns MSE distance to minimize -> Added normally.
    """
    def __init__(self, mi_bins=32, mind_radius=2, mind_dilation=2, mind_eps=1e-5, mi_weight=1.0, mind_weight=1.0):
        super(JointMIMINDLoss, self).__init__()
        self.mi_loss = MutualInformation(num_bin=mi_bins)
        self.mind_loss = MINDLoss(radius=mind_radius, dilation=mind_dilation, eps=mind_eps)
        self.mi_weight = mi_weight
        self.mind_weight = mind_weight
        
    def forward(self, target, source):
        # MutualInformation already returns -MI, so we minimize it directly
        loss_mi = self.mi_loss(target, source)
        loss_mind = self.mind_loss(target, source)
        
        return self.mi_weight * loss_mi + self.mind_weight * loss_mind


class BoundaryConsistencyLoss(nn.Module):
    """
    Boundary consistency loss on fixed gradient-magnitude maps.
    """

    def __init__(
        self,
        operator: str = 'sobel',
        metric: str = 'ncc',
        smooth_kernel_size: int = 3,
        normalize_edges: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        if metric not in {'ncc', 'l1'}:
            raise ValueError(f"metric must be 'ncc' or 'l1', got {metric!r}")

        self.metric = metric
        self.eps = eps
        self.extractor = FixedGradientMagnitude3D(
            operator=operator,
            smooth_kernel_size=smooth_kernel_size,
            normalize=normalize_edges,
            eps=eps,
        )

    def _prepare_mask(self, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if mask.shape[2:] != reference.shape[2:]:
            mask = F.interpolate(mask, size=reference.shape[2:], mode='nearest')
        if mask.shape[1] == 1 and reference.shape[1] != 1:
            mask = mask.expand(-1, reference.shape[1], -1, -1, -1)
        return mask.float()

    def _masked_ncc(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)
        else:
            mask = self._prepare_mask(mask, pred)

        reduce_dims = (2, 3, 4)
        weight_sum = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(self.eps)

        pred_mean = (pred * mask).sum(dim=reduce_dims, keepdim=True) / weight_sum
        target_mean = (target * mask).sum(dim=reduce_dims, keepdim=True) / weight_sum

        pred_centered = (pred - pred_mean) * mask
        target_centered = (target - target_mean) * mask

        numerator = (pred_centered * target_centered).sum(dim=reduce_dims)
        pred_var = pred_centered.square().sum(dim=reduce_dims)
        target_var = target_centered.square().sum(dim=reduce_dims)
        corr = numerator / (torch.sqrt(pred_var * target_var) + self.eps)
        return 1.0 - corr.mean()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        pred_edges = self.extractor(prediction.float())
        target_edges = self.extractor(target.float())

        if self.metric == 'l1':
            if mask is None:
                return torch.mean(torch.abs(pred_edges - target_edges))

            mask = self._prepare_mask(mask, pred_edges)
            diff = torch.abs(pred_edges - target_edges) * mask
            return diff.sum() / mask.sum().clamp_min(self.eps)

        return self._masked_ncc(pred_edges, target_edges, mask=mask)
