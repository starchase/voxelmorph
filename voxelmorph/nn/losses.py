"""
Loss functions for image registration.
"""

# Standard library imports
import math

# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np


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
class MIND(torch.nn.Module):
    """
    Modality Independent Neighbourhood Descriptor (MIND) Self-Similarity Context (SSC) Loss.
    Standard 3D implementation for PyTorch.
    Expects inputs of shape (B, C, D, H, W). Usually C=1.
    """
    def __init__(self, radius=2, dilation=2):
        super(MIND, self).__init__()
        self.radius = radius
        self.dilation = dilation
        
        six_neighborhood = torch.tensor(
            [[0, 1, 1],
             [1, 1, 0],
             [1, 0, 1],
             [1, 1, 2],
             [2, 1, 1],
             [1, 2, 1]],
            dtype=torch.long,
        )

        distances = torch.cdist(six_neighborhood.float(), six_neighborhood.float(), p=2)
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        edge_mask = (x > y) & torch.isclose(distances, torch.tensor(2.0).sqrt())

        shift1 = six_neighborhood[x[edge_mask]]
        shift2 = six_neighborhood[y[edge_mask]]

        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        for idx in range(12):
            mshift1[idx, 0, shift1[idx, 0], shift1[idx, 1], shift1[idx, 2]] = 1
            mshift2[idx, 0, shift2[idx, 0], shift2[idx, 1], shift2[idx, 2]] = 1

        self.register_buffer('mshift1', mshift1)
        self.register_buffer('mshift2', mshift2)
        
    def _forward_impl(self, y_true, y_pred, mask=None):
        loss_map = (self.mind_ssc(y_true) - self.mind_ssc(y_pred)) ** 2

        if mask is None:
            return torch.mean(loss_map)

        if mask.dim() != y_true.dim():
            raise ValueError(
                f'MIND mask must have shape compatible with inputs. Got mask {tuple(mask.shape)} '
                f'for input {tuple(y_true.shape)}.'
            )

        mask = mask.to(device=loss_map.device, dtype=loss_map.dtype)
        if mask.shape[1] == 1 and loss_map.shape[1] != 1:
            mask = mask.expand(-1, loss_map.shape[1], -1, -1, -1)

        masked_loss = loss_map * mask
        normalizer = mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def forward(self, y_true, y_pred, mask=None):
        """
        Computes the MSE of the MIND-SSC features between the fixed and moving images.
        Optionally restricts the loss to a foreground mask in target space.
        """
        return self._forward_impl(y_true, y_pred, mask=mask)

    def mind_ssc(self, img):
        """
        Compute the MIND-SSC descriptor for a 3D image.
        """
        if img.shape[1] != 1:
            raise ValueError(f'MIND expects single-channel inputs, got {img.shape[1]} channels.')

        padded_img = F.pad(
            img,
            [self.dilation] * 6,
            mode='replicate',
        )

        dist = (
            F.conv3d(padded_img, self.mshift1, dilation=self.dilation)
            - F.conv3d(padded_img, self.mshift2, dilation=self.dilation)
        ) ** 2
        dist = F.avg_pool3d(
            dist,
            kernel_size=2 * self.radius + 1,
            stride=1,
            padding=self.radius,
        )

        mind = dist - torch.min(dist, dim=1, keepdim=True)[0]
        mind_var = torch.mean(mind, dim=1, keepdim=True)
        mind_var_mean = torch.mean(mind_var.detach())
        mind_var = torch.clamp(mind_var, min=mind_var_mean * 0.001, max=mind_var_mean * 1000)
        mind = torch.exp(-mind / (mind_var + 1e-8))

        return mind[:, [6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], ...]
