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
import torch
import torch.nn as nn
import torch.nn.functional as F

class MINDLoss(nn.Module):
    """
    MIND-SSC (Modality Independent Neighbourhood Descriptor - Self Similarity Context) Loss.
    This calculates the MIND descriptor for both moving and fixed images and returns the Mean Squared Error between them.
    Suitable for multimodal image registration (e.g., CT-MRI).
    """
    def __init__(self, win=1, radius=2, dilation=2):
        super(MINDLoss, self).__init__()
        self.win = win
        self.radius = radius
        self.dilation = dilation
        
        # Define the displacement vectors (6-neighborhood in 3D)
        self.six_neighbourhood = torch.Tensor([[0,1,0], [1,0,0], [0,0,1], [0,-1,0], [-1,0,0], [0,0,-1]]).long()

    def pdist(self, x, y):
        return torch.exp(-torch.mean((x - y) ** 2, dim=1, keepdim=True))
        
    def _mind_ssc(self, img):
        # 1. Compute local variance (noise estimation)
        # Using a simple 3x3x3 smoothing average
        device = img.device
        kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
        
        # Pad image
        pad = 1
        img_pad = F.pad(img, (pad, pad, pad, pad, pad, pad), mode='replicate')
        
        # Local mean
        mean_img = F.conv3d(img_pad, kernel, padding=0)
        
        # Local variance
        var_img = F.conv3d((img_pad)**2, kernel, padding=0) - mean_img**2
        var_img = torch.clamp(var_img, min=1e-5)
        
        # 2. Compute MIND-SSC
        pad_d = self.dilation
        img_pad_d = F.pad(img, (pad_d, pad_d, pad_d, pad_d, pad_d, pad_d), mode='replicate')
        
        ssc = []
        for i in range(self.six_neighbourhood.shape[0]):
            shift = self.six_neighbourhood[i]
            # Shifted image
            shifted_img = img_pad_d[:, :, 
                pad_d + shift[0]*self.dilation : pad_d + shift[0]*self.dilation + img.shape[2],
                pad_d + shift[1]*self.dilation : pad_d + shift[1]*self.dilation + img.shape[3],
                pad_d + shift[2]*self.dilation : pad_d + shift[2]*self.dilation + img.shape[4]
            ]
            
            # Squared difference
            Dp = (img - shifted_img)**2
            
            # Smoothed squared difference
            Dp_smooth = F.conv3d(F.pad(Dp, (pad, pad, pad, pad, pad, pad), mode='replicate'), kernel, padding=0)
            
            # Normalized response (Self-Similarity Context)
            Cp = torch.exp(-Dp_smooth / var_img)
            ssc.append(Cp)
            
        ssc = torch.cat(ssc, dim=1)
        
        # Normalize sum to 1
        ssc = ssc / torch.max(ssc.max(dim=1, keepdim=True)[0], torch.tensor(1e-5, device=device))
        return ssc

    def forward(self, y_pred, y_true):
        # Expecting shape [B, C=1, D, H, W]
        mind_pred = self._mind_ssc(y_pred)
        mind_true = self._mind_ssc(y_true)
        
        # Return Mean Absolute Error or Mean Squared Error of descriptors
        return torch.mean((mind_pred - mind_true) ** 2)

