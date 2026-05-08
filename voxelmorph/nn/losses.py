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
        
        # 3D 12-neighborhood definition (MIND-SSC)
        # Represents shifts: (x, y, z)
        six_neighborhood = torch.Tensor([[0, 1, 0],
                                         [1, 0, 0],
                                         [0, 0, 1],
                                         [0,-1, 0],
                                         [-1,0, 0],
                                         [0, 0,-1]])
        
        # Generates the 12 edges of the cube (SSC patch differences)
        import math
        dist = F.pdist(six_neighborhood)
        edges = torch.combinations(torch.arange(6), 2)[dist == math.sqrt(2)]
        
        # Create displacement variables
        displacement = six_neighborhood[edges[:, 0]] - six_neighborhood[edges[:, 1]]
        
        # Shift masks for Conv3d
        self.register_buffer('displacement', displacement.view(12, 3, 1, 1, 1).long())
        
    def forward(self, y_true, y_pred):
        """
        Computes the MSE of the MIND-SSC features between the fixed and moving images.
        """
        return torch.mean((self.mind_ssc(y_true) - self.mind_ssc(y_pred)) ** 2)

    def mind_ssc(self, img):
        """
        Compute the MIND-SSC descriptor for a 3D image.
        """
        # Padding to avoid border issues during shifts
        pad = self.dilation
        img_padded = F.pad(img, (pad, pad, pad, pad, pad, pad), mode='replicate')
        
        # Compute patch distances
        Dp = []
        for i in range(12):
            dx, dy, dz = (self.displacement[i, :, 0, 0, 0] * self.dilation).tolist()
            
            # Shift image
            shifted_img = img_padded[
                :, :, 
                pad+dz : img_padded.shape[2]-pad+dz,
                pad+dy : img_padded.shape[3]-pad+dy,
                pad+dx : img_padded.shape[4]-pad+dx
            ]
            
            # Squared difference metric
            Dp.append((img - shifted_img)**2)
            
        Dp = torch.cat(Dp, dim=1)
        
        # Compute local variance using average pooling (equivalent to uniform convolution)
        kernel = torch.ones([1, 1, 3, 3, 3], device=img.device) / 27.0
        
        # Grouped conv to apply mean filter channel-wise
        V = F.conv3d(Dp, kernel.repeat(12, 1, 1, 1, 1), padding=1, groups=12)
        
        # Estimate variance (mean of min distances)
        # Using the smallest 6 distances to approximate variance structurally
        V_min, _ = torch.min(V, dim=1, keepdim=True)
        # Added small epsilon for numerical stability
        V = V_min + 1e-5 

        # Calculate MIND-SSC features (exponentiated normalized distances)
        mind = torch.exp(-Dp / V)
        
        return mind
