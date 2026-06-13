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


def symmetric_displacement_composition_loss(
    spatial_transform,
    forward_displacement,
    backward_displacement,
    forward_stages=(),
    backward_stages=(),
    stage_weight=0.5,
):
    """Robust inverse-composition consistency for final and coarse-to-fine fields."""

    def composition_loss(first, second):
        # The model composes displacements as second + warp(first, second).
        residual = second.float() + spatial_transform(first.float(), second.float())
        return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)

    final_loss = 0.5 * (
        composition_loss(forward_displacement, backward_displacement)
        + composition_loss(backward_displacement, forward_displacement)
    )

    paired_stages = list(zip(forward_stages, backward_stages))
    if not paired_stages or stage_weight <= 0:
        return final_loss

    stage_loss = final_loss.new_tensor(0.0)
    for forward_stage, backward_stage in paired_stages:
        stage_loss = stage_loss + 0.5 * (
            composition_loss(forward_stage, backward_stage)
            + composition_loss(backward_stage, forward_stage)
        )
    return final_loss + float(stage_weight) * stage_loss / len(paired_stages)


@torch.no_grad()
def random_diffeomorphic_displacement(
    reference,
    spatial_transform,
    max_displacement=3.0,
    coarse_scale=0.125,
    integration_steps=5,
):
    """Generate a smooth random displacement in full-resolution voxel units."""
    spatial_shape = reference.shape[2:]
    coarse_shape = tuple(max(4, int(round(size * coarse_scale))) for size in spatial_shape)
    velocity = torch.randn(
        reference.shape[0],
        len(spatial_shape),
        *coarse_shape,
        device=reference.device,
        dtype=torch.float32,
    )
    velocity = F.interpolate(velocity, size=spatial_shape, mode='trilinear', align_corners=True)
    velocity = float(max_displacement) * torch.tanh(velocity)
    displacement = velocity / (2 ** int(integration_steps))
    for _ in range(int(integration_steps)):
        displacement = displacement + spatial_transform(displacement, displacement)
    return displacement


def deformation_equivariance_loss(
    spatial_transform,
    base_displacement,
    perturbed_displacement,
    target_perturbation,
    structure_image=None,
    structure_aware=False,
    eps=1e-6,
):
    """Match a prediction to the known target-space deformation composition."""
    with torch.no_grad():
        expected = target_perturbation.float() + spatial_transform(
            base_displacement.detach().float(),
            target_perturbation.float(),
        )
        weight = None
        if structure_aware:
            if structure_image is None:
                raise ValueError('structure_image is required for structure-aware DESS')
            image = structure_image.detach().float()
            local_mean = F.avg_pool3d(image, kernel_size=5, stride=1, padding=2)
            local_second = F.avg_pool3d(image.square(), kernel_size=5, stride=1, padding=2)
            local_variance = (local_second - local_mean.square()).clamp_min(0.0)
            variance_scale = local_variance.mean(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
            structure_weight = local_variance / (local_variance + variance_scale)

            valid = spatial_transform(torch.ones_like(image), target_perturbation.float())
            valid_weight = ((valid - 0.999) / 0.001).clamp(0.0, 1.0)
            weight = structure_weight * valid_weight

    error = F.smooth_l1_loss(perturbed_displacement.float(), expected, beta=1.0, reduction='none')
    if weight is None:
        return error.mean()
    weight = weight.expand(-1, error.shape[1], *([-1] * (error.ndim - 2)))
    return (error * weight).sum() / weight.sum().clamp_min(eps)


class StructureAdaptiveOptimizationRegularization(nn.Module):
    """Structure-aware smoothness with high-deformation risk protection."""

    def __init__(self, alpha=3.0, risk_gamma=0.5, risk_threshold=0.5, eps=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.risk_gamma = float(risk_gamma)
        self.risk_threshold = float(risk_threshold)
        self.eps = float(eps)

    @staticmethod
    def _diff(value, dim):
        difference = value.diff(dim=dim)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim) + 1] = 1
        return F.pad(difference, tuple(pad), mode='replicate')

    def forward(self, displacement, fixed_image):
        displacement = displacement.float()
        fixed_image = fixed_image.float()
        image_gradient = torch.sqrt(sum(
            self._diff(fixed_image, dim).square() for dim in (2, 3, 4)
        ) + self.eps)
        gradient_scale = image_gradient.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        normalized_structure = (image_gradient / gradient_scale).clamp(max=5.0)
        structure_weight = torch.exp(-self.alpha * normalized_structure)

        flow_diffs = [self._diff(displacement, dim) for dim in (2, 3, 4)]
        flow_energy = sum(diff.square() for diff in flow_diffs).mean(dim=1, keepdim=True)
        risk_weight = torch.sigmoid(8.0 * (torch.sqrt(flow_energy + self.eps) - self.risk_threshold))
        weight = structure_weight + self.risk_gamma * risk_weight
        weight = weight / weight.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        return (weight * flow_energy).mean()


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


class BendingEnergyLoss(nn.Module):
    """
    Second-order bending energy regularization for displacement fields.

    Expects a displacement field with shape (B, C, *spatial_dims), where C is
    the number of spatial dimensions.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f'Unsupported reduction: {reduction!r}')
        self.reduction = reduction

    def _first_diff(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        head = [slice(None)] * field.dim()
        tail = [slice(None)] * field.dim()
        head[dim] = slice(1, None)
        tail[dim] = slice(None, -1)
        diff = field[tuple(head)] - field[tuple(tail)]

        pad = [0, 0] * (field.dim() - 2)
        axis = dim - 2
        pad_index = 2 * (field.dim() - 3 - axis)
        pad[pad_index + 1] = 1
        return F.pad(diff, tuple(pad))

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        if displacement.dim() < 4:
            raise ValueError(
                f'displacement must have shape (B, C, *spatial_dims), got {tuple(displacement.shape)}'
            )

        spatial_dims = list(range(2, displacement.dim()))
        first_diffs = {dim: self._first_diff(displacement, dim) for dim in spatial_dims}
        bending_terms = []

        for dim in spatial_dims:
            second = self._first_diff(first_diffs[dim], dim)
            bending_terms.append(second.pow(2))

        for idx, dim_i in enumerate(spatial_dims):
            for dim_j in spatial_dims[idx + 1:]:
                mixed = self._first_diff(first_diffs[dim_i], dim_j)
                bending_terms.append(2.0 * mixed.pow(2))

        loss = sum(bending_terms)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss.mean(dim=tuple(range(1, loss.dim())))
        return loss.mean()

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

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.register_buffer('vol_bin_centers', vol_bin_centers)

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
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
    deformable multimodal registration and matches the descriptor ordering used
    by the baseline branch. It builds self-similarity descriptors from pairwise
    patch SSDs in a 6-neighbourhood and compares descriptors with a voxel-wise
    squared error.
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
        distances = torch.cdist(neighbourhood.float(), neighbourhood.float(), p=2)
        first_index, second_index = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        pair_mask = (first_index > second_index) & torch.isclose(distances, torch.tensor(2.0).sqrt())
        shift_1 = neighbourhood[first_index[pair_mask]]
        shift_2 = neighbourhood[second_index[pair_mask]]

        kernel_1 = torch.zeros((shift_1.shape[0], 1, 3, 3, 3), dtype=torch.float32)
        kernel_2 = torch.zeros((shift_2.shape[0], 1, 3, 3, 3), dtype=torch.float32)
        for idx in range(shift_1.shape[0]):
            kernel_1[idx, 0, shift_1[idx, 0], shift_1[idx, 1], shift_1[idx, 2]] = 1.0
            kernel_2[idx, 0, shift_2[idx, 0], shift_2[idx, 1], shift_2[idx, 2]] = 1.0

        self.register_buffer('kernel_1', kernel_1)
        self.register_buffer('kernel_2', kernel_2)
        self.register_buffer('descriptor_permutation', torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long))

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
        mind_var_mean = mind_var.detach().mean()
        mind_var = torch.clamp(mind_var, min=mind_var_mean * 1e-3, max=mind_var_mean * 1e3)

        descriptor = torch.exp(-patch_ssd / (mind_var + self.eps))
        return descriptor[:, self.descriptor_permutation, ...]

    def forward(self, y_pred, y_true, mask=None):
        mind_pred = self._mind_ssc(y_pred)
        mind_true = self._mind_ssc(y_true)
        mse = (mind_pred - mind_true) ** 2
        
        if mask is None:
            return mse.mean()
            
        mask = mask.to(device=mse.device, dtype=mse.dtype)
        if mask.shape[1] == 1 and mse.shape[1] != 1:
            mask = mask.expand(-1, mse.shape[1], -1, -1, -1)

        mse = mse * mask
        return torch.sum(mse) / mask.sum().clamp_min(1.0)


class MIND(MINDLoss):
    """Backward-compatible alias for the baseline branch MIND-SSC loss."""


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
        
    def forward(self, target, source, mask=None):
        # MutualInformation already returns -MI, so we minimize it directly
        loss_mi = self.mi_loss(target, source)
        loss_mind = self.mind_loss(target, source, mask=mask)
        
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
