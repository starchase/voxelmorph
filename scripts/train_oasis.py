#!/usr/bin/env python3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import argparse
import glob
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import neurite as ne
import matplotlib.pyplot as plt
import voxelmorph as vxm

from data import datasets, trans
import utils


import matplotlib.pyplot as plt


class VxmBaselineAdapter(nn.Module):
    """Adapter that exposes the standard Voxelmorph baseline through the same interface as the Siamese model."""

    def __init__(self, inshape, enc_nf, dec_nf, int_steps=0, device='cpu'):
        super().__init__()
        self.inshape = inshape
        self.ndim = 3
        self.use_pyramid = False
        self.model = vxm.nn.models.VxmPairwise(
            ndim=3,
            source_channels=1,
            target_channels=1,
            nb_features=[enc_nf, dec_nf],
            integration_steps=int_steps,
            device=device,
        )
        self.integrate = getattr(self.model, 'velocity_field_integrator', None)
        self.spatial_transform = self.model.spatial_transformer

    def forward(
        self,
        source,
        target,
        return_warped_source=True,
        return_warped_target=False,
        return_field_type='displacement',
        return_coarse_flows=False,
    ):
        out = self.model(
            source,
            target,
            return_warped_source=return_warped_source,
            return_warped_target=return_warped_target,
            return_field_type=return_field_type,
        )

        outputs = list(out) if isinstance(out, tuple) else [out]
        if return_coarse_flows:
            outputs.append([])
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


def build_registration_model(args, device):
    """Create either the standard Voxelmorph baseline or the current Siamese dual-stream model."""
    enc_channels = [32, 64, 64, 64]
    dec_channels = [64, 64, 64, 32]
    inshape = (160, 192, 224)

    if args.model_config == 'voxelmorph_baseline':
        ignored_flags = []
        for flag in ('use_pdaps', 'use_daps', 'use_dsin', 'use_cmim', 'use_wmca', 'use_pyramid'):
            if getattr(args, flag):
                ignored_flags.append(f'--{flag.replace("_", "-")}')
        if ignored_flags:
            print(
                '[INFO] voxelmorph_baseline mode ignores Siamese-specific options: '
                + ', '.join(ignored_flags)
            )

        model = VxmBaselineAdapter(
            inshape=inshape,
            enc_nf=enc_channels,
            dec_nf=dec_channels,
            int_steps=args.integration_steps,
            device=device,
        )
    else:
        model = vxm.nn.SiameseUNetBaseline(
            inshape=inshape,
            in_channels=1,
            enc_nf=enc_channels,
            dec_nf=dec_channels,
            ndim=3,
            int_steps=args.integration_steps,
            decouple_layers=args.decouple_layers,
            use_daps=args.use_daps,
            use_pdaps=args.use_pdaps,
            use_dsin=args.use_dsin,
            use_cmim=args.use_cmim,
            use_wmca=args.use_wmca,
            use_pyramid=args.use_pyramid
        )

    return model.to(device)

def save_qualitative_results(model, dataset, output_dir, epoch, device='cuda', suffix='', best_sample_idx=None):
    """Save mid-slice images of samples."""
    
    samples_to_plot = []
    
    # Get the dataset from the dataloader if needed, but we'll adapt to just take a list of data items
    # For OASIS dataset, data is (x, y, x_seg, y_seg)
    
    # Default behavior for validation: Index 0
    sample_default = dataset[0]
    samples_to_plot.append(('default', sample_default))
        
    for name_tag, sample in samples_to_plot:
        # Sample contains: x, y, x_seg, y_seg
        source = sample[0].unsqueeze(0).to(device)
        target = sample[1].unsqueeze(0).to(device)
        source_label = sample[2].unsqueeze(0).to(device)
        target_label = sample[3].unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(source, target, return_warped_source=True, return_field_type='displacement')
            displacement, warped_source = out[0], out[1]
            
            warped_label = None
            if source_label is not None:
                 import voxelmorph as vxm
                 trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                 warped_label = trf(source_label.float(), displacement)
        
        # Determine the best slice index
        # Default: middle slice
        slice_idx = source.shape[4] // 2
        
        # Extract slices using the determined index
        def get_slice(img_tensor, z_idx, is_label=False):
            if img_tensor is None: return None
            # img_tensor: (1, 1, X, Y, Z) (or similar)
            slice_tensor = img_tensor[:, 0, :, :, z_idx] if len(img_tensor.shape) == 5 else img_tensor[:, :, :, z_idx]
            # Remove batch dims
            slice_np = slice_tensor.detach().cpu().numpy()[0]
            if len(slice_np.shape) == 3: # if channel dim still exists
                slice_np = slice_np[0]
                
            if is_label:
                # Filter labels to only include: 
                # Lateral Ventricles (3, 22)
                # 3rd Ventricle (11), Thalamus (7, 26), Hippocampus (14, 30)
                labels_to_keep = [3, 7, 11, 14, 22, 26, 30]
                mask = np.isin(slice_np, labels_to_keep)
                slice_np = np.where(mask, slice_np, 0)
                
            # X-Y plane is Axial
            return np.rot90(slice_np, -1)

        src_slice = get_slice(source, slice_idx)
        tgt_slice = get_slice(target, slice_idx)
        warped_slice = get_slice(warped_source, slice_idx)
        
        src_lbl_slice = get_slice(source_label, slice_idx, is_label=True)
        tgt_lbl_slice = get_slice(target_label, slice_idx, is_label=True)
        warped_lbl_slice = get_slice(warped_label, slice_idx, is_label=True)
        
        has_labels = (src_lbl_slice is not None) and (tgt_lbl_slice is not None)
        
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        
        for ax in axes.flatten():
            ax.axis('off')
            
        # Determine global min and max for consistent brightness plotting
        vmax_val = max(np.max(src_slice), np.max(tgt_slice), np.max(warped_slice))
        vmin_val = min(np.min(src_slice), np.min(tgt_slice), np.min(warped_slice))
        
        # --- Row 1: Images & Differences ---
        
        # [0,0] Source Image
        axes[0, 0].imshow(src_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[0, 0].set_title('Source Image')
        axes[0, 0].axis('off')

        # [0,1] Target Image
        axes[0, 1].imshow(tgt_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')

        # [0,2] Diff: Moving - Fixed
        diff_moving_fixed = src_slice - tgt_slice
        im_diff1 = axes[0, 2].imshow(diff_moving_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 2].set_title('Diff: Source - Target')
        axes[0, 2].axis('off')

        # [0,3] Diff: Deformed - Target
        diff_warp_fixed = warped_slice - tgt_slice
        im_diff2 = axes[0, 3].imshow(diff_warp_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 3].set_title('Diff: Deformed - Target')
        axes[0, 3].axis('off')
        
        # --- Row 2: Label Overlays & Result vs GT ---
        if has_labels:
            def get_color(lbl, is_fixed=False):
                # Group Left/Right variants of the same structure to the same color
                lbl_map = {3: 0, 22: 0, 7: 1, 26: 1, 11: 2, 14: 3, 30: 3}
                mapped_lbl = lbl_map.get(int(lbl), int(lbl))
                
                # Hand-picked colors to ensure high visibility on gray images
                # Format: (Source/Moving color, Target/Fixed color)
                # Ensure they are same color family but explicitly distinguishable and both bright
                color_pairs = {
                    0: ('#1f77b4', '#00bfff'), # Dark Blue vs Deep Sky Blue
                    1: ('#ff7f0e', '#ffd700'), # Orange vs Gold/Yellow
                    2: ('#2ca02c', '#32cd32'), # Green vs Lime Green
                    3: ('#d62728', '#ff69b4'), # Red vs Hot Pink
                }
                
                if mapped_lbl in color_pairs:
                    hex_color = color_pairs[mapped_lbl][1 if is_fixed else 0]
                else:
                    hex_color = '#ffffff'
                    
                import matplotlib.colors as mcolors
                return mcolors.to_rgba(hex_color)

            def plot_label_contour(ax, bg_img, label_img, title, is_fixed=False):
                ax.imshow(bg_img, cmap='gray', vmin=vmin_val, vmax=vmax_val)
                if label_img is not None:
                     unique_labels = np.unique(label_img)
                     unique_labels = unique_labels[unique_labels > 0]
                     
                     for lbl in unique_labels:
                         mask = (label_img == lbl)
                         c = get_color(lbl, is_fixed=is_fixed)
                         if np.any(mask):
                             ax.contour(mask, colors=[c], linewidths=1.2)
                ax.set_title(title)
                ax.axis('off')

            plot_label_contour(axes[1, 0], src_slice, src_lbl_slice, 'Source + Labels', is_fixed=False)
            plot_label_contour(axes[1, 1], tgt_slice, tgt_lbl_slice, 'Target + Labels', is_fixed=True)
            plot_label_contour(axes[1, 2], warped_slice, warped_lbl_slice, 'Deformed + Labels', is_fixed=False)
            
        # [1,3] Overlay Warped Label on Target Image
        axes[1, 3].imshow(warped_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        if has_labels:
             unique_labels = np.unique(np.concatenate([tgt_lbl_slice, warped_lbl_slice]))
             unique_labels = unique_labels[unique_labels > 0]
             
             for lbl in unique_labels:
                 mask_tgt = (tgt_lbl_slice == lbl)
                 c_fixed = get_color(lbl, is_fixed=True)
                 if np.any(mask_tgt):
                     axes[1, 3].contour(mask_tgt, colors=[c_fixed], linewidths=1.5, linestyles='dashed', alpha=0.8)
                     
                 mask_warp = (warped_lbl_slice == lbl)
                 c_moving = get_color(lbl, is_fixed=False)
                 if np.any(mask_warp):
                     axes[1, 3].contour(mask_warp, colors=[c_moving], linewidths=1.5, linestyles='solid')

        axes[1, 3].set_title('Result vs GT')
        axes[1, 3].axis('off')

        # --- Row 3: Flow, Grid & Jacobian ---
        
        H, W = src_slice.shape
        grid_spacing = 10
        raw_d_slice = displacement.detach().cpu().numpy()[0, :, :, :, slice_idx] 
        d_slice = np.rot90(raw_d_slice, -1, axes=(1, 2))
        d_slice[1] = -d_slice[1] # flip Y displacement to match 180 deg rotation
        d_slice[2] = -d_slice[2] # flip X displacement to match 180 deg rotation

        dx, dy, dz = d_slice[2], d_slice[1], d_slice[0]
        max_mag = np.max(np.abs(d_slice)) + 1e-5
        
        flow_vis = np.zeros((H, W, 3), dtype=np.float32)
        flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5
        flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5
        flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5
        flow_vis = np.clip(flow_vis, 0, 1)
        
        # [2,0] Flow
        axes[2, 0].imshow(flow_vis)
        axes[2, 0].set_title('RGB Displacement')
        axes[2, 0].axis('off')

        from matplotlib.colors import hsv_to_rgb
        
        # [2,1] 3D Vector Legend
        ax_legend_spot = axes[2, 1]
        ax_legend_spot.clear()
        ax_legend_spot.axis('off')
        ax_legend_spot.set_aspect('equal')
        ax_legend_spot.set_xlim(-1.2, 1.2)
        ax_legend_spot.set_ylim(-1.2, 1.2)

        x_wheel = np.linspace(-0.12, 0.12, 100) 
        y_wheel = np.linspace(-0.12, 0.12, 100)
        XW, YW = np.meshgrid(x_wheel, y_wheel)
        RW = np.sqrt(XW**2 + YW**2)
        TW = np.arctan2(YW, XW)
        TW[TW < 0] += 2*np.pi
        
        HW = TW / (2*np.pi)
        SW = np.ones_like(HW)
        VW = np.ones_like(HW)
        mask = (RW <= 0.12)
        
        HSV_W = np.stack((HW, SW, VW), axis=-1)
        RGB_W = hsv_to_rgb(HSV_W)
        RGBA_W = np.concatenate([RGB_W, mask[..., None].astype(float)], axis=-1)
        
        ax_legend_spot.imshow(RGBA_W, extent=[-0.12, 0.12, -0.12, 0.12], origin='lower')

        o_x, o_y = 0, 0
        vec_x = np.array([0.5, -0.2])
        vec_y = np.array([-0.4, -0.25])
        vec_z = np.array([0.0, 0.5])
        
        scale = 0.312
        
        ax_legend_spot.arrow(o_x, o_y, vec_x[0]*scale, vec_x[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_y[0]*scale, vec_y[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_z[0]*scale, vec_z[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        
        ax_legend_spot.text(vec_x[0]*scale*1.6, vec_x[1]*scale*1.6, 'x', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_y[0]*scale*1.6, vec_y[1]*scale*1.6, 'y', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_z[0]*scale*1.6, vec_z[1]*scale*1.4, 'z', fontweight='bold', fontsize=16, ha='center', va='bottom')
        
        range_text = f"[{ -max_mag:.2f}, {max_mag:.2f}]"
        ax_legend_spot.text(0, -0.35, range_text, ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        # [2,2] Deformed Grid
        axes[2, 2].imshow(np.zeros_like(src_slice), cmap='gray', vmin=0, vmax=1) 
        # Plot vertical lines
        for i in range(0, W, grid_spacing):
            if i < d_slice.shape[2]:
                x_plot = i + d_slice[2, :, i]
                y_plot = np.arange(H) + d_slice[1, :, i]
                axes[2, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        # Plot horizontal lines
        for j in range(0, H, grid_spacing):
            if j < d_slice.shape[1]:
                x_plot = np.arange(W) + d_slice[2, j, :]
                y_plot = j + d_slice[1, j, :]
                axes[2, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        axes[2, 2].set_title('Deformed Grid')
        axes[2, 2].set_ylim(H, 0)
        axes[2, 2].set_xlim(0, W)
        axes[2, 2].axis('off')

        # [2,3] Jacobian Determinant
        disp_np = displacement.detach().cpu().numpy()[0]
        dz_dz, dz_dy, dz_dx = np.gradient(disp_np[0])
        dy_dz, dy_dy, dy_dx = np.gradient(disp_np[1])
        dx_dz, dx_dy, dx_dx = np.gradient(disp_np[2])
        
        jac_det = ( (1 + dx_dx) * ((1 + dy_dy) * (1 + dz_dz) - dy_dz * dz_dy)
                  - dx_dy * (dy_dx * (1 + dz_dz) - dy_dz * dz_dx)
                  + dx_dz * (dy_dx * dz_dy - (1 + dy_dy) * dz_dx) )
                  
        jac_slice = jac_det[:, :, slice_idx]
        jac_slice = np.rot90(jac_slice, -1)
        
        jac_vis = np.zeros((H, W, 3), dtype=np.float32)
        color_red = np.array([1.0, 0.0, 0.0])
        color_green = np.array([0.4, 0.8, 0.4]) 
        color_blue = np.array([0.4, 0.6, 0.9])  
        
        jac_vis[jac_slice < 0] = color_red
        jac_vis[(jac_slice >= 0) & (jac_slice <= 1)] = color_green
        jac_vis[jac_slice > 1] = color_blue
        
        axes[2, 3].imshow(jac_vis)
        axes[2, 3].set_title('Jacobian Determinant')
        axes[2, 3].axis('off')

        plt.suptitle(f'Epoch {epoch} - Sample {name_tag} (Slice Z={slice_idx})', fontsize=16)
        
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except UserWarning:
            pass
        
        safe_tag = str(name_tag).replace('/', '_').replace('\\', '_')
        filename_out = f'vis_epoch_{epoch:04d}{suffix}_{safe_tag}.png'
        out_file = output_dir / filename_out
        plt.savefig(str(out_file))
        plt.close(fig)

import scipy.ndimage

def compute_hd95(ground_truth, prediction, spacing=None):
    if ground_truth.sum() == 0 or prediction.sum() == 0:
        return np.nan
    pred_border = prediction ^ scipy.ndimage.binary_erosion(prediction)
    gt_border = ground_truth ^ scipy.ndimage.binary_erosion(ground_truth)
    pts_pred = np.argwhere(pred_border)
    pts_gt = np.argwhere(gt_border)
    if pts_pred.shape[0] == 0 or pts_gt.shape[0] == 0:
        return np.nan
    if spacing is not None:
        pts_pred = pts_pred * np.array(spacing)
        pts_gt = pts_gt * np.array(spacing)
    from scipy.spatial import cKDTree
    kd_tree_gt = cKDTree(pts_gt)
    distances_pred_to_gt, _ = kd_tree_gt.query(pts_pred)
    kd_tree_pred = cKDTree(pts_pred)
    distances_gt_to_pred, _ = kd_tree_pred.query(pts_gt)
    return max(np.percentile(distances_pred_to_gt, 95), np.percentile(distances_gt_to_pred, 95))

def validate(

    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    compute_extra: bool = False
) -> tuple:
    model.eval()
    eval_dsc = utils.AverageMeter()
    eval_hd95 = utils.AverageMeter()
    eval_jac = utils.AverageMeter()
    eval_mag = utils.AverageMeter()

    # Spatial transformation for nearest neighbour
    reg_model = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)

    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].to(device)
            x_seg = data[2].to(device)
            y_seg = data[3].to(device)

            # Get the displacement fields
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement'
            )
            displacement, warped_source = out[0], out[1]

            # Warp the segmentations with nearest neighbour
            def_out = reg_model(x_seg.float(), displacement)

            # DSC
            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            eval_dsc.update(dsc.item(), x.size(0))

            if compute_extra:
                # Magnitude
                disp_mag = torch.sqrt(torch.sum(displacement ** 2, dim=1))
                eval_mag.update(disp_mag.mean().item(), x.size(0))

                # Jacobian
                disp_np = displacement.cpu().numpy()
                disp_np = np.transpose(disp_np, (0, 2, 3, 4, 1))
                target_np = y.cpu().numpy()
                batch_neg_jac = 0.0
                for i in range(disp_np.shape[0]):
                    jac_det = vxm.py.utils.jacobian_determinant(disp_np[i])
                    mask = target_np[i, 0] > 0.01
                    if jac_det.shape != mask.shape:
                        diff = np.array(mask.shape) - np.array(jac_det.shape)
                        ds, hs, ws = diff // 2
                        de, he, we = mask.shape[0] - (diff[0]-ds), mask.shape[1] - (diff[1]-hs), mask.shape[2] - (diff[2]-ws)
                        mask = mask[ds:de, hs:he, ws:we]
                    valid_sum = np.sum(mask)
                    if valid_sum > 0:
                        batch_neg_jac += np.sum((jac_det <= 0) & mask) / valid_sum
                eval_jac.update(batch_neg_jac / disp_np.shape[0], x.size(0))

                # HD95
                wl_np = def_out.cpu().numpy()
                tl_np = y_seg.cpu().numpy()
                batch_hd95_sum = 0.0
                batch_hd95_count = 0
                for b in range(wl_np.shape[0]):
                    u_labels = np.unique(np.concatenate((wl_np[b], tl_np[b])))
                    u_labels = u_labels[u_labels > 0.5]
                    for l in u_labels:
                        mask_pred = (wl_np[b, 0] == l)
                        mask_gt = (tl_np[b, 0] == l)
                        hd = compute_hd95(mask_gt, mask_pred)
                        if not np.isnan(hd):
                            batch_hd95_sum += hd
                            batch_hd95_count += 1
                if batch_hd95_count > 0:
                    eval_hd95.update(batch_hd95_sum / batch_hd95_count, x.size(0))

    if compute_extra:
        return eval_dsc.avg, eval_hd95.avg, eval_jac.avg, eval_mag.avg
    return eval_dsc.avg

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: list,
    pyramid_weight: float = 0.5,
    device: str = 'cuda',
    scaler = None,
    amp_enabled: bool = True,
    use_mask: bool = False,
    loss_type: str = 'mse'
) -> float:
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        # TransMorph OASISDataset returns: x, y, x_seg, y_seg
        # x: moving image, y: fixed image
        x = data[0].to(device)
        y = data[1].to(device)
        # y_seg 常作为脑内部组织的金标准，适合当做高信度的前景色 Mask 候选
        y_seg = data[3].to(device)

        # 使用 AMP autocast
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            # Get the displacement and the warped source image from the model
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=True
            )
            displacement, warped_source = out[0], out[1]
            coarse_flows = out[2] if len(out) > 2 else []

            # AMP 兼容性保护：强制将预测结果和 Loss 计算切回 float32
            target_float = y.float()
            warped_float = warped_source.float()
            
            # --- Mask 核心逻辑联动 ---
            # Q: 前景 Mask 的过滤规则如果是按 > 0.01 背景排查，图像内部会有 0 体素吗？
            # A: 会。脑室（CSF/脑脊液区域）、肿瘤病灶、或者是扫描伪影在某些 MRI 模态(如T1) 中，部分像素可能天然表现为黑(接近0)。
            # 由于简单的硬阈值 `> 0.01` 会不小心在内部抠出“空洞”，一般用下述 2 种方案处理：
            # 方法A (推荐): 如果你的数据集里自带真实器官的 Label `y_seg`，直接拿全器官 Label 生成 Mask `(y_seg > 0).float()` 即可完美覆盖目标实质区域！
            # 方法B (常规): 依然用阈值提取背景，但做膨胀/闭运算填补内部空洞(morphological hole filling)。但深度学习中往往算算算嫌麻烦。
            
            if use_mask:
                # 方案：既然你有 y_seg，直接使用有标注的组织所在合集作为精准的前景 Mask！
                fg_mask = (y_seg > 0).float()
                # 兜底：如果有些批次 y_seg 全0失效了，退化回阈值硬扣。这会保护防止分母为 0。
                if fg_mask.sum() < 1e-3:
                    fg_mask = (target_float > 0.01).float()
            else:
                # 也就是默认在全图 (Batchx1xHxWxD) 上一视同仁全部计算 Loss
                fg_mask = torch.ones_like(target_float)
            
            if loss_type == 'mse':
                # 手动计算前景/全局的加权 MSE
                squared_diff = (target_float - warped_float) ** 2
                img_loss = (squared_diff * fg_mask).sum() / (fg_mask.sum() + 1e-8)
            elif loss_type == 'ncc':
                if use_mask:
                    # 屏蔽掉非脑范围，强制外围全黑，中心有效，使得 NCC 计算更稳定且聚焦大脑
                    masked_target = target_float * fg_mask
                    masked_warped = warped_float * fg_mask
                    img_loss = -image_loss_fn(masked_target, masked_warped).mean()
                else:
                    img_loss = -image_loss_fn(target_float, warped_float).mean()
                
            grad_loss = grad_loss_fn(displacement.float()).mean()
            
            # --- Deep Supervision for Pyramid/Coarse flows ---
            deep_sup_loss = displacement.new_tensor(0.0)
            if len(coarse_flows) > 0:
                for c_flow in coarse_flows:
                    # Scale to full resolution to evaluate image metric directly
                    c_shape = c_flow.shape[2:]
                    t_shape = displacement.shape[2:]
                    if c_shape != t_shape:
                        scale_factor = t_shape[0] / c_shape[0] # assuming square/cube
                        mode = 'trilinear' if len(c_shape) == 3 else 'bilinear'
                        c_flow_up = torch.nn.functional.interpolate(c_flow.float(), size=t_shape, mode=mode, align_corners=False) * scale_factor
                    else:
                        c_flow_up = c_flow.float()
                        
                    c_grad_loss = grad_loss_fn(c_flow_up).mean()
                    
                    # Warp using intermediate flow
                    if getattr(model, 'integrate', None) is not None:
                        c_disp = model.integrate(c_flow_up)
                    else:
                        c_disp = c_flow_up
                    c_warp = model.spatial_transform(x.float(), c_disp)
                    
                    # 金字塔中间层的损失约束同样跟随 user 的 Loss 策略
                    c_warp_float = c_warp.float()
                    if loss_type == 'mse':
                        c_squared_diff = (target_float - c_warp_float) ** 2
                        c_img_loss = (c_squared_diff * fg_mask).sum() / (fg_mask.sum() + 1e-8)
                    else: # loss_type == 'ncc'
                        if use_mask:
                            masked_c_target = target_float * fg_mask
                            masked_c_warp = c_warp_float * fg_mask
                            c_img_loss = -image_loss_fn(masked_c_target, masked_c_warp).mean()
                        else:
                            c_img_loss = -image_loss_fn(target_float, c_warp_float).mean()

                    deep_sup_loss = deep_sup_loss + c_img_loss + loss_weights[1] * c_grad_loss
                        
                # 对整体深度监督求平均，并打上折扣权重（默认0.5）
                deep_sup_loss = (deep_sup_loss / len(coarse_flows)) * pyramid_weight

            loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
            loss = loss + deep_sup_loss # Add deep supervision component

        # 数值稳定性保护：发现非有限值则跳过该 batch，避免污染整轮 loss
        if not torch.isfinite(loss):
            print(
                f"[WARN] Non-finite loss at batch {batch_idx}: "
                f"img_loss={img_loss.item()}, grad_loss={grad_loss.item()}, total={loss.item()}"
            )
            # 彻底释放包含 NaN/Inf 计算图的所有局部变量，防止在遇到 NaN 直接 continue 时显存泄漏引发后续 OOM
            optimizer.zero_grad(set_to_none=True)
            del out, displacement, warped_source, coarse_flows, loss, img_loss, grad_loss, deep_sup_loss
            if 'c_warp' in locals():
                del c_disp, c_warp, c_img_loss, c_grad_loss, c_flow_up
            torch.cuda.empty_cache()
            continue

        if amp_enabled and scaler is not None:
            # 缩放 loss，反向传播
            scaler.scale(loss).backward()
            # 在执行梯度裁剪前，必须先 unscale 梯度
            scaler.unscale_(optimizer)
            # 增加梯度裁剪，防止黑背景区导致的除零或梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item()
        valid_batches += 1

    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches

def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('--output', type=str, default='/root/autodl-tmp/models/oasis_vxm.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ncc'], help='Image similarity loss')
    parser.add_argument('--use-mask', action='store_true', help='Use foreground mask to exclude black background in loss calculation')
    parser.add_argument('--disable-amp', action='store_true', help='Force disable AMP (Automatic Mixed Precision)')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01, help='Weight of gradient loss')
    parser.add_argument('--pyramid-weight', type=float, default=0.5, help='Weight for intermediate pyramid deep supervision loss')
    parser.add_argument('--use-pdaps', action='store_true', help='Use Pyramid-guided Deformation-Aware Progressive Skip')
    parser.add_argument('--use-daps', action='store_true', help='Use original DAPS')
    parser.add_argument('--use-dsin', action='store_true', help='Enable DSIN in the shallow decoupled encoder layers')
    parser.add_argument('--use-cmim', action='store_true', help='Enable CMIM at deep decoder scales')
    parser.add_argument('--use-wmca', action='store_true', help='Enable window cross-attention on shallow skip features')
    parser.add_argument('--use-pyramid', action='store_true', help='Enable pyramid coarse-to-fine flow prediction')
    parser.add_argument('--model-config', type=str, default='dual_stream', choices=['dual_stream', 'voxelmorph_baseline'], help='Choose between the current dual-stream Siamese setup and the standard Voxelmorph baseline')
    parser.add_argument('--decouple-layers', type=int, default=2, help='Number of shallow decoupled encoder layers used in dual-stream mode')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--fusion-method', type=str, default='compress_concat', choices=['add', 'concat', 'compress_concat'], help='Feature fusion method for Siamese encoder')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument('--warm-start', type=int, default=10, help='Early stopping warm start steps')
    parser.add_argument('--integration-steps', type=int, default=0, help='number of integration steps for diffeomorphic registration')
    parser.add_argument('--train-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    args = parser.parse_args()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model
    model = build_registration_model(args, device)
    print(f'Model config: {args.model_config}')

    # 统计并打印参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Total Trainable Parameters: {total_params:,}')

    # ==========================
    # AMP & Loss & Mask 核心联动策略
    # ==========================
    # 1. 损失函数策略
    if args.loss.lower() == 'ncc':
        # 增大 eps (默认是 1e-5)，防止由于图像大面积黑色背景(方差近乎 0)导致的除零/梯度爆炸
        image_loss_fn = ne.nn.modules.NCC(eps=1e-3)
    else:
        image_loss_fn = ne.nn.modules.MSE()
        
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler: Cosine annealing to gradually lower LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 2. AMP 策略与防爆保护
    amp_enabled = (device == 'cuda') and not args.disable_amp
    if amp_enabled and (args.loss.lower() == 'ncc') and (args.integration_steps > 0):
        print("\n\n[WARNING 🚨] 检测到高危配置组合：AMP(FP16) + NCC + DiffoIntegration(>0)")
        print("          微分同胚 7次 Squaring 极易在 FP16 的浮点指数上乘爆，同时 NCC 的方差计算本身更容易溢出。")
        print("          为了防止 NaN 崩盘，系统已强制关闭当前的 AMP 训练。\n\n")
        amp_enabled = False
        
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    # Dataloader identical to TransMorph
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    
    train_pattern = os.path.join(args.train_dir, '*.pkl')
    val_pattern = os.path.join(args.val_dir, '*.pkl')
    train_set = datasets.OASISBrainDataset(glob.glob(train_pattern), transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(glob.glob(val_pattern), transforms=val_composed)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    import datetime
    
    # Ensure output dir
    # Create a timestamped directory for this run to keep logs and checkpoints separate
    # Use Beijing Time (UTC+8)
    utc_now = datetime.datetime.utcnow()
    beijing_time = utc_now + datetime.timedelta(hours=8)
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    
    input_output_path = Path(args.output)

    # Structure: <parent>/<stem>_<timestamp>/<stem>.pt
    run_dir = input_output_path.parent / f"{input_output_path.stem}_{timestamp}"
    model_filename = f"{input_output_path.stem}{input_output_path.suffix}"
    out_path = run_dir / model_filename
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for this run: {out_path.parent}")

    # Logging setup
    log_file = out_path.parent / 'train_log.csv'
    
    # Save training configuration
    config_file = out_path.parent / 'config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Integration Steps: {args.integration_steps}\n")
        f.write(f"Arguments: {vars(args)}\n")

    # Initialize Logger
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_dsc', 'val_hd95', 'val_jac', 'val_mag'])

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_dsc = 0.0
    loss_history = []
    val_dsc_history = []
    
    epoch_times = []
    
    # 记录训练前的初始 GPU 显存
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            pyramid_weight=args.pyramid_weight,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            use_mask=args.use_mask,
            loss_type=args.loss.lower()
        )
        loss_history.append(avg_loss)
        
        # Calculate Validation metrics
        compute_extra = ((epoch + 1) % 5 == 0)
        val_res = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            compute_extra=compute_extra
        )
        if compute_extra:
            val_dsc, val_hd95, val_jac, val_mag = val_res
        else:
            val_dsc = val_res
            val_hd95, val_jac, val_mag = np.nan, np.nan, np.nan
            
        val_dsc_history.append(val_dsc)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        if compute_extra:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Val DSC: {val_dsc:.6f}, HD95: {val_hd95:.2f}, Jac: {val_jac:.4f}, Mag: {val_mag:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')
        else:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Val DSC: {val_dsc:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')

        # Save visualizations
        try:
            save_qualitative_results(model, val_set, out_path.parent, epoch=epoch+1, device=device)
        except Exception as e:
            print(f"Failed to save visualization: {e}")

        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_loss:.6f}", 
                f"{val_dsc:.6f}", 
                f"{val_hd95:.2f}" if compute_extra else "",
                f"{val_jac:.6f}" if compute_extra else "",
                f"{val_mag:.6f}" if compute_extra else ""
            ])

        # Early stopping check based on average loss
        if len(loss_history) >= args.warm_start + args.patience + 1:
            recent_losses = loss_history[-args.patience:]
            best_past_loss = min(loss_history[:-args.patience])
            if all(max(best_past_loss - loss, 0) < args.threshold for loss in recent_losses):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = out_path.parent / f'{out_path.stem}_epoch{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_path = out_path.parent / f'{out_path.stem}_best.pt'
            torch.save(model.state_dict(), best_path)
            print(f'Saved new best model with DSC: {best_dsc:.6f}')

        if (epoch + 1) % 10 == 0:
            try:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:red'
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Train Loss', color=color)
                ax1.plot(range(1, epoch + 2), loss_history, color=color, marker='o', markersize=4, label='Train Loss')
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Val DSC', color=color)
                ax2.plot(range(1, epoch + 2), val_dsc_history, color=color, marker='s', markersize=4, label='Val DSC')
                ax2.tick_params(axis='y', labelcolor=color)
                
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left' if loss_history[0] > loss_history[-1] else 'center right')
                
                plt.title(f'Learning Curves (Epoch 1 to {epoch + 1})')
                fig.tight_layout()
                ax1.grid(True, linestyle='--', alpha=0.6)
                
                plot_path = out_path.parent / f'learning_curves_epoch{epoch + 1}.png'
                plt.savefig(str(plot_path), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to save learning curve plot: {e}")

    # Save final model
    torch.save(model.state_dict(), out_path)
    print(f'Final model saved to {out_path}')
    
    finish_utc = datetime.datetime.utcnow()
    finish_beijing = finish_utc + datetime.timedelta(hours=8)
    finish_timestamp = finish_beijing.strftime('%Y%m%d_%H%M%S')
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    final_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
    
    with open(config_file, 'a') as f:
        f.write(f"End Time: {finish_timestamp}\n")
        f.write(f"Average Epoch Time: {avg_epoch_time:.2f} s\n")
        f.write(f"Peak GPU Memory: {final_peak_mem:.2f} MB\n")

if __name__ == '__main__':
    main()
