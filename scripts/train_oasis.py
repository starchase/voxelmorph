#!/usr/bin/env python3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import argparse
import glob
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
        return_residual_flows=False,
        return_feature_edge_loss=False,
        feature_edge_indices=(0, 1),
        swap_encoder_branches=False,
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
        if return_residual_flows:
            outputs.append([])
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


def build_registration_model(args, device):
    """Create either the standard Voxelmorph baseline or the current Siamese dual-stream model."""
    enc_channels = [32, 64, 64, 64]
    dec_channels = [64, 64, 64, 32]
    inshape = (160, 192, 224)

    if args.model_config == 'voxelmorph_baseline':
        ignored_flags = []
        for flag in ('use_pdaps', 'use_daps', 'use_dsin', 'use_cmim', 'use_cross_mamba', 'use_wcv', 'use_swcv', 'use_gcv', 'use_boundary_branch', 'use_ussc', 'use_dasr', 'use_drfc', 'use_sdmr', 'use_dpfc', 'use_miscv', 'use_sscc', 'use_cagr'):
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
            use_cross_mamba=args.use_cross_mamba,
            cross_mamba_scales=getattr(args, 'cross_mamba_scales', '1/16,1/8'),
            cross_mamba_offset_limit=getattr(args, 'cross_mamba_offset_limit', 0.0),
            cross_mamba_offset_smooth_kernel=getattr(args, 'cross_mamba_offset_smooth_kernel', 1),
            cross_mamba_use_resampling=getattr(args, 'cross_mamba_use_resampling', True),
            cross_mamba_structure_norm=getattr(args, 'cross_mamba_structure_norm', False),
            cross_mamba_residual_scale=getattr(args, 'cross_mamba_residual_scale', 1.0),
            use_wcv=args.use_wcv,
            use_swcv=args.use_swcv,
            use_gcv=args.use_gcv,
            encoder_type=getattr(args, 'encoder_type', 'cnn'),
            mamba_shallow_multi=getattr(args, 'mamba_shallow_multi', False),
            mamba_enc_shallow_multi=getattr(args, 'mamba_enc_shallow_multi', False),
            mamba_dec_shallow_multi=getattr(args, 'mamba_dec_shallow_multi', False),
            mamba_quarter_scale=getattr(args, 'mamba_quarter_scale', False),
            mamba_dec_quarter_scale=getattr(args, 'mamba_dec_quarter_scale', False),
            mamba_parallel_block=getattr(args, 'mamba_parallel_block', False),
            window_size=getattr(args, 'window_size', 9),
            pdaps_flow_limit=getattr(args, 'pdaps_flow_limit', 20.0),
            use_residual_flow_pyramid=getattr(args, 'use_residual_flow_pyramid', False),
            use_dpfc=getattr(args, 'use_dpfc', False),
            dpfc_flow_limit=getattr(args, 'dpfc_flow_limit', 8.0),
            use_miscv=getattr(args, 'use_miscv', False),
            miscv_scales=getattr(args, 'miscv_scales', '1/8,1/4'),
            miscv_projection_channels=getattr(args, 'miscv_projection_channels', 8),
            miscv_search_radius=getattr(args, 'miscv_search_radius', 2),
            miscv_temperature=getattr(args, 'miscv_temperature', 0.1),
            use_sscc=getattr(args, 'use_sscc', False),
            sscc_scales=getattr(args, 'sscc_scales', '1/16,1/8'),
            sscc_hidden_channels=getattr(args, 'sscc_hidden_channels', 16),
            sscc_strength=getattr(args, 'sscc_strength', 0.2),
            use_cagr=getattr(args, 'use_cagr', False),
            cagr_strength=getattr(args, 'cagr_strength', 0.5),
            use_error_guided_residual=getattr(args, 'use_error_guided_residual', False),
            residual_flow_limit=getattr(args, 'residual_flow_limit', 4.0),
            use_boundary_branch=getattr(args, 'use_boundary_branch', False),
            boundary_branch_scales=getattr(args, 'boundary_branch_scales', 'deep'),
            boundary_branch_strength=getattr(args, 'boundary_branch_strength', 0.5),
            boundary_kernel=getattr(args, 'boundary_kernel', 'sobel'),
            boundary_smooth_kernel=getattr(args, 'boundary_smooth_kernel', 3),
            use_ussc=getattr(args, 'use_ussc', False),
            ussc_scales=getattr(args, 'ussc_scales', '1/8'),
            ussc_search_radius=getattr(args, 'ussc_search_radius', 2),
            ussc_temperature=getattr(args, 'ussc_temperature', 0.1),
            ussc_guidance_strength=getattr(args, 'ussc_guidance_strength', 0.5),
            use_dasr=getattr(args, 'use_dasr', False),
            dasr_scales=getattr(args, 'dasr_scales', '1/2'),
            dasr_reduction=getattr(args, 'dasr_reduction', 4),
            dasr_strength=getattr(args, 'dasr_strength', 0.2),
            use_drfc=getattr(args, 'use_drfc', False),
            drfc_scale=getattr(args, 'drfc_scale', 0.25),
            drfc_hidden_channels=getattr(args, 'drfc_hidden_channels', 16),
            drfc_strength=getattr(args, 'drfc_strength', 0.5),
            use_sdmr=getattr(args, 'use_sdmr', False),
            sdmr_use_mind=getattr(args, 'sdmr_use_mind', False),
            sdmr_scale=getattr(args, 'sdmr_scale', 0.125),
            sdmr_hidden_channels=getattr(args, 'sdmr_hidden_channels', 16),
            sdmr_flow_limit=getattr(args, 'sdmr_flow_limit', 1.0),
            sdmr_alpha=getattr(args, 'sdmr_alpha', 0.5),
        )

    return model.to(device)

def save_qualitative_results(
    model,
    dataset,
    output_dir,
    epoch,
    device='cuda',
    suffix='',
    best_sample_idx=None,
    boundary_kernel='sobel',
    boundary_smooth_kernel=3,
    boundary_ring_inner_kernel=3,
    boundary_ring_outer_kernel=7,
):
    """Save mid-slice images of samples."""

    def _extract_error_guided_stage_slices(error_guided_maps, z_idx, reference_depth):
        stage_slices = []
        for item in error_guided_maps:
            error_map = item.get('error_map')
            gate_map = item.get('gate_map')
            if error_map is None or gate_map is None:
                continue
            map_depth = int(error_map.shape[4])
            if map_depth <= 1 or reference_depth <= 1:
                mapped_z_idx = 0
            else:
                mapped_z_idx = int(round(z_idx * (map_depth - 1) / (reference_depth - 1)))
            mapped_z_idx = max(0, min(mapped_z_idx, map_depth - 1))

            error_slice = np.rot90(error_map[0, 0, :, :, mapped_z_idx].detach().cpu().numpy(), -1)
            gate_slice = np.rot90(gate_map[0, 0, :, :, mapped_z_idx].detach().cpu().numpy(), -1)
            stage_slices.append({
                'stage_idx': item.get('stage_idx', -1),
                'error_slice': error_slice,
                'gate_slice': gate_slice,
            })
        stage_slices.sort(key=lambda x: x['stage_idx'])
        return stage_slices
    
    samples_to_plot = []
    
    # Get the dataset from the dataloader if needed, but we'll adapt to just take a list of data items
    # For OASIS dataset, data is (x, y, x_seg, y_seg)
    
    # Use medium validation sample for visualization: Index 9 (Initial DSC ~ 0.59)
    sample_default = dataset[9]
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
            error_guided_maps = list(getattr(model, 'latest_error_guided_residual_maps', []))
            boundary_extractor = vxm.nn.modules.FixedGradientMagnitude3D(
                operator=boundary_kernel,
                smooth_kernel_size=boundary_smooth_kernel,
                normalize=True,
            ).to(device)
            source_boundary = boundary_extractor(source.float())
            target_boundary = boundary_extractor(target.float())
            warped_boundary = boundary_extractor(warped_source.float())
            base_fg_mask = build_foreground_mask(target.float())
            boundary_ring_mask = build_boundary_ring_mask(
                base_fg_mask,
                inner_kernel_size=boundary_ring_inner_kernel,
                outer_kernel_size=boundary_ring_outer_kernel,
            )
            
            warped_label = None
            if source_label is not None:
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
        src_boundary_slice = get_slice(source_boundary, slice_idx)
        tgt_boundary_slice = get_slice(target_boundary, slice_idx)
        warped_boundary_slice = get_slice(warped_boundary, slice_idx)
        ring_mask_slice = get_slice(boundary_ring_mask, slice_idx)
        
        src_lbl_slice = get_slice(source_label, slice_idx, is_label=True)
        tgt_lbl_slice = get_slice(target_label, slice_idx, is_label=True)
        warped_lbl_slice = get_slice(warped_label, slice_idx, is_label=True)
        
        has_labels = (src_lbl_slice is not None) and (tgt_lbl_slice is not None)
        error_guided_stage_slices = _extract_error_guided_stage_slices(error_guided_maps, slice_idx, source.shape[4])
        
        rows = 5 if len(error_guided_stage_slices) > 0 else 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        
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

        # --- Row 4: Boundary Maps & Ring Mask ---
        boundary_vmax = max(
            np.max(src_boundary_slice),
            np.max(tgt_boundary_slice),
            np.max(warped_boundary_slice),
            1e-6,
        )

        axes[3, 0].imshow(src_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 0].set_title('Source Boundary Map')
        axes[3, 0].axis('off')

        axes[3, 1].imshow(tgt_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 1].set_title('Target Boundary Map')
        axes[3, 1].axis('off')

        axes[3, 2].imshow(warped_boundary_slice, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 2].set_title('Warped Boundary Map')
        axes[3, 2].axis('off')

        axes[3, 3].imshow(tgt_slice, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        axes[3, 3].imshow(ring_mask_slice, cmap='autumn', alpha=0.45, vmin=0.0, vmax=1.0)
        axes[3, 3].set_title('Boundary Ring Mask')
        axes[3, 3].axis('off')

        if len(error_guided_stage_slices) > 0:
            for plot_idx in range(4):
                axes[4, plot_idx].axis('off')

            for stage_pos, stage_item in enumerate(error_guided_stage_slices[:2]):
                col_offset = stage_pos * 2
                error_slice = stage_item['error_slice']
                gate_slice = stage_item['gate_slice']
                stage_idx = stage_item['stage_idx']

                error_vmax = max(float(np.max(error_slice)), 1e-6)
                axes[4, col_offset].imshow(error_slice, cmap='magma', vmin=0.0, vmax=error_vmax)
                axes[4, col_offset].set_title(f'Err Map (Stage {stage_idx})')
                axes[4, col_offset].axis('off')

                axes[4, col_offset + 1].imshow(gate_slice, cmap='viridis', vmin=0.5, vmax=1.5)
                axes[4, col_offset + 1].set_title(f'Gate Map (Stage {stage_idx})')
                axes[4, col_offset + 1].axis('off')

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
    compute_extra: bool = False,
    image_loss_fn: nn.Module = None,
    grad_loss_fn: nn.Module = None,
    loss_weights: list = None,
    image_loss_fn_coarse: nn.Module = None,
    boundary_loss_fn: nn.Module = None,
    boundary_loss_weight: float = 0.0,
    boundary_ring_inner_kernel: int = 3,
    boundary_ring_outer_kernel: int = 7,
    feature_edge_loss_weight: float = 0.0,
    feature_edge_indices: tuple = (0, 1),
    pyramid_weight: float = 0.5,
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
    use_mask: bool = False,
    loss_type: str = 'mse',
) -> tuple:
    model.eval()
    eval_loss = utils.AverageMeter()
    eval_boundary_loss = utils.AverageMeter()
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
            use_feature_edge_loss = feature_edge_loss_weight > 0 and hasattr(model, '_feature_edge_loss')
            use_residual_flow_pyramid = getattr(model, 'use_residual_flow_pyramid', False)
            use_dpfc = getattr(model, 'use_dpfc', False)
            supports_coarse_flows = getattr(model, 'use_pdaps', False) or use_residual_flow_pyramid or use_dpfc
            request_pyramid_outputs = supports_coarse_flows and (pyramid_weight > 0 or pyramid_image_weight > 0 or residual_flow_reg_weight > 0)
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=request_pyramid_outputs,
                return_residual_flows=(use_residual_flow_pyramid or use_dpfc) and request_pyramid_outputs,
                return_feature_edge_loss=use_feature_edge_loss,
                feature_edge_indices=feature_edge_indices,
            )
            displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss = unpack_model_outputs(
                out,
                expect_coarse_flows=request_pyramid_outputs,
                expect_residual_flows=(use_residual_flow_pyramid or use_dpfc) and request_pyramid_outputs,
                expect_feature_edge_loss=use_feature_edge_loss,
            )

            target_float = y.float()
            warped_float = warped_source.float()
            base_fg_mask = build_foreground_mask(target_float)
            boundary_ring_mask = build_boundary_ring_mask(
                base_fg_mask,
                inner_kernel_size=boundary_ring_inner_kernel,
                outer_kernel_size=boundary_ring_outer_kernel,
            )
            if use_mask:
                fg_mask = base_fg_mask
            else:
                fg_mask = torch.ones_like(target_float)

            boundary_loss = displacement.new_tensor(0.0)
            if image_loss_fn is not None and grad_loss_fn is not None and loss_weights is not None:
                if loss_type == 'mse':
                    squared_diff = (target_float - warped_float) ** 2
                    img_loss = (squared_diff * fg_mask).sum() / (fg_mask.sum() + 1e-8)
                elif loss_type == 'ncc':
                    if use_mask:
                        img_loss = -image_loss_fn(target_float * fg_mask, warped_float * fg_mask).mean()
                    else:
                        img_loss = -image_loss_fn(target_float, warped_float).mean()
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                grad_loss = grad_loss_fn(displacement.float()).mean()
                loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss

                deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss = compute_oasis_pyramid_losses(
                    model,
                    x.float(),
                    target_float,
                    fg_mask,
                    coarse_flows,
                    residual_flows,
                    image_loss_fn_coarse,
                    grad_loss_fn,
                    loss_weights,
                    use_mask,
                    loss_type,
                    device=device,
                    pyramid_weight=pyramid_weight,
                    pyramid_image_weight=pyramid_image_weight,
                    pyramid_image_weights=pyramid_image_weights,
                    residual_flow_reg_weight=residual_flow_reg_weight,
                )
                if deep_sup_loss.requires_grad or deep_sup_loss.item() != 0:
                    loss = loss + deep_sup_loss
                if pyramid_image_weight > 0:
                    loss = loss + pyramid_image_weight * pyramid_img_loss

                if boundary_loss_fn is not None and boundary_loss_weight > 0:
                    boundary_loss = boundary_loss_fn(warped_float, target_float, mask=boundary_ring_mask)
                    loss = loss + boundary_loss_weight * boundary_loss

                if feature_edge_loss_weight > 0:
                    loss = loss + feature_edge_loss_weight * feature_edge_loss.float()

                if residual_flow_reg_weight > 0:
                    loss = loss + residual_flow_reg_weight * residual_flow_reg_loss

                eval_loss.update(loss.item(), x.size(0))
                eval_boundary_loss.update(boundary_loss.item(), x.size(0))

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
        return eval_loss.avg, eval_boundary_loss.avg, eval_dsc.avg, eval_hd95.avg, eval_jac.avg, eval_mag.avg
    return eval_loss.avg, eval_boundary_loss.avg, eval_dsc.avg

def _binary_dilate_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def _binary_erode_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def build_foreground_mask(target: torch.Tensor) -> torch.Tensor:
    """Build a label-free foreground mask directly from image intensities."""
    bg_val = target.amin(dim=(2, 3, 4), keepdim=True)
    mask = (target > (bg_val + 1e-3)).float()
    mask = _binary_dilate_3d(mask, kernel_size=5)
    mask = _binary_erode_3d(mask, kernel_size=5)
    mask = _binary_dilate_3d(mask, kernel_size=3)
    return mask.clamp_(0.0, 1.0)

def build_boundary_ring_mask(
    foreground_mask: torch.Tensor,
    inner_kernel_size: int = 3,
    outer_kernel_size: int = 7,
) -> torch.Tensor:
    inner_kernel_size = max(1, int(inner_kernel_size))
    outer_kernel_size = max(inner_kernel_size, int(outer_kernel_size))

    if inner_kernel_size % 2 == 0:
        inner_kernel_size += 1
    if outer_kernel_size % 2 == 0:
        outer_kernel_size += 1

    foreground_mask = foreground_mask.float().clamp_(0.0, 1.0)
    outer_band = _binary_dilate_3d(foreground_mask, kernel_size=outer_kernel_size)
    inner_core = _binary_erode_3d(foreground_mask, kernel_size=inner_kernel_size)
    ring_mask = (outer_band - inner_core).clamp_(0.0, 1.0)

    if torch.count_nonzero(ring_mask).item() == 0:
        return foreground_mask
    return ring_mask

def parse_feature_edge_indices(scales: str) -> tuple:
    scale_to_index = {'1/2': 0, '1/4': 1, '1/8': 2, '1/16': 3}
    indices = []
    for token in scales.split(','):
        token = token.strip()
        if not token:
            continue
        if token in scale_to_index:
            indices.append(scale_to_index[token])
        else:
            indices.append(int(token))
    return tuple(indices)


def parse_pyramid_image_weights(weights: str) -> tuple:
    values = []
    for token in weights.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return tuple(values)


def _match_weight_count(weights: tuple, count: int) -> list:
    if count <= 0:
        return []
    if not weights:
        return [1.0] * count
    matched = list(weights[:count])
    if len(matched) < count:
        matched.extend([matched[-1]] * (count - len(matched)))
    return matched


def unpack_model_outputs(
    out,
    expect_coarse_flows: bool = False,
    expect_residual_flows: bool = False,
    expect_feature_edge_loss: bool = False,
):
    idx = 0
    displacement = out[idx]
    idx += 1
    warped_source = out[idx] if len(out) > idx else None
    idx += 1

    coarse_flows = []
    if expect_coarse_flows:
        coarse_flows = out[idx]
        idx += 1

    residual_flows = []
    if expect_residual_flows:
        residual_flows = out[idx]
        idx += 1

    feature_edge_loss = displacement.new_tensor(0.0)
    if expect_feature_edge_loss and len(out) > idx:
        feature_edge_loss = out[idx]

    return displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss


def global_ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    u_true = y_true.mean(dim=[1, 2, 3, 4], keepdim=True)
    u_pred = y_pred.mean(dim=[1, 2, 3, 4], keepdim=True)
    z_true = y_true - u_true
    z_pred = y_pred - u_pred
    cov = (z_true * z_pred).mean(dim=[1, 2, 3, 4])
    var_true = (z_true ** 2).mean(dim=[1, 2, 3, 4])
    var_pred = (z_pred ** 2).mean(dim=[1, 2, 3, 4])
    corr = cov / (torch.sqrt(var_true * var_pred) + eps)
    return -corr.mean()


def compute_oasis_pyramid_losses(
    model: nn.Module,
    source_float: torch.Tensor,
    target_float: torch.Tensor,
    fg_mask: torch.Tensor,
    coarse_flows: list,
    residual_flows: list,
    image_loss_fn_coarse: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: list,
    use_mask: bool,
    loss_type: str,
    device: str,
    pyramid_weight: float = 0.5,
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
):
    deep_sup_loss = target_float.new_tensor(0.0)
    pyramid_img_loss = target_float.new_tensor(0.0)
    residual_flow_reg_loss = target_float.new_tensor(0.0)

    if len(coarse_flows) > 0:
        scale_weights = _match_weight_count(pyramid_image_weights, len(coarse_flows))
        image_weight_sum = max(sum(scale_weights), 1e-6)
        st_cache = {}
        for scale_weight, c_flow in zip(scale_weights, coarse_flows):
            c_shape = c_flow.shape[2:]
            c_flow_float = c_flow.float()
            if pyramid_weight > 0:
                c_grad_loss = grad_loss_fn(c_flow_float).mean()
                deep_sup_loss = deep_sup_loss + c_grad_loss

            if pyramid_image_weight > 0:
                if target_float.shape[2:] != c_shape:
                    y_down = F.interpolate(target_float, size=c_shape, mode='trilinear', align_corners=False)
                    x_down = F.interpolate(source_float, size=c_shape, mode='trilinear', align_corners=False)
                    mask_down = F.interpolate(fg_mask, size=c_shape, mode='trilinear', align_corners=False) if use_mask else None
                else:
                    y_down = target_float
                    x_down = source_float
                    mask_down = fg_mask if use_mask else None

                if not getattr(model, 'use_dpfc', False) and hasattr(model, 'integrate') and model.integrate is not None:
                    c_disp = model.integrate(c_flow_float)
                else:
                    c_disp = c_flow_float

                if c_shape not in st_cache:
                    st_cache[c_shape] = vxm.nn.modules.SpatialTransformer().to(device)

                warped_x_down = st_cache[c_shape](x_down, c_disp)

                if loss_type == 'mse':
                    squared_diff = (y_down - warped_x_down) ** 2
                    if use_mask and mask_down is not None:
                        c_img_loss = (squared_diff * mask_down).sum() / (mask_down.sum() + 1e-8)
                    else:
                        c_img_loss = squared_diff.mean()
                else:
                    if use_mask and mask_down is not None:
                        if image_loss_fn_coarse is not None:
                            c_img_loss = -image_loss_fn_coarse(y_down * mask_down, warped_x_down * mask_down).mean()
                        else:
                            c_img_loss = global_ncc(y_down * mask_down, warped_x_down * mask_down)
                    else:
                        if image_loss_fn_coarse is not None:
                            c_img_loss = -image_loss_fn_coarse(y_down, warped_x_down).mean()
                        else:
                            c_img_loss = global_ncc(y_down, warped_x_down)

                pyramid_img_loss = pyramid_img_loss + scale_weight * c_img_loss

        if pyramid_weight > 0:
            deep_sup_loss = (deep_sup_loss / len(coarse_flows)) * pyramid_weight
        if pyramid_image_weight > 0:
            pyramid_img_loss = pyramid_img_loss / image_weight_sum

    if residual_flow_reg_weight > 0 and len(residual_flows) > 0:
        stage_weights = [float(idx + 1) for idx in range(len(residual_flows))]
        weight_sum = max(sum(stage_weights), 1e-6)
        for stage_weight, residual_flow in zip(stage_weights, residual_flows):
            residual_flow_reg_loss = residual_flow_reg_loss + stage_weight * grad_loss_fn(residual_flow.float()).mean()
        residual_flow_reg_loss = residual_flow_reg_loss / weight_sum

    return deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss

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
    loss_type: str = 'mse',
    image_loss_fn_coarse: nn.Module = None,
    boundary_loss_fn: nn.Module = None,
    boundary_loss_weight: float = 0.0,
    boundary_ring_inner_kernel: int = 3,
    boundary_ring_outer_kernel: int = 7,
    feature_edge_loss_weight: float = 0.0,
    feature_edge_indices: tuple = (0, 1),
    pyramid_image_weight: float = 0.0,
    pyramid_image_weights: tuple = (),
    residual_flow_reg_weight: float = 0.0,
    saor_loss_fn: nn.Module = None,
    sbc_weight: float = 0.0,
    sbc_stage_weight: float = 0.5,
    accumulation_steps: int = 1,
    dess_weight: float = 0.0,
    dess_probability: float = 0.5,
    dess_max_displacement: float = 3.0,
    dess_coarse_scale: float = 0.125,
    dess_structure_aware: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_img_loss = 0.0
    total_grad_loss = 0.0
    total_feature_edge_loss = 0.0
    total_dess_loss = 0.0
    dess_batches = 0
    valid_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, data in enumerate(dataloader):
        # TransMorph OASISDataset returns: x, y, x_seg, y_seg
        # x: moving image, y: fixed image
        x = data[0].to(device)
        y = data[1].to(device)
        dess_active = dess_weight > 0 and torch.rand((), device=device).item() < dess_probability
        if dess_active:
            target_perturbation = vxm.nn.losses.random_diffeomorphic_displacement(
                y,
                model.spatial_transform,
                max_displacement=dess_max_displacement,
                coarse_scale=dess_coarse_scale,
            )
            perturbed_target = model.spatial_transform(y.float(), target_perturbation)
        # 使用 AMP autocast
        with torch.autocast('cuda', enabled=amp_enabled, dtype=torch.bfloat16):
            use_feature_edge_loss = feature_edge_loss_weight > 0 and hasattr(model, '_feature_edge_loss')
            use_residual_flow_pyramid = getattr(model, 'use_residual_flow_pyramid', False)
            use_dpfc = getattr(model, 'use_dpfc', False)
            # Get the displacement and the warped source image from the model
            out = model(
                x,
                y,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=True,
                return_residual_flows=use_residual_flow_pyramid or use_dpfc,
                return_feature_edge_loss=use_feature_edge_loss,
                feature_edge_indices=feature_edge_indices,
            )
            displacement, warped_source, coarse_flows, residual_flows, feature_edge_loss = unpack_model_outputs(
                out,
                expect_coarse_flows=True,
                expect_residual_flows=use_residual_flow_pyramid or use_dpfc,
                expect_feature_edge_loss=use_feature_edge_loss,
            )
            if sbc_weight > 0:
                reverse_out = model(
                    y,
                    x,
                    return_warped_source=True,
                    return_field_type='displacement',
                    return_coarse_flows=True,
                    return_residual_flows=False,
                )
                reverse_displacement, reverse_warped, reverse_coarse_flows, _, _ = unpack_model_outputs(
                    reverse_out,
                    expect_coarse_flows=True,
                )
            if dess_active:
                perturbed_displacement = model(
                    x,
                    perturbed_target,
                    return_warped_source=False,
                    return_field_type='displacement',
                )

        # 🔥【关键修复】：在这里退出模型前向的 AMP autocast 作用域！
        # 如果把损失函数（不论是 NCC 还是 MSE）放在 autocast 里面算，因为 Loss 内部往往会有 Conv3d 或者平方项操作，
        # PyTorch 会非常聪明地强行把输入的 float32 降级成 BF16/FP16 再算，这会导致 NCC 的方差发生严重的精度截断甚至变成负数导致 NaN。
        # AMP 兼容性保护：强制将预测结果和 Loss 计算切回 float32
        target_float = y.float()
        warped_float = warped_source.float()
        
        # Label-free foreground mask used by optional image losses.
        base_fg_mask = build_foreground_mask(target_float)
        boundary_ring_mask = build_boundary_ring_mask(
            base_fg_mask,
            inner_kernel_size=boundary_ring_inner_kernel,
            outer_kernel_size=boundary_ring_outer_kernel,
        )

        if use_mask:
            fg_mask = base_fg_mask
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

        sbc_loss = displacement.new_tensor(0.0)
        if sbc_weight > 0:
            reverse_warped_float = reverse_warped.float()
            if loss_type == 'mse':
                reverse_img_loss = ((x.float() - reverse_warped_float) ** 2).mean()
            else:
                reverse_img_loss = -image_loss_fn(x.float(), reverse_warped_float).mean()
            img_loss = 0.5 * (img_loss + reverse_img_loss)
            sbc_loss = vxm.nn.losses.symmetric_displacement_composition_loss(
                model.spatial_transform,
                displacement,
                reverse_displacement,
                coarse_flows,
                reverse_coarse_flows,
                stage_weight=sbc_stage_weight,
            )
        dess_loss = displacement.new_tensor(0.0)
        if dess_active:
            dess_loss = vxm.nn.losses.deformation_equivariance_loss(
                model.spatial_transform,
                displacement,
                perturbed_displacement,
                target_perturbation,
                structure_image=target_float,
                structure_aware=dess_structure_aware,
            )
        
        grad_loss = saor_loss_fn(displacement.float(), target_float) if saor_loss_fn is not None else grad_loss_fn(displacement.float()).mean()
        boundary_loss = displacement.new_tensor(0.0)
        if boundary_loss_fn is not None and boundary_loss_weight > 0:
            boundary_loss = boundary_loss_fn(warped_float, target_float, mask=boundary_ring_mask)
        deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss = compute_oasis_pyramid_losses(
            model,
            x.float(),
            target_float,
            fg_mask,
            coarse_flows,
            residual_flows,
            image_loss_fn_coarse,
            grad_loss_fn,
            loss_weights,
            use_mask,
            loss_type,
            device=device,
            pyramid_weight=pyramid_weight,
            pyramid_image_weight=pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=residual_flow_reg_weight,
        )

        grad_loss = torch.clamp(grad_loss, min=0.0, max=100.0)
        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        if boundary_loss_weight > 0:
            loss = loss + boundary_loss_weight * boundary_loss
        if feature_edge_loss_weight > 0:
            loss = loss + feature_edge_loss_weight * feature_edge_loss.float()
        if deep_sup_loss.requires_grad or deep_sup_loss.item() != 0:
            loss = loss + deep_sup_loss # Add deep supervision component
        if pyramid_image_weight > 0:
            loss = loss + pyramid_image_weight * pyramid_img_loss
        if residual_flow_reg_weight > 0:
            loss = loss + residual_flow_reg_weight * residual_flow_reg_loss
        if sbc_weight > 0:
            loss = loss + sbc_weight * sbc_loss
        if dess_active:
            loss = loss + dess_weight * dess_loss

        # 数值稳定性保护：发现非有限值则跳过该 batch，避免污染整轮 loss
        if not torch.isfinite(loss):
            print(
                f"[WARN] Non-finite loss at batch {batch_idx}: "
                f"img_loss={img_loss.item()}, grad_loss={grad_loss.item()}, boundary_loss={boundary_loss.item()}, feature_edge_loss={feature_edge_loss.item()}, total={loss.item()}"
            )
            # 彻底释放包含 NaN/Inf 计算图的所有局部变量，防止在遇到 NaN 直接 continue 时显存泄漏引发后续 OOM
            optimizer.zero_grad(set_to_none=True)
            del out, displacement, warped_source, coarse_flows, residual_flows, loss, img_loss, grad_loss, deep_sup_loss, pyramid_img_loss, residual_flow_reg_loss
            torch.cuda.empty_cache()
            continue

        if amp_enabled and scaler is not None:
            # 缩放 loss，反向传播
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        should_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader)
        if should_step:
            if amp_enabled and scaler is not None:
                # 在执行梯度裁剪前，必须先 unscale 梯度
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        total_loss += loss.item()
        total_img_loss += img_loss.item()
        total_grad_loss += grad_loss.item()
        total_feature_edge_loss += feature_edge_loss.float().item()
        if dess_active:
            total_dess_loss += dess_loss.item()
            dess_batches += 1
        valid_batches += 1

    if valid_batches == 0:
            return float('nan'), float('nan'), float('nan'), float('nan')
    if dess_weight > 0:
        print(f"  [DESS] Loss: {total_dess_loss / max(dess_batches, 1):.6f}, Active: {dess_batches}/{valid_batches}")
    return (
        total_loss / valid_batches,
        total_img_loss / valid_batches,
        total_grad_loss / valid_batches,
        total_feature_edge_loss / valid_batches,
    )

def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('--output', type=str, default='/root/autodl-tmp/models/oasis_vxm.pt', help='Output model path')
    parser.add_argument('--resume', type=str, default=None, help='Load model weights from a checkpoint before training')
    parser.add_argument('--start-epoch', type=int, default=1, help='First displayed/training epoch when branching from a checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible branch experiments')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps used to preserve effective batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ncc'], help='Image similarity loss')
    parser.add_argument('--use-mask', action='store_true', help='Use foreground mask to exclude black background in loss calculation')
    parser.add_argument('--disable-amp', action='store_true', help='Force disable AMP (Automatic Mixed Precision)')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01, help='Weight of gradient loss')
    parser.add_argument('--pyramid-weight', type=float, default=0.5, help='Weight for intermediate pyramid flow-smoothness deep supervision loss')
    parser.add_argument('--use-residual-flow-pyramid', action='store_true', help='Use lightweight coarse-to-fine residual flow accumulation without warping skip features')
    parser.add_argument('--use-dpfc', action='store_true', help='Use deformation-aware coarse-to-fine residual diffeomorphic composition')
    parser.add_argument('--dpfc-flow-limit', type=float, default=8.0, help='Maximum full-resolution correction budget at the coarsest DPFC stage')
    parser.add_argument('--use-miscv', action='store_true', help='Use modality-invariant sparse cost-volume correspondence')
    parser.add_argument('--miscv-scales', type=str, default='1/8,1/4', help='Decoder scales for MISC-V; recommended: 1/8,1/4')
    parser.add_argument('--miscv-projection-channels', type=int, default=8, help='Shared low-dimensional projection channels for MISC-V')
    parser.add_argument('--miscv-search-radius', type=int, default=2, help='MISC-V search radius at 1/8; 1/4 always uses radius 1')
    parser.add_argument('--miscv-temperature', type=float, default=0.1, help='Softmax temperature for MISC-V local correspondence')
    parser.add_argument('--use-sscc', action='store_true', help='Use shared spatial coordinate calibration on dual-stream encoder features')
    parser.add_argument('--sscc-scales', type=str, default='1/16,1/8', help='Comma-separated encoder scales for SSCC')
    parser.add_argument('--sscc-hidden-channels', type=int, default=16, help='Hidden channels in SSCC coordinate projection')
    parser.add_argument('--sscc-strength', type=float, default=0.2, help='Maximum SSCC feature calibration strength')
    parser.add_argument('--use-cagr', action='store_true', help='Use confidence-aware gated refinement inside C2F-RDC')
    parser.add_argument('--cagr-strength', type=float, default=0.5, help='Maximum multiplicative confidence modulation around neutral scale 1')
    parser.add_argument('--use-error-guided-residual', action='store_true', help='Enhance residual flow pyramid with Structure-aware & Error-guided side branch')
    parser.add_argument("--error-guided-metric", type=str, default="feature_ncc", choices=["feature_ncc", "mind"], help="Metric to compute error maps for guidance")
    parser.add_argument('--residual-flow-limit', type=float, default=4.0, help='Per-stage magnitude cap for residual flow heads when residual flow pyramid is enabled')
    parser.add_argument('--pyramid-image-weight', type=float, default=0.0, help='Weight for multi-scale image supervision on intermediate pyramid displacements')
    parser.add_argument('--pyramid-image-weights', type=str, default='0.2,0.35,0.5', help='Comma-separated relative weights for intermediate pyramid image supervision from coarse to fine')
    parser.add_argument('--pyramid-image-start-epoch', type=int, default=1, help='Enable pyramid image supervision starting from this 1-based epoch')
    parser.add_argument('--residual-flow-reg-weight', type=float, default=0.0, help='Weight for gradient regularization on per-stage residual flow heads')
    parser.add_argument('--use-saor', action='store_true', help='Replace fixed final-flow smoothness with structure-adaptive optimization regularization')
    parser.add_argument('--saor-alpha', type=float, default=3.0, help='Structure sensitivity of SAOR')
    parser.add_argument('--saor-risk-gamma', type=float, default=0.5, help='Strength of SAOR high-deformation risk protection')
    parser.add_argument('--saor-risk-threshold', type=float, default=0.5, help='Gradient-magnitude threshold for SAOR risk protection')
    parser.add_argument('--use-sbc', action='store_true', help='Use stage-level symmetric bidirectional composition training')
    parser.add_argument('--sbc-weight', type=float, default=0.05, help='Weight of symmetric inverse-composition consistency')
    parser.add_argument('--sbc-stage-weight', type=float, default=0.5, help='Relative weight of C2F stage-level composition consistency')
    parser.add_argument('--sbc-start-epoch', type=int, default=5, help='First epoch that enables SBC training')
    parser.add_argument('--use-dess', action='store_true', help='Use target-conditioned deformation-equivariant spatial supervision')
    parser.add_argument('--dess-weight', type=float, default=0.03, help='Weight of DESS final-displacement supervision')
    parser.add_argument('--dess-probability', type=float, default=0.5, help='Probability of applying DESS to a training pair')
    parser.add_argument('--dess-max-displacement', type=float, default=3.0, help='Maximum synthetic target perturbation in full-resolution voxels')
    parser.add_argument('--dess-coarse-scale', type=float, default=0.125, help='Low-resolution scale used to generate smooth DESS perturbations')
    parser.add_argument('--dess-start-epoch', type=int, default=10, help='First epoch that enables DESS training')
    parser.add_argument('--dess-structure-aware', action='store_true', help='DESS v2: supervise only valid structure-informative target regions')
    parser.add_argument('--use-pdaps', action='store_true', help='Use Pyramid-guided Deformation-Aware Progressive Skip')
    parser.add_argument('--pdaps-flow-limit', type=float, default=20.0, help='Maximum physical flow limit for P-DAPS. E.g., 20.0 for Brain, 30.0+ for Abdomen CT/MRI.')
    parser.add_argument('--use-daps', action='store_true', help='Use original DAPS')
    parser.add_argument('--use-dsin', action='store_true', help='Enable DSIN in the shallow decoupled encoder layers')
    parser.add_argument('--use-cmim', action='store_true', help='Enable CMIM at deep decoder scales')
    parser.add_argument('--use-cross-mamba', action='store_true', help='Enable Cross-Mamba at deep decoder scales')
    parser.add_argument('--cross-mamba-scales', type=str, default='1/16,1/8', help='Comma-separated deep scales for Cross-Mamba, e.g. 1/16 or 1/16,1/8,1/4,1/2')
    parser.add_argument('--cross-mamba-offset-limit', type=float, default=0.0, help='Feature-space resampling offset cap inside Cross-Mamba; 0 disables offset limiting/resampling regularization for cleaner ablations')
    parser.add_argument('--cross-mamba-offset-smooth-kernel', type=int, default=1, help='Odd average-pooling kernel for smoothing Cross-Mamba offsets; 1 disables smoothing')
    parser.add_argument('--cross-mamba-no-resampling', dest='cross_mamba_use_resampling', action='store_false', help='Disable Cross-Mamba feature-grid resampling and keep only semantic cross-context fusion')
    parser.set_defaults(cross_mamba_use_resampling=True)
    parser.add_argument('--cross-mamba-structure-norm', action='store_true', help='Use instance-normalized structure features for Cross-Mamba token interaction')
    parser.add_argument('--cross-mamba-residual-scale', type=float, default=1.0, help='Residual injection scale for Cross-Mamba output')
    parser.add_argument('--use-wcv', action='store_true', help='Enable window cross-attention on shallow skip features')
    parser.add_argument('--use-swcv', action='store_true', help='Enable Structure-Aware WCV on deep skip features')
    parser.add_argument('--use-gcv', action='store_true', help='Enable global cost volume on deep features')
    parser.add_argument('--use-boundary-branch', action='store_true', help='Enable lightweight boundary-guided feature modulation branch')
    parser.add_argument('--boundary-branch-scales', type=str, default='deep', choices=['deep', 'all'], help='Apply boundary branch on deep features only or all encoder scales')
    parser.add_argument('--boundary-branch-strength', type=float, default=0.5, help='Residual gate strength for the boundary feature branch')
    parser.add_argument('--use-boundary-loss', action='store_true', help='Enable explicit boundary consistency loss between warped source and target')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.1, help='Weight for boundary consistency loss when enabled')
    parser.add_argument('--boundary-loss-start-epoch', type=int, default=10, help='Enable boundary loss only after this 1-based epoch index; e.g. 10 starts boundary loss from epoch 11')
    parser.add_argument('--boundary-loss-metric', type=str, default='l1', choices=['ncc', 'l1'], help='Metric used by the boundary consistency loss; `l1` compares 3D Sobel gradient maps more directly and usually overlaps less with the main image NCC term')
    parser.add_argument('--boundary-kernel', type=str, default='sobel', choices=['sobel', 'diff'], help='Fixed operator used to extract 3D boundary maps')
    parser.add_argument('--boundary-smooth-kernel', type=int, default=3, help='Odd smoothing kernel size applied before boundary extraction')
    parser.add_argument('--boundary-ring-inner-kernel', type=int, default=3, help='Inner erosion kernel size for the foreground boundary ring mask')
    parser.add_argument('--boundary-ring-outer-kernel', type=int, default=7, help='Outer dilation kernel size for the foreground boundary ring mask')
    parser.add_argument('--use-ussc', '--use-cross-frequency-modulation', '--use-frequency-modulation', dest='use_ussc', action='store_true', help='Enable uncertainty-aware self-similarity correspondence guidance')
    parser.add_argument('--ussc-scales', '--cross-frequency-scales', '--frequency-modulation-scales', dest='ussc_scales', type=str, default='1/8', help='Comma-separated decoder scales for USSC; 1/8 is recommended for both tasks')
    parser.add_argument('--ussc-search-radius', type=int, default=2, help='USSC local search radius; 2 produces a 5x5x5 search neighborhood')
    parser.add_argument('--ussc-temperature', '--spectral-temperature', dest='ussc_temperature', type=float, default=0.1, help='Softmax temperature for USSC local correspondence probabilities')
    parser.add_argument('--ussc-guidance-strength', '--spectral-guidance-strength', dest='ussc_guidance_strength', type=float, default=0.5, help='Residual strength of uncertainty-weighted USSC guidance')
    parser.add_argument('--use-dasr', action='store_true', help='Enable structure-driven competitive decoder-adaptive skip routing (DASR v2)')
    parser.add_argument('--dasr-scales', type=str, default='1/2', help='Comma-separated decoder skip scales routed by DASR; 1/2 avoids Cross-Mamba scales')
    parser.add_argument('--dasr-reduction', type=int, default=4, help='Channel reduction ratio of DASR routing predictors')
    parser.add_argument('--dasr-strength', type=float, default=0.2, help='Maximum conservative modulation strength of competitive DASR v2 gates')
    parser.add_argument('--use-drfc', action='store_true', help='Enable deformation reliability field calibration before diffeomorphic integration')
    parser.add_argument('--drfc-scale', type=float, default=0.25, help='Low-resolution scale used to calibrate velocity reliability')
    parser.add_argument('--drfc-hidden-channels', type=int, default=16, help='Hidden channels of the DRFC reliability predictor')
    parser.add_argument('--drfc-strength', type=float, default=0.5, help='Maximum strength for calibrating local velocity residuals')
    parser.add_argument('--use-sdmr', action='store_true', help='Enable Structure-error-guided Diffeomorphic Mamba Refinement on the predicted velocity field')
    parser.add_argument('--sdmr-use-mind', action='store_true', help='Include low-resolution MIND residual maps in SDMR inputs')
    parser.add_argument('--sdmr-scale', type=float, default=0.125, help='Low-resolution scale used by SDMR refiner, e.g. 0.125 or 0.25')
    parser.add_argument('--sdmr-hidden-channels', type=int, default=16, help='Hidden channels of the SDMR refiner')
    parser.add_argument('--sdmr-flow-limit', type=float, default=1.0, help='Magnitude cap for SDMR residual velocity in full-resolution voxel units')
    parser.add_argument('--sdmr-alpha', type=float, default=0.5, help='Residual velocity blending weight for SDMR')
    parser.add_argument('--use-feature-edge-loss', action='store_true', help='Enable feature-level multi-scale gradient magnitude consistency on encoder features')
    parser.add_argument('--feature-edge-loss-weight', type=float, default=0.05, help='Weight for feature-level multi-scale gradient magnitude consistency loss')
    parser.add_argument('--feature-edge-scales', type=str, default='1/2,1/4', help='Comma-separated encoder scales or indices for feature edge loss, e.g. 1/2,1/4 or 0,1')
    parser.add_argument('--feature-edge-start-epoch', type=int, default=1, help='Enable feature-edge loss from this 1-based epoch index; e.g. 5 starts applying it at epoch 5')
    parser.add_argument('--encoder-type', type=str, default='cnn', choices=['cnn', 'mamba'], help='Backbone type for feature extraction.')
    parser.add_argument('--mamba-shallow-multi', action='store_true', help='Legacy shorthand: enable multi-axis (d,h,w) scanning for both encoder and decoder 1/8-scale Mamba blocks.')
    parser.add_argument('--mamba-enc-shallow-multi', action='store_true', help='Enable multi-axis (d,h,w) scanning for the encoder 1/8-scale Mamba block.')
    parser.add_argument('--mamba-dec-shallow-multi', action='store_true', help='Enable multi-axis (d,h,w) scanning for the decoder 1/8-scale Mamba block.')
    parser.add_argument('--mamba-quarter-scale', action='store_true', help='Enable Mamba scanning (single-axis) at 1/4 scale (i=1) to improve fine structural alignment.')
    parser.add_argument('--mamba-dec-quarter-scale', action='store_true', help='Enable decoder Mamba scanning (single-axis) at 1/4 scale (i=1) to improve fine structural alignment.')
    parser.add_argument('--mamba-parallel-block', action='store_true', help='Use Parallel Local(CNN)-Global(Mamba) Block instead of serial Mamba to protect local edges.')
    parser.add_argument('--model-config', type=str, default='dual_stream', choices=['dual_stream', 'voxelmorph_baseline'], help='Choose between the current dual-stream Siamese setup and the standard Voxelmorph baseline')
    parser.add_argument('--decouple-layers', type=int, default=2, help='Number of shallow decoupled encoder layers used in dual-stream mode')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--window-size', type=int, default=9, help='Window size for WCV and S-WCV')
    parser.add_argument('--fusion-method', type=str, default='compress_concat', choices=['add', 'concat', 'compress_concat'], help='Feature fusion method for Siamese encoder')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--vis-every', type=int, default=5, help='Save qualitative visualization every N epochs; set 0 to disable')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument('--warm-start', type=int, default=10, help='Early stopping warm start steps')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of epochs for learning rate warmup')
    parser.add_argument('--integration-steps', type=int, default=0, help='number of integration steps for diffeomorphic registration')
    parser.add_argument('--train-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    args = parser.parse_args()

    if args.mamba_shallow_multi:
        args.mamba_enc_shallow_multi = True
        args.mamba_dec_shallow_multi = True

    if args.use_daps and args.use_pdaps:
        parser.error('--use-daps and --use-pdaps are mutually exclusive. Use --use-pdaps for the current pyramid design.')
    if args.use_cmim and args.use_cross_mamba:
        parser.error('--use-cmim and --use-cross-mamba are mutually exclusive interaction modules.')
    if args.use_error_guided_residual and not args.use_residual_flow_pyramid:
        parser.error('--use-error-guided-residual requires --use-residual-flow-pyramid.')
    if args.use_dpfc and (args.use_pdaps or args.use_residual_flow_pyramid):
        parser.error('--use-dpfc is mutually exclusive with --use-pdaps and --use-residual-flow-pyramid.')
    if args.use_dpfc and args.integration_steps <= 0:
        parser.error('--use-dpfc requires --integration-steps greater than zero.')
    if args.use_dpfc and args.dpfc_flow_limit <= 0:
        parser.error('--dpfc-flow-limit must be positive.')
    if args.use_miscv and args.miscv_projection_channels < 1:
        parser.error('--miscv-projection-channels must be positive.')
    if args.use_miscv and args.miscv_search_radius < 1:
        parser.error('--miscv-search-radius must be at least 1.')
    if args.use_miscv and args.miscv_temperature <= 0:
        parser.error('--miscv-temperature must be positive.')
    if args.sscc_hidden_channels <= 0 or args.sscc_strength <= 0:
        parser.error('--sscc-hidden-channels and --sscc-strength must be positive.')
    if args.use_cagr and not args.use_dpfc:
        parser.error('--use-cagr requires --use-dpfc.')
    if args.cagr_strength < 0 or args.cagr_strength >= 1:
        parser.error('--cagr-strength must be in [0, 1).')
    if args.use_saor and (args.saor_alpha < 0 or args.saor_risk_gamma < 0 or args.saor_risk_threshold < 0):
        parser.error('SAOR parameters must be non-negative.')
    if args.use_sbc and not args.use_dpfc:
        parser.error('--use-sbc requires --use-dpfc for stage-level composition consistency.')
    if args.sbc_weight < 0 or args.sbc_stage_weight < 0 or args.sbc_start_epoch < 1:
        parser.error('SBC weights must be non-negative and --sbc-start-epoch must be positive.')
    if args.accumulation_steps < 1:
        parser.error('--accumulation-steps must be positive.')
    if args.dess_weight < 0 or not (0 <= args.dess_probability <= 1) or args.dess_max_displacement <= 0:
        parser.error('DESS weight/probability/displacement parameters are invalid.')
    if not (0 < args.dess_coarse_scale <= 1) or args.dess_start_epoch < 1:
        parser.error('--dess-coarse-scale must be in (0, 1] and --dess-start-epoch must be positive.')
    if args.boundary_ring_inner_kernel < 1 or args.boundary_ring_outer_kernel < 1:
        parser.error('--boundary-ring-inner-kernel and --boundary-ring-outer-kernel must be positive integers.')
    if args.boundary_ring_outer_kernel < args.boundary_ring_inner_kernel:
        parser.error('--boundary-ring-outer-kernel must be greater than or equal to --boundary-ring-inner-kernel.')
    if args.feature_edge_start_epoch < 1:
        parser.error('--feature-edge-start-epoch must be a positive integer.')
    if args.start_epoch < 1 or args.start_epoch > args.epochs:
        parser.error('--start-epoch must be between 1 and --epochs.')
    if args.use_ussc and args.ussc_search_radius < 1:
        parser.error('--ussc-search-radius must be at least 1.')
    if args.use_ussc and args.ussc_temperature <= 0:
        parser.error('--ussc-temperature must be positive.')
    if args.use_ussc and args.ussc_guidance_strength < 0:
        parser.error('--ussc-guidance-strength must be non-negative.')
    if args.use_dasr and args.dasr_reduction < 1:
        parser.error('--dasr-reduction must be positive.')
    if args.use_dasr and args.dasr_strength < 0:
        parser.error('--dasr-strength must be non-negative.')
    if args.use_drfc and not (0 < args.drfc_scale <= 1):
        parser.error('--drfc-scale must be in (0, 1].')
    if args.use_drfc and args.drfc_hidden_channels < 1:
        parser.error('--drfc-hidden-channels must be positive.')
    if args.use_drfc and args.drfc_strength < 0:
        parser.error('--drfc-strength must be non-negative.')
    if args.use_sdmr and not (0 < args.sdmr_scale <= 1.0):
        parser.error('--sdmr-scale must be in (0, 1].')
    if args.use_sdmr and args.sdmr_hidden_channels < 1:
        parser.error('--sdmr-hidden-channels must be a positive integer.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 加速补丁：为固定尺寸的 3D 输入寻找最优 CUDA 卷积算子
    import torch.backends.cudnn as cudnn
    if device == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = False
        print("🚀 cuDNN Benchmark enabled for Cuda acceleration!")

    # Create model
    model = build_registration_model(args, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f'Loaded model weights from: {args.resume}')
        if incompatible.missing_keys:
            print(f'Initialized new checkpoint-missing parameters: {incompatible.missing_keys}')
        if incompatible.unexpected_keys:
            print(f'Ignored unexpected checkpoint parameters: {incompatible.unexpected_keys}')
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
        image_loss_fn_coarse = ne.nn.modules.NCC(eps=1e-2, window_size=3)
    else:
        image_loss_fn = ne.nn.modules.MSE()
        image_loss_fn_coarse = ne.nn.modules.MSE()
        
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    saor_loss_fn = None
    if args.use_saor:
        saor_loss_fn = vxm.nn.losses.StructureAdaptiveOptimizationRegularization(
            alpha=args.saor_alpha,
            risk_gamma=args.saor_risk_gamma,
            risk_threshold=args.saor_risk_threshold,
        ).to(device)
    boundary_loss_fn = None
    if args.use_boundary_loss:
        boundary_loss_fn = vxm.nn.losses.BoundaryConsistencyLoss(
            operator=args.boundary_kernel,
            metric=args.boundary_loss_metric,
            smooth_kernel_size=args.boundary_smooth_kernel,
        ).to(device)
    loss_weights = [1.0, args.lambda_param]
    feature_edge_indices = parse_feature_edge_indices(args.feature_edge_scales)
    pyramid_image_weights = parse_pyramid_image_weights(args.pyramid_image_weights)
    base_feature_edge_loss_weight = args.feature_edge_loss_weight if args.use_feature_edge_loss else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler: Warmup (Linear) then Cosine annealing to gradually lower LR
    warmup_epochs = args.warmup_epochs
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # 2. AMP 策略与防爆保护
    amp_enabled = (device == 'cuda') and not args.disable_amp
    if amp_enabled and (args.loss.lower() == 'ncc') and (args.integration_steps > 0):
        print("\n\n[INFO 🚀] 使用了 AMP + NCC。系统已修复精度范围域，请放心训练！\n\n")
        
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Dataloader identical to TransMorph
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    
    train_pattern = os.path.join(args.train_dir, '*.pkl')
    val_pattern = os.path.join(args.val_dir, '*.pkl')
    train_set = datasets.OASISBrainDataset(glob.glob(train_pattern), transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(glob.glob(val_pattern), transforms=val_composed)
    
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, generator=train_generator)
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
        f.write(f"Start Epoch: {args.start_epoch}\n")
        f.write(f"Resume: {args.resume}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"Pyramid Weight: {args.pyramid_weight}\n")
        f.write(f"Pyramid Image Weight: {args.pyramid_image_weight}\n")
        f.write(f"Pyramid Image Start Epoch: {args.pyramid_image_start_epoch}\n")
        f.write(f"Pyramid Image Weights: {args.pyramid_image_weights}\n")
        f.write(f"Feature Edge Loss Weight: {base_feature_edge_loss_weight}\n")
        f.write(f"Feature Edge Start Epoch: {args.feature_edge_start_epoch}\n")
        f.write(f"Feature Edge Scales: {args.feature_edge_scales}\n")
        f.write(f"Use USSC: {args.use_ussc}\n")
        f.write(f"USSC Scales: {args.ussc_scales}\n")
        f.write(f"USSC Search Radius: {args.ussc_search_radius}\n")
        f.write(f"USSC Temperature: {args.ussc_temperature}\n")
        f.write(f"USSC Guidance Strength: {args.ussc_guidance_strength}\n")
        f.write(f"Use DASR: {args.use_dasr}\n")
        f.write("DASR Version: v2-structure-competitive\n")
        f.write(f"DASR Scales: {args.dasr_scales}\n")
        f.write(f"DASR Reduction: {args.dasr_reduction}\n")
        f.write(f"DASR Strength: {args.dasr_strength}\n")
        f.write(f"Use DRFC: {args.use_drfc}\n")
        f.write(f"DRFC Scale: {args.drfc_scale}\n")
        f.write(f"DRFC Hidden Channels: {args.drfc_hidden_channels}\n")
        f.write(f"DRFC Strength: {args.drfc_strength}\n")
        f.write(f"Use Residual Flow Pyramid: {args.use_residual_flow_pyramid}\n")
        f.write(f"Use DPFC: {args.use_dpfc}\n")
        f.write("DPFC Version: v2-deformation-aware-c2f\n")
        f.write(f"DPFC Flow Limit: {args.dpfc_flow_limit}\n")
        f.write(f"Use MISC-V: {args.use_miscv}\n")
        f.write(f"MISC-V Scales: {args.miscv_scales}\n")
        f.write(f"MISC-V Projection Channels: {args.miscv_projection_channels}\n")
        f.write(f"MISC-V Search Radius: {args.miscv_search_radius}\n")
        f.write(f"MISC-V Temperature: {args.miscv_temperature}\n")
        f.write(f"Use SSCC: {args.use_sscc}\n")
        f.write(f"SSCC Scales: {args.sscc_scales}\n")
        f.write(f"SSCC Hidden Channels: {args.sscc_hidden_channels}\n")
        f.write(f"SSCC Strength: {args.sscc_strength}\n")
        f.write(f"Use CAGR: {args.use_cagr}\n")
        f.write(f"CAGR Strength: {args.cagr_strength}\n")
        f.write(f"Use Error-Guided Residual: {args.use_error_guided_residual}\n")
        f.write(f"Residual Flow Limit: {args.residual_flow_limit}\n")
        f.write(f"Residual Flow Reg Weight: {args.residual_flow_reg_weight}\n")
        f.write(f"Use SAOR: {args.use_saor}\n")
        f.write(f"SAOR Alpha: {args.saor_alpha}\n")
        f.write(f"SAOR Risk Gamma: {args.saor_risk_gamma}\n")
        f.write(f"SAOR Risk Threshold: {args.saor_risk_threshold}\n")
        f.write(f"Use SBC: {args.use_sbc}\n")
        f.write(f"SBC Weight: {args.sbc_weight}\n")
        f.write(f"SBC Stage Weight: {args.sbc_stage_weight}\n")
        f.write(f"SBC Start Epoch: {args.sbc_start_epoch}\n")
        f.write(f"Gradient Accumulation Steps: {args.accumulation_steps}\n")
        f.write(f"Use DESS: {args.use_dess}\n")
        f.write(f"DESS Weight: {args.dess_weight}\n")
        f.write(f"DESS Probability: {args.dess_probability}\n")
        f.write(f"DESS Max Displacement: {args.dess_max_displacement}\n")
        f.write(f"DESS Coarse Scale: {args.dess_coarse_scale}\n")
        f.write(f"DESS Start Epoch: {args.dess_start_epoch}\n")
        f.write(f"DESS Structure Aware: {args.dess_structure_aware}\n")
        f.write(f"Cross-Mamba Offset Limit: {args.cross_mamba_offset_limit}\n")
        f.write(f"Cross-Mamba Offset Smooth Kernel: {args.cross_mamba_offset_smooth_kernel}\n")
        f.write(f"Cross-Mamba Use Resampling: {args.cross_mamba_use_resampling}\n")
        f.write(f"Cross-Mamba Structure Norm: {args.cross_mamba_structure_norm}\n")
        f.write(f"Cross-Mamba Residual Scale: {args.cross_mamba_residual_scale}\n")
        f.write(f"Use SDMR: {args.use_sdmr}\n")
        f.write(f"SDMR Use MIND: {args.sdmr_use_mind}\n")
        f.write(f"SDMR Scale: {args.sdmr_scale}\n")
        f.write(f"SDMR Hidden Channels: {args.sdmr_hidden_channels}\n")
        f.write(f"SDMR Flow Limit: {args.sdmr_flow_limit}\n")
        f.write(f"SDMR Alpha: {args.sdmr_alpha}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Integration Steps: {args.integration_steps}\n")
        f.write(f"Arguments: {vars(args)}\n")

    # Initialize Logger
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_img_loss', 'train_grad_loss', 'train_feature_edge_loss', 'val_dsc', 'val_hd95', 'val_jac', 'val_mag'])

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_dsc = 0.0
    loss_history = []
    val_dsc_history = []
    
    epoch_times = []
    
    # 记录训练前的初始 GPU 显存
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(args.start_epoch - 1, args.epochs):
        epoch_start_time = time.time()
        epoch_num = epoch + 1
        boundary_loss_active = args.use_boundary_loss and (epoch_num > args.boundary_loss_start_epoch)
        active_boundary_loss_weight = args.boundary_loss_weight if boundary_loss_active else 0.0
        feature_edge_active = args.use_feature_edge_loss and (epoch_num >= args.feature_edge_start_epoch)
        active_feature_edge_loss_weight = base_feature_edge_loss_weight if feature_edge_active else 0.0
        pyramid_image_active = epoch_num >= args.pyramid_image_start_epoch
        active_pyramid_image_weight = args.pyramid_image_weight if pyramid_image_active else 0.0
        active_sbc_weight = args.sbc_weight if args.use_sbc and epoch_num >= args.sbc_start_epoch else 0.0
        active_dess_weight = args.dess_weight if args.use_dess and epoch_num >= args.dess_start_epoch else 0.0
        
        avg_loss, train_img_loss, train_grad_loss, train_feature_edge_loss = train_epoch(
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
            loss_type=args.loss.lower(),
            image_loss_fn_coarse=image_loss_fn_coarse,
            boundary_loss_fn=boundary_loss_fn,
            boundary_loss_weight=active_boundary_loss_weight,
            boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
            boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
            feature_edge_loss_weight=active_feature_edge_loss_weight,
            feature_edge_indices=feature_edge_indices,
            pyramid_image_weight=active_pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=args.residual_flow_reg_weight,
            saor_loss_fn=saor_loss_fn,
            sbc_weight=active_sbc_weight,
            sbc_stage_weight=args.sbc_stage_weight,
            accumulation_steps=args.accumulation_steps,
            dess_weight=active_dess_weight,
            dess_probability=args.dess_probability,
            dess_max_displacement=args.dess_max_displacement,
            dess_coarse_scale=args.dess_coarse_scale,
            dess_structure_aware=args.dess_structure_aware,
        )
        loss_history.append(avg_loss)
        
        # Calculate Validation metrics (Fast: only DSC)
        _, _, val_dsc = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            compute_extra=False,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            boundary_loss_fn=boundary_loss_fn,
            boundary_loss_weight=active_boundary_loss_weight,
            boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
            boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
            feature_edge_loss_weight=active_feature_edge_loss_weight,
            feature_edge_indices=feature_edge_indices,
            pyramid_weight=args.pyramid_weight,
            pyramid_image_weight=active_pyramid_image_weight,
            pyramid_image_weights=pyramid_image_weights,
            residual_flow_reg_weight=args.residual_flow_reg_weight,
            use_mask=args.use_mask,
            loss_type=args.loss.lower(),
            image_loss_fn_coarse=image_loss_fn_coarse,
        )
        
        # Decide if we need to compute heavy extra metrics (HD95, Jac, Mag)
        # Condition: 第1, 3, 5, 7个epoch，之后每10个epoch，或者dice达到最佳且大于0.77
        is_new_best = val_dsc > best_dsc
        condition_epoch = epoch_num in [1, 3, 5, 7] or epoch_num % 10 == 0
        condition_best = is_new_best and (val_dsc > 0.77)
        compute_extra = condition_epoch or condition_best
        
        if compute_extra:
            # Rerun validate to get the extra outputs. 
            # (Slight overhead of redoing model forward & DSC, but totally negligible compared to HD95 calculation)
            val_res_extra = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                compute_extra=True,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                boundary_loss_fn=boundary_loss_fn,
                boundary_loss_weight=active_boundary_loss_weight,
                boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
                boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
                feature_edge_loss_weight=active_feature_edge_loss_weight,
                feature_edge_indices=feature_edge_indices,
                pyramid_weight=args.pyramid_weight,
                residual_flow_reg_weight=args.residual_flow_reg_weight,
                use_mask=args.use_mask,
                loss_type=args.loss.lower(),
            )
            _, _, _, val_hd95, val_jac, val_mag = val_res_extra
        else:
            val_hd95, val_jac, val_mag = np.nan, np.nan, np.nan
            
        val_dsc_history.append(val_dsc)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        if compute_extra:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Img: {train_img_loss:.6f}, Grad: {train_grad_loss:.6f}, FeatureEdge: {train_feature_edge_loss:.6f}, Val DSC: {val_dsc:.6f}, HD95: {val_hd95:.2f}, Jac: {val_jac:.4f}, Mag: {val_mag:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')
        else:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Img: {train_img_loss:.6f}, Grad: {train_grad_loss:.6f}, FeatureEdge: {train_feature_edge_loss:.6f}, Val DSC: {val_dsc:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')

        # Save visualizations periodically or when a new best model is found to reduce epoch overhead
        save_vis = compute_extra
        if save_vis:
            try:
                save_qualitative_results(
                    model,
                    val_set,
                    out_path.parent,
                    epoch=epoch+1,
                    device=device,
                    boundary_kernel=args.boundary_kernel,
                    boundary_smooth_kernel=args.boundary_smooth_kernel,
                    boundary_ring_inner_kernel=args.boundary_ring_inner_kernel,
                    boundary_ring_outer_kernel=args.boundary_ring_outer_kernel,
                )
            except Exception as e:
                print(f"Failed to save visualization: {e}")

        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_loss:.6f}", 
                f"{train_img_loss:.6f}",
                f"{train_grad_loss:.6f}",
                f"{train_feature_edge_loss:.6f}",
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

        if is_new_best:
            best_dsc = val_dsc
            best_path = out_path.parent / f'{out_path.stem}_best.pt'
            torch.save(model.state_dict(), best_path)
            print(f'Saved new best model with DSC: {best_dsc:.6f} (HD95: {val_hd95:.2f}, Jac: {val_jac:.4f})')

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
