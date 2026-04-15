#!/usr/bin/env python3
"""
Train VoxelMorph for multimodal registration (source: CT, target: MR).

This script assumes preprocessed volumes live in `processed/ct/image` and
`processed/mr/image` with matching filename prefixes like
`AbdomenMRCT_0001_0001.nii.gz` (CT) and `AbdomenMRCT_0001_0000.nii.gz` (MR).
"""
import argparse
import collections
import csv
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from pathlib import Path
from typing import Sequence, List, Optional
import scipy.ndimage

# Set allocator to avoid fragmentation issues
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import neurite as ne
import voxelmorph as vxm


class MultimodalTrainDataset(torch.utils.data.Dataset):
    """Dataset yielding CT (source) / MR (target) pairs.
    
    Dynamically samples random pairs from unpaired lists, plus a set of fixed paired data.
    """

    def __init__(self, ct_dir: str, mr_dir: str, paired_ct_dir: str = None, paired_mr_dir: str = None, device: str = 'cpu', unpaired: bool = True, max_samples: int = 100):
        self.ct_dir = Path(ct_dir)
        self.mr_dir = Path(mr_dir)
        # self.device = device # Don't move to GPU in dataset, do it in training loop
        self.max_samples = max_samples if max_samples is not None else 100
        
        # 1. Load Unpaired Pool
        self.ct_paths = sorted([p for p in self.ct_dir.iterdir() if p.suffix and not p.name.startswith('.')])
        self.mr_paths = sorted([p for p in self.mr_dir.iterdir() if p.suffix and not p.name.startswith('.')])
        
        # 2. Load Fixed Semi-Supervised Pairs
        self.fixed_pairs = []
        if paired_ct_dir and paired_mr_dir:
            self.fixed_ct_dir = Path(paired_ct_dir)
            self.fixed_mr_dir = Path(paired_mr_dir)
            
            if self.fixed_ct_dir.exists() and self.fixed_mr_dir.exists():
                f_cts = sorted([p for p in self.fixed_ct_dir.iterdir() if p.suffix and not p.name.startswith('.')])
                # Minimal matching logic for these fixed pairs (assuming 0001 <-> 0000 or same name stem)
                mr_map = {}
                for p in self.fixed_mr_dir.iterdir():
                    if p.suffix and not p.name.startswith('.'):
                        # Key extraction: AbdomenMRCT_0009_0000 -> AbdomenMRCT_0009
                        stem = p.name.replace('.nii.gz', '').replace('.nii', '')
                        if stem.endswith('_0000'): stem = stem[:-5]
                        mr_map[stem] = p
                
                for ct in f_cts:
                    stem = ct.name.replace('.nii.gz', '').replace('.nii', '')
                    if stem.endswith('_0001'): stem = stem[:-5]
                    
                    if stem in mr_map:
                        self.fixed_pairs.append((ct, mr_map[stem]))
        
        print(f"Training Dataset: {len(self.ct_paths)} Unpaired CTs, {len(self.mr_paths)} Unpaired MRs.")
        print(f"                  + {len(self.fixed_pairs)} Fixed Pairs found in {paired_ct_dir if paired_ct_dir else 'None'}")
        print(f"                  Epoch Length: {self.max_samples} Random + {len(self.fixed_pairs)} Fixed = {self.max_samples + len(self.fixed_pairs)}")

    def __len__(self):
        # Epoch length = Random Samples + Fixed Pairs
        return self.max_samples + len(self.fixed_pairs)

    def __getitem__(self, idx):
        # Strategy: 
        # Indices [0 ... max_samples-1] -> Randomly sampled unpaired data
        # Indices [max_samples ... end] -> Fixed paired data
        
        if idx < self.max_samples:
            # Random Unpaired Mode
            import random
            ct_path = random.choice(self.ct_paths)
            mr_path = random.choice(self.mr_paths)
        else:
            # Fixed Paired Mode
            # Map idx back to 0..N range
            fixed_idx = idx - self.max_samples
            ct_path, mr_path = self.fixed_pairs[fixed_idx]
            
        ct_nii = nib.load(str(ct_path))
        mr_nii = nib.load(str(mr_path))

        # print(f'Loaded CT: {ct_path.name}, MR: {mr_path.name}') # Too verbose for full training

        ct_np = ct_nii.get_fdata().astype(np.float32)
        mr_np = mr_nii.get_fdata().astype(np.float32)

        # Data is already normalized to [0, 1] during preprocessing (e.g. normalize_for_abdomenmrct.py)
        # Bounding to expected range just in case of float drifts
        ct_np = np.clip(ct_np, 0.0, 1.0)
        mr_np = np.clip(mr_np, 0.0, 1.0)

        ct = torch.from_numpy(ct_np).float().unsqueeze(0)
        mr = torch.from_numpy(mr_np).float().unsqueeze(0)

        return {'source': ct, 'target': mr}
def save_qualitative_results(model, dataset, output_dir, epoch, device='cuda', suffix='', best_sample_idx=None):
    """Save mid-slice images of samples."""
    
    samples_to_plot = []
    
    # If it is test set (suffix contains 'test'), plot all samples
    if 'test' in suffix:
        for i in range(len(dataset)):
            sample = dataset[i]
            # Use filename as tag if available
            tag = sample.get('filename', f'{i:04d}')
            
            # Plot all test cases
            samples_to_plot.append((tag, sample))
    else:
        # Default behavior for validation: Index 0 and Best Dice
        sample_default = dataset[0]
        samples_to_plot.append(('default', sample_default))
        
        if best_sample_idx is not None and best_sample_idx < len(dataset):
            sample_best = dataset[best_sample_idx]
            samples_to_plot.append(('best_dice', sample_best))
        
    for name_tag, sample in samples_to_plot:
        # Dataset returns (C, D, H, W), so unsqueeze for batch dim -> (1, C, D, H, W)
        source = sample['source'].unsqueeze(0).to(device)
        target = sample['target'].unsqueeze(0).to(device)
        
        source_label = None
        target_label = None
        if 'source_label' in sample:
            source_label = sample['source_label'].unsqueeze(0).to(device)
        if 'target_label' in sample:
            target_label = sample['target_label'].unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            displacement, warped_source = model(source, target, return_warped_source=True, return_field_type='displacement')
            
            warped_label = None
            if source_label is not None:
                 trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                 warped_label = trf(source_label, displacement)
        
        # Determine the best slice index
        # Default: middle slice
        slice_idx = source.shape[4] // 2
        
        # Optimization: Find best slice based on Label Richness (Primary) and Dice (Secondary)
        # Goal: Find slices where CT and MR both have labels, preferring more labels, then better overlap.
        if source_label is not None and target_label is not None:
             # GPU-accelerated search for best slice
            best_score = -1.0
            best_idx = slice_idx
            
            D = target_label.shape[4]
            # Search from middle outwards
            search_indices = sorted(range(D), key=lambda i: abs(i - slice_idx))
            
            for z in search_indices:
                # Get slices on GPU
                s_slice = source_label[0, 0, :, :, z]
                t_slice = target_label[0, 0, :, :, z]
                
                # Check for content in both
                if s_slice.sum() == 0 or t_slice.sum() == 0:
                    continue

                # Unique labels
                s_uniq = torch.unique(s_slice)
                t_uniq = torch.unique(t_slice)
                s_uniq = s_uniq[s_uniq > 0.5] # Exclude bg
                t_uniq = t_uniq[t_uniq > 0.5] # Exclude bg
                
                s_count = len(s_uniq)
                t_count = len(t_uniq)
                
                if s_count == 0 or t_count == 0:
                    continue
                
                # Intersection (Common labels)
                common_count = 0
                for lbl in s_uniq:
                    if (t_uniq == lbl).any():
                        common_count += 1
                
                # Compute Dice for this slice if warped label is available
                avg_slice_dice = 0.0
                if warped_label is not None:
                    w_slice = warped_label[0, 0, :, :, z]
                    dice_sum = 0.0
                    dice_n = 0
                    for lbl in t_uniq:
                        m1 = (w_slice == lbl)
                        m2 = (t_slice == lbl)
                        inter = (m1 & m2).sum().float()
                        union = m1.sum().float() + m2.sum().float()
                        if union > 0:
                            dice_sum += (2.0 * inter / union).item()
                            dice_n += 1
                    if dice_n > 0:
                        avg_slice_dice = dice_sum / dice_n

                # Scoring Formula:
                # 1. Base Score: Number of common labels (Most important) -> Weight 10
                # 2. Secondary: Total distinct labels -> Weight 1
                # 3. Tie-breaker: Dice score -> Weight 0.5 (Max contribution 0.5)
                # Example: 4 common labels > 3 common labels regardless of Dice
                # Example: 4 common labels + Dice 0.9 > 4 common labels + Dice 0.8
                
                score = (common_count * 10.0) + (s_count + t_count) + (avg_slice_dice * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_idx = z
            
            slice_idx = best_idx

        # Extract slices using the determined index
        # OPTIMIZATION: Slice on GPU first, then move to CPU
        def get_slice(img_tensor, z_idx):
            if img_tensor is None: return None
            # img_tensor: (1, 1, X, Y, Z)
            # Slice the Z dimension on GPU
            # shape becomes (1, 1, X, Y)
            slice_tensor = img_tensor[:, :, :, :, z_idx]
            # Remove batch and channel dims -> (X, Y)
            slice_np = slice_tensor.detach().cpu().numpy()[0, 0]
            # X-Y plane is Axial
            # Typically requires rotation to align with standard visualization
            return np.rot90(slice_np)

        src_slice = get_slice(source, slice_idx)
        tgt_slice = get_slice(target, slice_idx)
        warped_slice = get_slice(warped_source, slice_idx)
        
        src_lbl_slice = get_slice(source_label, slice_idx)
        tgt_lbl_slice = get_slice(target_label, slice_idx)
        warped_lbl_slice = get_slice(warped_label, slice_idx)
        
        # Plot - Dynamic Rows setup
        has_labels = (src_lbl_slice is not None) and (tgt_lbl_slice is not None)
        
        # Re-organized Layout: 3 Rows x 4 Cols (12 plots total) to fit Jacobian
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        
        # Turn off all axes initially
        for ax in axes.flatten():
            ax.axis('off')
            
        from matplotlib.colors import ListedColormap
        # Create a custom colormap for up to 4 labels + background
        base_colors = np.array([
            [1, 0, 0, 0.5],   # Label 1: Red
            [0, 1, 0, 0.5],   # Label 2: Green
            [0, 0, 1, 0.5],   # Label 3: Blue
            [1, 1, 0, 0.5],   # Label 4: Yellow
            [0, 1, 1, 0.5],   # Label 5: Cyan
            [1, 0, 1, 0.5],   # Label 6: Magenta
        ])
        
        # --- Row 1: Differences & Grid ---
        
        # 1. [0,0] Diff: Moving - Fixed
        diff_moving_fixed = src_slice - tgt_slice
        im_diff1 = axes[0, 0].imshow(diff_moving_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 0].set_title('Diff: Source - Target')
        axes[0, 0].axis('off')

        # 2. [0,1] Diff: Deformed - Target
        diff_warp_fixed = warped_slice - tgt_slice
        im_diff2 = axes[0, 1].imshow(diff_warp_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 1].set_title('Diff: Deformed - Target')
        axes[0, 1].axis('off')

        # 3. [0,2] Deformed Grid

        # Get dimensions from slice
        H, W = src_slice.shape
        # Create a regular grid
        grid_spacing = 10
        xx, yy = np.meshgrid(np.arange(0, W, grid_spacing), np.arange(0, H, grid_spacing))
        
        # d_slice indices: 0=Z, 1=Y, 2=X
        raw_d_slice = displacement.detach().cpu().numpy()[0, :, :, :, slice_idx] 
        d_slice = np.rot90(raw_d_slice, axes=(1, 2))
        
        axes[0, 2].imshow(np.zeros_like(src_slice), cmap='gray', vmin=0, vmax=1) # Black background
        # Plot vertical lines
        for i in range(0, W, grid_spacing):
            if i < d_slice.shape[2]:
                x_plot = i + d_slice[2, :, i]
                y_plot = np.arange(H) + d_slice[1, :, i]
                axes[0, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        # Plot horizontal lines
        for j in range(0, H, grid_spacing):
            if j < d_slice.shape[1]:
                x_plot = np.arange(W) + d_slice[2, j, :]
                y_plot = j + d_slice[1, j, :]
                axes[0, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
            
        axes[0, 2].set_title('Deformed Grid')
        axes[0, 2].set_ylim(H, 0)
        axes[0, 2].set_xlim(0, W)
        axes[0, 2].axis('off')

        # 4. [0,3] Jacobian Determinant
        # Calculate Jacobian determinant of the displacement field
        # displacement is (1, 3, D, H, W)
        # We need to compute spatial gradients.
        # VoxelMorph provides a utility for this, or we can do it manually.
        # Let's use numpy gradient on the 3D displacement field, then slice it.
        disp_np = displacement.detach().cpu().numpy()[0] # (3, D, H, W)
        
        # Compute gradients along spatial dimensions (D, H, W)
        # np.gradient returns a list of arrays, one for each dimension.
        # We want gradients of each component (x, y, z) with respect to each spatial dimension.
        # disp_np[0] is Z displacement, disp_np[1] is Y, disp_np[2] is X
        # Spatial dims are 0:Z, 1:Y, 2:X
        
        # Gradients of Z displacement
        dz_dz, dz_dy, dz_dx = np.gradient(disp_np[0])
        # Gradients of Y displacement
        dy_dz, dy_dy, dy_dx = np.gradient(disp_np[1])
        # Gradients of X displacement
        dx_dz, dx_dy, dx_dx = np.gradient(disp_np[2])
        
        # Jacobian matrix J = I + \nabla u
        # J = [[1 + dx_dx, dx_dy, dx_dz],
        #      [dy_dx, 1 + dy_dy, dy_dz],
        #      [dz_dx, dz_dy, 1 + dz_dz]]
        
        # Compute determinant
        jac_det = ( (1 + dx_dx) * ((1 + dy_dy) * (1 + dz_dz) - dy_dz * dz_dy)
                  - dx_dy * (dy_dx * (1 + dz_dz) - dy_dz * dz_dx)
                  + dx_dz * (dy_dx * dz_dy - (1 + dy_dy) * dz_dx) )
                  
        # Extract the slice
        jac_slice = jac_det[:, :, slice_idx]
        # Rotate to match image orientation
        jac_slice = np.rot90(jac_slice)
        
        # Create RGB image for Jacobian visualization
        # Red: Jac < 0 (Folding)
        # Green: 0 < Jac < 1 (Contraction)
        # Blue: Jac > 1 (Expansion)
        jac_vis = np.zeros((H, W, 3), dtype=np.float32)
        
        # Define colors
        color_red = np.array([1.0, 0.0, 0.0])
        color_green = np.array([0.4, 0.8, 0.4]) # Slightly muted green for better visibility
        color_blue = np.array([0.4, 0.6, 0.9])  # Slightly muted blue
        
        # Apply colors based on conditions
        jac_vis[jac_slice < 0] = color_red
        jac_vis[(jac_slice >= 0) & (jac_slice <= 1)] = color_green
        jac_vis[jac_slice > 1] = color_blue
        
        axes[0, 3].imshow(jac_vis)
        axes[0, 3].set_title('Jacobian Determinant')
        axes[0, 3].axis('off')

        # --- Row 2: Flow & Legend ---

        # 5. [1,0] RGB Displacement
        # Normalize flow for visualization
        dx, dy, dz = d_slice[2], d_slice[1], d_slice[0]
        max_mag = np.max(np.abs(d_slice)) + 1e-5
        
        flow_vis = np.zeros((H, W, 3), dtype=np.float32)
        flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5 # X -> R
        flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5 # Y -> G
        flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5 # Z -> B
        flow_vis = np.clip(flow_vis, 0, 1)
        
        axes[1, 0].imshow(flow_vis)
        axes[1, 0].set_title('RGB Displacement')
        axes[1, 0].axis('off')

        from matplotlib.colors import hsv_to_rgb

        # 6. [1,1] 3D Legend (Center)
        # Note: We skip the HSV mag/angle calculation code from original
        # as we are plotting RGB flow directly.

        ax_legend_spot = axes[1, 1]
        ax_legend_spot.clear() # Clear any previous content
        ax_legend_spot.axis('off')
        # Set aspect equal to ensure circular color wheel
        ax_legend_spot.set_aspect('equal')
        ax_legend_spot.set_xlim(-1.2, 1.2)
        ax_legend_spot.set_ylim(-1.2, 1.2)

        # Draw RGB Color Wheel
        # Fixed Standard Color Wheel (Direction Legend)
        # Explicitly ignore flow magnitude for the wheel itself
        
        # Use simple polar loop to draw segments or a mesh
        # Make the wheel smaller: r from 0.12 to 0.12
        x_wheel = np.linspace(-0.12, 0.12, 100) 
        y_wheel = np.linspace(-0.12, 0.12, 100)
        XW, YW = np.meshgrid(x_wheel, y_wheel)
        RW = np.sqrt(XW**2 + YW**2)
        TW = np.arctan2(YW, XW)
        TW[TW < 0] += 2*np.pi
        
        HW = TW / (2*np.pi)
        SW = np.ones_like(HW)
        VW = np.ones_like(HW)
        # Create mask for ring - Make it smaller/thinner? 
        # User said "光圈图例改成实心的，中间的白色去掉" (solid circle, remove white center). 
        # Let's make outer radius 0.12, inner 0.0
        mask = (RW <= 0.12)
        
        HSV_W = np.stack((HW, SW, VW), axis=-1)
        RGB_W = hsv_to_rgb(HSV_W)
        # Apply alpha channel for mask
        RGBA_W = np.concatenate([RGB_W, mask[..., None].astype(float)], axis=-1)
        
        # Extent matches the meshgrid range
        ax_legend_spot.imshow(RGBA_W, extent=[-0.12, 0.12, -0.12, 0.12], origin='lower')

        # Draw 3D Axes - Keep consistent
        # Origin
        o_x, o_y = 0, 0
        
        # Perspective projection manually:
        # X: right-down
        # Y: left-down
        # Z: up
        
        # Vectors (x, y components for 2D plot)
        # Adjust these angles to match the 3D look in example
        vec_x = np.array([0.5, -0.2])  # Right and slightly down
        vec_y = np.array([-0.4, -0.25]) # Left and slightly down
        vec_z = np.array([0.0, 0.5])   # Up
        
        # Increase arrow length by 30% (0.24 * 1.3 = 0.312)
        scale = 0.312
        
        # Draw Arrows
        ax_legend_spot.arrow(o_x, o_y, vec_x[0]*scale, vec_x[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_y[0]*scale, vec_y[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        ax_legend_spot.arrow(o_x, o_y, vec_z[0]*scale, vec_z[1]*scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
        
        # Labels
        # Offset labels slightly from arrow tips (increase distance)
        ax_legend_spot.text(vec_x[0]*scale*1.6, vec_x[1]*scale*1.6, 'x', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_y[0]*scale*1.6, vec_y[1]*scale*1.6, 'y', fontweight='bold', fontsize=16, ha='center', va='center')
        ax_legend_spot.text(vec_z[0]*scale*1.6, vec_z[1]*scale*1.4, 'z', fontweight='bold', fontsize=16, ha='center', va='bottom')
        
        # Add Range Text at bottom
        range_text = f"[{ -max_mag:.2f}, {max_mag:.2f}]"
        ax_legend_spot.text(0, -0.35, range_text, ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        # 7. [1,2] Overlay Warped Label on Target Image (Result vs GT)
        if has_labels:
             # Color definitions
            base_colors = {
                1: '#8B0000', # Dark Red (Liver)
                2: '#228B22', # Forest Green (Spleen) - Brighter than Dark Green
                3: '#4682B4', # Steel Blue (R-Kidney) - Lighter/Darker Blue mix, easier to see than Dark Blue
                4: '#DAA520', # Goldenrod (L-Kidney) - Brighter than Dark Goldenrod
                5: '#008B8B'  # Dark Cyan
            }
            # Bright colors for moving/warped (source/pred) labels - Standard colors, closer to Base than Neon
            bright_colors = {
                1: '#FF0000', # Red
                2: '#00FF00', # Lime
                3: '#0000FF', # Blue
                4: '#FFD700', # Gold
                5: '#00FFFF'  # Cyan
            }
            # Dark colors for text to ensure readability
            text_colors = {
                1: '#A52A2A', # Brown/Red (Higher saturation than 660000)
                2: '#2E8B57', # Sea Green (Higher saturation than 1A521A)
                3: '#4682B4', # Steel Blue (Lower saturation than Royal Blue)
                4: '#CD853F', # Peru/Golden (Higher saturation than 8B6914)
                5: '#20B2AA'  # Light Sea Green (Higher saturation than 005C5C)
            }
        
        axes[1, 2].imshow(warped_slice, cmap='gray')
        
        if has_labels and tgt_lbl_slice is not None and warped_lbl_slice is not None:
             unique_labels = np.unique(np.concatenate([tgt_lbl_slice, warped_lbl_slice]))
             unique_labels = unique_labels[unique_labels > 0]
             
             # Calculate total width needed for horizontal layout
             num_labels = len(unique_labels)
             
             dice_results = []
             for lbl in unique_labels:
                 mask_tgt = (tgt_lbl_slice == lbl)
                 c_base = base_colors.get(int(lbl), 'white')
                 if np.any(mask_tgt):
                     # Solid line for GT (Target)
                     axes[1, 2].contour(mask_tgt, colors=[c_base], linewidths=1.5, linestyles='solid')
                     
                 mask_warp = (warped_lbl_slice == lbl)
                 c_bright = bright_colors.get(int(lbl), 'white')
                 if np.any(mask_warp):
                     # Solid line for Prediction
                     axes[1, 2].contour(mask_warp, colors=[c_bright], linewidths=1.5, linestyles='solid')
                     
                 # Calculate and display Dice
                 intersection = np.logical_and(mask_warp, mask_tgt).sum()
                 union = mask_warp.sum() + mask_tgt.sum()
                 if union > 0:
                     dice = 2.0 * intersection / union
                     c_text = text_colors.get(int(lbl), 'white')
                     dice_results.append((dice, c_text))
             
             # Sort by dice descending
             dice_results.sort(key=lambda x: x[0], reverse=True)
             
             spacing = 0.20 # Increased spacing
             total_text_width = len(dice_results) * spacing
             x_offset = (1.0 - total_text_width) / 2.0 + spacing / 2.0
             
             for dice, color in dice_results:
                 axes[1, 2].text(x_offset, 0.02, f'{dice:.2f}', color=color, transform=axes[1, 2].transAxes, fontsize=24, fontweight='bold', ha='center')
                 x_offset += spacing

        axes[1, 2].set_title('Result vs GT')
        axes[1, 2].axis('off')
        
        # 8. [1,3] Empty or something else
        axes[1, 3].axis('off')

        # --- Row 3: Label Overlays (Individual) ---
        if has_labels:
            def plot_label_contour(ax, bg_img, label_img, title, color_lookup, target_img=None):
                ax.imshow(bg_img, cmap='gray')
                if label_img is not None:
                     if target_img is not None:
                         unique_labels = np.unique(np.concatenate([label_img, target_img]))
                     else:
                         unique_labels = np.unique(label_img)
                     unique_labels = unique_labels[unique_labels > 0]
                     
                     dice_results = []
                     for lbl in unique_labels:
                         mask = (label_img == lbl)
                         c = color_lookup.get(int(lbl), 'white')
                         if np.any(mask):
                             ax.contour(mask, colors=[c], linewidths=1.2)
                             
                         if target_img is not None:
                             mask_tgt = (target_img == lbl)
                             intersection = np.logical_and(mask, mask_tgt).sum()
                             union = mask.sum() + mask_tgt.sum()
                             if union > 0:
                                 dice = 2.0 * intersection / union
                                 c_text = text_colors.get(int(lbl), 'white')
                                 dice_results.append((dice, c_text))
                     
                     if dice_results:
                         dice_results.sort(key=lambda x: x[0], reverse=True)
                         spacing = 0.20 # Increased spacing
                         total_text_width = len(dice_results) * spacing
                         x_offset = (1.0 - total_text_width) / 2.0 + spacing / 2.0
                         
                         for dice, color in dice_results:
                             ax.text(x_offset, 0.02, f'{dice:.2f}', color=color, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='center')
                             x_offset += spacing
                ax.set_title(title)
                ax.axis('off')

            # 9. [2,0] Source + Labels
            plot_label_contour(axes[2, 0], src_slice, src_lbl_slice, 'Source + Labels', bright_colors, target_img=tgt_lbl_slice)
            
            # 10. [2,1] Target + Labels
            plot_label_contour(axes[2, 1], tgt_slice, tgt_lbl_slice, 'Target + Labels', base_colors)
            
            # 11. [2,2] Warped Source + Warped Labels
            plot_label_contour(axes[2, 2], warped_slice, warped_lbl_slice, 'Deformed + Labels', bright_colors, target_img=tgt_lbl_slice)
            
            # 12. [2,3] Empty
            axes[2, 3].axis('off')

        plt.suptitle(f'Epoch {epoch} - Sample {name_tag} (Slice Z={slice_idx})', fontsize=16)
        
        # Use tight_layout with rect to avoid overlapping suptitle, but wrap in try-except to handle
        # incompatibilities with inset_axes (which cause the UserWarning)
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except UserWarning:
            pass # Ignore warning about incompatible axes (inset_axes)
        
        # Clean filename to be safe for filesystem
        safe_tag = str(name_tag).replace('/', '_').replace('\\', '_')
        filename_out = f'vis_epoch_{epoch:04d}{suffix}_{safe_tag}.png'
        out_file = output_dir / filename_out
        plt.savefig(str(out_file))
        plt.close(fig)


def plot_history(log_file, output_dir):
    """Plot training history from log file."""
    epochs = []
    train_loss = []
    val_loss = []
    val_dice = []
    val_jac = []
    val_mag = []
    test_loss = []
    test_dice = []
    test_jac = []
    test_mag = []

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            val_dice.append(float(row.get('val_dice', 0.0)))
            val_jac.append(float(row.get('val_neg_jac_ratio', 0.0)))
            # Handle potentially missing columns if log file format changed mid-way
            val_mag.append(float(row.get('val_mag', 0.0)))
            test_loss.append(float(row.get('test_loss', 0.0)))
            test_dice.append(float(row.get('test_dice', 0.0)))
            test_jac.append(float(row.get('test_jac', 0.0)))
            test_mag.append(float(row.get('test_mag', 0.0)))

    # Plot Loss
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    if any(l != 0 for l in test_loss):
        plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # Plot Dice
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_dice, label='Val Dice')
    if any(d != 0 for d in test_dice):
        plt.plot(epochs, test_dice, label='Test Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot Jacobian
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_jac, label='Val Neg Jac')
    if any(j != 0 for j in test_jac):
        plt.plot(epochs, test_jac, label='Test Neg Jac')
    plt.title('Negative Jacobian Ratio (Folding)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot Magnitude
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_mag, label='Val Mag')
    if any(m != 0 for m in test_mag):
        plt.plot(epochs, test_mag, label='Test Mag')
    plt.title('Deformation Magnitude')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()


class MultimodalValidationDataset(torch.utils.data.Dataset):
    """Dataset yielding all fixed pairs of CT (source) / MR (target) for validation.
       Optionally yields labels for Dice calculation."""

    def __init__(self, ct_dir: str, mr_dir: str, ct_label_dir: str = None, mr_label_dir: str = None, device: str = 'cpu', paired: bool = False):
        self.ct_dir = Path(ct_dir)
        self.mr_dir = Path(mr_dir)
        self.ct_label_dir = Path(ct_label_dir) if ct_label_dir else None
        self.mr_label_dir = Path(mr_label_dir) if mr_label_dir else None
        self.device = device
        self.ct_paths = sorted([p for p in self.ct_dir.iterdir() if p.suffix and not p.name.startswith('.')])
        self.mr_paths = sorted([p for p in self.mr_dir.iterdir() if p.suffix and not p.name.startswith('.')])
        
        # Verify labels exist if requested
        self.ct_labels = {}
        if self.ct_label_dir:
            for p in self.ct_label_dir.iterdir():
                 if p.suffix:
                     # Assumes label filename matches image filename or is discoverable
                     self.ct_labels[p.name] = p

        self.mr_labels = {}
        if self.mr_label_dir:
            for p in self.mr_label_dir.iterdir():
                 if p.suffix:
                     self.mr_labels[p.name] = p

        self.pairs = []
        
        def get_stem(p):
            # robustly remove extensions
            name = p.name
            if name.endswith('.nii.gz'):
                return name[:-7]
            elif name.endswith('.nii'):
                return name[:-4]
            return p.stem

        if paired:
            # Paired mode: Match files by common identifier
             print(f"DEBUG: Paired Mode Enabled.")
             print(f"DEBUG: Found {len(self.ct_paths)} CT candidates and {len(self.mr_paths)} MR candidates.")
             
             mr_map = {}
             for p in self.mr_paths:
                stem = get_stem(p)
                if stem.endswith('_0000'):
                    key = stem[:-5]
                else:
                    key = stem
                mr_map[key] = p

             for ct in self.ct_paths:
                stem = get_stem(ct)
                if stem.endswith('_0001'):
                    key = stem[:-5]
                else:
                    key = stem
                
                if key in mr_map:
                    self.pairs.append((ct, mr_map[key]))
                else:
                    # Fallback: try direct name match if prefixes not heavily used
                    if ct.name in mr_map:
                         self.pairs.append((ct, mr_map[ct.name]))
                    else:
                        print(f"DEBUG: Could not find match for CT: {ct.name} (Key: {key})")
        else:
            # Unpaired (Cross-Product) mode: All CTs with all MRs
            for ct in self.ct_paths:
                for mr in self.mr_paths:
                    self.pairs.append((ct, mr))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mr_path = self.pairs[idx]
        
        ct_nii = nib.load(str(ct_path))
        mr_nii = nib.load(str(mr_path))

        ct_np = ct_nii.get_fdata().astype(np.float32)
        mr_np = mr_nii.get_fdata().astype(np.float32)
        
        # Data is already normalized to [0, 1] during preprocessing
        # Bounding to expected range just in case of float drifts
        ct_np = np.clip(ct_np, 0.0, 1.0)
        mr_np = np.clip(mr_np, 0.0, 1.0)

        ct = torch.from_numpy(ct_np).float().unsqueeze(0)
        mr = torch.from_numpy(mr_np).float().unsqueeze(0)
        
        sample = {'source': ct, 'target': mr}

        # Load labels if available and filenames match
        if self.ct_label_dir and ct_path.name in self.ct_labels:
            ct_lbl_path = self.ct_labels[ct_path.name]
            ct_lbl = torch.from_numpy(nib.load(str(ct_lbl_path)).get_fdata()).float().unsqueeze(0)
            sample['source_label'] = ct_lbl
            
        if self.mr_label_dir and mr_path.name in self.mr_labels:
             mr_lbl_path = self.mr_labels[mr_path.name]
             mr_lbl = torch.from_numpy(nib.load(str(mr_lbl_path)).get_fdata()).float().unsqueeze(0)
             sample['target_label'] = mr_lbl

        sample['filename'] = ct_path.name.replace('.nii.gz', '').replace('.nii', '')
        return sample


def check_early_stopping(loss_history, patience=20, threshold=0.0, warm_start_steps=10):
    """
    Early stopping function.
    Returns True if training should stop.
    """
    if len(loss_history) < warm_start_steps:
        return False

    best_loss_idx = np.argmin(loss_history)
    epochs_since_best = len(loss_history) - 1 - best_loss_idx

    if epochs_since_best >= patience:
        return True
    
    return False


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    steps_per_epoch: int,   # Used only for progress bar calculation now if we iterate full loader
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = 'cuda'
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 0
    total_grad_norm = 0.0
    num_grad_norm = 0
    effective_update_steps = 0
    nonfinite_steps = 0

    # Keep AMP scaler persistent across epochs. If not provided, fallback to local scaler.
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Prefer bfloat16 for better numeric range when available (especially for MI/localMI).
    amp_dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    # Track a reference parameter to monitor whether optimizer steps are actually changing weights.
    ref_param = next((p for p in model.parameters() if p.requires_grad), None)

    # Iterate through the entire dataloader (all 1500+ pairs if unpaired)
    # The dataloader is now a finite Map-style Dataset, not an infinite Iterable
    for batch in tqdm(dataloader, desc="Training Batch"):
        optimizer.zero_grad(set_to_none=True)

        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(scaler is not None)):
            displacement, warped_source, coarse_flows = model(
                source,
                target,
                return_warped_source=True,
                return_field_type='displacement',
                return_coarse_flows=True
            )

        # AMP 兼容性保护：强制将预测结果和 Loss 计算切回 float32。
        # 这是配准任务的常见坑，因为形变场和损失求导在 float16 下极易精度溢出
        target_float = target.float()
        warped_source_float = warped_source.float()
        displacement_float = displacement.float()
        
        img_loss = image_loss_fn(target_float, warped_source_float).mean()
        
        if isinstance(image_loss_fn, ne.nn.modules.NCC):
            img_loss = -img_loss
            
        grad_loss = grad_loss_fn(displacement_float).mean()
        
        # --- Deep Supervision for DAPS coarse flows ---
        daps_reg_loss = 0.0
        if len(coarse_flows) > 0:
            for c_flow in coarse_flows:
                daps_reg_loss += grad_loss_fn(c_flow.float()).mean()
            daps_reg_loss = daps_reg_loss / len(coarse_flows)
            
        loss = loss_weights[0] * img_loss + loss_weights[1] * (grad_loss + daps_reg_loss)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            nonfinite_steps += 1
            num_steps += 1
            continue
        
        # Optimization: Scaled Backward
        scaler.scale(loss).backward()

        # Unscale before grad-norm computation so the value is meaningful.
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)

        # 增加梯度裁剪，防止黑背景区导致的除零或梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        grad_sq_sum = 0.0
        has_nonfinite_grad = False
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    has_nonfinite_grad = True
                    break
                grad_sq_sum += torch.sum(g * g).item()

        if has_nonfinite_grad:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            nonfinite_steps += 1
            num_steps += 1
            continue

        if grad_sq_sum > 0 and np.isfinite(grad_sq_sum):
            total_grad_norm += grad_sq_sum ** 0.5
            num_grad_norm += 1

        # Monitor whether parameters are updated this step.
        before_ref = ref_param.detach().clone() if ref_param is not None else None
        scaler.step(optimizer)
        scaler.update()

        if before_ref is not None:
            delta = (ref_param.detach() - before_ref).abs().mean().item()
            if delta > 0:
                effective_update_steps += 1

        total_loss += loss.item()
        # Track components for debugging
        # Note: We need to detach to avoid accumulation
        num_steps += 1
        
    # Return breakdown (approximated from last batch or accumulate if needed, 
    # but strictly we just need to see the scale)
    avg_grad_norm = total_grad_norm / num_grad_norm if num_grad_norm > 0 else 0.0
    update_ratio = effective_update_steps / max(num_steps, 1)
    if nonfinite_steps > 0:
        print(f'  [Warning] Non-finite train steps: {nonfinite_steps}/{num_steps}. Consider reducing lr or disabling AMP for this loss.')
    return total_loss / num_steps, img_loss.item(), grad_loss.item() + (daps_reg_loss.item() if isinstance(daps_reg_loss, torch.Tensor) else 0), avg_grad_norm, update_ratio


def compute_hd95(ground_truth, prediction, spacing=None):
    """
    Compute 95th Hausdorff Distance between two binary masks.
    Uses K-D tree for much faster computation.
    """
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
    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)
    
    dist_pred_to_gt, _ = tree_gt.query(pts_pred, k=1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt, k=1)
    
    hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)
    
    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    device: str = 'cuda',
    fast: bool = False  # Optimization: Skip slow metrics
):
    """
    Run lightweight validation for model selection.
    Returns: avg_loss, avg_dice, avg_hd95, avg_time, avg_jac, avg_mag
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_hd95 = 0.0 
    total_time = 0.0
    total_neg_jac = 0.0
    total_mag = 0.0
    
    num_batches = 0
    num_dice_batches = 0
    num_hd95_batches = 0
    num_jac_batches = 0
    
    # Optimization: Pre-fetch to GPU
    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)

            start_time = time.time()
            if device == 'cuda':
                torch.cuda.synchronize()
            
            displacement, warped_source = model(
                source,
                target,
                return_warped_source=True,
                return_field_type='displacement'
            )
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            # 1. Metrics: Magnitude
            # Optimization: Skip magnitude for validation set to save time
            # disp_mag = torch.sqrt(torch.sum(displacement ** 2, dim=1))
            # total_mag += disp_mag.mean().item()

            # 2. Time
            batch_time = end_time - start_time
            total_time += (batch_time / source.shape[0])
            
            # 3. Loss
            if isinstance(image_loss_fn, ne.nn.modules.NCC):
                img_loss = -image_loss_fn(target, warped_source)
            else:
                img_loss = image_loss_fn(target, warped_source)
            grad_loss = grad_loss_fn(displacement)
            loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
            total_loss += loss.item()
            num_batches += 1

            # Optimization: Skip Jacobian for validation set.
            if False and not fast:
                # 4. Jacobian (Slow: CPU Transfer + Numpy)
                disp_np = displacement.detach().cpu().numpy()
                disp_np = np.transpose(disp_np, (0, 2, 3, 4, 1))
                target_np = batch['target'].detach().cpu().numpy() # Use original batch valid target
                
                batch_neg_jac = 0.0
                for i in range(disp_np.shape[0]):
                    jac_det = vxm.py.utils.jacobian_determinant(disp_np[i])
                    mask = target_np[i, 0] > 0.01
                    if jac_det.shape != mask.shape:
                        diff = np.array(mask.shape) - np.array(jac_det.shape)
                        d_start, h_start, w_start = diff // 2
                        d_end = mask.shape[0] - (diff[0] - d_start)
                        h_end = mask.shape[1] - (diff[1] - h_start)
                        w_end = mask.shape[2] - (diff[2] - w_start)
                        mask = mask[d_start:d_end, h_start:h_end, w_start:w_end]

                    valid_mask_sum = np.sum(mask)
                    if valid_mask_sum > 0:
                        batch_neg_jac += np.sum((jac_det <= 0) & mask) / valid_mask_sum
                
                total_neg_jac += (batch_neg_jac / disp_np.shape[0])
                num_jac_batches += 1

            # 5. Dice & HD95
            # Optimization: Skip Dice and HD95 for validation set, only compute loss.
            if False and 'source_label' in batch and 'target_label' in batch:
                source_label = batch['source_label'].to(device, non_blocking=True)
                target_label = batch['target_label'].to(device, non_blocking=True)
                
                trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                warped_label = trf(source_label, displacement)
                
                wl_np = warped_label.cpu().numpy()
                tl_np = target_label.cpu().numpy()
                
                # Simple batch average dice (Always compute)
                dice_score = vxm.py.utils.dice(wl_np, tl_np)
                total_dice += dice_score.mean()
                num_dice_batches += 1
                
                if not fast:
                    # HD95 (Very Slow: Scipy)
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
                        total_hd95 += (batch_hd95_sum / batch_hd95_count)
                        num_hd95_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_dice_batches if num_dice_batches > 0 else 0.0
    avg_hd95 = total_hd95 / num_hd95_batches if num_hd95_batches > 0 else 0.0
    avg_time = total_time / num_batches if num_batches > 0 else 0.0
    avg_neg_jac = total_neg_jac / num_jac_batches if num_jac_batches > 0 else 0.0
    avg_mag = total_mag / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice, avg_hd95, avg_time, avg_neg_jac, avg_mag


def test_evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    device: str = 'cuda',
    fast: bool = False
):
    """
    Run Comprehensive Testing. 
    Returns detailed metrics: avg_loss, avg_dice, avg_hd95, avg_time, avg_reg_time, avg_jac, avg_mag, 
                              best_sample_idx, per_label_dice_dict, raw_sample_results
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_dice_per_label = {} 
    total_label_counts = {} 
    raw_sample_results = []
    
    total_hd95 = 0.0
    total_time = 0.0
    total_reg_time = 0.0 
    total_neg_jac = 0.0
    total_mag = 0.0
    
    num_batches = 0
    num_dice_batches = 0
    num_hd95_batches = 0
    num_jac_batches = 0
    
    best_dice = -1.0
    best_dice_idx = None
    current_idx_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            # 1. Measure Pure Registration Time
            start_time_reg = time.time()
            if device == 'cuda': torch.cuda.synchronize()
            _ = model(source, target, return_warped_source=False, return_field_type='displacement')
            if device == 'cuda': torch.cuda.synchronize()
            end_time_reg = time.time()
            total_reg_time += ((end_time_reg - start_time_reg) / source.shape[0])

            # 2. Measure Full Inference Time & Forward
            start_time = time.time()
            if device == 'cuda': torch.cuda.synchronize()
            displacement, warped_source = model(source, target, return_warped_source=True, return_field_type='displacement')
            if device == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            total_time += ((end_time - start_time) / source.shape[0])
            
            # Metrics: Magnitude
            disp_mag = torch.sqrt(torch.sum(displacement ** 2, dim=1))
            total_mag += disp_mag.mean().item()

            # Loss
            if isinstance(image_loss_fn, ne.nn.modules.NCC):
                img_loss = -image_loss_fn(target, warped_source)
            else:
                img_loss = image_loss_fn(target, warped_source)
            grad_loss = grad_loss_fn(displacement)
            loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
            total_loss += loss.item()
            num_batches += 1

            # Jacobian (Conditional)
            disp_np = displacement.detach().cpu().numpy()
            disp_np = np.transpose(disp_np, (0, 2, 3, 4, 1))

            if not fast:
                target_np = target.detach().cpu().numpy()
                batch_neg_jac = 0.0
                for i in range(disp_np.shape[0]):
                    jac_det = vxm.py.utils.jacobian_determinant(disp_np[i])
                    mask = target_np[i, 0] > 0.01
                    if jac_det.shape != mask.shape:
                        diff = np.array(mask.shape) - np.array(jac_det.shape)
                        d_start, h_start, w_start = diff // 2
                        d_end = mask.shape[0] - (diff[0] - d_start)
                        h_end = mask.shape[1] - (diff[1] - h_start)
                        w_end = mask.shape[2] - (diff[2] - w_start)
                        mask = mask[d_start:d_end, h_start:h_end, w_start:w_end]

                    valid_mask_sum = np.sum(mask)
                    if valid_mask_sum > 0:
                        batch_neg_jac += np.sum((jac_det <= 0) & mask) / valid_mask_sum
                total_neg_jac += (batch_neg_jac / disp_np.shape[0])
                num_jac_batches += 1

            # Detailed Dice & HD95
            if 'source_label' in batch and 'target_label' in batch:
                source_label = batch['source_label'].to(device)
                target_label = batch['target_label'].to(device)
                trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                warped_label = trf(source_label, displacement)
                
                wl_np = warped_label.cpu().numpy()
                tl_np = target_label.cpu().numpy()
                sl_np = source_label.cpu().numpy()
                
                batch_dice_sum = 0.0
                batch_dice_count = 0 
                
                # Per-sample processing
                for b in range(wl_np.shape[0]):
                    # 只评价源图像和目标图像中同时存在的标签 (Fair Dice)
                    u_s = np.unique(sl_np[b])
                    u_t = np.unique(tl_np[b])
                    u_labels = np.intersect1d(u_s, u_t)
                    u_labels = u_labels[u_labels > 0.5]
                    
                    # Dice (Always Compute)
                    if len(u_labels) > 0:
                        dice_scores = vxm.py.utils.dice(wl_np[b], tl_np[b], labels=u_labels)
                        sample_mean_dice = dice_scores.mean()
                        batch_dice_sum += sample_mean_dice
                        batch_dice_count += 1
                        
                        # Accumulate per-label
                        label_dice_dict = {}
                        for l_idx, label_val in enumerate(u_labels):
                            l_key = int(label_val)
                            d_val = dice_scores[l_idx]
                            label_dice_dict[l_key] = d_val
                            if l_key not in total_dice_per_label:
                                total_dice_per_label[l_key] = 0.0
                                total_label_counts[l_key] = 0
                            total_dice_per_label[l_key] += d_val
                            total_label_counts[l_key] += 1
                    else:
                        sample_mean_dice = 0.0
                        label_dice_dict = {}

                    # Raw result init
                    sample_filename = ""
                    if 'filename' in batch:
                        if isinstance(batch['filename'], (list, tuple)):
                            sample_filename = batch['filename'][b]
                        elif isinstance(batch['filename'], str):
                            sample_filename = batch['filename']

                    sample_res = {
                        'sample_idx': current_idx_offset + b,
                        'filename': sample_filename,
                        'dice': sample_mean_dice,
                        'hd95': np.nan,
                        'label_dice': label_dice_dict
                    }
                    
                    # HD95 (Conditional)
                    if not fast:
                        sample_hd95_sum = 0.0
                        sample_hd95_count = 0
                        for l in u_labels:
                            mask_pred = (wl_np[b, 0] == l)
                            mask_gt = (tl_np[b, 0] == l)
                            hd = compute_hd95(mask_gt, mask_pred)
                            if not np.isnan(hd):
                                sample_hd95_sum += hd
                                sample_hd95_count += 1
                        
                        if sample_hd95_count > 0:
                            sample_avg_hd95 = sample_hd95_sum / sample_hd95_count
                            sample_res['hd95'] = sample_avg_hd95
                            # Accumulate - but wait, how to average globally?
                            # previous code logic was a bit messy on global HD95 average
                            # Let's just track valid batches or sum of means?
                            # Simplified: accumulated into total_hd95 if valid
                            pass 

                    # Add to list
                    raw_sample_results.append(sample_res)

                if batch_dice_count > 0:
                    total_dice += (batch_dice_sum / batch_dice_count)
                    num_dice_batches += 1
                
                # Check Best Dice
                if batch_dice_count > 0:
                    current_batch_mean = batch_dice_sum / batch_dice_count
                    if current_batch_mean > best_dice:
                        best_dice = current_batch_mean
                        best_dice_idx = current_idx_offset
                
                # HD95 Batch Accumulation (Conditional)
                if not fast:
                     # Re-compute batch mean HD95 from sample results for this batch
                     # The last batch_dice_count items in raw_sample_results correspond to this batch
                     # Filter those with valid HD95
                     current_batch_results = raw_sample_results[-len(wl_np):] # approximation
                     valid_batch_hds = [r['hd95'] for r in current_batch_results if not np.isnan(r['hd95'])]
                     if valid_batch_hds:
                         total_hd95 += np.mean(valid_batch_hds)
                         num_hd95_batches += 1

            current_idx_offset += source.shape[0]

    # Post-process HD95 from raw results to ensure consistency
    valid_hd95_values = [r['hd95'] for r in raw_sample_results if not np.isnan(r['hd95'])]
    avg_hd95 = np.mean(valid_hd95_values) if valid_hd95_values else 0.0
    std_hd95 = np.std(valid_hd95_values) if valid_hd95_values else 0.0

    valid_dice_values = [r['dice'] for r in raw_sample_results]
    std_dice = np.std(valid_dice_values) if valid_dice_values else 0.0

    valid_jac_values = [r['jac'] for r in raw_sample_results if 'jac' in r]
    std_neg_jac = np.std(valid_jac_values) if valid_jac_values else 0.0

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_dice_batches if num_dice_batches > 0 else 0.0
    avg_time = total_time / num_batches if num_batches > 0 else 0.0
    avg_reg_time = total_reg_time / num_batches if num_batches > 0 else 0.0
    avg_neg_jac = total_neg_jac / num_jac_batches if num_jac_batches > 0 else 0.0
    avg_mag = total_mag / num_batches if num_batches > 0 else 0.0
    
    # Calculate Per-Label Average Dice and Std
    avg_dice_per_label = {}
    std_dice_per_label = {}
    # Extract ALL dice records for each label across batches
    label_dice_lists = collections.defaultdict(list)
    for res in raw_sample_results:
        for k, v in res.get('label_dice', {}).items():
            label_dice_lists[k].append(v)
            
    for l_key in total_dice_per_label:
        if total_label_counts[l_key] > 0:
            avg_dice_per_label[l_key] = total_dice_per_label[l_key] / total_label_counts[l_key]
            std_dice_per_label[l_key] = np.std(label_dice_lists[l_key]) if label_dice_lists[l_key] else 0.0
    
    return avg_loss, avg_dice, std_dice, avg_hd95, std_hd95, avg_time, avg_reg_time, avg_neg_jac, std_neg_jac, avg_mag, best_dice_idx, avg_dice_per_label, std_dice_per_label, raw_sample_results


def main():
    parser = argparse.ArgumentParser(description='Train multimodal CT->MR VoxelMorph')
    parser.add_argument('--ct-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/train/images/ct', help='CT train images (source)')
    parser.add_argument('--mr-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/train/images/mr', help='MR train images (target)')
    parser.add_argument('--paired-ct-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images/ct', help='Paired CT train images (source)')
    parser.add_argument('--paired-mr-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images/mr', help='Paired MR train images (target)')
    parser.add_argument('--ct-val-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/images/ct', help='Validation CT images')
    parser.add_argument('--mr-val-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/images/mr', help='Validation MR images')
    parser.add_argument('--ct-val-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/labels/ct', help='Validation CT labels')
    parser.add_argument('--mr-val-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/val/labels/mr', help='Validation MR labels')
    parser.add_argument('--ct-test-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/images/ct', help='Test CT images (for monitoring)')
    parser.add_argument('--mr-test-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/images/mr', help='Test MR images (for monitoring)')
    parser.add_argument('--ct-test-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/labels/ct', help='Test CT labels')
    parser.add_argument('--mr-test-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/labels/mr', help='Test MR labels')
    parser.add_argument('--output', type=str, default='/root/autodl-tmp/models/multimodal_vxm.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Max training samples to use (subsample)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01, help='Regularization weight (0.01 for smooth, 1.0 for rigid)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--image-loss', type=str, choices=['mse', 'ncc', 'mi', 'local_mi'], default='ncc', help='Image similarity loss')
    parser.add_argument('--ncc-win', type=int, default=9, help='NCC window size')
    parser.add_argument('--patch-size', type=int, default=9, help='Local Mutual Information patch size')
    parser.add_argument('--mi-bins', type=int, default=32, help='Bins for Mutual Information')
    parser.add_argument('--unpaired', action='store_true', default=False, help='If set, force unpaired training even if filenames match.')
    parser.add_argument('--val-paired', action='store_true', default=False, help='Validation data is paired')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument(
        '--warm-start', type=int, default=10, help='Early stopping warm start steps'
    )
    parser.add_argument('--integration-steps', type=int, default=0, help='number of integration steps for diffeomorphic registration')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Optimization: Enable cuDNN benchmark
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Dataloader Dataset Init (Before model to grab shape)
    # Switch to Map-style Dataset for consistent epoch definition
    dataset = MultimodalTrainDataset(args.ct_dir, args.mr_dir, paired_ct_dir=args.paired_ct_dir, paired_mr_dir=args.paired_mr_dir, device=device, unpaired=args.unpaired, max_samples=args.max_train_samples)
    
    # Grab inshape from dataset
    sample = dataset[0]
    # sample['source'] shape is (1, D, H, W). We want (D, H, W)
    inshape = tuple(sample['source'].shape[1:])

    # Model: SiameseUNetBaseline (原双流共享权重编码器)
    # model = vxm.nn.SiameseUNetBaseline(
    #     inshape=inshape,
    #     ndim=3,
    #     enc_nf=[32, 64, 64, 64],
    #     dec_nf=[64, 64, 64, 32],
    #     int_steps=args.integration_steps,
    # ).to(device)
    
    # 替换为：调用浅层独立深层共享编码器 (DecoupledEncoder)
    enc_channels = [32, 64, 64, 64]
    dec_channels = [64, 64, 64, 32]
    model = vxm.nn.SiameseUNetBaseline(
        inshape=inshape,
        in_channels=1,
        enc_nf=enc_channels,
        dec_nf=dec_channels,
        ndim=3,
        int_steps=args.integration_steps,
        decouple_layers=2,
        use_daps=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Total Trainable Parameters: {total_params:,}')

    # Losses
    if args.image_loss == 'mse':
        image_loss_fn = ne.nn.modules.MSE().to(device)
    elif args.image_loss == 'mi':
        image_loss_fn = vxm.nn.losses.MutualInformation(num_bin=args.mi_bins).to(device)
    elif args.image_loss == 'local_mi':
        image_loss_fn = vxm.nn.losses.localMutualInformation(patch_size=args.patch_size, num_bin=args.mi_bins).to(device)
    else:
        # neurite NCC expects window size; default None will choose automatic
        image_loss_fn = ne.nn.modules.NCC(window_size=args.ncc_win, eps=1e-3).to(device)

    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Dataloader Datasets were initialized above
    # Shuffle is KEY here: it mixes Easy and Hard pairs within each epoch
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # Important: Shuffle every epoch
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0)
    )
    
    # Validation Dataloader
    val_loader = None
    if args.ct_val_dir and args.mr_val_dir:
        val_ct_path = Path(args.ct_val_dir)
        val_mr_path = Path(args.mr_val_dir)
        if val_ct_path.exists() and val_mr_path.exists():
            val_dataset = MultimodalValidationDataset(
                args.ct_val_dir, 
                args.mr_val_dir, 
                ct_label_dir=args.ct_val_label_dir,
                mr_label_dir=args.mr_val_label_dir,
                device=device,
                paired=args.val_paired
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            print(f'Validation set loaded: {len(val_dataset)} pairs.')
        else:
            print(f'Warning: Validation directories not found ({args.ct_val_dir}, {args.mr_val_dir}). Validation skipped.')

    # Test/Monitor Dataloader
    test_loader = None
    if args.ct_test_dir and args.mr_test_dir:
        test_ct_path = Path(args.ct_test_dir)
        test_mr_path = Path(args.mr_test_dir)
        if test_ct_path.exists() and test_mr_path.exists():
            test_dataset = MultimodalValidationDataset(
                args.ct_test_dir, 
                args.mr_test_dir, 
                ct_label_dir=args.ct_test_label_dir,
                mr_label_dir=args.mr_test_label_dir,
                device=device,
                paired=True
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            print(f'Test (Monitor) set loaded: {len(test_dataset)} pairs.')
        else:
            print(f'Warning: Test directories not found ({args.ct_test_dir}, {args.mr_test_dir}). Test monitoring skipped.')

    import datetime
    
    # Ensure output dir
    # Create a timestamped directory for this run to keep logs and checkpoints separate
    # Use Beijing Time (UTC+8)
    utc_now = datetime.datetime.utcnow()
    beijing_time = utc_now + datetime.timedelta(hours=8)
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    
    input_output_path = Path(args.output)
    loss_tag = args.image_loss.lower()

    # Structure: <parent>/<stem>_<loss>_<timestamp>/<stem>_<loss>.pt
    run_dir = input_output_path.parent / f"{input_output_path.stem}_{loss_tag}_{timestamp}"
    model_filename = f"{input_output_path.stem}_{loss_tag}{input_output_path.suffix}"
    out_path = run_dir / model_filename
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for this run: {out_path.parent}")

    # Logging setup
    log_file = out_path.parent / 'train_log.csv'
    
    # Save training configuration
    config_file = out_path.parent / 'config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Start Time: {timestamp}\n")
        f.write(f"Dataset: {Path(args.ct_dir).parent.parent.parent.name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Image Loss: {args.image_loss}\n")
        f.write(f"NCC Window: {args.ncc_win}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Integration Steps: {args.integration_steps}\n")
        f.write(f"Unpaired: {args.unpaired}\n")
        f.write(f"Val Paired: {args.val_paired}\n")
        f.write(f"Model Architecture: SiameseUNetBaseline\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Output Path: {out_path}\n")
        f.write(f"Arguments: {vars(args)}\n")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_grad_norm', 'train_update_ratio', 'val_loss', 'test_loss', 'test_dice', 'test_dice_std', 'test_hd95', 'test_hd95_std', 'test_time_sec', 'test_reg_time_sec', 'test_jac', 'test_jac_std', 'test_mag', 'test_dice_per_label', 'test_dice_per_label_std'])

    best_loss = float('inf')
    best_test_dice = 0.0
    # Threshold for "acceptable" folding (NegJac ratio). 
    # If NegJac > 0.01 (1%), we consider the deformation unrealistic despite high Dice.
    NEG_JAC_THRESHOLD = 0.01 
    
    loss_history: List[float] = []
    epoch_times = []

    # Persistent scaler across all epochs (important for stable AMP training)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # 记录训练前的初始 GPU 显存
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    print(f'Training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        avg_loss, last_img_loss, last_grad_loss, avg_grad_norm, update_ratio = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=len(train_loader),
            scaler=scaler,
            device=device,
        )

        # -----------------------------
        # Validation Phase
        # -----------------------------
        # Default values if val_loader is None
        val_loss, val_dice, val_hd95, val_time, val_jac, val_mag = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        current_monitor_loss = avg_loss

        if val_loader:
            # Optimization: Skip expensive metrics (HD95, Jacobian) 
            # We now skip Dice and HD95 and Mag for validation, only computing loss.
            val_loss, val_dice, val_hd95, val_time, val_jac, val_mag = validate(
                model=model,
                dataloader=val_loader,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                device=device,
                fast=True
            )
            
            print(f'Epoch {epoch + 1} | TrainTotal: {avg_loss:.4f} (Img: {last_img_loss:.4f}, Grad: {last_grad_loss:.6f}) | GradNorm: {avg_grad_norm:.6f}, UpdateRatio: {update_ratio:.2%}')
            metric_suffix = " (Fast Val, Loss Only)"
            print(f'         | Val Loss: {val_loss:.4f}{metric_suffix}')
            current_monitor_loss = val_loss
        else:
            print(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}, GradNorm: {avg_grad_norm:.6f}, UpdateRatio: {update_ratio:.2%}')

        if update_ratio < 0.1:
            print(f'  [Warning] Low effective update ratio ({update_ratio:.2%}). Model parameters may not be updating properly.')
        
        # -----------------------------
        # Test Phase (Detailed Evaluation)
        # -----------------------------
        test_loss, test_dice, test_hd95, test_time, test_reg_time, test_jac, test_mag = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_best_idx = None
        test_dice_per_label = {}
        test_raw_results = []
        is_best_test_dice = False
        
        # Optimization: Always run test evaluation to get Dice and HD95 for CSV, skip slow metrics (Jac) unless run_full_metrics
        run_full_metrics = ((epoch + 1) == 1) or ((epoch + 1) % 5 == 0)
        run_test_eval = (test_loader is not None)

        if run_test_eval:
             # Evaluate on the test dataset
             fast_eval = not run_full_metrics
             test_loss, test_dice, test_dice_std, test_hd95, test_hd95_std, test_time, test_reg_time, test_jac, test_jac_std, test_mag, test_best_idx, test_dice_per_label, test_dice_per_label_std, test_raw_results = test_evaluate(
                model=model,
                dataloader=test_loader,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                device=device,
                fast=fast_eval # HD95 only calculated if fast=False
             )
             
             if test_dice > best_test_dice:
                 best_test_dice = test_dice
                 is_best_test_dice = True
                 print(f'  [Monitor] * New best Test Dice: {best_test_dice:.6f} *')
                 if fast_eval:
                     pass # 关掉新高test_dice时计算hd95和jac，最大程度加快进度

             test_label_metrics_str = ""
             # 关掉明细dice打印
             # if test_dice_per_label:
             #    test_label_metrics_str = " | LabelDice: " + ", ".join([f"{k}:{v:.3f}±{test_dice_per_label_std[k]:.3f}" for k, v in test_dice_per_label.items()])

             metric_suffix = "" if run_full_metrics else (" (Fast Test -> Full)" if is_best_test_dice else " (Fast Test)")
             print(f'  [Monitor] Test Dice: {test_dice:.6f}±{test_dice_std:.6f}, HD95: {test_hd95:.6f}±{test_hd95_std:.6f}, Loss: {test_loss:.6f}, Jac: {test_jac:.6f}±{test_jac_std:.6f}, Time: {test_time:.4f}s{test_label_metrics_str}{metric_suffix}')
             
             if is_best_test_dice:
                 # Save best pt model based on test dice
                 best_model_path = out_path.parent / f'{out_path.stem}_best_test.pt'
                 torch.save(model.state_dict(), best_model_path)
                 print(f'  [Monitor] Saved new best model to {best_model_path.name}')

             # Save detailed per-sample results for Boxplot ONLY when it's the best test dice
             # if is_best_test_dice or run_full_metrics:
             #     detailed_log_file = out_path.parent / f'test_results_detailed_epoch{epoch+1}.csv'
             #     if test_raw_results:
             #         all_label_keys = set()
             #         for res in test_raw_results:
             #             all_label_keys.update(res['label_dice'].keys())
             #         sorted_keys = sorted(list(all_label_keys))
             #         
             #         header = ['sample_idx', 'filename', 'dice', 'hd95', 'jac'] + [f'dice_label_{k}' for k in sorted_keys]
             #         
             #         with open(detailed_log_file, 'w', newline='') as f_detail:
             #             w_detail = csv.DictWriter(f_detail, fieldnames=header)
             #             w_detail.writeheader()
             #             for res in test_raw_results:
             #                 row_data = {
             #                     'sample_idx': res['sample_idx'],
             #                     'filename': res.get('filename', ''),
             #                     'dice': f"{res['dice']:.5f}",
             #                     'hd95': f"{res.get('hd95', 0):.5f}" if 'hd95' in res and not np.isnan(res.get('hd95', np.nan)) else '',
             #                     'jac': f"{res['jac']:.5f}" if 'jac' in res else ''
             #                 }
             #                 for k in sorted_keys:
             #                     val = res['label_dice'].get(k, '')
             #                     if isinstance(val, (float, np.floating)):
             #                         row_data[f'dice_label_{k}'] = f"{val:.5f}"
             #                     else:
             #                         row_data[f'dice_label_{k}'] = val
             #                 w_detail.writerow(row_data)
        

        # -----------------------------
        # Logging
        # -----------------------------
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            def fmt(x):
                # Helper to format floats
                return f"{x:.5f}" if isinstance(x, (float, np.float32, np.float64)) else x
            
            # Prepare dice per label string for CSV (semicolon separated to avoid csv conflict)
            test_dice_per_label_str = ""
            test_dice_per_label_std_str = ""
            if test_dice_per_label:
                 # Manually format dictionary to avoid 'np.float64(...)' in output
                 items = [f"{k}: {v:.5f}" for k, v in test_dice_per_label.items()]
                 test_dice_per_label_str = "{" + "; ".join(items) + "}"
                 
                 items_std = [f"{k}: {v:.5f}" for k, v in test_dice_per_label_std.items()]
                 test_dice_per_label_std_str = "{" + "; ".join(items_std) + "}"

            row = [
                epoch + 1, 
                fmt(avg_loss), 
                fmt(avg_grad_norm),
                fmt(update_ratio),
                fmt(val_loss), 
                fmt(test_loss), 
                fmt(test_dice), 
                fmt(test_dice_std),
                fmt(test_hd95), 
                fmt(test_hd95_std),
                fmt(test_time), 
                fmt(test_reg_time), 
                fmt(test_jac), 
                fmt(test_jac_std),
                fmt(test_mag), 
                test_dice_per_label_str,
                test_dice_per_label_std_str
            ]
            writer.writerow(row)

        loss_history.append(current_monitor_loss)
        
        # -----------------------------
        # Visualization
        # -----------------------------
        do_visualization = False # 快速验证超参期间关闭可视化 (原为: is_best_test_dice or run_full_metrics)
        
        if do_visualization:
            vis_dataset = None
            vis_suffix = ''
            best_idx_to_plot = None
            
            # Prefer test set for visualization if available
            if test_loader and hasattr(test_loader.dataset, '__getitem__'):
                vis_dataset = test_loader.dataset
                vis_suffix = '_test'
                best_idx_to_plot = test_best_idx
            elif val_loader and hasattr(val_loader.dataset, '__getitem__'):
                vis_dataset = val_loader.dataset
                vis_suffix = '_val'
                
            if vis_dataset:
                 save_qualitative_results(
                     model, 
                     vis_dataset, 
                     out_path.parent, 
                     epoch + 1, 
                     device=device, 
                     suffix=vis_suffix,
                     best_sample_idx=best_idx_to_plot
                 )

        # -----------------------------
        # Model Saving & Stopping
        # -----------------------------
        # Early stopping
        if check_early_stopping(
            loss_history,
            patience=args.patience,
            threshold=args.threshold,
            warm_start_steps=args.warm_start
        ):
            print(f'Early stopping at epoch {epoch + 1}')
            break
            
        # Periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt = out_path.parent / f'{out_path.stem}_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), ckpt)
            print(f'Checkpoint saved to {ckpt}')

        # Best Val Loss track for early stopping
        if current_monitor_loss < best_loss:
            best_loss = current_monitor_loss
            print(f'New best val loss: {best_loss:.6f}.')

        # Scheduler step based on Test Dice (maximize)
        scheduler.step(test_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0
        
        print(f"Current LR: {current_lr:.6f} | Epoch Time: {epoch_time:.2f}s | Peak GPU Mem: {peak_gpu_mem:.2f} MB")


    # Remove the final epoch arbitrary saving except the very last one
    torch.save(model.state_dict(), out_path.parent / f'{out_path.stem}_final.pt')
    print(f'Final model saved to {out_path.parent / f"{out_path.stem}_final.pt"}')
    
    # Save end time to config
    finish_utc = datetime.datetime.utcnow()
    finish_beijing = finish_utc + datetime.timedelta(hours=8)
    finish_timestamp = finish_beijing.strftime('%Y%m%d_%H%M%S')
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    final_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 0.0

    with open(config_file, 'a') as f:
        f.write(f"End Time: {finish_timestamp}\n")
        f.write(f"Average Epoch Time: {avg_epoch_time:.2f} s\n")
        f.write(f"Peak GPU Memory: {final_peak_mem:.2f} MB\n")

    # Plot history
    try:
        plot_history(log_file, out_path.parent)
        print(f'Training curves saved to {out_path.parent / "training_curves.png"}')
    except Exception as e:
        print(f'Failed to plot training history: {e}')


if __name__ == '__main__':
    main()
