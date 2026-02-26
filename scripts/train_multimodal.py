#!/usr/bin/env python3
"""
Train VoxelMorph for multimodal registration (source: CT, target: MR).

This script assumes preprocessed volumes live in `processed/ct/image` and
`processed/mr/image` with matching filename prefixes like
`AbdomenMRCT_0001_0001.nii.gz` (CT) and `AbdomenMRCT_0001_0000.nii.gz` (MR).
"""
import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Sequence, List

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
        self.device = device
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

        # Robust Min-Max Normalization to force background to 0
        # This fixes "Background Warping" by ensuring empty space is actually 0.
        def normalize(vol):
            v_min = vol.min()
            v_max = vol.max()
            if v_max - v_min > 1e-6:
                return (vol - v_min) / (v_max - v_min)
            return vol

        ct_np = normalize(ct_np)
        mr_np = normalize(mr_np)

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
        
        # If labels are available, try to find a slice with most labels (1,2,3,4)
        lbl_for_search = None
        if target_label is not None:
            lbl_for_search = target_label
        elif source_label is not None:
            lbl_for_search = source_label
            
        if lbl_for_search is not None:
            # OPTIMIZATION: Computer unique labels per slice on GPU
            # lbl_for_search: (1, 1, X, Y, Z)
            # We want to count non-zero unique elements along Z
            
            # Simple approach on GPU: Iterate Z slices, but keep data on GPU
            best_idx = slice_idx
            max_unique_labels = -1
            
            D = lbl_for_search.shape[4] # Z dimension
            
            # Start search from middle outwards
            search_indices = sorted(range(D), key=lambda i: abs(i - slice_idx))
            
            for z in search_indices:
                # Get slice on GPU
                slice_data = lbl_for_search[0, 0, :, :, z]
                
                # Check unique elements. torch.unique is supported on GPU
                unique = torch.unique(slice_data)
                
                # Count non-zero labels (> 0.5 to be safe with float)
                count = (unique > 0.5).sum().item()
                
                if count > max_unique_labels:
                    max_unique_labels = count
                    best_idx = z
                    if max_unique_labels >= 4: # Found all 4
                        break
            
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
        
        rows = 3 if has_labels else 2
        cols = 4
        # Use simple subplots. To ensure uniform image size, we must be careful with colorbars.
        # We will use ImageGrid or just handle colorbars carefully.
        # But for simplicity, let's stick to subplots and make the figure larger.
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        
        # Turn off all axes initially
        for ax in axes.flatten():
            ax.axis('off')
        from matplotlib.colors import ListedColormap
        # Create a custom colormap for up to 4 labels + background
        # Background should be transparent (alpha=0)
        # Colors: [Red, Green, Blue, Yellow]
        base_colors = np.array([
            [1, 0, 0, 0.5],   # Label 1: Red
            [0, 1, 0, 0.5],   # Label 2: Green
            [0, 0, 1, 0.5],   # Label 3: Blue
            [1, 1, 0, 0.5],   # Label 4: Yellow
            [0, 1, 1, 0.5],   # Label 5: Cyan
            [1, 0, 1, 0.5],   # Label 6: Magenta
        ])
        
        def plot_multi_label(ax, bg_img, label_img, title):
            ax.imshow(bg_img, cmap='gray')
            if label_img is not None:
                 # Mask out background (0)
                 masked_lbl = np.ma.masked_where(label_img < 0.5, label_img)
                 # We want discrete colors for integer labels 1, 2, 3, 4
                 # Normalize max label to fit color map indices?
                 # Simple approach: float overlay
                 ax.imshow(masked_lbl, cmap='tab10', alpha=0.5, interpolation='nearest', vmin=1, vmax=10)
            ax.set_title(title)
            ax.axis('off')

        # Row 1: Basic Images
        # 1. Moving
        axes[0, 0].imshow(src_slice, cmap='gray')
        axes[0, 0].set_title('Moving')
        axes[0, 0].axis('off')
        
        # 2. Fixed
        axes[0, 1].imshow(tgt_slice, cmap='gray')
        axes[0, 1].set_title('Fixed')
        axes[0, 1].axis('off')
        
        # 3. Warp
        axes[0, 2].imshow(warped_slice, cmap='gray')
        axes[0, 2].set_title('Deformed')
        axes[0, 2].axis('off')

        # 4. Moving - Fixed
        diff_moving_fixed = src_slice - tgt_slice
        im_diff1 = axes[0, 3].imshow(diff_moving_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 3].set_title('Source - Target')
        axes[0, 3].axis('off')
        fig.colorbar(im_diff1, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # Row 2: Advanced Analysis
        
        # 5. Deformed Grid
        # Get dimensions from slice
        H, W = src_slice.shape
        # Create a regular grid
        grid_spacing = 10
        xx, yy = np.meshgrid(np.arange(0, W, grid_spacing), np.arange(0, H, grid_spacing))
        # Get displacement for this slice (need to correspond to Meshgrid usage)
        # displacement shape is (1, 3, Dim0, Dim1, Dim2). We need 2D vectors on the last dimension slice.
        # Spatial dims are indices 2, 3, 4. We want to slice index 4.
        
        # d_slice should be taken from the same slice index we determined above
        raw_d_slice = displacement.detach().cpu().numpy()[0, :, :, :, slice_idx] # (3, Dim0, Dim1)
        
        # Rotate spatial dimensions to align with image visualization (np.rot90 on axes 0,1 of image)
        # raw_d_slice has channels at axis 0. So we rotate axes 1 and 2.
        d_slice = np.rot90(raw_d_slice, axes=(1, 2))
        
        # Note on orientation: Matplotlib imshow origin is 'upper'. 
        # Grid: xx is col index (X), yy is row index (Y).
        # d_slice[2] is X-displacement, d_slice[1] is Y-displacement.
        
        # Warped grid positions
        # We sample displacement at grid points
        # Simple nearest sampling for visualization
        # Note: d_slice is (3, H, W). 
        dx_grid = d_slice[2, ::grid_spacing, ::grid_spacing]
        dy_grid = d_slice[1, ::grid_spacing, ::grid_spacing]
        
        # Add displacement to original grid
        # Note: VoxelMorph fields are usually 'pull' fields (map target to source)
        # To visualize 'push' (source to target) geometry simply, we can just subtract flow 
        # or just plot the grid lines modified by flow.
        # Standard 'Deformed Grid' usually shows how a regular grid on Target maps back to Source.
        
        axes[1, 0].imshow(np.zeros_like(src_slice), cmap='gray', vmin=0, vmax=1) # Black background
        # Plot vertical lines (using dense sampling for smooth curves)
        for i in range(0, W, grid_spacing):
            # x is constant i, y varies 0..H
            # d_slice[2, :, i] is X-displacement along the vertical line at x=i
            # d_slice[1, :, i] is Y-displacement along the vertical line at x=i
            # Check bounds
            if i < d_slice.shape[2]:
                x_plot = i + d_slice[2, :, i]
                y_plot = np.arange(H) + d_slice[1, :, i]
                axes[1, 0].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9) # White lines
            
        # Plot horizontal lines (using dense sampling)
        for j in range(0, H, grid_spacing):
            # y is constant j, x varies 0..W
            # d_slice[2, j, :] is X-displacement along the horizontal line at y=j
            # d_slice[1, j, :] is Y-displacement along the horizontal line at y=j
            if j < d_slice.shape[1]:
                x_plot = np.arange(W) + d_slice[2, j, :]
                y_plot = j + d_slice[1, j, :]
                axes[1, 0].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9) # White lines
            
        axes[1, 0].set_title('Deformed Grid')
        axes[1, 0].set_ylim(H, 0) # Flip Y to match image coordinates
        axes[1, 0].set_xlim(0, W)
        axes[1, 0].axis('off')

        # Create HSV image
        # Standard flow visualization:
        # Hue = direction (0..1)
        # Saturation = magnitude (0..1)
        # Value = 1.0 (constant bright)
        
        from matplotlib.colors import hsv_to_rgb
        
        # Calculate magnitude and angle
        # d_slice indices: 0=Z, 1=Y, 2=X
        dx = d_slice[2]
        dy = d_slice[1]
        
        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Normalize magnitude for saturation
        # Clip outliers to visualize variations better
        p99 = np.percentile(mag, 99) + 1e-5
        mag_norm = np.clip(mag / p99, 0, 1)

        hsv = np.zeros((H, W, 3), dtype=np.float32)

        # --- Plot 6: RGB Displacement Field with 3D Axis Legend ---
        # Normalize flow for visualization: Map X->Red, Y->Green, Z->Blue
        # d_slice indices: 0=Z, 1=Y, 2=X
        dx = d_slice[2]
        dy = d_slice[1]
        dz = d_slice[0]
        
        # Calculate max magnitude for normalization
        max_mag = np.max(np.abs(d_slice)) + 1e-5
        
        # Create RGB image
        # Center 0 displacement at 0.5 (Gray) to show positive/negative direction
        # Mapping: [-max, max] -> [0, 1]
        flow_vis = np.zeros((H, W, 3), dtype=np.float32)
        flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5 # X -> R
        flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5 # Y -> G
        flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5 # Z -> B
        
        # Clip to ensure valid range
        flow_vis = np.clip(flow_vis, 0, 1)
        
        # Display the flow
        axes[1, 1].imshow(flow_vis)
        axes[1, 1].set_title('RGB Displacement', color='black') # Default color
        axes[1, 1].axis('off')

        # Add legend if this is a test set image
        if 'test' in suffix:
            # We now use a dedicated subplot for the legend (axes[1, 2])
            pass

        # 7. Warp - Fixed
        diff_warp_fixed = warped_slice - tgt_slice
        # Move to position [1, 3] (was [1, 2])
        im_diff2 = axes[1, 3].imshow(diff_warp_fixed, cmap='bwr', vmin=-1, vmax=1)
        axes[1, 3].set_title('Deformed - Target')
        axes[1, 3].axis('off')
        # Create colorbar axis adjacent to image or just inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(axes[1, 3], width="5%", height="100%", loc='center right', borderpad=-1.5)
        fig.colorbar(im_diff2, cax=cax)

        # 8. RGB Flow Legend (Using the empty spot at [1, 2] since Edges removed)
        # We removed "Edges" ([1, 3] originally), moved Diff2 to [1, 3].
        
        ax_legend_spot = axes[1, 2]
        ax_legend_spot.clear() # Clear any previous content
        ax_legend_spot.axis('off')
        ax_legend_spot.set_xlim(0, 1)
        ax_legend_spot.set_ylim(0, 1)
        
        # Draw Arrows at center of this subplot
        # Y axis (Green) - pointing Down visually to match image coordinates
        ax_legend_spot.arrow(0.5, 0.7, 0, -0.4, head_width=0.05, head_length=0.05, fc='lime', ec='lime', width=0.01)
        ax_legend_spot.text(0.5, 0.2, 'Y', color='lime', ha='center', va='top', fontweight='bold', fontsize=12)

        # X axis (Red) - pointing Right
        ax_legend_spot.arrow(0.5, 0.7, 0.4, 0, head_width=0.05, head_length=0.05, fc='red', ec='red', width=0.01)
        ax_legend_spot.text(0.95, 0.7, 'X', color='red', ha='left', va='center', fontweight='bold', fontsize=12)
        
        # Z axis (Blue) - Dot
        ax_legend_spot.plot(0.5, 0.7, 'o', color='blue', markersize=15)
        ax_legend_spot.text(0.45, 0.75, 'Z', color='blue', ha='right', va='bottom', fontweight='bold', fontsize=12)
        
        # Add Range Text
        range_text = f"Displacement\nRange:\n[-{max_mag:.1f}, {max_mag:.1f}]"
        ax_legend_spot.text(0.5, 0.9, range_text, ha='center', va='center', fontsize=12, fontweight='bold', color='black')

        # Row 3: Label Analysis
        if has_labels:
            # Color definitions
            # 5 distinct hues.
            # Base colors for fixed (GT) labels - Deep/Dark saturated colors for clear visibility on grey
            # 1: Liver, 2: Spleen, 3: R-Kidney, 4: L-Kidney
            # Deep/Dark saturated colors for clear visibility on grey
            base_colors = {
                1: '#8B0000', # Dark Red (Liver)
                2: '#006400', # Dark Green (Spleen)
                3: '#4682B4', # Steel Blue (R-Kidney) - Lighter/Darker Blue mix, easier to see than Dark Blue
                4: '#B8860B', # Dark Goldenrod (L-Kidney) - Darker but related to yellow/gold
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
            
            def plot_label_contour(ax, bg_img, label_img, title, color_lookup):
                ax.imshow(bg_img, cmap='gray')
                if label_img is not None:
                     unique_labels = np.unique(label_img)
                     unique_labels = unique_labels[unique_labels > 0]
                     for lbl in unique_labels:
                         mask = (label_img == lbl)
                         c = color_lookup.get(int(lbl), 'white')
                         if np.any(mask):
                             ax.contour(mask, colors=[c], linewidths=1.2)
                ax.set_title(title)
                ax.axis('off')

            # 9. Source + Label Overlay (Contour, Bright)
            plot_label_contour(axes[2, 0], src_slice, src_lbl_slice, 'Source + Labels', bright_colors)
            
            # 10. Target + Label Overlay (Contour, Base)
            plot_label_contour(axes[2, 1], tgt_slice, tgt_lbl_slice, 'Target + Labels', base_colors)
            
            # 11. Warped Source + Warped Label (Contour, Bright - same as Source)
            plot_label_contour(axes[2, 2], warped_slice, warped_lbl_slice, 'Deformed + Deformed Labels', bright_colors)
            
            # 12. Overlay Warped Label on Target Image
            # Logic: Show Warped Image (Result) in Gray. 
            # Show Target Labels (GT) as contours (Base Colors, Dashed).
            # Show Warped Labels (Pred) as contours (Bright Colors, Solid).
            axes[2, 3].imshow(warped_slice, cmap='gray')
            
            # Ground Truth Contours
            if tgt_lbl_slice is not None:
                 unique_labels = np.unique(tgt_lbl_slice)
                 unique_labels = unique_labels[unique_labels > 0]
                 for lbl in unique_labels:
                     mask = (tgt_lbl_slice == lbl)
                     c = base_colors.get(int(lbl), 'white')
                     if np.any(mask):
                         # Dashed line for GT (Target)
                         # Increase dash spacing via linestyles tuple (offset, (on_off_seq))
                         # 'dashed' is roughly (0, (5, 5)). Let's make it distinct.
                         axes[2, 3].contour(mask, colors=[c], linewidths=1.5, linestyles='dashed')
            
            # Prediction Contours
            if warped_lbl_slice is not None:
                 unique_labels = np.unique(warped_lbl_slice)
                 unique_labels = unique_labels[unique_labels > 0]
                 for lbl in unique_labels:
                     mask = (warped_lbl_slice == lbl)
                     c = bright_colors.get(int(lbl), 'white')
                     if np.any(mask):
                         axes[2, 3].contour(mask, colors=[c], linewidths=1.5, linestyles='solid')

            axes[2, 3].set_title('Deformed Image + Target(Dash) & Deformed(Solid)')
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
            val_dice.append(float(row['val_dice']))
            val_jac.append(float(row['val_neg_jac_ratio']))
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
        
        # Robust Min-Max Normalization to force background to 0
        def normalize(vol):
            v_min = vol.min()
            v_max = vol.max()
            if v_max - v_min > 1e-6:
                return (vol - v_min) / (v_max - v_min)
            return vol

        ct_np = normalize(ct_np)
        mr_np = normalize(mr_np)

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
    device: str = 'cuda'
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 0

    # Iterate through the entire dataloader (all 1500+ pairs if unpaired)
    # The dataloader is now a finite Map-style Dataset, not an infinite Iterable
    for batch in tqdm(dataloader, desc="Training Batch"):
        optimizer.zero_grad()

        source = batch['source'].to(device)
        target = batch['target'].to(device)

        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement'
        )

        if isinstance(image_loss_fn, ne.nn.modules.NCC):
            img_loss = -image_loss_fn(target, warped_source)
        else:
            img_loss = image_loss_fn(target, warped_source)
        
        grad_loss = grad_loss_fn(displacement)

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Track components for debugging
        # Note: We need to detach to avoid accumulation
        num_steps += 1
        
    # Return breakdown (approximated from last batch or accumulate if needed, 
    # but strictly we just need to see the scale)
    return total_loss / num_steps, img_loss.item(), grad_loss.item()


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    device: str = 'cuda'
):
    """Run validation on a fixed dataset. Returns avg_loss, avg_dice, avg_time, avg_neg_jac.
       Also returns the index of the sample with the best Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_time = 0.0
    total_neg_jac = 0.0
    total_mag = 0.0  # magnitude of deformation
    num_batches = 0
    num_dice_batches = 0
    num_jac_batches = 0
    
    best_dice = -1.0
    best_dice_idx = None
    # Keep track of global index
    current_idx_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            # Measure inference time
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
            
            # Calculate mean deformation magnitude
            # displacement shape: (B, C, D, H, W)
            disp_mag = torch.sqrt(torch.sum(displacement ** 2, dim=1))
            total_mag += disp_mag.mean().item()

            # Time per batch
            batch_time = end_time - start_time
            # Normalize by batch size to get sec/volume (optional, but requested per image usually)
            # Keeping per batch for now, will divide by total volumes later if needed, 
            # but standard is usually seconds per inference call. 
            # User asks for "GPU sec", usually means per volume time.
            total_time += (batch_time / source.shape[0])
            
            if isinstance(image_loss_fn, ne.nn.modules.NCC):
                img_loss = -image_loss_fn(target, warped_source)
            else:
                img_loss = image_loss_fn(target, warped_source)
            
            grad_loss = grad_loss_fn(displacement)

            loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
            total_loss += loss.item()
            num_batches += 1

            # Jacobian Determinant
            # Displacement is (B, C, D, H, W) -> need (D, H, W, C) for utils
            disp_np = displacement.detach().cpu().numpy()
            disp_np = np.transpose(disp_np, (0, 2, 3, 4, 1))
            
            batch_neg_jac = 0.0
            for i in range(disp_np.shape[0]):
                jac_det = vxm.py.utils.jacobian_determinant(disp_np[i])
                # Count non-positive
                batch_neg_jac += np.sum(jac_det <= 0) / np.prod(jac_det.shape)
            
            total_neg_jac += (batch_neg_jac / disp_np.shape[0])
            num_jac_batches += 1

            # Compute Dice if labels validation is available
            if 'source_label' in batch and 'target_label' in batch:
                source_label = batch['source_label'].to(device)
                target_label = batch['target_label'].to(device)
                
                # Warping labels using Nearest Neighbor interpolation
                # SpatialTransformer in this version does not take size in __init__, but behaves dynamically.
                # It takes interpolation_mode as 'nearest' for labels.
                trf = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest').to(device)
                warped_label = trf(source_label, displacement)
                
                # Compute Dice for each label and average
                # Assuming labels are integers. If 0 is background, we might want to skip it.
                # Here we use a simple dice implementation
                dice_score = vxm.py.utils.dice(warped_label.cpu().numpy(), target_label.cpu().numpy())
                mean_dice = dice_score.mean()
                total_dice += mean_dice
                
                # Update Best Dice tracking
                # Note: dice_score is an array of dices per label, we take mean.
                # If batch size > 1, we strictly need to check each sample.
                # For simplicity here, if batch size > 1, we take the batch mean dice as the metric for the batch.
                # But to return a specific index, let's assume batch size = 1 or we just take the first one of the batch.
                # The 'idx' in dataset is needed.
                # Let's approximate: if this batch's mean dice is the best, we mark the index of the first item in this batch.
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_dice_idx = current_idx_offset # Point to the first element of this batch
                
                num_dice_batches += 1
            
            current_idx_offset += source.shape[0]

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_dice_batches if num_dice_batches > 0 else 0.0
    avg_time = total_time / num_batches if num_batches > 0 else 0.0
    avg_neg_jac = total_neg_jac / num_jac_batches if num_jac_batches > 0 else 0.0
    avg_mag = total_mag / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice, avg_time, avg_neg_jac, avg_mag, best_dice_idx


def main():
    parser = argparse.ArgumentParser(description='Train multimodal CT->MR VoxelMorph')
    parser.add_argument('--ct-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/train/images/ct', help='CT train images (source)')
    parser.add_argument('--mr-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/train/images/mr', help='MR train images (target)')
    parser.add_argument('--paired-ct-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/trainPairs/images/ct', help='Paired CT train images (source)')
    parser.add_argument('--paired-mr-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/trainPairs/images/mr', help='Paired MR train images (target)')
    parser.add_argument('--ct-val-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/val/images/ct', help='Validation CT images')
    parser.add_argument('--mr-val-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/val/images/mr', help='Validation MR images')
    parser.add_argument('--ct-val-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/val/labels/ct', help='Validation CT labels')
    parser.add_argument('--mr-val-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/val/labels/mr', help='Validation MR labels')
    parser.add_argument('--ct-test-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/test/images/ct', help='Test CT images (for monitoring)')
    parser.add_argument('--mr-test-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/test/images/mr', help='Test MR images (for monitoring)')
    parser.add_argument('--ct-test-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/test/labels/ct', help='Test CT labels')
    parser.add_argument('--mr-test-label-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm/test/labels/mr', help='Test MR labels')
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
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Model: pairwise VoxelMorph
    # model = vxm.nn.models.VxmPairwise(
    #     ndim=3,
    #     source_channels=1,
    #     target_channels=1,
    #     nb_features=[16, 16, 16, 16, 16],
    #     integration_steps=0,
    # ).to(device)
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        # Symmetric 5-level UNet (Encoder: [16, 32, 32, 32, 32] -> Decoder auto-reversed)
        # This provides standard VoxelMorph capacity (5 downsamples) without crashing on image size.
        nb_features=[16, 32, 32, 32, 32], 
        integration_steps=7,
    ).to(device)

    # Losses
    if args.image_loss == 'mse':
        image_loss_fn = ne.nn.modules.MSE().to(device)
    elif args.image_loss == 'mi':
        image_loss_fn = vxm.nn.losses.MutualInformation(num_bin=args.mi_bins).to(device)
    elif args.image_loss == 'local_mi':
        image_loss_fn = vxm.nn.losses.localMutualInformation(patch_size=args.patch_size, num_bin=args.mi_bins).to(device)
    else:
        # neurite NCC expects window size; default None will choose automatic
        image_loss_fn = ne.nn.modules.NCC(window_size=args.ncc_win).to(device)

    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dataloader
    # Switch to Map-style Dataset for consistent epoch definition
    dataset = MultimodalTrainDataset(args.ct_dir, args.mr_dir, paired_ct_dir=args.paired_ct_dir, paired_mr_dir=args.paired_mr_dir, device=device, unpaired=args.unpaired, max_samples=args.max_train_samples)
    # Shuffle is KEY here: it mixes Easy and Hard pairs within each epoch
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # Important: Shuffle every epoch
        num_workers=args.workers
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
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
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
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
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
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Image Loss: {args.image_loss}\n")
        f.write(f"NCC Window: {args.ncc_win}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Unpaired: {args.unpaired}\n")
        f.write(f"Val Paired: {args.val_paired}\n")
        f.write(f"Model Architecture: VxmPairwise\n")
        f.write(f"Output Path: {out_path}\n")
        f.write(f"Arguments: {vars(args)}\n")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_time_sec', 'val_neg_jac_ratio', 'val_mag', 'test_loss', 'test_dice', 'test_jac', 'test_mag'])

    best_loss = float('inf')
    best_dice = 0.0
    # Threshold for "acceptable" folding (NegJac ratio). 
    # If NegJac > 0.01 (1%), we consider the deformation unrealistic despite high Dice.
    NEG_JAC_THRESHOLD = 0.01 
    
    loss_history: List[float] = []

    print(f'Training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        avg_loss, last_img_loss, last_grad_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=len(train_loader), # Now unused inside function, but passed for API
            device=device,
        )

        # Validation
        current_monitor_loss = avg_loss
        val_loss_scalar = 0.0
        val_dice_scalar = 0.0
        val_time_scalar = 0.0
        val_jac_scalar = 0.0
        val_mag_scalar = 0.0
        
        if val_loader:
            val_loss, val_dice, val_time, val_jac, val_mag, _ = validate(
                model=model,
                dataloader=val_loader,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                device=device
            )
            # Enhanced printing
            print(f'Epoch {epoch + 1} | TrainTotal: {avg_loss:.4f} (Img: {last_img_loss:.4f}, Grad: {last_grad_loss:.6f})')
            print(f'         | Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Mag: {val_mag:.2f}, NegJac: {val_jac:.4f}')
            
            current_monitor_loss = val_loss
            val_loss_scalar = val_loss
            val_dice_scalar = val_dice
            val_time_scalar = val_time
            val_jac_scalar = val_jac
            val_mag_scalar = val_mag
        else:
            print(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}')
        
        # Test / Monitor Evaluation
        test_loss_scalar = 0.0
        test_dice_scalar = 0.0
        test_jac_scalar = 0.0
        test_mag_scalar = 0.0
        
        test_best_idx = None

        if test_loader:
             test_loss, test_dice, _, test_jac, test_mag, test_best_idx = validate(
                model=model,
                dataloader=test_loader,
                image_loss_fn=image_loss_fn,
                grad_loss_fn=grad_loss_fn,
                loss_weights=loss_weights,
                device=device
            )
             print(f'  [Monitor] Test Dice: {test_dice:.6f}, Test Loss: {test_loss:.6f}, Test Jac: {test_jac:.6f}, Test Mag: {test_mag:.4f}')
             test_loss_scalar = test_loss
             test_dice_scalar = test_dice
             test_jac_scalar = test_jac
             test_mag_scalar = test_mag

             # If we have a test set, use it for "Best Dice" tracking instead of the (possibly unpaired) validation set
             # This overrides the decision logic below for 'best_dice'
             # BUT we keep 'current_monitor_loss' (used for best_loss) as val_loss if available, or train_loss, 
             # because we usually want 'best_loss' to be about the optimization objective.
             
        
        # Log to file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, val_loss_scalar, val_dice_scalar, val_time_scalar, val_jac_scalar, val_mag_scalar, test_loss_scalar, test_dice_scalar, test_jac_scalar, test_mag_scalar])

        loss_history.append(current_monitor_loss)
        
        # Save visualization (updated logic: every 5 epochs OR always for epoch 1)
        # We use test_loader primarily if available, else val_loader
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            vis_dataset = None
            vis_suffix = ''
            best_idx_to_plot = None
            
            if test_loader and hasattr(test_loader.dataset, '__getitem__'):
                vis_dataset = test_loader.dataset
                vis_suffix = '_test'
                best_idx_to_plot = test_best_idx
            elif val_loader and hasattr(val_loader.dataset, '__getitem__'):
                vis_dataset = val_loader.dataset
                vis_suffix = '_val'
                # val metric doesn't return best idx currently unless we changed calling convention above too, 
                # but we only updated validate sig. In call above used `_` for val.
                # If you want best val sample too, need to capture it.
                # For now, let's just use test best idx if available.
                
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

        # Early stopping check
        if check_early_stopping(
            loss_history,
            patience=args.patience,
            threshold=args.threshold,
            warm_start_steps=args.warm_start
        ):
            print(f'Early stopping at epoch {epoch + 1}')
            break
            
        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt = out_path.parent / f'{out_path.stem}_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), ckpt)
            print(f'Checkpoint saved to {ckpt}')

        # Strategy: "Best Loss" (Unsupervised Model Selection)
        # Always track the model with lowest validation loss as the main model.
        if current_monitor_loss < best_loss:
            best_loss = current_monitor_loss
            # Save as the main output model
            torch.save(model.state_dict(), out_path)
            print(f'New best val loss: {best_loss:.6f}. Saved to {out_path.name}')


    torch.save(model.state_dict(), out_path)
    print(f'Final model saved to {out_path}')
    
    # Plot history
    try:
        plot_history(log_file, out_path.parent)
        print(f'Training curves saved to {out_path.parent / "training_curves.png"}')
    except Exception as e:
        print(f'Failed to plot training history: {e}')


if __name__ == '__main__':
    main()
