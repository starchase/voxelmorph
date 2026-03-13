#!/usr/bin/env python3

import os
import sys
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
            
        # --- Row 1: Images & Differences ---
        
        # [0,0] Source Image
        axes[0, 0].imshow(src_slice, cmap='gray')
        axes[0, 0].set_title('Source Image')
        axes[0, 0].axis('off')

        # [0,1] Target Image
        axes[0, 1].imshow(tgt_slice, cmap='gray')
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
                ax.imshow(bg_img, cmap='gray')
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
        axes[1, 3].imshow(warped_slice, cmap='gray')
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

def validate(

    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> float:
    model.eval()
    eval_dsc = utils.AverageMeter()

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

            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            eval_dsc.update(dsc.item(), x.size(0))

    return eval_dsc.avg

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: list,
    device: str = 'cuda'
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        # TransMorph OASISDataset returns: x, y, x_seg, y_seg
        # x: moving image, y: fixed image
        x = data[0].to(device)
        y = data[1].to(device)

        # Get the displacement and the warped source image from the model
        out = model(
            x,
            y,
            return_warped_source=True,
            return_field_type='displacement'
        )
        displacement, warped_source = out[0], out[1]

        img_loss = image_loss_fn(y, warped_source).mean()
        
        # If the loss function is NCC (which computes similarity), we need to minimize -NCC
        if isinstance(image_loss_fn, ne.nn.modules.NCC):
            img_loss = -img_loss
            
        grad_loss = grad_loss_fn(displacement).mean()

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        
        # 增加梯度裁剪，防止 NCC 在背景区域计算导致梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('--output', type=str, default='/root/autodl-tmp/models/oasis_vxm.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='ncc', choices=['mse', 'ncc'], help='Image similarity loss')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01, help='Weight of gradient loss')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.0, help='Early stopping threshold')
    parser.add_argument('--warm-start', type=int, default=10, help='Early stopping warm start steps')
    parser.add_argument('--train-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', type=str, default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    args = parser.parse_args()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model (using VoxelMorph-2 extended capacity)
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        # 使用更大的接收野和更宽的通道，这是能压平复杂脑回的关键
        nb_features=[
            # 编码器4层，刚好对应图像尺寸 160(可整除16)，避免空间维度拼接不匹配
            [32, 64, 64, 64], 
            # 解码器4层，与编码器对称
            [64, 64, 64, 32]
        ],
        integration_steps=0, # set to 7 if you want diffeomorphic fields
    ).to(device)

    # Setup losses and optimizer
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
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Lambda: {args.lambda_param}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Arguments: {vars(args)}\n")

    # Initialize Logger
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_dsc'])

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_dsc = 0.0
    loss_history = []
    val_dsc_history = []
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            device=device
        )
        loss_history.append(avg_loss)
        
        # Calculate Validation metrics
        val_dsc = validate(
            model=model,
            dataloader=val_loader,
            device=device
        )
        val_dsc_history.append(val_dsc)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}, Val DSC: {val_dsc:.6f}, LR: {current_lr:.6f}')

        # Save visualizations
        try:
            save_qualitative_results(model, val_set, out_path.parent, epoch=epoch+1, device=device)
        except Exception as e:
            print(f"Failed to save visualization: {e}")

        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{val_dsc:.6f}"])

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
    with open(config_file, 'a') as f:
        f.write(f"End Time: {finish_timestamp}\n")

if __name__ == '__main__':
    main()
