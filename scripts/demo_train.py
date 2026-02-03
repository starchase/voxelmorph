#!/usr/bin/env python3
"""Minimal demo trainer for VoxelMorph using a local dataset directory.

This script performs a short training run on volumes found in a directory
(e.g. `./AbdomenMRCT/imagesTr`) and saves a `state_dict` checkpoint.

Designed for quick demos only.
"""
import argparse
import glob
import os
import torch
import numpy as np

import voxelmorph as vxm
import neurite as ne


def collect_vols(data_dir):
    exts = ('*.nii', '*.nii.gz', '*.npz')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(data_dir, e))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f'No volume files found in {data_dir}')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./AbdomenMRCT/imagesTr')
    parser.add_argument('--output', type=str, default='./models/demo_abdomen.pt')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps-per-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of volume files (for quick demos)')
    args = parser.parse_args()

    # device
    if args.gpu is not None and args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    vols = collect_vols(args.data_dir)
    if args.max_files is not None:
        vols = vols[: args.max_files]
        print(f'Using first {len(vols)} files for demo training')

    # build model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    image_loss = ne.nn.modules.MSE()
    grad_loss = ne.nn.modules.SpatialGradient('l2')

    # create generator (uses voxelmorph's generators)
    gen = vxm.py.generators.scan_to_scan(vols, batch_size=args.batch_size, add_feat_axis=True)

    print(f'Found {len(vols)} volumes. Training on device: {device}')

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step in range(args.steps_per_epoch):
            batch = next(gen)
            # batch is (invols, outvols); invols is [scan1, scan2]
            scan1 = batch[0][0]
            scan2 = batch[0][1]

            src = torch.from_numpy(scan1).float().to(device).permute(0, 4, 1, 2, 3)
            tgt = torch.from_numpy(scan2).float().to(device).permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            displacement, warped = model(src, tgt, return_warped_source=True, return_field_type='displacement')

            img_loss = image_loss(tgt, warped)
            g_loss = grad_loss(displacement)
            loss = img_loss + 0.01 * g_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg = total_loss / args.steps_per_epoch
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {avg:.6f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f'Model state_dict saved to {args.output}')


if __name__ == '__main__':
    main()
