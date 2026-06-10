#!/usr/bin/env python3
"""
Compare original image spectrum consistency vs MIND structural spectrum consistency
for CT-MR pairs under a dataset folder.

Saves results to CSV with columns: sample_id, raw_spec_corr, mind_spec_corr

Usage:
  python scripts/compare_spectrum_consistency.py --data-dir /root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images \
      --out results.csv --max-samples 50
"""
import argparse
import csv
import os
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None

import torch

# Try to import local vxm losses for MIND extractor
try:
    import vxm
    from voxelmorph.nn.losses import MINDLoss
except Exception:
    # Fallback: try import from vxm.nn.losses if package name differs
    try:
        from voxelmorph.nn.losses import MINDLoss
    except Exception:
        MINDLoss = None


def load_nifti(path: Path):
    if nib is None:
        raise RuntimeError('nibabel is required to load NIfTI files (pip install nibabel)')
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


def fft_magnitude(volume: np.ndarray):
    # Compute 3D FFT magnitude and return log-magnitude flattened vector
    # Ensure we operate on float32 and avoid huge memory by downcasting if necessary
    arr = np.asarray(volume, dtype=np.float32)
    # Remove singleton channel dims if present
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    f = np.fft.fftn(arr)
    mag = np.abs(f)
    mag = np.fft.fftshift(mag)
    # log scaling for numeric stability
    mag = np.log1p(mag)
    return mag.flatten()


def compute_corr(a: np.ndarray, b: np.ndarray):
    # Pearson correlation between two flat vectors
    a = a.ravel()
    b = b.ravel()
    if a.size != b.size:
        # crop to smallest
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    a_mean = a.mean()
    b_mean = b.mean()
    a_c = a - a_mean
    b_c = b - b_mean
    denom = (np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def mind_descriptor_volume(mind_module, volume: np.ndarray, device='cpu'):
    # mind_module: instance of MINDLoss
    # volume: 3D numpy array
    t = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    t = t.to(device)
    with torch.no_grad():
        desc = mind_module._mind_ssc(t)
        # desc shape: [1, C, D, H, W]
        desc = desc.cpu().numpy()
    # average across channels to a single volume
    desc_mean = desc.mean(axis=1)[0]
    return desc_mean


def find_pairs(data_dir: Path):
    ct_dir = data_dir / 'ct'
    mr_dir = data_dir / 'mr'
    if not ct_dir.exists() or not mr_dir.exists():
        # try flat structure: pair by matching prefixes
        files = list(data_dir.glob('*.nii*'))
        pairs = []
        # naive pairing: match by prefix up to last underscore
        base_map = {}
        for f in files:
            name = f.name
            key = '_'.join(name.split('_')[:-1])
            base_map.setdefault(key, []).append(f)
        for key, fl in base_map.items():
            if len(fl) >= 2:
                # try to find ct and mr by presence of 0001/0000
                ct = None
                mr = None
                for f in fl:
                    if f.name.endswith('_0001.nii') or f.name.endswith('_0001.nii.gz'):
                        ct = f
                    if f.name.endswith('_0000.nii') or f.name.endswith('_0000.nii.gz'):
                        mr = f
                if ct and mr:
                    pairs.append((key, ct, mr))
        return pairs

    ct_files = sorted(ct_dir.glob('*.nii*'))
    mr_files = sorted(mr_dir.glob('*.nii*'))
    # build map by prefix
    ct_map = {('_'.join(p.name.split('_')[:-1])): p for p in ct_files}
    mr_map = {('_'.join(p.name.split('_')[:-1])): p for p in mr_files}
    common = set(ct_map.keys()).intersection(mr_map.keys())
    pairs = []
    for key in sorted(common):
        pairs.append((key, ct_map[key], mr_map[key]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images')
    parser.add_argument('--out-csv', type=str, default='spectrum_consistency.csv')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--mind-radius', type=int, default=2)
    parser.add_argument('--mind-dilation', type=int, default=2)
    parser.add_argument('--mind-eps', type=float, default=1e-8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pairs = find_pairs(data_dir)
    if not pairs:
        print(f'No CT/MR pairs found under {data_dir}')
        return

    max_samples = args.max_samples
    pairs = pairs[:max_samples]

    # prepare MIND module
    if MINDLoss is None:
        print('Warning: MINDLoss implementation not available in this environment; MIND-derived metrics will be skipped.')
        mind_module = None
    else:
        mind_module = MINDLoss(radius=args.mind_radius, dilation=args.mind_dilation, eps=args.mind_eps)
        mind_module = mind_module.to(args.device)

    out_rows = []

    for key, ct_path, mr_path in pairs:
        try:
            ct = load_nifti(ct_path)
            mr = load_nifti(mr_path)
        except Exception as e:
            print(f'Failed to load {ct_path} or {mr_path}: {e}')
            continue

        # ensure same shape by cropping or padding to min shape
        if ct.shape != mr.shape:
            # crop to min along each dim
            min_shape = tuple(min(a, b) for a, b in zip(ct.shape, mr.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            ct = ct[slices]
            mr = mr[slices]

        raw_ct_spec = fft_magnitude(ct)
        raw_mr_spec = fft_magnitude(mr)
        raw_corr = compute_corr(raw_ct_spec, raw_mr_spec)

        mind_corr = None
        if mind_module is not None:
            try:
                ct_desc = mind_descriptor_volume(mind_module, ct, device=args.device)
                mr_desc = mind_descriptor_volume(mind_module, mr, device=args.device)
                ct_desc_spec = fft_magnitude(ct_desc)
                mr_desc_spec = fft_magnitude(mr_desc)
                mind_corr = compute_corr(ct_desc_spec, mr_desc_spec)
            except Exception as e:
                print(f'Failed to compute MIND descriptor for {key}: {e}')
                mind_corr = None

        out_rows.append({'sample_id': key, 'raw_spec_corr': raw_corr, 'mind_spec_corr': mind_corr})
        print(f'{key}: raw_corr={raw_corr:.4f}, mind_corr={mind_corr}')

    # write CSV
    out_path = Path(args.out_csv)
    with open(out_path, 'w', newline='') as csvfile:
        fieldnames = ['sample_id', 'raw_spec_corr', 'mind_spec_corr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f'Wrote results to {out_path}')


if __name__ == '__main__':
    main()
