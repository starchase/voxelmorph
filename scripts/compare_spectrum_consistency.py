#!/usr/bin/env python3
"""
Compare CT/MR image spectra with modality-invariant MIND structure spectra.

The reported ``*_consistency`` metric matches the normalized-amplitude
consistency used by CrossImageFrequencyConsistencyModulation3D. Pearson
correlation and low/mid/high-band consistency are included as diagnostics.

Example:
  python scripts/compare_spectrum_consistency.py \
      --data-dir /root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images \
      --out-csv spectrum_consistency.csv --max-samples 50
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import nibabel as nib
except Exception:
    nib = None

METRIC_NAMES = (
    'corr',
    'consistency',
    'low_consistency',
    'mid_consistency',
    'high_consistency',
)
PRIMARY_METRIC = 'consistency'


class MINDDescriptor3D(nn.Module):
    """Standalone copy of the repository's 12-channel MIND-SSC extractor."""

    def __init__(self, radius=2, dilation=2, eps=1e-8):
        super().__init__()
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
        first, second = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = (first > second) & torch.isclose(distances, torch.tensor(2.0).sqrt())
        shift_1 = neighbourhood[first[mask]]
        shift_2 = neighbourhood[second[mask]]
        kernel_1 = torch.zeros((shift_1.shape[0], 1, 3, 3, 3), dtype=torch.float32)
        kernel_2 = torch.zeros_like(kernel_1)
        for index in range(shift_1.shape[0]):
            d1, h1, w1 = shift_1[index].tolist()
            d2, h2, w2 = shift_2[index].tolist()
            kernel_1[index, 0, d1, h1, w1] = 1.0
            kernel_2[index, 0, d2, h2, w2] = 1.0
        self.register_buffer('kernel_1', kernel_1)
        self.register_buffer('kernel_2', kernel_2)
        self.register_buffer(
            'permutation',
            torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long),
        )

    def forward(self, image):
        pad = self.dilation
        image = F.pad(image, (pad, pad, pad, pad, pad, pad), mode='replicate')
        shifted_1 = F.conv3d(image, self.kernel_1.to(image.dtype), dilation=self.dilation)
        shifted_2 = F.conv3d(image, self.kernel_2.to(image.dtype), dilation=self.dilation)
        patch_ssd = F.avg_pool3d(
            (shifted_1 - shifted_2).square(),
            kernel_size=2 * self.radius + 1,
            stride=1,
            padding=self.radius,
        )
        patch_ssd = patch_ssd - patch_ssd.amin(dim=1, keepdim=True)
        mind_var = patch_ssd.mean(dim=1, keepdim=True)
        mean_var = mind_var.detach().mean()
        mind_var = torch.clamp(mind_var, min=mean_var * 1e-3, max=mean_var * 1e3)
        return torch.exp(-patch_ssd / (mind_var + self.eps))[:, self.permutation]


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return path.stem


def pair_key(path: Path) -> str:
    stem = nifti_stem(path)
    for suffix in ('_0000', '_0001'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_nifti(path: Path):
    if nib is None:
        raise RuntimeError('nibabel is required to load NIfTI files')
    image = nib.as_closest_canonical(nib.load(str(path)))
    return image.get_fdata(dtype=np.float32), image.affine


def center_crop_pair(first: np.ndarray, second: np.ndarray):
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError(f'Expected 3D volumes, got {first.shape} and {second.shape}')
    shape = tuple(min(a, b) for a, b in zip(first.shape, second.shape))

    def crop(volume):
        starts = tuple((size - target) // 2 for size, target in zip(volume.shape, shape))
        return volume[tuple(slice(start, start + target) for start, target in zip(starts, shape))]

    return crop(first), crop(second)


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    foreground = np.abs(volume) > 1e-6
    values = volume[foreground] if foreground.any() else volume.ravel()
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(volume)
    return np.clip((volume - low) / (high - low), 0.0, 1.0)


def prepare_tensor(volume: np.ndarray, scale: float, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(robust_normalize(volume)).unsqueeze(0).unsqueeze(0).to(device)
    if scale != 1.0:
        size = tuple(max(8, int(round(dim * scale))) for dim in tensor.shape[-3:])
        tensor = F.interpolate(tensor, size=size, mode='trilinear', align_corners=False)
    return tensor


def radial_bands(spatial_shape, device):
    depth, height, width = spatial_shape
    fd = torch.fft.fftfreq(depth, device=device)
    fh = torch.fft.fftfreq(height, device=device)
    fw = torch.fft.rfftfreq(width, device=device)
    radius = torch.sqrt(
        fd[:, None, None].square()
        + fh[None, :, None].square()
        + fw[None, None, :].square()
    )
    radius = radius / radius.max().clamp_min(1e-6)
    return (
        radius <= 1.0 / 3.0,
        (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        radius > 2.0 / 3.0,
    )


def spectrum_metrics(first: torch.Tensor, second: torch.Tensor):
    """Average metrics across descriptor channels without destroying their ordering."""
    if first.shape[1] != second.shape[1]:
        raise ValueError(f'Spectrum channel mismatch: {first.shape} vs {second.shape}')
    if first.shape[-3:] != second.shape[-3:]:
        second = F.interpolate(second, size=first.shape[-3:], mode='trilinear', align_corners=False)
    first_amp = torch.fft.rfftn(first.float(), dim=(-3, -2, -1), norm='ortho').abs()
    second_amp = torch.fft.rfftn(second.float(), dim=(-3, -2, -1), norm='ortho').abs()
    first_amp = first_amp / first_amp.mean(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
    second_amp = second_amp / second_amp.mean(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)

    consistency = (
        2.0 * first_amp * second_amp
        / (first_amp.square() + second_amp.square() + 1e-6)
    )

    first_log = torch.log1p(first_amp).flatten(start_dim=2)
    second_log = torch.log1p(second_amp).flatten(start_dim=2)
    first_log = first_log - first_log.mean(dim=2, keepdim=True)
    second_log = second_log - second_log.mean(dim=2, keepdim=True)
    corr = (
        (first_log * second_log).sum(dim=2)
        / (
            first_log.square().sum(dim=2).sqrt()
            * second_log.square().sum(dim=2).sqrt()
        ).clamp_min(1e-6)
    ).mean()

    bands = radial_bands(first.shape[-3:], first.device)
    band_values = [consistency[..., mask].mean() for mask in bands]
    return {
        'corr': corr.item(),
        'consistency': consistency.mean().item(),
        'low_consistency': band_values[0].item(),
        'mid_consistency': band_values[1].item(),
        'high_consistency': band_values[2].item(),
    }


def find_pairs(data_dir: Path, pair_mode: str, max_samples: int, seed: int):
    ct_dir = data_dir / 'ct'
    mr_dir = data_dir / 'mr'
    if not ct_dir.exists() or not mr_dir.exists():
        raise ValueError(f'Expected separate CT/MR folders under {data_dir}')

    ct_map = {pair_key(path): path for path in sorted(ct_dir.glob('*.nii*'))}
    mr_map = {pair_key(path): path for path in sorted(mr_dir.glob('*.nii*'))}
    exact = [(key, ct_map[key], mr_map[key]) for key in sorted(ct_map.keys() & mr_map.keys())]
    if pair_mode == 'auto':
        pair_mode = 'exact' if exact else 'shuffled'
    if pair_mode == 'exact':
        return exact[:max_samples], pair_mode

    ct_files = sorted(ct_map.values())
    mr_files = sorted(mr_map.values())
    if not ct_files or not mr_files:
        return [], pair_mode
    rng = np.random.default_rng(seed)
    count = max_samples
    ct_indices = rng.integers(0, len(ct_files), size=count)
    mr_indices = rng.integers(0, len(mr_files), size=count)
    pairs = []
    for index in range(count):
        ct_path = ct_files[ct_indices[index]]
        mr_path = mr_files[mr_indices[index]]
        pairs.append((f'{pair_key(ct_path)}__{pair_key(mr_path)}', ct_path, mr_path))
    return pairs, pair_mode


def prefixed(prefix, metrics):
    return {f'{prefix}_{name}': metrics[name] for name in METRIC_NAMES}


def load_prepared_pair(ct_path, mr_path, scale, mind_input_scale, device):
    ct, ct_affine = load_nifti(ct_path)
    mr, mr_affine = load_nifti(mr_path)
    ct, mr = center_crop_pair(ct, mr)
    return (
        prepare_tensor(ct, scale, device),
        prepare_tensor(mr, scale, device),
        prepare_tensor(ct, mind_input_scale, device),
        prepare_tensor(mr, mind_input_scale, device),
        ct_affine,
        mr_affine,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/root/autodl-tmp/classedAbdomenMRCT_norm_300/trainPairs/images')
    parser.add_argument('--out-csv', default='spectrum_consistency.csv')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument(
        '--pair-mode',
        choices=('auto', 'exact', 'shuffled'),
        default='auto',
        help='Use same-case pairs when available or deterministic random inter-patient pairs.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--downsample-factor',
        type=float,
        default=0.125,
        help='Evaluate at the module scale; 0.125 corresponds to the recommended 1/8 scale.',
    )
    parser.add_argument(
        '--mind-input-factor',
        type=float,
        default=0.25,
        help='Extract MIND before resizing it to the evaluation scale; 0.25 preserves more local structure.',
    )
    parser.add_argument('--mind-radius', type=int, default=2)
    parser.add_argument('--mind-dilation', type=int, default=2)
    parser.add_argument('--mind-eps', type=float, default=1e-8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not (0 < args.downsample_factor <= 1):
        parser.error('--downsample-factor must be in (0, 1]')
    if not (args.downsample_factor <= args.mind_input_factor <= 1):
        parser.error('--mind-input-factor must be between --downsample-factor and 1')
    pairs, resolved_pair_mode = find_pairs(
        Path(args.data_dir),
        args.pair_mode,
        args.max_samples,
        args.seed,
    )
    if not pairs:
        raise RuntimeError(f'No CT/MR pairs found under {args.data_dir} with mode {resolved_pair_mode}')
    print(f'Using {len(pairs)} {resolved_pair_mode} CT/MR pairs from {args.data_dir}')

    mind = MINDDescriptor3D(
        radius=args.mind_radius,
        dilation=args.mind_dilation,
        eps=args.mind_eps,
    ).to(args.device).eval()
    rows = []
    prepared = []

    # Precompute the small evaluation-scale tensors so each same-case pair can
    # also be compared with a deterministic shuffled inter-patient pair. The
    # latter reflects the random-pair inputs used by multimodal training.
    for key, ct_path, mr_path in pairs:
        ct_tensor, mr_tensor, ct_mind_input, mr_mind_input, ct_affine, mr_affine = load_prepared_pair(
            ct_path,
            mr_path,
            args.downsample_factor,
            args.mind_input_factor,
            args.device,
        )
        if not np.allclose(ct_affine, mr_affine, atol=1e-3):
            print(f'Warning: {key} has different CT/MR affine matrices; spectrum comparison ignores this.')
        with torch.no_grad():
            ct_mind = mind(ct_mind_input)
            mr_mind = mind(mr_mind_input)
            if ct_mind.shape[-3:] != ct_tensor.shape[-3:]:
                ct_mind = F.interpolate(ct_mind, size=ct_tensor.shape[-3:], mode='trilinear', align_corners=False)
                mr_mind = F.interpolate(mr_mind, size=mr_tensor.shape[-3:], mode='trilinear', align_corners=False)
            prepared.append({
                'key': key,
                'ct': ct_tensor,
                'mr': mr_tensor,
                'ct_mind': ct_mind,
                'mr_mind': mr_mind,
            })

    for index, item in enumerate(prepared):
        shuffled = prepared[(index + 1) % len(prepared)] if len(prepared) > 1 else None
        with torch.no_grad():
            raw_metrics = spectrum_metrics(item['ct'], item['mr'])
            mind_metrics = spectrum_metrics(item['ct_mind'], item['mr_mind'])

        row = {'sample_id': item['key']}
        row.update(prefixed('raw', raw_metrics))
        row.update(prefixed('mind', mind_metrics))
        if shuffled is not None:
            with torch.no_grad():
                raw_shuffled = spectrum_metrics(item['ct'], shuffled['mr'])
                mind_shuffled = spectrum_metrics(item['ct_mind'], shuffled['mr_mind'])
            row.update(prefixed('raw_shuffled', raw_shuffled))
            row.update(prefixed('mind_shuffled', mind_shuffled))
            row['raw_pair_margin'] = raw_metrics[PRIMARY_METRIC] - raw_shuffled[PRIMARY_METRIC]
            row['mind_pair_margin'] = mind_metrics[PRIMARY_METRIC] - mind_shuffled[PRIMARY_METRIC]
        rows.append(row)
        print(
            f'{item["key"]}: raw={raw_metrics["consistency"]:.4f}, '
            f'MIND={mind_metrics["consistency"]:.4f}, '
            f'delta={mind_metrics["consistency"] - raw_metrics["consistency"]:+.4f}, '
            f'MIND pair margin={row.get("mind_pair_margin", float("nan")):+.4f}'
        )

    fields = ['sample_id'] + [
        f'{prefix}_{name}' for prefix in ('raw', 'mind') for name in METRIC_NAMES
    ]
    if len(prepared) > 1:
        fields += [
            f'{prefix}_{name}'
            for prefix in ('raw_shuffled', 'mind_shuffled')
            for name in METRIC_NAMES
        ]
        fields += ['raw_pair_margin', 'mind_pair_margin']
    with open(args.out_csv, 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f'\nCompared {len(rows)} {resolved_pair_mode} pairs at scale {args.downsample_factor:g} '
        f'with MIND extracted at {args.mind_input_factor:g}:'
    )
    for name in METRIC_NAMES:
        raw = np.asarray([row[f'raw_{name}'] for row in rows])
        mind_values = np.asarray([row[f'mind_{name}'] for row in rows])
        delta = mind_values - raw
        print(
            f'{name:>16}: raw={raw.mean():.4f}, MIND={mind_values.mean():.4f}, '
            f'delta={delta.mean():+.4f}, improved={100.0 * (delta > 0).mean():.1f}%'
        )
    if len(prepared) > 1:
        raw_margin = np.asarray([row['raw_pair_margin'] for row in rows])
        mind_margin = np.asarray([row['mind_pair_margin'] for row in rows])
        print(
            f'{"pair margin":>16}: raw={raw_margin.mean():+.4f}, '
            f'MIND={mind_margin.mean():+.4f}, '
            f'MIND positive={100.0 * (mind_margin > 0).mean():.1f}%\n'
            f'{"shuffled MIND":>16}: '
            f'{np.mean([row["mind_shuffled_consistency"] for row in rows]):.4f} '
            f'(this must remain usable because training uses random inter-patient pairs)'
        )
    print(f'Wrote results to {args.out_csv}')


if __name__ == '__main__':
    main()
