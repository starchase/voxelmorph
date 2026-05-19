#!/usr/bin/env python3
"""Visualize multimodal registration inference results for all test cases.

This script reuses the dataset and rendering utilities from
`scripts/train_multimodal.py`. It reconstructs the model from a saved training
`config.txt` plus checkpoint/state-dict inspection, loads the specified weight,
and generates one visualization per paired test case.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import neurite as ne

import voxelmorph as vxm

from train_multimodal import MultimodalValidationDataset, save_qualitative_results


class ScriptSpatialTransformer(torch.nn.Module):
    def __init__(self, interpolation_mode: str = 'linear', align_corners: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def forward(self, moving_image: torch.Tensor, deformation_field: torch.Tensor) -> torch.Tensor:
        spatial_shape = moving_image.shape[2:]
        if not hasattr(self, 'meshgrid') or self.meshgrid.shape[1:] != spatial_shape:
            self.meshgrid = ne.volshape_to_ndgrid(
                size=spatial_shape,
                device=moving_image.device,
                dtype=moving_image.dtype,
                stack=True,
            )

        return vxm.spatial_transform(
            image=moving_image,
            trf=deformation_field,
            mode=self.interpolation_mode,
            isdisp=True,
            meshgrid=self.meshgrid,
            non_spatial_dims=(0, 1),
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )


def _binary_dilate_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _binary_erode_3d(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return 1.0 - F.max_pool3d(1.0 - mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def soften_source_boundary(
    source: torch.Tensor,
    fg_threshold: float = 0.03,
    band_kernel: int = 7,
    blur_kernel: int = 5,
    soften_strength: float = 1.0,
) -> torch.Tensor:
    if source.dim() != 4:
        raise ValueError(f'source must have shape (C, D, H, W), got {tuple(source.shape)}')

    if band_kernel % 2 == 0 or blur_kernel % 2 == 0:
        raise ValueError('band_kernel and blur_kernel must be odd numbers.')

    source_batched = source.unsqueeze(0)
    foreground = (source_batched > fg_threshold).float()
    if torch.count_nonzero(foreground).item() == 0:
        return source

    outer_band = (_binary_dilate_3d(foreground, band_kernel) - foreground).clamp_(0.0, 1.0)
    if torch.count_nonzero(outer_band).item() == 0:
        return source

    blurred = F.avg_pool3d(source_batched, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    blend_weight = F.avg_pool3d(outer_band, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    blend_weight = blend_weight.clamp_(0.0, 1.0) * soften_strength
    softened = source_batched * (1.0 - blend_weight) + blurred * blend_weight
    return softened.squeeze(0)


class InferencePreprocessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: MultimodalValidationDataset,
        soften_source_boundary_enabled: bool = False,
        soften_fg_threshold: float = 0.03,
        soften_band_kernel: int = 7,
        soften_blur_kernel: int = 5,
        soften_strength: float = 1.0,
    ):
        self.base_dataset = base_dataset
        self.soften_source_boundary_enabled = soften_source_boundary_enabled
        self.soften_fg_threshold = soften_fg_threshold
        self.soften_band_kernel = soften_band_kernel
        self.soften_blur_kernel = soften_blur_kernel
        self.soften_strength = soften_strength

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = dict(self.base_dataset[index])
        if self.soften_source_boundary_enabled:
            sample['source'] = soften_source_boundary(
                sample['source'],
                fg_threshold=self.soften_fg_threshold,
                band_kernel=self.soften_band_kernel,
                blur_kernel=self.soften_blur_kernel,
                soften_strength=self.soften_strength,
            )
        return sample


DEFAULT_ARGS: Dict[str, Any] = {
    'ct_test_dir': '/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/images/ct',
    'mr_test_dir': '/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/images/mr',
    'ct_test_label_dir': '/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/labels/ct',
    'mr_test_label_dir': '/root/autodl-tmp/classedAbdomenMRCT_norm_300/test/labels/mr',
    'network': 'siamese',
    'workers': 4,
    'batch_size': 1,
    'integration_steps': 0,
    'decouple_layers': 2,
    'use_daps': False,
    'use_pdaps': False,
    'use_dsin': False,
    'use_cmim': False,
    'use_cross_mamba': False,
    'cross_mamba_scales': '1/16,1/8',
    'cross_mamba_offset_limit': 0.0,
    'cross_mamba_offset_smooth_kernel': 1,
    'use_wcv': False,
    'use_swcv': False,
    'use_gcv': False,
    'use_wmca': False,
    'encoder_type': 'cnn',
    'mamba_shallow_multi': False,
    'mamba_enc_shallow_multi': False,
    'mamba_dec_shallow_multi': False,
    'mamba_quarter_scale': False,
    'mamba_parallel_block': False,
    'fusion_method': 'compress_concat',
    'window_size': 9,
    'pdaps_flow_limit': 20.0,
    'use_residual_flow_pyramid': False,
    'residual_flow_limit': 4.0,
    'use_boundary_branch': False,
    'boundary_branch_scales': 'deep',
    'boundary_branch_strength': 0.5,
    'boundary_kernel': 'sobel',
    'boundary_smooth_kernel': 3,
}


def parse_training_args(config_path: Path) -> Dict[str, Any]:
    args_dict: Dict[str, Any] = {}
    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('Arguments: '):
                payload = line.split('Arguments: ', 1)[1].strip()
                args_dict = ast.literal_eval(payload)
                break
    merged = dict(DEFAULT_ARGS)
    merged.update(args_dict)
    return merged


def infer_model_settings(args_dict: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = set(state_dict.keys())

    inferred = dict(args_dict)

    if any(k.startswith('encoder.') for k in keys):
        inferred['network'] = 'siamese'
    elif any(k.startswith('model.') or k.startswith('flow_layer.') for k in keys):
        inferred['network'] = 'vxm'

    inferred['use_daps'] = inferred.get('use_daps', False) or any(k.startswith('daps_plr_blocks.') for k in keys)
    inferred['use_pdaps'] = inferred.get('use_pdaps', False) or any(k.startswith('pyramid_flows.') for k in keys)
    inferred['use_residual_flow_pyramid'] = inferred.get('use_residual_flow_pyramid', False) or any(k.startswith('residual_flow_heads.') for k in keys)
    inferred['use_dsin'] = inferred.get('use_dsin', False) or any('dsin' in k.lower() for k in keys)
    inferred['use_cmim'] = inferred.get('use_cmim', False) or any(k.startswith('cmim_blocks.') for k in keys)
    inferred['use_cross_mamba'] = inferred.get('use_cross_mamba', False) or any(k.startswith('cross_mamba_blocks.') for k in keys)
    inferred['use_wcv'] = inferred.get('use_wcv', False) or any(k.startswith('wcv_blocks.') for k in keys)
    inferred['use_swcv'] = inferred.get('use_swcv', False) or any(k.startswith('swcv_blocks.') for k in keys)
    inferred['use_gcv'] = inferred.get('use_gcv', False) or any(k.startswith('gcv_blocks.') for k in keys)
    inferred['use_boundary_branch'] = inferred.get('use_boundary_branch', False) or any(k.startswith('boundary_guidance.') for k in keys)

    if any('mamba' in k.lower() for k in keys):
        inferred['encoder_type'] = 'mamba'

    return inferred


def adapt_legacy_state_dict(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> tuple[Dict[str, torch.Tensor], list[str]]:
    """Backfill known missing parameters from older checkpoints.

    Older CMIM checkpoints were saved before `fuse_conv` was introduced.
    To preserve the old behavior as closely as possible, initialize
    `fuse_conv` to pass through only the attention output half of the
    concatenated tensor `[source, out]`, so the module computes
    `norm(out + source)`.
    """
    adapted = dict(state_dict)
    notes: list[str] = []

    model_state = model.state_dict()
    for key, tensor in model_state.items():
        if key in adapted:
            continue

        if key.endswith('fuse_conv.weight') and key.startswith('cmim_blocks.'):
            out_channels, in_channels, *kernel = tensor.shape
            if len(kernel) != 3 or tuple(kernel) != (1, 1, 1):
                continue
            if in_channels != out_channels * 2:
                continue

            weight = torch.zeros_like(tensor)
            for channel in range(out_channels):
                weight[channel, out_channels + channel, 0, 0, 0] = 1.0
            adapted[key] = weight
            notes.append(f'Initialized legacy missing key: {key}')
        elif key.endswith('fuse_conv.bias') and key.startswith('cmim_blocks.'):
            adapted[key] = torch.zeros_like(tensor)
            notes.append(f'Initialized legacy missing key: {key}')

    return adapted, notes


def build_model(args_dict: Dict[str, Any], inshape: tuple[int, int, int], device: str) -> torch.nn.Module:
    if args_dict['network'] == 'vxm':
        model = vxm.nn.models.VxmPairwise(
            ndim=3,
            source_channels=1,
            target_channels=1,
            nb_features=([32, 64, 64, 64], [64, 64, 64, 32]),
            integration_steps=args_dict['integration_steps'],
        )
    else:
        model = vxm.nn.SiameseUNetBaseline(
            inshape=inshape,
            in_channels=1,
            enc_nf=[32, 64, 64, 64],
            dec_nf=[64, 64, 64, 32],
            ndim=3,
            int_steps=args_dict['integration_steps'],
            decouple_layers=args_dict['decouple_layers'],
            use_daps=args_dict['use_daps'],
            use_pdaps=args_dict['use_pdaps'],
            use_dsin=args_dict['use_dsin'],
            use_cmim=args_dict['use_cmim'],
            use_cross_mamba=args_dict['use_cross_mamba'],
            cross_mamba_scales=args_dict['cross_mamba_scales'],
            cross_mamba_offset_limit=args_dict['cross_mamba_offset_limit'],
            cross_mamba_offset_smooth_kernel=args_dict['cross_mamba_offset_smooth_kernel'],
            use_wcv=(args_dict['use_wcv'] or args_dict['use_wmca']),
            use_swcv=args_dict['use_swcv'],
            use_gcv=args_dict['use_gcv'],
            encoder_type=args_dict['encoder_type'],
            mamba_shallow_multi=args_dict['mamba_shallow_multi'],
            mamba_enc_shallow_multi=args_dict['mamba_enc_shallow_multi'],
            mamba_dec_shallow_multi=args_dict['mamba_dec_shallow_multi'],
            mamba_quarter_scale=args_dict['mamba_quarter_scale'],
            mamba_parallel_block=args_dict['mamba_parallel_block'],
            fusion_method=args_dict['fusion_method'],
            window_size=args_dict['window_size'],
            pdaps_flow_limit=args_dict['pdaps_flow_limit'],
            use_residual_flow_pyramid=args_dict['use_residual_flow_pyramid'],
            residual_flow_limit=args_dict['residual_flow_limit'],
            use_boundary_branch=args_dict['use_boundary_branch'],
            boundary_branch_scales=args_dict['boundary_branch_scales'],
            boundary_branch_strength=args_dict['boundary_branch_strength'],
            boundary_kernel=args_dict['boundary_kernel'],
            boundary_smooth_kernel=args_dict['boundary_smooth_kernel'],
        )
    return model.to(device)


def build_test_dataset(args_dict: Dict[str, Any], device: str) -> MultimodalValidationDataset:
    return MultimodalValidationDataset(
        args_dict['ct_test_dir'],
        args_dict['mr_test_dir'],
        ct_label_dir=args_dict.get('ct_test_label_dir'),
        mr_label_dir=args_dict.get('mr_test_label_dir'),
        device=device,
        paired=True,
    )


def apply_warp_padding_mode(model: torch.nn.Module, padding_mode: str) -> None:
    if hasattr(model, 'spatial_transformer'):
        model.spatial_transformer = ScriptSpatialTransformer(
            interpolation_mode='linear',
            align_corners=True,
            padding_mode=padding_mode,
        ).to(next(model.parameters()).device)

    if hasattr(model, 'spatial_transform'):
        model.spatial_transform = ScriptSpatialTransformer(
            interpolation_mode='linear',
            align_corners=True,
            padding_mode=padding_mode,
        ).to(next(model.parameters()).device)


def get_rotated_slice(img_tensor: torch.Tensor | None, z_idx: int) -> np.ndarray | None:
    if img_tensor is None:
        return None

    slice_tensor = img_tensor[:, :, :, :, z_idx]
    slice_np = slice_tensor.detach().cpu().numpy()[0, 0]
    return np.rot90(slice_np)


def save_padding_comparison_results(
    model: torch.nn.Module,
    dataset: MultimodalValidationDataset,
    output_dir: Path,
    device: str,
    padding_a: str = 'zeros',
    padding_b: str = 'border',
) -> None:
    warp_a = ScriptSpatialTransformer(padding_mode=padding_a).to(device)
    warp_b = ScriptSpatialTransformer(padding_mode=padding_b).to(device)

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        name_tag = sample.get('filename', f'{sample_idx:04d}')
        source = sample['source'].unsqueeze(0).to(device)
        target = sample['target'].unsqueeze(0).to(device)

        with torch.no_grad():
            displacement = model(source, target, return_warped_source=False, return_field_type='displacement')
            warped_a = warp_a(source, displacement)
            warped_b = warp_b(source, displacement)

        padding_diff = (warped_a - warped_b).abs()[0, 0]
        slice_scores = padding_diff.mean(dim=(0, 1))
        slice_idx = int(torch.argmax(slice_scores).item())
        slice_score = float(slice_scores[slice_idx].item())

        src_slice = get_rotated_slice(source, slice_idx)
        tgt_slice = get_rotated_slice(target, slice_idx)
        warped_a_slice = get_rotated_slice(warped_a, slice_idx)
        warped_b_slice = get_rotated_slice(warped_b, slice_idx)
        pad_diff_slice = get_rotated_slice((warped_a - warped_b).abs(), slice_idx)
        diff_a_slice = warped_a_slice - tgt_slice
        diff_b_slice = warped_b_slice - tgt_slice

        vmax_pad_diff = max(float(np.max(pad_diff_slice)), 1e-6)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        axes[0, 0].imshow(src_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Source\nSlice {slice_idx}')
        axes[0, 1].imshow(tgt_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Target')
        axes[0, 2].imshow(pad_diff_slice, cmap='hot', vmin=0, vmax=vmax_pad_diff)
        axes[0, 2].set_title(f'|{padding_a} - {padding_b}|\nmean={slice_score:.4f}')

        axes[1, 0].imshow(warped_a_slice, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Warped ({padding_a})')
        axes[1, 1].imshow(warped_b_slice, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Warped ({padding_b})')
        axes[1, 2].imshow(diff_a_slice, cmap='bwr', vmin=-1, vmax=1, alpha=0.5)
        axes[1, 2].imshow(diff_b_slice, cmap='viridis', vmin=-1, vmax=1, alpha=0.5)
        axes[1, 2].set_title(f'Diff to Target\n{padding_a}=bwr, {padding_b}=viridis')

        for ax in axes.flat:
            ax.axis('off')

        fig.suptitle(f'{name_tag} | padding comparison at max-diff slice', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_dir / f'vis_padding_compare_{name_tag}.png', dpi=150)
        plt.close(fig)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize multimodal inference for all test cases.')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model checkpoint, e.g. multimodal_vxm_mi_mind_best_test.pt',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional config.txt path. Defaults to sibling config.txt next to weights.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save visualizations. Defaults to <weights_dir>/inference_vis_<checkpoint_stem>.',
    )
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use when CUDA is available.')
    parser.add_argument('--ct-test-dir', type=str, default=None, help='Override CT test image directory.')
    parser.add_argument('--mr-test-dir', type=str, default=None, help='Override MR test image directory.')
    parser.add_argument('--ct-test-label-dir', type=str, default=None, help='Override CT test label directory.')
    parser.add_argument('--mr-test-label-dir', type=str, default=None, help='Override MR test label directory.')
    parser.add_argument('--boundary-kernel', type=str, default=None, choices=['sobel', 'diff'], help='Override boundary visualization operator.')
    parser.add_argument('--boundary-smooth-kernel', type=int, default=None, help='Override boundary smoothing kernel size.')
    parser.add_argument('--warp-padding-mode', type=str, default='zeros', choices=['zeros', 'border', 'reflection'], help='Padding mode used by the final image warp during inference visualization.')
    parser.add_argument('--slice-selection-mode', type=str, default='default', choices=['default', 'padding-diff-max'], help='Use default training-style visualization or compare two padding modes at the slice with maximum warp difference.')
    parser.add_argument('--compare-padding-a', type=str, default='zeros', choices=['zeros', 'border', 'reflection'], help='First padding mode for padding-diff-max comparison.')
    parser.add_argument('--compare-padding-b', type=str, default='border', choices=['zeros', 'border', 'reflection'], help='Second padding mode for padding-diff-max comparison.')
    parser.add_argument('--soften-source-boundary', action='store_true', help='Apply inference-only softening to the source foreground/background boundary before model inference.')
    parser.add_argument('--soften-fg-threshold', type=float, default=0.03, help='Foreground threshold used to detect the source body region for boundary softening.')
    parser.add_argument('--soften-band-kernel', type=int, default=7, help='Odd kernel size used to define the boundary band on the source volume.')
    parser.add_argument('--soften-blur-kernel', type=int, default=5, help='Odd kernel size for local averaging when softening the source boundary.')
    parser.add_argument('--soften-strength', type=float, default=1.0, help='Blend strength for inference-only source boundary softening, typically in [0, 1].')
    return parser.parse_args()


def main() -> None:
    cli = parse_cli()

    weights_path = Path(cli.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {weights_path}')

    config_path = Path(cli.config).resolve() if cli.config else (weights_path.parent / 'config.txt')
    if not config_path.exists():
        raise FileNotFoundError(f'config.txt not found: {config_path}')

    args_dict = parse_training_args(config_path)
    for key in ['ct_test_dir', 'mr_test_dir', 'ct_test_label_dir', 'mr_test_label_dir', 'boundary_kernel', 'boundary_smooth_kernel']:
        override = getattr(cli, key)
        if override is not None:
            args_dict[key] = override

    if torch.cuda.is_available():
        torch.cuda.set_device(int(cli.gpu))
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Using device: {device}')
    print(f'Loading config from: {config_path}')
    print(f'Loading weights from: {weights_path}')

    base_dataset = build_test_dataset(args_dict, device=device)
    if len(base_dataset) == 0:
        raise RuntimeError('Test dataset is empty. Please verify test directories in config or CLI overrides.')

    dataset = InferencePreprocessDataset(
        base_dataset,
        soften_source_boundary_enabled=cli.soften_source_boundary,
        soften_fg_threshold=cli.soften_fg_threshold,
        soften_band_kernel=cli.soften_band_kernel,
        soften_blur_kernel=cli.soften_blur_kernel,
        soften_strength=cli.soften_strength,
    )

    sample = dataset[0]
    inshape = tuple(sample['source'].shape[1:])

    state_dict = torch.load(weights_path, map_location=device)
    args_dict = infer_model_settings(args_dict, state_dict)
    model = build_model(args_dict, inshape=inshape, device=device)
    state_dict, compat_notes = adapt_legacy_state_dict(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unresolved_missing = [key for key in missing if key not in state_dict]
    if unresolved_missing or unexpected:
        details = []
        if unresolved_missing:
            details.append(f'missing keys: {unresolved_missing}')
        if unexpected:
            details.append(f'unexpected keys: {unexpected}')
        raise RuntimeError('Failed to load checkpoint strictly after compatibility adaptation: ' + '; '.join(details))
    model.eval()

    for note in compat_notes:
        print(f'[Compat] {note}')

    if cli.output_dir:
        output_dir = Path(cli.output_dir).resolve()
    elif cli.slice_selection_mode == 'padding-diff-max':
        suffix = '_softsrc' if cli.soften_source_boundary else ''
        output_dir = weights_path.parent / f'inference_vis_{weights_path.stem}_padding_compare_{cli.compare_padding_a}_vs_{cli.compare_padding_b}{suffix}'
    else:
        soften_suffix = '_softsrc' if cli.soften_source_boundary else ''
        output_dir = weights_path.parent / f'inference_vis_{weights_path.stem}_{cli.warp_padding_mode}{soften_suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Network: {args_dict["network"]}')
    print(f'Found {len(dataset)} paired test cases.')
    print(f'Saving visualizations to: {output_dir}')
    if cli.soften_source_boundary:
        print(
            'Inference source boundary softening: '
            f'threshold={cli.soften_fg_threshold}, '
            f'band_kernel={cli.soften_band_kernel}, '
            f'blur_kernel={cli.soften_blur_kernel}, '
            f'strength={cli.soften_strength}'
        )

    if cli.slice_selection_mode == 'padding-diff-max':
        print(f'Padding comparison mode: {cli.compare_padding_a} vs {cli.compare_padding_b}')
        save_padding_comparison_results(
            model,
            dataset,
            output_dir=output_dir,
            device=device,
            padding_a=cli.compare_padding_a,
            padding_b=cli.compare_padding_b,
        )
    else:
        apply_warp_padding_mode(model, cli.warp_padding_mode)
        print(f'Warp padding mode: {cli.warp_padding_mode}')
        save_qualitative_results(
            model,
            dataset,
            output_dir=output_dir,
            epoch=0,
            device=device,
            suffix='_test_infer',
            best_sample_idx=None,
            boundary_kernel=args_dict['boundary_kernel'],
            boundary_smooth_kernel=args_dict['boundary_smooth_kernel'],
        )

    print('Done.')


if __name__ == '__main__':
    main()