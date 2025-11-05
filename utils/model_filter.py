"""
Model filtering utilities for OHDGS
Provides functionality to save filtered versions of trained models
by removing Gaussians based on importance scores
"""

import torch
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import json


def filter_gaussians_by_importance(gaussians, filter_top_percent=0.0, filter_bottom_percent=0.0):
    """
    Filter Gaussians based on importance scores

    Args:
        gaussians: GaussianModel object
        filter_top_percent: Remove top X% most important Gaussians (0-1)
        filter_bottom_percent: Remove bottom X% least important Gaussians (0-1)

    Returns:
        filtered_gaussians: GaussianModel object with filtered Gaussians
        filter_stats: Dictionary with filtering statistics
    """
    # Compute importance scores
    try:
        gaussians.compute_importance()
    except AttributeError as e:
        print(f"Error computing importance: {e}")
        # Fallback: create dummy importance scores
        importance_scores = torch.ones(len(gaussians._xyz), device=gaussians._xyz.device)
    else:
        importance_scores = gaussians.importance

    # Ensure importance_scores is a tensor
    if not isinstance(importance_scores, torch.Tensor):
        importance_scores = torch.tensor(importance_scores, device=gaussians._xyz.device)

    total_gaussians = len(importance_scores)

    # Convert to numpy for percentile computation
    if isinstance(importance_scores, torch.Tensor):
        importance_np = importance_scores.cpu().numpy()
    else:
        importance_np = importance_scores

    # Determine indices to keep
    keep_indices = np.arange(total_gaussians)

    # Remove top X% most important
    if filter_top_percent > 0:
        top_threshold = np.percentile(importance_np, 100 * (1 - filter_top_percent))
        top_mask = importance_np >= top_threshold
        keep_indices = keep_indices[~top_mask]

    # Remove bottom X% least important
    if filter_bottom_percent > 0:
        bottom_threshold = np.percentile(importance_np, 100 * filter_bottom_percent)
        bottom_mask = importance_np <= bottom_threshold
        keep_indices = keep_indices[~bottom_mask]

    # Create filtered model
    # GaussianModel expects args parameter, not sh_degree
    filtered_gaussians = type(gaussians)(gaussians.args)
    filtered_gaussians.active_sh_degree = gaussians.active_sh_degree

    # Copy only the Gaussians we want to keep
    if len(keep_indices) > 0:
        # Convert to tensor for indexing
        if isinstance(keep_indices, np.ndarray):
            keep_indices = torch.from_numpy(keep_indices).to(gaussians._xyz.device)

        # Basic Gaussian attributes (always exist)
        filtered_gaussians._xyz = gaussians._xyz[keep_indices].clone()
        filtered_gaussians._features_dc = gaussians._features_dc[keep_indices].clone()
        filtered_gaussians._features_rest = gaussians._features_rest[keep_indices].clone()
        filtered_gaussians._opacity = gaussians._opacity[keep_indices].clone()
        filtered_gaussians._scaling = gaussians._scaling[keep_indices].clone()
        filtered_gaussians._rotation = gaussians._rotation[keep_indices].clone()

        # Gradient accumulation tensors (check existence)
        if hasattr(gaussians, '_opacity_gradient_accum'):
            filtered_gaussians._opacity_gradient_accum = gaussians._opacity_gradient_accum[keep_indices].clone()
        if hasattr(gaussians, '_max_radii2D') and len(gaussians._max_radii2D) > 0:
            if len(gaussians._max_radii2D) == len(gaussians._xyz):
                filtered_gaussians._max_radii2D = gaussians._max_radii2D[keep_indices].clone()
            else:
                filtered_gaussians._max_radii2D = gaussians._max_radii2D.clone()
        if hasattr(gaussians, '_xyz_gradient_accum') and len(gaussians._xyz_gradient_accum) > 0:
            if len(gaussians._xyz_gradient_accum) == len(gaussians._xyz):
                filtered_gaussians._xyz_gradient_accum = gaussians._xyz_gradient_accum[keep_indices].clone()
            else:
                filtered_gaussians._xyz_gradient_accum = gaussians._xyz_gradient_accum.clone()
        if hasattr(gaussians, '_denom') and len(gaussians._denom) > 0:
            if len(gaussians._denom) == len(gaussians._xyz):
                filtered_gaussians._denom = gaussians._denom[keep_indices].clone()
            else:
                filtered_gaussians._denom = gaussians._denom.clone()

        # OHDGS specific attributes (check existence)
        if hasattr(gaussians, 'importance_lambda'):
            filtered_gaussians.importance_lambda = gaussians.importance_lambda
        if hasattr(gaussians, 'confidence'):
            if len(gaussians.confidence) > 0:
                if len(gaussians.confidence) == len(gaussians._xyz):
                    filtered_gaussians.confidence = gaussians.confidence[keep_indices].clone()
                else:
                    filtered_gaussians.confidence = gaussians.confidence.clone()
            else:
                filtered_gaussians.confidence = torch.empty(0, device=gaussians._xyz.device)
        if hasattr(gaussians, 'bg_color'):
            if len(gaussians.bg_color) > 0:
                if len(gaussians.bg_color) == len(gaussians._xyz):
                    filtered_gaussians.bg_color = gaussians.bg_color[keep_indices].clone()
                else:
                    filtered_gaussians.bg_color = gaussians.bg_color.clone()
            else:
                filtered_gaussians.bg_color = torch.empty(0, device=gaussians._xyz.device)
    else:
        # Edge case: no Gaussians left
        print("Warning: All Gaussians filtered out!")

    # Compute statistics
    # Use torch methods for tensors
    if isinstance(importance_scores, torch.Tensor):
        stats_min = float(torch.min(importance_scores))
        stats_max = float(torch.max(importance_scores))
        stats_mean = float(torch.mean(importance_scores))
        stats_median = float(torch.median(importance_scores))
    else:
        stats_min = float(np.min(importance_scores))
        stats_max = float(np.max(importance_scores))
        stats_mean = float(np.mean(importance_scores))
        stats_median = float(np.median(importance_scores))

    filter_stats = {
        'total_original': total_gaussians,
        'total_filtered': len(keep_indices),
        'removed_top': int(total_gaussians * filter_top_percent) if filter_top_percent > 0 else 0,
        'removed_bottom': int(total_gaussians * filter_bottom_percent) if filter_bottom_percent > 0 else 0,
        'filter_percentage': 100 * (total_gaussians - len(keep_indices)) / total_gaussians,
        'importance_range': {
            'min': stats_min,
            'max': stats_max,
            'mean': stats_mean,
            'median': stats_median
        }
    }

    return filtered_gaussians, filter_stats


def evaluate_filtered_model(filtered_gaussians, scene, pipe, background, device='cuda'):
    """
    Evaluate the performance of a filtered model

    Args:
        filtered_gaussians: Filtered GaussianModel object
        scene: Scene object with test views
        pipe: Pipeline parameters
        background: Background tensor
        device: Device to run on

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    from gaussian_renderer import render

    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'l1': []
    }

    # Get test views
    test_views = scene.getTestCameras()

    print(f"Evaluating on {len(test_views)} test views...")

    for viewpoint_cam in tqdm(test_views, desc="Evaluating"):
        with torch.no_grad():
            # Render
            render_pkg = render(viewpoint_cam, filtered_gaussians, pipe, background)
            rendered_image = render_pkg["render"]

            # Get ground truth
            gt_image = viewpoint_cam.original_image.to(device)

            # Compute metrics
            l1 = l1_loss(rendered_image, gt_image)
            psnr_value = psnr(rendered_image, gt_image).mean()
            ssim_value = ssim(rendered_image, gt_image).mean()

            metrics['l1'].append(l1.item())
            metrics['psnr'].append(psnr_value.item())
            metrics['ssim'].append(ssim_value.item())

    # Aggregate metrics
    avg_metrics = {
        'psnr_mean': np.mean(metrics['psnr']),
        'psnr_std': np.std(metrics['psnr']),
        'ssim_mean': np.mean(metrics['ssim']),
        'ssim_std': np.std(metrics['ssim']),
        'l1_mean': np.mean(metrics['l1']),
        'l1_std': np.std(metrics['l1']),
        'num_views': len(test_views)
    }

    return avg_metrics


def save_filtered_model_and_evaluate(original_gaussians, scene, pipe, background,
                                  save_path, filter_top_percent=0.0,
                                  filter_bottom_percent=0.0, device='cuda'):
    """
    Save filtered model versions and evaluate them

    Args:
        original_gaussians: Original trained GaussianModel
        scene: Scene object
        pipe: Pipeline parameters
        background: Background tensor
        save_path: Base path for saving models
        filter_top_percent: Remove top X% most important Gaussians
        filter_bottom_percent: Remove bottom X% least important Gaussians
        device: Device to run on

    Returns:
        results: Dictionary with filtering results and metrics
    """
    results = {
        'filter_configs': [],
        'filter_results': []
    }

    # Create results directory
    results_dir = os.path.join(save_path, 'filtered_models')
    os.makedirs(results_dir, exist_ok=True)

    # Determine filter configurations to test
    filter_configs = []

    if filter_top_percent > 0:
        filter_configs.append({
            'name': f'top_{filter_top_percent*100:.0f}%_removed',
            'filter_top_percent': filter_top_percent,
            'filter_bottom_percent': 0.0
        })

    if filter_bottom_percent > 0:
        filter_configs.append({
            'name': f'bottom_{filter_bottom_percent*100:.0f}%_removed',
            'filter_top_percent': 0.0,
            'filter_bottom_percent': filter_bottom_percent
        })

    if filter_top_percent > 0 and filter_bottom_percent > 0:
        filter_configs.append({
            'name': f'top_{filter_top_percent*100:.0f}%_and_bottom_{filter_bottom_percent*100:.0f}%_removed',
            'filter_top_percent': filter_top_percent,
            'filter_bottom_percent': filter_bottom_percent
        })

    # Evaluate original model as baseline
    print("\nEvaluating original model...")
    baseline_metrics = evaluate_filtered_model(original_gaussians, scene, pipe, background, device)

    results['baseline_metrics'] = baseline_metrics
    results['filter_configs'] = filter_configs

    # Process each filter configuration
    for config in filter_configs:
        print(f"\nProcessing filter configuration: {config['name']}")

        # Filter Gaussians
        filtered_gaussians, filter_stats = filter_gaussians_by_importance(
            original_gaussians,
            filter_top_percent=config['filter_top_percent'],
            filter_bottom_percent=config['filter_bottom_percent']
        )

        print(f"Original: {filter_stats['total_original']:,} Gaussians")
        print(f"Filtered: {filter_stats['total_filtered']:,} Gaussians")
        print(f"Removed: {filter_stats['total_original'] - filter_stats['total_filtered']:,} ({filter_stats['filter_percentage']:.1f}%)")

        # Evaluate filtered model
        filtered_metrics = evaluate_filtered_model(filtered_gaussians, scene, pipe, background, device)

        # Save filtered model checkpoint
        model_path = os.path.join(results_dir, f"filtered_{config['name']}.pth")

        # Build checkpoint dict with only existing attributes
        checkpoint_dict = {
            'xyz': filtered_gaussians._xyz,
            'features_dc': filtered_gaussians._features_dc,
            'features_rest': filtered_gaussians._features_rest,
            'opacity': filtered_gaussians._opacity,
            'scaling': filtered_gaussians._scaling,
            'rotation': filtered_gaussians._rotation,
            'active_sh_degree': filtered_gaussians.active_sh_degree,
        }

        # Add gradient accum tensors if they exist
        if hasattr(filtered_gaussians, '_max_radii2D') and len(filtered_gaussians._max_radii2D) > 0:
            checkpoint_dict['max_radii2D'] = filtered_gaussians._max_radii2D
        if hasattr(filtered_gaussians, '_xyz_gradient_accum') and len(filtered_gaussians._xyz_gradient_accum) > 0:
            checkpoint_dict['xyz_gradient_accum'] = filtered_gaussians._xyz_gradient_accum
        if hasattr(filtered_gaussians, '_denom') and len(filtered_gaussians._denom) > 0:
            checkpoint_dict['denom'] = filtered_gaussians._denom
        if hasattr(filtered_gaussians, '_opacity_gradient_accum'):
            checkpoint_dict['opacity_gradient_accum'] = filtered_gaussians._opacity_gradient_accum

        # Add OHDGS specific attributes
        checkpoint_dict['importance_lambda'] = getattr(filtered_gaussians, 'importance_lambda', 1.0)
        if hasattr(filtered_gaussians, 'confidence'):
            checkpoint_dict['confidence'] = filtered_gaussians.confidence if len(filtered_gaussians.confidence) > 0 else torch.empty(0)
        else:
            checkpoint_dict['confidence'] = torch.empty(0)
        if hasattr(filtered_gaussians, 'bg_color'):
            checkpoint_dict['bg_color'] = filtered_gaussians.bg_color if len(filtered_gaussians.bg_color) > 0 else torch.empty(0)
        else:
            checkpoint_dict['bg_color'] = torch.empty(0)
        checkpoint_dict['layer_assignments'] = torch.zeros(len(filtered_gaussians._xyz), dtype=torch.long)
        checkpoint_dict['importance'] = torch.zeros(len(filtered_gaussians._xyz))

        torch.save(checkpoint_dict, model_path)

        # Store results
        result = {
            'config_name': config['name'],
            'filter_stats': filter_stats,
            'metrics': filtered_metrics,
            'model_path': model_path
        }

        # Compute performance degradation
        result['psnr_degradation'] = baseline_metrics['psnr_mean'] - filtered_metrics['psnr_mean']
        result['ssim_degradation'] = baseline_metrics['ssim_mean'] - filtered_metrics['ssim_mean']

        results['filter_results'].append(result)

        # Print summary
        print(f"\nResults for {config['name']}:")
        print(f"  PSNR: {filtered_metrics['psnr_mean']:.2f} (degradation: {result['psnr_degradation']:.2f} dB)")
        print(f"  SSIM: {filtered_metrics['ssim_mean']:.4f} (degradation: {result['ssim_degradation']:.4f})")
        print(f"  Model saved to: {model_path}")

    # Save comprehensive results
    results_file = os.path.join(results_dir, 'filtering_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"Original model:")
    print(f"  PSNR: {baseline_metrics['psnr_mean']:.2f} ± {baseline_metrics['psnr_std']:.2f}")
    print(f"  SSIM: {baseline_metrics['ssim_mean']:.4f} ± {baseline_metrics['ssim_std']:.4f}")
    print(f"  Gaussians: {original_gaussians._xyz.shape[0]:,}")

    for result in results['filter_results']:
        print(f"\n{result['config_name']}:")
        print(f"  PSNR: {result['metrics']['psnr_mean']:.2f} (Δ{result['psnr_degradation']:.2f})")
        print(f"  SSIM: {result['metrics']['ssim_mean']:.4f} (Δ{result['ssim_degradation']:.4f})")
        print(f"  Gaussians: {result['filter_stats']['total_filtered']:,} ({result['filter_stats']['filter_percentage']:.1f}% removed)")

    print(f"\nDetailed results saved to: {results_file}")

    return results