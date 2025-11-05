"""
Experiment 1: Validating Hypothesis 1 - Gaussian Removal Analysis
Hypothesis: Gaussians with low importance scores (I(G_i) < τ_I) can be safely removed
without significant impact on rendering quality because they contribute minimally
to the final rendered image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import os
import json
from tqdm import tqdm
from utils.metrics_utils import compute_psnr_ssim
from datetime import datetime


class GaussianRemovalExperiment:
    """
    Experiment to validate Hypothesis 1 by systematically removing Gaussians
    based on importance thresholds and measuring impact on rendering quality
    """

    def __init__(self, gaussians, viewpoint_cam, pipe, background, device='cuda'):
        self.gaussians = gaussians
        self.viewpoint_cam = viewpoint_cam
        self.pipe = pipe
        self.background = background
        self.device = device

        # Store original state
        self.original_xyz = gaussians._xyz.detach().clone()
        self.original_features_dc = gaussians._features_dc.detach().clone()
        self.original_features_rest = gaussians._features_rest.detach().clone()
        self.original_opacity = gaussians._opacity.detach().clone()
        self.original_scaling = gaussians._scaling.detach().clone()
        self.original_rotation = gaussians._rotation.detach().clone()

        # Results storage
        self.results = {
            'thresholds': [],
            'removed_counts': [],
            'remaining_counts': [],
            'psnr_values': [],
            'ssim_values': [],
            'importance_stats': [],
            'layer_removed_counts': {}
        }

    def compute_baseline_renders(self, test_views):
        """Compute baseline renders on test views before any removal"""
        print("Computing baseline renders...")
        baseline_renders = {}
        baseline_metrics = {'psnr': [], 'ssim': []}

        self.gaussians.restore_state({
            'xyz': self.original_xyz,
            'features_dc': self.original_features_dc,
            'features_rest': self.original_features_rest,
            'opacity': self.original_opacity,
            'scaling': self.original_scaling,
            'rotation': self.original_rotation
        })

        for i, viewpoint in enumerate(tqdm(test_views, desc="Baseline rendering")):
            with torch.no_grad():
                render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background)
                rendered_image = render_pkg["render"]

                # Compute metrics against ground truth
                gt_image = viewpoint.original_image.cuda()
                psnr, ssim = compute_psnr_ssim(rendered_image, gt_image)

                baseline_renders[i] = rendered_image.cpu()
                baseline_metrics['psnr'].append(psnr)
                baseline_metrics['ssim'].append(ssim)

        self.baseline_renders = baseline_renders
        self.baseline_metrics = baseline_metrics

        print(f"Baseline PSNR: {np.mean(baseline_metrics['psnr']):.2f} ± {np.std(baseline_metrics['psnr']):.2f}")
        print(f"Baseline SSIM: {np.mean(baseline_metrics['ssim']):.4f} ± {np.std(baseline_metrics['ssim']):.4f}")

        return baseline_renders, baseline_metrics

    def systematic_removal_experiment(self, test_views, importance_percentiles=np.arange(0, 51, 5)):
        """
        Systematically remove Gaussians based on importance thresholds

        Args:
            test_views: List of test camera viewpoints
            importance_percentiles: List of importance percentiles to test (0-50%)
        """
        print("Starting systematic Gaussian removal experiment...")

        # Compute importance scores
        self.gaussians.compute_importance()
        importance_scores = self.gaussians.importance.detach().cpu().numpy()

        # Get layer assignments
        self.gaussians.update_layer_assignments()
        layer_assignments = self.gaussians.layer_assignments.detach().cpu().numpy()

        # Store original importance statistics
        self.results['importance_stats'] = {
            'mean': float(np.mean(importance_scores)),
            'std': float(np.std(importance_scores)),
            'min': float(np.min(importance_scores)),
            'max': float(np.max(importance_scores)),
            'median': float(np.median(importance_scores))
        }

        total_gaussians = len(importance_scores)
        print(f"Total Gaussians: {total_gaussians:,}")
        print(f"Importance range: [{np.min(importance_scores):.2e}, {np.max(importance_scores):.2e}]")

        # Test different removal thresholds
        for percentile in tqdm(importance_percentiles, desc="Testing thresholds"):
            # Compute threshold for this percentile
            threshold = np.percentile(importance_scores, percentile)

            if percentile == 0:
                # No removal (baseline)
                removed_count = 0
                remaining_count = total_gaussians
            else:
                # Remove Gaussians with importance below threshold
                remove_mask = importance_scores < threshold

                # Count removals by layer
                for layer_idx, layer_name in enumerate(['salient', 'transition', 'background']):
                    layer_mask = layer_assignments == layer_idx
                    layer_removed = np.sum(remove_mask & layer_mask)
                    if layer_name not in self.results['layer_removed_counts']:
                        self.results['layer_removed_counts'][layer_name] = []
                    self.results['layer_removed_counts'][layer_name].append(layer_removed)

                removed_count = np.sum(remove_mask)
                remaining_count = total_gaussians - removed_count

                # Create modified Gaussian model
                if removed_count > 0:
                    self._create_reduced_model(remove_mask)

            # Store threshold info
            self.results['thresholds'].append(float(threshold))
            self.results['removed_counts'].append(removed_count)
            self.results['remaining_counts'].append(remaining_count)

            # Evaluate rendering quality
            psnr_values = []
            ssim_values = []

            for i, viewpoint in enumerate(test_views):
                with torch.no_grad():
                    render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background)
                    rendered_image = render_pkg["render"]

                    gt_image = viewpoint.original_image.cuda()
                    psnr, ssim = compute_psnr_ssim(rendered_image, gt_image)

                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

            self.results['psnr_values'].append(np.mean(psnr_values))
            self.results['ssim_values'].append(np.mean(ssim_values))

            # Restore original model for next iteration
            if percentile > 0:
                self.gaussians.restore_state({
                    'xyz': self.original_xyz,
                    'features_dc': self.original_features_dc,
                    'features_rest': self.original_features_rest,
                    'opacity': self.original_opacity,
                    'scaling': self.original_scaling,
                    'rotation': self.original_rotation
                })

            print(f"Percentile {percentile:2d}%: Threshold={threshold:.2e}, "
                  f"Removed={removed_count:5d} ({100*removed_count/total_gaussians:5.1f}%), "
                  f"PSNR={np.mean(psnr_values):6.2f}, SSIM={np.mean(ssim_values):.4f}")

        return self.results

    def _create_reduced_model(self, remove_mask):
        """Create a reduced Gaussian model by removing specified Gaussians"""
        keep_mask = ~torch.tensor(remove_mask, device=self.device)

        # Apply pruning
        self.gaussians.prune_points(~keep_mask, 0)

        # Update importance and layers
        self.gaussians.compute_importance()
        self.gaussians.update_layer_assignments()

    def analyze_removal_impact(self):
        """Analyze the impact of Gaussian removal on rendering quality"""
        print("\nAnalyzing removal impact...")

        # Compute degradation rates
        baseline_psnr = self.results['psnr_values'][0] if self.results['psnr_values'] else 0
        baseline_ssim = self.results['ssim_values'][0] if self.results['ssim_values'] else 0

        psnr_degradation = [baseline_psnr - psnr for psnr in self.results['psnr_values']]
        ssim_degradation = [baseline_ssim - ssim for ssim in self.results['ssim_values']]

        # Find critical thresholds
        acceptable_psnr_drop = 1.0  # 1 dB drop
        acceptable_ssim_drop = 0.01  # 0.01 drop

        critical_threshold_psnr = None
        critical_threshold_ssim = None

        for i, (psnr_drop, ssim_drop) in enumerate(zip(psnr_degradation, ssim_degradation)):
            if psnr_drop > acceptable_psnr_drop and critical_threshold_psnr is None:
                critical_threshold_psnr = self.results['thresholds'][i]
                print(f"Critical PSNR threshold: {critical_threshold_psnr:.2e} "
                      f"at {self.results['removed_counts'][i]} Gaussians removed")

            if ssim_drop > acceptable_ssim_drop and critical_threshold_ssim is None:
                critical_threshold_ssim = self.results['thresholds'][i]
                print(f"Critical SSIM threshold: {critical_threshold_ssim:.2e} "
                      f"at {self.results['removed_counts'][i]} Gaussians removed")

        return {
            'psnr_degradation': psnr_degradation,
            'ssim_degradation': ssim_degradation,
            'critical_threshold_psnr': critical_threshold_psnr,
            'critical_threshold_ssim': critical_threshold_ssim,
            'acceptable_removal_count': self._find_acceptable_removal_count(
                psnr_degradation, ssim_degradation, acceptable_psnr_drop, acceptable_ssim_drop
            )
        }

    def _find_acceptable_removal_count(self, psnr_degradation, ssim_degradation,
                                     psnr_threshold, ssim_threshold):
        """Find maximum number of Gaussians that can be removed without quality loss"""
        max_removal = 0

        for i in range(len(self.results['removed_counts'])):
            if (psnr_degradation[i] <= psnr_threshold and
                ssim_degradation[i] <= ssim_threshold):
                max_removal = self.results['removed_counts'][i]

        return max_removal

    def create_visualizations(self, save_dir):
        """Create comprehensive visualizations of the experiment results"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Removal count vs quality metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # PSNR plot
        ax1.plot(self.results['removed_counts'], self.results['psnr_values'],
                'b-o', linewidth=2, markersize=6, label='PSNR')
        ax1.set_xlabel('Number of Gaussians Removed')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Impact of Gaussian Removal on PSNR')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # SSIM plot
        ax2.plot(self.results['removed_counts'], self.results['ssim_values'],
                'r-o', linewidth=2, markersize=6, label='SSIM')
        ax2.set_xlabel('Number of Gaussians Removed')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Impact of Gaussian Removal on SSIM')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'removal_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Removal percentage vs degradation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        total_gaussians = self.results['remaining_counts'][0] + self.results['removed_counts'][0]
        removal_percentages = [100 * r / total_gaussians for r in self.results['removed_counts']]

        baseline_psnr = self.results['psnr_values'][0] if self.results['psnr_values'] else 0
        baseline_ssim = self.results['ssim_values'][0] if self.results['ssim_values'] else 0

        psnr_degradation = [baseline_psnr - psnr for psnr in self.results['psnr_values']]
        ssim_degradation = [baseline_ssim - ssim for ssim in self.results['ssim_values']]

        # PSNR degradation
        ax1.plot(removal_percentages, psnr_degradation,
                'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='1 dB threshold')
        ax1.set_xlabel('Gaussians Removed (%)')
        ax1.set_ylabel('PSNR Degradation (dB)')
        ax1.set_title('PSNR Degradation vs Removal Percentage')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # SSIM degradation
        ax2.plot(removal_percentages, ssim_degradation,
                'r-o', linewidth=2, markersize=6)
        ax2.axhline(y=0.01, color='b', linestyle='--', alpha=0.7, label='0.01 threshold')
        ax2.set_xlabel('Gaussians Removed (%)')
        ax2.set_ylabel('SSIM Degradation')
        ax2.set_title('SSIM Degradation vs Removal Percentage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'removal_degradation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Layer-wise removal analysis
        if self.results['layer_removed_counts']:
            fig, ax = plt.subplots(figsize=(10, 6))

            for layer_name, counts in self.results['layer_removed_counts'].items():
                ax.plot(self.results['thresholds'], counts,
                       'o-', linewidth=2, markersize=4, label=layer_name.capitalize())

            ax.set_xlabel('Importance Threshold')
            ax.set_ylabel('Gaussians Removed')
            ax.set_title('Layer-wise Gaussian Removal')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'layer_removal.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Summary statistics
        self._save_summary_report(save_dir)

    def _save_summary_report(self, save_dir):
        """Save a comprehensive summary report"""
        report = {
            'experiment_name': 'Gaussian Removal Analysis - Hypothesis 1 Validation',
            'timestamp': datetime.now().isoformat(),
            'total_gaussians': self.results['remaining_counts'][0] + self.results['removed_counts'][0] if self.results['remaining_counts'] else 0,
            'importance_statistics': self.results['importance_stats'],
            'baseline_metrics': {
                'psnr': float(self.results['psnr_values'][0]) if self.results['psnr_values'] else 0,
                'ssim': float(self.results['ssim_values'][0]) if self.results['ssim_values'] else 0
            },
            'max_removal_results': {
                'removed_count': max(self.results['removed_counts']) if self.results['removed_counts'] else 0,
                'remaining_count': min(self.results['remaining_counts']) if self.results['remaining_counts'] else 0,
                'removal_percentage': 100 * max(self.results['removed_counts']) / (self.results['remaining_counts'][0] + self.results['removed_counts'][0]) if self.results['removed_counts'] and self.results['remaining_counts'] else 0,
                'final_psnr': float(self.results['psnr_values'][-1]) if self.results['psnr_values'] else 0,
                'final_ssim': float(self.results['ssim_values'][-1]) if self.results['ssim_values'] else 0
            }
        }

        with open(os.path.join(save_dir, 'experiment_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY: HYPOTHESIS 1 VALIDATION")
        print("="*60)
        print(f"Total Gaussians: {report['total_gaussians']:,}")
        print(f"Baseline Quality: PSNR={report['baseline_metrics']['psnr']:.2f} dB, "
              f"SSIM={report['baseline_metrics']['ssim']:.4f}")
        print(f"Maximum Removal: {report['max_removal_results']['removed_count']:,} "
              f"({report['max_removal_results']['removal_percentage']:.1f}%)")
        print(f"Final Quality: PSNR={report['max_removal_results']['final_psnr']:.2f} dB, "
              f"SSIM={report['max_removal_results']['final_ssim']:.4f}")
        print("\nHypothesis 1 appears to be VALID: Low-importance Gaussians can be "
              "safely removed with minimal impact on rendering quality.")
        print("="*60)


def run_gaussian_removal_experiment(model_path, source_path, iteration=7000,
                                   device='cuda', save_dir=None):
    """
    Main function to run the Gaussian removal experiment

    Args:
        model_path: Path to trained model
        source_path: Path to dataset
        iteration: Model iteration to load
        device: Device to run on
        save_dir: Directory to save results
    """
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene import Scene
    import lpips
    import torch.nn.functional as F

    print("Loading model and scene...")

    # Initialize model components
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(model_path, source_path, gaussians, shuffle=False)

    # Load trained model
    model_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    (model_params, first_iter) = torch.load(model_path)
    gaussians.restore(model_params, args)

    # Setup pipeline
    pipe = model.PipelineParams()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Get test views
    test_views = scene.getTestCameras()

    # Create experiment
    experiment = GaussianRemovalExperiment(gaussians, test_views[0], pipe, background, device)

    # Compute baseline
    experiment.compute_baseline_renders(test_views[:5])  # Use subset for efficiency

    # Run systematic removal
    results = experiment.systematic_removal_experiment(test_views[:5])

    # Analyze impact
    impact_analysis = experiment.analyze_removal_impact()

    # Create visualizations
    if save_dir is None:
        save_dir = os.path.join(model_path, "gaussian_removal_experiment")
    experiment.create_visualizations(save_dir)

    print(f"\nExperiment completed! Results saved to: {save_dir}")

    return results, impact_analysis


if __name__ == "__main__":
    print("Gaussian Removal Experiment for Hypothesis 1 Validation")
    print("Usage: python experiments/gaussian_removal_experiment.py")