"""
Floating Artifact Detection and Suppression for OHDGS Background Layer
Implements Section 3.4.3: Background Layer Strategy
"""

import torch
import numpy as np


class FloaterDetector:
    """
    Detects and suppresses floating artifacts in the background layer
    """

    def __init__(self, device='cuda'):
        self.device = device
        # Thresholds from Observation 1
        self.opacity_percentile = 90  # τ_ω
        self.volume_percentile = 90   # τ_V
        self.importance_percentile = 50  # Median importance threshold

    def detect_floaters(self, gaussians):
        """
        Detect floating artifacts based on Observation 1

        Args:
            gaussians: GaussianModel instance

        Returns:
            floater_mask: Boolean mask indicating floaters
            floater_indices: Indices of floating artifacts
        """
        if len(gaussians.get_xyz) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device), []

        # Get background layer Gaussians
        background_indices = gaussians.get_layer_gaussians(2)  # 2 = background
        if len(background_indices) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device), []

        # Extract background layer properties
        opacity = gaussians.get_opacity[background_indices].squeeze()
        volume = gaussians.compute_volume()[background_indices]
        importance = gaussians.importance[background_indices]

        # Compute thresholds
        omega_threshold = torch.quantile(opacity, self.opacity_percentile / 100.0)
        volume_threshold = torch.quantile(volume, self.volume_percentile / 100.0)

        # Extract scalar values from PyTorch return types
        omega_threshold = omega_threshold.item() if hasattr(omega_threshold, 'item') else omega_threshold
        volume_threshold = volume_threshold.item() if hasattr(volume_threshold, 'item') else volume_threshold

        floater_mask = (opacity > omega_threshold) & (volume > volume_threshold)

        # Additional check: low importance (characteristic of floaters)
        importance_threshold = torch.quantile(gaussians.importance, self.importance_percentile / 100.0)
        importance_threshold = importance_threshold.item() if hasattr(importance_threshold, 'item') else importance_threshold
        floater_mask = floater_mask & (importance < importance_threshold)

        # Get global indices
        all_mask = torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device=self.device)
        all_mask[background_indices] = floater_mask

        floater_indices = torch.where(all_mask)[0]

        return all_mask, floater_indices

    def analyze_floater_characteristics(self, gaussians):
        """
        Analyze characteristics of detected floaters

        Args:
            gaussians: GaussianModel instance

        Returns:
            dict: Floater statistics
        """
        _, floater_indices = self.detect_floaters(gaussians)

        if len(floater_indices) == 0:
            return {
                'count': 0,
                'percentage': 0.0,
                'mean_opacity': 0.0,
                'mean_volume': 0.0,
                'mean_importance': 0.0
            }

        total_gaussians = len(gaussians.get_xyz)

        # Extract floater properties
        opacity = gaussians.get_opacity[floater_indices].squeeze()
        volume = gaussians.compute_volume()[floater_indices]
        importance = gaussians.importance[floater_indices]
        scaling = gaussians.get_scaling[floater_indices]

        return {
            'count': len(floater_indices),
            'percentage': 100.0 * len(floater_indices) / total_gaussians,
            'mean_opacity': opacity.mean().item(),
            'mean_volume': volume.mean().item(),
            'mean_importance': importance.mean().item(),
            'opacity_std': opacity.std().item(),
            'volume_std': volume.std().item(),
            'importance_std': importance.std().item(),
            'mean_scale': scaling.mean(dim=1).tolist(),
            'scale_std': scaling.std(dim=1).tolist()
        }

    def visualize_floater_distribution(self, gaussians, save_path=None):
        """
        Visualize floater distribution in (ω, V) space

        Args:
            gaussians: GaussianModel instance
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        if len(gaussians.get_xyz) == 0:
            print("[Floater Detector] No Gaussians to analyze")
            return

        opacity = gaussians.get_opacity.squeeze().cpu().numpy()
        volume = gaussians.compute_volume().cpu().numpy()
        importance = gaussians.importance.cpu().numpy()

        # Detect floaters
        floater_mask, _ = self.detect_floaters(gaussians)
        floater_mask = floater_mask.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: All Gaussians with floaters highlighted
        axes[0].scatter(opacity[~floater_mask], volume[~floater_mask],
                       c='blue', s=1, alpha=0.5, label='Normal')
        axes[0].scatter(opacity[floater_mask], volume[floater_mask],
                       c='red', s=10, alpha=0.8, label='Floaters')
        axes[0].set_xlabel('Opacity (ω)')
        axes[0].set_ylabel('Volume (V)')
        axes[0].set_title('Gaussian Distribution with Floaters')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Color by importance
        scatter = axes[1].scatter(opacity, volume, c=importance, s=1, alpha=0.6, cmap='viridis')
        axes[1].scatter(opacity[floater_mask], volume[floater_mask],
                       c='red', s=10, alpha=0.8, edgecolors='black', linewidths=0.5)
        axes[1].set_xlabel('Opacity (ω)')
        axes[1].set_ylabel('Volume (V)')
        axes[1].set_title('Colored by Importance (Red = Floaters)')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1], label='Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Floater Detector] Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

        # Print statistics
        stats = self.analyze_floater_characteristics(gaussians)
        print(f"\nFloater Statistics:")
        print(f"  Count: {stats['count']:,} ({stats['percentage']:.1f}%)")
        print(f"  Mean Opacity: {stats['mean_opacity']:.4f} ± {stats['opacity_std']:.4f}")
        print(f"  Mean Volume: {stats['mean_volume']:.2e} ± {stats['volume_std']:.2e}")
        print(f"  Mean Importance: {stats['mean_importance']:.6f} ± {stats['importance_std']:.6f}")


class FloaterSuppressor:
    """
    Suppresses floating artifacts in the background layer
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.detector = FloaterDetector(device=device)
        # Suppression parameters
        self.max_scale = 0.05      # Maximum scale after suppression
        self.opacity_lr_factor = 0.5  # Learning rate reduction factor for opacity
        self.start_iteration = 500
        self.end_iteration = 5000

    def suppress_floaters(self, gaussians, iteration):
        """
        Apply suppression strategies to detected floaters

        Args:
            gaussians: GaussianModel instance
            iteration: Current training iteration

        Returns:
            Number of suppressed floaters
        """
        if iteration < self.start_iteration or iteration > self.end_iteration:
            return 0

        # Detect floaters
        floater_mask, floater_indices = self.detector.detect_floaters(gaussians)

        if len(floater_indices) == 0:
            return 0

        num_suppressed = 0

        # Strategy 1: Scale clamping (prevent unbounded growth)
        if hasattr(gaussians, '_scaling'):
            scaling = gaussians.get_scaling
            with torch.no_grad():
                # Clamp scaling to maximum value
                clamped_scaling = torch.clamp(scaling[floater_indices], 0.001, self.max_scale)
                gaussians._scaling[floater_indices] = torch.log(clamped_scaling)
            num_suppressed += len(floater_indices)

        # Strategy 2: Reduce opacity learning rate (slower opacity increase)
        if hasattr(gaussians, 'optimizer'):
            # Find optimizer parameter group for opacity
            for group in gaussians.optimizer.param_groups:
                if group.get("name") == "opacity":
                    # Get opacity parameter indices
                    for param in group['params']:
                        if hasattr(param, 'grad') and param.grad is not None:
                            # Find which parameters correspond to floaters
                            if hasattr(gaussians, '_opacity'):
                                opacity_param = gaussians._opacity
                                # Check if this parameter is for a floater
                                if torch.any(torch.isclose(param, opacity_param[floater_indices])):
                                    # Reduce gradient for floater parameters
                                    param.grad *= self.opacity_lr_factor
                                    num_suppressed += 1
                    break

        return num_suppressed

    def aggressive_pruning(self, gaussians, iteration, prune_interval=2000):
        """
        Aggressively prune persistent floaters

        Args:
            gaussians: GaussianModel instance
            iteration: Current training iteration
            prune_interval: Interval between pruning operations

        Returns:
            Number of pruned floaters
        """
        if iteration % prune_interval != 0:
            return 0

        # Detect floaters
        floater_mask, floater_indices = self.detector.detect_floaters(gaussians)

        if len(floater_indices) == 0:
            return 0

        # Additional filtering: remove floaters with very low importance
        background_indices = gaussians.get_layer_gaussians(2)
        importance = gaussians.importance[background_indices]
        importance_threshold = torch.quantile(importance, 10.0 / 100.0)  # Bottom 10%
        importance_threshold = float(importance_threshold)

        # Find floaters with very low importance
        low_importance_floaters = []
        for idx in range(len(background_indices)):
            if floater_mask[background_indices[idx]] and importance[idx] < importance_threshold:
                low_importance_floaters.append(background_indices[idx])

        if low_importance_floaters:
            # Create mask for pruning
            prune_mask = torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device=self.device)
            prune_mask[low_importance_floaters] = True

            # Apply pruning
            gaussians.prune_points(prune_mask, iteration)

            return len(low_importance_floaters)

        return 0

    def progressive_suppression(self, gaussians, iteration):
        """
        Progressive suppression with varying intensity

        Args:
            gaussians: GaussianModel instance
            iteration: Current training iteration

        Returns:
            dict: Suppression statistics
        """
        stats = {
            'floaters_detected': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0
        }

        # Detect floaters
        _, floater_indices = self.detector.detect_floaters(gaussians)
        stats['floaters_detected'] = len(floater_indices)

        # Progressive suppression intensity
        progress = (iteration - self.start_iteration) / (self.end_iteration - self.start_iteration)
        progress = max(0, min(1, progress))

        # Scale clamping with progressive intensity
        max_scale = 0.05 * (1 - progress) + 0.01 * progress  # Start strict, relax later

        if hasattr(gaussians, '_scaling') and len(floater_indices) > 0:
            scaling = gaussians.get_scaling
            with torch.no_grad():
                clamped_scaling = torch.clamp(scaling[floater_indices], 0.001, max_scale)
                old_scaling = gaussians._scaling[floater_indices]
                gaussians._scaling[floater_indices] = torch.log(clamped_scaling)

                # Count how many were actually clamped
                diff = torch.any(old_scaling != gaussians._scaling[floater_indices], dim=1)
                suppressed_count = diff.sum()
                # sum() already returns a scalar tensor, convert to int directly
                stats['floaters_suppressed'] = int(suppressed_count)

        # Progressive pruning (more aggressive towards end)
        if iteration > self.end_iteration * 0.7:  # Start pruning after 70% of suppression phase
            prune_threshold = 0.01 + 0.04 * progress  # Increase threshold over time
            background_indices = gaussians.get_layer_gaussians(2)

            # Find floaters with very large scales
            if len(background_indices) > 0:
                scaling = gaussians.get_scaling[background_indices]
                scaling_max_values = scaling.max(dim=1).values  # Get max values directly
                large_scale_mask = (scaling_max_values > prune_threshold)

                # Count large scale floaters
                large_scale_count = int(large_scale_mask.sum())

                if large_scale_count > 0:
                    prune_indices = background_indices[large_scale_mask]
                    prune_mask = torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device=self.device)
                    prune_mask[prune_indices] = True

                    # Keep only if high importance
                    importance = gaussians.importance[prune_indices]
                    importance_quantile = torch.quantile(importance, 20.0 / 100.0)
                    # Extract quantile value properly
                    importance_quantile = float(importance_quantile)
                    high_importance_mask = importance > importance_quantile

                    # Count values properly
                    high_count = int(high_importance_mask.sum())
                    total_count = int(large_scale_mask.sum())

                    if high_count < total_count:
                        final_prune_indices = prune_indices[~high_importance_mask]
                        stats['floaters_pruned'] = len(final_prune_indices)

                        # Actually prune
                        gaussians.prune_points(
                            torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device=self.device),
                            iteration
                        )
                        # Manually remove the specified indices
                        self._manual_prune(gaussians, final_prune_indices)

        return stats

    def _manual_prune(self, gaussians, indices):
        """Manual pruning of specific Gaussian indices"""
        if len(indices) == 0:
            return

        # Create mask to keep all except specified indices
        mask = torch.ones(len(gaussians.get_xyz), dtype=torch.bool, device=self.device)
        mask[indices] = False

        # Prune
        gaussians.prune_points(~mask, 0)  # iteration=0 to force pruning


def detect_and_analyze_floaters(gaussians, save_path=None):
    """
    Convenience function to detect and analyze floaters

    Args:
        gaussians: GaussianModel instance
        save_path: Path to save visualization

    Returns:
        dict: Floater statistics
    """
    detector = FloaterDetector(device=gaussians._xyz.device)
    detector.visualize_floater_distribution(gaussians, save_path)
    return detector.analyze_floater_characteristics(gaussians)


if __name__ == "__main__":
    print("Floating Artifact Detection utilities for OHDGS")
    print("Usage: from utils.floater_utils import FloaterDetector, FloaterSuppressor")