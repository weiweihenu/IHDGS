"""
Hierarchical Densification Scheduler for OHDGS
Coordinates densification strategies across different layers
"""

import torch
import time
from utils.cdd_utils import CDDDensifier
from utils.nod_utils_simple import SimpleNODDensifier, apply_nod_densification
from utils.floater_utils import FloaterSuppressor


class HierarchicalDensifier:
    """
    Manages hierarchical densification strategies for different layers
    """

    def __init__(self, device='cuda',
                 # CDD parameters
                 cdd_max_new=500, cdd_percentile=90, cdd_min_error=0.1,
                 cdd_color_threshold=0.05, cdd_depth_threshold=0.02,
                 # NOD parameters
                 nod_max_new=300,
                 # Scheduling parameters
                 cdd_interval=100, nod_interval=200, floater_interval=50,
                 # Time control parameters
                 cdd_start_iter=500, cdd_end_iter=15000,
                 nod_start_iter=1000, nod_end_iter=15000,
                 floater_start_iter=500, floater_end_iter=8000):
        self.device = device

        # Initialize layer-specific densifiers with constraints
        self.cdd_densifier = CDDDensifier(device=device)
        self.nod_densifier = SimpleNODDensifier(device=device)
        self.floater_suppressor = FloaterSuppressor(device=device)

        # Apply CDD constraints
        self.cdd_densifier.max_new_gaussians_per_view = cdd_max_new
        self.cdd_densifier.error_percentile = cdd_percentile
        self.cdd_densifier.min_error_threshold = cdd_min_error

        # Apply CDD basic thresholds
        self.cdd_densifier.color_threshold = cdd_color_threshold
        self.cdd_densifier.depth_threshold = cdd_depth_threshold

        # Store NOD constraint
        self.nod_max_new = nod_max_new

        # Densification schedule (use provided parameters)
        self.cdd_interval = cdd_interval
        self.nod_interval = nod_interval
        self.floater_interval = floater_interval

        # Densification control (use provided parameters)
        self.cdd_start_iter = cdd_start_iter
        self.cdd_end_iter = cdd_end_iter
        self.nod_start_iter = nod_start_iter
        self.nod_end_iter = nod_end_iter
        self.floater_start_iter = floater_start_iter
        self.floater_end_iter = floater_end_iter

        # Statistics
        self.stats = {
            'cdd_added': 0,
            'nod_added': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0,
            'last_cdd_iter': 0,
            'last_nod_iter': 0,
            'last_floater_iter': 0
        }

    def should_densify_cdd(self, iteration):
        """Check if CDD should be applied"""
        return (self.cdd_start_iter <= iteration <= self.cdd_end_iter and
                iteration % self.cdd_interval == 0 and
                iteration != self.stats['last_cdd_iter'])

    def should_densify_nod(self, iteration):
        """Check if NOD should be applied"""
        return (self.nod_start_iter <= iteration <= self.nod_end_iter and
                iteration % self.nod_interval == 0 and
                iteration != self.stats['last_nod_iter'])

    def should_suppress_floaters(self, iteration):
        """Check if floater suppression should be applied"""
        return (self.floater_start_iter <= iteration <= self.floater_end_iter and
                iteration % self.floater_interval == 0 and
                iteration != self.stats['last_floater_iter'])

    def densify_salient_layer(self, gaussians, viewpoint_cam, pipe, background, iteration):
        """
        Apply CDD densification to salient layer with constraints
        """
        if not self.should_densify_cdd(iteration):
            return 0

        try:
            # Get current layer statistics
            stats = gaussians.get_layer_statistics()
            salient_count = stats.get('salient', {}).get('count', 0)

            if salient_count == 0:
                return 0

            # 计算当前总数和允许的最大新增数量（2.5%限制）
            current_total = len(gaussians.get_xyz)
            max_new_total = int(current_total * 0.025)  # 2.5%限制

            # 临时修改CDD的max_new_gaussians_per_view
            original_max = self.cdd_densifier.max_new_gaussians_per_view
            self.cdd_densifier.max_new_gaussians_per_view = min(original_max, max_new_total)

            # Apply CDD from current viewpoint with constraints
            new_params = self.cdd_densifier.densify_salient_layer(
                gaussians, viewpoint_cam, pipe, background
            )

            # 恢复原始值
            self.cdd_densifier.max_new_gaussians_per_view = original_max

            if new_params and isinstance(new_params, dict) and 'xyz' in new_params:
                num_new = new_params['xyz'].shape[0]

                if num_new > 0:
                    # Add to model
                    gaussians.densification_postfix(
                        new_params['xyz'],
                        new_params['features_dc'],
                        new_params['features_rest'],
                        new_params['opacity'],
                        new_params['scaling'],
                        new_params['rotation']
                    )

                # Update importance and layer assignments
                gaussians.compute_importance()
                gaussians.update_layer_assignments()

                self.stats['cdd_added'] += num_new
                self.stats['last_cdd_iter'] = iteration

                print(f"[CDD] Added {num_new} Gaussians to salient layer "
                      f"(total: {len(gaussians.get_xyz)})")
                return num_new

        except Exception as e:
            print(f"[CDD Error] Densification failed: {e}")

        return 0

    def densify_transition_layer(self, gaussians, iteration):
        """
        Apply NOD densification to transition layer
        """
        if not self.should_densify_nod(iteration):
            return 0

        try:
            # 使用简化的NOD实现，传递max_new_gaussians参数
            num_new = apply_nod_densification(gaussians, max_new_gaussians=self.nod_max_new)

            if num_new > 0:
                self.stats['nod_added'] += num_new
                self.stats['last_nod_iter'] = iteration
                print(f"[NOD] Added {num_new} Gaussians to transition layer "
                      f"(total: {len(gaussians.get_xyz)})")
                return num_new

        except Exception as e:
            print(f"[NOD Error] Densification failed: {e}")

        return 0

    def suppress_background_layer(self, gaussians, iteration):
        """
        Apply floater suppression to background layer

        Returns:
            dict: Suppression statistics with keys:
                - floaters_detected
                - floaters_suppressed
                - floaters_pruned
                - total_handled
        """
        if not self.should_suppress_floaters(iteration):
            return {
                'floaters_detected': 0,
                'floaters_suppressed': 0,
                'floaters_pruned': 0,
                'total_handled': 0
            }

        try:
            # Progressive suppression
            suppression_stats = self.floater_suppressor.progressive_suppression(gaussians, iteration)

            self.stats['floaters_suppressed'] += suppression_stats.get('floaters_suppressed', 0)
            self.stats['floaters_pruned'] += suppression_stats.get('floaters_pruned', 0)
            self.stats['last_floater_iter'] = iteration

            if suppression_stats.get('floaters_detected', 0) > 0:
                print(f"[Floater] Detected {suppression_stats['floaters_detected']} floaters, "
                      f"suppressed {suppression_stats['floaters_suppressed']}, "
                      f"pruned {suppression_stats['floaters_pruned']}")

            # Add total_handled field
            suppression_stats['total_handled'] = suppression_stats.get('floaters_suppressed', 0) + suppression_stats.get('floaters_pruned', 0)
            return suppression_stats

        except Exception as e:
            print(f"[Floater Error] Suppression failed: {e}")

        return {
            'floaters_detected': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0,
            'total_handled': 0
        }

    def densify_all_layers(self, gaussians, viewpoint_cam, pipe, background, iteration):
        """
        Apply appropriate densification strategies to all layers

        Args:
            gaussians: GaussianModel instance
            viewpoint_cam: Current camera viewpoint
            pipe: Pipeline parameters
            background: Background color
            iteration: Current iteration

        Returns:
            dict: Summary of densification actions
        """
        summary = {
            'iteration': iteration,
            'cdd_added': 0,
            'nod_added': 0,
            'floaters_handled': 0,
            'floaters_detected': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0,
            'total_gaussians_before': len(gaussians.get_xyz),
            'layer_stats_before': gaussians.get_layer_statistics()
        }

        # Only update layers when we're actually going to run a densification strategy
        should_update_layers = False

        # Check if any densification strategy should run
        if self.should_densify_cdd(iteration) or self.should_densify_nod(iteration) or self.should_suppress_floaters(iteration):
            should_update_layers = True

        # Apply densification strategies only if they are scheduled to run
        cdd_count = 0
        nod_count = 0
        floater_stats = {
            'floaters_detected': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0,
            'total_handled': 0
        }

        if self.should_densify_cdd(iteration):
            cdd_count = self.densify_salient_layer(gaussians, viewpoint_cam, pipe, background, iteration)

        if self.should_densify_nod(iteration):
            nod_count = self.densify_transition_layer(gaussians, iteration)

        if self.should_suppress_floaters(iteration):
            floater_stats = self.suppress_background_layer(gaussians, iteration)

        summary['cdd_added'] = cdd_count
        summary['nod_added'] = nod_count
        summary['floaters_handled'] = floater_stats['total_handled']
        summary['floaters_detected'] = floater_stats['floaters_detected']
        summary['floaters_suppressed'] = floater_stats['floaters_suppressed']
        summary['floaters_pruned'] = floater_stats['floaters_pruned']
        summary['total_gaussians_after'] = len(gaussians.get_xyz)
        summary['layer_stats_after'] = gaussians.get_layer_statistics() if should_update_layers else summary['layer_stats_before']

        # Update cumulative stats
        self.stats['cdd_added'] += cdd_count
        self.stats['nod_added'] += nod_count

        return summary

    def get_densification_report(self):
        """
        Generate a report of densification activities

        Returns:
            str: Formatted report
        """
        report = f"""
Hierarchical Densification Report
=============================
CDD (Salient Layer):
  - Total Gaussians Added: {self.stats['cdd_added']}
  - Interval: Every {self.cdd_interval} iterations
  - Active Range: {self.cdd_start_iter} - {self.cdd_end_iter}

NOD (Transition Layer):
  - Total Gaussians Added: {self.stats['nod_added']}
  - Interval: Every {self.nod_interval} iterations
  - Active Range: {self.nod_start_iter} - {self.nod_end_iter}

Floater Suppression (Background Layer):
  - Total Suppressed: {self.stats['floaters_suppressed']}
  - Total Pruned: {self.stats['floaters_pruned']}
  - Interval: Every {self.floater_interval} iterations
  - Active Range: {self.floater_start_iter} - {self.floater_end_iter}

Last Updates:
  - CDD: Iteration {self.stats['last_cdd_iter']}
  - NOD: Iteration {self.stats['last_nod_iter']}
  - Floater: Iteration {self.stats['last_floater_iter']}
"""
        return report

    def reset_stats(self):
        """Reset densification statistics"""
        self.stats = {
            'cdd_added': 0,
            'nod_added': 0,
            'floaters_suppressed': 0,
            'floaters_pruned': 0,
            'last_cdd_iter': 0,
            'last_nod_iter': 0,
            'last_floater_iter': 0
        }


def apply_hierarchical_densification(gaussians, viewpoint_cam, pipe, background, iteration,
                                    # CDD parameters
                                    cdd_max_new=500, cdd_percentile=90, cdd_min_error=0.1,
                                    cdd_color_threshold=0.05, cdd_depth_threshold=0.02,
                                    # NOD parameters
                                    nod_max_new=300,
                                    # Scheduling parameters
                                    cdd_interval=100, nod_interval=200, floater_interval=50,
                                    # Time control parameters
                                    cdd_start_iter=500, cdd_end_iter=15000,
                                    nod_start_iter=1000, nod_end_iter=15000,
                                    floater_start_iter=500, floater_end_iter=8000):
    """
    Convenience function to apply hierarchical densification with constraints

    Args:
        gaussians: GaussianModel instance
        viewpoint_cam: Camera viewpoint
        pipe: Pipeline parameters
        background: Background color
        iteration: Current iteration
        cdd_max_new: Maximum new Gaussians per CDD call
        cdd_percentile: Error percentile threshold for CDD
        cdd_min_error: Minimum error threshold for CDD
        cdd_color_threshold: Color difference threshold for CDD (τ_c)
        cdd_depth_threshold: Depth difference threshold for CDD (τ_d)
        nod_max_new: Maximum new Gaussians per NOD call
        cdd_interval: CDD densification interval
        nod_interval: NOD densification interval
        floater_interval: Floater suppression interval
        cdd_start_iter: When to start CDD
        cdd_end_iter: When to end CDD
        nod_start_iter: When to start NOD
        nod_end_iter: When to end NOD
        floater_start_iter: When to start floater suppression
        floater_end_iter: When to end floater suppression

    Returns:
        dict: Summary of densification actions
    """
    densifier = HierarchicalDensifier(
        device=gaussians._xyz.device,
        cdd_max_new=cdd_max_new,
        cdd_percentile=cdd_percentile,
        cdd_min_error=cdd_min_error,
        cdd_color_threshold=cdd_color_threshold,
        cdd_depth_threshold=cdd_depth_threshold,
        nod_max_new=nod_max_new,
        cdd_interval=cdd_interval,
        nod_interval=nod_interval,
        floater_interval=floater_interval,
        cdd_start_iter=cdd_start_iter,
        cdd_end_iter=cdd_end_iter,
        nod_start_iter=nod_start_iter,
        nod_end_iter=nod_end_iter,
        floater_start_iter=floater_start_iter,
        floater_end_iter=floater_end_iter
    )
    return densifier.densify_all_layers(gaussians, viewpoint_cam, pipe, background, iteration)


if __name__ == "__main__":
    print("Hierarchical Densification utilities for OHDGS")
    print("Usage: from utils.hierarchical_densification import HierarchicalDensifier")