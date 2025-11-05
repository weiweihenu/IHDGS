"""
Importance Space Visualization for OHDGS
Provides comprehensive visualization tools for analyzing Gaussian importance
distribution and layer assignments in importance space (I, ω, V)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scene.gaussian_model import GaussianModel
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime


class ImportanceSpaceVisualizer:
    """
    Comprehensive visualization toolkit for importance space analysis
    """

    def __init__(self, gaussians, device='cuda'):
        self.gaussians = gaussians
        self.device = device

        # Ensure importance and layers are computed
        self.gaussians.compute_importance()
        self.gaussians.update_layer_assignments()

        # Extract data for visualization
        self.importance = self.gaussians.importance.cpu().numpy()
        self.opacity = self.gaussians.get_opacity.squeeze().cpu().numpy()
        self.volume = self.gaussians.compute_volume().cpu().numpy()
        self.layers = self.gaussians.layer_assignments.cpu().numpy()
        self.xyz = self.gaussians.get_xyz.cpu().numpy()

        # Layer colors and names
        self.layer_names = ['salient', 'transition', 'background']
        self.layer_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        self.layer_colors_rgb = [
            (1.0, 0.42, 0.42),  # Red
            (0.31, 0.8, 0.77),  # Teal
            (0.27, 0.72, 0.82)   # Blue
        ]

    def create_importance_space_3d(self, save_path=None, interactive=False):
        """
        Create 3D visualization of importance space (I, ω, V)

        Args:
            save_path: Path to save visualization
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            return self._create_interactive_3d(save_path)
        else:
            return self._create_static_3d(save_path)

    def _create_static_3d(self, save_path=None):
        """Create static 3D scatter plot"""
        fig = plt.figure(figsize=(15, 12))

        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Plot each layer with different colors
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                ax.scatter(self.importance[layer_mask],
                          self.opacity[layer_mask],
                          self.volume[layer_mask],
                          c=color, s=2, alpha=0.6, label=layer_name.capitalize())

        ax.set_xlabel('Importance (I)')
        ax.set_ylabel('Opacity (ω)')
        ax.set_zlabel('Volume (V)')
        ax.set_title('Importance Space Distribution by Layer')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_zscale('log')

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Importance space 3D visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def _create_interactive_3d(self, save_path=None):
        """Create interactive 3D plotly visualization"""
        fig = go.Figure()

        # Add each layer as separate trace
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                fig.add_trace(go.Scatter3d(
                    x=self.importance[layer_mask],
                    y=self.opacity[layer_mask],
                    z=self.volume[layer_mask],
                    mode='markers',
                    name=layer_name.capitalize(),
                    marker=dict(
                        size=2,
                        color=color,
                        opacity=0.6
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Importance: %{x:.2e}<br>' +
                                 'Opacity: %{y:.4f}<br>' +
                                 'Volume: %{z:.2e}<extra></extra>'
                ))

        fig.update_layout(
            title='Importance Space Distribution by Layer',
            scene=dict(
                xaxis_title='Importance (I)',
                xaxis_type='log',
                yaxis_title='Opacity (ω)',
                yaxis_type='log',
                zaxis_title='Volume (V)',
                zaxis_type='log',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_html(save_path + '.html')
            print(f"Interactive importance space visualization saved to {save_path}")
        else:
            fig.show()

        return fig

    def create_2d_projections(self, save_dir=None):
        """
        Create 2D projections of importance space

        Args:
            save_dir: Directory to save plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Importance Space 2D Projections', fontsize=16)

        # I vs ω
        ax1 = axes[0, 0]
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                ax1.scatter(self.importance[layer_mask], self.opacity[layer_mask],
                          c=color, s=1, alpha=0.6, label=layer_name.capitalize())
        ax1.set_xlabel('Importance (I)')
        ax1.set_ylabel('Opacity (ω)')
        ax1.set_title('Importance vs Opacity')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # I vs V
        ax2 = axes[0, 1]
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                ax2.scatter(self.importance[layer_mask], self.volume[layer_mask],
                          c=color, s=1, alpha=0.6, label=layer_name.capitalize())
        ax2.set_xlabel('Importance (I)')
        ax2.set_ylabel('Volume (V)')
        ax2.set_title('Importance vs Volume')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ω vs V
        ax3 = axes[1, 0]
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                ax3.scatter(self.opacity[layer_mask], self.volume[layer_mask],
                          c=color, s=1, alpha=0.6, label=layer_name.capitalize())
        ax3.set_xlabel('Opacity (ω)')
        ax3.set_ylabel('Volume (V)')
        ax3.set_title('Opacity vs Volume')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # I vs (ω * V^-λ) - showing the importance calculation
        ax4 = axes[1, 1]
        lambda_param = getattr(self.gaussians, 'importance_lambda', 1.0)
        importance_calc = self.opacity * (self.volume ** (-lambda_param))
        scatter = ax4.scatter(self.importance, importance_calc,
                             c=self.layers, s=1, alpha=0.6, cmap='viridis')
        ax4.plot([min(self.importance), max(self.importance)],
                [min(self.importance), max(self.importance)],
                'r--', alpha=0.5, label='I = ω * V^(-λ)')
        ax4.set_xlabel('Computed Importance (I)')
        ax4.set_ylabel('Importance Formula (ω * V^(-λ))')
        ax4.set_title(f'Importance Validation (λ={lambda_param})')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Layer')

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'importance_space_2d_projections.png'),
                       dpi=300, bbox_inches='tight')
            print(f"2D projections saved to {save_dir}")
        else:
            plt.show()
        plt.close()

    def create_distribution_analysis(self, save_dir=None):
        """
        Create distribution analysis plots for each layer

        Args:
            save_dir: Directory to save plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Layer-wise Distribution Analysis', fontsize=16)

        metrics = ['Importance', 'Opacity', 'Volume']
        data = [self.importance, self.opacity, self.volume]
        log_scale = [True, False, True]

        for i, (metric, metric_data, use_log) in enumerate(zip(metrics, data, log_scale)):
            for j, layer_name in enumerate(self.layer_names):
                ax = axes[i, j]
                layer_mask = self.layers == j

                if np.any(layer_mask):
                    # Histogram
                    ax.hist(metric_data[layer_mask], bins=50, alpha=0.7,
                           color=self.layer_colors[j], edgecolor='black')
                    ax.set_title(f'{layer_name.capitalize()} Layer {metric}')
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Count')

                    if use_log:
                        ax.set_xscale('log')

                    # Add statistics
                    mean_val = np.mean(metric_data[layer_mask])
                    median_val = np.median(metric_data[layer_mask])
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                              label=f'Mean: {mean_val:.2e}')
                    ax.axvline(median_val, color='blue', linestyle='--', alpha=0.7,
                              label=f'Median: {median_val:.2e}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'layer_distribution_analysis.png'),
                       dpi=300, bbox_inches='tight')
            print(f"Distribution analysis saved to {save_dir}")
        else:
            plt.show()
        plt.close()

    def create_density_heatmaps(self, save_dir=None):
        """
        Create density heatmaps of importance space

        Args:
            save_dir: Directory to save plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Importance Space Density Heatmaps', fontsize=16)

        # I vs ω density
        ax1 = axes[0, 0]
        hb1 = ax1.hexbin(self.importance, self.opacity, gridsize=50,
                        cmap='viridis', bins='log')
        ax1.set_xlabel('Importance (I)')
        ax1.set_ylabel('Opacity (ω)')
        ax1.set_title('Importance vs Opacity Density')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.colorbar(hb1, ax=ax1, label='Log Count')

        # I vs V density
        ax2 = axes[0, 1]
        hb2 = ax2.hexbin(self.importance, self.volume, gridsize=50,
                        cmap='viridis', bins='log')
        ax2.set_xlabel('Importance (I)')
        ax2.set_ylabel('Volume (V)')
        ax2.set_title('Importance vs Volume Density')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        plt.colorbar(hb2, ax=ax2, label='Log Count')

        # ω vs V density
        ax3 = axes[1, 0]
        hb3 = ax3.hexbin(self.opacity, self.volume, gridsize=50,
                        cmap='viridis', bins='log')
        ax3.set_xlabel('Opacity (ω)')
        ax3.set_ylabel('Volume (V)')
        ax3.set_title('Opacity vs Volume Density')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        plt.colorbar(hb3, ax=ax3, label='Log Count')

        # Layer-wise density
        ax4 = axes[1, 1]
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                sns.kdeplot(x=np.log10(self.importance[layer_mask]),
                          y=np.log10(self.opacity[layer_mask]),
                          cmap='Reds' if layer_idx == 0 else 'Blues' if layer_idx == 1 else 'Greens',
                          alpha=0.5, ax=ax4, label=layer_name.capitalize())
        ax4.set_xlabel('log10(Importance)')
        ax4.set_ylabel('log10(Opacity)')
        ax4.set_title('Layer-wise Density (Log Scale)')
        ax4.legend()

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'importance_space_density.png'),
                       dpi=300, bbox_inches='tight')
            print(f"Density heatmaps saved to {save_dir}")
        else:
            plt.show()
        plt.close()

    def create_temporal_evolution(self, save_dir=None, iterations=None):
        """
        Create temporal evolution visualization (placeholder for future extension)

        Args:
            save_dir: Directory to save plots
            iterations: List of iterations to analyze
        """
        # This would require storing importance data across training iterations
        # For now, create a placeholder showing current state
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Show current importance distribution
        for layer_idx, (layer_name, color) in enumerate(zip(self.layer_names, self.layer_colors)):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                ax.hist(self.importance[layer_mask], bins=30, alpha=0.6,
                       color=color, label=f'{layer_name.capitalize()} (Current)')

        ax.set_xlabel('Importance')
        ax.set_ylabel('Count')
        ax.set_title('Importance Distribution at Current Iteration')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'importance_temporal_evolution.png'),
                       dpi=300, bbox_inches='tight')
            print(f"Temporal evolution visualization saved to {save_dir}")
        else:
            plt.show()
        plt.close()

    def create_comprehensive_report(self, save_dir=None):
        """
        Create comprehensive importance space analysis report

        Args:
            save_dir: Directory to save all visualizations
        """
        if save_dir is None:
            save_dir = f"importance_space_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(save_dir, exist_ok=True)

        print("Creating comprehensive importance space analysis...")

        # Create all visualizations
        self.create_importance_space_3d(os.path.join(save_dir, 'importance_space_3d.png'))
        self.create_2d_projections(save_dir)
        self.create_distribution_analysis(save_dir)
        self.create_density_heatmaps(save_dir)
        self.create_temporal_evolution(save_dir)

        # Create interactive visualization
        interactive_path = os.path.join(save_dir, 'importance_space_3d_interactive')
        self.create_importance_space_3d(interactive_path, interactive=True)

        # Save summary statistics
        self._save_summary_statistics(save_dir)

        print(f"\nComprehensive importance space analysis completed!")
        print(f"All visualizations saved to: {save_dir}")

        return save_dir

    def _save_summary_statistics(self, save_dir):
        """Save summary statistics to JSON file"""
        import json

        stats = {
            'total_gaussians': len(self.importance),
            'layer_statistics': {},
            'global_statistics': {
                'importance': {
                    'mean': float(np.mean(self.importance)),
                    'std': float(np.std(self.importance)),
                    'min': float(np.min(self.importance)),
                    'max': float(np.max(self.importance)),
                    'median': float(np.median(self.importance))
                },
                'opacity': {
                    'mean': float(np.mean(self.opacity)),
                    'std': float(np.std(self.opacity)),
                    'min': float(np.min(self.opacity)),
                    'max': float(np.max(self.opacity)),
                    'median': float(np.median(self.opacity))
                },
                'volume': {
                    'mean': float(np.mean(self.volume)),
                    'std': float(np.std(self.volume)),
                    'min': float(np.min(self.volume)),
                    'max': float(np.max(self.volume)),
                    'median': float(np.median(self.volume))
                }
            }
        }

        # Layer-wise statistics
        for layer_idx, layer_name in enumerate(self.layer_names):
            layer_mask = self.layers == layer_idx
            if np.any(layer_mask):
                stats['layer_statistics'][layer_name] = {
                    'count': int(np.sum(layer_mask)),
                    'percentage': 100.0 * np.sum(layer_mask) / len(self.importance),
                    'importance': {
                        'mean': float(np.mean(self.importance[layer_mask])),
                        'std': float(np.std(self.importance[layer_mask])),
                        'min': float(np.min(self.importance[layer_mask])),
                        'max': float(np.max(self.importance[layer_mask])),
                        'median': float(np.median(self.importance[layer_mask]))
                    },
                    'opacity': {
                        'mean': float(np.mean(self.opacity[layer_mask])),
                        'std': float(np.std(self.opacity[layer_mask])),
                        'min': float(np.min(self.opacity[layer_mask])),
                        'max': float(np.max(self.opacity[layer_mask])),
                        'median': float(np.median(self.opacity[layer_mask]))
                    },
                    'volume': {
                        'mean': float(np.mean(self.volume[layer_mask])),
                        'std': float(np.std(self.volume[layer_mask])),
                        'min': float(np.min(self.volume[layer_mask])),
                        'max': float(np.max(self.volume[layer_mask])),
                        'median': float(np.median(self.volume[layer_mask]))
                    }
                }

        with open(os.path.join(save_dir, 'importance_space_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("IMPORTANCE SPACE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Gaussians: {stats['total_gaussians']:,}")
        for layer_name, layer_stats in stats['layer_statistics'].items():
            print(f"\n{layer_name.capitalize()} Layer:")
            print(f"  Count: {layer_stats['count']:,} ({layer_stats['percentage']:.1f}%)")
            print(f"  Importance: {layer_stats['importance']['median']:.2e} (median)")
            print(f"  Opacity: {layer_stats['opacity']['median']:.4f} (median)")
            print(f"  Volume: {layer_stats['volume']['median']:.2e} (median)")
        print("="*60)


def analyze_importance_space(gaussians, save_dir=None, interactive=False):
    """
    Convenience function to analyze importance space

    Args:
        gaussians: GaussianModel instance
        save_dir: Directory to save visualizations
        interactive: Whether to create interactive plots

    Returns:
        ImportanceSpaceVisualizer instance
    """
    visualizer = ImportanceSpaceVisualizer(gaussians)

    if save_dir:
        visualizer.create_comprehensive_report(save_dir)
    else:
        # Just show basic visualizations
        visualizer.create_importance_space_3d(interactive=interactive)
        visualizer.create_2d_projections()

    return visualizer


if __name__ == "__main__":
    print("Importance Space Visualization for OHDGS")
    print("Usage: from utils.importance_space_visualization import ImportanceSpaceVisualizer, analyze_importance_space")