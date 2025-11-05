#
# OHDGS Visualization Utilities
# Author: Your Name
# Date: 2025
#

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_importance_distribution(gaussians, save_path=None, show=True):
    """
    Plot the distribution of importance scores across all Gaussians

    Args:
        gaussians: GaussianModel instance
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if len(gaussians.importance) == 0:
        print("No importance data available. Computing importance...")
        gaussians.compute_importance()

    importance = gaussians.importance.cpu().numpy()
    layer_assignments = gaussians.layer_assignments.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Overall importance distribution
    axes[0, 0].hist(importance, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Overall Importance Distribution')
    axes[0, 0].axvline(np.mean(importance), color='red', linestyle='--',
                       label=f'Mean: {np.mean(importance):.4f}')
    axes[0, 0].legend()

    # Plot 2: Layer-specific importance distributions
    layer_names = ['Salient', 'Transition', 'Background']
    colors = ['green', 'blue', 'gray']

    for i, (name, color) in enumerate(zip(layer_names, colors)):
        mask = layer_assignments == i
        if mask.sum() > 0:
            axes[0, 1].hist(importance[mask], bins=30, alpha=0.6,
                           label=f'{name} ({mask.sum()})', color=color)

    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Layer-specific Importance Distributions')
    axes[0, 1].legend()

    # Plot 3: Importance percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    importance_percentiles = [np.percentile(importance, p) for p in percentiles]

    axes[1, 0].bar(range(len(percentiles)), importance_percentiles,
                   color='lightcoral', alpha=0.7)
    axes[1, 0].set_xticks(range(len(percentiles)))
    axes[1, 0].set_xticklabels([f'{p}%' for p in percentiles])
    axes[1, 0].set_ylabel('Importance Value')
    axes[1, 0].set_title('Importance Percentiles')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Layer composition pie chart
    layer_counts = [np.sum(layer_assignments == i) for i in range(3)]
    total = sum(layer_counts)
    layer_percentages = [100 * count / total for count in layer_counts]

    axes[1, 1].pie(layer_percentages, labels=layer_names, colors=colors,
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Layer Composition')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved importance distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_omega_volume_scatter(gaussians, save_path=None, show=True, use_plotly=True):
    """
    Plot Gaussians in the (ω, V) importance space

    Args:
        gaussians: GaussianModel instance
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        use_plotly: Use plotly for interactive visualization
    """
    if len(gaussians.importance) == 0:
        gaussians.compute_importance()

    opacity = gaussians.get_opacity.squeeze().cpu().numpy()
    volume = gaussians.compute_volume().cpu().numpy()
    importance = gaussians.importance.cpu().numpy()
    layer_assignments = gaussians.layer_assignments.cpu().numpy()

    # Take logarithm for better visualization
    log_volume = np.log(volume + 1e-10)
    log_importance = np.log(importance + 1e-10)

    layer_names = ['Salient', 'Transition', 'Background']
    colors = ['green', 'blue', 'gray']

    if use_plotly:
        # Create interactive plotly visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Linear Scale', 'Log Scale'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )

        for i, (name, color) in enumerate(zip(layer_names, colors)):
            mask = layer_assignments == i
            if mask.sum() > 0:
                # Linear scale plot
                fig.add_trace(
                    go.Scatter(
                        x=opacity[mask],
                        y=volume[mask],
                        mode='markers',
                        name=f'{name} ({mask.sum()})',
                        marker=dict(
                            size=3,
                            color=importance[mask],
                            colorscale='Viridis',
                            showscale=True if i == 0 else False,
                            colorbar=dict(title="Importance", x=0.48) if i == 0 else None
                        ),
                        text=f'Importance: {importance[mask]:.4f}',
                        hovertemplate='Opacity: %{x:.4f}<br>Volume: %{y:.4f}<br>%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Log scale plot
                fig.add_trace(
                    go.Scatter(
                        x=opacity[mask],
                        y=log_volume[mask],
                        mode='markers',
                        name=f'{name} (log)',
                        marker=dict(size=3, color=color),
                        showlegend=False
                    ),
                    row=1, col=2
                )

        fig.update_xaxes(title_text="Opacity (ω)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (V)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Opacity (ω)", row=1, col=2)
        fig.update_yaxes(title_text="Log Volume", row=1, col=2)

        fig.update_layout(
            title="Gaussians in (ω, V) Importance Space",
            height=500,
            width=1200
        )

        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"Saved interactive plot to {save_path.replace('.png', '.html')}")

        if show:
            fig.show()

    else:
        # Create matplotlib visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: All points with layer colors
        for i, (name, color) in enumerate(zip(layer_names, colors)):
            mask = layer_assignments == i
            if mask.sum() > 0:
                scatter = axes[0].scatter(opacity[mask], volume[mask],
                                         c=color, label=f'{name} ({mask.sum()})',
                                         s=1, alpha=0.6)

        axes[0].set_xlabel('Opacity (ω)')
        axes[0].set_ylabel('Volume (V)')
        axes[0].set_title('Gaussians in (ω, V) Space')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Color by importance
        scatter = axes[1].scatter(opacity, volume, c=importance, s=1,
                                 cmap='viridis', alpha=0.6)
        axes[1].set_xlabel('Opacity (ω)')
        axes[1].set_ylabel('Volume (V)')
        axes[1].set_title('Colored by Importance')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1], label='Importance')

        # Plot 3: Highlight potential floaters
        omega_thresh = np.percentile(opacity, 90)
        volume_thresh = np.percentile(volume, 90)

        normal_mask = ~((opacity > omega_thresh) & (volume > volume_thresh))
        floater_mask = (opacity > omega_thresh) & (volume > volume_thresh)

        axes[2].scatter(opacity[normal_mask], volume[normal_mask],
                       c='blue', s=1, alpha=0.6, label='Normal')
        axes[2].scatter(opacity[floater_mask], volume[floater_mask],
                       c='red', s=5, alpha=0.8, label=f'Potential Floaters ({floater_mask.sum()})')

        axes[2].set_xlabel('Opacity (ω)')
        axes[2].set_ylabel('Volume (V)')
        axes[2].set_title('Floater Detection (Top 10% ω and V)')
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved omega-volume scatter plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def visualize_layers_in_3d(gaussians, viewpoint_camera=None, save_path=None, show=True):
    """
    3D visualization of layer assignments

    Args:
        gaussians: GaussianModel instance
        viewpoint_camera: Camera viewpoint for visualization (optional)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if len(gaussians.layer_assignments) == 0:
        print("No layer assignments available. Updating layers...")
        gaussians.update_layer_assignments()

    xyz = gaussians.get_xyz.cpu().numpy()
    layer_assignments = gaussians.layer_assignments.cpu().numpy()
    importance = gaussians.importance.cpu().numpy()

    layer_names = ['Salient', 'Transition', 'Background']
    colors = ['green', 'blue', 'gray']

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: 3D scatter by layers
    ax1 = fig.add_subplot(131, projection='3d')
    for i, (name, color) in enumerate(zip(layer_names, colors)):
        mask = layer_assignments == i
        if mask.sum() > 0:
            ax1.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
                       c=color, label=f'{name} ({mask.sum()})', s=1, alpha=0.6)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Layer Visualization')
    ax1.legend()

    # Plot 2: XY projection
    ax2 = fig.add_subplot(132)
    for i, (name, color) in enumerate(zip(layer_names, colors)):
        mask = layer_assignments == i
        if mask.sum() > 0:
            ax2.scatter(xyz[mask, 0], xyz[mask, 1],
                       c=color, label=f'{name}', s=1, alpha=0.6)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: XZ projection colored by importance
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(xyz[:, 0], xyz[:, 2], c=importance, s=1,
                         cmap='viridis', alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection (colored by importance)')
    ax3.set_aspect('equal')
    plt.colorbar(scatter, ax=ax3, label='Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D layer visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_layer_statistics(gaussians, save_path=None, show=True):
    """
    Plot detailed statistics for each layer

    Args:
        gaussians: GaussianModel instance
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    stats = gaussians.get_layer_statistics()

    if not stats:
        print("No statistics available. Computing layer assignments...")
        gaussians.update_layer_assignments()
        stats = gaussians.get_layer_statistics()

    layer_names = ['salient', 'transition', 'background']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Extract data for plotting
    counts = [stats[name]['count'] for name in layer_names]
    importance_means = [stats[name]['importance_mean'] for name in layer_names]
    opacity_means = [stats[name]['opacity_mean'] for name in layer_names]
    volume_means = [stats[name]['volume_mean'] for name in layer_names]
    volume_medians = [stats[name]['volume_median'] for name in layer_names]
    importance_stds = [stats[name]['importance_std'] for name in layer_names]

    colors = ['green', 'blue', 'gray']

    # Plot 1: Count distribution
    axes[0, 0].bar(layer_names, counts, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Gaussian Count per Layer')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Importance statistics
    x_pos = np.arange(len(layer_names))
    axes[0, 1].bar(x_pos, importance_means, yerr=importance_stds,
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 1].set_ylabel('Mean Importance')
    axes[0, 1].set_title('Importance Statistics')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(layer_names)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Opacity distribution
    axes[0, 2].bar(layer_names, opacity_means, color=colors, alpha=0.7)
    axes[0, 2].set_ylabel('Mean Opacity')
    axes[0, 2].set_title('Opacity Distribution')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Plot 4: Volume comparison (mean vs median)
    x_pos = np.arange(len(layer_names))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, volume_means, width, label='Mean',
                   color=colors, alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, volume_medians, width, label='Median',
                   color=colors, alpha=0.4)
    axes[1, 0].set_ylabel('Volume')
    axes[1, 0].set_title('Volume Distribution (Mean vs Median)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(layer_names)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 5: Layer size distribution
    total = sum(counts)
    percentages = [100 * count / total for count in counts]
    axes[1, 1].pie(percentages, labels=layer_names, colors=colors,
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Layer Size Distribution')

    # Plot 6: Summary table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')

    table_data = []
    headers = ['Layer', 'Count', 'Imp. Mean', 'Opac. Mean', 'Vol. Mean']

    for i, name in enumerate(layer_names):
        table_data.append([
            name.capitalize(),
            f"{counts[i]:,}",
            f"{importance_means[i]:.4f}",
            f"{opacity_means[i]:.4f}",
            f"{volume_means[i]:.2e}"
        ])

    table = axes[1, 2].table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer statistics plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_comprehensive_report(gaussians, output_dir="./ohdgs_analysis"):
    """
    Create a comprehensive analysis report of the current Gaussian model

    Args:
        gaussians: GaussianModel instance
        output_dir: Directory to save all plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating comprehensive OHDGS analysis report...")
    print(f"Output directory: {output_dir}")

    # Compute importance if not already computed
    if len(gaussians.importance) == 0:
        print("Computing importance metrics...")
        gaussians.compute_importance()

    # Update layer assignments if not done
    if len(gaussians.layer_assignments) == 0:
        print("Updating layer assignments...")
        gaussians.update_layer_assignments()

    # Generate all visualizations
    print("Generating importance distribution plot...")
    plot_importance_distribution(gaussians,
                                save_path=os.path.join(output_dir, "importance_distribution.png"),
                                show=False)

    print("Generating omega-volume scatter plot...")
    plot_omega_volume_scatter(gaussians,
                             save_path=os.path.join(output_dir, "omega_volume_scatter.png"),
                             show=False)

    print("Generating 3D layer visualization...")
    visualize_layers_in_3d(gaussians,
                          save_path=os.path.join(output_dir, "3d_layers.png"),
                          show=False)

    print("Generating layer statistics...")
    plot_layer_statistics(gaussians,
                         save_path=os.path.join(output_dir, "layer_statistics.png"),
                         show=False)

    # Print summary statistics
    stats = gaussians.get_layer_statistics()
    print("\n" + "="*50)
    print("OHDGS ANALYSIS SUMMARY")
    print("="*50)

    total_gaussians = sum([stats[name]['count'] for name in ['salient', 'transition', 'background']])
    print(f"Total Gaussians: {total_gaussians:,}")

    for name in ['salient', 'transition', 'background']:
        count = stats[name]['count']
        percentage = 100 * count / total_gaussians
        print(f"\n{name.capitalize()} Layer:")
        print(f"  Count: {count:,} ({percentage:.1f}%)")
        print(f"  Mean Importance: {stats[name]['importance_mean']:.4f} ± {stats[name]['importance_std']:.4f}")
        print(f"  Mean Opacity: {stats[name]['opacity_mean']:.4f}")
        print(f"  Mean Volume: {stats[name]['volume_mean']:.2e}")

    print(f"\nAnalysis complete! Check {output_dir} for detailed visualizations.")


if __name__ == "__main__":
    # Example usage
    print("OHDGS Visualization Utilities")
    print("Import this module and use the functions to analyze your Gaussian model")
    print("Example: from utils.visualization import create_comprehensive_report")