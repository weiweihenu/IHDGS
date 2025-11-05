#!/usr/bin/env python3
"""
OHDGS Analysis Script
Usage: python analyze_ohdgs.py --model_path output/model/chkpntXXXX.pth --output_dir ./analysis
"""

import argparse
import torch
import os
from scene.gaussian_model import GaussianModel
from utils.visualization import create_comprehensive_report
from arguments import ModelParams


def analyze_checkpoint(model_path, output_dir="./ohdgs_analysis", show_plots=True):
    """
    Analyze a trained OHDGS model checkpoint

    Args:
        model_path: Path to the checkpoint file
        output_dir: Directory to save analysis results
        show_plots: Whether to display plots interactively
    """
    print("="*60)
    print("OHDGS Model Analysis")
    print("="*60)

    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cuda')
        print(f"Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Create GaussianModel instance
    print("Initializing GaussianModel...")
    class Args:
        sh_degree = 3

    args = Args()
    gaussians = GaussianModel(args)

    # Restore model parameters
    print("Restoring model parameters...")
    try:
        # Handle different checkpoint formats
        if isinstance(checkpoint, tuple):
            # Standard FSGS checkpoint is saved as tuple
            print(f"Checkpoint is a tuple with {len(checkpoint)} elements")
            model_args = checkpoint
        elif isinstance(checkpoint, dict):
            # Check if it's a wrapped tuple
            if 'model_args' in checkpoint:
                print("Found wrapped tuple in 'model_args' key")
                model_args = checkpoint['model_args']
            else:
                print("Dict format with keys:", list(checkpoint.keys())[:10])  # Show first 10 keys
                # Try to find the actual model data
                if '_xyz' in checkpoint:
                    print("Direct state dict format")
                    # This is a direct state dict, not a tuple
                    model_args = checkpoint
                else:
                    print("Unknown dict format")
                    return
        else:
            print(f"Unexpected checkpoint format: {type(checkpoint)}")
            return

        # Now handle the model_args
        if isinstance(model_args, tuple):
            # Standard FSGS tuple format
            print(f"Processing tuple format with {len(model_args)} elements")
            # Create dummy training args
            from arguments import OptimizationParams
            class DummyTrainingArgs:
                pass
            training_args = DummyTrainingArgs()
            training_args.position_lr_init = 0.00016
            training_args.position_lr_final = 0.0000016
            training_args.position_lr_delay_mult = 0.01
            training_args.position_lr_max_steps = 30000
            training_args.feature_lr = 0.0025
            training_args.opacity_lr = 0.05
            training_args.scaling_lr = 0.005
            training_args.rotation_lr = 0.001
            training_args.percent_dense = 0.01

            # Use the restore method
            gaussians.restore(model_args, training_args)

            # Check if OHDGS data was included
            if len(model_args) >= 15:
                print("OHDGS data found in checkpoint")
                gaussians.importance = model_args[12]
                gaussians.layer_assignments = model_args[13]
                gaussians.importance_lambda = model_args[14]
            else:
                print("No OHDGS data in checkpoint, initializing...")
                num_gaussians = gaussians.get_xyz.shape[0]
                gaussians.importance = torch.zeros(num_gaussians, device='cuda')
                gaussians.layer_assignments = torch.zeros(num_gaussians, dtype=torch.long, device='cuda')
                gaussians.importance_lambda = 1.0

        elif isinstance(model_args, dict):
            # Direct state dict format
            print("Processing state dict format")
            # Manually restore parameters
            if '_xyz' in model_args:
                gaussians._xyz = model_args['_xyz']
                gaussians._features_dc = model_args['_features_dc']
                gaussians._features_rest = model_args['_features_rest']
                gaussians._scaling = model_args['_scaling']
                gaussians._rotation = model_args['_rotation']
                gaussians._opacity = model_args['_opacity']

                # Restore optional attributes
                if 'max_radii2D' in model_args:
                    gaussians.max_radii2D = model_args['max_radii2D']
                if 'xyz_gradient_accum' in model_args:
                    gaussians.xyz_gradient_accum = model_args['xyz_gradient_accum']
                if 'denom' in model_args:
                    gaussians.denom = model_args['denom']
                if 'confidence' in model_args:
                    gaussians.confidence = model_args['confidence']

                # Initialize OHDGS attributes
                num_gaussians = gaussians.get_xyz.shape[0]
                gaussians.importance = model_args.get('importance', torch.zeros(num_gaussians, device='cuda'))
                gaussians.layer_assignments = model_args.get('layer_assignments', torch.zeros(num_gaussians, dtype=torch.long, device='cuda'))
                gaussians.importance_lambda = model_args.get('importance_lambda', 1.0)
            else:
                print("State dict doesn't contain expected keys")
                return
        else:
            print(f"Unknown model_args format: {type(model_args)}")
            return

    except Exception as e:
        print(f"Error restoring model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Basic model info
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"\nModel Information:")
    print(f"  Total Gaussians: {num_gaussians:,}")
    print(f"  Active SH degree: {gaussians.active_sh_degree}")
    print(f"  Max SH degree: {gaussians.max_sh_degree}")

    # Compute importance if needed
    if len(gaussians.importance) == 0 or gaussians.importance.shape[0] != num_gaussians:
        print("\nComputing importance metrics...")
        gaussians.compute_importance()
        print(f"  Importance range: [{gaussians.importance.min():.6f}, {gaussians.importance.max():.6f}]")
        print(f"  Mean importance: {gaussians.importance.mean():.6f}")

    # Update layer assignments if needed
    if len(gaussians.layer_assignments) == 0 or gaussians.layer_assignments.shape[0] != num_gaussians:
        print("\nUpdating layer assignments...")
        layer_assignments, tau_s, tau_t = gaussians.update_layer_assignments()
        print(f"  Salient threshold (τ_s): {tau_s:.6f}")
        print(f"  Transition threshold (τ_t): {tau_t:.6f}")

    # Print layer statistics
    print("\nLayer Statistics:")
    stats = gaussians.get_layer_statistics()
    if stats:
        for layer_name in ['salient', 'transition', 'background']:
            layer_stats = stats[layer_name]
            percentage = 100 * layer_stats['count'] / num_gaussians
            print(f"\n{layer_name.capitalize()} Layer:")
            print(f"  Count: {layer_stats['count']:,} ({percentage:.1f}%)")
            print(f"  Importance: {layer_stats['importance_mean']:.6f} ± {layer_stats['importance_std']:.6f}")
            print(f"  Opacity: {layer_stats['opacity_mean']:.4f}")
            print(f"  Volume: {layer_stats['volume_mean']:.2e} (median: {layer_stats['volume_median']:.2e})")

    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        create_comprehensive_report(gaussians, output_dir=output_dir)
        print(f"\nAll visualizations saved to: {output_dir}")

        # List generated files
        output_files = os.listdir(output_dir)
        print("\nGenerated files:")
        for f in sorted(output_files):
            if f.endswith(('.png', '.html')):
                print(f"  - {f}")

    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # Additional analysis
    print("\n" + "="*60)
    print("Advanced Analysis")
    print("="*60)

    # Analyze potential floaters
    opacity = gaussians.get_opacity.squeeze().cpu().numpy()
    volume = gaussians.compute_volume().cpu().numpy()
    importance = gaussians.importance.cpu().numpy()

    # Detect potential floaters (high opacity, high volume, low importance)
    omega_thresh = torch.quantile(gaussians.get_opacity.squeeze(), 0.9)
    volume_thresh = torch.quantile(gaussians.compute_volume(), 0.9)
    importance_thresh = torch.quantile(gaussians.importance, 0.5)

    floater_mask = (opacity > omega_thresh.item()) & (volume > volume_thresh.item()) & (importance < importance_thresh.item())
    num_floaters = floater_mask.sum()
    floater_percentage = 100 * num_floaters / num_gaussians

    print(f"Potential Floating Artifacts:")
    print(f"  Count: {num_floaters:,} ({floater_percentage:.1f}%)")
    print(f"  Thresholds: ω > {omega_thresh.item():.4f}, V > {volume_thresh.item():.2e}, I < {importance_thresh.item():.6f}")

    # Importance distribution analysis
    print(f"\nImportance Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(gaussians.importance, p/100).item()
        print(f"  {p}th percentile: {val:.6f}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


def list_checkpoints(model_dir):
    """List all available checkpoints in a model directory"""
    print(f"\nAvailable checkpoints in {model_dir}:")
    if not os.path.exists(model_dir):
        print(f"  Directory not found: {model_dir}")
        return

    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('chkpnt') and f.endswith('.pth')]
    checkpoint_files.sort()

    if not checkpoint_files:
        print("  No checkpoint files found.")
    else:
        for i, f in enumerate(checkpoint_files, 1):
            iteration = f.replace('chkpnt', '').replace('.pth', '')
            print(f"  {i}. {f} (iteration {iteration})")


def main():
    parser = argparse.ArgumentParser(description="Analyze OHDGS trained models")
    parser.add_argument("--model_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--model_dir", type=str, help="Model directory (will list checkpoints)")
    parser.add_argument("--output_dir", type=str, default="./ohdgs_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots interactively")

    args = parser.parse_args()

    if args.list or args.model_dir:
        if args.model_dir:
            list_checkpoints(args.model_dir)
        else:
            # Extract model directory from model_path
            if args.model_path:
                model_dir = os.path.dirname(args.model_path)
                list_checkpoints(model_dir)
            else:
                print("Please specify --model_dir or --model_path")
        return

    if not args.model_path:
        print("Usage: python analyze_ohdgs.py --model_path path/to/chkpntXXXX.pth")
        print("   or: python analyze_ohdgs.py --model_dir path/to/model --list")
        return

    analyze_checkpoint(
        model_path=args.model_path,
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )


if __name__ == "__main__":
    main()