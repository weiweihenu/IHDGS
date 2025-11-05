#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    # Initialize performance optimizer
    from utils.performance_optimizer import get_optimizer, monitor_performance
    optimizer = get_optimizer("cuda")  # Use default CUDA device

    # Initialize threshold tracking for visualization
    threshold_history = {
        'iterations': [],
        'salient_thresholds': [],
        'transition_thresholds': [],
        'total_gaussians': []
    } if opt.track_layer_thresholds else None

    # Initialize floater suppression tracking for visualization
    floater_history = {
        'iterations': [],
        'floaters_detected': [],
        'floaters_suppressed': [],
        'floaters_pruned': [],
        'total_handled': []
    } if opt.track_floater_suppression else None

    # Track NaN occurrences to reduce log spam
    nan_stats = {
        'depth_loss_count': 0,
        'depth_loss_pseudo_count': 0,
        'last_report_iter': 0
    }

    for iteration in range(first_iter, opt.iterations + 1):
        # Performance monitoring and optimization
        monitor_performance(gaussians, iteration, "cuda")

        # Apply training step optimizations
        gaussians = optimizer.optimize_training_step(gaussians, iteration)

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # OHDGS: Update layer assignments only when densification happens
        # This ensures layer assignments are always up-to-date when needed
        need_layer_update = False

        # Check if any densification strategy will be applied
        cdd_will_run = (opt.cdd_start_iter <= iteration <= opt.cdd_end_iter and
                       iteration % opt.cdd_interval == 0)
        nod_will_run = (opt.nod_start_iter <= iteration <= opt.nod_end_iter and
                       iteration % opt.nod_interval == 0)
        floater_will_run = (opt.floater_start_iter <= iteration <= opt.floater_end_iter and
                          iteration % opt.floater_interval == 0)

        # Also update at regular intervals to keep assignments fresh
        regular_update = iteration % opt.layer_update_interval == 0

        if cdd_will_run or nod_will_run or floater_will_run or regular_update:
            # Only compute importance and update layers if needed
            if len(gaussians.importance) == 0 or regular_update:
                layer_assignments, tau_s, tau_t = gaussians.update_layer_assignments(
                    alpha=opt.salient_percentile,
                    beta=opt.transition_percentile
                )

                if regular_update:
                    print(f"[Iteration {iteration}] Updated layer assignments:")
                    print(f"  λ (importance_lambda): {gaussians.importance_lambda}")
                    print(f"  α (salient_percentile): {opt.salient_percentile}%")
                    print(f"  β (transition_percentile): {opt.transition_percentile}%")
                    print(f"  Salient threshold (τ_s): {tau_s:.6f}")
                    print(f"  Transition threshold (τ_t): {tau_t:.6f}")

                    # Print layer statistics
                    stats = gaussians.get_layer_statistics()
                    total_gaussians = gaussians.get_xyz.shape[0]
                    for layer_name, count in [('salient', stats['salient']['count']),
                                             ('transition', stats['transition']['count']),
                                             ('background', stats['background']['count'])]:
                        percentage = 100 * count / total_gaussians
                        print(f"  {layer_name.capitalize()}: {count:,} ({percentage:.1f}%)")
                    print("-" * 50)

                    # Track thresholds for visualization if enabled
                    if threshold_history is not None:
                        threshold_history['iterations'].append(iteration)
                        threshold_history['salient_thresholds'].append(tau_s.item() if hasattr(tau_s, 'item') else tau_s)
                        threshold_history['transition_thresholds'].append(tau_t.item() if hasattr(tau_t, 'item') else tau_t)
                        threshold_history['total_gaussians'].append(total_gaussians)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)

        # Compute SSIM with NaN check
        ssim_value = ssim(image, gt_image)
        if torch.isnan(ssim_value):
            print(f"[Warning] SSIM is NaN at iteration {iteration}, using 0 instead")
            ssim_value = torch.tensor(0.0, device=image.device)

        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value))

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Error] Loss is NaN/Inf at iteration {iteration}!")
            print(f"  Ll1: {Ll1}")
            print(f"  SSIM: {ssim_value}")
            print(f"  Loss: {loss}")
            # Skip optimizer step if loss is invalid
            continue


        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )

        # Check depth_loss for NaN before adding to loss
        if torch.isnan(depth_loss) or torch.isinf(depth_loss):
            nan_stats['depth_loss_count'] += 1
            # Only print the first time and periodic summaries
            if nan_stats['depth_loss_count'] == 1:
                print(f"[Warning] depth_loss is NaN/Inf at iteration {iteration}, will skip when detected")
            depth_loss = torch.tensor(0.0, device=loss.device)

        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001



        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            # Check depth_loss_pseudo for NaN
            if torch.isnan(depth_loss_pseudo) or torch.isinf(depth_loss_pseudo):
                nan_stats['depth_loss_pseudo_count'] += 1
                # Only print the first time
                if nan_stats['depth_loss_pseudo_count'] == 1:
                    print(f"[Warning] depth_loss_pseudo is NaN/Inf at iteration {iteration}, will skip when detected")
            else:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

        # Final NaN check before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Error] Final loss is NaN/Inf at iteration {iteration}, skipping backward")
            continue

        loss.backward()

        # Gradient clipping to prevent numerical instability
        # Clip gradients of all parameters managed by the optimizer
        params_with_grad = []
        for param_group in gaussians.optimizer.param_groups:
            params_with_grad.extend([p for p in param_group['params'] if p.grad is not None])
        if params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background), opt, args)

            # Report NaN statistics at test iterations
            if iteration in testing_iterations:
                total_nan = nan_stats['depth_loss_count'] + nan_stats['depth_loss_pseudo_count']
                if total_nan > 0:
                    print(f"\n[NaN Statistics up to iteration {iteration}]")
                    print(f"  depth_loss NaN count: {nan_stats['depth_loss_count']}")
                    print(f"  depth_loss_pseudo NaN count: {nan_stats['depth_loss_pseudo_count']}")
                    print(f"  Total NaN occurrences: {total_nan}")
                    if total_nan > 100:
                        print(f"  Note: High NaN count indicates depth estimation instability")
                    print("")

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                # OHDGS: Save filtered model versions if enabled
                if opt.save_filtered_models and (opt.filter_top_percent > 0 or opt.filter_bottom_percent > 0):
                    print(f"\n[ITER {iteration}] Saving Filtered Model Versions...")
                    try:
                        from utils.model_filter import save_filtered_model_and_evaluate

                        # Ensure importance is computed
                        if len(gaussians.importance) == 0:
                            gaussians.compute_importance()

                        # Save and evaluate filtered models
                        save_filtered_model_and_evaluate(
                            gaussians,
                            scene,
                            pipe,
                            background,
                            scene.model_path,
                            filter_top_percent=opt.filter_top_percent,
                            filter_bottom_percent=opt.filter_bottom_percent,
                            device="cuda"
                        )

                    except Exception as e:
                        print(f"[Warning] Filtered model saving failed: {e}")
                        import traceback
                        traceback.print_exc()

            # OHDGS: Save visualizations at checkpoint iterations with performance optimization
            if iteration > first_iter and (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Generating OHDGS Visualizations...")
                try:
                    # Use performance optimization for visualization generation
                    with optimize_context("visualization_generation", "cuda"):
                        # Compute importance and update layers if needed
                        if len(gaussians.importance) == 0:
                            gaussians.compute_importance()
                        if len(gaussians.layer_assignments) == 0:
                            gaussians.update_layer_assignments()

                        # Create visualization directory
                        import os
                        vis_dir = os.path.join(scene.model_path, "visualizations", f"iter_{iteration}")
                        os.makedirs(vis_dir, exist_ok=True)

                        # Note: Visualizations will use tensors on their current device
                        # The visualization functions handle moving to CPU internally if needed

                        # Import visualization functions
                        from utils.visualization import (
                            plot_importance_distribution,
                            plot_omega_volume_scatter,
                            visualize_layers_in_3d,
                            plot_layer_statistics
                        )

                        # Generate all visualizations
                        print(f"  - Saving importance distribution...")
                        plot_importance_distribution(
                            gaussians,
                            save_path=os.path.join(vis_dir, "importance_distribution.png"),
                            show=False
                        )

                        print(f"  - Saving omega-volume scatter plot...")
                        plot_omega_volume_scatter(
                            gaussians,
                            save_path=os.path.join(vis_dir, "omega_volume_scatter.png"),
                            show=False,
                            use_plotly=False  # Use matplotlib to avoid dependencies
                        )

                        print(f"  - Saving 3D layer visualization...")
                        visualize_layers_in_3d(
                            gaussians,
                            save_path=os.path.join(vis_dir, "3d_layers.png"),
                            show=False
                        )

                        print(f"  - Saving layer statistics...")
                        plot_layer_statistics(
                            gaussians,
                            save_path=os.path.join(vis_dir, "layer_statistics.png"),
                            show=False
                        )

                    # Save text statistics
                        stats = gaussians.get_layer_statistics()
                        with open(os.path.join(vis_dir, "statistics.txt"), 'w') as f:
                            f.write(f"OHDGS Statistics - Iteration {iteration}\n")
                            f.write("="*50 + "\n")
                            f.write(f"Total Gaussians: {gaussians.get_xyz.shape[0]:,}\n\n")

                            for layer_name in ['salient', 'transition', 'background']:
                                if stats[layer_name]['count'] > 0:
                                    f.write(f"{layer_name.capitalize()} Layer:\n")
                                    f.write(f"  Count: {stats[layer_name]['count']:,}\n")
                                    f.write(f"  Percentage: {100*stats[layer_name]['count']/gaussians.get_xyz.shape[0]:.1f}%\n")
                                    f.write(f"  Mean Importance: {stats[layer_name]['importance_mean']:.6f}\n")
                                    f.write(f"  Std Importance: {stats[layer_name]['importance_std']:.6f}\n")
                                    f.write(f"  Mean Opacity: {stats[layer_name]['opacity_mean']:.4f}\n")
                                    f.write(f"  Mean Volume: {stats[layer_name]['volume_mean']:.2e}\n")
                                    f.write(f"  Median Volume: {stats[layer_name]['volume_median']:.2e}\n\n")

                        # Note: Tensors remain on their original device
                        print(f"  - Visualizations saved to: {vis_dir}")

                except Exception as e:
                    print(f"  [Warning] Visualization generation failed: {e}")
                    # Continue training even if visualization fails

            # OHDGS: Hierarchical Densification
            # Replace vanilla densification with layer-specific strategies
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # OHDGS: Apply hierarchical densification only when strategies are scheduled
                # Check if any strategy should run before applying hierarchical densification
                cdd_will_run = (opt.cdd_start_iter <= iteration <= opt.cdd_end_iter and
                               iteration % opt.cdd_interval == 0)
                nod_will_run = (opt.nod_start_iter <= iteration <= opt.nod_end_iter and
                               iteration % opt.nod_interval == 0)
                floater_will_run = (opt.floater_start_iter <= iteration <= opt.floater_end_iter and
                                  iteration % opt.floater_interval == 0)

                # Initialize default summary
                densify_summary = {
                    'cdd_added': 0,
                    'nod_added': 0,
                    'floaters_handled': 0,
                    'floaters_detected': 0,
                    'floaters_suppressed': 0,
                    'floaters_pruned': 0,
                    'layer_stats_after': gaussians.get_layer_statistics() if len(gaussians.layer_assignments) > 0 else {'salient': {'count': 0}, 'transition': {'count': 0}, 'background': {'count': 0}},
                    'total_gaussians_after': len(gaussians.get_xyz)
                }

                # Only run hierarchical densification when at least one strategy is active
                if cdd_will_run or nod_will_run or floater_will_run:
                    from utils.hierarchical_densification import apply_hierarchical_densification
                    from utils.performance_optimizer import optimize_context

                    with optimize_context("hierarchical_densification", "cuda"):
                        densify_summary = apply_hierarchical_densification(
                            gaussians, viewpoint_cam, pipe, background, iteration,
                            # CDD parameters
                            cdd_max_new=opt.cdd_max_new_per_view,
                            cdd_percentile=opt.cdd_error_percentile,
                            cdd_min_error=opt.cdd_min_error_threshold,
                            cdd_color_threshold=opt.cdd_color_threshold,
                            cdd_depth_threshold=opt.cdd_depth_threshold,
                            # NOD parameters
                            nod_max_new=opt.nod_max_new_per_call,
                            # Scheduling parameters
                            cdd_interval=opt.cdd_interval,
                            nod_interval=opt.nod_interval,
                            floater_interval=opt.floater_interval,
                            # Time control parameters
                            cdd_start_iter=opt.cdd_start_iter,
                            cdd_end_iter=opt.cdd_end_iter,
                            nod_start_iter=opt.nod_start_iter,
                            nod_end_iter=opt.nod_end_iter,
                            floater_start_iter=opt.floater_start_iter,
                            floater_end_iter=opt.floater_end_iter
                        )

                # GPU memory cleanup after densification to prevent CUDA errors
                if densify_summary['cdd_added'] > 0 or densify_summary['nod_added'] > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Track floater suppression statistics if enabled
                if floater_history is not None and densify_summary['floaters_detected'] > 0:
                    floater_history['iterations'].append(iteration)
                    floater_history['floaters_detected'].append(densify_summary['floaters_detected'])
                    floater_history['floaters_suppressed'].append(densify_summary['floaters_suppressed'])
                    floater_history['floaters_pruned'].append(densify_summary['floaters_pruned'])
                    floater_history['total_handled'].append(densify_summary['floaters_handled'])

                # Log densification activities
                if densify_summary['cdd_added'] > 0 or densify_summary['nod_added'] > 0 or densify_summary['floaters_handled'] > 0:
                    print(f"[OHDGS Iter {iteration}] Densification Summary:")
                    if densify_summary['cdd_added'] > 0:
                        print(f"  CDD (Salient): +{densify_summary['cdd_added']} Gaussians")
                    if densify_summary['nod_added'] > 0:
                        print(f"  NOD (Transition): +{densify_summary['nod_added']} Gaussians")
                    if densify_summary['floaters_handled'] > 0:
                        print(f"  Floaters (Background): {densify_summary['floaters_handled']} handled")

                    # Show layer statistics
                    stats_after = densify_summary['layer_stats_after']
                    total_after = densify_summary['total_gaussians_after']
                    for layer_name in ['salient', 'transition', 'background']:
                        if stats_after[layer_name]['count'] > 0:
                            percentage = 100 * stats_after[layer_name]['count'] / total_after
                            print(f"  {layer_name.capitalize()}: {stats_after[layer_name]['count']:,} ({percentage:.1f}%)")

            # Original densification for backward compatibility (reduced intensity)
            if iteration > opt.densify_from_iter and iteration % (opt.densification_interval * 2) == 0:  # Less frequent
                size_threshold = None
                gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

    # Print final NaN statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    total_nan = nan_stats['depth_loss_count'] + nan_stats['depth_loss_pseudo_count']
    if total_nan > 0:
        print(f"NaN Statistics:")
        print(f"  depth_loss NaN count: {nan_stats['depth_loss_count']}")
        print(f"  depth_loss_pseudo NaN count: {nan_stats['depth_loss_pseudo_count']}")
        print(f"  Total NaN occurrences: {total_nan}")
        percentage = 100.0 * total_nan / opt.iterations
        print(f"  NaN percentage: {percentage:.2f}%")
        if total_nan > 100:
            print(f"  Note: High NaN count suggests depth estimation instability due to low variance")
    else:
        print(f"No NaN occurrences detected during training")
    print("="*60 + "\n")

    # Plot threshold history if tracking is enabled
    if threshold_history is not None and len(threshold_history['iterations']) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt

            print("="*60)
            print("Generating Layer Threshold Evolution Plot")
            print("="*60)

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            iterations = threshold_history['iterations']
            salient_thresholds = threshold_history['salient_thresholds']
            transition_thresholds = threshold_history['transition_thresholds']
            total_gaussians = threshold_history['total_gaussians']

            # Plot 1: Thresholds over time
            ax1.plot(iterations, salient_thresholds, 'b-o', label='Salient Threshold (τ_s)', linewidth=2, markersize=6)
            ax1.plot(iterations, transition_thresholds, 'r-s', label='Transition Threshold (τ_t)', linewidth=2, markersize=6)
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Threshold Value', fontsize=12)
            ax1.set_title('Layer Assignment Thresholds Evolution', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # Use log scale for better visualization

            # Plot 2: Total Gaussians over time
            ax2.plot(iterations, total_gaussians, 'g-^', label='Total Gaussians', linewidth=2, markersize=6)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Number of Gaussians', fontsize=12)
            ax2.set_title('Total Gaussian Count Evolution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            import os
            plot_path = os.path.join(dataset.model_path, 'threshold_evolution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Threshold evolution plot saved to: {plot_path}")
            print(f"Total data points recorded: {len(iterations)}")
            print("="*60 + "\n")

            # Also save the raw data as CSV for further analysis
            import csv
            csv_path = os.path.join(dataset.model_path, 'threshold_evolution.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Iteration', 'Salient_Threshold', 'Transition_Threshold', 'Total_Gaussians'])
                for i in range(len(iterations)):
                    writer.writerow([iterations[i], salient_thresholds[i], transition_thresholds[i], total_gaussians[i]])
            print(f"Threshold data saved to: {csv_path}\n")

        except Exception as e:
            print(f"[Warning] Failed to generate threshold plot: {e}")

    # Plot floater suppression history if tracking is enabled
    if floater_history is not None and len(floater_history['iterations']) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt

            print("="*60)
            print("Generating Floater Suppression Evolution Plot")
            print("="*60)

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            iterations = floater_history['iterations']
            floaters_detected = floater_history['floaters_detected']
            floaters_suppressed = floater_history['floaters_suppressed']
            floaters_pruned = floater_history['floaters_pruned']
            total_handled = floater_history['total_handled']

            # Plot 1: Floater counts over time
            ax1.plot(iterations, floaters_detected, 'r-o', label='Floaters Detected', linewidth=2, markersize=5)
            ax1.plot(iterations, floaters_suppressed, 'b-s', label='Floaters Suppressed', linewidth=2, markersize=5)
            ax1.plot(iterations, floaters_pruned, 'g-^', label='Floaters Pruned', linewidth=2, markersize=5)
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Number of Floaters', fontsize=12)
            ax1.set_title('Floater Detection and Suppression Over Time', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Total handled over time
            ax2.plot(iterations, total_handled, 'm-D', label='Total Handled (Suppressed + Pruned)', linewidth=2, markersize=5)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Total Floaters Handled', fontsize=12)
            ax2.set_title('Total Floater Handling Over Time', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            import os
            plot_path = os.path.join(dataset.model_path, 'floater_suppression_evolution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Floater suppression plot saved to: {plot_path}")
            print(f"Total floater suppression events: {len(iterations)}")

            # Calculate cumulative statistics
            total_detected = sum(floaters_detected)
            total_suppressed = sum(floaters_suppressed)
            total_pruned = sum(floaters_pruned)
            print(f"Cumulative floaters detected: {total_detected}")
            print(f"Cumulative floaters suppressed: {total_suppressed}")
            print(f"Cumulative floaters pruned: {total_pruned}")
            print("="*60 + "\n")

            # Also save the raw data as CSV for further analysis
            import csv
            csv_path = os.path.join(dataset.model_path, 'floater_suppression_evolution.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Iteration', 'Floaters_Detected', 'Floaters_Suppressed', 'Floaters_Pruned', 'Total_Handled'])
                for i in range(len(iterations)):
                    writer.writerow([
                        iterations[i],
                        floaters_detected[i],
                        floaters_suppressed[i],
                        floaters_pruned[i],
                        total_handled[i]
                    ])
            print(f"Floater suppression data saved to: {csv_path}\n")

        except Exception as e:
            print(f"[Warning] Failed to generate floater suppression plot: {e}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, opt, args):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # OHDGS: Generate final visualizations at training completion
    if iteration == args.iterations:
        gaussians = scene.gaussians
        print("\n" + "="*60)
        print("TRAINING COMPLETE - Generating Final OHDGS Visualizations")
        print("="*60)
        try:
            # Compute importance and update layers if needed
            if len(gaussians.importance) == 0:
                gaussians.compute_importance()
            if len(gaussians.layer_assignments) == 0:
                gaussians.update_layer_assignments()

            # Create final visualization directory
            import os
            vis_dir = os.path.join(scene.model_path, "final_visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            # Move tensors to CPU for visualization (GaussianModel doesn't have .to() method)
            # Store original device info
            device_info = {
                'xyz_device': gaussians._xyz.device,
                'is_cuda': gaussians._xyz.is_cuda
            }

            # Import visualization functions
            from utils.visualization import (
                plot_importance_distribution,
                plot_omega_volume_scatter,
                visualize_layers_in_3d,
                plot_layer_statistics
            )

            # Generate all visualizations
            print("Generating final OHDGS analysis...")
            plot_importance_distribution(
                gaussians,
                save_path=os.path.join(vis_dir, "final_importance_distribution.png"),
                show=False
            )

            plot_omega_volume_scatter(
                gaussians,
                save_path=os.path.join(vis_dir, "final_omega_volume_scatter.png"),
                show=False,
                use_plotly=False
            )

            visualize_layers_in_3d(
                gaussians,
                save_path=os.path.join(vis_dir, "final_3d_layers.png"),
                show=False
            )

            plot_layer_statistics(
                gaussians,
                save_path=os.path.join(vis_dir, "final_layer_statistics.png"),
                show=False
            )

            # Generate comprehensive summary
            stats = gaussians.get_layer_statistics()
            num_gaussians = gaussians.get_xyz.shape[0]

            print("\nFinal OHDGS Statistics:")
            print("-"*60)
            print(f"Total Gaussians: {num_gaussians:,}")
            print(f"Active SH degree: {gaussians.active_sh_degree}")
            print(f"Importance λ: {gaussians.importance_lambda}")

            for layer_name in ['salient', 'transition', 'background']:
                count = stats[layer_name]['count']
                percentage = 100 * count / num_gaussians
                print(f"\n{layer_name.capitalize()} Layer:")
                print(f"  Count: {count:,} ({percentage:.1f}%)")
                print(f"  Importance: {stats[layer_name]['importance_mean']:.6f} ± {stats[layer_name]['importance_std']:.6f}")
                print(f"  Opacity: {stats[layer_name]['opacity_mean']:.4f}")
                print(f"  Volume: {stats[layer_name]['volume_mean']:.2e}")

            # Save summary to file
            with open(os.path.join(vis_dir, "final_summary.txt"), 'w') as f:
                f.write(f"OHDGS Final Summary - Iteration {iteration}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Model Path: {scene.model_path}\n")
                f.write(f"Total Iterations: {iteration}\n")
                f.write(f"Total Gaussians: {num_gaussians:,}\n")
                f.write(f"Active SH degree: {gaussians.active_sh_degree}\n")
                f.write(f"Importance λ: {gaussians.importance_lambda}\n")
                f.write(f"Salient percentile (α): {opt.salient_percentile}%\n")
                f.write(f"Transition percentile (β): {opt.transition_percentile}%\n")
                f.write(f"Layer update interval: {opt.layer_update_interval} iterations\n\n")

                for layer_name in ['salient', 'transition', 'background']:
                    count = stats[layer_name]['count']
                    percentage = 100 * count / num_gaussians
                    f.write(f"{layer_name.capitalize()} Layer:\n")
                    f.write(f"  Count: {count:,} ({percentage:.1f}%)\n")
                    f.write(f"  Mean Importance: {stats[layer_name]['importance_mean']:.6f}\n")
                    f.write(f"  Std Importance: {stats[layer_name]['importance_std']:.6f}\n")
                    f.write(f"  Mean Opacity: {stats[layer_name]['opacity_mean']:.4f}\n")
                    f.write(f"  Mean Volume: {stats[layer_name]['volume_mean']:.2e}\n")
                    f.write(f"  Median Volume: {stats[layer_name]['volume_median']:.2e}\n\n")

            # Note: Tensors are already on the correct device
            print(f"\nAll final visualizations saved to: {vis_dir}")
            print("="*60)

        except Exception as e:
            print(f"[Warning] Final visualization generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            # OHDGS: Add importance and layer tracking
            if len(scene.gaussians.importance) > 0:
                tb_writer.add_histogram("scene/importance_histogram", scene.gaussians.importance, iteration)
                tb_writer.add_scalar('ohdgs/importance_mean', scene.gaussians.importance.mean(), iteration)
                tb_writer.add_scalar('ohdgs/importance_std', scene.gaussians.importance.std(), iteration)

            if len(scene.gaussians.layer_assignments) > 0:
                # Count Gaussians in each layer
                layer_counts = []
                layer_names = ['salient', 'transition', 'background']
                for i in range(3):
                    count = (scene.gaussians.layer_assignments == i).sum().item()
                    layer_counts.append(count)
                    tb_writer.add_scalar(f'ohdgs/layer_{layer_names[i]}_count', count, iteration)

                # Layer statistics
                stats = scene.gaussians.get_layer_statistics()
                if stats:
                    for layer_name in layer_names:
                        if stats[layer_name]['count'] > 0:
                            tb_writer.add_scalar(f'ohdgs/{layer_name}_importance_mean',
                                               stats[layer_name]['importance_mean'], iteration)
                            tb_writer.add_scalar(f'ohdgs/{layer_name}_opacity_mean',
                                               stats[layer_name]['opacity_mean'], iteration)
                            tb_writer.add_scalar(f'ohdgs/{layer_name}_volume_mean',
                                               stats[layer_name]['volume_mean'], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")