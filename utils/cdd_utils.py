"""
Color-Depth Difference-aware Densification (CDD) for OHDGS
Implements Section 3.4.1: Salient Layer Densification Strategy
"""

import torch
import torch.nn.functional as F
import numpy as np
from utils.depth_utils import estimate_depth


class CDDDensifier:
    """
    Color-Depth Difference-aware Densification for salient layer
    """

    def __init__(self, device='cuda'):
        self.device = device
        # Thresholds from paper
        self.color_threshold = 0.05  # τ_c
        self.depth_threshold = 0.02   # τ_d
        # Parameter weights
        self.alpha_scale = 0.3       # Scale reduction factor
        self.gamma_color = 0.7       # Color difference weight
        self.eta_depth = 0.3         # Depth difference weight

        # Constraints to prevent over-densification
        self.max_new_gaussians_per_view = 1000  # Maximum new Gaussians per CDD call
        self.error_percentile = 95  # Only consider top 95% highest error regions
        self.min_error_threshold = 0.1  # Minimum error to consider for densification

    def align_depth(self, rendered_depth, observed_depth):
        """
        Align observed depth to rendered depth scale

        Args:
            rendered_depth: Rendered depth from Gaussian model
            observed_depth: Estimated depth from monocular model

        Returns:
            Aligned observed depth
        """
        # Flatten and filter valid pixels
        mask = (rendered_depth > 0) & (observed_depth > 0)
        if mask.sum() == 0:
            return observed_depth

        r_d = rendered_depth[mask].flatten()
        o_d = observed_depth[mask].flatten()

        # Least squares alignment: s * observed + t ≈ rendered
        A = torch.stack([o_d, torch.ones_like(o_d)], dim=1)
        b = r_d

        # Solve for s and t
        try:
            solution, _ = torch.lstsq(A, b)
            s, t = solution[0], solution[1]
        except:
            # Fallback to simple scaling
            s = r_d.mean() / (o_d.mean() + 1e-8)
            t = 0.0

        aligned_depth = s * observed_depth + t
        return aligned_depth

    def compute_differences(self, rendered_image, gt_image, rendered_depth, observed_depth):
        """
        Compute color and depth differences

        Args:
            rendered_image: Rendered RGB image [C, H, W] or [H, W, C]
            gt_image: Ground truth RGB image [C, H, W] or [H, W, C]
            rendered_depth: Rendered depth [H, W] or [1, H, W]
            observed_depth: Aligned observed depth [H, W] or [1, H, W]

        Returns:
            color_diff: Color difference per pixel [H, W]
            depth_diff: Depth difference per pixel [H, W]
        """
        # Ensure both images are in CHW format for consistent processing
        if rendered_image.dim() == 3:
            if rendered_image.shape[2] == 3:  # HWC format
                rendered_image = rendered_image.permute(2, 0, 1)  # Convert to CHW
            if gt_image.shape[2] == 3:  # HWC format
                gt_image = gt_image.permute(2, 0, 1)  # Convert to CHW

        # Ensure depth tensors are 2D [H, W]
        if rendered_depth.dim() == 3:
            rendered_depth = rendered_depth.squeeze(0)  # Remove channel dimension if present
        if observed_depth.dim() == 3:
            observed_depth = observed_depth.squeeze(0)  # Remove channel dimension if present

        # Debug: Print tensor shapes to understand the issue
        # Debug info removed to reduce log noise

        # Color difference (L2 norm along channel dimension)
        # rendered_image and gt_image are now [3, H, W]
        try:
            color_diff = torch.norm(rendered_image - gt_image, dim=0)  # Compute along channel dimension (dim=0)
        except Exception as e:
            print(f"[CDD Error] Color difference computation failed: {e}")
            print(f"[CDD Error] rendered_image: {rendered_image.shape}, gt_image: {gt_image.shape}")
            # Fallback: compute difference channel by channel if needed
            if rendered_image.shape[0] == 3 and gt_image.shape[0] == 3:
                color_diff = torch.sqrt(
                    (rendered_image[0] - gt_image[0])**2 +
                    (rendered_image[1] - gt_image[1])**2 +
                    (rendered_image[2] - gt_image[2])**2
                )
            else:
                raise e

        # Depth difference
        try:
            depth_diff = torch.abs(rendered_depth - observed_depth)
        except Exception as e:
            print(f"[CDD Error] Depth difference computation failed: {e}")
            print(f"[CDD Error] rendered_depth: {rendered_depth.shape}, observed_depth: {observed_depth.shape}")
            raise e

        return color_diff, depth_diff

    def identify_high_error_regions(self, color_diff, depth_diff):
        """
        Identify regions requiring additional Gaussians

        Args:
            color_diff: Color difference map
            depth_diff: Depth difference map

        Returns:
            mask: Boolean mask of high-error pixels
        """
        mask = (color_diff > self.color_threshold) & (depth_diff > self.depth_threshold)
        return mask

    def identify_high_error_regions_constrained(self, color_diff, depth_diff,
                                             max_new_gaussians, error_percentile, min_error_threshold):
        """
        Identify regions requiring additional Gaussians with constraints

        Args:
            color_diff: Color difference map
            depth_diff: Depth difference map
            max_new_gaussians: Maximum number of new Gaussians to add
            error_percentile: Error percentile threshold
            min_error_threshold: Minimum error threshold

        Returns:
            mask: Boolean mask of high-error pixels
        """
        # Basic threshold-based mask
        basic_mask = (color_diff > self.color_threshold) & (depth_diff > self.depth_threshold)

        if basic_mask.sum() == 0:
            return basic_mask

        # Combined error metric
        combined_error = color_diff + depth_diff

        # Apply minimum error threshold
        valid_mask = combined_error > min_error_threshold

        # Combine masks
        candidate_mask = basic_mask & valid_mask

        if candidate_mask.sum() == 0:
            return candidate_mask

        # Get error values for candidate regions
        candidate_errors = combined_error[candidate_mask]

        # If too many candidates, select top percentile
        if len(candidate_errors) > max_new_gaussians:
            # Determine threshold based on percentile
            error_threshold = torch.quantile(candidate_errors, error_percentile / 100.0)
            if hasattr(error_threshold, 'item'):
                error_threshold = error_threshold.item()

            # Create final mask with both percentile and maximum constraints
            high_error_mask = candidate_mask & (combined_error >= error_threshold)

            # If still too many, keep only the highest errors
            if high_error_mask.sum() > max_new_gaussians:
                # Get all error values in the high_error_mask
                high_errors = combined_error[high_error_mask]

                # Get the top K error values
                _, top_indices_in_mask = torch.topk(high_errors, k=max_new_gaussians, largest=True)

                # Get the coordinates of high_error_mask pixels
                y_coords, x_coords = torch.where(high_error_mask)

                # Create final mask with only top K pixels
                final_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
                final_mask[y_coords[top_indices_in_mask], x_coords[top_indices_in_mask]] = True

                return final_mask
            else:
                return high_error_mask
        else:
            return candidate_mask

    def backproject_to_3d(self, pixel_coords, depth, camera_params):
        """
        Back-project 2D pixels to 3D points

        Args:
            pixel_coords: [N, 2] pixel coordinates (u, v)
            depth: [N] depth values
            camera_params: Camera intrinsics

        Returns:
            3D points [N, 3]
        """
        u, v = pixel_coords[:, 0], pixel_coords[:, 1]

        # Extract camera parameters
        fx, fy = camera_params[0, 0], camera_params[1, 1]
        cx, cy = camera_params[0, 2], camera_params[1, 2]

        # Back-projection
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        points_3d = torch.stack([x, y, z], dim=1)
        return points_3d

    def initialize_new_gaussians(self, points_3d, color_diff, depth_diff,
                               existing_gaussians, camera_params):
        """
        Initialize new Gaussian parameters

        Args:
            points_3d: 3D positions for new Gaussians [N, 3]
            color_diff: Color difference values (flattened 1D tensor)
            depth_diff: Depth difference values (flattened 1D tensor)
            existing_gaussians: Existing GaussianModel
            camera_params: Camera intrinsics

        Returns:
            dict: New Gaussian parameters
        """
        num_new = points_3d.shape[0]
        if num_new == 0:
            return {
                'xyz': torch.empty(0, 3),
                'features_dc': torch.empty(0, 3, 1),
                'features_rest': torch.empty(0, 3, 15),
                'opacity': torch.empty(0, 1),
                'scaling': torch.empty(0, 3),
                'rotation': torch.empty(0, 4)
            }

        # Debug info removed to reduce log noise

        # Handle both flattened and potentially non-flattened inputs
        if color_diff.dim() > 1:
            color_diff = color_diff.flatten()
        if depth_diff.dim() > 1:
            depth_diff = depth_diff.flatten()

        # Ensure we have the right number of values
        if len(color_diff) != num_new:
            print(f"[CDD Warning] color_diff length ({len(color_diff)}) != num_new ({num_new})")
            # Adjust or pad as needed
            if len(color_diff) > num_new:
                color_diff = color_diff[:num_new]
            else:
                color_diff = torch.cat([color_diff, torch.zeros(num_new - len(color_diff), device=color_diff.device)])

        if len(depth_diff) != num_new:
            print(f"[CDD Warning] depth_diff length ({len(depth_diff)}) != num_new ({num_new})")
            # Adjust or pad as needed
            if len(depth_diff) > num_new:
                depth_diff = depth_diff[:num_new]
            else:
                depth_diff = torch.cat([depth_diff, torch.zeros(num_new - len(depth_diff), device=depth_diff.device)])

        # Normalize differences for parameter computation
        if color_diff.std() > 1e-8:
            color_norm = (color_diff - color_diff.mean()) / (color_diff.std() + 1e-8)
        else:
            color_norm = torch.zeros_like(color_diff)

        if depth_diff.std() > 1e-8:
            depth_norm = (depth_diff - depth_diff.mean()) / (depth_diff.std() + 1e-8)
        else:
            depth_norm = torch.zeros_like(depth_diff)

        # Find nearest existing Gaussian for parameter inheritance
        with torch.no_grad():
            existing_xyz = existing_gaussians.get_xyz
            if len(existing_xyz) > 0:
                # Compute distances to all existing Gaussians
                dists = torch.cdist(points_3d, existing_xyz)
                nearest_idx = torch.argmin(dists, dim=1)

                # Inherit parameters from nearest Gaussians
                nearest_scaling = existing_gaussians.get_scaling[nearest_idx]
                nearest_rotation = existing_gaussians.get_rotation[nearest_idx]
                nearest_opacity = existing_gaussians.get_opacity[nearest_idx]
                nearest_features_dc = existing_gaussians._features_dc[nearest_idx]
                nearest_features_rest = existing_gaussians._features_rest[nearest_idx]
            else:
                # Fallback to default initialization
                nearest_scaling = torch.ones(num_new, 3) * 0.01
                nearest_rotation = torch.zeros(num_new, 4)
                nearest_rotation[:, 0] = 1.0  # Quaternion identity
                nearest_opacity = torch.ones(num_new, 1) * 0.1
                nearest_features_dc = torch.zeros(num_new, 3, 1)
                nearest_features_rest = torch.zeros(num_new, 3, 15)

        # Initialize parameters
        new_xyz = points_3d

        # Handle scaling - ensure proper dimensions
        scale_factor = (1 - self.alpha_scale * (color_norm + depth_norm) / 2)
        scale_factor = scale_factor.unsqueeze(1)  # [num_new, 1] for proper broadcasting
        new_scaling = nearest_scaling * scale_factor
        new_scaling = torch.clamp(new_scaling, 0.001, 0.1)  # Prevent too small/large scales

        # Handle opacity - ensure proper dimensions
        new_opacity = torch.clamp(
            self.gamma_color * color_norm + self.eta_depth * depth_norm + 0.1,
            0.05, 0.95
        ).unsqueeze(1)  # [num_new, 1]

        # Handle other parameters
        new_rotation = nearest_rotation
        new_features_dc = nearest_features_dc
        new_features_rest = nearest_features_rest

        return {
            'xyz': new_xyz,
            'features_dc': new_features_dc,
            'features_rest': new_features_rest,
            'opacity': new_opacity,
            'scaling': new_scaling,
            'rotation': new_rotation
        }

    def densify_salient_layer(self, gaussians, viewpoint_cam, pipe, background,
                          max_new_gaussians=None, error_percentile=None, min_error_threshold=None):
        """
        Perform CDD densification on salient layer

        Args:
            gaussians: GaussianModel instance
            viewpoint_cam: Camera viewpoint
            pipe: Pipeline parameters
            background: Background color
            max_new_gaussians: Maximum new Gaussians per call (optional)
            error_percentile: Error percentile threshold (optional)
            min_error_threshold: Minimum error threshold (optional)

        Returns:
            dict: New Gaussian parameters to add
        """
        # Use provided parameters or defaults
        max_new = max_new_gaussians if max_new_gaussians is not None else self.max_new_gaussians_per_view
        error_perc = error_percentile if error_percentile is not None else self.error_percentile
        min_error = min_error_threshold if min_error_threshold is not None else self.min_error_threshold

        # Get salient layer Gaussians
        salient_indices = gaussians.get_layer_gaussians(0)  # 0 = salient
        if len(salient_indices) == 0:
            return {}

        # Render current view
        from gaussian_renderer import render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_image = render_pkg["render"]
        rendered_depth = render_pkg["depth"][0]

        # Get ground truth
        gt_image = viewpoint_cam.original_image.to(self.device)

        # Estimate depth from ground truth image
        try:
            observed_depth = estimate_depth(gt_image, mode='train')
            if observed_depth is None:
                return {}

            # Convert to tensor if needed
            if isinstance(observed_depth, np.ndarray):
                observed_depth = torch.from_numpy(observed_depth).float().to(self.device)

            # Align depth scales
            observed_depth = self.align_depth(rendered_depth, observed_depth)

            # Compute differences
            # Pass images in their original format, function will handle conversion
            color_diff, depth_diff = self.compute_differences(
                rendered_image,  # [H, W, 3] from renderer
                gt_image,        # [H, W, 3] from camera
                rendered_depth,
                observed_depth
            )

            # Identify high-error regions with constraints
            high_error_mask = self.identify_high_error_regions_constrained(
                color_diff, depth_diff, max_new, error_perc, min_error
            )

            if high_error_mask.sum() == 0:
                return {}

            # Get pixel coordinates for high-error regions
            y_coords, x_coords = torch.where(high_error_mask)
            pixel_coords = torch.stack([x_coords.float(), y_coords.float()], dim=1)

            # Get depth values for these pixels
            pixel_depths = observed_depth[high_error_mask]

            # Back-project to 3D
            # Get camera intrinsics matrix
            try:
                # Try to get intrinsics from camera
                if hasattr(viewpoint_cam, 'intrinsics'):
                    intrinsics = viewpoint_cam.intrinsics
                else:
                    # Use a default camera intrinsics matrix
                    # This should be adapted based on actual camera parameters
                    H, W = gt_image.shape[:2]
                    intrinsics = torch.tensor([
                        [max(H, W), 0, W/2],
                        [0, max(H, W), H/2],
                        [0, 0, 1]
                    ], dtype=torch.float32, device=observed_depth.device)
            except:
                intrinsics = torch.eye(3, device=observed_depth.device)

            points_3d = self.backproject_to_3d(
                pixel_coords,
                pixel_depths,
                intrinsics
            )

            # Initialize new Gaussians
            new_params = self.initialize_new_gaussians(
                points_3d,
                color_diff[high_error_mask],
                depth_diff[high_error_mask],
                gaussians,
                intrinsics
            )

            return new_params

        except Exception as e:
            print(f"[CDD Warning] Densification failed: {e}")
            return {}

    def densify_multiple_views(self, gaussians, cameras, pipe, background, max_views=3):
        """
        Perform CDD densification from multiple viewpoints

        Args:
            gaussians: GaussianModel instance
            cameras: List of camera viewpoints
            pipe: Pipeline parameters
            background: Background color
            max_views: Maximum number of views to process

        Returns:
            dict: Aggregated new Gaussian parameters
        """
        all_new_params = []

        # Randomly sample views
        if len(cameras) > max_views:
            indices = torch.randperm(len(cameras))[:max_views]
            cameras = [cameras[i] for i in indices]

        for cam in cameras:
            new_params = self.densify_salient_layer(gaussians, cam, pipe, background)
            if new_params and new_params['xyz'].shape[0] > 0:
                all_new_params.append(new_params)

        # Aggregate parameters from all views
        if not all_new_params:
            return {}

        aggregated = {
            'xyz': torch.cat([p['xyz'] for p in all_new_params], dim=0),
            'features_dc': torch.cat([p['features_dc'] for p in all_new_params], dim=0),
            'features_rest': torch.cat([p['features_rest'] for p in all_new_params], dim=0),
            'opacity': torch.cat([p['opacity'] for p in all_new_params], dim=0),
            'scaling': torch.cat([p['scaling'] for p in all_new_params], dim=0),
            'rotation': torch.cat([p['rotation'] for p in all_new_params], dim=0)
        }

        return aggregated


def apply_cdd_densification(gaussians, viewpoint_cam, pipe, background):
    """
    Convenience function to apply CDD densification

    Args:
        gaussians: GaussianModel instance
        viewpoint_cam: Camera viewpoint
        pipe: Pipeline parameters
        background: Background color

    Returns:
        Number of new Gaussians added
    """
    densifier = CDDDensifier(device=gaussians._xyz.device)
    new_params = densifier.densify_salient_layer(gaussians, viewpoint_cam, pipe, background)

    if new_params and new_params['xyz'].shape[0] > 0:
        # Add new Gaussians to the model
        num_new = new_params['xyz'].shape[0]
        gaussians.densification_postfix(
            new_params['xyz'],
            new_params['features_dc'],
            new_params['features_rest'],
            new_params['opacity'],
            new_params['scaling'],
            new_params['rotation']
        )

        # Update importance and layers for new Gaussians
        gaussians.compute_importance()
        gaussians.update_layer_assignments()

        return num_new

    return 0


if __name__ == "__main__":
    print("CDD Densification utilities for OHDGS")
    print("Usage: from utils.cdd_utils import CDDDensifier, apply_cdd_densification")