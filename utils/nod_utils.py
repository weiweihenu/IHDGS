"""
Normal-Orthogonal Plane Densification (NOD) for OHDGS
Implements Section 3.4.2: Transition Layer Densification Strategy
"""

import torch
import numpy as np
from scipy.linalg import eigh
import math


class NODDensifier:
    """
    Normal-Orthogonal Plane Densification for transition layer
    """

    def __init__(self, device='cuda'):
        self.device = device

    def extract_surface_normal(self, gaussian):
        """
        Extract surface normal from Gaussian covariance matrix

        Args:
            gaussian: GaussianModel instance or specific Gaussian parameters

        Returns:
            normals: Surface normals [N, 3]
        """
        # Get covariance matrices
        if hasattr(gaussian, 'get_scaling'):
            # GaussianModel instance
            scales = gaussian.get_scaling
            rotation = gaussian.get_rotation

            # Build covariance matrices
            L = torch.bmm(
                rotation,
                torch.diag_embed(scales)
            )
            covariances = torch.bmm(L, L.transpose(1, 2))
        else:
            # Direct covariance matrices
            covariances = gaussian

        # Eigendecomposition to find principal axes
        # The eigenvector with smallest eigenvalue corresponds to surface normal
        normals = []
        for i in range(covariances.shape[0]):
            cov = covariances[i].cpu().numpy()
            eigenvalues, eigenvectors = eigh(cov)

            # Sort by eigenvalues (ascending)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Surface normal is eigenvector with smallest eigenvalue
            normal = eigenvectors[:, 0]
            normals.append(normal)

        return torch.tensor(np.array(normals), dtype=torch.float32).to(self.device)

    def build_orthogonal_basis(self, normal):
        """
        Build orthonormal basis for plane perpendicular to normal

        Args:
            normal: Surface normal vector [3] or [N, 3]

        Returns:
            t1, t2: Two orthogonal tangent vectors
        """
        if normal.dim() == 1:
            # Single normal vector
            n = normal
            e_x = torch.tensor([1.0, 0.0, 0.0], device=self.device)

            # Handle case where normal is parallel to e_x
            if torch.allclose(torch.abs(torch.dot(n, e_x)), torch.tensor(1.0, device=self.device)):
                e_x = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # First tangent vector
            t1 = torch.cross(n, e_x)
            t1 = t1 / (torch.norm(t1) + 1e-8)

            # Second tangent vector (perpendicular to both n and t1)
            t2 = torch.cross(n, t1)
            t2 = t2 / (torch.norm(t2) + 1e-8)

            return t1, t2
        else:
            # Multiple normal vectors
            n = normal
            e_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(n)

            # Handle parallel case
            parallel_mask = torch.allclose(torch.abs(torch.sum(n * e_x, dim=1)), torch.tensor(1.0, device=self.device))
            e_x[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # First tangent vectors
            t1 = torch.cross(n, e_x)
            t1 = t1 / (torch.norm(t1, dim=1, keepdim=True) + 1e-8)

            # Second tangent vectors
            t2 = torch.cross(n, t1)
            t2 = t2 / (torch.norm(t2, dim=1, keepdim=True) + 1e-8)

            return t1, t2

    def sample_on_plane(self, center, normal, scaling, num_samples=4):
        """
        Sample new Gaussian positions on orthogonal plane

        Args:
            center: Center position [3]
            normal: Surface normal [3]
            scaling: Scaling factors [3]
            num_samples: Number of samples (default: 4 for 90-degree intervals)

        Returns:
            new_positions: New positions on plane [num_samples, 3]
        """
        # Build orthogonal basis
        t1, t2 = self.build_orthogonal_basis(normal)

        # Use average scale as sampling radius
        radius = scaling.mean()

        # Sample at regular angles
        angles = torch.linspace(0, 2 * math.pi, num_samples, endpoint=False, device=self.device)

        new_positions = []
        for angle in angles:
            offset = radius * (torch.cos(angle) * t1 + torch.sin(angle) * t2)
            new_pos = center + offset
            new_positions.append(new_pos)

        return torch.stack(new_positions, dim=0)

    def densify_transition_gaussian(self, gaussian_idx, gaussians):
        """
        Densify a single transition layer Gaussian

        Args:
            gaussian_idx: Index of Gaussian to densify
            gaussians: GaussianModel instance

        Returns:
            dict: New Gaussian parameters
        """
        # Input validation
        if gaussian_idx < 0 or gaussian_idx >= len(gaussians.get_xyz):
            return {}

        # Get Gaussian parameters
        xyz = gaussians.get_xyz[gaussian_idx:gaussian_idx+1]
        scaling = gaussians.get_scaling[gaussian_idx:gaussian_idx+1]
        rotation = gaussians.get_rotation[gaussian_idx:gaussian_idx+1]
        opacity = gaussians.get_opacity[gaussian_idx:gaussian_idx+1]
        features_dc = gaussians._features_dc[gaussian_idx:gaussian_idx+1]
        features_rest = gaussians._features_rest[gaussian_idx:gaussian_idx+1]

        # Validate tensors
        if xyz.shape[0] == 0 or scaling.shape[0] == 0:
            return {}

        # Extract surface normal
        # Directly compute normal for this single Gaussian
        scales = scaling[0]  # Get [3] from [1, 3]
        rotation_quat = rotation[0]  # Get [4] from [1, 4]

        # Convert quaternion to rotation matrix
        # Using quaternion format: [w, x, y, z] or [x, y, z, w]?
        # Assuming format is [x, y, z, w] as in Gaussian Splatting
        x, y, z, w = rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3]

        # Build rotation matrix
        rotation_matrix = torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], device=scaling.device, dtype=torch.float32)

        # Build covariance matrix using the formula: C = R * diag(s^2) * R^T
        cov = rotation_matrix @ torch.diag(scales**2) @ rotation_matrix.t()

        # Eigendecomposition
        cov_np = cov.cpu().numpy()
        eigenvalues, eigenvectors = eigh(cov_np)

        # Sort by eigenvalues (ascending)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Surface normal is eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]
        normal = torch.tensor(normal, dtype=torch.float32, device=scaling.device)

        # Sample new positions on orthogonal plane
        try:
            new_xyz = self.sample_on_plane(xyz[0], normal, scaling[0])
        except Exception as e:
            print(f"[NOD Warning] Failed to sample positions for Gaussian {gaussian_idx}: {e}")
            return {}

        # Create new Gaussian parameters (inherit from parent)
        num_new = new_xyz.shape[0]

        new_params = {
            'xyz': new_xyz,
            'features_dc': features_dc.repeat(num_new, 1, 1),
            'features_rest': features_rest.repeat(num_new, 1, 1),
            'opacity': opacity.repeat(num_new, 1),
            'scaling': scaling.repeat(num_new, 1),
            'rotation': rotation.repeat(num_new, 1)
        }

        return new_params

    def densify_transition_layer(self, gaussians, max_density=10000):
        """
        Densify all Gaussians in the transition layer

        Args:
            gaussians: GaussianModel instance
            max_density: Maximum total density after densification

        Returns:
            dict: All new Gaussian parameters
        """
        # Get transition layer Gaussians
        transition_indices = gaussians.get_layer_gaussians(1)  # 1 = transition
        if len(transition_indices) == 0:
            return {}

        current_total = len(gaussians.get_xyz)

        # 新增高斯数量不超过模型总数的5%
        max_new_gaussians = int(current_total * 0.05)

        # 计算基于5%限制的最大可处理数量
        # 每个高斯致密化会产生3个新高斯，所以最多处理 max_new_gaussians // 3 个
        max_process = min(len(transition_indices), max_new_gaussians // 3)
        if max_process == 0:
            return {}

        # Randomly sample Gaussians to densify
        if len(transition_indices) > max_process:
            indices = torch.randperm(len(transition_indices))[:max_process]
            transition_indices = [transition_indices[i].item() if hasattr(transition_indices[i], 'item') else transition_indices[i] for i in indices]

        all_new_params = []

        for idx in transition_indices:
            try:
                new_params = self.densify_transition_gaussian(idx, gaussians)
                if new_params and isinstance(new_params, dict):
                    xyz_tensor = new_params.get('xyz')
                    if xyz_tensor is not None and isinstance(xyz_tensor, torch.Tensor) and xyz_tensor.shape[0] > 0:
                        all_new_params.append(new_params)
            except Exception as e:
                print(f"[NOD Warning] Failed to densify Gaussian {idx}: {e}")
                continue

        # Aggregate all new parameters
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

    def adaptive_densification(self, gaussians, iteration_interval=1000,
                             start_iteration=1000, end_iteration=15000):
        """
        Perform adaptive NOD densification based on training progress

        Args:
            gaussians: GaussianModel instance
            iteration_interval: Frequency of densification
            start_iteration: When to start densification
            end_iteration: When to stop densification

        Returns:
            Number of new Gaussians added
        """
        # Get current iteration from training
        current_iteration = getattr(gaussians, 'current_iteration', 0)

        # Check if we should densify
        if (current_iteration < start_iteration or
            current_iteration > end_iteration or
            current_iteration % iteration_interval != 0):
            return 0

        # Get transition layer statistics
        stats = gaussians.get_layer_statistics()
        transition_count = stats.get('transition', {}).get('count', 0)

        # Skip if no transition layer Gaussians
        if transition_count == 0:
            return 0

        # Perform densification
        new_params = self.densify_transition_layer(gaussians)

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

            print(f"[NOD] Added {num_new} Gaussians to transition layer")
            return num_new

        return 0


def apply_nod_densification(gaussians):
    """
    Convenience function to apply NOD densification

    Args:
        gaussians: GaussianModel instance

    Returns:
        Number of new Gaussians added
    """
    densifier = NODDensifier(device=gaussians._xyz.device)
    return densifier.densify_transition_layer(gaussians)


if __name__ == "__main__":
    print("NOD Densification utilities for OHDGS")
    print("Usage: from utils.nod_utils import NODDensifier, apply_nod_densification")