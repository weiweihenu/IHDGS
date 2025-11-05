"""
简化的NOD实现
基于CDD的克隆方法，但只在法向量平面上生成新高斯
"""

import torch
import numpy as np
from scipy.linalg import eigh
import math


class SimpleNODDensifier:
    """
    简化的Normal-Orthogonal Plane Densification
    参考CDD的克隆方法，但只在法向量平面生成
    """

    def __init__(self, device='cuda'):
        self.device = device

    def get_normal_from_gaussian(self, scaling, rotation):
        """从单个高斯的scaling和rotation获取法向量"""
        # scaling: [1, 3] or [3]
        # rotation: [1, 4] or [4]

        if len(scaling.shape) > 1:
            scales = scaling[0]
            quat = rotation[0]
        else:
            scales = scaling
            quat = rotation

        # 四元数转旋转矩阵 (假设格式为 [x, y, z, w])
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]

        # 标准化四元数
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm

        # 构建旋转矩阵
        R = torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], device=self.device, dtype=torch.float32)

        # 协方差矩阵 C = R * diag(scales^2) * R^T
        C = R @ torch.diag(scales**2) @ R.t()

        # 特征分解，最小特征值对应的特征向量就是法向量
        C_np = C.cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(C_np)
        normal_idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, normal_idx]

        return torch.tensor(normal, device=self.device, dtype=torch.float32)

    def densify_single_gaussian(self, gaussians, idx):
        """
        致密化单个高斯，在法向量平面上生成一个新高斯

        Returns:
            dict: 新高斯的参数，或None如果失败
        """
        try:
            # 获取高斯参数
            xyz = gaussians.get_xyz[idx:idx+1]  # [1, 3]
            scaling = gaussians.get_scaling[idx:idx+1]  # [1, 3]
            rotation = gaussians.get_rotation[idx:idx+1]  # [1, 4]
            opacity = gaussians.get_opacity[idx:idx+1]  # [1, 1]
            features_dc = gaussians._features_dc[idx:idx+1]  # [1, 3, 1]
            features_rest = gaussians._features_rest[idx:idx+1]  # [1, (max_sh_degree+1)^2-3, 1]

            # 获取法向量
            normal = self.get_normal_from_gaussian(scaling, rotation)

            # 在法向量平面上生成新位置
            # 使用CDD中的offset方法，但限制在法向量平面
            offset_scale = scaling.mean() * 0.5  # 使用平均缩放的一半作为偏移距离

            # 生成两个正交向量
            if abs(normal[0]) < 0.9:
                u = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                u = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # Gram-Schmidt正交化
            u = u - torch.dot(u, normal) * normal
            u = u / (torch.norm(u) + 1e-8)
            v = torch.cross(normal, u)
            v = v / (torch.norm(v) + 1e-8)

            # 在平面上随机采样位置
            theta = torch.rand(1, device=self.device) * 2 * math.pi
            # 使用更小的偏移，避免位置不合理
            offset = offset_scale * 0.1 * (torch.cos(theta) * u + torch.sin(theta) * v)
            new_xyz = xyz + offset

            # 确保新位置在合理范围内
            if torch.any(torch.isnan(new_xyz)) or torch.any(torch.isinf(new_xyz)):
                return None

            # 复制其他属性（同CDD）
            new_features_dc = features_dc.clone()
            new_features_rest = features_rest.clone()
            new_opacity = opacity.clone()
            new_scaling = scaling.clone()
            new_rotation = rotation.clone()

            return {
                'xyz': new_xyz,
                'features_dc': new_features_dc,
                'features_rest': new_features_rest,
                'opacity': new_opacity,
                'scaling': new_scaling,
                'rotation': new_rotation
            }

        except Exception as e:
            print(f"[SimpleNOD] Failed to densify Gaussian {idx}: {e}")
            return None

    def densify_transition_layer(self, gaussians, max_new_gaussians=300):
        """
        致密化过渡层

        Args:
            gaussians: GaussianModel instance
            max_new_gaussians: 最多生成的新高斯点数量（默认300，与CDD一致）
        """
        # 获取过渡层高斯索引
        transition_indices = gaussians.get_layer_gaussians(1)
        if len(transition_indices) == 0:
            return {}

        # 使用固定的最大值，而不是百分比
        # 避免生成过多高斯点导致内存问题
        max_new = max_new_gaussians

        # 随机选择要致密化的高斯
        num_to_process = min(len(transition_indices), max_new)
        if num_to_process == 0:
            return {}

        # 随机采样
        perm = torch.randperm(len(transition_indices))
        selected_indices = transition_indices[perm[:num_to_process]]

        # 收集新高斯
        new_gaussians = []
        for idx in selected_indices:
            new_gaussian = self.densify_single_gaussian(gaussians, idx.item())
            if new_gaussian is not None:
                new_gaussians.append(new_gaussian)

        # 聚合结果
        if not new_gaussians:
            return {}

        aggregated = {
            'xyz': torch.cat([g['xyz'] for g in new_gaussians], dim=0),
            'features_dc': torch.cat([g['features_dc'] for g in new_gaussians], dim=0),
            'features_rest': torch.cat([g['features_rest'] for g in new_gaussians], dim=0),
            'opacity': torch.cat([g['opacity'] for g in new_gaussians], dim=0),
            'scaling': torch.cat([g['scaling'] for g in new_gaussians], dim=0),
            'rotation': torch.cat([g['rotation'] for g in new_gaussians], dim=0)
        }

        # 参数有效性检查，避免NaN/Inf导致GPU内存错误
        for key, tensor in aggregated.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[NOD Warning] {key} contains NaN/Inf values, skipping this batch")
                return {}

        return aggregated


def apply_nod_densification(gaussians, max_new_gaussians=300):
    """
    应用NOD致密化的便捷函数

    Args:
        gaussians: GaussianModel instance
        max_new_gaussians: 最多生成的新高斯点数量（默认300）
    """
    densifier = SimpleNODDensifier()
    new_params = densifier.densify_transition_layer(gaussians, max_new_gaussians=max_new_gaussians)

    if new_params and new_params['xyz'].shape[0] > 0:
        # print moved to hierarchical_densification.py

        # 添加到模型
        gaussians.densification_postfix(
            new_params['xyz'],
            new_params['features_dc'],
            new_params['features_rest'],
            new_params['opacity'],
            new_params['scaling'],
            new_params['rotation']
        )

        # 更新层级分配
        gaussians.update_layer_assignments()

        return new_params['xyz'].shape[0]

    return 0