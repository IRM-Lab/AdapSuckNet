# ----------------------
# utils/preprocess.py
# ----------------------
import open3d as o3d
import numpy as np
import json
import os
import torch


class Preprocessor:
    def __init__(self, norm_param_path="configs/norm_params.json"):
        self.norm_param_path = os.path.abspath(norm_param_path)  # 强制使用绝对路径
        self.mean = None
        self.std = None

    def load_ply(self, path):
        """加载PLY文件并返回毫米单位的点云"""
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)  # 形状为 (N,3)

    def downsample(self, points, target_num=3072):
        """均匀下采样到指定点数"""
        if len(points) >= target_num:
            indices = np.random.choice(len(points), target_num, replace=False)
        else:
            indices = np.arange(len(points))
            # 补零填充
            padding = np.zeros((target_num - len(points), 3))
            points = np.concatenate([points, padding])
        return points[indices]

    def compute_and_save_normalization(self, points):
        """计算并覆盖保存归一化参数"""
        # 输入校验
        assert isinstance(points, np.ndarray), "输入必须为numpy数组"
        assert points.ndim == 2 and points.shape[1] == 3, "输入形状应为 (N,3)"

        # 计算参数
        self.mean = np.mean(points, axis=0).astype(np.float64)
        self.std = np.std(points, axis=0).astype(np.float64)
        self.std[self.std < 1e-8] = 1.0  # 防止除零

        # 强制覆盖保存
        os.makedirs(os.path.dirname(self.norm_param_path), exist_ok=True)
        with open(self.norm_param_path, 'w') as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "unit": "mm"
            }, f, indent=2)
        print(f"归一化参数已覆盖保存至：{self.norm_param_path}")

    def normalize(self, points):
        """应用归一化（输入输出均为毫米单位）"""
        return (points - self.mean) / self.std

    def __call__(self, ply_path):
        """返回归一化后的点云及参数"""
        # 1. 加载原始点云
        raw_points = self.load_ply(ply_path)

        # 2. 下采样
        down_points = self.downsample(raw_points)

        # 3. 计算参数（基于下采样数据）
        self.compute_and_save_normalization(down_points)

        # 4. 应用归一化
        norm_points = self.normalize(down_points)

        return (
            torch.tensor(norm_points, dtype=torch.float32),
            self.mean.copy(),  # 返回均值和标准差
            self.std.copy()
        )
