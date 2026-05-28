import numpy as np
import json
import os


class PointCloudNormalizer:
    def __init__(self, save_path=None):
        """
        点云归一化处理器
        :param save_path: 归一化参数保存路径（可选）
        """
        self.mean = None
        self.std = None
        self.save_path = save_path

    def fit(self, point_cloud):
        """
        计算归一化参数
        :param point_cloud: (N,3) numpy数组 原始点云坐标（毫米单位）
        """
        # 输入校验
        assert isinstance(point_cloud, np.ndarray), "输入必须为numpy数组"
        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "输入形状应为(N,3)"

        # 转换为米单位计算
        point_cloud_m = point_cloud

        self.mean = np.mean(point_cloud_m, axis=0)
        self.std = np.std(point_cloud_m, axis=0)

        # 防止除零
        self.std[self.std < 1e-8] = 1.0

        # 自动保存参数
        if self.save_path:
            self._save_params()

    def transform(self, point_cloud):
        """
        应用归一化转换
        :param point_cloud: (N,3) numpy数组 原始点云坐标（毫米单位）
        :return: 归一化后的点云（numpy数组）
        """
        assert self.mean is not None and self.std is not None, "请先调用fit方法"
        point_cloud_m = point_cloud
        return (point_cloud_m - self.mean) / self.std

    def _save_params(self):
        """保存归一化参数到文件"""
        params = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "unit": "meters"  # 明确参数计算单位
        }

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        with open(self.save_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"归一化参数已保存至：{self.save_path}")

    @classmethod
    def load(cls, param_path):
        """
        从文件加载归一化参数
        :return: PointCloudNormalizer实例
        """
        with open(param_path) as f:
            params = json.load(f)

        normalizer = cls()
        normalizer.mean = np.array(params['mean'])
        normalizer.std = np.array(params['std'])
        return normalizer


# 使用示例
if __name__ == "__main__":
    # 生成示例数据（毫米单位）
    raw_pc = np.random.rand(1000, 3) * 1000  # 0-1000mm范围

    # 初始化处理器（自动保存参数）
    normalizer = PointCloudNormalizer(save_path="configs/norm_params.json")

    # 计算并保存参数
    normalizer.fit(raw_pc)

    # 应用归一化
    normalized_pc = normalizer.transform(raw_pc)

    # 验证归一化效果
    print("归一化后均值：", np.mean(normalized_pc, axis=0))  # 应接近 [0,0,0]
    print("归一化后方差：", np.std(normalized_pc, axis=0))  # 应接近 [1,1,1]
