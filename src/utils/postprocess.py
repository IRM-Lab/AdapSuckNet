# ----------------------
# utils/postprocess.py
# ----------------------
import numpy as np
import torch


class Postprocessor:
    def __init__(self):
        """
        后处理器（不再从文件加载参数）
        """
        self.mean = None
        self.std = None
        self.nms_thresh = 5  # 毫米单位
        self.z_range = (500, 2000)  # 毫米单位
        print("后处理器初始化完成（参数需动态设置）")

    def set_normalization_params(self, mean, std):
        """
        设置归一化参数（必须在使用前调用）
        :param mean: 归一化均值 (3,)
        :param std:  归一化标准差 (3,)
        """
        # 输入校验
        assert mean.shape == (3,), "均值参数形状应为(3,)"
        assert std.shape == (3,), "标准差参数形状应为(3,)"

        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        print(f"后处理参数已设置 - 均值: {self.mean}, 标准差: {self.std}")

    def denormalize(self, norm_points):
        """
        反归一化到原始毫米坐标
        :param norm_points: 归一化后的点坐标 (N,3)
        :return: 原始毫米坐标 (N,3)
        """
        # 参数校验
        if self.mean is None or self.std is None:
            raise RuntimeError("请先调用set_normalization_params设置参数")

        return norm_points * self.std + self.mean

    def nms(self, positions, scores):
        """非极大值抑制（毫米单位）"""
        keep = []
        sorted_ids = np.argsort(scores)[::-1]  # 按置信度降序排序

        while len(sorted_ids) > 0:
            keep_id = sorted_ids[0]
            keep.append(keep_id)

            # 计算欧氏距离
            dists = np.linalg.norm(
                positions[keep_id] - positions[sorted_ids],
                axis=1
            )

            # 移除邻近项
            remove_mask = dists < self.nms_thresh
            sorted_ids = sorted_ids[~remove_mask]

        return np.array(keep)

    def physical_check(self, positions, normals):
        """物理约束检查（毫米单位）"""
        valid_mask = []
        for pos, norm in zip(positions, normals):
            # Z轴高度检查
            z_valid = self.z_range[0] < pos[2] < self.z_range[1]

            # 法向量垂直分量检查
            normal_z = norm[2] / (np.linalg.norm(norm) + 1e-8)
            normal_valid = normal_z > 0.6

            valid_mask.append(z_valid and normal_valid)

        return np.array(valid_mask)

    def __call__(self, pred_pos, pred_norm, pred_conf, mean, std):
        """
        后处理主函数
        输入:
            pred_pos: (N,3) 归一化预测位置
            pred_norm: (N,3) 单位法向量
            pred_conf: (N,) 置信度
            mean: 归一化均值 (3,)
            std: 归一化标准差 (3,)
        输出:
            List[Dict] 处理后的抓取位姿
        """
        # 设置归一化参数
        self.set_normalization_params(mean, std)

        # 转换为numpy数组
        pred_pos_np = pred_pos.numpy() if isinstance(pred_pos, torch.Tensor) else pred_pos
        pred_norm_np = pred_norm.numpy() if isinstance(pred_norm, torch.Tensor) else pred_norm
        pred_conf_np = pred_conf.numpy() if isinstance(pred_conf, torch.Tensor) else pred_conf

        # 反归一化到毫米坐标
        positions_mm = self.denormalize(pred_pos_np)

        # 置信度过滤 (阈值0.5)
        valid_mask = pred_conf_np > 0.5
        positions = positions_mm[valid_mask]
        normals = pred_norm_np[valid_mask]
        scores = pred_conf_np[valid_mask]

        if len(positions) == 0:
            return []

        # 非极大值抑制
        keep_ids = self.nms(positions, scores)
        positions = positions[keep_ids]
        normals = normals[keep_ids]
        scores = scores[keep_ids]

        # 物理约束过滤
        valid_mask = self.physical_check(positions, normals)

        # 格式转换
        results = []
        for pos, norm, score in zip(
                positions[valid_mask],
                normals[valid_mask],
                scores[valid_mask]
        ):
            results.append({
                "position": pos.round(3).tolist(),  # 毫米单位保留3位小数
                "normal": norm.round(4).tolist(),  # 单位向量保留4位小数
                "confidence": float(round(score, 2))  # 置信度保留2位小数
            })

        return results
