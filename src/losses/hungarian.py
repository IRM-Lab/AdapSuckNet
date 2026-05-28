# losses/hungarian.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianLoss(nn.Module):
    def __init__(self, pos_weight=1.0, norm_weight=0.5, conf_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.norm_weight = norm_weight
        self.conf_weight = conf_weight

    def forward(self, pred, target):
        """
        输入:
            pred: {
                'positions': (B, K, 3),
                'normals': (B, K, 3),
                'confidences': (B, K)
            }
            target: {
                'positions': (B, M, 3),
                'normals': (B, M, 3),
                'mask': (B, M)  # 有效位姿掩码
            }
        输出: 标量损失值
        """
        assert pred['positions'].shape[-1] == 3, "位置预测维度错误"
        assert target['positions'].shape[-1] == 3, "目标位置维度错误"

        B, K, _ = pred['positions'].size()
        total_loss = 0.0

        for b in range(B):
            # 提取有效真实位姿
            valid_mask = target['mask'][b].bool()
            gt_pos = target['positions'][b][valid_mask]  # (N, 3)
            gt_norm = target['normals'][b][valid_mask]  # (N, 3)
            N = gt_pos.size(0)

            if N == 0:
                # 处理无有效标注情况
                conf_loss = F.binary_cross_entropy(
                    pred['confidences'][b],
                    torch.zeros_like(pred['confidences'][b])
                )
                total_loss += self.conf_weight * conf_loss
                continue

            pred_pos = pred['positions'][b]  # (K, 3)
            pred_norm = pred['normals'][b]  # (K, 3)

            # 构建成本矩阵
            pos_cost = torch.cdist(pred_pos, gt_pos)  # (K, N)
            norm_sim = F.cosine_similarity(
                pred_norm.unsqueeze(1),
                gt_norm.unsqueeze(0),
                dim=-1
            )  # (K, N)
            norm_cost = 1 - norm_sim
            cost_matrix = self.pos_weight * pos_cost + self.norm_weight * norm_cost

            # 匈牙利匹配
            with torch.no_grad():
                cost_np = cost_matrix.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)
                if K < N:  # 处理预测数少于真实数的情况
                    row_idx = np.concatenate([row_idx, np.arange(K, N)])

            # 计算匹配损失
            matched_pos = pred_pos[row_idx[:N]]
            matched_norm = pred_norm[row_idx[:N]]

            pos_loss = F.mse_loss(matched_pos, gt_pos)
            norm_loss = 1 - F.cosine_similarity(matched_norm, gt_norm).mean()

            # 置信度损失
            conf_target = torch.zeros(K, device=pred_pos.device)
            conf_target[row_idx[:N]] = 1.0
            conf_loss = F.binary_cross_entropy(
                pred['confidences'][b],
                conf_target
            )

            total_loss += (
                    self.pos_weight * pos_loss +
                    self.norm_weight * norm_loss +
                    self.conf_weight * conf_loss
            )

        return total_loss / B
