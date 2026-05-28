# losses/geometric.py
import torch
import torch.nn as nn


class GeometricLoss(nn.Module):
    def __init__(self, min_dist=0.02, ortho_weight=0.1):
        super().__init__()
        self.min_dist = min_dist
        self.ortho_weight = ortho_weight

    def forward(self, positions, normals):
        """
        输入:
            positions: (B, K, 3) 预测抓取点坐标
            normals: (B, K, 3) 预测法向量
        输出: 标量损失值
        """
        B, K, _ = positions.size()

        # 间距约束
        dists = torch.cdist(positions, positions)  # (B, K, K)
        eye = torch.eye(K, device=positions.device).unsqueeze(0)
        dist_loss = torch.mean((dists < self.min_dist).float() * (1 - eye))

        # 法向量正交约束
        cos_sim = torch.bmm(normals, normals.transpose(1, 2))  # (B, K, K)
        ortho_loss = torch.mean(torch.abs(cos_sim) * (1 - eye))

        return dist_loss + self.ortho_weight * ortho_loss
