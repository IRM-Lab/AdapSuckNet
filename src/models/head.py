# models/head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspPredictHead(nn.Module):
    def __init__(self, max_grasps=5):
        super().__init__()
        self.max_grasps = max_grasps
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.pos = nn.Linear(64, max_grasps * 3)
        self.normal = nn.Linear(64, max_grasps * 3)
        self.conf = nn.Sequential(
            nn.Linear(64, max_grasps),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入: (B, 256)
        输出:
            positions: (B, K, 3)
            normals: (B, K, 3)
            confidences: (B, K)
        """
        x = self.shared(x)
        positions = self.pos(x).view(-1, self.max_grasps, 3)
        normals = F.normalize(self.normal(x).view(-1, self.max_grasps, 3), dim=-1)
        confs = self.conf(x)
        return positions, normals, confs
