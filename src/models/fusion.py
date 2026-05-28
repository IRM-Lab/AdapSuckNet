# models/fusion.py
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, geo_feat, sem_feat):
        """
        输入:
            geo_feat: (B, 256)
            sem_feat: (B, 256)
        输出: (B, 256)
        """
        combined = torch.cat([geo_feat, sem_feat], dim=1)
        gates = self.gate(combined)  # (B, 2)
        return gates[:, 0:1] * geo_feat + gates[:, 1:2] * sem_feat
