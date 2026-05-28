# models/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 局部特征提取网络
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # 适配PyTorch 1.8的注意力层
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4)

    def forward(self, x):
        """
        输入: (B, N, 3)
        输出: (B, 256)
        """
        B, N, _ = x.size()

        # 特征提取
        x = self.mlp(x.transpose(1, 2))  # (B, 256, N)
        x = x.transpose(1, 2)  # (B, N, 256)

        # 维度转换适配PyTorch 1.8
        attn_input = x.permute(1, 0, 2)  # (N, B, 256)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # (B, N, 256)

        # 残差连接
        x = x + attn_output
        # 全局特征
        return torch.max(x, dim=1)[0]
