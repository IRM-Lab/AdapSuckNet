# models/embedding.py
import torch
import torch.nn as nn


class CategoryEmbedding(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 64)
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )

    def forward(self, class_ids):
        """
        输入: (B,)
        输出: (B, 256)
        """
        return self.proj(self.embed(class_ids))
