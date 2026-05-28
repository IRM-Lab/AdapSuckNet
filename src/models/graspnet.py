# models/graspnet.py
import torch
import torch.nn as nn
from src.models.backbone import LocalFeatureExtractor
from src.models.embedding import CategoryEmbedding
from src.models.fusion import GatedFusion
from src.models.head import GraspPredictHead


class GraspNet(nn.Module):
    def __init__(self, num_classes=10, max_grasps=5):
        super().__init__()
        self.geo_extractor = LocalFeatureExtractor()
        self.sem_embed = CategoryEmbedding(num_classes)
        self.fusion = GatedFusion()
        self.head = GraspPredictHead(max_grasps)

    def forward(self, points, class_ids):
        """
        输入:
            points: (B, N, 3)
            class_ids: (B,)
        输出:
            positions: (B, K, 3)
            normals: (B, K, 3)
            confidences: (B, K)
        """
        geo = self.geo_extractor(points)
        sem = self.sem_embed(class_ids)
        fused = self.fusion(geo, sem)
        return self.head(fused)


# 单元测试
if __name__ == "__main__":
    B, N, K = 2, 1024, 5
    model = GraspNet()
    points = torch.randn(B, N, 3)
    cls_ids = torch.randint(0, 10, (B,))

    pos, norm, conf = model(points, cls_ids)

    assert pos.shape == (B, K, 3)
    assert norm.shape == (B, K, 3)
    assert conf.shape == (B, K)
    print("所有维度校验通过！")
