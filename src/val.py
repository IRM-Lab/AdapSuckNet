# ----------------------
# val.py
# ----------------------
import torch
import yaml
import os
from models.graspnet import GraspNet
from utils.preprocess import Preprocessor
from utils.postprocess import Postprocessor
from utils.visualize import create_grasp_vis, visualize


class Validator:
    def __init__(self, ckpt_path, config):
        # 初始化预处理（强制生成参数）
        self.preprocessor = Preprocessor()

        # 模型加载
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraspNet(
            num_classes=config['model']['num_classes'],
            max_grasps=config['model']['max_grasps']
        ).to(self.device)

        # 加载权重
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        # 后处理初始化
        self.postprocessor = Postprocessor()

    def process(self, ply_path, class_id):
        """完整处理流程"""
        # 1. 预处理（获取点云及参数）
        input_tensor, mean, std = self.preprocessor(ply_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        class_tensor = torch.tensor([class_id], device=self.device)

        # 2. 模型推理
        with torch.no_grad():
            pred_pos, pred_norm, pred_conf = self.model(input_tensor, class_tensor)

        # 3. 后处理（直接传递参数）
        grasps = self.postprocessor(
            pred_pos.squeeze(0).cpu(),
            pred_norm.squeeze(0).cpu(),
            pred_conf.squeeze(0).cpu(),
            mean=mean,
            std=std
        )

        # 4. 可视化
        orig_points = self.preprocessor.load_ply(ply_path)
        geometries = create_grasp_vis(orig_points, grasps)
        visualize(geometries)

        return grasps


if __name__ == "__main__":
    # 加载配置
    config_path = os.path.abspath("configs/graspnet.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 初始化验证器
    validator = Validator(
        ckpt_path=os.path.abspath("checkpoints/best_model.pth"),
        config=config
    )

    # 处理示例点云
    result = validator.process(
        # ply_path=os.path.abspath("/home/user/wff/data/raw/test/clouds/test0.ply"),
        ply_path=os.path.abspath("/home/user/wff/data/raw/train/clouds/radish-6.ply"),
        class_id=9
    )

    # 打印结果
    print("预测结果: ")
    for i, grasp in enumerate(result[:3]):
        print(f"{i}. Position: {grasp['position']}")
        print(f"   Normal: {grasp['normal']}")
        print(f"   Confidence: {grasp['confidence']:.2f}")
