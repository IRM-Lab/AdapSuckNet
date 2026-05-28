# train.py
import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import get_dataloader
from models.graspnet import GraspNet
from losses.hungarian import HungarianLoss
from losses.geometric import GeometricLoss
import datetime
import torch.nn.functional as F


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = GraspNet(
            num_classes=config['model']['num_classes'],
            max_grasps=config['model']['max_grasps']
        ).to(self.device)

        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optim']['lr'],
            weight_decay=config['optim']['weight_decay']
        )

        # 损失函数
        self.hungarian_loss = HungarianLoss()
        self.geo_loss = GeometricLoss()

        # 数据加载
        self.train_loader, self.val_loader = get_dataloader(config)

        # 创建日志和保存目录
        os.makedirs(config['training']['log_dir'], exist_ok=True)
        os.makedirs(config['training']['save_dir'], exist_ok=True)
        self.writer = SummaryWriter(config['training']['log_dir'])
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            # 数据迁移到设备
            points = batch['point_cloud'].to(self.device)
            class_ids = batch['class_ids'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            # 前向传播
            self.optimizer.zero_grad()
            pred_pos, pred_norm, pred_conf = self.model(points, class_ids)

            # 计算损失
            hung_loss = self.hungarian_loss({
                'positions': pred_pos,
                'normals': pred_norm,
                'confidences': pred_conf
            }, targets)

            geo_loss = self.geo_loss(pred_pos, pred_norm)
            total_loss += hung_loss + 0.1 * geo_loss

            # 反向传播
            (hung_loss + 0.1 * geo_loss).backward()
            self.optimizer.step()

            # 记录日志
            self.writer.add_scalar('Loss/train', hung_loss.item(), epoch)
            self.writer.add_scalar('GeoLoss/train', geo_loss.item(), epoch)

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_pos_err = 0.0
        total_norm_sim = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                points = batch['point_cloud'].to(self.device)
                class_ids = batch['class_ids'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                pred_pos, pred_norm, pred_conf = self.model(points, class_ids)
                loss = self.hungarian_loss({
                    'positions': pred_pos,
                    'normals': pred_norm,
                    'confidences': pred_conf
                }, targets)
                total_loss += loss.item()

                # 计算位置误差和法向量相似度
                pos_err = torch.norm(pred_pos - targets['positions'], dim=-1)
                pos_err = pos_err[targets['mask'].bool()].mean()
                norm_sim = F.cosine_similarity(pred_norm, targets['normals'], dim=-1)
                norm_sim = norm_sim[targets['mask'].bool()].mean()

                total_pos_err += pos_err.item() * points.size(0)
                total_norm_sim += norm_sim.item() * points.size(0)
                total_samples += points.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_pos_err = total_pos_err / total_samples
        avg_norm_sim = total_norm_sim / total_samples

        # 记录指标
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Metric/Position_Error', avg_pos_err, epoch)
        self.writer.add_scalar('Metric/Normal_Similarity', avg_norm_sim, epoch)

        # 保存最佳模型
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            save_path = os.path.join(
                self.config['training']['save_dir'],
                'best_model.pth'
            )
            torch.save(self.model.state_dict(), save_path)

        return avg_loss, avg_pos_err, avg_norm_sim

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"Training Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Pos Err(mm)':<12}{'Norm Sim':<12}")
        print('-' * 60)

        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss, pos_err, norm_sim = self.validate(epoch)

            # 格式化输出
            log_str = (
                f"{epoch + 1:03d}/{self.config['training']['epochs']:<6}"
                f"{train_loss:.4f}{'':<6}"
                f"{val_loss:.4f}{'':<6}"
                f"{pos_err:.2f}{'':<8}"
                f"{norm_sim:.4f}"
            )
            tqdm.write(log_str)

        print('=' * 60)
        print(f"Training Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best Validation Loss: {self.best_loss:.4f}")


if __name__ == "__main__":
    with open('configs/graspnet.yaml') as f:
        config = yaml.safe_load(f)

    config['training']['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(config)
    trainer.run()
