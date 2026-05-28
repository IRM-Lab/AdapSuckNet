# data/dataloader.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GraspDataset(Dataset):
    def __init__(self, root_dir, max_grasps=5):
        self.cloud_dir = os.path.join(root_dir, 'clouds')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.file_list = [f.split('.')[0] for f in os.listdir(self.cloud_dir)]
        self.max_grasps = max_grasps

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]

        # 加载点云
        cloud = np.load(os.path.join(self.cloud_dir, base_name + '.npy'))
        cloud = torch.from_numpy(cloud).float()  # (N,3)

        # 加载标注
        with open(os.path.join(self.label_dir, base_name + '.json')) as f:
            label = json.load(f)

        # 处理抓取位姿
        grasps = label['grasps']
        M = len(grasps)
        positions = np.zeros((self.max_grasps, 3))
        normals = np.zeros((self.max_grasps, 3))
        mask = np.zeros(self.max_grasps)

        for i in range(min(M, self.max_grasps)):
            positions[i] = grasps[i]['position']
            normals[i] = grasps[i]['normal']
            mask[i] = 1.0

        return {
            'point_cloud': cloud,
            'class_id': torch.tensor(label['classID'], dtype=torch.long),
            'target_pos': torch.from_numpy(positions).float(),
            'target_norm': torch.from_numpy(normals).float(),
            'target_mask': torch.from_numpy(mask).float()
        }


def collate_fn(batch):
    # 处理变长点云
    clouds = [item['point_cloud'] for item in batch]
    class_ids = torch.stack([item['class_id'] for item in batch])

    # 零填充到最大点数
    N = max(c.shape[0] for c in clouds)
    padded_clouds = []
    for c in clouds:
        if c.shape[0] < N:
            pad = torch.zeros(N - c.shape[0], 3)
            padded_clouds.append(torch.cat([c, pad]))
        else:
            padded_clouds.append(c[:N])

    return {
        'point_cloud': torch.stack(padded_clouds),
        'class_ids': class_ids,
        'targets': {
            'positions': torch.stack([item['target_pos'] for item in batch]),
            'normals': torch.stack([item['target_norm'] for item in batch]),
            'mask': torch.stack([item['target_mask'] for item in batch])
        }
    }


def get_dataloader(config):
    train_set = GraspDataset(
        config['data']['train_path'],
        config['model']['max_grasps']
    )
    val_set = GraspDataset(
        config['data']['val_path'],
        config['model']['max_grasps']
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    return train_loader, val_loader
