import os
import json
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def uniform_sampling_with_annotations(points, annotations, target_points=3072):
    """保证标注点不被去除的均匀采样"""
    grasp_positions = np.array([g['position'] for g in annotations['grasps']])

    nbrs = NearestNeighbors(n_neighbors=1).fit(points)
    _, indices = nbrs.kneighbors(grasp_positions)
    grasp_indices = np.unique(indices.flatten())

    required_points = points[grasp_indices]
    remaining_indices = np.setdiff1d(np.arange(len(points)), grasp_indices)

    need_points = target_points - len(required_points)
    if need_points <= 0:
        return required_points[:target_points]

    if len(remaining_indices) > need_points:
        selected = np.random.choice(remaining_indices, need_points, replace=False)
    else:
        selected = np.random.choice(remaining_indices, need_points, replace=True)

    return np.concatenate([required_points, points[selected]])


def normalize_point_cloud(pc):
    """归一化到单位球体"""
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    max_distance = np.max(np.linalg.norm(pc, axis=1))
    pc /= max_distance
    return pc, centroid, max_distance


def rotate_z(points, angle):
    """绕Z轴旋转"""
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return points @ rot_mat.T


def process_single_file(raw_path, label_path, save_path, augment_angles, target_points=3072):
    """处理单个文件，支持自定义增强角度"""
    # 读取数据
    ply_data = PlyData.read(raw_path)
    vertex = ply_data['vertex']
    original_points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    with open(label_path) as f:
        annotations = json.load(f)

    # 采样处理
    sampled_points = uniform_sampling_with_annotations(original_points, annotations, target_points)

    # 确保点数准确
    if len(sampled_points) < target_points:
        indices = np.random.choice(len(sampled_points),
                                   target_points - len(sampled_points),
                                   replace=True)
        sampled_points = np.concatenate([sampled_points, sampled_points[indices]])
    elif len(sampled_points) > target_points:
        sampled_points = sampled_points[:target_points]

    # 归一化
    normalized_points, centroid, scale = normalize_point_cloud(sampled_points)

    # 数据增强（使用自定义角度）
    augmented_data = []
    for angle in augment_angles:
        rotated_points = rotate_z(normalized_points, angle)
        augmented_data.append(rotated_points)

    # 保存点云
    base_name = os.path.basename(raw_path).split('.')[0]
    for i, data in enumerate(augmented_data):
        np.save(os.path.join(save_path, 'clouds', f'{base_name}_aug{i}.npy'), data.astype(np.float32))

    return centroid, scale, annotations


def process_annotation(annotations, save_path, centroid, scale, base_name, augment_angles):
    """处理标注，与点云增强角度对应"""
    processed_grasps = []
    for grasp in annotations['grasps']:
        position = (np.array(grasp['position']) - centroid) / scale
        normal = np.array(grasp['normal'])
        normal_norm = np.linalg.norm(normal)
        normal = normal / (normal_norm + 1e-6)

        # 应用所有增强角度
        for angle in augment_angles:
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            rotated_pos = rotate_z(position, angle)
            rotated_normal = rot_mat @ normal
            rotated_normal /= (np.linalg.norm(rotated_normal) + 1e-6)

            processed_grasps.append({
                'position': rotated_pos.tolist(),
                'normal': rotated_normal.tolist()
            })

    # 按增强角度数量分割标注
    num_aug = len(augment_angles)
    for i in range(num_aug):
        new_ann = {
            "classID": annotations["classID"],
            "grasps": processed_grasps[i::num_aug],
            # 新增归一化参数字段
            # "normalization": {
            #     "centroid": centroid.tolist(),  # 转换为list类型
            #     "scale": float(scale)           # 转换为Python原生float
            # }
        }
        with open(os.path.join(save_path, 'labels', f'{base_name}_aug{i}.json'), 'w') as f:
            json.dump(new_ann, f, indent=2)


def batch_processing(raw_root, processed_root, augment_angles=None):
    """批量处理，可传入自定义增强角度列表"""
    # 默认增强角度（当不指定时使用）
    if augment_angles is None:
        augment_angles = np.linspace(0, 2 * np.pi, 4)[:-1]  # 默认3个角度

    for phase in ['train', 'val', 'test']:
        raw_dir = os.path.join(raw_root, phase)
        processed_dir = os.path.join(processed_root, phase)

        os.makedirs(os.path.join(processed_dir, 'clouds'), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, 'labels'), exist_ok=True)

        cloud_dir = os.path.join(raw_dir, 'clouds')
        label_dir = os.path.join(raw_dir, 'labels')

        files = [f for f in os.listdir(cloud_dir) if f.endswith('.ply')]
        for filename in tqdm(files, desc=f'Processing {phase} data'):
            raw_path = os.path.join(cloud_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.ply', '.json'))

            if not os.path.exists(label_path):
                continue

            centroid, scale, ann = process_single_file(
                raw_path, label_path, processed_dir, augment_angles)

            base_name = filename.split('.')[0]
            process_annotation(ann, processed_dir, centroid, scale,
                               base_name, augment_angles)


if __name__ == '__main__':
    # 示例用法：自定义增强角度（单位：弧度）
    custom_angles = [
        0,  # 0度
        np.radians(10),  # 45度
        np.radians(20),  # 45度
        np.radians(30),  # 45度
        np.radians(-10),  # 45度
        np.radians(-20),  # 45度
        np.radians(-30),  # 45度
    ]

    batch_processing(
        raw_root='/home/user/wff/grasp-net/dataset/raw_data',
        processed_root='/home/user/wff/grasp-net/dataset/processed_data',
        augment_angles=custom_angles  # 传入自定义角度列表
    )