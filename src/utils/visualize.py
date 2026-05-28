# ----------------------
# utils/visualize.py
# ----------------------
import open3d as o3d
import numpy as np
import json
import os

# 可视化参数配置
VISUAL_CONFIG = {
    "grasp_radius": 10,  # 抓取点周围高亮区域半径（毫米）
    "point_size": 2.0,  # 点云显示大小（像素）
    "colors": {
        "cloud": [0.6, 0.6, 0.6],  # 基础点云颜色（灰色）
        "highlight": [1, 193 / 255, 193 / 255],  # 高亮区域颜色（粉红）
        "grasp": [0, 1, 0],  # 抓取点颜色（绿色）
        "normal": [0, 245 / 255, 1],  # 法向量颜色（蓝色）
    },
    "normal_scale": 15,  # 法向量显示总长度（毫米）
    "coord_frame_size": 10,  # 坐标系尺寸（毫米）
    "grasp_sphere_radius": 1.5,  # 抓取点球体半径（毫米）
    "view_params": {  # 视角配置
        "zoom": 0.5,
        "front": [0.5, -0.3, 0.7],  # 相机前方方向向量
        "up": [0, 0, 1]  # 相机上方向量
    }
}


def create_grasp_vis(orig_points, grasps):
    """创建可视化元素（毫米单位）"""
    # 创建基础点云
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(orig_points)
    base_pcd.paint_uniform_color(VISUAL_CONFIG["colors"]["cloud"])

    # 初始化几何体集合
    geometries = [base_pcd]

    # 如果没有抓取位姿，直接返回基础点云
    if not grasps:
        return geometries

    # 查找高亮点区域
    highlight_idx = find_radius_points(
        orig_points,
        [g["position"] for g in grasps],
        VISUAL_CONFIG["grasp_radius"]
    )

    # 创建高亮点云
    highlight_pcd = base_pcd.select_by_index(highlight_idx)
    highlight_pcd.paint_uniform_color(VISUAL_CONFIG["colors"]["highlight"])
    geometries.append(highlight_pcd)

    # 添加抓取元素
    grasp_positions = []
    for grasp in grasps:
        pos = np.array(grasp["position"])
        normal = np.array(grasp["normal"])
        grasp_positions.append(pos)

        # 抓取点球体
        sphere = create_grasp_sphere(
            pos,
            VISUAL_CONFIG["colors"]["grasp"],
            VISUAL_CONFIG["grasp_sphere_radius"]
        )
        geometries.append(sphere)

        # 法向量箭头
        arrow = create_normal_arrow(
            pos,
            normal,
            VISUAL_CONFIG["colors"]["normal"],
            VISUAL_CONFIG["normal_scale"]
        )
        geometries.append(arrow)

        # 坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=VISUAL_CONFIG["coord_frame_size"],
            origin=pos
        )
        # coord_frame.paint_uniform_color(VISUAL_CONFIG["colors"]["coord_frame"])
        geometries.append(coord_frame)

    return geometries


def find_radius_points(pcd_points, grasp_points, radius):
    """查找抓取点周围半径内的点"""
    highlight_indices = set()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for grasp in grasp_points:
        center = np.array(grasp).reshape(3, 1)
        [_, idx, _] = kdtree.search_radius_vector_3d(center, radius)
        highlight_indices.update(idx)

    return list(highlight_indices)


def create_grasp_sphere(position, color, radius):
    """创建抓取点球体"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere


def create_normal_arrow(position, normal, color, scale):
    """创建法向量箭头（修正版：箭头尾部起始于球体中心）"""
    # 单位化法向量
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # 计算箭头总长度
    arrow_length = scale  # 直接使用配置的长度参数
    cylinder_height = arrow_length * 0.8  # 圆柱部分占80%
    cone_height = arrow_length * 0.2  # 锥体部分占20%

    # 创建箭头几何体（默认沿Z轴方向）
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_height * 0.03,  # 圆柱半径与长度比例
        cone_radius=cylinder_height * 0.05,  # 锥体底部半径
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )

    # 计算旋转矩阵：将默认Z轴方向对齐到法向量方向
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal)
    rotation_angle = np.arccos(np.dot(z_axis, normal))

    # 处理法向量与Z轴平行的情况
    if np.linalg.norm(rotation_axis) < 1e-6:
        if normal[2] > 0:
            rotation_axis = np.array([0, 1, 0])  # 向上方向
        else:
            rotation_axis = np.array([0, -1, 0])  # 向下方向

    # 计算旋转矩阵
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        rotation_axis * rotation_angle
    )

    # 应用旋转（以箭头底部为中心）
    arrow.rotate(rotation_matrix, center=(0, 0, 0))

    # 平移箭头到抓取点中心位置
    arrow.translate(position)

    arrow.paint_uniform_color(color)
    return arrow


def visualize(geometries):
    """执行可视化"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加所有几何元素
    for geom in geometries:
        vis.add_geometry(geom)

    # 设置渲染参数
    render_opt = vis.get_render_option()
    render_opt.point_size = VISUAL_CONFIG["point_size"]

    # 设置视角参数
    if len(geometries) > 0:
        view_ctl = vis.get_view_control()
        view_params = VISUAL_CONFIG["view_params"]
        view_ctl.set_front(view_params["front"])
        view_ctl.set_up(view_params["up"])
        view_ctl.set_zoom(view_params["zoom"])

        # 自动计算视角中心点
        all_points = np.concatenate([
            np.asarray(geom.points)
            for geom in geometries
            if isinstance(geom, o3d.geometry.PointCloud)
        ], axis=0)
        if len(all_points) > 0:
            view_ctl.set_lookat(np.mean(all_points, axis=0))

    vis.run()
    vis.destroy_window()
