"""
Mesh processing utilities for Uni3D scorer
"""
import torch
import numpy as np
from typing import Union, List
from kiui.mesh import Mesh


def mesh_to_pointcloud(mesh: Mesh, num_points: int = 8192, with_color: bool = True) -> torch.Tensor:
    """
    将 kiui mesh 转换为点云格式，适用于 Uni3D 模型
    
    Args:
        mesh: kiui Mesh 对象
        num_points: 采样点数，默认 8192
        with_color: 是否包含颜色信息
        
    Returns:
        torch.Tensor: 点云数据，形状为 (num_points, 6) 或 (num_points, 3)
                     前3列为 xyz 坐标，后3列为 rgb 颜色（如果 with_color=True）
    """
    # 获取顶点和面信息
    vertices = mesh.v  # (N, 3)
    faces = mesh.f     # (F, 3)
    
    # 如果网格有颜色信息，使用它；否则使用默认颜色
    if hasattr(mesh, 'vc') and mesh.vc is not None:
        vertex_colors = mesh.vc  # (N, 3)
    else:
        # 默认使用白色
        vertex_colors = torch.ones_like(vertices) * 0.7
    
    # 从面片表面采样点云
    # 计算每个面片的面积
    face_vertices = vertices[faces]  # (F, 3, 3)
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    
    # 计算面积
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = 0.5 * torch.norm(cross_product, dim=1)
    
    # 根据面积加权采样
    face_probs = face_areas / face_areas.sum()
    
    # 采样面片
    sampled_face_indices = torch.multinomial(face_probs, num_points, replacement=True)
    
    # 在每个采样的面片上随机采样点
    sampled_faces = face_vertices[sampled_face_indices]  # (num_points, 3, 3)
    
    # 重心坐标采样
    r1 = torch.rand(num_points, 1, device=vertices.device)
    r2 = torch.rand(num_points, 1, device=vertices.device)
    
    # 确保 r1 + r2 <= 1
    mask = (r1 + r2) > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    
    r3 = 1 - r1 - r2
    
    # 计算采样点
    sampled_points = (
        r1 * sampled_faces[:, 0] + 
        r2 * sampled_faces[:, 1] + 
        r3 * sampled_faces[:, 2]
    )  # (num_points, 3)
    
    if with_color:
        # 为采样点计算颜色（使用面片顶点颜色的插值）
        sampled_face_colors = vertex_colors[faces[sampled_face_indices]]  # (num_points, 3, 3)
        sampled_colors = (
            r1 * sampled_face_colors[:, 0] + 
            r2 * sampled_face_colors[:, 1] + 
            r3 * sampled_face_colors[:, 2]
        )  # (num_points, 3)
        
        # 合并坐标和颜色
        pointcloud = torch.cat([sampled_points, sampled_colors], dim=1)  # (num_points, 6)
    else:
        pointcloud = sampled_points  # (num_points, 3)
    
    return pointcloud


def normalize_pointcloud(pointcloud: torch.Tensor) -> torch.Tensor:
    """
    标准化点云数据
    
    Args:
        pointcloud: 点云数据，形状为 (num_points, 6) 或 (num_points, 3)
        
    Returns:
        torch.Tensor: 标准化后的点云数据
    """
    # 分离坐标和颜色
    xyz = pointcloud[:, :3]  # (num_points, 3)
    
    # 标准化坐标到 [-1, 1] 范围
    xyz_min = xyz.min(dim=0, keepdim=True)[0]
    xyz_max = xyz.max(dim=0, keepdim=True)[0]
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_scale = (xyz_max - xyz_min).max()
    
    normalized_xyz = (xyz - xyz_center) / (xyz_scale / 2)
    
    if pointcloud.shape[1] > 3:
        # 保持颜色信息不变，假设已经在 [0, 1] 范围内
        colors = pointcloud[:, 3:]
        normalized_pointcloud = torch.cat([normalized_xyz, colors], dim=1)
    else:
        normalized_pointcloud = normalized_xyz
    
    return normalized_pointcloud


def prepare_pointcloud_batch(meshes: List[Mesh], num_points: int = 8192) -> torch.Tensor:
    """
    准备点云批次数据用于 Uni3D 模型
    
    Args:
        meshes: kiui Mesh 对象列表
        num_points: 每个点云的采样点数
        
    Returns:
        torch.Tensor: 批次点云数据，形状为 (batch_size, num_points, 6)
    """
    pointclouds = []
    
    for mesh in meshes:
        pc = mesh_to_pointcloud(mesh, num_points=num_points, with_color=True)
        pc = normalize_pointcloud(pc)
        pointclouds.append(pc)
    
    # 堆叠成批次
    batch_pointclouds = torch.stack(pointclouds, dim=0)
    
    return batch_pointclouds 