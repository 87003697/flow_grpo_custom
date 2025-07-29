"""
Mesh utilities for converting kiui meshes to point clouds
"""
import torch
import numpy as np
from typing import Tuple, Optional

# 导入kiui mesh
from kiui.mesh import Mesh

def sample_points_from_mesh(mesh: Mesh, num_points: int = 10000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 kiui mesh 对象采样点云
    
    Args:
        mesh: kiui mesh 对象
        num_points: 采样点数量
        
    Returns:
        Tuple[points, colors]: (点坐标 [N, 3], 颜色 [N, 3]) 或 (None, None) 如果失败
    """
    if mesh is None:
        return None, None
    
    try:
        # 从mesh中获取顶点和面
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            if mesh.v is None or mesh.f is None:
                return None, None
                
            vertices = mesh.v.cpu().numpy() if torch.is_tensor(mesh.v) else mesh.v
            faces = mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else mesh.f
        else:
            print("⚠️ Mesh对象没有顶点(v)或面(f)属性")
            return None, None
        
        if len(vertices) == 0 or len(faces) == 0:
            print("⚠️ 空的mesh数据")
            return None, None
        
        # 使用面积加权的方式从三角面上采样点
        points = _sample_points_from_faces(vertices, faces, num_points)
        
        # 检查是否有颜色信息
        colors = None
        if hasattr(mesh, 'vc') and mesh.vc is not None:
            # 顶点颜色存在，通过插值获取采样点的颜色
            vertex_colors = mesh.vc.cpu().numpy() if torch.is_tensor(mesh.vc) else mesh.vc
            colors = _interpolate_vertex_colors(points, vertices, faces, vertex_colors)
        else:
            # 没有颜色信息，设置为白色
            colors = np.ones((len(points), 3), dtype=np.float32)
        
        return points.astype(np.float32), colors.astype(np.float32)
        
    except Exception as e:
        print(f"⚠️ 点云采样失败: {e}")
        return None, None

def _sample_points_from_faces(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """
    使用面积加权从三角面上采样点
    """
    # 计算每个三角面的面积
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # 计算三角形面积
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # 处理面积为0的情况
    areas = np.clip(areas, 1e-10, None)
    
    # 根据面积权重选择面
    probabilities = areas / np.sum(areas)
    face_indices = np.random.choice(len(faces), size=num_points, p=probabilities)
    
    # 在选定的面上均匀采样点
    sampled_points = []
    for face_idx in face_indices:
        # 获取三角形的三个顶点
        v0 = vertices[faces[face_idx, 0]]
        v1 = vertices[faces[face_idx, 1]]
        v2 = vertices[faces[face_idx, 2]]
        
        # 使用重心坐标采样
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        
        # 重心坐标
        r3 = 1 - r1 - r2
        point = r1 * v0 + r2 * v1 + r3 * v2
        sampled_points.append(point)
    
    return np.array(sampled_points)

def _interpolate_vertex_colors(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray) -> np.ndarray:
    """
    通过重心坐标插值获取采样点的颜色
    """
    # 简化版本：对于每个采样点，找到最近的顶点并使用其颜色
    # 更精确的方法需要记录采样点来自哪个面以及重心坐标
    
    colors = []
    for point in points:
        # 找到最近的顶点
        distances = np.linalg.norm(vertices - point, axis=1)
        nearest_vertex_idx = np.argmin(distances)
        colors.append(vertex_colors[nearest_vertex_idx])
    
    return np.array(colors)

def test_mesh_sampling():
    """测试mesh采样功能"""
    import trimesh
    
    # 创建测试mesh
    mesh_trimesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = Mesh(v=mesh_trimesh.vertices, f=mesh_trimesh.faces)
    
    # 测试采样
    points, colors = sample_points_from_mesh(mesh, num_points=1000)
    
    if points is not None:
        print(f"✅ 采样成功: {len(points)} 个点")
        print(f"   点云形状: {points.shape}")
        print(f"   颜色形状: {colors.shape}")
        print(f"   点云范围: {points.min(axis=0)} ~ {points.max(axis=0)}")
    else:
        print("❌ 采样失败")

if __name__ == "__main__":
    test_mesh_sampling() 