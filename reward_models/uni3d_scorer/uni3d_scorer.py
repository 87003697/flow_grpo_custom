"""
Uni3D Scorer - 🚀 超高效的3D mesh语义质量评分器
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Union, Tuple
from pathlib import Path

# 导入正确的模型
import open_clip
from .models.uni3d import create_uni3d, Uni3D
from .models.mesh_utils import Mesh

def _resolve_weights_path(*filename_candidates: str) -> Path:
    """在项目根目录 `pretrained_weights/` 中按顺序查找候选文件。
    若均不存在，则返回第一个候选在该目录下的默认路径。
    """
    project_root = Path(__file__).resolve().parents[2]
    root_dir = project_root / "pretrained_weights"
    for name in filename_candidates:
        cand = root_dir / name
        if cand.exists():
            return cand
    return root_dir / filename_candidates[0]

def _fps_pytorch(xyz, npoint):
    """Furthest Point Sampling using PyTorch - Optimized Version"""
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    
    return centroids

def _gather_pytorch(points, idx):
    """
    Gather operation using PyTorch - Official Uni3D Logic
    """
    B, N, C = points.shape
    idx = idx.unsqueeze(-1).expand(-1, -1, C)
    new_points = torch.gather(points, 1, idx)
    return new_points

class Uni3DScorer:
    """🚀 超高效的Uni3D评分器"""
    
    def __init__(self, device="cuda", target_device="cuda"):
        # 🔧 设备配置
        self.target_device = torch.device(target_device if torch.cuda.is_available() else "cpu")
        
        # 🔧 模型缓存状态
        self._models_initialized = False
        
        self._init_models()
        
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def _init_models(self):
        """一次性初始化所有模型，避免重复加载"""
        if self._models_initialized:
            return
            
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 1. 初始化CLIP模型
        clip_weights_path = _resolve_weights_path("eva02_e_14_plus_laion2b_s9b_b144k.pt")
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'EVA02-E-14-plus', 
            pretrained=None
        )
        
        state_dict = torch.load(clip_weights_path, map_location='cpu', weights_only=False)
        self.clip_model.load_state_dict(state_dict, strict=True)
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. 初始化Uni3D模型
        eva_weights_path = _resolve_weights_path("eva_giant_patch14_560.pt")
        uni3d_weights_path = _resolve_weights_path("uni3d-g.pt")
        
        class Args:
            pc_model = "eva_giant_patch14_560"
            pretrained_pc = str(eva_weights_path)
            drop_path_rate = 0.0
            pc_feat_dim = 1408
            embed_dim = 1024
            group_size = 64
            num_group = 512
            pc_encoder_dim = 512
            patch_dropout = 0.0
        
        args = Args()
        self.uni3d_model = create_uni3d(args)
        
        checkpoint = torch.load(uni3d_weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['module']
        self.uni3d_model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        self.clip_model.eval()
        self.uni3d_model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 直接将模型移动到目标设备
        self.device = self.target_device
        self.clip_model = self.clip_model.to(self.target_device, dtype=torch.bfloat16)
        self.uni3d_model = self.uni3d_model.to(self.target_device, dtype=torch.bfloat16)  # 形状: 模型
        
        self._models_initialized = True


    @torch.no_grad()
    def __call__(self, 
                 meshes: Union[Mesh, List[Mesh]], 
                 images: Union[str, List[str]],
                 metadata: dict = None,
                 openshape_setting: bool = False) -> Tuple[List[float], dict]:
        """使用官方Uni3D流程的图像-3D评分器"""
        
        # 确保模型已初始化
        self._init_models()
        
        start_time = time.time()
        
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(images, str):
            images = [images]
            
        # 确保数量匹配
        assert len(meshes) % len(images) == 0, "Mesh 数量 ({len(meshes)}) 与 image 数量 ({len(images)}) 不匹配"

        
        # 使用官方流程处理点云
        pc_tensor = prepare_pointcloud_batch(meshes, num_points=10000, 
                                           openshape_setting=openshape_setting)
        pc_tensor = pc_tensor.to(self.device, dtype=torch.bfloat16)  # 形状: (B,N,6)
 
        # 批量处理图像
        # images现在直接是PIL对象列表，不需要Image.open()
        image_tensors = torch.stack([
            self.clip_preprocess(img) for img in images
        ]).to(self.device, dtype=torch.bfloat16)
 
        # 批量推理
        with torch.no_grad():
            # 提取特征
            image_features = self.clip_model.encode_image(image_tensors)  # 形状: (B,D)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 形状: (B,D)
            image_features = image_features.to(torch.bfloat16)  # 形状: (B,D)

            # 重复特征
            image_features = image_features.repeat_interleave(len(meshes) // len(images), dim=0)
            
            pc_features = self.uni3d_model.encode_pc(pc_tensor)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = torch.cosine_similarity(image_features, pc_features, dim=-1)
            scores = similarity.cpu().tolist()
        
        elapsed = time.time() - start_time
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return scores, {
            "num_meshes": len(meshes), 
            "avg_score": avg_score,
            "eval_time": elapsed
        }

def prepare_pointcloud_batch(meshes: List[Mesh], num_points: int = 10000, 
                          openshape_setting: bool = False) -> torch.Tensor:
    """
    将多个 mesh 转换为批量点云数据，完全按照官方 Uni3D 流程
    
    Args:
        meshes: kiui mesh 对象列表
        num_points: 每个点云的采样点数（官方默认 10000）
        openshape_setting: 是否使用OpenShape设置（Y-Z轴翻转）
        
    Returns:
        torch.Tensor: 批量点云数据，形状 (batch_size, num_points, 6)
    """
    pointclouds = []
    
    for mesh in meshes:
        # 按照官方流程处理每个mesh
        pointcloud = _sample_points_from_mesh_official(mesh, num_points, openshape_setting)
        pointclouds.append(pointcloud)
    
    return torch.stack(pointclouds, dim=0)

def _sample_points_from_mesh_official(mesh: Mesh, num_points: int = 10000, 
                                   openshape_setting: bool = False) -> torch.Tensor:
    """
    从单个mesh采样点云，完全按照官方Uni3D流程
    """
    vertices = mesh.v if torch.is_tensor(mesh.v) else torch.from_numpy(mesh.v).float()
    faces = mesh.f if torch.is_tensor(mesh.f) else torch.from_numpy(mesh.f).long()
    
    initial_num_points = max(num_points * 3, 30000)
    
    # 处理颜色信息（官方方式）
    if hasattr(mesh, 'vc') and mesh.vc is not None:
        vertex_colors = mesh.vc if torch.is_tensor(mesh.vc) else torch.from_numpy(mesh.vc).float()
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
    else:
        vertex_colors = torch.ones_like(vertices) * 0.4
    
    # 面积加权采样
    face_vertices = vertices[faces]
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = 0.5 * torch.norm(cross_product, dim=1)
    face_probs = face_areas / face_areas.sum()
    
    selected_faces = torch.multinomial(face_probs, initial_num_points, replacement=True)
    selected_face_vertices = face_vertices[selected_faces]
    
    # 重心坐标采样
    u = torch.rand(initial_num_points, device=vertices.device)
    v = torch.rand(initial_num_points, device=vertices.device)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    w = 1.0 - u - v
    
    sampled_points = (
        w.unsqueeze(-1) * selected_face_vertices[:, 0] +
        u.unsqueeze(-1) * selected_face_vertices[:, 1] +
        v.unsqueeze(-1) * selected_face_vertices[:, 2]
    )
    
    selected_face_colors = vertex_colors[faces[selected_faces]]
    sampled_colors = (
        w.unsqueeze(-1) * selected_face_colors[:, 0] +
        u.unsqueeze(-1) * selected_face_colors[:, 1] +
        v.unsqueeze(-1) * selected_face_colors[:, 2]
    )
    
    # FPS下采样
    if sampled_points.shape[0] > num_points:
        xyz_for_fps = sampled_points.unsqueeze(0)
        fps_indices = _fps_pytorch(xyz_for_fps, num_points)
        sampled_points = _gather_pytorch(sampled_points.unsqueeze(0), fps_indices)[0]
        sampled_colors = _gather_pytorch(sampled_colors.unsqueeze(0), fps_indices)[0]
    
    # 官方坐标处理流程
    xyz = sampled_points
    rgb = sampled_colors
    
    if openshape_setting:
        xyz[:, [1, 2]] = xyz[:, [2, 1]]  # Y-Z轴交换
        # normalize_pc
        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
        xyz = xyz / m
    else:
        # pc_normalize  
        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
        xyz = xyz / m
    
    # 最终拼接
    pointcloud = torch.cat([xyz, rgb], dim=1)
    return pointcloud

def main():
    """测试 Uni3D 评分器"""
    scorer = Uni3DScorer()

if __name__ == "__main__":
    main() 