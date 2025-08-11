"""
最简单的Uni3D评分器 - 无缓存版本
"""

import torch
import numpy as np
import time
from pathlib import Path
from kiui.mesh import Mesh
import open_clip
from .models.uni3d import create_uni3d


class SimpleUni3DScorer:
    """最简单的Uni3D评分器 - 每次调用都重新初始化"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        print(f"🔧 初始化SimpleUni3DScorer: {self.device}")
        
        # 初始化CLIP模型
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'EVA02-E-14-plus', 
            pretrained=None
        )
        
        # 加载CLIP权重
        clip_weights_path = Path("pretrained_weights/eva02_e_14_plus_laion2b_s9b_b144k.pt")
        state_dict = torch.load(clip_weights_path, map_location='cpu', weights_only=False)
        self.clip_model.load_state_dict(state_dict, strict=True)
        self.clip_model.to(self.device).eval()
        
        # 初始化Uni3D模型
        eva_weights_path = Path("pretrained_weights/eva_giant_patch14_560.pt")
        uni3d_weights_path = Path("pretrained_weights/uni3d-g.pt")
        
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
        
        # 加载Uni3D权重
        checkpoint = torch.load(uni3d_weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['module']
        self.uni3d_model.load_state_dict(state_dict, strict=True)
        self.uni3d_model.to(self.device).eval()
        
        print(f"✅ SimpleUni3DScorer初始化完成")
    
    def mesh_to_pointcloud_simple(self, mesh, num_points=10000):
        """最简单的mesh转点云 - 使用随机采样"""
        # 统一设备：以 mesh.v 的 device 为准
        mesh_device = mesh.v.device if torch.is_tensor(mesh.v) else self.device
        
        vertices = mesh.v if torch.is_tensor(mesh.v) else torch.from_numpy(mesh.v).float()
        faces = mesh.f if torch.is_tensor(mesh.f) else torch.from_numpy(mesh.f).long()
        vertices = vertices.to(mesh_device)
        faces = faces.to(mesh_device)
        
        # 处理颜色
        if hasattr(mesh, 'vc') and mesh.vc is not None:
            vertex_colors = mesh.vc if torch.is_tensor(mesh.vc) else torch.from_numpy(mesh.vc).float()
            if vertex_colors.max() > 1.0:
                vertex_colors = vertex_colors / 255.0
            vertex_colors = vertex_colors.to(mesh_device)
        else:
            vertex_colors = torch.ones_like(vertices, device=mesh_device) * 0.4
        
        # 随机采样面（放到同一设备）
        num_faces = faces.shape[0]
        selected_face_ids = torch.randint(0, num_faces, (num_points,), device=mesh_device)
        selected_faces = faces[selected_face_ids]
        
        # 重心坐标随机采样（放到同一设备）
        u = torch.rand(num_points, device=mesh_device)
        v = torch.rand(num_points, device=mesh_device)
        mask = u + v > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        w = 1.0 - u - v
        
        # 采样点
        face_vertices = vertices[selected_faces]
        points = (
            w.unsqueeze(-1) * face_vertices[:, 0] +
            u.unsqueeze(-1) * face_vertices[:, 1] +
            v.unsqueeze(-1) * face_vertices[:, 2]
        )
        
        # 采样颜色
        face_colors = vertex_colors[selected_faces]
        colors = (
            w.unsqueeze(-1) * face_colors[:, 0] +
            u.unsqueeze(-1) * face_colors[:, 1] +
            v.unsqueeze(-1) * face_colors[:, 2]
        )
        
        # 标准化坐标
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        max_dist = torch.max(torch.sqrt(torch.sum(points**2, dim=1)))
        if max_dist > 0:
            points = points / max_dist
        
        # 拼接xyz和rgb
        pointcloud = torch.cat([points, colors], dim=1)
        return pointcloud
    
    @torch.no_grad()
    def compute_scores(self, meshes, images):
        """计算评分"""
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        
        # 处理图像特征
        image_tensors = torch.stack([
            self.clip_preprocess(img) for img in images
        ]).to(self.device)
        
        image_features = self.clip_model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 处理每个mesh
        for i, mesh in enumerate(meshes):
            # 转换为点云
            pointcloud = self.mesh_to_pointcloud_simple(mesh, num_points=10000)
            pc_tensor = pointcloud.unsqueeze(0).to(self.device)
            
            # 提取3D特征
            pc_features = self.uni3d_model.encode_pc(pc_tensor)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 选择对应图像特征
            image_idx = i % len(images)
            current_image_features = image_features[image_idx:image_idx+1]
            
            # 计算相似度
            similarity = torch.cosine_similarity(current_image_features, pc_features, dim=-1)
            score = similarity.cpu().item()
            scores.append(score)
        
        return scores 