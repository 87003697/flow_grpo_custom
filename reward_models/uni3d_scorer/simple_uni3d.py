"""
最简单的Uni3D评分器 - 无缓存版本
"""

import torch
import numpy as np
from pathlib import Path
from kiui.mesh import Mesh
import open_clip
from .models.uni3d import create_uni3d
from typing import List


class SimpleUni3DScorer:
    """最简单的Uni3D评分器 - 每次调用都重新初始化"""
    
    def __init__(self, device="cuda", verbose: bool = False):
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        if self.verbose:
            print(f"🔧 初始化SimpleUni3DScorer: {self.device}")
        
        # 初始化CLIP模型
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'EVA02-E-14-plus', 
            pretrained=None
        )
        
        # 加载CLIP权重
        repo_root = Path(__file__).resolve().parents[2]
        weights_dir = repo_root / "pretrained_weights"
        clip_weights_path = weights_dir / "eva02_e_14_plus_laion2b_s9b_b144k.pt"
        state_dict = torch.load(clip_weights_path, map_location='cpu', weights_only=False)
        self.clip_model.load_state_dict(state_dict, strict=True)
        self.clip_model.to(self.device, dtype=torch.bfloat16).eval()
        
        # 初始化Uni3D模型
        eva_weights_path = weights_dir / "eva_giant_patch14_560.pt"
        uni3d_weights_path = weights_dir / "uni3d-g.pt"
        
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
        self.uni3d_model.to(self.device, dtype=torch.bfloat16).eval()
        
        if self.verbose:
            print(f"✅ SimpleUni3DScorer初始化完成")
    
    def mesh_to_pointcloud_simple(self, mesh, num_points=10000):
        """最简单的mesh转点云 - 使用随机采样
        
        兼容两种 mesh 格式：
        - KiuiMesh: 使用 .v 和 .f 属性
        - MeshWithVoxel: 使用 .vertices 和 .faces 属性
        """
        # 获取顶点和面（兼容 .v/.f 和 .vertices/.faces）
        if hasattr(mesh, 'v') and mesh.v is not None:
            v_attr = mesh.v
            f_attr = mesh.f
        elif hasattr(mesh, 'vertices') and mesh.vertices is not None:
            v_attr = mesh.vertices
            f_attr = mesh.faces
        else:
            raise AttributeError(f"mesh 对象缺少 v/f 或 vertices/faces 属性: {type(mesh)}")
        
        # 统一设备
        mesh_device = v_attr.device if torch.is_tensor(v_attr) else self.device
        
        vertices = mesh.v if torch.is_tensor(mesh.v) else torch.from_numpy(mesh.v).float()
        faces = mesh.f if torch.is_tensor(mesh.f) else torch.from_numpy(mesh.f).long()
        vertices = vertices.to(mesh_device)
        faces = faces.to(mesh_device)
        
        # 处理颜色（兼容 .vc 和 .vertex_colors）
        vc_attr = getattr(mesh, 'vc', None) or getattr(mesh, 'vertex_colors', None)
        if vc_attr is not None:
            vertex_colors = vc_attr if torch.is_tensor(vc_attr) else torch.from_numpy(vc_attr).float()
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
    def compute_scores(self, meshes, images, metadata=None) -> List[float]:
        """计算 Uni3D 评分（批并行版：统一点数 N，按 chunk 分块前向）。

        Args:
            meshes: mesh 列表
            images: 图像列表
            metadata: 元数据列表（可选，未使用）

        Returns:
            scores_list: 长度为 M 的分数列表
        """
        if isinstance(meshes, Mesh):
            meshes = [meshes]  # 形状: 长度 M 的列表

        # 常量：统一点数与分块大小
        N_POINTS = 10000  # 形状: 标量
        CHUNK = 8  # 形状: 标量（按显存可调）

        # 批大小
        M = len(meshes)  # 形状: 标量
        B = len(images)  # 形状: 标量

        # 图像特征（一次性批处理）
        image_tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device, dtype=torch.bfloat16)  # 形状: (B,3,H,W)
        image_features = self.clip_model.encode_image(image_tensors)  # 形状: (B,D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 形状: (B,D)
        image_features = image_features.to(torch.bfloat16)  # 形状: (B,D)

        # 将所有 mesh 采样为固定 N 点并堆叠
        pcs = []  # 形状: 长度 M 的列表
        for mesh in meshes:
            pc = self.mesh_to_pointcloud_simple(mesh, num_points=N_POINTS)  # 形状: (N,6)
            pcs.append(pc.to(torch.float32))  # 形状: (N,6)
        pc_batch = torch.stack(pcs, dim=0).to(self.device, dtype=torch.bfloat16, non_blocking=True)  # 形状: (M,N,6)

        # 按分块进行 Uni3D 前向并与对应图像计算相似度
        scores_vec = torch.empty(M, device=self.device, dtype=torch.float32)  # 形状: (M,)
        for start in range(0, M, CHUNK):
            end = min(start + CHUNK, M)  # 形状: 标量
            pc_chunk = pc_batch[start:end]  # 形状: (m,N,6)

            pc_features = self.uni3d_model.encode_pc(pc_chunk)  # 形状: (m,D)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)  # 形状: (m,D)
            pc_features = pc_features.to(torch.bfloat16)  # 形状: (m,D)

            idx = torch.arange(start, end, device=self.device)  # 形状: (m,)
            img_idx = idx % max(1, B)  # 形状: (m,)
            cur_img = image_features[img_idx].to(torch.bfloat16)  # 形状: (m,D)

            sim = torch.sum(cur_img * pc_features, dim=-1)  # 形状: (m,)
            scores_vec[start:end] = sim  # 形状: (m,)

        scores_list = scores_vec.detach().cpu().tolist()  # 形状: 长度 M 的列表
        return scores_list