"""
最简单的Uni3D评分器 - 无缓存版本
"""

import torch
import numpy as np
import time
from pathlib import Path
from kiui.mesh import Mesh
from PIL import Image
import open_clip
from .models.uni3d import create_uni3d
from torch.nn import functional as F
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from reward_models.camera_normal_scorer.render.adapter import to_mesh_extract, KiuiMeshLike

# 为每组构造 image_normal_pil（使用输入 RGB 居中裁剪）
def crop_rgb_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size  # 形状: 标量, 标量
    side = min(w, h)  # 形状: 标量
    left = (w - side) // 2  # 形状: 标量
    top = (h - side) // 2  # 形状: 标量
    return img.crop((left, top, left + side, top + side)).resize((256, 256), Image.BICUBIC)


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
        self.clip_model.to(self.device).eval()
        
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
        self.uni3d_model.to(self.device).eval()
        
        if self.verbose:
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
    def compute_scores(self, meshes, images, metadata=None):
        """计算评分（批并行版：统一点数 N，按 chunk 分块前向），并返回分组元数据。

        返回: (scores_list, grouped_meta)
        grouped_meta: List[{
            "image_path": str,
            "image_normal_pil": PIL(R,R,3)  # 这里使用输入 RGB 的居中裁剪方图
            "candidates": List[{"mesh_index": int, "score": float, "rendered_normal_pil": PIL(R,R,3)}]
        }]
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
        image_tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)  # 形状: (B,3,H,W)
        image_features = self.clip_model.encode_image(image_tensors)  # 形状: (B,D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 形状: (B,D)

        # 将所有 mesh 采样为固定 N 点并堆叠
        pcs = []  # 形状: 长度 M 的列表
        for mesh in meshes:
            pc = self.mesh_to_pointcloud_simple(mesh, num_points=N_POINTS)  # 形状: (N,6)
            pcs.append(pc.to(torch.float32))  # 形状: (N,6)
        pc_batch = torch.stack(pcs, dim=0).to(self.device, non_blocking=True)  # 形状: (M,N,6)

        # 按分块进行 Uni3D 前向并与对应图像计算相似度
        scores_vec = torch.empty(M, device=self.device, dtype=torch.float32)  # 形状: (M,)
        for start in range(0, M, CHUNK):
            end = min(start + CHUNK, M)  # 形状: 标量
            pc_chunk = pc_batch[start:end]  # 形状: (m,N,6)

            pc_features = self.uni3d_model.encode_pc(pc_chunk)  # 形状: (m,D)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)  # 形状: (m,D)

            idx = torch.arange(start, end, device=self.device)  # 形状: (m,)
            img_idx = idx % max(1, B)  # 形状: (m,)
            cur_img = image_features[img_idx]  # 形状: (m,D)

            sim = torch.sum(cur_img * pc_features, dim=-1)  # 形状: (m,)
            scores_vec[start:end] = sim  # 形状: (m,)

        scores_list = scores_vec.detach().cpu().tolist()  # 形状: 长度 M 的列表

        # ===== 构造分组元数据（与 camera_normal_scorer 风格一致） =====
        grouped_meta = []
        if metadata is None:
            metadata = [{} for _ in range(len(images))]

        # 分组键：优先使用 image_path，其次 image_name
        # 收集每组的索引
        groups = {}
        for idx, meta in enumerate(metadata):
            key = str(meta.get("image_path", meta.get("image_name", f"idx_{idx}")))
            groups.setdefault(key, []).append(idx)

        for image_path, idxs in groups.items():
            if len(idxs) == 0:
                continue
            # 取该组第一个索引对应的 RGB 作为显示用“图像侧法线”占位
            img_left = crop_rgb_square(images[idxs[0]], 256)  # 形状: PIL(256,256,3)
            group_entry = {
                "image_path": image_path,
                "image_normal_pil": img_left,
                "candidates": [],
            }
            for i in idxs:
                cand = {
                    "mesh_index": int(i),
                    "score": float(scores_list[i]),
                    # 提前渲染每个候选的法线，便于上层直接筛选 best/worst 可视化
                    "rendered_normal_pil": self._render_normal_pil(meshes[i], R=256),
                }
                if cand["rendered_normal_pil"] is None:
                    # 若渲染失败，保持键存在但设为 None，避免 logger 报错
                    cand["rendered_normal_pil"] = img_left
                group_entry["candidates"].append(cand)
            grouped_meta.append(group_entry)

        return scores_list, grouped_meta

    # ============== 可视化：迁移 best/worst mesh 配对（与 camera_normal_scorer 风格一致） ==============
    def _render_normal_pil(self, mesh, R: int = 256):
        """渲染单视角法线为 PIL(R,R,3)。失败返回 None。"""

        device_str = ("cuda" if (self.device.type == "cuda") else "cpu")  # 形状: 字符串
        renderer = RefMeshRenderer(img_size=int(R), device=device_str)  # 形状: 渲染器(R,R)
        cams = renderer.sample_camera_poses(
            num_random_views=0,
            predefined_poses=[{"distance": 3.0, "fovy": 50.0, "elevation": 15.0, "azimuth": 0.0}],
        )  # 形状: 列表(1)
        mesh_ex = to_mesh_extract(mesh, self.device)  # 形状: 顶点:(V,3), 面:(F,3)
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: kiui Mesh
        out = renderer.render_mesh(
            mesh_kiui,
            cams,
            return_depth=False,
            return_normals=True,
            return_positions=False,
            return_masks=True,
        )  # 形状: images(1,3,R,R)
        img_t = out["images"][0]  # 形状: (3,R,R)
        img01 = img_t.clamp(0, 1)  # 形状: (3,R,R)
        img255 = (img01 * 255.0).round().to(torch.uint8)  # 形状: (3,R,R)
        img_hwc = img255.permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
        return Image.fromarray(img_hwc)


    @torch.no_grad()
    def build_best_worst_pairs(self, meshes, images, metadata, scores_arr, R: int = 256):
        """直接使用 compute_scores 返回的 grouped_meta 生成 best/worst 配对。"""
        pairs_best, pairs_worst = [], []
        for grp in metadata:  # metadata 即 grouped_meta
            cands = grp.get("candidates", [])
            if not cands:
                continue
            best_c = max(cands, key=lambda x: float(x.get("score", 0.0)))
            worst_c = min(cands, key=lambda x: float(x.get("score", 0.0)))
            pairs_best.append({
                "image_path": grp.get("image_path", ""),
                "image_normal_pil": grp.get("image_normal_pil", None),
                "rendered_normal_pil": best_c.get("rendered_normal_pil", None),
                "mesh_index": int(best_c.get("mesh_index", -1)),
                "score": float(best_c.get("score", 0.0)),
            })
            pairs_worst.append({
                "image_path": grp.get("image_path", ""),
                "image_normal_pil": grp.get("image_normal_pil", None),
                "rendered_normal_pil": worst_c.get("rendered_normal_pil", None),
                "mesh_index": int(worst_c.get("mesh_index", -1)),
                "score": float(worst_c.get("score", 0.0)),
            })
        return pairs_best, pairs_worst