import os
from typing import Any, Dict, List

import torch
from PIL import Image
import torchvision.transforms as T

from .config import ScorerConfig
from .camera.vggt_estimator import VGGTSearchEstimator
from .encoders.dino_encoder import DinoNormalEncoder
from .camera.support import build_support_batches, load_fixed_poses_and_renderer
from .camera.estimate_utils import batch_estimate_camera
from .render.render_normals import render_normals_batched
from .vis.save import save_camera_search_visualization
from .vis.normal_convert import tensor_from_normal_pil as shared_tensor_from_normal_pil, normal_tensor_to_pil as shared_normal_tensor_to_pil


class CameraNormalScorer:
    """按 image_path 分组、基于 normal PIL 搜索相机并用 DINO 计算相似度的简化实现。

    简化要点:
        - 仅使用 metadata.normal_pil 提供的图像侧法线，禁止回退到磁盘缓存。
        - 先对所有分组逐组完成相机估计与法线渲染，再一次性并行编码 DINO：
          1) 所有组的图像侧法线（G 张） 2) 所有渲染法线（∑K 张）。
        - 用向量化余弦相似度得到每个 mesh 的分数。
    """

    def __init__(self, device: torch.device, cfg: Dict[str, Any]) -> None:
        self.device = device
        self.cfg = ScorerConfig(**cfg)  # 形状: 配置对象

        # DINO 编码器
        if self.cfg.encoder == "dino_v2":
            model_id = self.cfg.dino_v2_path  # 形状: 标量
        else:
            model_id = self.cfg.dino_v3_path  # 形状: 标量
        self.encoder = DinoNormalEncoder(model_id=model_id, device=device)  # 形状: 编码器

        # 相机估计器（必须提供 camera_ckpt）
        self.camera = VGGTSearchEstimator(
            device,
            camera_param_dim=int(self.cfg.camera_param_dim),
            img_size=int(self.cfg.img_size),
            ckpt=getattr(self.cfg, "camera_ckpt", ""),
        )  # 形状: 相机估计器
        # 单例渲染器（与 support 渲染一致的 img_size/device）
        _, self._renderer = load_fixed_poses_and_renderer(self.cfg.camera_config_py, int(self.cfg.img_size), self.device)

    # -------------------- 基础工具 --------------------
    def _get_image_path(self, meta: Dict[str, Any]) -> str:
        if isinstance(meta, dict) and "image_path" in meta:
            return str(meta["image_path"])  # 形状: 标量
        if isinstance(meta, dict) and "image_name" in meta:
            base_dir = os.environ.get("FLOW_GRPO_DATA_DIR", "dataset/eval3d_hunyuan3d")  # 形状: 标量
            return os.path.join(base_dir, "images", str(meta["image_name"]))  # 形状: 标量
        raise ValueError("metadata 缺少 image_path 或 image_name")

    def _tensor_from_normal_pil(self, normal_pil: Image.Image, R: int) -> torch.Tensor:
        """将 normal PIL 变换为 (3,R,R) 且值域在 [-1,1]。

        备注: 不做回退与额外读取，严格依赖传入的 PIL。
        """
        return shared_tensor_from_normal_pil(normal_pil, int(R), self.device)  # 形状: (3,R,R)

    def _normal_tensor_to_pil(self, n: torch.Tensor) -> Image.Image:
        """将法线张量 [-1,1] 的 (3,R,R) 转为 RGB PIL。

        输入:
            n: (3,R,R) in [-1,1]
        输出:
            PIL(R,R,3)
        """
        return shared_normal_tensor_to_pil(n)  # 形状: PIL(R,R,3)

    def _build_query_from_metadata(self, meta: Dict[str, Any]) -> torch.Tensor:
        """从 metadata.normal_pil 构造 VGGT 的 query 输入 (1,3,H,W)。"""
        if ("normal_pil" not in meta) or (meta["normal_pil"] is None):
            raise ValueError("metadata 必须包含 normal_pil（不支持回退）")
        H, W = int(self.cfg.img_size), int(self.cfg.img_size)  # 形状: 标量, 标量
        transform = T.Compose([
            T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),  # 形状: -> PIL(H,W)
            T.ToTensor(),  # 形状: -> (3,H,W) in [0,1]
        ])
        q = transform(meta["normal_pil"]).to(self.device)  # 形状: (3,H,W)
        return q.unsqueeze(0)  # 形状: (1,3,H,W)

    def _encode_normals_in_chunks(self, normals: torch.Tensor, bs: int) -> torch.Tensor:
        """分块编码法线图像，避免一次性占用过多显存。

        输入:
            normals: (B,3,R,R)
            bs: 分块大小
        输出:
            (B,D)
        """
        B = normals.shape[0]  # 形状: 标量
        feats: List[torch.Tensor] = []  # 形状: 列表
        for s in range(0, B, int(bs)):
            e = min(B, s + int(bs))  # 形状: 标量
            f = self.encoder.features_from_normals(normals[s:e])  # 形状: (b,D)
            feats.append(f)  # 形状: 追加
        return torch.cat(feats, dim=0)  # 形状: (B,D)

    # -------------------- 主流程 --------------------
    @torch.no_grad()
    def compute_scores(
        self,
        meshes: List[Any],
        images: List[Image.Image],
        metadata: List[Dict[str, Any]],
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """为 (mesh, image, meta) 列表计算法线相似度奖励。

        流程:
            1) 按 image_path 分组。
            2) 每组用 normal_pil 作为 query，批量估计相机并渲染得到该组 K 张渲染法线。
            3) 收集所有组的图像侧法线与渲染法线，合并为一次 DINO 前向编码后再拆分。
            4) 向量化计算每个 mesh 与其所属组图像法线的余弦相似度并映射到 [0,1]。
        """
        assert len(meshes) == len(images) == len(metadata), "输入列表长度需一致"  # 形状: 断言

        R = int(self.cfg.normal_resolution)  # 形状: 标量
        H, W = int(self.cfg.img_size), int(self.cfg.img_size)  # 形状: 标量, 标量

        # 1) 分组
        groups: Dict[str, List[int]] = {}
        for idx, meta in enumerate(metadata):
            p = self._get_image_path(meta)  # 形状: 标量
            groups.setdefault(p, []).append(idx)  # 形状: 追加

        image_paths = list(groups.keys())  # 形状: 长度 G

        # 收集：组级图像法线、渲染法线，以及映射关系
        group_normals: List[torch.Tensor] = []  # 每组 (3,R,R)
        rendered_normals_all: List[torch.Tensor] = []  # 多组拼接后 (sumK,3,R,R)
        mesh_global_indices: List[int] = []  # 长度 sumK
        mesh_group_indices: List[int] = []  # 长度 sumK，对应每个渲染法线所属组 id

        # 2) 逐组估计相机 + 渲染
        # 分组式元数据：每张图像仅保存一次 image_path 与 image_normal_pil
        grouped_meta: List[Dict[str, Any]] = []  # 形状: 长度 G
        pair_j_to_group_local: List[tuple[int, int]] = []  # 形状: 长度 M，映射合并序 j -> (gid, local_idx)
        for gid, image_path in enumerate(image_paths):
            idxs = groups[image_path]  # 形状: 长度 K
            meta0 = metadata[idxs[0]]  # 形状: 字典

            # 图像侧法线（来自 normal_pil）
            assert ("normal_pil" in meta0) and (meta0["normal_pil"] is not None), "metadata.normal_pil 是必需的"
            n_img = self._tensor_from_normal_pil(meta0["normal_pil"], R)  # 形状: (3,R,R)
            group_normals.append(n_img)  # 形状: 追加
            # 分组元数据记录（只存一次图像侧信息）
            img_pil = self._normal_tensor_to_pil(n_img)  # 形状: PIL(R,R,3)
            grouped_meta.append({
                "group_id": int(gid),                  # 形状: 标量
                "image_path": str(image_path),         # 形状: 字符串
                "image_normal_pil": img_pil,           # 形状: PIL(R,R,3)
                "candidates": [],                      # 形状: 长度 K 的列表（稍后填充）
            })  # 形状: 追加

            # query 构造并构建支持批次
            imgs_query = self._build_query_from_metadata(meta0)  # 形状: (1,3,H,W)
            images_batched, support = build_support_batches(
                meshes=meshes,
                idxs=idxs,
                imgs_query=imgs_query,
                H=H,
                W=W,
                camera_config_py=self.cfg.camera_config_py,
                camera_param_dim=int(self.cfg.camera_param_dim),
                img_size=int(self.cfg.img_size),
                device=self.device,
            )  # 形状: (K,S,3,H,W), (K,S-1,D)

            # 相机估计
            extri_all, intr_all, intr_pix_all = batch_estimate_camera(
                self.camera, images_batched, support, H, W, R, int(self.cfg.cam_batch_size)
            )  # 形状: (K,4,4),(K,3,3),(K,3,3)

            # 渲染法线（参考渲染器使用像素内参 intr_pix_all 和 C2W）
            n_mesh_all = render_normals_batched(
                meshes, idxs, extri_all, intr_pix_all, H, R, self.device, renderer=self._renderer
            )  # 形状: (K,3,R,R)

            rendered_normals_all.append(n_mesh_all)  # 形状: 追加
            mesh_global_indices.extend(idxs)  # 形状: 追加 K 个
            mesh_group_indices.extend([gid] * n_mesh_all.shape[0])  # 形状: 追加 K 个
            # 记录每个候选（仅保存渲染法线；图像侧法线保存在组级）
            for j in range(n_mesh_all.shape[0]):
                mesh_norm_pil = self._normal_tensor_to_pil(n_mesh_all[j])  # 形状: PIL(R,R,3)
                local_idx = len(grouped_meta[gid]["candidates"])  # 形状: 标量
                grouped_meta[gid]["candidates"].append({
                    "mesh_index": int(idxs[j]),            # 形状: 标量
                    "rendered_normal_pil": mesh_norm_pil, # 形状: PIL(R,R,3)
                    "score": None,                         # 形状: 占位
                })  # 形状: 追加
                pair_j_to_group_local.append((gid, local_idx))  # 形状: 追加

            # 可视化保存（每组示例保存第一个样本）
            if bool(self.cfg.save_vis):
                os.makedirs(self.cfg.vis_dir, exist_ok=True)
                tag = os.path.splitext(os.path.basename(image_path))[0]
                save_camera_search_visualization(
                    images_batched,                # 形状: (K,S,3,H,W)
                    n_img,                          # 形状: (3,R,R)
                    n_mesh_all[0],                  # 形状: (3,R,R)
                    self.cfg.vis_dir,
                    tag,
                )

        # 3) DINO 并行编码：组图像法线 + 所有渲染法线（合并一次前向）
        G = len(group_normals)  # 形状: 标量
        if G == 0:
            return [], []  # 形状: 空列表

        n_groups = torch.stack(group_normals, dim=0)  # 形状: (G,3,R,R)
        n_mesh_cat = torch.cat(rendered_normals_all, dim=0)  # 形状: (M,3,R,R)
        normals_all = torch.cat([n_groups, n_mesh_cat], dim=0)  # 形状: (G+M,3,R,R)

        f_all = self._encode_normals_in_chunks(normals_all, int(self.cfg.dino_batch_size))  # 形状: (G+M,D)
        f_groups = f_all[:G]  # 形状: (G,D)
        f_mesh_all = f_all[G:]  # 形状: (M,D)

        # 4) 计算相似度并回填到对应 mesh 位置
        group_idx_tensor = torch.tensor(mesh_group_indices, device=f_groups.device, dtype=torch.long)  # 形状: (M,)
        f_img_per_mesh = f_groups.index_select(0, group_idx_tensor)  # 形状: (M,D)
        cos_sim = (f_mesh_all * f_img_per_mesh).sum(dim=-1)  # 形状: (M,)
        rewards_vec = (cos_sim + 1.0) * 0.5  # 形状: (M,)

        rewards_all: List[float] = [0.0 for _ in range(len(meshes))]  # 形状: 长度 N_total
        for j, midx in enumerate(mesh_global_indices):
            score_j = float(rewards_vec[j].item())  # 形状: 标量
            rewards_all[midx] = score_j  # 形状: 标量
            # 回填分数到对应组的候选项
            if j < len(pair_j_to_group_local):
                gid, lidx = pair_j_to_group_local[j]  # 形状: 标量, 标量
                grouped_meta[gid]["candidates"][lidx]["score"] = score_j  # 形状: 标量

        return rewards_all, grouped_meta

