import inspect
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

from .config import ScorerConfig
from .camera.vggt_estimator import VGGTSearchEstimator
from .camera.support import build_support_batches
from .camera.estimate_utils import batch_estimate_camera
from .render.render_normals import render_normals_batched, render_normals_predefined
from .vis.save import save_camera_search_visualization
from .utils.transforms import (
    to_tensor_from_normal_pil,
    normal_tensor_to_pil,
)


_GEMINI_ALLOWED_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-pro",
)


def _parse_gemini_model(enc_name: str) -> str:
    """Extract a valid Gemini model name from encoder string."""
    enc_lower = str(enc_name).lower()
    for model in _GEMINI_ALLOWED_MODELS:
        if model in enc_lower:
            return model
    allowed = ", ".join(_GEMINI_ALLOWED_MODELS)
    raise ValueError(
        f"Gemini 编码器名称必须包含以下任一模型子串 [{allowed}]，实际: {enc_name}"
    )


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
        self.camera_type = str(self.cfg.camera_type).lower()  # 形状: 字符串
        self._use_camera_search = (self.camera_type == "search")  # 形状: 布尔
        self._use_max_score = ("max" in self.camera_type)  # 形状: 布尔

        # 编码器选择：PickScore / DINOv2 / DINOv3
        enc_name_raw = str(self.cfg.encoder)
        enc_name = enc_name_raw.lower()  # 形状: 字符串
        if enc_name == "pickscore":
            from .encoders.pickscore_image_encoder import PickScoreImageEncoder
            self.encoder = PickScoreImageEncoder(  # 形状: 编码器
                device=device,
                model_id=self.cfg.pickscore_model_id,
                processor_id=self.cfg.pickscore_processor_id,
                dtype=torch.float32,
                similarity_type="cls",
            )
        elif enc_name == "clip":
            from .encoders.clip_image_encoder import ClipImageEncoder
            self.encoder = ClipImageEncoder(  # 形状: 编码器
                device=device,
                model_id=self.cfg.clip_model_id,
                processor_id=self.cfg.clip_processor_id,
                dtype=torch.float32,
                similarity_type="cls",
            )
        elif enc_name == "hpsv2":
            from .encoders.hpsv2_image_encoder import HPSV2ImageEncoder
            self.encoder = HPSV2ImageEncoder(  # 形状: 编码器
                device=device,
                ckpt_path=self.cfg.hpsv2_ckpt_path,
                dtype=torch.float32,
                similarity_type="cls",
            )
        elif enc_name == "dino_v2":
            from .encoders.dino_encoder import DinoNormalEncoder
            self.encoder = DinoNormalEncoder(  # 形状: 编码器
                model_id=self.cfg.dino_v2_path,  # 形状: 标量
                device=device,
                similarity_type=self.cfg.dino_similarity_type,
                dense_match_chunk_size=self.cfg.dense_match_chunk_size,
            )
        elif enc_name == "dino_v3":
            from .encoders.dino_encoder import DinoNormalEncoder
            self.encoder = DinoNormalEncoder(  # 形状: 编码器
                model_id=self.cfg.dino_v3_path,  # 形状: 标量
                device=device,
                similarity_type=self.cfg.dino_similarity_type,
                dense_match_chunk_size=self.cfg.dense_match_chunk_size,
            )
        elif "gemini" in enc_name:
            from .encoders.vlm_encoder_v2 import GeminiOpenAIEncoder, GeminiOpenAIGroupEncoder
            vlm_model = _parse_gemini_model(enc_name_raw)
            common_kwargs = {
                "device": device,
                "api_source": self.cfg.vlm_api_source,
                "model": vlm_model,
                "max_concurrent": self.cfg.vlm_max_concurrent,
                "timeout": self.cfg.vlm_timeout,
                "prompt_version": self.cfg.vlm_prompt_version,
                "max_tokens": int(self.cfg.vlm_max_tokens),
                "thinking_enabled": bool(self.cfg.vlm_enable_thinking),
            }  # 形状: 字典
            debug_raw_resp = bool(self.cfg.vlm_debug_raw_response)  # 形状: 布尔
            if "group" in enc_name:
                self.encoder = GeminiOpenAIGroupEncoder(  # 形状: 编码器
                    debug_raw_response=debug_raw_resp,
                    **common_kwargs,
                )
            else:
                self.encoder = GeminiOpenAIEncoder(  # 形状: 编码器
                        debug_raw_response=debug_raw_resp,
                    **common_kwargs,
                )
        else:
            raise ValueError(f"不支持的编码器类型: {self.cfg.encoder}")

        # 相机估计器（必须提供 camera_ckpt）
        if self._use_camera_search:
            self.camera = VGGTSearchEstimator(
                device,
                camera_param_dim=int(self.cfg.camera_param_dim),
                img_size=int(self.cfg.img_size),
                ckpt=self.cfg.camera_ckpt,
            )  # 形状: 相机估计器
        else:
            self.camera = None  # 形状: None

    def _get_fixed_predefined_poses(self) -> List[Dict[str, float]]:
        """返回与预览一致的固定视角。"""
        if "fixed_v0" in self.camera_type:
            return self._get_fixed_predefined_poses_v0()
        elif "fixed_v1" in self.camera_type:
            return self._get_fixed_predefined_poses_v1()
        else:
            raise ValueError(f"不支持的相机类型: {self.camera_type}")

    def _get_fixed_predefined_poses_v0(self) -> List[Dict[str, float]]:
        """返回与预览一致的固定视角（单视角）。"""
        elevation = 0.0  # 形状: 标量
        distance = 3.0  # 形状: 标量
        fovy = 50.0  # 形状: 标量
        azimuth = 0.0  # 形状: 标量
        return [{"distance": distance, "fovy": fovy, "elevation": elevation, "azimuth": azimuth}]  # 形状: 列表(1)
    
    def _get_fixed_predefined_poses_v1(self) -> List[Dict[str, float]]:
        """返回与预览一致的固定视角。"""
        elevation = 0.0  # 形状: 标量
        distance = 3.0  # 形状: 标量
        fovy = 50.0  # 形状: 标量
        azimuths = [0.0, 90.0, 180.0, 270.0]  # 形状: 列表(3)
        return [
            {"distance": distance, "fovy": fovy, "elevation": elevation, "azimuth": a} for a in azimuths
        ]  # 形状: 列表(4)

    def _render_normals_fixed(
        self,
        meshes: List[Any],
        idxs: List[int],
        H: int,
        W: int,
        R: int,
    ) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
        """使用固定相机渲染法线。"""
        return render_normals_predefined(
            meshes=meshes,
            idxs=idxs,
            pose_list=self._get_fixed_predefined_poses(),
            img_size=int(R),
            R=int(R),
            device=self.device,
        )

    def _render_normals_estimated(
        self,
        meshes: List[Any],
        idxs: List[int],
        meta0: Dict[str, Any],
        H: int,
        W: int,
        R: int,
    ) -> tuple[torch.Tensor, torch.Tensor, List[int], Optional[Dict[str, Any]], torch.Tensor]:
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
        extri_all, intr_all, intr_pix_all = batch_estimate_camera(
            self.camera, images_batched, support, H, W, R, int(self.cfg.cam_batch_size)
        )  # 形状: (K,4,4),(K,3,3),(K,3,3)
        avg_camera = None
        if bool(self.cfg.avg_camera_per_group):
            w2c_mean, Kpix_mean = self._avg_w2c_K(extri_all, intr_pix_all)
            extri_use = w2c_mean.unsqueeze(0).expand(extri_all.shape[0], -1, -1).contiguous()
            intr_pix_use = Kpix_mean.unsqueeze(0).expand(intr_pix_all.shape[0], -1, -1).contiguous()
            avg_camera = {
                "w2c": w2c_mean.detach().cpu().tolist(),
                "K_pix": Kpix_mean.detach().cpu().tolist(),
            }
        else:
            extri_use = extri_all
            intr_pix_use = intr_pix_all
        n_mesh_all, m_mesh_all = render_normals_batched(
            meshes, idxs, extri_use, intr_pix_use, H, R, self.device
        )
        return n_mesh_all, m_mesh_all, list(idxs), avg_camera, images_batched

    # -------------------- 基础工具 --------------------
    def _get_image_path(self, meta: Dict[str, Any]) -> str:
        if isinstance(meta, dict) and "image_path" in meta:
            return str(meta["image_path"])  # 形状: 标量
        if isinstance(meta, dict) and "image_name" in meta:
            base_dir = os.environ.get("FLOW_GRPO_DATA_DIR", "dataset/eval3d_hunyuan3d")  # 形状: 标量
            return os.path.join(base_dir, "images", str(meta["image_name"]))  # 形状: 标量
        raise ValueError("metadata 缺少 image_path 或 image_name")

    # 删除本类内的转换方法，直接在调用点使用 utils.transforms

    def _avg_w2c_K(self, w2c_all: torch.Tensor, Kpix_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对同组 K 个相机做均值：旋转取最近旋转（SVD 投影），平移与内参算术平均。

        输入:
            w2c_all: (K,4,4) OpenCV 世界到相机变换
            Kpix_all: (K,3,3) 像素坐标内参（H×W 基准）
        输出:
            w2c_mean: (4,4)
            Kpix_mean: (3,3)
        """
        R_stack = w2c_all[:, :3, :3]  # 形状: (K,3,3)
        t_stack = w2c_all[:, :3, 3]   # 形状: (K,3)

        A = R_stack.mean(dim=0)  # 形状: (3,3)
        U, S, Vh = torch.linalg.svd(A)  # 形状: U(3,3), S(3,), Vh(3,3)
        Rm = U @ Vh  # 形状: (3,3)
        if torch.det(Rm) < 0:  # 形状: 标量
            U_fix = U.clone()  # 形状: (3,3)
            U_fix[:, -1] = -U_fix[:, -1]  # 形状: (3,)
            Rm = U_fix @ Vh  # 形状: (3,3)

        tm = t_stack.mean(dim=0)  # 形状: (3)
        w2c_mean = torch.eye(4, device=w2c_all.device, dtype=w2c_all.dtype)  # 形状: (4,4)
        w2c_mean = w2c_mean.clone()  # 形状: (4,4)
        w2c_mean[:3, :3] = Rm  # 形状: (4,4)
        w2c_mean[:3, 3] = tm   # 形状: (4,4)

        Kpix_mean = Kpix_all.mean(dim=0)  # 形状: (3,3)
        return w2c_mean, Kpix_mean  # 形状: (4,4), (3,3)

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

    

    def build_best_worst_pairs(self, grouped_meta: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """从 compute_scores 返回的 grouped_meta 中，挑选每组分数最高/最低的候选，
        并构造成用于可视化/记录的展平列表。

        返回:
            filtered_meta_best, filtered_meta_worst
        其中每个元素包含键:
            - image_path: str
            - image_normal_pil: PIL(R,R,3)
            - rendered_normal_pil: PIL(R,R,3)
            - mesh_index: int
            - score: float
        """
        filtered_meta_best: List[Dict[str, Any]] = []  # 形状: 长度 G 的列表
        filtered_meta_worst: List[Dict[str, Any]] = []  # 形状: 长度 G 的列表
        for grp in grouped_meta:
            image_path = str(grp["image_path"])   # 形状: 字符串
            img_pil = grp["image_normal_pil"]  # 形状: PIL(R,R,3)
            cands = grp["candidates"]  # 形状: 长度 K 的列表
            if len(cands) == 0:
                continue
            best = cands[0]
            worst = cands[0]
            for cand in cands[1:]:
                score_c = float(cand.get("score", -1.0))
                if score_c > float(best.get("score", -1.0)):
                    best = cand
                if score_c < float(worst.get("score", 1e9)):
                    worst = cand
            filtered_meta_best.append({
                "image_path": image_path,
                "image_normal_pil": img_pil,
                "rendered_normal_pil": best.get("rendered_normal_pil"),
                "mesh_index": int(best.get("mesh_index", -1)),
                "score": float(best.get("score", 0.0)),
            })
            filtered_meta_worst.append({
                "image_path": image_path,
                "image_normal_pil": img_pil,
                "rendered_normal_pil": worst.get("rendered_normal_pil"),
                "mesh_index": int(worst.get("mesh_index", -1)),
                "score": float(worst.get("score", 0.0)),
            })
        return filtered_meta_best, filtered_meta_worst  # 形状: 长度 G 的列表, 长度 G 的列表

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

        # 收集：渲染法线，以及映射关系
        rendered_normals_all: List[torch.Tensor] = []  # 多组拼接后 (sumK,3,R,R)
        rendered_masks_all: List[torch.Tensor] = []    # 多组拼接后 (sumK,R,R)
        mesh_global_indices: List[int] = []  # 长度 sumK
        mesh_group_indices: List[int] = []  # 长度 sumK，对应每个渲染法线所属组 id

        # 2) 逐组估计相机 + 渲染
        # 分组式元数据：每张图像仅保存一次 image_path 与 image_normal_pil
        grouped_meta: List[Dict[str, Any]] = []  # 形状: 长度 G
        pair_j_to_group_local: List[tuple[int, int]] = []  # 形状: 长度 M，映射合并序 j -> (gid, local_idx)
        for gid, image_path in enumerate(image_paths):
            idxs = groups[image_path]  # 形状: 长度 K
            # 图像侧法线（来自 normal_pil）
            meta0 = metadata[idxs[0]]  # 形状: 字典
            assert ("normal_pil" in meta0) and (meta0["normal_pil"] is not None), "metadata.normal_pil 是必需的"
            n_img = to_tensor_from_normal_pil(meta0["normal_pil"], R, self.device)  # 形状: (3,R,R)
            # 分组元数据记录（只存一次图像侧信息）
            normal_pil = normal_tensor_to_pil(n_img)  # 形状: PIL(R,R,3)
            # 额外存储组级 RGB PIL（供 PickScore 走 RGB 对比路径时使用）
            rgb_pil = images[idxs[0]]  # 形状: PIL(H,W,3|4)
            grouped_meta.append({
                "group_id": int(gid),                  # 形状: 标量
                "image_path": str(image_path),         # 形状: 字符串
                "image_normal_pil": normal_pil,        # 形状: PIL(R,R,3)
                "image_rgb_pil": rgb_pil,              # 形状: PIL(H,W,3)
                "candidates": [],                      # 形状: 长度 K 的列表（稍后填充）
            })  # 形状: 追加

            avg_camera = None
            images_batched_vis = None
            if self._use_camera_search:
                (
                    n_mesh_all,
                    m_mesh_all,
                    mesh_indices_for_outputs,
                    avg_camera,
                    images_batched_vis,
                ) = self._render_normals_estimated(
                    meshes=meshes,
                    idxs=idxs,
                    meta0=meta0,
                    H=H,
                    W=W,
                    R=R,
                )
            else:
                n_mesh_all, m_mesh_all, mesh_indices_for_outputs = self._render_normals_fixed(
                    meshes=meshes,
                    idxs=idxs,
                    H=H,
                    W=W,
                    R=R,
                )  # 形状: (K*V,3,R,R),(K*V,R,R),列表(K*V)

            rendered_normals_all.append(n_mesh_all)  # 形状: 追加
            rendered_masks_all.append(m_mesh_all)    # 形状: 追加
            mesh_global_indices.extend(mesh_indices_for_outputs)  # 形状: 追加条目
            mesh_group_indices.extend([gid] * n_mesh_all.shape[0])  # 形状: 追加
            # 记录每个候选（仅保存渲染法线；图像侧法线保存在组级）
            for j in range(n_mesh_all.shape[0]):
                mesh_norm_pil = normal_tensor_to_pil(n_mesh_all[j])  # 形状: PIL(R,R,3)
                local_idx = len(grouped_meta[gid]["candidates"])  # 形状: 标量
                grouped_meta[gid]["candidates"].append({
                    "mesh_index": int(mesh_indices_for_outputs[j]),            # 形状: 标量
                    "rendered_normal_pil": mesh_norm_pil, # 形状: PIL(R,R,3)
                    "score": None,                         # 形状: 占位
                })  # 形状: 追加
                pair_j_to_group_local.append((gid, local_idx))  # 形状: 追加

            if avg_camera is not None:
                grouped_meta[gid]["avg_camera"] = avg_camera

            # 可视化保存（方案2：为每张图像建立一层子目录，按每个 mesh 单独保存）
            if bool(self.cfg.save_vis) and self._use_camera_search and images_batched_vis is not None:
                os.makedirs(self.cfg.vis_dir, exist_ok=True)
                # 目标：与 mesh 导出一致：.../generated_meshes/eval_epoch_{epoch}/{safe_base}/
                # 在此处根据 image_path 追加 safe_base 子目录，确保与 save_meshes_for_preview 对齐
                base = os.path.splitext(os.path.basename(image_path))[0]
                safe_base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).rstrip()
                case_dir = os.path.join(self.cfg.vis_dir, safe_base)
                os.makedirs(case_dir, exist_ok=True)
                for j in range(n_mesh_all.shape[0]):
                    save_camera_search_visualization(
                        images_batched_vis,            # 形状: (K,S,3,H,W)
                        n_img,                     # 形状: (3,R,R)
                        n_mesh_all[j],             # 形状: (3,R,R)
                        case_dir,
                        f"camera_{j}",
                    )

        # 3) 统一编码：构造 PIL 列表并进入编码器（DINO/PickScore 均接受 PIL）
        G = len(grouped_meta)  # 形状: 标量
        if G == 0:
            return [], []  # 形状: 空列表

        # 组 PIL：仅依据 use_RGB_for_comparison 选择 RGB 或法线
        key = ("image_rgb_pil" if bool(self.cfg.use_RGB_for_comparison) else "image_normal_pil")  # 形状: 字符串
        group_pils: List[Image.Image] = [grp[key] for grp in grouped_meta]  # 形状: 长度 G

        # 候选 PIL：直接复用前面保存的渲染法线 PIL（顺序由 pair_j_to_group_local 保证与 mesh_group_indices 对齐）
        mesh_pils: List[Image.Image] = []  # 形状: 长度 M
        for (gid, lidx) in pair_j_to_group_local:
            mesh_pils.append(grouped_meta[gid]["candidates"][lidx]["rendered_normal_pil"])  # 形状: PIL(R,R,3)

        # 掩码（供 DINO match_* 使用）；PickScore 将忽略
        mask_mesh_cat = torch.cat(rendered_masks_all, dim=0) if len(rendered_masks_all) > 0 else torch.zeros(0, R, R, device=self.device, dtype=torch.bool)  # 形状: (M,R,R)

        # 相似度：完全委托给编码器，由其内部依据 sim_type 决策
        bs = int(self.cfg.encoding_batch_size)  # 形状: 标量
        rewards_vec = self.encoder.score_pairs(
            group_pils,  # 形状: 长度 G
            mesh_pils,  # 形状: 长度 M
            mesh_group_indices,  # 形状: 长度 M
            mask_meshes=(mask_mesh_cat if mask_mesh_cat.numel() > 0 else None),  # 形状: (M,R,R) 或 None
            batch_size=bs,  # 形状: 标量
        )  # 形状: (M,)

        rewards_all: List[float] = [0.0 for _ in range(len(meshes))]  # 形状: 长度 N_total
        per_mesh_scores: Dict[int, List[float]] = {}  # 形状: 字典
        for j, midx in enumerate(mesh_global_indices):
            score_j = float(rewards_vec[j].item())  # 形状: 标量
            per_mesh_scores.setdefault(int(midx), []).append(score_j)  # 形状: 列表追加
            if j < len(pair_j_to_group_local):
                gid, lidx = pair_j_to_group_local[j]  # 形状: 标量, 标量
                grouped_meta[gid]["candidates"][lidx]["score"] = score_j  # 形状: 标量
        
        for midx, score_list in per_mesh_scores.items():
            if self._use_max_score:
                final_score = float(max(score_list))  # 形状: 标量
            else:
                final_score = float(sum(score_list) / max(1, len(score_list)))  # 形状: 标量
            rewards_all[midx] = final_score  # 形状: 标量

        return rewards_all, grouped_meta

