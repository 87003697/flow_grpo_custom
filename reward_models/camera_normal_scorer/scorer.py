import os
from typing import Any, Dict, List
import torch
from PIL import Image
import torchvision.transforms as T

from .config import ScorerConfig
from .normal_io.cache import load_normal_from_cache
from .camera.vggt_estimator import VGGTSearchEstimator
from .encoders.dino_encoder import DinoNormalEncoder
from .camera.support import build_support_batches
from .camera.estimate_utils import batch_estimate_camera
from .render.render_normals import render_normals_batched
from .vis.save import save_camera_search_visualization


class CameraNormalScorer:
    """基于相机搜索与法线相似度的网格评分器。

    功能:
        - 为每个 mesh 生成固定 support 视角并与 query 图像共同输入 VGGT 相机搜索模型，估计 query 视角相机参数。
        - 使用参考渲染器按估计相机渲染法线图，将其与图像侧法线特征做余弦相似度得到奖励分数。

    参考:
        - 参考渲染器: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py`
        - 姿态编码: `_reference_codes/VGGTObj/vggt/utils/pose_enc.py`
        - 坐标系转换: `_reference_codes/VGGTObj/training/utils/coordinate_conversion.py`
    """
    def __init__(self, device: torch.device, cfg: Dict[str, Any]) -> None:
        self.device = device
        self.cfg = ScorerConfig(**cfg)

        # 编码器选择
        if self.cfg.encoder == "dino_v2":
            model_id = self.cfg.dino_v2_path  # 形状: 标量
        else:
            model_id = self.cfg.dino_v3_path  # 形状: 标量
        self.encoder = DinoNormalEncoder(model_id=model_id, device=device)  # 形状: 编码器

        # 相机估计器（固定 camera_param_dim=9, img_size=518），支持外部 checkpoint
        self.camera = VGGTSearchEstimator(
            device,
            camera_param_dim=int(self.cfg.camera_param_dim),
            img_size=int(self.cfg.img_size),
            ckpt=getattr(self.cfg, 'camera_ckpt', ''),
        )  # 形状: 相机估计器

    # -------------------- 私有工具函数：输入/支持/相机/渲染/相似度 --------------------
    def _get_image_path(self, meta: Dict[str, Any]) -> str:
        """从 metadata 中解析图像路径。

        输入:
            meta: 元数据字典，包含 `image_path` 或 `image_name`。
        输出:
            字符串图像路径。
        功能:
            支持两种来源，优先使用显式 `image_path`，否则拼接 `FLOW_GRPO_DATA_DIR/images/{image_name}`。
        参考: 无直接参考（项目内约定）。
        """
        if isinstance(meta, dict) and "image_path" in meta:
            return str(meta["image_path"])  # 形状: 标量
        if isinstance(meta, dict) and "image_name" in meta:
            base_dir = os.environ.get("FLOW_GRPO_DATA_DIR", "dataset/eval3d")  # 形状: 标量
            return os.path.join(base_dir, "images", str(meta["image_name"]))  # 形状: 标量
        raise ValueError("metadata 缺少 image_path 或 image_name")

    def _build_query_from_metadata(self, meta: Dict[str, Any]) -> torch.Tensor:
        """从 metadata 构造 query 张量，不依赖 cfg.query_input。

        规则:
            - 若包含 `normal_pil`，直接用该 PIL 转为 (1,3,H,W)
            - 否则若包含 `normal_path`，从路径读入并转为 (1,3,H,W)
            - 以上都不存在则报错（不做回退）
        """
        H, W = int(self.cfg.img_size), int(self.cfg.img_size)  # 形状: 标量, 标量
        transform = T.Compose([
            T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        if ("normal_pil" in meta) and (meta["normal_pil"] is not None):
            q = transform(meta["normal_pil"]).to(self.device)  # 形状: (3,H,W)
            return q.unsqueeze(0)  # 形状: (1,3,H,W)
        if ("normal_path" in meta) and (meta["normal_path"] is not None):
            img = Image.open(meta["normal_path"]).convert("RGB")  # 形状: (h,w,3)
            q = transform(img).to(self.device)  # 形状: (3,H,W)
            return q.unsqueeze(0)  # 形状: (1,3,H,W)
        raise ValueError("metadata 必须包含 normal_pil 或 normal_path")


    def _build_support_batches(self, meshes: List[Any], idxs: List[int], imgs_query: torch.Tensor, H: int, W: int):
        """为一组 mesh 构建 support 批次并拼接 query。

        输入:
            meshes: 原始 mesh 列表。
            idxs: 本组对应的索引列表。
            imgs_query: (1,3,H,W) 的 query 图像张量。
            H, W: 图像高宽。
        输出:
            images_batched: (K,S,3,H,W)
            support: (K,S-1,D)
        参考:
            - 渲染与相机采样: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L102-L157, L179-L215
            - 姿态编码: `_reference_codes/VGGTObj/vggt/utils/pose_enc.py` L11-L41
        """
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
        )
        return images_batched, support

    def _batch_estimate_camera(self, images_batched: torch.Tensor, support: torch.Tensor, H: int, W: int, R: int):
        """分批估计相机并返回像素/归一化内参与外参。

        输入:
            images_batched: (K,S,3,H,W)
            support: (K,S-1,D)
            H, W: 图像尺寸；R: 渲染分辨率。
        输出:
            extri_all: (K,4,4) OpenCV W2C(4x4)
            intr_all: (K,3,3) 归一化到 R×R 的内参
            intr_pix_all: (K,3,3) 像素内参（基于 H×W）
        参考:
            - 归一化内参: `reward_models/camera_normal_scorer/camera_estimation.py` L14-L25
        """
        return batch_estimate_camera(self.camera, images_batched, support, H, W, R, int(self.cfg.cam_batch_size))

    def _render_normals_batched(self, meshes: List[Any], idxs: List[int], extri_all: torch.Tensor, intr_pix_all: torch.Tensor, R: int, W: int) -> torch.Tensor:
        """按估计相机渲染法线，输出 [-1,1] 取值。

        输入:
            meshes, idxs: mesh 列表与当前组索引。
            extri_all: (K,4,4) OpenCV W2C。
            intr_pix_all: (K,3,3) 像素内参。
            R, W: 渲染分辨率与原图宽。
        输出:
            n_mesh_all: (K,3,R,R)
        参考:
            - 坐标系转换: `_reference_codes/VGGTObj/training/utils/coordinate_conversion.py` L21-L69
            - 渲染接口: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L179-L215
        """
        return render_normals_batched(meshes, idxs, extri_all, intr_pix_all, R, W, self.device)

    def _compute_rewards_from_normals(self, n_mesh_all: torch.Tensor, f_img: torch.Tensor) -> List[float]:
        """将渲染法线编码并与图像法线特征做余弦相似度，映射到 [0,1]。

        输入:
            n_mesh_all: (K,3,R,R)
            f_img: (1,D)
        输出:
            rewards_k: 长度 K 的打分列表。
        参考:
            - 余弦实现: `reward_models/camera_normal_scorer/similarity/cosine.py` L4-L7
        """
        # 输出: rewards_k 长度 K 的列表
        K = n_mesh_all.shape[0]  # 形状: 标量
        rewards_chunks = []
        bs_dino = int(self.cfg.dino_batch_size)  # 形状: 标量
        for s in range(0, K, bs_dino):
            e = min(K, s + bs_dino)  # 形状: 标量
            f_mesh = self.encoder.features_from_normals(n_mesh_all[s:e])  # 形状: (b,D)
            rewards_b = ((f_mesh @ f_img.t()).squeeze(-1) + 1.0) * 0.5  # 形状: (b,)
            rewards_chunks.append(rewards_b)
        rewards_k = torch.cat(rewards_chunks, dim=0).tolist()  # 形状: 长度 K
        return rewards_k

    @torch.no_grad()
    def compute_scores(
        self,
        meshes: List[Any],
        images: List[Image.Image],
        metadata: List[Dict[str, Any]],
    ) -> List[float]:
        """为一组 (mesh, image, meta) 计算相似度奖励。

        输入:
            meshes: 待评估 mesh 列表。
            images: 原图列表（目前未直接使用）。
            metadata: 含 image_path/image_name 的字典列表。
            renderer: 兼容接口的渲染器（此实现使用参考渲染器）。
        输出:
            rewards_all: 与 meshes 等长的分数列表，范围约 [0,1]。
        参考:
            - 参考渲染器: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py`
            - VGGT 相机搜索: `reward_models/camera_normal_scorer/camera/vggt_estimator.py`
        """
        assert len(meshes) == len(images) == len(metadata), "输入列表长度需一致"  # 形状: 断言
        R = int(self.cfg.resolution)  # 形状: 标量

        groups: Dict[str, List[int]] = {}
        for idx, meta in enumerate(metadata):
            p = self._get_image_path(meta)  # 形状: 标量
            groups.setdefault(p, []).append(idx)  # 形状: 追加

        rewards_all: List[float] = [0.0 for _ in range(len(meshes))]  # 形状: (K,)
        for image_path, idxs in groups.items():
            n_img = load_normal_from_cache(image_path, self.cfg.cache_dir, R).to(self.device)  # 形状: (3,R,R)
            f_img = self.encoder.feature_from_normal(n_img)  # 形状: (1,D)

            # 构造 query：仅基于 metadata 内容（normal_pil > normal_path），不依赖 cfg.query_input
            meta0 = metadata[idxs[0]]  # 形状: 字典
            imgs_query = self._build_query_from_metadata(meta0)  # 形状: (1,3,H,W)

            K = len(idxs)  # 形状: 标量
            H, W = int(self.cfg.img_size), int(self.cfg.img_size)  # 形状: 标量, 标量
            images_batched, support = self._build_support_batches(meshes, idxs, imgs_query, H, W)

            extri_all, intr_all, intr_pix_all = self._batch_estimate_camera(images_batched, support, H, W, R)

            n_mesh_all = self._render_normals_batched(meshes, idxs, extri_all, intr_pix_all, R, W)

            rewards_k = self._compute_rewards_from_normals(n_mesh_all, f_img)

            if self.cfg.save_vis:
                os.makedirs(self.cfg.vis_dir, exist_ok=True)
                tag = os.path.splitext(os.path.basename(image_path))[0]
                save_camera_search_visualization(images_batched, n_img, n_mesh_all[0], self.cfg.vis_dir, tag)

            for loc, score in enumerate(rewards_k):
                rewards_all[idxs[loc]] = float(score)  # 形状: 标量

        return rewards_all


