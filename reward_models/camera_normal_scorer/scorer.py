import os
from typing import Any, Dict, List
import importlib
import sys
import torch
from PIL import Image

from .config import ScorerConfig
from .types import RendererProtocol
from .normal_io.cache import load_normal_from_cache
from .camera.vggt_estimator import VGGTSearchEstimator
from .encoders.dino_encoder import DinoNormalEncoder
from .camera_estimation import normalize_intrinsics_to_R
from .render.adapter import to_mesh_extract, compose_white_background
# 确保参考库根目录在 sys.path，便于其内部使用 from training.xxx 导入
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from _reference_codes.VGGTObj.vggt.utils.pose_enc import extri_intri_to_pose_encoding
from _reference_codes.VGGTObj.vggt_camera_search.normal_predictor import create_normal_predictor
from _reference_codes.VGGTObj.training.utils.coordinate_conversion import CoordinateConverter
import torchvision.transforms as T


class CameraNormalScorer:
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

    @torch.no_grad()
    def compute_scores(
        self,
        meshes: List[Any],
        images: List[Image.Image],
        metadata: List[Dict[str, Any]],
        renderer: RendererProtocol,
    ) -> List[float]:
        assert len(meshes) == len(images) == len(metadata), "输入列表长度需一致"  # 形状: 断言
        R = int(self.cfg.resolution)  # 形状: 标量

        def get_image_path(meta: Dict[str, Any]) -> str:
            if isinstance(meta, dict) and "image_path" in meta:
                return str(meta["image_path"])  # 形状: 标量
            if isinstance(meta, dict) and "image_name" in meta:
                base_dir = os.environ.get("FLOW_GRPO_DATA_DIR", "dataset/eval3d")  # 形状: 标量
                return os.path.join(base_dir, "images", str(meta["image_name"]))  # 形状: 标量
            raise ValueError("metadata 缺少 image_path 或 image_name")

        groups: Dict[str, List[int]] = {}
        for idx, meta in enumerate(metadata):
            p = get_image_path(meta)  # 形状: 标量
            groups.setdefault(p, []).append(idx)  # 形状: 追加

        rewards_all: List[float] = [0.0 for _ in range(len(meshes))]  # 形状: (K,)
        for image_path, idxs in groups.items():
            n_img = load_normal_from_cache(image_path, self.cfg.cache_dir, R).to(self.device)  # 形状: (3,R,R)
            f_img = self.encoder.feature_from_normal(n_img)  # 形状: (1,D)

            # Query 图像与参考脚本一致：支持 rgb / normal_pred / normal_image
            if self.cfg.query_input not in {"rgb", "normal_pred", "normal_image"}:
                raise ValueError("query_input 必须为 {'rgb','normal_pred','normal_image'}")
            img_size = int(self.cfg.img_size)  # 形状: 标量
            if self.cfg.query_input == "rgb":
                img = Image.open(image_path).convert("RGB")
                transform = T.Compose([
                    T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                ])
                query_tensor = transform(img).to(self.device)  # 形状: (3,H,W)
            else:
                if self.cfg.query_input == "normal_pred":
                    predictor = create_normal_predictor(
                        weights_dir=self.cfg.normal_weights_dir,
                        yoso_version=self.cfg.normal_version,
                        device=str(self.device),
                    )
                    rgb_img = Image.open(image_path).convert("RGB")
                    normal_img = predictor.predict(
                        rgb_img,
                        resolution=img_size,
                        match_input_resolution=True,
                        data_type="object",
                    )
                    img_for_tensor = normal_img
                else:  # normal_image
                    img_for_tensor = Image.open(image_path).convert("RGB")

                transform = T.Compose([
                    T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                ])
                query_tensor = transform(img_for_tensor).to(self.device)  # 形状: (3,H,W)

            imgs_query = query_tensor.unsqueeze(0)  # 形状: (1,3,H,W)
            H, W = img_size, img_size  # 形状: 标量, 标量
            K = len(idxs)  # 形状: 标量

            # 为每个 mesh 使用参考渲染器构造 support（S-1 固定来自 config）并拼接 query
            images_seqs = []  # 形状: K*[S,3,H,W]
            supports = []     # 形状: K*[S-1,D]

            # 读取参考配置中的预设相机，并初始化参考渲染器
            mod = importlib.import_module(self.cfg.camera_config_py.replace('/', '.').replace('.py', ''))  # 形状: 模块
            if hasattr(mod, 'get_camera_search_seven_view_config'):
                cfg_ref = mod.get_camera_search_seven_view_config()  # 形状: 配置
                fixed_poses = getattr(cfg_ref.render, 'predefined_poses', [])
            else:
                raise ValueError(f"未找到 get_camera_search_seven_view_config 于 {self.cfg.camera_config_py}")
            ref_renderer = RefMeshRenderer(img_size=int(self.cfg.img_size), device=str(self.device))  # 形状: 渲染器

            for j in idxs:
                mesh_ex = to_mesh_extract(meshes[j], self.device)  # 形状: MeshExtractResult

                # 适配为参考渲染器可用的mesh对象（鸭子类型：需 .v/.f，可无 .vn）
                class _KiuiMeshLike:
                    def __init__(self, v: torch.Tensor, f: torch.Tensor) -> None:
                        self.v = v  # 形状: (V,3)
                        self.f = f  # 形状: (F,3)
                        self.vn = None  # 形状: 可为空

                mesh_kiui = _KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: MeshLike

                # 生成固定 support 相机并渲染（输出OpenCV W2C与像素内参）
                cams_fixed = ref_renderer.sample_camera_poses(num_random_views=0, predefined_poses=fixed_poses)  # 形状: 列表
                sup_out = ref_renderer.render_mesh(
                    mesh=mesh_kiui,
                    cameras=cams_fixed,
                    return_depth=False,
                    return_normals=False,
                    return_positions=False,
                    return_masks=False,
                )
                images_s = sup_out['images'].to(self.device)  # 形状: (S-1,3,H,W)
                extr_s = sup_out['extrinsics'].to(self.device)  # 形状: (S-1,3,4)
                intr_s = sup_out['intrinsics'].to(self.device)  # 形状: (S-1,4) -> [fx,fy,cx,cy]

                # 组装像素内参为 (1,S-1,3,3)，并编码为 9 维 support cameras
                Ssup = images_s.shape[0]  # 形状: 标量
                intr_33 = torch.zeros(1, Ssup, 3, 3, device=self.device, dtype=intr_s.dtype)  # 形状: (1,S-1,3,3)
                intr_33[:, :, 0, 0] = intr_s[:, 0].unsqueeze(0)  # 形状: (1,S-1)
                intr_33[:, :, 1, 1] = intr_s[:, 1].unsqueeze(0)  # 形状: (1,S-1)
                intr_33[:, :, 0, 2] = intr_s[:, 2].unsqueeze(0)  # 形状: (1,S-1)
                intr_33[:, :, 1, 2] = intr_s[:, 3].unsqueeze(0)  # 形状: (1,S-1)
                intr_33[:, :, 2, 2] = 1.0  # 形状: (1,S-1)

                if int(self.cfg.camera_param_dim) == 9:
                    pose_sup = extri_intri_to_pose_encoding(
                        extr_s.unsqueeze(0),  # 形状: (1,S-1,3,4)
                        intr_33,              # 形状: (1,S-1,3,3)
                        image_size_hw=(H, W),
                    )[0]  # 形状: (S-1,9)
                else:
                    pose_sup = extr_s.reshape(Ssup, -1)  # 形状: (S-1,12)

                images_seq = torch.cat([images_s.unsqueeze(0), imgs_query.unsqueeze(0)], dim=1)  # 形状: (1,S,3,H,W)
                images_seqs.append(images_seq)
                supports.append(pose_sup)  # 形状: (S-1,D)

            images_batched = torch.cat(images_seqs, dim=0).to(self.device)  # 形状: (K,S,3,H,W)
            support = torch.stack(supports, dim=0).to(self.device)  # 形状: (K,S-1,D)

            cam_bs = int(self.cfg.cam_batch_size)  # 形状: 标量
            extri_list, intr_list, intr_pix_list = [], [], []
            for s in range(0, K, cam_bs):
                e = min(K, s + cam_bs)  # 形状: 标量
                extri_4x4, intr_3x3 = self.camera.estimate(images_batched[s:e], support[s:e], (H, W))  # 形状: (b,4,4),(b,3,3)
                intr_R = normalize_intrinsics_to_R(intr_3x3, H, W, R)  # 形状: (b,3,3)
                extri_list.append(extri_4x4)  # 形状: 追加
                intr_list.append(intr_R)  # 形状: 追加
                intr_pix_list.append(intr_3x3)  # 形状: 追加（像素内参，基于 (H,W)=img_size）
            extri_all = torch.cat(extri_list, dim=0)  # 形状: (K,4,4)
            intr_all = torch.cat(intr_list, dim=0)  # 形状: (K,3,3)
            intr_pix_all = torch.cat(intr_pix_list, dim=0)  # 形状: (K,3,3)

            # 使用参考渲染器在评分阶段渲染（与训练一致）。渲染尺寸为 R，并将像素K按比例从 (H,W) 重标到 (R,R)。
            ref_renderer_score = RefMeshRenderer(img_size=R, device=str(self.device))  # 形状: 渲染器
            render_bs = int(self.cfg.render_batch_size)  # 形状: 标量
            n_mesh_list = []
            for s in range(0, K, render_bs):
                e = min(K, s + render_bs)  # 形状: 标量
                for j in range(s, e):
                    mesh_ex = to_mesh_extract(meshes[idxs[j]], self.device)  # 形状: MeshExtractResult

                    # 适配为参考渲染器可用mesh
                    class _KiuiMeshLike:
                        def __init__(self, v: torch.Tensor, f: torch.Tensor) -> None:
                            self.v = v  # 形状: (V,3)
                            self.f = f  # 形状: (F,3)
                            self.vn = None

                    mesh_kiui = _KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)

                    # OpenCV W2C(3x4) -> OpenGL C2W(4x4)
                    w2c34 = extri_all[j][:3, :]  # 形状: (3,4)
                    w2c_bv = w2c34.view(1, 1, 3, 4)  # 形状: (1,1,3,4)
                    c2w_bv = CoordinateConverter.opencv_w2c_to_opengl_c2w(w2c_bv)  # 形状: (1,1,4,4)
                    c2w_b = c2w_bv.view(1, 4, 4)  # 形状: (1,4,4)

                    # 将像素内参从 (H,W)=img_size 缩放到 (R,R)
                    K_pix = intr_pix_all[j].clone()  # 形状: (3,3)
                    scale = float(R) / float(W)
                    K_pix[0, 0] = K_pix[0, 0] * scale  # 形状: 标量
                    K_pix[1, 1] = K_pix[1, 1] * scale  # 形状: 标量
                    K_pix[0, 2] = K_pix[0, 2] * scale  # 形状: 标量
                    K_pix[1, 2] = K_pix[1, 2] * scale  # 形状: 标量
                    K_b = K_pix.view(1, 3, 3)  # 形状: (1,3,3)

                    sup_out = ref_renderer_score.render_mesh(
                        mesh=mesh_kiui,
                        c2w=c2w_b,  # 形状: (1,4,4)
                        K=K_b,      # 形状: (1,3,3)
                        return_depth=False,
                        return_normals=False,
                        return_positions=False,
                        return_masks=False,
                    )
                    img01 = sup_out['images'][0]  # 形状: (3,R,R) in [0,1]
                    n_mesh = (img01 * 2.0 - 1.0).clamp(-1, 1)  # 形状: (3,R,R)
                    n_mesh_list.append(n_mesh)

            n_mesh_all = torch.stack(n_mesh_list, dim=0)  # 形状: (K,3,R,R)

            rewards_chunks = []
            bs_dino = int(self.cfg.dino_batch_size)  # 形状: 标量
            for s in range(0, K, bs_dino):
                e = min(K, s + bs_dino)  # 形状: 标量
                f_mesh = self.encoder.features_from_normals(n_mesh_all[s:e])  # 形状: (b,D)
                rewards_b = ((f_mesh @ f_img.t()).squeeze(-1) + 1.0) * 0.5  # 形状: (b,)
                rewards_chunks.append(rewards_b)
            rewards_k = torch.cat(rewards_chunks, dim=0).tolist()  # 形状: 长度 K

            if self.cfg.save_vis:
                from .vis.save import save_similarity_inputs
                os.makedirs(self.cfg.vis_dir, exist_ok=True)
                tag = os.path.splitext(os.path.basename(image_path))[0]
                save_similarity_inputs(n_img, n_mesh_all[0], self.cfg.vis_dir, tag)

            for loc, score in enumerate(rewards_k):
                rewards_all[idxs[loc]] = float(score)  # 形状: 标量

        return rewards_all


