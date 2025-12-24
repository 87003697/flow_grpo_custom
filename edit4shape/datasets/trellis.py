from utils3d.torch.transforms import intrinsics_from_fov_xy

import glob
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import utils3d
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .utils import build_mvp_from_w2c


def build_trellis_intrinsics(fovy_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fov_tensor = torch.tensor([math.radians(fovy_deg)], device=device, dtype=dtype)  # [1]
    intrinsics = intrinsics_from_fov_xy(fov_tensor, fov_tensor)[0]  # [3,3]
    return intrinsics  # [3,3]


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float = 1.0,
    far: float = 100.0,
) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # [], []
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # [], []
    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)  # [4,4]
    proj[0, 0] = 2 * fx  # []
    proj[1, 1] = 2 * fy  # []
    proj[0, 2] = 2 * cx - 1  # []
    proj[1, 2] = -2 * cy + 1  # []
    proj[2, 2] = far / (far - near)  # []
    proj[2, 3] = near * far / (near - far)  # []
    proj[3, 2] = 1.0  # []
    return proj  # [4,4]


class BaseImageDatasetTrellis(Dataset):
    """Trellis 变体数据集（去除 threestudio / default 依赖）"""

    def __init__(self, cfg: "TrellisDataConfig", split: str):
        self.cfg = cfg
        self.split = split
        self.image_paths = self._load_image_paths()  # [N]

    def _load_image_paths(self) -> List[str]:
        """从 eval_image_path 或 image_dataset_dir 读取图像列表"""
        image_paths: List[str] = []

        # eval 模式优先 eval_image_path
        if self.split == "test" and self.cfg.eval_image_path:
            if os.path.exists(self.cfg.eval_image_path):
                if os.path.isfile(self.cfg.eval_image_path):
                    image_paths = [self.cfg.eval_image_path]
                else:
                    extensions = list(set(["png", "jpg", "jpeg", self.cfg.image_file_extension]))
                    for ext in extensions:
                        image_paths.extend(glob.glob(os.path.join(self.cfg.eval_image_path, f"*.{ext}")))

        # 仅在 train/val 或未提供 eval_image_path 时使用 image_dataset_dir（不做二次 fallback）
        if not image_paths and self.cfg.image_dataset_dir:
            image_dir = self.cfg.image_dataset_dir
            if os.path.exists(image_dir):
                extensions = list(set(["png", "jpg", "jpeg", self.cfg.image_file_extension]))
                for ext in extensions:
                    image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_paths:
            raise FileNotFoundError(f"No images found under eval_image_path={self.cfg.eval_image_path} or image_dataset_dir={self.cfg.image_dataset_dir}")

        return sorted(list(set(image_paths)))  # [N]

    def compute_views_uniform(self, num_views: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        统一采样相机参数: yaw, pitch, r, fov。
        训练时从各自 range 随机采样；评估时使用固定值。

        Returns:
            yaws_deg: [V] yaw 角度 (度)
            pitch_deg: [V] pitch 角度 (度)
            r: float 相机距离
            fov: float 视场角 (度)
        """
        is_eval = self.split in ("test", "val")

        if is_eval:
            # 评估时使用固定值
            eval_cfg = self.cfg.eval
            yaws_deg = torch.full((num_views,), eval_cfg.yaw, dtype=torch.float32)  # [V]
            pitch_deg = torch.full((num_views,), eval_cfg.pitch, dtype=torch.float32)  # [V]
            r = eval_cfg.r  # float
            fov = eval_cfg.fov  # float
        else:
            # 训练时均匀随机采样
            train_cfg = self.cfg.train
            yaw_min, yaw_max = train_cfg.yaw_range
            pitch_min, pitch_max = train_cfg.pitch_range
            yaws_deg = torch.rand(num_views) * (yaw_max - yaw_min) + yaw_min  # [V]
            pitch_deg = torch.rand(num_views) * (pitch_max - pitch_min) + pitch_min  # [V]
            r = float(np.random.uniform(*train_cfg.r_range))  # float
            fov = float(np.random.uniform(*train_cfg.fov_range))  # float

        return yaws_deg, pitch_deg, r, fov

    def _build_camera(
        self,
        fovy: float,
        yaw_deg: float,
        pitch_deg: float,
        distance: float,
        device: torch.device,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:

        # 与 TRELLIS render_utils.py 中 yaw_pitch_r_fov_to_extrinsics_intrinsics 对齐:
        # yaw, pitch in radians
        # orig = [sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch)] * r
        # extr = look_at(orig, 0, up=[0,0,1])  # Z-up 坐标系
        #
        # yaw=0 时相机在 +Y 轴方向
        yaw = torch.deg2rad(torch.tensor(yaw_deg, dtype=torch.float32, device=device))  # []
        pitch = torch.deg2rad(torch.tensor(pitch_deg, dtype=torch.float32, device=device))  # []
        fovy_rad = torch.deg2rad(torch.tensor(fovy, dtype=torch.float32, device=device))  # []
        
        # Calculate camera position (origin)
        # 与 TRELLIS render_utils.py 一致: [sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]
        orig = torch.stack([
            torch.sin(yaw) * torch.cos(pitch),  # []
            torch.cos(yaw) * torch.cos(pitch),  # []
            torch.sin(pitch),  # []
        ]).to(device) * distance  # [3]

        # Extrinsics (World-to-Camera)
        # 与 TRELLIS render_utils.py 一致: up = [0, 0, 1] (Z-up)
        w2c = utils3d.torch.extrinsics_look_at(
            orig,  # [3]
            torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),  # [3]
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),  # [3]
        )  # [4,4]
        
        c2w = torch.inverse(w2c)  # [4,4]
        camera_position = c2w[:3, 3]  # [3]

        # Intrinsics (Normalized)
        # Trellis uses square fov for intrinsics calculation in render_utils
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fovy_rad, fovy_rad)  # [3,3]

        # Projection Matrix
        # Use local intrinsics_to_projection; near/far 放宽以与 blender_script 一致
        proj = intrinsics_to_projection(
            intrinsics,  # [3,3]
            near=1.0,
            far=100.0,
        )  # [4,4]
        
        mvp = build_mvp_from_w2c(w2c.unsqueeze(0), proj.unsqueeze(0)).squeeze(0)  # [4,4]

        return {
            "c2w_matrix": c2w,  # [4,4]
            "w2c_matrix": w2c,  # [4,4]
            "intrinsics": intrinsics,  # [3,3]
            "camera_positions": camera_position,  # [3]
            "mvp_matrix": mvp,  # [4,4]
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_paths[index]  # str
        pil_image = Image.open(image_path).convert("RGB")  # PIL

        is_eval = self.split in ("test", "val")
        num_views = self.cfg.eval.n_view if is_eval else self.cfg.train.n_view  # int
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device

        # 统一采样相机参数
        yaws_deg, pitch_deg, r, fov = self.compute_views_uniform(num_views)  # [V], [V], float, float

        cameras = [
            self._build_camera(
                fovy=fov,
                yaw_deg=float(yaws_deg[i].item()),
                pitch_deg=float(pitch_deg[i].item()),
                distance=r,
                device=device,
                width=self.cfg.width,
                height=self.cfg.height,
            )
            for i in range(num_views)
        ]  # list[len=V]

        mesh_c2w = torch.stack([cam["c2w_matrix"] for cam in cameras])  # [V,4,4]
        mesh_w2c = torch.stack([cam["w2c_matrix"] for cam in cameras])  # [V,4,4]
        mesh_intrinsics = torch.stack([cam["intrinsics"] for cam in cameras])  # [V,3,3]
        mesh_mvp_mtx = torch.stack([cam["mvp_matrix"] for cam in cameras])  # [V,4,4]
        mesh_camera_positions = torch.stack([cam["camera_positions"] for cam in cameras]).unsqueeze(1)  # [V,1,3]

        return {
            "index": index,
            "name": os.path.basename(image_path),
            "image_path": image_path,
            "pixel_values": pil_image,
            "num_views": num_views,
            "mesh_c2w": mesh_c2w,
            "mesh_w2c": mesh_w2c,
            "mesh_intrinsics": mesh_intrinsics,
            "mesh_mvp_mtx": mesh_mvp_mtx,
            "mesh_camera_positions": mesh_camera_positions,
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def collate(batch) -> Dict[str, Any]:
        if not batch:
            return {}
        pixel_values = [item["pixel_values"] for item in batch]  # list[len=B]
        batch_no_img = [{k: v for k, v in item.items() if k != "pixel_values"} for item in batch]  # list[len=B]
        collated = torch.utils.data.default_collate(batch_no_img)  # dict
        collated["pixel_values"] = pixel_values  # list[len=B]
        return collated


# === 轻量配置与 DataModule（仅 train / eval） ===
# 使用 TRELLIS 原生命名: yaw, pitch, r, fov
@dataclass
class TrellisCameraTrainConfig:
    """训练时相机参数配置"""
    n_view: int = 4                       # 训练时视角数
    yaw_range: List[float] = None         # yaw 采样范围 (度)
    pitch_range: List[float] = None       # pitch 采样范围 (度)
    r_range: List[float] = None           # 相机距离范围
    fov_range: List[float] = None         # 视场角范围 (度)

    def __post_init__(self):
        if self.yaw_range is None:
            self.yaw_range = [0.0, 360.0]
        if self.pitch_range is None:
            self.pitch_range = [-15.0, 45.0]
        if self.r_range is None:
            self.r_range = [2.0, 2.0]
        if self.fov_range is None:
            self.fov_range = [40.0, 40.0]


@dataclass
class TrellisCameraEvalConfig:
    """评估时相机参数配置"""
    n_view: int = 4                       # 评估时视角数
    yaw: float = 0.0                      # 评估时固定 yaw (度)
    pitch: float = 15.0                   # 评估时固定 pitch (度)
    r: float = 2.0                        # 评估时相机距离
    fov: float = 40.0                     # 评估时视场角 (度)


@dataclass
class TrellisDataConfig:
    batch_size: int = 1
    eval_batch_size: int = 1
    width: int = 512
    height: int = 512
    ray_height: int = 256
    ray_width: int = 256
    image_dataset_dir: str = "test_images"
    image_file_extension: str = "png"
    eval_image_path: Optional[str] = None
    # 分离的相机配置
    train: TrellisCameraTrainConfig = None
    eval: TrellisCameraEvalConfig = None

    def __post_init__(self):
        if self.train is None:
            self.train = TrellisCameraTrainConfig()
        if self.eval is None:
            self.eval = TrellisCameraEvalConfig()


class TrellisDataModule:
    def __init__(self, cfg: TrellisDataConfig, num_replicas: int = 1, rank: int = 0):
        self.cfg = cfg
        self.num_replicas = num_replicas
        self.rank = rank
        self.train_dataset: Optional[BaseImageDatasetTrellis] = None
        self.eval_dataset: Optional[BaseImageDatasetTrellis] = None
        self.train_sampler: Optional[DistributedSampler] = None
        self.eval_sampler: Optional[DistributedSampler] = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = BaseImageDatasetTrellis(self.cfg, "train")
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
        if stage in (None, "eval", "test", "predict"):
            # 统一使用 test split 作为评估
            self.eval_dataset = BaseImageDatasetTrellis(self.cfg, "test")
            self.eval_sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.train_dataset.collate if self.train_dataset else None,
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.eval_batch_size,
            sampler=self.eval_sampler,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.eval_dataset.collate if self.eval_dataset else None,
        )
