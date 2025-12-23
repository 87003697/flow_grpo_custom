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


def _radical_inverse(base: int, n: int) -> float:
    """Halton 基元。"""
    val = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def _halton_sequence(dim: int, n: int) -> List[float]:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    return [_radical_inverse(primes[d], n) for d in range(dim)]


def _hammersley_sequence(dim: int, n: int, num_samples: int) -> List[float]:
    return [n / num_samples] + _halton_sequence(dim - 1, n)


def sphere_hammersley_sequence(
    n: int,
    num_samples: int,
    offset: Tuple[float, float] = (0.0, 0.0),
    remap: bool = False,
) -> Tuple[float, float]:
    """
    球面 Hammersley 采样，返回 (yaw=phi, pitch=theta)（弧度）。

    Args:
        n: 当前采样点索引
        num_samples: 总采样点数
        offset: 随机偏移 (u_offset, v_offset)
        remap: 若为 True，使用与 TRELLIS 训练数据生成一致的分布（赤道密集）；
               若为 False，使用与 TRELLIS 推理渲染一致的均匀分布。
    """
    u, v = _hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        # 与 TRELLIS dataset_toolkits/utils.py 一致，使采样点更集中于赤道
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2  # [-pi/2, pi/2]
    phi = v * 2 * np.pi  # [0, 2pi]
    return float(phi), float(theta)


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

    def compute_views_hammersley(self, num_views: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 Hammersley 采样生成 yaw/pitch（度），对齐参考 TRELLIS。
        训练时加入随机 offset 并使用 remap 以复现参考数据生成脚本的随机相机分布；
        推理时不使用 remap，与 TRELLIS 官方推理渲染一致。
        """
        is_train = self.split == "train"
        offset = (float(np.random.rand()), float(np.random.rand())) if is_train else (0.0, 0.0)  # tuple()
        # 训练时 remap=True（与训练数据生成一致），推理时 remap=False（与官方推理一致）
        cams = [
            sphere_hammersley_sequence(i, num_views, offset=offset, remap=is_train)
            for i in range(num_views)
        ]  # list[len=V] of (phi, theta)
        yaws_rad = torch.tensor([c[0] for c in cams], dtype=torch.float32)  # [V]
        pitch_rad = torch.tensor([c[1] for c in cams], dtype=torch.float32)  # [V]
        yaws_deg = torch.rad2deg(yaws_rad)  # [V]
        pitch_deg = torch.rad2deg(pitch_rad)  # [V]
        return yaws_deg, pitch_deg

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

        num_views = self.cfg.n_val_views if self.split in ("test", "val") else self.cfg.n_view  # []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
        yaws_deg, pitch_deg = self.compute_views_hammersley(num_views)  # [V], [V]
        camera_distance = self.cfg.eval_camera_distance if self.split in ("test", "val") else float(sum(self.cfg.camera_distance_range) / 2.0)  # []
        fovy = self.cfg.eval_fovy_deg if self.split in ("test", "val") else float(sum(self.cfg.fovy_range) / 2.0)  # []

        cameras = [
            self._build_camera(
                fovy=fovy,
                yaw_deg=float(yaws_deg[i].item()),
                pitch_deg=float(pitch_deg[i].item()),
                distance=camera_distance,
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
@dataclass
class TrellisDataConfig:
    batch_size: int = 1
    eval_batch_size: int = 1
    n_view: int = 4
    width: int = 512
    height: int = 512
    ray_height: int = 256
    ray_width: int = 256
    image_dataset_dir: str = "test_images"
    image_file_extension: str = "png"
    eval_image_path: Optional[str] = None
    elevation_range: List[float] = None
    frontal_azimuth_range: List[float] = None
    camera_distance_range: List[float] = None
    fovy_range: List[float] = None
    eval_camera_distance: float = 2.0
    eval_fovy_deg: float = 40.0
    eval_elevation_deg: float = 0.0
    n_val_views: int = 4

    def __post_init__(self):
        # 默认范围填充
        if self.elevation_range is None:
            self.elevation_range = [0.0, 30.0]
        if self.frontal_azimuth_range is None:
            self.frontal_azimuth_range = [-15.0, 15.0]
        if self.camera_distance_range is None:
            self.camera_distance_range = [2.0, 2.0]
        if self.fovy_range is None:
            self.fovy_range = [40.0, 40.0]


class TrellisDataModule:
    def __init__(self, cfg: TrellisDataConfig):
        self.cfg = cfg
        self.train_dataset: Optional[BaseImageDatasetTrellis] = None
        self.eval_dataset: Optional[BaseImageDatasetTrellis] = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = BaseImageDatasetTrellis(self.cfg, "train")
        if stage in (None, "eval", "test", "predict"):
            # 统一使用 test split 作为评估
            self.eval_dataset = BaseImageDatasetTrellis(self.cfg, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.train_dataset.collate if self.train_dataset else None,
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.eval_dataset.collate if self.eval_dataset else None,
        )
