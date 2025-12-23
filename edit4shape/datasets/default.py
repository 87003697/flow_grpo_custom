"""
简化的多视角图像数据模块
用于Gen2Turbo框架的图像到3D生成
"""

import os
import math
import glob
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.ops import get_ray_directions, get_rays
from threestudio.utils.config import parse_structured

from PIL import Image
import kiui
import kiui.cam
import torchvision.transforms as T
from .utils import (
    build_perspective_matrix,
    build_w2c_from_c2w,
    build_mvp_from_w2c,
)


def apply_camera_pose_alignment(camera_pose: torch.Tensor, alignment: str) -> torch.Tensor:
    return camera_pose


def build_trellis_intrinsics(fovy_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raise NotImplementedError("Trellis intrinsics moved to perview_vd_trellis; import from there")

@dataclass
class MultiviewImageDualRenderMultiStepDataModuleConfig:
    # 基本配置
    batch_size: int = 1
    eval_batch_size: int = 1
    n_view: int = 4
    width: int = 512
    height: int = 512
    ray_height: int = 256
    ray_width: int = 256
    
    # 图像数据配置
    image_dataset_dir: str = "test_images"
    eval_image_path: Optional[str] = None
    specific_image_path: Optional[str] = None
    image_file_extension: str = "png"
    
    # 相机配置
    elevation_range: List[float] = field(default_factory=lambda: [0, 30])
    frontal_azimuth_range: List[float] = field(default_factory=lambda: [-15, 15])
    camera_distance_range: List[float] = field(default_factory=lambda: [1.5, 2.5])
    fovy_range: List[float] = field(default_factory=lambda: [40, 60])
    mesh_alignment: str = "default"

    # 评估配置
    eval_camera_distance: float = 2.0
    eval_fovy_deg: float = 50.0
    eval_elevation_deg: float = 0.0
    n_val_views: int = 4
    n_test_views: int = 8

class BaseImageDataset(Dataset, Updateable):
    """简化的图像数据集类（仅输出图像+相机几何）"""
    
    def __init__(self, cfg: MultiviewImageDualRenderMultiStepDataModuleConfig, split: str):
        self.cfg = cfg
        self.split = split
        
        # 相机配置
        self.elevation_range = cfg.elevation_range
        self.frontal_azimuth_range = cfg.frontal_azimuth_range
        self.camera_distance_range = cfg.camera_distance_range
        self.fovy_range = cfg.fovy_range
        
        # 加载图像路径
        self.image_paths = self._load_image_paths()
        threestudio.info(f"Loaded {len(self.image_paths)} images for {split} split")
        
        # 图像预处理
        self.transform = T.Compose([
            T.Resize((cfg.height, cfg.width)),
            # T.ToTensor(),
        ])

    def compute_azimuths(self, num_views: int) -> torch.Tensor:
        """统一的方位角生成（度）。返回 [V]
        - val/test：等间隔 [0,360)
        - train：等间隔 + 前视范围内随机整体偏移
        """
        if self.split in ["test", "val"]:
            idx = torch.arange(num_views, dtype=torch.float32)  # [V]
            return (360.0 / num_views) * idx  # [V]
        base_low = float(self.frontal_azimuth_range[0])  # 标量
        base_high = float(self.frontal_azimuth_range[1])  # 标量
        base = torch.empty((), dtype=torch.float32).uniform_(base_low, base_high)  # []
        idx = torch.arange(num_views, dtype=torch.float32)  # [V]
        return ((360.0 / num_views) * idx + base) % 360.0  # [V]
    
    def _load_image_paths(self) -> List[str]:
        """简化的图像路径加载"""
        image_paths = []
        
        # 检查具体的图像路径配置
        if self.split == "test" and self.cfg.eval_image_path:
            if os.path.exists(self.cfg.eval_image_path):
                if os.path.isfile(self.cfg.eval_image_path):
                    image_paths = [self.cfg.eval_image_path]
                else:
                    # 目录，获取所有图像
                    extensions = list(set([self.cfg.image_file_extension, "png", "jpg", "jpeg"]))
                    for ext in extensions:
                        image_paths.extend(glob.glob(os.path.join(self.cfg.eval_image_path, f"*.{ext}")))
        
        # 如果没有找到图像，使用默认目录
        if not image_paths:
            image_dir = self.cfg.image_dataset_dir
            if os.path.exists(image_dir):
                extensions = list(set([self.cfg.image_file_extension, "png", "jpg", "jpeg"]))
                for ext in extensions:
                    image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        
        # 如果还是没有图像，直接报错，避免静默回退
        if not image_paths:
            raise FileNotFoundError(f"No images found under eval_image_path or image_dataset_dir: {self.cfg.image_dataset_dir}")
        
        return sorted(list(set(image_paths)))  # 限制最多20张图片用于测试

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_paths[index]
        
        # 加载和处理图像
        pil_image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(pil_image)
        
        # === Multi-view data generation ===
        if self.split in ["test", "val"]:
            # 多视角相机数据生成
            num_views = self.cfg.n_test_views if self.split == "test" else self.cfg.n_val_views
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 生成两套不同分辨率的相机数据
            sdf_cameras, mesh_cameras = self.generate_dual_test_cameras(num_views=num_views, device=device)
            
            # 转换为批量张量 - SDF渲染器数据 (低分辨率)
            sdf_c2w_matrices = torch.stack([cam['c2w_matrix'] for cam in sdf_cameras])
            sdf_w2c_matrices = torch.stack([cam['w2c_matrix'] for cam in sdf_cameras])
            sdf_rays_o = torch.stack([cam['rays_o'] for cam in sdf_cameras])
            sdf_rays_d = torch.stack([cam['rays_d'] for cam in sdf_cameras])
            
            # 转换为批量张量 - Mesh渲染器数据 (高分辨率)
            mesh_mvp_matrices = torch.stack([cam['mvp_matrix'] for cam in mesh_cameras])
            mesh_c2w_matrices = torch.stack([cam['c2w_matrix'] for cam in mesh_cameras])
            mesh_rays_d_rasterize = torch.stack([cam['rays_d'] for cam in mesh_cameras])
            azimuths = torch.tensor([cam['azimuth_deg'] for cam in mesh_cameras], dtype=torch.float32)  # [V]
            
            # 共享数据
            camera_positions = torch.stack([cam['camera_positions'] for cam in mesh_cameras]).squeeze(1)  # [V,3]
            # 使用相机局部坐标轴放置点光源（OpenGL 约定: c2w 的第0/1/2列分别为 right/up/(-forward)）
            c2w_all = torch.stack([cam['c2w_matrix'] for cam in mesh_cameras])  # [V,4,4]
            cam_right = c2w_all[:, :3, 0]  # [V,3]
            cam_up = c2w_all[:, :3, 1]  # [V,3]
            cam_back = c2w_all[:, :3, 2]  # [V,3]  # 第2列为 -forward，即相机背后方向
            light_positions = (
                camera_positions
                + 0.8 * cam_back   # [V,3]  相机“背后”
                + 0.6 * cam_right  # [V,3]  右侧偏移
                + 0.7 * cam_up     # [V,3]  上方偏移
            )  # [V,3]
            
            result = {
                # === 基本信息 ===
                "index": index,
                "name": os.path.basename(image_path),
                "image_path": image_path,
                "pixel_values": pixel_values,
                "num_views": num_views,
                
                # === SDF渲染器数据 (sdf_前缀) ===
                "sdf_rays_o": sdf_rays_o,               # [V, 256, 256, 3]
                "sdf_rays_d": sdf_rays_d,               # [V, 256, 256, 3]
                "sdf_c2w": sdf_c2w_matrices,            # [V, 4, 4] 基于256x256
                "sdf_w2c": sdf_w2c_matrices,            # [V, 4, 4] 基于256x256
                
                # === Mesh渲染器数据 (mesh_前缀) ===
                "mesh_mvp_mtx": mesh_mvp_matrices,      # [V, 4, 4] 基于512x512
                "mesh_height": self.cfg.height,         # 512
                "mesh_width": self.cfg.width,           # 512
                "mesh_c2w": mesh_c2w_matrices,          # [V, 4, 4] 基于512x512
                "mesh_rays_d_rasterize": mesh_rays_d_rasterize, # [V, 512, 512, 3]
                
                # === 共享数据 (无前缀) ===
                "camera_positions": camera_positions,   # [V, 3]
                "light_positions": light_positions,     # [V, 3]
                "camera_distances": torch.full((num_views,), self.cfg.eval_camera_distance),  # [V]
                "azimuths": azimuths,                   # [V]
                
                # === 其他数据 ===
                "fovy": mesh_cameras[0]['fovy'],
            }
            return result
        
        # === Multi-view logic for training ===
        else:
            num_views = self.cfg.n_view
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 智能采样 - 一次性生成所有参数
            # azimuth: 统一通过 compute_azimuths 生成
            azimuths = self.compute_azimuths(num_views)  # [V]
            
            # 其他参数：随机一次，所有视角共享
            shared_elevation = np.random.uniform(*self.elevation_range)
            shared_distance = np.random.uniform(*self.camera_distance_range)  
            shared_fovy = np.random.uniform(*self.fovy_range)
            
            # 生成相机数据
            sdf_cameras = []   # SDF渲染器用的低分辨率相机数据
            mesh_cameras = []  # Mesh渲染器用的高分辨率相机数据
            
            for i in range(num_views):
                # 使用统一的相机生成方法
                sdf_camera_data = self._compute_low_res_camera(
                    shared_fovy, float(azimuths[i].item()), shared_elevation, shared_distance, device
                )
                sdf_cameras.append(sdf_camera_data)
                
                mesh_camera_data = self._compute_high_res_camera(
                    shared_fovy, float(azimuths[i].item()), shared_elevation, shared_distance, device
                )
                mesh_cameras.append(mesh_camera_data)
            
            # 转换为批量张量 - SDF渲染器数据 (低分辨率)
            sdf_c2w_matrices = torch.stack([cam['c2w_matrix'] for cam in sdf_cameras])
            sdf_w2c_matrices = torch.stack([cam['w2c_matrix'] for cam in sdf_cameras])
            sdf_rays_o = torch.stack([cam['rays_o'] for cam in sdf_cameras])
            sdf_rays_d = torch.stack([cam['rays_d'] for cam in sdf_cameras])
            
            # 转换为批量张量 - Mesh渲染器数据 (高分辨率)
            mesh_mvp_matrices = torch.stack([cam['mvp_matrix'] for cam in mesh_cameras])
            mesh_c2w_matrices = torch.stack([cam['c2w_matrix'] for cam in mesh_cameras])
            mesh_rays_d_rasterize = torch.stack([cam['rays_d'] for cam in mesh_cameras])
            azimuths = torch.tensor([cam['azimuth_deg'] for cam in mesh_cameras], dtype=torch.float32)  # [V]
            
            # 共享数据 (使用mesh cameras的位置数据，因为位置应该相同)
            camera_positions = torch.stack([cam['camera_positions'] for cam in mesh_cameras])  # [V,3]
            # 训练阶段同样使用相机局部坐标的侧后上方灯位
            c2w_all = torch.stack([cam['c2w_matrix'] for cam in mesh_cameras])  # [V,4,4]
            cam_right = c2w_all[:, :3, 0]  # [V,3]
            cam_up = c2w_all[:, :3, 1]  # [V,3]
            cam_back = c2w_all[:, :3, 2]  # [V,3]
            light_positions = (
                camera_positions
                + 0.8 * cam_back   # [V,3]
                + 0.6 * cam_right  # [V,3]
                + 0.7 * cam_up     # [V,3]
            )  # [V,3]
            
            result = {
                # === 基本信息 ===
                "index": index,
                "name": os.path.basename(image_path),
                "image_path": image_path,
                "pixel_values": pixel_values,
                "num_views": num_views,
                
                # === SDF渲染器数据 (sdf_前缀) ===
                "sdf_rays_o": sdf_rays_o,               # [V, 256, 256, 3]
                "sdf_rays_d": sdf_rays_d,               # [V, 256, 256, 3]
                "sdf_c2w": sdf_c2w_matrices,            # [V, 4, 4] 基于256x256
                "sdf_w2c": sdf_w2c_matrices,            # [V, 4, 4] 基于256x256
                
                # === Mesh渲染器数据 (mesh_前缀) ===
                "mesh_mvp_mtx": mesh_mvp_matrices,      # [V, 4, 4] 基于512x512
                "mesh_height": self.cfg.height,         # 512
                "mesh_width": self.cfg.width,           # 512
                "mesh_c2w": mesh_c2w_matrices,          # [V, 4, 4] 基于512x512
                "mesh_rays_d_rasterize": mesh_rays_d_rasterize, # [V, 512, 512, 3]
                
                # === 共享数据 (无前缀) ===
                "camera_positions": camera_positions,    # [V, 3]
                "light_positions": light_positions,      # [V, 3]
                "camera_distances": torch.full((num_views,), shared_distance),  # [V]
                "azimuths": azimuths,                    # [V]
                
                # === 其他数据 ===
                "fovy": mesh_cameras[0]['fovy'],
            }
            return result

    def _compute_camera(self, fovy: float, azimuth: float, elevation: float, distance: float, 
                       device: torch.device, width: int, height: int, 
                       include_mvp: bool = False) -> Dict[str, torch.Tensor]:
        """统一的相机参数计算方法"""
        # 创建相机
        camera = kiui.cam.OrbitCamera(W=width, H=height, r=distance, fovy=fovy)
        camera.from_angle(elevation=-elevation, azimuth=azimuth, is_degree=True)
        
        # 获取相机矩阵
        camera_pose = torch.from_numpy(camera.pose).float().to(device)  # [4,4]
        camera_pose = apply_camera_pose_alignment(camera_pose, self.cfg.mesh_alignment)  # [4,4]

        aspect = float(width) / float(height)  # 标量
        w2c = build_w2c_from_c2w(camera_pose.unsqueeze(0)).squeeze(0)  # [4,4]
        camera_position = camera_pose[:3, 3]  # [3]
        
        # 生成rays - 统一使用动态生成
        fovy_rad = torch.tensor(fovy * math.pi / 180, device=device, dtype=torch.float32)
        focal_length_y = 0.5 * height / torch.tan(0.5 * fovy_rad)
        
        directions = get_ray_directions(H=height, W=width, focal=1.0).to(device)
        directions[:, :, :2] = directions[:, :, :2] / focal_length_y
        
        rays_o, rays_d = get_rays(directions.unsqueeze(0), camera_pose.unsqueeze(0), keepdim=True)  # rays_o:[1,H,W,3], rays_d:[1,H,W,3]
        rays_o = rays_o.squeeze(0)  # [H,W,3]
        rays_d = rays_d.squeeze(0)  # [H,W,3]
        
        # 构建返回结果
        result = {
            'c2w_matrix': camera_pose,  # [4,4]
            'w2c_matrix': w2c,  # [4,4]
            'camera_positions': camera_position,  # [3]
            'rays_o': rays_o,  # [H,W,3]
            'rays_d': rays_d,  # [H,W,3]
            'fovy': fovy,
            'azimuth_deg': float(azimuth),
        }
        
        if include_mvp:
            proj = build_perspective_matrix(
                fovy_deg=torch.tensor(fovy, dtype=camera_pose.dtype, device=device),  # []
                aspect=aspect,  # 标量
                znear=0.1,
                zfar=100.0,
                dtype=camera_pose.dtype,
                device=device,
                batch_size=1,
            )  # [1,4,4]
            mvp_matrix = build_mvp_from_w2c(w2c=w2c.unsqueeze(0), proj=proj).squeeze(0)  # [4,4]
            result['mvp_matrix'] = mvp_matrix  # [4,4]
            
        return result


    def _compute_low_res_camera(self, fovy: float, azimuth: float, 
                              elevation: float, distance: float, device: torch.device) -> Dict[str, torch.Tensor]:
        """生成SDF渲染器用的低分辨率相机数据 (256x256)"""
        return self._compute_camera(
            fovy, azimuth, elevation, distance, device,
            width=self.cfg.ray_width,
            height=self.cfg.ray_height,
            include_mvp=False
        )

    def _compute_high_res_camera(self, fovy: float, azimuth: float, 
                               elevation: float, distance: float, device: torch.device) -> Dict[str, torch.Tensor]:
        """生成Mesh渲染器用的高分辨率相机数据 (512x512)"""
        return self._compute_camera(
            fovy, azimuth, elevation, distance, device,
            width=self.cfg.width,
            height=self.cfg.height,
            include_mvp=True
        )

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        # 简化版本，暂时不需要分辨率更新
        pass



    def generate_dual_test_cameras(self, num_views: int = None, device: torch.device = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        为测试生成两套不同分辨率的多视角相机参数
        
        Args:
            num_views: 视角数量
            device: 计算设备
            
        Returns:
            (sdf_cameras, mesh_cameras): 两套相机数据列表
        """
        if num_views is None:
            num_views = self.cfg.n_test_views
            
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取相机配置参数
        camera_distance = self.cfg.eval_camera_distance
        elevation_deg = self.cfg.eval_elevation_deg
        fovy_deg = self.cfg.eval_fovy_deg
        
        threestudio.info(f"Data module generating {num_views} dual test cameras: SDF(256x256) + Mesh(512x512)")
        
        sdf_cameras = []   # SDF渲染器用的低分辨率相机
        mesh_cameras = []  # Mesh渲染器用的高分辨率相机
        
        azimuths = self.compute_azimuths(num_views)  # [V]
        for i in range(num_views):
            azimuth = float(azimuths[i].item())  # 标量
            
            # 使用统一的相机生成方法
            sdf_camera_data = self._compute_low_res_camera(
                fovy_deg, azimuth, elevation_deg, camera_distance, device
            )
            # 需要为测试数据调整格式
            sdf_camera_data['camera_positions'] = sdf_camera_data['camera_positions'].unsqueeze(0)
            sdf_cameras.append(sdf_camera_data)
            
            mesh_camera_data = self._compute_high_res_camera(
                fovy_deg, azimuth, elevation_deg, camera_distance, device
            )
            # 需要为测试数据调整格式
            mesh_camera_data['camera_positions'] = mesh_camera_data['camera_positions'].unsqueeze(0)
            mesh_cameras.append(mesh_camera_data)
        
        return sdf_cameras, mesh_cameras

    def generate_eval_cameras(self, num_views: int = None, device: torch.device = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        为验证生成相机参数（复用测试逻辑，因为评估和测试使用相同的固定分布）
        """
        if num_views is None:
            num_views = self.cfg.n_val_views
        
        return self.generate_dual_test_cameras(num_views, device)

    def collate(self, batch) -> Dict[str, Any]:
        """
        处理包含 PIL Image 的 batch。
        """
        if not batch:
            return {}
            
        # 提取 PIL images (pixel_values)
        pixel_values = [item['pixel_values'] for item in batch]
        
        # 创建不含 PIL 的新 batch 列表进行默认 collate
        # 过滤掉 'pixel_values'，让 default_collate 处理剩下的 tensor/numpy/str
        batch_no_img = [{k: v for k, v in item.items() if k != 'pixel_values'} for item in batch]
        
        # 使用默认 collate 处理其他数据
        collated = torch.utils.data.default_collate(batch_no_img)
        
        # 把 PIL list 放回去
        collated['pixel_values'] = pixel_values 
        return collated

@threestudio.register("multiview-image-dualrender-multistep-datamodule") 
class MultiviewImageDualRenderMultiStepDataModule(pl.LightningDataModule):
    """简化的数据模块"""

    def __init__(self, cfg: Union[Dict[str, Any], DictConfig]):
        super().__init__()
        self.cfg = parse_structured(MultiviewImageDualRenderMultiStepDataModuleConfig, cfg)

    def setup(self, stage: Optional[str] = None):
        """简化的 setup：仅创建数据集"""
        if stage in (None, "test", "predict"):
            self.test_dataset = BaseImageDataset(self.cfg, "test")
        if stage in (None, "fit", "validate"):
            self.val_dataset = BaseImageDataset(self.cfg, "val")
            self.train_dataset = BaseImageDataset(self.cfg, "train")

    def train_dataloader(self):
        """简化的训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self):
        """简化的验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self):
        """简化的测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.test_dataset.collate,  # 🔥 改为数据集的 collate
        )

    def predict_dataloader(self):
        """预测阶段的数据加载器：复用测试集配置"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.test_dataset.collate,
        )

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # 将更新步骤委托给数据集
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if hasattr(dataset, 'update_step'):
                dataset.update_step(epoch, global_step, on_load_weights)

    # 🔥 数据模块对外提供的相机生成接口
    def generate_test_cameras(self, num_views: int = None, device: torch.device = None) -> List[Dict[str, Any]]:
        """
        为测试生成多视角相机参数（向后兼容接口，返回高分辨率相机数据）
        
        Args:
            num_views: 视角数量，如果为None则使用配置中的n_test_views
            device: 计算设备
            
        Returns:
            高分辨率相机数据列表
        """
        if num_views is None:
            num_views = self.cfg.n_test_views
            
        # 🔥 使用新的双相机生成方法，但只返回高分辨率相机数据以保持兼容性
        sdf_cameras, mesh_cameras = self.test_dataset.generate_dual_test_cameras(num_views, device)
        return mesh_cameras
    
    def generate_eval_cameras(self, num_views: int = None, device: torch.device = None) -> List[Dict[str, Any]]:
        """
        为验证生成相机参数（向后兼容接口，返回高分辨率相机数据）
        
        Args:
            num_views: 视角数量，如果为None则使用配置中的n_val_views
            device: 计算设备
            
        Returns:
            高分辨率相机数据列表
        """
        if num_views is None:
            num_views = self.cfg.n_val_views
            
        # 🔥 使用新的双相机生成方法，但只返回高分辨率相机数据以保持兼容性
        sdf_cameras, mesh_cameras = self.val_dataset.generate_eval_cameras(num_views, device)
        return mesh_cameras
