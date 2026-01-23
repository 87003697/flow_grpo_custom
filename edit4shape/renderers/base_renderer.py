"""
渲染器基类 - 定义统一的渲染流水线接口

渲染流水线 7 阶段:
    Stage 1: Input Preparation (输入准备)
    Stage 2: Camera Transform (相机变换)
    Stage 3: Geometry Processing (几何处理)
    Stage 4: Rasterization Core (光栅化/渲染核心)
    Stage 5: Attribute Interpolation (属性插值/采样)
    Stage 6: Post-processing (后处理)
    Stage 7: Output Assembly (输出组装)

使用示例:
    class MyRenderer(BaseRenderer):
        def _rasterize_core(self, processed_geometry, camera_data):
            # 实现光栅化核心逻辑
            ...
    
    renderer = MyRenderer(RenderConfig(resolution=512))
    output = renderer.render(geometry, extrinsics, intrinsics)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class RenderConfig:
    """
    渲染器配置
    
    Attributes:
        resolution: 输出图像分辨率 (像素)
        near: 近裁剪面距离
        far: 远裁剪面距离
        ssaa: 超采样抗锯齿倍数 (1=无超采样)
        bg_color: 背景颜色，支持:
            - float: 单一灰度值
            - Tuple[float, float, float]: RGB 值
            - 'random': 随机黑白背景
            - 'envmap': 从环境贴图采样
    """
    resolution: int = 512
    near: float = 0.1
    far: float = 10.0
    ssaa: int = 1
    bg_color: Union[float, Tuple[float, float, float], str] = 0.0


@dataclass
class CameraData:
    """
    相机数据容器
    
    Attributes:
        extrinsics: (4, 4) W2C 相机外参矩阵 (OpenCV 约定)
        intrinsics: (3, 3) 相机内参矩阵 (归一化到 [0, 1])
        projection: (4, 4) OpenGL 投影矩阵
        mvp: (4, 4) Model-View-Projection 矩阵
        camera_center: (3,) 相机世界坐标
        fov: Tuple[float, float] (fov_x, fov_y) 弧度
        rays_o: Optional[(H, W, 3)] 射线原点
        rays_d: Optional[(H, W, 3)] 射线方向
    """
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    projection: torch.Tensor
    mvp: torch.Tensor
    camera_center: torch.Tensor
    fov: Tuple[float, float]
    rays_o: Optional[torch.Tensor] = None
    rays_d: Optional[torch.Tensor] = None


@dataclass
class RasterOutput:
    """
    光栅化输出容器
    
    Attributes:
        rast: 光栅化结果 (取决于具体实现)
        depth_buffer: (H, W) 深度缓冲
        primitive_id: (H, W) 图元 ID
        barycentric: Optional[(H, W, 3)] 重心坐标
    """
    rast: Any
    depth_buffer: torch.Tensor
    primitive_id: torch.Tensor
    barycentric: Optional[torch.Tensor] = None


@dataclass
class RenderOutput:
    """
    渲染输出容器
    
    维度规范:
        - 2D 属性 (标量): (H, W) - depth, alpha, mask
        - 3D 属性 (向量): (H, W, 3) - normal, color, shaded
    
    Attributes:
        depth: (H, W) 深度图
        alpha: (H, W) 透明度/覆盖度
        mask: (H, W) 二值掩码
        normal: Optional[(H, W, 3)] 法线图 (可视化格式 [0, 1])
        color: Optional[(H, W, 3)] 颜色图
        shaded: Optional[(H, W, 3)] PBR 着色结果
        extras: Dict[str, Tensor] 其他自定义属性
    """
    depth: torch.Tensor
    alpha: torch.Tensor
    mask: torch.Tensor
    normal: Optional[torch.Tensor] = None
    color: Optional[torch.Tensor] = None
    shaded: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def to_edict(self) -> edict:
        """转换为 edict 格式 (兼容旧接口)"""
        result = edict(
            depth=self.depth,
            alpha=self.alpha,
            mask=self.mask,
        )
        if self.normal is not None:
            result.normal = self.normal
        if self.color is not None:
            result.color = self.color
        if self.shaded is not None:
            result.shaded = self.shaded
        result.update(self.extras)
        return result


# ============================================================================
# 工具函数
# ============================================================================

def intrinsics_to_projection(
    intrinsics: torch.Tensor,  # (3, 3)
    near: float,
    far: float,
) -> torch.Tensor:
    """
    OpenCV 内参矩阵 → OpenGL 投影矩阵
    
    Args:
        intrinsics: (3, 3) 归一化相机内参 (fx, fy, cx, cy 在 [0, 1])
        near: 近裁剪面
        far: 远裁剪面
    
    Returns:
        projection: (4, 4) OpenGL 投影矩阵
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # 标量
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # 标量
    
    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)  # (4, 4)
    proj[0, 0] = 2 * fx
    proj[1, 1] = 2 * fy
    proj[0, 2] = 2 * cx - 1
    proj[1, 2] = -2 * cy + 1
    proj[2, 2] = far / (far - near)
    proj[2, 3] = near * far / (near - far)
    proj[3, 2] = 1.0
    
    return proj  # (4, 4)


def depth_to_normal(
    depth: torch.Tensor,       # (H, W)
    intrinsics: torch.Tensor,  # (3, 3)
    mask: Optional[torch.Tensor] = None,  # (H, W)
) -> torch.Tensor:
    """
    从深度图估算相机空间法线
    
    Args:
        depth: (H, W) 深度图
        intrinsics: (3, 3) 归一化相机内参
        mask: (H, W) 可选掩码
    
    Returns:
        normal: (H, W, 3) 相机空间法线，朝向相机为正
    """
    H, W = depth.shape[-2:]
    depth = depth.reshape(H, W)  # (H, W)
    device = depth.device
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 像素网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H, W), (H, W)
    
    # 反投影到相机空间
    z = depth  # (H, W)
    x = (x_grid / W - cx) * z / fx  # (H, W)
    y = (y_grid / H - cy) * z / fy  # (H, W)
    
    # 中心差分计算梯度
    dx = torch.zeros(H, W, 3, device=device)  # (H, W, 3)
    dy = torch.zeros(H, W, 3, device=device)  # (H, W, 3)
    
    dx[:, 1:-1, 0] = (x[:, 2:] - x[:, :-2]) / 2
    dx[:, 1:-1, 1] = (y[:, 2:] - y[:, :-2]) / 2
    dx[:, 1:-1, 2] = (z[:, 2:] - z[:, :-2]) / 2
    
    dy[1:-1, :, 0] = (x[2:, :] - x[:-2, :]) / 2
    dy[1:-1, :, 1] = (y[2:, :] - y[:-2, :]) / 2
    dy[1:-1, :, 2] = (z[2:, :] - z[:-2, :]) / 2
    
    # 法线 = dy × dx
    normal = torch.linalg.cross(dy, dx)  # (H, W, 3)
    normal = F.normalize(normal, dim=-1).reshape(H, W, 3)  # (H, W, 3)
    
    # 确保法线朝向相机（z < 0）
    normal = torch.where(normal[..., 2:3] > 0, -normal, normal)  # (H, W, 3)
    
    if mask is not None:
        mask = mask.reshape(H, W)  # (H, W)
        normal = normal * mask[..., None]  # (H, W, 3)
    
    return normal  # (H, W, 3)


def camera_normal_to_vis(normal: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    相机空间法线转可视化格式
    
    Args:
        normal: (H, W, 3) 相机空间法线 (朝向相机 z < 0)
        mask: (H, W) 可选掩码
    
    Returns:
        normal_vis: (H, W, 3) 可视化格式 [0, 1]
    """
    normal_vis = -normal * 0.5 + 0.5  # (H, W, 3)
    if mask is not None:
        bg = torch.tensor([0.5, 0.5, 1.0], device=normal.device)  # 中性蓝色背景
        normal_vis = normal_vis * mask[..., None] + bg * (1 - mask[..., None])  # (H, W, 3)
    return normal_vis  # (H, W, 3)


def ssaa_downsample(
    img: torch.Tensor,  # (C, H*ssaa, W*ssaa) 或 (H*ssaa, W*ssaa, C)
    target_size: int,
    channel_first: bool = True,
) -> torch.Tensor:
    """
    SSAA 下采样
    
    Args:
        img: 高分辨率图像
        target_size: 目标分辨率
        channel_first: 是否 channel-first 格式
    
    Returns:
        下采样后的图像
    """
    if not channel_first:
        img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    img = F.interpolate(
        img[None],  # (1, C, H, W)
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False,
        antialias=True
    ).squeeze(0)  # (C, H, W)
    
    if not channel_first:
        img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    return img


# ============================================================================
# 渲染器基类
# ============================================================================

class BaseRenderer(ABC):
    """
    渲染器抽象基类
    
    定义了统一的 7 阶段渲染流水线:
        1. prepare_inputs: 输入准备
        2. compute_camera_data: 相机变换
        3. process_geometry: 几何处理
        4. rasterize_core: 光栅化核心 (抽象方法)
        5. interpolate_attributes: 属性插值
        6. post_process: 后处理
        7. assemble_output: 输出组装
    
    子类必须实现:
        - _rasterize_core: 核心光栅化逻辑
        - _interpolate_attributes: 属性插值逻辑
    
    子类可选重写:
        - _process_geometry: 自定义几何处理
        - _post_process: 自定义后处理
    """
    
    def __init__(self, config: RenderConfig, device: str = 'cuda'):
        """
        Args:
            config: 渲染配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        self._bg_color: Optional[torch.Tensor] = None
    
    # ========== Stage 1: Input Preparation ==========
    
    def prepare_inputs(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """
        Stage 1: 输入准备
        
        - 验证输入
        - 初始化背景颜色
        - 处理空输入
        
        Args:
            geometry: 几何表示 (Mesh/Gaussian/Voxel)
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
        
        Returns:
            处理后的 (geometry, extrinsics, intrinsics)
        """
        # 初始化背景色
        self._init_bg_color(extrinsics.device)
        
        return geometry, extrinsics, intrinsics
    
    def _init_bg_color(self, device: torch.device) -> None:
        """初始化背景颜色"""
        bg = self.config.bg_color
        
        if isinstance(bg, str):
            if bg == 'random':
                import numpy as np
                val = 1.0 if np.random.rand() < 0.5 else 0.0
                self._bg_color = torch.full((3,), val, device=device)  # (3,)
            else:
                self._bg_color = torch.zeros(3, device=device)  # (3,)
        elif isinstance(bg, (int, float)):
            self._bg_color = torch.full((3,), float(bg), device=device)  # (3,)
        else:
            self._bg_color = torch.tensor(bg, device=device, dtype=torch.float32)  # (3,)
    
    def _is_empty_geometry(self, geometry: Any) -> bool:
        """检查几何体是否为空 (子类可重写)"""
        return False
    
    def _get_empty_output(self) -> RenderOutput:
        """返回空输出"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),  # (H, W)
            alpha=torch.zeros(res, res, device=device),  # (H, W)
            mask=torch.zeros(res, res, device=device),   # (H, W)
        )
    
    # ========== Stage 2: Camera Transform ==========
    
    def compute_camera_data(
        self,
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
    ) -> CameraData:
        """
        Stage 2: 相机变换
        
        计算投影矩阵、MVP、相机中心等
        
        Args:
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
        
        Returns:
            CameraData: 相机数据容器
        """
        near, far = self.config.near, self.config.far
        
        # 投影矩阵
        projection = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        
        # MVP
        mvp = projection @ extrinsics  # (4, 4)
        
        # 相机中心 (世界坐标)
        camera_center = torch.inverse(extrinsics)[:3, 3]  # (3,)
        
        # FoV
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        fov_x = 2 * torch.atan(0.5 / fx).item()
        fov_y = 2 * torch.atan(0.5 / fy).item()
        
        return CameraData(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            projection=projection,
            mvp=mvp,
            camera_center=camera_center,
            fov=(fov_x, fov_y),
        )
    
    # ========== Stage 3: Geometry Processing ==========
    
    def process_geometry(
        self,
        geometry: Any,
        camera_data: CameraData,
    ) -> Any:
        """
        Stage 3: 几何处理
        
        默认实现直接返回原始几何体，子类可重写
        
        Args:
            geometry: 原始几何表示
            camera_data: 相机数据
        
        Returns:
            处理后的几何数据
        """
        return self._process_geometry(geometry, camera_data)
    
    def _process_geometry(self, geometry: Any, camera_data: CameraData) -> Any:
        """几何处理实现 (子类可重写)"""
        return geometry
    
    # ========== Stage 4: Rasterization Core ==========
    
    def rasterize(
        self,
        processed_geometry: Any,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        Stage 4: 光栅化
        
        Args:
            processed_geometry: 处理后的几何数据
            camera_data: 相机数据
        
        Returns:
            RasterOutput: 光栅化结果
        """
        return self._rasterize_core(processed_geometry, camera_data)
    
    @abstractmethod
    def _rasterize_core(
        self,
        processed_geometry: Any,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        光栅化核心实现 (子类必须实现)
        
        不同渲染器使用不同的光栅化技术:
        - Mesh: nvdiffrast
        - Gaussian: diff_gaussian_rasterization
        - Voxel: o_voxel 或 soft z-buffer
        """
        pass
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 5: 属性插值
        
        Args:
            raster_output: 光栅化结果
            geometry: 原始几何体
            camera_data: 相机数据
            return_types: 需要返回的属性列表
        
        Returns:
            Dict[str, Tensor]: 插值后的属性
        """
        return self._interpolate_attributes(raster_output, geometry, camera_data, return_types)
    
    @abstractmethod
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """属性插值实现 (子类必须实现)"""
        pass
    
    # ========== Stage 6: Post-processing ==========
    
    def post_process(
        self,
        raw_attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 6: 后处理
        
        - SSAA 下采样
        - 法线空间变换
        - Tone mapping
        - Gamma 校正
        
        Args:
            raw_attrs: 原始属性
            camera_data: 相机数据
        
        Returns:
            处理后的属性
        """
        result = {}
        res = self.config.resolution
        ssaa = self.config.ssaa
        
        for key, value in raw_attrs.items():
            # SSAA 下采样
            if ssaa > 1 and value.numel() > 0:
                if value.dim() == 2:  # (H, W)
                    value = ssaa_downsample(value[None], res, channel_first=True).squeeze(0)  # (H, W)
                elif value.dim() == 3:  # (H, W, C)
                    value = ssaa_downsample(value, res, channel_first=False)  # (H, W, C)
            
            result[key] = value
        
        # 子类可扩展后处理
        result = self._post_process(result, camera_data)
        
        return result
    
    def _post_process(
        self,
        attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """后处理扩展点 (子类可重写)"""
        return attrs
    
    # ========== Stage 7: Output Assembly ==========
    
    def assemble_output(
        self,
        attrs: Dict[str, torch.Tensor],
    ) -> RenderOutput:
        """
        Stage 7: 输出组装
        
        Args:
            attrs: 所有属性
        
        Returns:
            RenderOutput: 统一输出格式
        """
        # 提取标准属性
        depth = attrs.pop('depth', torch.zeros(self.config.resolution, self.config.resolution, device=self.device))
        alpha = attrs.pop('alpha', torch.zeros(self.config.resolution, self.config.resolution, device=self.device))
        mask = attrs.pop('mask', (alpha > 0.5).float())
        normal = attrs.pop('normal', None)
        color = attrs.pop('color', None)
        shaded = attrs.pop('shaded', None)
        
        return RenderOutput(
            depth=depth,
            alpha=alpha,
            mask=mask,
            normal=normal,
            color=color,
            shaded=shaded,
            extras=attrs,  # 剩余属性放入 extras
        )
    
    # ========== 主渲染入口 ==========
    
    def render(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        主渲染入口
        
        执行完整的 7 阶段渲染流水线
        
        Args:
            geometry: 几何表示 (Mesh/Gaussian/Voxel)
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
            return_types: 需要返回的属性列表
        
        Returns:
            RenderOutput: 渲染结果
        """
        if return_types is None:
            return_types = ['depth', 'alpha', 'mask', 'normal']
        
        # Stage 1: Input Preparation
        geometry, extrinsics, intrinsics = self.prepare_inputs(geometry, extrinsics, intrinsics)
        
        # 空输入检查
        if self._is_empty_geometry(geometry):
            return self._get_empty_output()
        
        # Stage 2: Camera Transform
        camera_data = self.compute_camera_data(extrinsics, intrinsics)
        
        # Stage 3: Geometry Processing
        processed_geom = self.process_geometry(geometry, camera_data)
        
        # Stage 4: Rasterization Core
        raster_output = self.rasterize(processed_geom, camera_data)
        
        # Stage 5: Attribute Interpolation
        raw_attrs = self.interpolate_attributes(raster_output, processed_geom, camera_data, return_types)
        
        # Stage 6: Post-processing
        processed_attrs = self.post_process(raw_attrs, camera_data)
        
        # Stage 7: Output Assembly
        return self.assemble_output(processed_attrs)
    
    # ========== 兼容接口 ==========
    
    def render_edict(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types: List[str] = None,
    ) -> edict:
        """
        兼容旧接口，返回 edict
        """
        output = self.render(geometry, extrinsics, intrinsics, return_types)
        return output.to_edict()


# ============================================================================
# PBR 后处理 Mixin
# ============================================================================

class PBRPostProcessMixin:
    """
    PBR 后处理混入类
    
    提供 tone mapping 和 gamma 校正功能
    """
    
    @staticmethod
    def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
        """ACES tone mapping"""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    
    @staticmethod
    def gamma_correction(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """Gamma correction"""
        return torch.clamp(x ** (1.0 / gamma), 0.0, 1.0)
    
    def apply_pbr_post_process(
        self,
        shaded: torch.Tensor,  # (H, W, 3)
        mask: torch.Tensor = None,  # (H, W)
        bg_color: torch.Tensor = None,  # (3,)
    ) -> torch.Tensor:
        """
        应用 PBR 后处理
        
        Args:
            shaded: (H, W, 3) 线性空间着色结果
            mask: (H, W) 可选掩码
            bg_color: (3,) 可选背景色
        
        Returns:
            (H, W, 3) 后处理结果
        """
        # Tone mapping
        shaded = self.aces_tonemapping(shaded)
        
        # Gamma correction
        shaded = self.gamma_correction(shaded)
        
        # 背景合成
        if mask is not None and bg_color is not None:
            shaded = shaded * mask[..., None] + bg_color * (1 - mask[..., None])
        
        return shaded


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 数据结构
    'RenderConfig',
    'CameraData',
    'RasterOutput',
    'RenderOutput',
    # 工具函数
    'intrinsics_to_projection',
    'depth_to_normal',
    'camera_normal_to_vis',
    'ssaa_downsample',
    # 基类
    'BaseRenderer',
    'PBRPostProcessMixin',
]
