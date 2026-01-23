"""
Gaussian Splatting Renderer for Trellis.

重构版：继承 BaseRenderer，遵循 7 阶段渲染流水线。

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

# 从参考代码 TRELLIS 导入 Gaussian 类
from trellis.representations.gaussian import Gaussian
from trellis.renderers.sh_utils import eval_sh

from edit4shape.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    intrinsics_to_projection,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GaussianPipeConfig:
    """
    Gaussian Splatting 管线配置
    
    Attributes:
        kernel_size: 光栅化核大小
        convert_SHs_python: 是否在 Python 中进行 SH -> RGB 转换
        compute_cov3D_python: 是否在 Python 中计算 3D 协方差
        scale_modifier: 缩放修正因子
        debug: 调试模式
    """
    kernel_size: float = 0.1
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    scale_modifier: float = 1.0
    debug: bool = False


@dataclass
class GaussianCameraData(CameraData):
    """
    Gaussian 渲染专用相机数据
    
    扩展 CameraData，增加 3DGS 需要的字段
    """
    image_height: int = 512
    image_width: int = 512
    world_view_transform: torch.Tensor = None  # (4, 4) 转置后的 view 矩阵
    full_proj_transform: torch.Tensor = None   # (4, 4) 转置后的 MVP 矩阵


@dataclass
class GaussianRasterResult:
    """
    Gaussian 光栅化结果
    
    Attributes:
        rendered_image: (3, H, W) 渲染图像
        viewspace_points: (N, 3) 屏幕空间点（用于梯度）
        visibility_filter: (N,) 可见性掩码
        radii: (N,) 屏幕空间半径
    """
    rendered_image: torch.Tensor      # (3, H, W)
    viewspace_points: torch.Tensor    # (N, 3)
    visibility_filter: torch.Tensor   # (N,)
    radii: torch.Tensor               # (N,)


# ============================================================================
# 核心光栅化函数
# ============================================================================

def gaussian_rasterize(
    camera_data: GaussianCameraData,
    gaussian: Gaussian,
    pipe_config: GaussianPipeConfig,
    bg_color: torch.Tensor,
    override_color: torch.Tensor = None,
) -> GaussianRasterResult:
    """
    Gaussian Splatting 光栅化核心函数
    
    Args:
        camera_data: 相机数据
        gaussian: Gaussian 表示
        pipe_config: 管线配置
        bg_color: (3,) 背景颜色
        override_color: (N, 3) 可选的覆盖颜色
    
    Returns:
        GaussianRasterResult: 光栅化结果
    """
    # Lazy import
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # 创建用于获取 2D 梯度的零张量
    screenspace_points = torch.zeros_like(
        gaussian.get_xyz, 
        dtype=gaussian.get_xyz.dtype, 
        requires_grad=True, 
        device="cuda"
    ) + 0  # (N, 3)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # 计算 tan(fov/2)
    tanfovx = math.tan(camera_data.fov[0] * 0.5)
    tanfovy = math.tan(camera_data.fov[1] * 0.5)
    
    # Subpixel offset
    subpixel_offset = torch.zeros(
        (camera_data.image_height, camera_data.image_width, 2),
        dtype=torch.float32,
        device="cuda"
    )  # (H, W, 2)
    
    # 光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=camera_data.image_height,
        image_width=camera_data.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pipe_config.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=pipe_config.scale_modifier,
        viewmatrix=camera_data.world_view_transform,
        projmatrix=camera_data.full_proj_transform,
        sh_degree=gaussian.active_sh_degree,
        campos=camera_data.camera_center,
        prefiltered=False,
        debug=pipe_config.debug,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # 提取 Gaussian 属性
    means3D = gaussian.get_xyz  # (N, 3)
    means2D = screenspace_points  # (N, 3)
    opacity = gaussian.get_opacity  # (N, 1)
    
    # 协方差
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe_config.compute_cov3D_python:
        cov3D_precomp = gaussian.get_covariance(pipe_config.scale_modifier)
    else:
        scales = gaussian.get_scaling
        rotations = gaussian.get_rotation
    
    # 颜色
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe_config.convert_SHs_python:
            shs_view = gaussian.get_features.transpose(1, 2).view(
                -1, 3, (gaussian.max_sh_degree + 1) ** 2
            )  # (N, 3, D^2)
            dir_pp = gaussian.get_xyz - camera_data.camera_center.repeat(
                gaussian.get_features.shape[0], 1
            )  # (N, 3)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # (N, 3)
            sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)  # (N, 3)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
        else:
            shs = gaussian.get_features
    else:
        colors_precomp = override_color  # (N, 3)
    
    # 执行光栅化
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )  # rendered_image: (3, H, W), radii: (N,)
    
    return GaussianRasterResult(
        rendered_image=rendered_image,
        viewspace_points=screenspace_points,
        visibility_filter=(radii > 0),
        radii=radii,
    )


# ============================================================================
# GaussianRenderer
# ============================================================================

class GaussianRenderer(BaseRenderer):
    """
    3D Gaussian Splatting 渲染器
    
    继承 BaseRenderer，实现 7 阶段渲染流水线:
        Stage 1: prepare_inputs - 初始化背景色
        Stage 2: compute_camera_data - 计算 3DGS 专用相机参数
        Stage 3: process_geometry - 直接返回 Gaussian（无需预处理）
        Stage 4: rasterize_core - 调用 diff_gaussian_rasterization
        Stage 5: interpolate_attributes - 提取颜色（3DGS 已完成插值）
        Stage 6: post_process - SSAA 下采样
        Stage 7: assemble_output - 组装 RenderOutput
    
    使用示例:
        config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1)
        renderer = GaussianRenderer(config)
        output = renderer.render(gaussian, extrinsics, intrinsics)
        # output.color: (512, 512, 3)
    """
    
    def __init__(
        self,
        config: RenderConfig = None,
        device: str = "cuda",
        pipe_config: GaussianPipeConfig = None,
    ):
        """
        Args:
            config: 渲染配置
            device: 计算设备
            pipe_config: 管线配置
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1, bg_color='random')
        super().__init__(config, device)
        
        self.pipe_config = pipe_config or GaussianPipeConfig()
        self._colors_overwrite: Optional[torch.Tensor] = None
    
    # ========== Stage 1: Input Preparation ==========
    
    def prepare_inputs(
        self,
        geometry: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """准备输入，重置颜色覆盖"""
        self._colors_overwrite = None
        return super().prepare_inputs(geometry, extrinsics, intrinsics)
    
    def _is_empty_geometry(self, geometry: Gaussian) -> bool:
        """检查 Gaussian 是否为空"""
        return geometry.get_xyz.shape[0] == 0
    
    # ========== Stage 2: Camera Transform ==========
    
    def compute_camera_data(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> GaussianCameraData:
        """
        计算 3DGS 专用相机数据
        
        Args:
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
        
        Returns:
            GaussianCameraData: 扩展的相机数据
        """
        resolution = self.config.resolution
        ssaa = self.config.ssaa
        near, far = self.config.near, self.config.far
        
        # 基础相机数据
        projection = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        mvp = projection @ extrinsics  # (4, 4)
        camera_center = torch.inverse(extrinsics)[:3, 3]  # (3,)
        
        # FoV
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        fov_x = 2 * torch.atan(0.5 / fx).item()
        fov_y = 2 * torch.atan(0.5 / fy).item()
        
        # 3DGS 专用：转置的矩阵
        world_view_transform = extrinsics.T.contiguous()  # (4, 4)
        full_proj_transform = mvp.T.contiguous()  # (4, 4)
        
        return GaussianCameraData(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            projection=projection,
            mvp=mvp,
            camera_center=camera_center,
            fov=(fov_x, fov_y),
            image_height=resolution * ssaa,
            image_width=resolution * ssaa,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(self, geometry: Gaussian, camera_data: CameraData) -> Gaussian:
        """Gaussian 无需预处理"""
        return geometry
    
    # ========== Stage 4: Rasterization Core ==========
    
    def _rasterize_core(
        self,
        processed_geometry: Gaussian,
        camera_data: GaussianCameraData,
    ) -> RasterOutput:
        """
        调用 3DGS 光栅化
        
        Args:
            processed_geometry: Gaussian 表示
            camera_data: 相机数据
        
        Returns:
            RasterOutput: 光栅化结果
        """
        # 执行光栅化
        result = gaussian_rasterize(
            camera_data=camera_data,
            gaussian=processed_geometry,
            pipe_config=self.pipe_config,
            bg_color=self._bg_color,
            override_color=self._colors_overwrite,
        )
        
        # 包装为 RasterOutput
        return RasterOutput(
            rast=result,  # 存储完整结果
            depth_buffer=torch.zeros(camera_data.image_height, camera_data.image_width, device=self.device),  # 3DGS 不输出深度
            primitive_id=torch.zeros(camera_data.image_height, camera_data.image_width, device=self.device, dtype=torch.long),
        )
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Gaussian,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        提取渲染结果
        
        3DGS 的光栅化已经完成了属性插值，这里只需要提取
        """
        result: GaussianRasterResult = raster_output.rast
        
        # 颜色图像: (3, H, W) -> (H, W, 3)
        color = result.rendered_image.permute(1, 2, 0)  # (H, W, 3)
        
        # 3DGS 不直接输出 mask，使用全 1
        H, W = color.shape[:2]
        mask = torch.ones(H, W, device=self.device)  # (H, W)
        alpha = mask  # (H, W)
        
        # 占位深度
        depth = torch.zeros(H, W, device=self.device)  # (H, W)
        
        return {
            'color': color,
            'mask': mask,
            'alpha': alpha,
            'depth': depth,
        }
    
    # ========== 扩展接口 ==========
    
    def render_with_color_override(
        self,
        geometry: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        colors_overwrite: torch.Tensor,
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        使用覆盖颜色渲染
        
        Args:
            geometry: Gaussian 表示
            extrinsics: (4, 4) 相机外参
            intrinsics: (3, 3) 相机内参
            colors_overwrite: (N, 3) 覆盖颜色
            return_types: 返回类型列表
        
        Returns:
            RenderOutput: 渲染结果
        """
        self._colors_overwrite = colors_overwrite
        result = self.render(geometry, extrinsics, intrinsics, return_types)
        self._colors_overwrite = None
        return result
    
    # ========== 空输出 ==========
    
    def _get_empty_output(self) -> RenderOutput:
        """返回空 Gaussian 的输出"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),
            alpha=torch.zeros(res, res, device=device),
            mask=torch.zeros(res, res, device=device),
            color=torch.zeros(res, res, 3, device=device),
        )
