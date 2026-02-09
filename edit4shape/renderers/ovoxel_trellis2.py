"""
Voxel 渲染器 + PBR 着色

重构版：继承 BaseRenderer，遵循 7 阶段渲染流水线。

维度规范：
    - depth, alpha, mask: 2D (H, W)
    - normal, color, shaded: 3D (H, W, 3)
    - 相机参数: extrinsics (4, 4), intrinsics (3, 3)
"""

import logging
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from edit4shape.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    PBRPostProcessMixin,
    depth_to_normal,
    camera_normal_to_vis,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class VoxelProxy:
    """
    从 FDG Decoder 输出构建的伪体素对象。
    
    Attributes:
        position: (N, 3) 体素位置，可微
        opacities: (N,) 体素不透明度，可微
        voxel_size: 体素大小
        batch_indices: (N,) batch 索引
    """
    position: torch.Tensor      # (N, 3)
    opacities: torch.Tensor     # (N,)
    voxel_size: float
    batch_indices: torch.Tensor  # (N,)
    
    @classmethod
    def from_fdg_decoder(
        cls,
        h_feats: torch.Tensor,    # (N, 7)
        coords: torch.Tensor,      # (N, 4) [batch_idx, x, y, z]
        resolution: int,
        voxel_margin: float = 0.5,
    ) -> "VoxelProxy":
        """
        从 FDG Decoder 输出构建 VoxelProxy。
        
        Args:
            h_feats: (N, 7) decoder 输出，[0:3] dual_vertices, [3:6] intersected
            coords: (N, 4) 稀疏坐标
            resolution: 网格分辨率
            voxel_margin: 顶点偏移范围
        
        Returns:
            VoxelProxy 对象
        """
        device = h_feats.device
        origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
        voxel_size = 1.0 / resolution
        
        # 位置: base_position + dual_vertices 偏移 (可微)
        dual_vertices = (1 + 2 * voxel_margin) * F.sigmoid(h_feats[..., 0:3]) - voxel_margin  # (N, 3)
        base_position = (coords[:, 1:4].float() + 0.5) * voxel_size + origin  # (N, 3)
        position = base_position + (dual_vertices - 0.5) * voxel_size  # (N, 3)
        
        # 不透明度: sigmoid(max(intersected_logits)) (可微)
        intersected_logits = h_feats[..., 3:6]  # (N, 3)
        max_logit = intersected_logits.max(dim=-1).values  # (N,)
        opacities = torch.sigmoid(max_logit * 10.0)  # (N,)
        
        return cls(position, opacities, voxel_size, coords[:, 0])
    
    def filter_by_batch(self, batch_idx: int) -> "VoxelProxy":
        """
        过滤指定 batch 的体素。
        
        Args:
            batch_idx: batch 索引
        
        Returns:
            只包含该 batch 的 VoxelProxy
        """
        mask = self.batch_indices == batch_idx
        return VoxelProxy(
            self.position[mask],
            self.opacities[mask],
            self.voxel_size,
            self.batch_indices[mask],
        )


@dataclass
class VoxelRasterData:
    """
    Voxel 光栅化中间数据
    
    Attributes:
        positions: (N, 3) 体素位置
        attrs: (N, C) 体素属性
        voxel_size: 体素大小
        layout: Dict[str, slice] 属性布局
    """
    positions: torch.Tensor
    attrs: torch.Tensor
    voxel_size: float
    layout: Optional[Dict[str, slice]] = None


# ============================================================================
# 工具函数
# ============================================================================

def load_envmap(envmap_path: str, device: str = 'cuda'):
    """
    加载 PBR 环境贴图（使用 TRELLIS.2 的 EnvMap 类）。
    
    Args:
        envmap_path: 环境贴图路径（支持 .exr, .hdr 格式）
        device: 目标设备
    
    Returns:
        EnvMap: TRELLIS.2 的 EnvMap 对象
    """
    import os
    import cv2
    
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from trellis2.renderers import EnvMap
    
    env_bgr = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)  # (H, W, 3) BGR
    if env_bgr is None:
        raise FileNotFoundError(f"无法加载环境贴图: {envmap_path}")
    
    env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3)
    env_tensor = torch.tensor(env_rgb, dtype=torch.float32, device=device)  # (H, W, 3)
    
    return EnvMap(env_tensor)


# ============================================================================
# VoxelRenderer - 基础体素渲染器
# ============================================================================

class VoxelRenderer(BaseRenderer):
    """
    基础 Voxel 渲染器（使用 o_voxel 后端）
    
    继承 BaseRenderer，实现 7 阶段渲染流水线:
        Stage 1: prepare_inputs - 检查空体素
        Stage 2: compute_camera_data - 计算相机参数
        Stage 3: process_geometry - 提取体素位置和属性
        Stage 4: rasterize_core - 调用 o_voxel 光栅化
        Stage 5: interpolate_attributes - 解析属性布局
        Stage 6: post_process - 可选后处理
        Stage 7: assemble_output - 组装 RenderOutput
    
    输出维度:
        depth: (H, W)
        alpha: (H, W)
        base_color: (H, W, 3)
        metallic: (H, W)
        roughness: (H, W)
    """
    
    def __init__(self, config: RenderConfig = None, device: str = 'cuda'):
        """
        Args:
            config: 渲染配置
            device: 计算设备
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1)
        super().__init__(config, device)
        
        self._colors_overwrite: Optional[torch.Tensor] = None
    
    # ========== Stage 1: Input Preparation ==========
    
    def _is_empty_geometry(self, geometry: Any) -> bool:
        """检查体素是否为空"""
        if hasattr(geometry, 'position'):
            return geometry.position.shape[0] == 0
        return False
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(
        self,
        geometry: Any,  # Voxel-like
        camera_data: CameraData,
    ) -> VoxelRasterData:
        """
        提取体素数据
        """
        positions = geometry.position  # (N, 3)
        attrs = self._colors_overwrite if self._colors_overwrite is not None else geometry.attrs  # (N, C)
        voxel_size = geometry.voxel_size
        layout = getattr(geometry, 'layout', None)
        
        return VoxelRasterData(
            positions=positions,
            attrs=attrs,
            voxel_size=voxel_size,
            layout=layout,
        )
    
    # ========== Stage 4: Rasterization Core ==========
    
    def _rasterize_core(
        self,
        processed_geometry: VoxelRasterData,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        调用 o_voxel 光栅化
        """
        import o_voxel
        
        # 构建 o_voxel 期望的 rendering_options
        rendering_options = {
            "resolution": self.config.resolution,
            "near": self.config.near,
            "far": self.config.far,
            "ssaa": self.config.ssaa,
        }
        
        renderer = o_voxel.rasterize.VoxelRenderer(rendering_options)
        render_ret = renderer.render(
            processed_geometry.positions,
            processed_geometry.attrs,
            processed_geometry.voxel_size,
            camera_data.extrinsics,
            camera_data.intrinsics,
        )
        
        H = W = self.config.resolution
        depth = render_ret['depth'].reshape(H, W)  # (H, W)
        alpha = render_ret['alpha'].reshape(H, W)  # (H, W)
        
        return RasterOutput(
            rast={
                'depth': depth,
                'alpha': alpha,
                'attr': render_ret.get('attr'),  # (C, H, W) 或 None
                'layout': processed_geometry.layout,
            },
            depth_buffer=depth,
            primitive_id=torch.zeros(H, W, device=self.device, dtype=torch.long),
        )
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        解析属性布局
        """
        rast = raster_output.rast
        H = W = self.config.resolution
        
        result = {
            'depth': rast['depth'],  # (H, W)
            'alpha': rast['alpha'],  # (H, W)
        }
        
        attr = rast.get('attr')
        layout = rast.get('layout')
        
        if self._colors_overwrite is not None and attr is not None:
            # 覆盖模式：attr -> color
            result['color'] = attr.permute(1, 2, 0).reshape(H, W, -1)  # (H, W, C)
        elif layout is not None and attr is not None:
            # 按 layout 解析各属性
            for k, s in layout.items():
                attr_k = attr[s]  # (C_k, H, W)
                if attr_k.shape[0] == 1:
                    result[k] = attr_k.squeeze(0)  # (H, W)
                else:
                    result[k] = attr_k.permute(1, 2, 0)  # (H, W, C)
        
        return result
    
    # ========== 扩展接口 ==========
    
    def render_with_color_override(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        colors_overwrite: torch.Tensor,
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        使用覆盖颜色渲染
        """
        self._colors_overwrite = colors_overwrite
        result = self.render(geometry, extrinsics, intrinsics, return_types)
        self._colors_overwrite = None
        return result


# ============================================================================
# PbrVoxelRenderer - PBR 着色体素渲染器
# ============================================================================

class PbrVoxelRenderer(VoxelRenderer, PBRPostProcessMixin):
    """
    PBR Voxel 渲染器（带 IBL 着色）
    
    继承 VoxelRenderer + PBRPostProcessMixin
    
    输出维度:
        shaded: (H, W, 3)
        normal: (H, W, 3)
        depth: (H, W)
        alpha: (H, W)
        mask: (H, W)
    
    使用示例:
        config = RenderConfig(resolution=512)
        renderer = PbrVoxelRenderer(config)
        renderer.load_envmap('path/to/envmap.exr')
        output = renderer.render(voxel, extrinsics, intrinsics)
    """
    
    def __init__(self, config: RenderConfig = None, device: str = 'cuda'):
        super().__init__(config, device)
        self.envmap = None
        self._shade: bool = True
        self._use_envmap_bg: bool = False
        self._external_envmap = None
    
    def load_envmap(self, envmap_path: str) -> "PbrVoxelRenderer":
        """
        加载 PBR 环境贴图。
        """
        self.envmap = load_envmap(envmap_path, device=self.device)
        logging.info(f"[PbrVoxelRenderer] 加载环境贴图: {envmap_path}")
        return self
    
    # ========== Stage 6: Post-processing ==========
    
    def _post_process(
        self,
        attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """
        PBR 后处理：depth → normal → IBL shading
        """
        import utils3d
        
        H = W = self.config.resolution
        
        depth = attrs.get('depth', torch.zeros(H, W, device=self.device))  # (H, W)
        alpha = attrs.get('alpha', torch.zeros(H, W, device=self.device))  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        attrs['mask'] = mask
        
        # 从 depth 估算 normal
        normal_cam = depth_to_normal(depth, camera_data.intrinsics, mask)  # (H, W, 3)
        attrs['normal'] = camera_normal_to_vis(normal_cam, mask)  # (H, W, 3)
        
        # 选择环境贴图
        envmap = self._external_envmap if self._external_envmap is not None else self.envmap
        
        # 如果不需要着色或没有环境贴图，返回
        if not self._shade or envmap is None:
            return attrs
        
        # PBR 着色
        if not isinstance(envmap, dict):
            envmap = {'': envmap}
        
        # 获取 PBR 属性
        base_color = attrs.get('base_color', torch.ones(H, W, 3, device=self.device) * 0.5)  # (H, W, 3)
        metallic = attrs.get('metallic', torch.zeros(H, W, device=self.device))  # (H, W)
        roughness = attrs.get('roughness', torch.ones(H, W, device=self.device) * 0.5)  # (H, W)
        
        # 确保维度正确
        if base_color.dim() == 2:
            base_color = base_color[..., None].expand(H, W, 3)  # (H, W, 3)
        base_color = base_color.reshape(H, W, 3)  # (H, W, 3)
        metallic = metallic.reshape(H, W)  # (H, W)
        roughness = roughness.reshape(H, W)  # (H, W)
        
        # 获取射线
        rays_o, rays_d = utils3d.torch.get_image_rays(
            camera_data.extrinsics, camera_data.intrinsics, H, W
        )  # (H, W, 3)
        
        # 重建 3D 位置
        pos = rays_o + rays_d * depth[..., None]  # (H, W, 3)
        
        # 转换 normal 到世界空间
        R = camera_data.extrinsics[:3, :3]  # (3, 3)
        normal_world = normal_cam @ R  # (H, W, 3)
        
        # sRGB -> Linear
        base_color_clamped = torch.clamp(base_color, 0.0, 1.0)  # (H, W, 3)
        base_color_linear = base_color_clamped ** 2.2  # (H, W, 3)
        
        # ORM 格式
        orm = torch.stack([
            torch.zeros_like(metallic),  # Occlusion
            roughness,
            metallic,
        ], dim=-1)  # (H, W, 3)
        
        # IBL 着色
        for name, env in envmap.items():
            shaded = env.shade(
                pos[None],              # (1, H, W, 3)
                normal_world[None],     # (1, H, W, 3)
                base_color_linear[None],# (1, H, W, 3)
                orm[None],              # (1, H, W, 3)
                rays_o,                 # (H, W, 3)
                specular=True,
            )[0]  # (H, W, 3)
            
            shaded = shaded.reshape(H, W, 3)  # (H, W, 3)
            shaded = shaded * mask[..., None]  # (H, W, 3)
            
            if self._use_envmap_bg:
                bg = env.sample(rays_d).reshape(H, W, 3)  # (H, W, 3)
                shaded = shaded + (1 - mask[..., None]) * bg  # (H, W, 3)
            
            # Tone mapping + Gamma correction
            shaded = self.aces_tonemapping(shaded)
            shaded = self.gamma_correction(shaded)
            
            key = f"shaded_{name}" if name else "shaded"
            attrs[key] = shaded
        
        return attrs
    
    # ========== 扩展接口 ==========
    
    def render_with_options(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        envmap=None,
        colors_overwrite: torch.Tensor = None,
        shade: bool = True,
        use_envmap_bg: bool = False,
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        带选项渲染
        """
        self._colors_overwrite = colors_overwrite
        self._shade = shade
        self._use_envmap_bg = use_envmap_bg
        self._external_envmap = envmap
        
        result = self.render(geometry, extrinsics, intrinsics, return_types)
        
        self._colors_overwrite = None
        self._shade = True
        self._use_envmap_bg = False
        self._external_envmap = None
        
        return result


# ============================================================================
# DiffVoxelRenderer - 可微体素渲染器
# ============================================================================

class DiffVoxelRenderer(VoxelRenderer):
    """
    可微体素渲染器（近似版本）
    
    渲染流程: VoxelProxy → o_voxel 渲染深度 → depth_to_normal → Normal
    
    梯度流: Loss → Normal → depth_to_normal → Depth → STE → opacities → Decoder
    
    注意: 使用 STE (Straight-Through Estimator) 建立梯度连接，
    因为 o_voxel 渲染器本身不可微。
    """
    
    def __init__(self, config: RenderConfig = None, device: str = 'cuda'):
        if config is None:
            config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1)
        super().__init__(config, device)
        self._voxel_proxy: Optional[VoxelProxy] = None
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(
        self,
        geometry: VoxelProxy,
        camera_data: CameraData,
    ) -> VoxelRasterData:
        """
        处理 VoxelProxy，过滤低透明度体素
        """
        self._voxel_proxy = geometry  # 保存用于 STE
        
        # 过滤低不透明度体素（加速渲染）
        visible_mask = geometry.opacities > 0.01  # (N,)
        positions = geometry.position[visible_mask]  # (M, 3)
        
        # 使用全 1 属性（只渲染深度）
        attrs = torch.ones(positions.shape[0], 1, device=self.device)  # (M, 1)
        
        return VoxelRasterData(
            positions=positions,
            attrs=attrs,
            voxel_size=geometry.voxel_size,
            layout=None,
        )
    
    # ========== Stage 6: Post-processing ==========
    
    def _post_process(
        self,
        attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """
        Depth → Normal + STE 梯度连接
        """
        depth = attrs['depth']  # (H, W)
        alpha = attrs['alpha']  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        attrs['mask'] = mask
        
        # Depth → Normal
        normal_cam = depth_to_normal(depth, camera_data.intrinsics, mask)  # (H, W, 3)
        normal_vis = camera_normal_to_vis(normal_cam, mask)  # (H, W, 3)
        
        # STE: 建立 opacities 梯度连接
        if self._voxel_proxy is not None and self._voxel_proxy.opacities.requires_grad:
            visible_mask = self._voxel_proxy.opacities > 0.01
            opacities_visible = self._voxel_proxy.opacities[visible_mask]
            if opacities_visible.numel() > 0:
                mean_opacity = opacities_visible.mean()
                normal_vis = normal_vis + (mean_opacity - mean_opacity.detach()) * 0
        
        attrs['normal'] = normal_vis
        
        return attrs
    
    # ========== 批量渲染 ==========
    
    def render_batch(
        self,
        voxel_proxy: VoxelProxy,
        extrinsics: torch.Tensor,  # (B, V, 4, 4)
        intrinsics: torch.Tensor,  # (B, V, 3, 3)
    ) -> RenderOutput:
        """
        批量渲染多个视角
        
        Args:
            voxel_proxy: VoxelProxy 对象（包含多个 batch）
            extrinsics: (B, V, 4, 4) 相机外参
            intrinsics: (B, V, 3, 3) 相机内参
        
        Returns:
            RenderOutput: 包含 normal: (B, V, H, W, 3)
        """
        B, V = extrinsics.shape[:2]
        unique_batches = voxel_proxy.batch_indices.unique().tolist()
        
        all_normals = []
        for b_idx, batch_id in enumerate(unique_batches):
            proxy_b = voxel_proxy.filter_by_batch(batch_id)
            view_normals = []
            for v in range(V):
                output = self.render(
                    proxy_b, 
                    extrinsics[b_idx, v], 
                    intrinsics[b_idx, v],
                    return_types=['normal'],
                )
                view_normals.append(output.normal)  # (H, W, 3)
            all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)
        
        normal_batch = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
        
        res = self.config.resolution
        return RenderOutput(
            depth=torch.zeros(B, V, res, res, device=self.device),
            alpha=torch.zeros(B, V, res, res, device=self.device),
            mask=torch.zeros(B, V, res, res, device=self.device),
            normal=normal_batch,
        )
