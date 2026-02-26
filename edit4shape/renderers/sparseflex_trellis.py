"""
Trellis Mesh Renderer using nvdiffrast.

重构版：继承 BaseRenderer，遵循 7 阶段渲染流水线。
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import nvdiffrast.torch as dr

from edit4shape.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
)


@dataclass
class MeshRasterData:
    """
    Mesh 光栅化中间数据
    
    Attributes:
        vertices_clip: (1, V, 4) clip space 顶点
        vertices_cam: (1, V, 4) camera space 顶点
        faces: (F, 3) 面索引
        mesh: 原始 mesh (MeshExtractResult)
    """
    vertices_clip: torch.Tensor  # (1, V, 4)
    vertices_cam: torch.Tensor   # (1, V, 4)
    faces: torch.Tensor          # (F, 3)
    mesh: Any


class TrellisMeshRasterizer(BaseRenderer):
    """
    Trellis 专用 Mesh 光栅化器 (nvdiffrast)
    
    继承 BaseRenderer，实现 7 阶段渲染流水线:
        Stage 1: prepare_inputs - 检查空 mesh
        Stage 2: compute_camera_data - 计算 MVP 矩阵
        Stage 3: process_geometry - 顶点变换到 clip/camera space
        Stage 4: rasterize_core - nvdiffrast 光栅化
        Stage 5: interpolate_attributes - 插值 depth/normal/color
        Stage 6: post_process - SSAA 下采样
        Stage 7: assemble_output - 组装 RenderOutput
    
    使用示例:
        config = RenderConfig(resolution=512, near=0.01, far=100.0, ssaa=2)
        renderer = TrellisMeshRasterizer(config)
        output = renderer.render(mesh, extrinsics, intrinsics)
        # output.depth: (512, 512)
        # output.normal: (512, 512, 3)
    """
    
    def __init__(self, config: RenderConfig = None, device: str = "cuda"):
        """
        Args:
            config: 渲染配置，None 时使用默认值
            device: 计算设备
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.01, far=100.0, ssaa=1)
        super().__init__(config, device)
        
        # 初始化 nvdiffrast 上下文
        self.glctx = dr.RasterizeCudaContext(device=device)
    
    # ========== Stage 1: Input Preparation ==========
    
    def _is_empty_geometry(self, geometry: Any) -> bool:
        """检查 mesh 是否为空"""
        return geometry.vertices.shape[0] == 0 or geometry.faces.shape[0] == 0
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(
        self,
        geometry: Any,  # MeshExtractResult
        camera_data: CameraData,
    ) -> MeshRasterData:
        """
        顶点变换到 clip space 和 camera space
        
        Args:
            geometry: MeshExtractResult，包含 vertices (V, 3), faces (F, 3)
            camera_data: 相机数据
        
        Returns:
            MeshRasterData: 变换后的顶点数据
        """
        # 顶点齐次坐标
        vertices = geometry.vertices.unsqueeze(0)  # (1, V, 3)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)  # (1, V, 4)
        
        # Clip Space: v @ MVP.T
        vertices_clip = torch.bmm(
            vertices_homo, 
            camera_data.mvp.unsqueeze(0).transpose(-1, -2)
        )  # (1, V, 4)
        
        # Camera Space (for depth): v @ Extrinsics.T
        vertices_cam = torch.bmm(
            vertices_homo, 
            camera_data.extrinsics.unsqueeze(0).transpose(-1, -2)
        )  # (1, V, 4)
        
        return MeshRasterData(
            vertices_clip=vertices_clip,
            vertices_cam=vertices_cam,
            faces=geometry.faces.int(),
            mesh=geometry,
        )
    
    # ========== Stage 4: Rasterization Core ==========
    
    def _rasterize_core(
        self,
        processed_geometry: MeshRasterData,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        nvdiffrast 光栅化
        
        Args:
            processed_geometry: 变换后的顶点数据
            camera_data: 相机数据
        
        Returns:
            RasterOutput: 光栅化结果
        """
        resolution = self.config.resolution
        ssaa = self.config.ssaa
        render_res = resolution * ssaa
        
        # 光栅化
        rast, _ = dr.rasterize(
            self.glctx,
            processed_geometry.vertices_clip,
            processed_geometry.faces,
            (render_res, render_res)
        )  # (1, H, W, 4) - [u, v, z, triangle_id]
        
        return RasterOutput(
            rast=rast,
            depth_buffer=rast[..., 2],       # (1, H, W)
            primitive_id=rast[..., 3].long(),  # (1, H, W)
        )
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,  # MeshRasterData
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        属性插值
        
        Args:
            raster_output: 光栅化结果
            geometry: 原始 mesh
            camera_data: 相机数据
            return_types: 需要返回的属性列表
        
        Returns:
            Dict[str, Tensor]: 插值后的属性
        """
        rast = raster_output.rast  # (1, H, W, 4)
        
        processed = geometry
        mesh = geometry.mesh
        v_clip = geometry.vertices_clip
        v_cam = geometry.vertices_cam
        faces = geometry.faces
        cam_rot = camera_data.extrinsics[:3, :3].contiguous()  # (3, 3)
        
        result = {}
        
        for attr_type in return_types:
            img = self._interpolate_single_attribute(
                attr_type, rast, v_clip, v_cam, mesh, faces, cam_rot
            )  # (1, H, W, C)
            
            # 去除 batch 维度，转换为 (H, W, C) 或 (H, W)
            img = img.squeeze(0)  # (H, W, C)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)  # (H, W)
            
            result[attr_type] = img
        
        # 确保有 alpha
        if 'alpha' not in result and 'mask' in result:
            result['alpha'] = result['mask']
        elif 'alpha' not in result:
            result['alpha'] = (rast[..., -1] > 0).float().squeeze(0)  # (H, W)
        
        return result
    
    def _interpolate_single_attribute(
        self,
        attr_type: str,
        rast: torch.Tensor,      # (1, H, W, 4)
        v_clip: torch.Tensor,    # (1, V, 4)
        v_cam: torch.Tensor,     # (1, V, 4)
        mesh: Any,
        faces: torch.Tensor,     # (F, 3)
        cam_rot: torch.Tensor,  # (3, 3)
    ) -> torch.Tensor:
        """
        插值单个属性
        
        Args:
            attr_type: 属性类型 ('mask', 'depth', 'normal', 'color', 'normal_map')
            rast: 光栅化结果
            v_clip: clip space 顶点
            v_cam: camera space 顶点
            mesh: 原始 mesh
            faces: 面索引
        
        Returns:
            img: (1, H, W, C) 插值结果
        """
        if attr_type == "mask":
            # 掩码：triangle_id > 0
            img = dr.antialias(
                (rast[..., -1:] > 0).float(), 
                rast, v_clip, faces
            )  # (1, H, W, 1)
            
        elif attr_type == "depth":
            # 深度：camera space z
            depth = v_cam[..., 2:3].contiguous()  # (1, V, 1)
            img = dr.interpolate(depth, rast, faces)[0]  # (1, H, W, 1)
            img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 1)
            
        elif attr_type == "normal":
            # 面法线插值
            # 构建 per-face-vertex 法线索引
            face_normal_indices = torch.arange(
                mesh.faces.shape[0] * 3, 
                device=self.device, 
                dtype=torch.int
            ).reshape(-1, 3)  # (F, 3)

            normals_world = mesh.face_normal  # (F, 3) or (F, 3, 3)
            normals_cam = -(normals_world @ cam_rot.T)  # (F, 3) or (F, 3, 3)
            normals_cam = normals_cam.reshape(1, -1, 3)  # (1, F*3, 3)

            normals = dr.interpolate(
                normals_cam,
                rast,
                face_normal_indices,
            )[0]  # (1, H, W, 3)
            img = dr.antialias(normals, rast, v_clip, faces)  # (1, H, W, 3)
            # 转换到可视化范围 [0, 1]
            img = (img + 1) / 2  # (1, H, W, 3)

            # 背景混合（用 mask）
            mask = dr.antialias(
                (rast[..., -1:] > 0).float(),
                rast,
                v_clip,
                faces
            )  # (1, H, W, 1)
            bg = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 1, 1, 3)  # (1, 1, 1, 3)
            img = img * mask + bg * (1 - mask)  # (1, H, W, 3)
            
        elif attr_type == "normal_map":
            # 顶点法线贴图
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 6:
                nm = mesh.vertex_attrs[:, 3:6].unsqueeze(0).contiguous()  # (1, V, 3)
                img = dr.interpolate(nm, rast, faces)[0]  # (1, H, W, 3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1, H, W, 3)
                
        elif attr_type == "color":
            # 顶点颜色
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 3:
                color = mesh.vertex_attrs[:, :3].unsqueeze(0).contiguous()  # (1, V, 3)
                img = dr.interpolate(color, rast, faces)[0]  # (1, H, W, 3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1, H, W, 3)
                
        else:
            # 未知属性类型
            img = torch.zeros_like(rast[..., :1])  # (1, H, W, 1)
        
        return img
    
    # ========== 空输出 ==========
    
    def _get_empty_output(self) -> RenderOutput:
        """返回空 mesh 的输出"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),        # (H, W)
            alpha=torch.zeros(res, res, device=device),        # (H, W)
            mask=torch.zeros(res, res, device=device),         # (H, W)
            normal=torch.zeros(res, res, 3, device=device),    # (H, W, 3)
            color=torch.zeros(res, res, 3, device=device),     # (H, W, 3)
        )
