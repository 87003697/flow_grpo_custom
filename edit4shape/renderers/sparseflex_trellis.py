"""
Trellis Mesh Renderer using nvdiffrast.
Independent implementation without threestudio base dependency.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


@dataclass
class TrellisRendererConfig:
    """渲染器配置"""
    resolution: int = 512
    near: float = 0.01
    far: float = 100.0
    ssaa: int = 1
    bg_color: float = 0.0


class TrellisMeshRasterizer:
    """
    Trellis 专用光栅化器 (nvdiffrast)。
    不依赖 geometry.isosurface，直接渲染 MeshExtractResult。
    """

    def __init__(self, cfg: Optional[TrellisRendererConfig] = None, device: str = "cuda"):
        self.cfg = cfg or TrellisRendererConfig()
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)

    def render(
        self,
        mesh: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types: List[str] = ["mask", "depth", "normal", "color"],
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个 Mesh 的单视角。
        
        Args:
            mesh: MeshExtractResult
            extrinsics: (4, 4) W2C Matrix (OpenCV)
            intrinsics: (3, 3) Camera Intrinsics
        Returns:
            Dict[str, Tensor]: (H, W, C)
        """
        resolution = self.cfg.resolution  # 标量 ()
        ssaa = self.cfg.ssaa  # 标量 ()

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return self._get_empty_output(resolution, return_types)

        # 1. 构建矩阵 (像素坐标系 intrinsics -> OpenGL)
        proj = self._get_projection_matrix(intrinsics)  # (4,4)
        mvp = proj @ extrinsics  # (4,4)

        # 2. 顶点变换
        vertices = mesh.vertices.unsqueeze(0)  # (1,Nv,3)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)  # (1,Nv,4)

        # Clip Space: v * MVP.T
        vertices_clip = torch.bmm(vertices_homo, mvp.unsqueeze(0).transpose(-1, -2))  # (1,Nv,4)
        
        # Camera Space (for depth): v * Ext.T
        vertices_cam = torch.bmm(vertices_homo, extrinsics.unsqueeze(0).transpose(-1, -2))  # (1,Nv,4)

        # 3. 光栅化
        faces_int = mesh.faces.int()  # (F,3)
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa)
        )  # (1, H*ssaa, W*ssaa, 4)

        # 4. 插值与后处理
        out = {}
        for k in return_types:
            img = self._process_channel(k, rast, vertices_clip, vertices_cam, mesh, faces_int)  # (1,H*ssaa,W*ssaa,C)

            # SSAA 下采样
            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),  # (1,C,H*ssaa,W*ssaa)
                    (resolution, resolution),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )  # (1,C,H,W)
            else:
                img = img.permute(0, 3, 1, 2)  # (1,C,H,W)

            out[k] = img.squeeze(0).permute(1, 2, 0)  # (H,W,C)

        return out

    def _get_projection_matrix(self, K):
        """
        像素坐标 Intrinsics -> OpenGL Projection（对齐参考 TRELLIS）。
        """
        fx, fy = K[0, 0], K[1, 1]  # 标量 (), ()
        cx, cy = K[0, 2], K[1, 2]  # 标量 (), ()
        n, f = self.cfg.near, self.cfg.far  # 标量 (), ()

        ret = torch.zeros((4, 4), device=K.device, dtype=K.dtype)  # (4,4)
        ret[0, 0] = 2 * fx  # 标量 ()
        ret[1, 1] = 2 * fy  # 标量 ()
        ret[0, 2] = 2 * cx - 1  # 标量 ()
        ret[1, 2] = -2 * cy + 1  # 标量 ()
        ret[2, 2] = f / (f - n)  # 标量 ()
        ret[2, 3] = n * f / (n - f)  # 标量 ()
        ret[3, 2] = 1.0  # 标量 ()
        return ret  # (4,4)

    def _process_channel(self, type_name, rast, v_clip, v_cam, mesh, faces):
        if type_name == "mask":
            img = dr.antialias((rast[..., -1:] > 0).float(), rast, v_clip, faces)  # (1,H,W,1)
        elif type_name == "depth":
            depth = v_cam[..., 2:3].contiguous()  # (1,Nv,1)
            img = dr.interpolate(depth, rast, faces)[0]  # (1,H,W,1)
            img = dr.antialias(img, rast, v_clip, faces)  # (1,H,W,1)
        elif type_name == "normal":
            normals = dr.interpolate(
                mesh.face_normal.reshape(1, -1, 3),  # (1,F*3,3)
                rast,
                torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)  # (F,3)
            )[0]  # (1,H,W,3)
            img = dr.antialias(normals, rast, v_clip, faces)  # (1,H,W,3)
            img = (img + 1) / 2  # (1,H,W,3)
        elif type_name == "normal_map":
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 6:
                nm = mesh.vertex_attrs[:, 3:6].contiguous()  # (Nv,3)
                img = dr.interpolate(nm, rast, faces)[0]  # (1,H,W,3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1,H,W,3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1,H,W,3)
        elif type_name == "color":
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 3:
                color = mesh.vertex_attrs[:, :3].contiguous()  # (Nv,3)
                img = dr.interpolate(color, rast, faces)[0]  # (1,H,W,3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1,H,W,3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1,H,W,3)
        else:
            img = torch.zeros_like(rast[..., :1])  # (1,H,W,1)
        return img  # (1,H,W,C)

    def _get_empty_output(self, res, types):
        out = {}
        for t in types:
            c = 3 if t in ['color', 'normal'] else 1
            out[t] = torch.zeros((res, res, c), device=self.device)  # (res,res,c)
        return out

