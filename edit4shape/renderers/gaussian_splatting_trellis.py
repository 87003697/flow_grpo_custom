#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F

# 从参考代码 TRELLIS 导入 Gaussian 类
from trellis.representations.gaussian import Gaussian
from trellis.renderers.sh_utils import eval_sh


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # 标量 (), ()
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # 标量 (), ()
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)  # (4,4)
    ret[0, 0] = 2 * fx  # (4,4)
    ret[1, 1] = 2 * fy  # (4,4)
    ret[0, 2] = 2 * cx - 1  # (4,4)
    ret[1, 2] = - 2 * cy + 1  # (4,4)
    ret[2, 2] = far / (far - near)  # (4,4)
    ret[2, 3] = near * far / (near - far)  # (4,4)
    ret[3, 2] = 1.  # (4,4)
    return ret  # (4,4)


def render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # lazy import
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0  # (N,3)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 标量 ()
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 标量 ()
    
    kernel_size = pipe.kernel_size
    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")  # (H,W,2)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz  # (N,3)
    means2D = screenspace_points  # (N,3)
    opacity = pc.get_opacity  # (N,1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # (N,3,(D^2))
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # (N,3)
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)  # (N,3)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  # (N,3)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N,3)
        else:
            shs = pc.get_features  # (N,C,? )
    else:
        colors_precomp = override_color  # (N,3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,  # (N,3)
        means2D = means2D,  # (N,3)
        shs = shs,  # (N,C,?)
        colors_precomp = colors_precomp,  # (N,3) 或 None
        opacities = opacity,  # (N,1)
        scales = scales,  # (N,3) or None
        rotations = rotations,  # (N,4) or None
        cov3D_precomp = cov3D_precomp  # (N,3,3) or None
    )  # rendered_image: (3,H,W), radii: (N,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return edict({"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]  # 标量 ()
        near = self.rendering_options["near"]  # 标量 ()
        far = self.rendering_options["far"]  # 标量 ()
        ssaa = self.rendering_options["ssaa"]  # 标量 ()
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")  # (3,)
            if np.random.rand() < 0.5:
                self.bg_color += 1  # (3,)
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")  # (3,)

        view = extrinsics  # (4,4)
        perspective = intrinsics_to_projection(intrinsics, near, far)  # (4,4)
        camera = torch.inverse(view)[:3, 3]  # (3,)
        focalx = intrinsics[0, 0]  # 标量 ()
        focaly = intrinsics[1, 1]  # 标量 ()
        fovx = 2 * torch.atan(0.5 / focalx)  # 标量 ()
        fovy = 2 * torch.atan(0.5 / focaly)  # 标量 ()
            
        camera_dict = edict({
            "image_height": resolution * ssaa,  # 标量 ()
            "image_width": resolution * ssaa,  # 标量 ()
            "FoVx": fovx,  # 标量 ()
            "FoVy": fovy,  # 标量 ()
            "znear": near,  # 标量 ()
            "zfar": far,  # 标量 ()
            "world_view_transform": view.T.contiguous(),  # (4,4)
            "projection_matrix": perspective.T.contiguous(),  # (4,4)
            "full_proj_transform": (perspective @ view).T.contiguous(),  # (4,4)
            "camera_center": camera  # (3,)
        })

        # Render
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)  # render: (3,H*ssaa,W*ssaa)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()  # (3,H,W)
            
        ret = edict({
            'color': render_ret['render']  # (3,H,W)
        })
        return ret
