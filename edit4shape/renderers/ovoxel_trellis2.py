import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, TYPE_CHECKING
from easydict import EasyDict as edict

# 使用 TYPE_CHECKING 避免运行时导入问题
# Voxel 类型可来自 trellis2.representations 或其他兼容实现
if TYPE_CHECKING:
    from typing import Any
    Voxel = Any  # Duck typing: 任何具有 position, attrs, voxel_size, layout 属性的对象


class VoxelRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,
            "far": 10.0,
            "ssaa": 1,
        })
        self.rendering_options.update(rendering_options)
    
    def render(
            self,
            voxel,  # Voxel-like object with position, attrs, voxel_size, layout
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the voxel.

        Args:
            voxel: Voxel representation (requires position, attrs, voxel_size, layout).
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
                depth (torch.Tensor): (H, W) rendered depth
                alpha (torch.Tensor): (H, W) rendered alpha
                ...
        """ 
        # lazy import
        if 'o_voxel' not in globals():
            import o_voxel
        renderer = o_voxel.rasterize.VoxelRenderer(self.rendering_options)
        positions = voxel.position
        attrs = voxel.attrs if colors_overwrite is None else colors_overwrite
        voxel_size = voxel.voxel_size
        
        # Render
        render_ret = renderer.render(positions, attrs, voxel_size, extrinsics, intrinsics)
        
        ret = {
            'depth': render_ret['depth'],
            'alpha': render_ret['alpha'],
        }
        if colors_overwrite is not None:
            ret['color'] = render_ret['attr']
        else:
            for k, s in voxel.layout.items():
                ret[k] = render_ret['attr'][s]
        
        return ret


def depth_to_normal(
    depth: torch.Tensor, 
    intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    从 depth 图估算相机空间法线
    
    Args:
        depth: [H, W] 深度图
        intrinsics: [3, 3] 相机内参
        mask: [H, W] 可选的有效区域 mask
        
    Returns:
        normal: [3, H, W] 相机空间法线，朝向相机为正 Z
    """
    # 确保输入是 dense tensor 并且是 contiguous
    depth = depth.contiguous()
    if mask is not None:
        mask = mask.contiguous()
    
    H, W = depth.shape
    device = depth.device
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 创建像素网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # y_grid: [H, W], x_grid: [H, W]
    
    # 反投影到相机空间 3D 点
    z = depth  # [H, W]
    x = (x_grid - cx * W) * z / (fx * W)  # [H, W]
    y = (y_grid - cy * H) * z / (fy * H)  # [H, W]
    
    # 计算偏导数（中心差分）
    # dP/dx 和 dP/dy
    dx = torch.zeros(H, W, 3, device=device)  # [H, W, 3]
    dy = torch.zeros(H, W, 3, device=device)  # [H, W, 3]
    
    # X 方向梯度
    dx[:, 1:-1, 0] = (x[:, 2:] - x[:, :-2]) / 2  # [H, W-2]
    dx[:, 1:-1, 1] = (y[:, 2:] - y[:, :-2]) / 2  # [H, W-2]
    dx[:, 1:-1, 2] = (z[:, 2:] - z[:, :-2]) / 2  # [H, W-2]
    
    # Y 方向梯度
    dy[1:-1, :, 0] = (x[2:, :] - x[:-2, :]) / 2  # [H-2, W]
    dy[1:-1, :, 1] = (y[2:, :] - y[:-2, :]) / 2  # [H-2, W]
    dy[1:-1, :, 2] = (z[2:, :] - z[:-2, :]) / 2  # [H-2, W]
    
    # 确保 dx 和 dy 是正确形状
    dx = dx.contiguous()  # [H, W, 3]
    dy = dy.contiguous()  # [H, W, 3]
    
    # 法线 = dy × dx（注意顺序，使法线朝向相机）
    normal = torch.linalg.cross(dy, dx)  # [H, W, 3]，使用 linalg.cross 更稳定
    normal = F.normalize(normal, dim=-1)  # [H, W, 3]
    
    # 确保法线朝向相机（z 分量为负）
    normal = torch.where(normal[..., 2:3] > 0, -normal, normal)  # [H, W, 3]
    
    if mask is not None:
        # 确保 mask 是 2D [H, W]
        if mask.dim() > 2:
            mask = mask.squeeze()
        normal = normal * mask.unsqueeze(-1)  # [H, W, 3]
    
    # 确保输出是正确形状 [H, W, 3]
    while normal.dim() > 3:
        normal = normal.squeeze(0)
    
    return normal.permute(2, 0, 1).contiguous()  # [3, H, W]


def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
    """ACES tone mapping"""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def gamma_correction(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Gamma correction"""
    return torch.clamp(x ** (1.0 / gamma), 0.0, 1.0)


class PbrVoxelRenderer(VoxelRenderer):
    """
    PBR Renderer for the Voxel representation with IBL shading.
    
    Inherits from VoxelRenderer and adds PBR shading capabilities.
    
    Args:
        rendering_options (dict): Rendering options.
        device (str): Device to use.
    """
    
    def __init__(self, rendering_options={}, device='cuda') -> None:
        super().__init__(rendering_options)
        self.device = device
    
    def render(
        self,
        voxel,  # Voxel-like object with position, attrs, voxel_size, layout
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        envmap=None,  # EnvMap 或 Dict[str, EnvMap]
        colors_overwrite: torch.Tensor = None,
        shade: bool = True,
        use_envmap_bg: bool = False,
    ) -> edict:
        """
        Render the voxel with optional PBR shading.

        Args:
            voxel: Voxel representation (requires position, attrs, voxel_size, layout).
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            envmap: EnvMap or Dict[str, EnvMap] for IBL shading
            colors_overwrite (torch.Tensor): (N, 3) override color
            shade (bool): Whether to apply PBR shading
            use_envmap_bg (bool): Whether to use envmap as background

        Returns:
            edict containing:
                shaded (torch.Tensor): [3, H, W] PBR shaded image (if shade=True)
                normal (torch.Tensor): [3, H, W] normal image
                base_color, metallic, roughness, alpha, depth, ...
        """
        # 1. 调用父类获取基础渲染结果
        ret = super().render(voxel, extrinsics, intrinsics, colors_overwrite)
        
        resolution = self.rendering_options.resolution
        
        # 2. 从 depth 估算 normal
        depth = ret['depth']  # [H, W]
        alpha = ret['alpha']  # [H, W]
        
        # 确保是 dense tensor
        if depth.is_sparse:
            depth = depth.to_dense()
        if alpha.is_sparse:
            alpha = alpha.to_dense()
            
        mask = (alpha > 0.5).float()  # [H, W]，确保是 float 类型的 dense tensor
        
        normal = depth_to_normal(depth, intrinsics, mask)  # [3, H, W]
        
        # 转换到相机空间的可视化格式
        ret['normal'] = (-normal * 0.5 + 0.5) * mask.unsqueeze(0)  # [3, H, W]
        
        # 3. 如果不需要着色或没有 envmap，直接返回
        if not shade or envmap is None:
            return ret
        
        # 4. 准备 PBR 着色
        import utils3d
        
        if not isinstance(envmap, dict):
            envmap = {'': envmap}
        
        # 获取 PBR 属性
        base_color = ret.get('base_color', torch.ones(3, resolution, resolution, device=self.device) * 0.5)
        metallic = ret.get('metallic', torch.zeros(1, resolution, resolution, device=self.device))
        roughness = ret.get('roughness', torch.ones(1, resolution, resolution, device=self.device) * 0.5)
        
        # 确保维度正确
        if base_color.dim() == 2:
            base_color = base_color.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        if metallic.dim() == 2:
            metallic = metallic.unsqueeze(0)  # [1, H, W]
        if roughness.dim() == 2:
            roughness = roughness.unsqueeze(0)  # [1, H, W]
        
        # 获取射线
        rays_o, rays_d = utils3d.torch.get_image_rays(
            extrinsics, intrinsics, resolution, resolution
        )  # rays_o: [H, W, 3], rays_d: [H, W, 3]
        
        # 重建 3D 位置
        pos = rays_o + rays_d * depth.unsqueeze(-1)  # [H, W, 3]
        
        # 转换 normal 到世界空间
        R = extrinsics[:3, :3]  # [3, 3]
        normal_cam = normal.permute(1, 2, 0)  # [H, W, 3]
        normal_world = (normal_cam @ R)  # [H, W, 3]，相机空间 -> 世界空间
        
        # Gamma 校正输入（sRGB -> Linear）
        base_color_linear = base_color.permute(1, 2, 0) ** 2.2  # [H, W, 3]
        
        # 准备 ORM 格式 (Occlusion, Roughness, Metallic)
        orm = torch.cat([
            torch.zeros_like(metallic),  # Occlusion = 0, [1, H, W]
            roughness,                   # [1, H, W]
            metallic,                    # [1, H, W]
        ], dim=0).permute(1, 2, 0)  # [H, W, 3]
        
        # 5. IBL 着色
        shaded_results = {}
        for name, env in envmap.items():
            gb_shaded = env.shade(
                pos.unsqueeze(0),           # [1, H, W, 3]
                normal_world.unsqueeze(0),  # [1, H, W, 3]
                base_color_linear.unsqueeze(0),  # [1, H, W, 3]
                orm.unsqueeze(0),           # [1, H, W, 3]
                rays_o,                     # [H, W, 3]
                specular=True,
            )[0]  # [H, W, 3]
            
            # 确保 gb_shaded 是 3D [H, W, 3]
            while gb_shaded.dim() > 3:
                gb_shaded = gb_shaded.squeeze(0)
            
            # 确保 mask 是 2D
            mask_2d = mask
            while mask_2d.dim() > 2:
                mask_2d = mask_2d.squeeze(0)
            
            # 应用 mask
            gb_shaded = gb_shaded * mask_2d.unsqueeze(-1)  # [H, W, 3]
            
            # 背景
            if use_envmap_bg:
                bg = env.sample(rays_d)  # [H, W, 3]
                while bg.dim() > 3:
                    bg = bg.squeeze(0)
                gb_shaded = gb_shaded + (1 - mask_2d.unsqueeze(-1).float()) * bg  # [H, W, 3]
            
            # 后处理
            gb_shaded = aces_tonemapping(gb_shaded)  # [H, W, 3]
            gb_shaded = gamma_correction(gb_shaded)  # [H, W, 3]
            
            key = f"shaded_{name}" if name else "shaded"
            shaded_results[key] = gb_shaded.permute(2, 0, 1).contiguous()  # [3, H, W]
        
        ret.update(shaded_results)
        ret['mask'] = mask.float()  # [H, W]
        
        return ret
