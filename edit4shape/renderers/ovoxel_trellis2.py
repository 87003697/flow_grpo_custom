"""
Voxel 渲染器 + PBR 着色（适配 trellis2.py 的维度规范）

维度规范：
    - depth, alpha, mask: 2D (H, W)
    - normal, color, shaded: 3D (H, W, 3)  ← 通道在最后
    - 相机参数: extrinsics (4, 4), intrinsics (3, 3)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass
from easydict import EasyDict as edict


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


class VoxelRenderer:
    """
    基础 Voxel 渲染器。
    
    输出维度:
        depth: (H, W)
        alpha: (H, W)
        base_color: (H, W, 3)
        metallic: (H, W, 1)
        roughness: (H, W, 1)
    """

    def __init__(self, rendering_options: Dict = None) -> None:
        self.rendering_options = edict({
            "resolution": 512,
            "near": 0.1,
            "far": 10.0,
            "ssaa": 1,
        })
        if rendering_options:
            self.rendering_options.update(rendering_options)
    
    def render(
        self,
        voxel,  # Voxel-like: 需要 position, attrs, voxel_size, layout
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
        colors_overwrite: torch.Tensor = None,
    ) -> edict:
        """
        渲染 Voxel。
        
        Returns:
            edict: {
                depth: (H, W),
                alpha: (H, W),
                base_color: (H, W, 3),
                metallic: (H, W, 1),
                ...
            }
        """
        import o_voxel
        
        H = W = self.rendering_options.resolution
        renderer = o_voxel.rasterize.VoxelRenderer(self.rendering_options)
        
        positions = voxel.position  # (N, 3)
        attrs = voxel.attrs if colors_overwrite is None else colors_overwrite  # (N, C)
        voxel_size = voxel.voxel_size
        
        render_ret = renderer.render(positions, attrs, voxel_size, extrinsics, intrinsics)
        
        # 统一为 2D: (H, W)
        depth = render_ret['depth'].reshape(H, W)  # (H, W)
        alpha = render_ret['alpha'].reshape(H, W)  # (H, W)
        
        ret = edict(depth=depth, alpha=alpha)
        
        if colors_overwrite is not None:
            # attr 输出: (C, H, W) -> (H, W, C)
            ret['color'] = render_ret['attr'].permute(1, 2, 0).reshape(H, W, -1)  # (H, W, C)
        else:
            # 按 layout 解析各属性
            for k, s in voxel.layout.items():
                attr = render_ret['attr'][s]  # (C_k, H, W)
                ret[k] = attr.permute(1, 2, 0).reshape(H, W, -1).squeeze(-1) if attr.shape[0] == 1 else attr.permute(1, 2, 0)  # (H, W) 或 (H, W, C)
        
        return ret


def depth_to_normal(
    depth: torch.Tensor,  # (H, W)
    intrinsics: torch.Tensor,  # (3, 3)
    mask: Optional[torch.Tensor] = None,  # (H, W)
) -> torch.Tensor:
    """
    从 depth 图估算相机空间法线。
    
    Returns:
        normal: (H, W, 3) 相机空间法线，朝向相机为正
    """
    H, W = depth.shape[-2:]
    depth = depth.reshape(H, W)  # 确保 2D
    device = depth.device
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 像素网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H, W)
    
    # 反投影到相机空间
    z = depth  # (H, W)
    x = (x_grid - cx * W) * z / (fx * W)  # (H, W)
    y = (y_grid - cy * H) * z / (fy * H)  # (H, W)
    
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
        mask = mask.reshape(H, W)  # 确保 2D
        normal = normal * mask[..., None]  # (H, W, 3)
    
    return normal  # (H, W, 3)


def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
    """ACES tone mapping"""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def gamma_correction(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Gamma correction"""
    return torch.clamp(x ** (1.0 / gamma), 0.0, 1.0)


def load_envmap(envmap_path: str, device: str = 'cuda'):
    """
    加载 PBR 环境贴图（使用 TRELLIS.2 的 EnvMap 类）。
    
    TRELLIS.2 的 EnvMap 类需要：
    1. 用 cv2.imread 读取 .exr/.hdr 文件
    2. 转换为 RGB 格式的 torch.Tensor
    3. 包装成 EnvMap 对象
    
    Args:
        envmap_path: 环境贴图路径（支持 .exr, .hdr 格式）
        device: 目标设备
    
    Returns:
        EnvMap: TRELLIS.2 的 EnvMap 对象，包含 shade() 和 sample() 方法
    """
    import os
    import cv2
    
    # 确保 OpenCV 可以读取 EXR 文件
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    
    # 导入 TRELLIS.2 的 EnvMap 类
    from trellis2.renderers import EnvMap
    
    # 用 cv2 读取 HDR/EXR 文件
    env_bgr = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)  # (H, W, 3) BGR
    if env_bgr is None:
        raise FileNotFoundError(f"无法加载环境贴图: {envmap_path}")
    
    # BGR -> RGB
    env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3)
    
    # 转换为 torch.Tensor
    env_tensor = torch.tensor(env_rgb, dtype=torch.float32, device=device)  # (H, W, 3)
    
    return EnvMap(env_tensor)


class PbrVoxelRenderer(VoxelRenderer):
    """
    PBR Voxel 渲染器（带 IBL 着色）。
    
    输出维度:
        shaded: (H, W, 3)
        normal: (H, W, 3)
        depth: (H, W)
        alpha: (H, W)
        mask: (H, W)
    
    使用示例:
        renderer = PbrVoxelRenderer(rendering_options, device='cuda')
        renderer.load_envmap('path/to/envmap.exr')
        out = renderer.render(voxel, extrinsics, intrinsics)
    """
    
    def __init__(self, rendering_options: Dict = None, device: str = 'cuda') -> None:
        super().__init__(rendering_options)
        self.device = device
        self.envmap = None  # 环境贴图，通过 load_envmap 加载
    
    def load_envmap(self, envmap_path: str) -> "PbrVoxelRenderer":
        """
        加载 PBR 环境贴图。
        
        Args:
            envmap_path: 环境贴图路径（支持 .exr, .hdr 格式）
        
        Returns:
            self: 支持链式调用
        """
        self.envmap = load_envmap(envmap_path, device=self.device)
        print(f"[PbrVoxelRenderer] 加载环境贴图: {envmap_path}")
        return self
    
    def render(
        self,
        voxel,
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
        envmap=None,
        colors_overwrite: torch.Tensor = None,
        shade: bool = True,
        use_envmap_bg: bool = False,
    ) -> edict:
        """
        渲染 Voxel 并进行 PBR 着色。
        
        Args:
            voxel: Voxel-like 对象（需要 position, attrs, voxel_size, layout）
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
            envmap: 环境贴图，None 时使用 self.envmap
            colors_overwrite: 覆盖 voxel attrs
            shade: 是否进行 PBR 着色
            use_envmap_bg: 是否使用环境贴图作为背景
        
        Returns:
            edict: {
                shaded: (H, W, 3),
                normal: (H, W, 3),
                depth: (H, W),
                alpha: (H, W),
                mask: (H, W),
                base_color: (H, W, 3),
                ...
            }
        """
        import utils3d
        
        H = W = self.rendering_options.resolution
        
        # 1. 基础渲染
        ret = super().render(voxel, extrinsics, intrinsics, colors_overwrite)
        
        depth = ret['depth']  # (H, W)
        alpha = ret['alpha']  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        
        # 2. 从 depth 估算 normal
        normal_cam = depth_to_normal(depth, intrinsics, mask)  # (H, W, 3)
        ret['normal'] = (-normal_cam * 0.5 + 0.5) * mask[..., None]  # (H, W, 3)，可视化格式
        ret['mask'] = mask  # (H, W)
        
        # 3. 使用 self.envmap 作为默认值
        if envmap is None:
            envmap = self.envmap
        
        # 4. 如果不需要着色或没有环境贴图，返回
        if not shade or envmap is None:
            return ret
        
        # 5. PBR 着色准备
        if not isinstance(envmap, dict):
            envmap = {'': envmap}
        
        # 获取属性，确保维度正确
        base_color = ret.get('base_color', torch.ones(H, W, 3, device=self.device) * 0.5)  # (H, W, 3)
        metallic = ret.get('metallic', torch.zeros(H, W, device=self.device))  # (H, W)
        roughness = ret.get('roughness', torch.ones(H, W, device=self.device) * 0.5)  # (H, W)
        
        # 确保 base_color 是 (H, W, 3)
        if base_color.dim() == 2:
            base_color = base_color[..., None].expand(H, W, 3)  # (H, W, 3)
        base_color = base_color.reshape(H, W, 3)  # (H, W, 3)
        
        # 确保 metallic/roughness 是 (H, W)
        metallic = metallic.reshape(H, W)  # (H, W)
        roughness = roughness.reshape(H, W)  # (H, W)
        
        # 获取射线
        rays_o, rays_d = utils3d.torch.get_image_rays(extrinsics, intrinsics, H, W)  # (H, W, 3)
        
        # 重建 3D 位置
        pos = rays_o + rays_d * depth[..., None]  # (H, W, 3)
        
        # 转换 normal 到世界空间
        R = extrinsics[:3, :3]  # (3, 3)
        normal_world = normal_cam @ R  # (H, W, 3)
        
        # sRGB -> Linear (clamp to avoid NaN from negative values)
        base_color_clamped = torch.clamp(base_color, 0.0, 1.0)  # (H, W, 3)
        base_color_linear = base_color_clamped ** 2.2  # (H, W, 3)
        
        # ORM 格式: (H, W, 3)
        orm = torch.stack([
            torch.zeros_like(metallic),  # Occlusion
            roughness,
            metallic,
        ], dim=-1)  # (H, W, 3)
        
        # 6. IBL 着色
        for name, env in envmap.items():
            shaded = env.shade(
                pos[None],              # (1, H, W, 3)
                normal_world[None],     # (1, H, W, 3)
                base_color_linear[None],# (1, H, W, 3)
                orm[None],              # (1, H, W, 3)
                rays_o,                 # (H, W, 3)
                specular=True,
            )[0]  # (H, W, 3)
            
            shaded = shaded.reshape(H, W, 3)  # 确保 (H, W, 3)
            shaded = shaded * mask[..., None]  # (H, W, 3)
            
            if use_envmap_bg:
                bg = env.sample(rays_d).reshape(H, W, 3)  # (H, W, 3)
                shaded = shaded + (1 - mask[..., None]) * bg  # (H, W, 3)
            
            # 后处理
            shaded = aces_tonemapping(shaded)  # (H, W, 3)
            shaded = gamma_correction(shaded)  # (H, W, 3)
            
            key = f"shaded_{name}" if name else "shaded"
            ret[key] = shaded  # (H, W, 3)
        
        return ret


class DiffVoxelRenderer(VoxelRenderer):
    """
    可微体素渲染器（近似版本）。
    
    渲染流程: VoxelProxy → o_voxel 渲染深度 → depth_to_normal → Normal
    
    梯度流: Loss → Normal → depth_to_normal → Depth → STE → opacities → Decoder
    
    注意: 使用 STE (Straight-Through Estimator) 建立梯度连接，
    因为 o_voxel 渲染器本身不可微。
    """
    
    def __init__(self, rendering_options: Dict = None, device: str = 'cuda') -> None:
        super().__init__(rendering_options)
        self.device = device
        self.bg_color = torch.tensor([0.5, 0.5, 1.0])  # 中性法线背景 (朝向相机)
    
    def _render_single(
        self,
        voxel_proxy: "VoxelProxy",
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
    ) -> edict:
        """
        渲染单个视角。
        
        Args:
            voxel_proxy: VoxelProxy 对象
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参
        
        Returns:
            edict: {normal: (H, W, 3), depth: (H, W), alpha: (H, W), mask: (H, W)}
        """
        import o_voxel
        
        H = W = self.rendering_options.resolution
        device = voxel_proxy.position.device
        bg = self.bg_color.to(device)
        
        # 过滤低不透明度体素（加速渲染）
        mask_visible = voxel_proxy.opacities > 0.01  # (N,)
        positions = voxel_proxy.position[mask_visible]  # (M, 3)
        opacities_visible = voxel_proxy.opacities[mask_visible]  # (M,)
        
        # 调用 o_voxel 渲染器
        attrs = torch.ones(positions.shape[0], 1, device=device)  # (M, 1)
        renderer = o_voxel.rasterize.VoxelRenderer(self.rendering_options)
        ret = renderer.render(positions, attrs, voxel_proxy.voxel_size, extrinsics, intrinsics)
        
        depth = ret['depth'].reshape(H, W)  # (H, W)
        alpha = ret['alpha'].reshape(H, W)  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        
        # Depth → Normal（可微）
        normal_cam = depth_to_normal(depth, intrinsics, mask)  # (H, W, 3)
        normal_vis = (-normal_cam * 0.5 + 0.5) * mask[..., None] + bg * (1 - mask[..., None])  # (H, W, 3)
        
        # STE: 建立 opacities 梯度连接（值不变，梯度流向 opacities）
        if voxel_proxy.opacities.requires_grad:
            mean_opacity = opacities_visible.mean()
            normal_vis = normal_vis + (mean_opacity - mean_opacity.detach()) * 0
        
        return edict(depth=depth, alpha=alpha, normal=normal_vis, mask=mask)
    
    def render_batch(
        self,
        voxel_proxy: "VoxelProxy",
        extrinsics: torch.Tensor,  # (B, V, 4, 4)
        intrinsics: torch.Tensor,  # (B, V, 3, 3)
    ) -> edict:
        """
        批量渲染多个视角。
        
        Args:
            voxel_proxy: VoxelProxy 对象（包含多个 batch）
            extrinsics: (B, V, 4, 4) 相机外参
            intrinsics: (B, V, 3, 3) 相机内参
        
        Returns:
            edict: {normal: (B, V, H, W, 3)}
        """
        B, V = extrinsics.shape[:2]
        unique_batches = voxel_proxy.batch_indices.unique().tolist()
        
        all_normals = []
        for b_idx, batch_id in enumerate(unique_batches):
            proxy_b = voxel_proxy.filter_by_batch(batch_id)
            view_normals = [
                self._render_single(proxy_b, extrinsics[b_idx, v], intrinsics[b_idx, v]).normal
                for v in range(V)
            ]  # List[(H, W, 3)]
            all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)
        
        return edict(normal=torch.stack(all_normals, dim=0))  # (B, V, H, W, 3)
