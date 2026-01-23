"""
纯 PyTorch 可微体素渲染器

重构版：继承 BaseRenderer，遵循 7 阶段渲染流水线。

核心原理：Soft Z-buffer
1. 投影体素到屏幕
2. 用 scatter_add 累积到像素
3. 用 exp(-depth) 加权处理遮挡

梯度特性：
- opacities: ✅ 完全可微
- positions.z: ✅ 可微（影响深度权重和深度值）
- positions.x/y: ❌ 不可微（只影响像素索引，使用整数操作）

限制：
- 不处理精确的 ray-box intersection
- 每个体素只影响 1 个像素（最近的）
- 速度慢，仅用于验证梯度流
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F

from edit4shape.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    depth_to_normal,
    camera_normal_to_vis,
)
from edit4shape.renderers.ovoxel_trellis2 import VoxelProxy


# ============================================================================
# 核心渲染函数
# ============================================================================

def soft_voxel_render(
    positions: torch.Tensor,   # (N, 3) 体素位置
    opacities: torch.Tensor,   # (N,) 体素不透明度
    extrinsics: torch.Tensor,  # (4, 4) W2C 相机外参
    intrinsics: torch.Tensor,  # (3, 3) 相机内参（归一化到 [0,1]）
    H: int = 512,
    W: int = 512,
    temperature: float = 50.0,  # 控制软硬程度
) -> dict:
    """
    极简可微体素渲染。
    
    Args:
        positions: (N, 3) 体素世界坐标
        opacities: (N,) 体素不透明度 [0, 1]
        extrinsics: (4, 4) W2C 相机外参
        intrinsics: (3, 3) 相机内参（归一化，fx/fy/cx/cy 在 [0,1]）
        H, W: 输出图像尺寸
        temperature: 深度加权温度，越大越硬
    
    Returns:
        dict: {depth: (H, W), alpha: (H, W)}
    """
    device = positions.device
    N = positions.shape[0]
    
    if N == 0:
        return {
            'depth': torch.zeros(H, W, device=device),
            'alpha': torch.zeros(H, W, device=device),
        }
    
    # 1. 世界坐标 → 相机坐标
    ones = torch.ones(N, 1, device=device)  # (N, 1)
    positions_homo = torch.cat([positions, ones], dim=-1)  # (N, 4)
    positions_cam = (extrinsics @ positions_homo.T).T[:, :3]  # (N, 3)
    
    # 2. 过滤掉相机后面的点
    z = positions_cam[:, 2]  # (N,)
    valid_mask = z > 0.01  # (N,)
    
    if valid_mask.sum() == 0:
        return {
            'depth': torch.zeros(H, W, device=device),
            'alpha': torch.zeros(H, W, device=device),
        }
    
    positions_cam = positions_cam[valid_mask]  # (M, 3)
    opacities_valid = opacities[valid_mask]    # (M,)
    z = z[valid_mask]                          # (M,)
    
    # 3. 相机坐标 → 像素坐标
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_ndc = positions_cam[:, 0] / z  # (M,)
    y_ndc = positions_cam[:, 1] / z  # (M,)
    
    x_pix = (x_ndc * fx + cx) * W  # (M,)
    y_pix = (y_ndc * fy + cy) * H  # (M,)
    
    # 4. 过滤屏幕外的点
    in_screen = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
    if in_screen.sum() == 0:
        return {
            'depth': torch.zeros(H, W, device=device),
            'alpha': torch.zeros(H, W, device=device),
        }
    
    x_pix = x_pix[in_screen]
    y_pix = y_pix[in_screen]
    z = z[in_screen]
    opacities_valid = opacities_valid[in_screen]
    
    # 5. 量化到像素索引
    x_idx = x_pix.long().clamp(0, W - 1)  # (M,)
    y_idx = y_pix.long().clamp(0, H - 1)  # (M,)
    pixel_idx = y_idx * W + x_idx         # (M,)
    
    # 6. Soft Z-buffer: 用 softmax 处理同一像素的多个体素
    z_min = z.min()
    z_relative = z - z_min  # 相对深度，最小值为 0
    depth_weights = torch.exp(-z_relative * temperature)  # (M,) 近处权重大
    weighted_opacity = depth_weights * opacities_valid  # (M,)
    
    # 7. Scatter 累积到像素
    dtype = opacities_valid.dtype
    depth_sum = torch.zeros(H * W, device=device, dtype=dtype)   # (H*W,)
    weight_sum = torch.zeros(H * W, device=device, dtype=dtype)  # (H*W,)
    alpha_sum = torch.zeros(H * W, device=device, dtype=dtype)   # (H*W,)
    
    depth_sum.scatter_add_(0, pixel_idx, (weighted_opacity * z).to(dtype))  # (H*W,)
    weight_sum.scatter_add_(0, pixel_idx, weighted_opacity.to(dtype))       # (H*W,)
    alpha_sum.scatter_add_(0, pixel_idx, opacities_valid.to(dtype))         # (H*W,)
    
    # 8. 归一化
    depth = (depth_sum / (weight_sum + 1e-8)).reshape(H, W)  # (H, W)
    alpha = alpha_sum.clamp(0, 1).reshape(H, W)              # (H, W)
    
    return {'depth': depth, 'alpha': alpha}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SoftVoxelRasterData:
    """
    Soft Voxel 光栅化中间数据
    """
    positions: torch.Tensor   # (N, 3)
    opacities: torch.Tensor   # (N,)


# ============================================================================
# SoftVoxelRenderer
# ============================================================================

class SoftVoxelRenderer(BaseRenderer):
    """
    纯 PyTorch 可微体素渲染器
    
    继承 BaseRenderer，实现 7 阶段渲染流水线:
        Stage 1: prepare_inputs - 检查空体素
        Stage 2: compute_camera_data - 计算相机参数
        Stage 3: process_geometry - 过滤低透明度体素
        Stage 4: rasterize_core - Soft Z-buffer 光栅化
        Stage 5: interpolate_attributes - 计算 depth/alpha
        Stage 6: post_process - depth → normal
        Stage 7: assemble_output - 组装 RenderOutput
    
    输出维度:
        depth: (H, W)
        alpha: (H, W)
        normal: (H, W, 3)
        mask: (H, W)
    
    使用示例:
        config = RenderConfig(resolution=512)
        renderer = SoftVoxelRenderer(config, temperature=50.0)
        output = renderer.render(voxel_proxy, extrinsics, intrinsics)
    """
    
    def __init__(
        self,
        config: RenderConfig = None,
        device: str = 'cuda',
        temperature: float = 50.0,
    ):
        """
        Args:
            config: 渲染配置
            device: 计算设备
            temperature: Soft Z-buffer 温度参数，越大越硬
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1)
        super().__init__(config, device)
        
        self.temperature = temperature
    
    # ========== Stage 1: Input Preparation ==========
    
    def _is_empty_geometry(self, geometry: VoxelProxy) -> bool:
        """检查体素是否为空"""
        return geometry.position.shape[0] == 0
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(
        self,
        geometry: VoxelProxy,
        camera_data: CameraData,
    ) -> SoftVoxelRasterData:
        """
        过滤低透明度体素
        """
        # 过滤低不透明度体素（加速渲染）
        visible_mask = geometry.opacities > 0.01  # (N,)
        positions = geometry.position[visible_mask]  # (M, 3)
        opacities = geometry.opacities[visible_mask]  # (M,)
        
        return SoftVoxelRasterData(
            positions=positions,
            opacities=opacities,
        )
    
    # ========== Stage 4: Rasterization Core ==========
    
    def _rasterize_core(
        self,
        processed_geometry: SoftVoxelRasterData,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        Soft Z-buffer 光栅化
        """
        H = W = self.config.resolution
        
        out = soft_voxel_render(
            processed_geometry.positions,
            processed_geometry.opacities,
            camera_data.extrinsics,
            camera_data.intrinsics,
            H, W,
            self.temperature,
        )
        
        return RasterOutput(
            rast=out,
            depth_buffer=out['depth'],
            primitive_id=torch.zeros(H, W, device=self.device, dtype=torch.long),
        )
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: VoxelProxy,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        提取 depth 和 alpha
        """
        rast = raster_output.rast
        
        return {
            'depth': rast['depth'],  # (H, W)
            'alpha': rast['alpha'],  # (H, W)
        }
    
    # ========== Stage 6: Post-processing ==========
    
    def _post_process(
        self,
        attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """
        Depth → Normal
        """
        depth = attrs['depth']  # (H, W)
        alpha = attrs['alpha']  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        attrs['mask'] = mask
        
        # Depth → Normal
        normal_cam = depth_to_normal(depth, camera_data.intrinsics, mask)  # (H, W, 3)
        attrs['normal'] = camera_normal_to_vis(normal_cam, mask)  # (H, W, 3)
        
        return attrs


# ============================================================================
# 多尺度 Occupancy 监督
# ============================================================================

def expand_subdivision_to_voxels(
    parent_coords: torch.Tensor,  # (N, 3) 父 voxel 整数坐标
    sub_logits: torch.Tensor,     # (N, 8) 子 voxel logits
    parent_resolution: int,       # 父层分辨率
) -> tuple:
    """
    将 subdivision 展开成子 voxel 列表。
    
    Args:
        parent_coords: (N, 3) 父 voxel 整数坐标
        sub_logits: (N, 8) 8 个子 voxel 的 logits
        parent_resolution: 父层网格分辨率
    
    Returns:
        positions: (N*8, 3) 子 voxel 世界坐标
        occupancies: (N*8,) 子 voxel 占用概率
    """
    device = parent_coords.device
    
    # 8 个子 voxel 的偏移
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], device=device, dtype=torch.float32) * 0.5  # (8, 3)
    
    voxel_size = 1.0 / parent_resolution
    child_size = voxel_size / 2
    
    # 父 voxel 角点世界坐标
    parent_origin = parent_coords.float() * voxel_size - 0.5  # (N, 3)
    
    # 子 voxel 中心
    positions = (
        parent_origin.unsqueeze(1) +
        offsets.unsqueeze(0) * voxel_size +
        child_size / 2
    ).reshape(-1, 3)  # (N*8, 3)
    
    occupancies = torch.sigmoid(sub_logits).reshape(-1)  # (N*8,)
    
    return positions, occupancies


def multiscale_occupancy_loss(
    subs: list,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    target_alpha: torch.Tensor,
    base_resolution: int = 64,
    max_render_size: int = 256,
    temperature: float = 50.0,
) -> torch.Tensor:
    """
    多尺度 occupancy 监督（Soft Z-buffer）。
    
    Args:
        subs: Decoder 各层 subdivision
        extrinsics: (4, 4) W2C 相机外参
        intrinsics: (3, 3) 归一化相机内参
        target_alpha: (H, W) 目标 alpha
        base_resolution: 第 0 层父分辨率
        max_render_size: 渲染尺寸上限
        temperature: Soft Z-buffer 温度
    
    Returns:
        loss: 加权平均 loss
    """
    total_loss = 0.0
    weight_sum = 0.0
    
    for i, sub in enumerate(subs):
        parent_res = base_resolution * (2 ** i)
        render_size = min(parent_res * 2, max_render_size)
        
        coords = sub.coords[:, 1:] if sub.coords.shape[1] == 4 else sub.coords
        positions, occupancies = expand_subdivision_to_voxels(coords, sub.feats, parent_res)
        
        out = soft_voxel_render(
            positions, occupancies, extrinsics, intrinsics,
            render_size, render_size, temperature
        )
        
        target_i = F.interpolate(
            target_alpha.unsqueeze(0).unsqueeze(0),
            size=(render_size, render_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        layer_loss = F.mse_loss(out['alpha'], target_i)
        layer_weight = 2 ** i
        total_loss += layer_weight * layer_loss
        weight_sum += layer_weight
    
    return total_loss / weight_sum if weight_sum > 0 else total_loss


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试纯 PyTorch 可微体素渲染器")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 模拟输入
    N = 5000
    positions_raw = torch.randn(N, 3, device=device) * 0.2
    positions_raw.requires_grad_(True)
    
    opacities_logits = torch.randn(N, device=device)
    opacities_logits.requires_grad_(True)
    
    positions = positions_raw
    opacities = torch.sigmoid(opacities_logits)
    
    # 相机参数
    extrinsics = torch.eye(4, device=device)
    extrinsics[2, 3] = 2.0
    
    intrinsics = torch.tensor([
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)
    
    # 渲染
    print("\n渲染中...")
    out = soft_voxel_render(positions, opacities, extrinsics, intrinsics, H=128, W=128)
    
    print(f"depth shape: {out['depth'].shape}")
    print(f"alpha shape: {out['alpha'].shape}")
    
    # 测试梯度
    loss = out['depth'].sum() + out['alpha'].sum()
    grads = torch.autograd.grad(
        loss,
        [positions_raw, opacities_logits],
        retain_graph=True,
        allow_unused=True
    )
    
    print(f"grad_positions is None: {grads[0] is None}")
    print(f"grad_opacities is None: {grads[1] is None}")
    
    if grads[1] is not None:
        print(f"grad_opacities norm: {grads[1].norm().item():.6f}")
    
    print("\n🎉 测试完成!")
