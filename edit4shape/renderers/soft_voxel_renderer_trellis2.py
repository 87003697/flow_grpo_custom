"""
纯 PyTorch 可微体素渲染器（极简版）

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

使用示例：
    from edit4shape.renderers.soft_voxel_renderer_trellis2 import SoftVoxelRenderer
    
    renderer = SoftVoxelRenderer(resolution=512)
    out = renderer.render(voxel_proxy, extrinsics, intrinsics)
    # out.depth: (H, W), out.alpha: (H, W), out.normal: (H, W, 3)
"""

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from typing import Optional


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
    M = positions_cam.shape[0]  # noqa: F841
    
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
    # 深度越小（越近），权重越大
    # 使用相对深度，避免 exp(-large_number) ≈ 0
    z_min = z.min()
    z_relative = z - z_min  # 相对深度，最小值为 0
    depth_weights = torch.exp(-z_relative * temperature)  # (M,) 近处权重大
    weighted_opacity = depth_weights * opacities_valid  # (M,)
    
    # 7. Scatter 累积到像素
    dtype = opacities_valid.dtype  # 确保 dtype 一致
    depth_sum = torch.zeros(H * W, device=device, dtype=dtype)   # (H*W,)
    weight_sum = torch.zeros(H * W, device=device, dtype=dtype)  # (H*W,)
    alpha_sum = torch.zeros(H * W, device=device, dtype=dtype)   # (H*W,)
    
    # depth * weight
    depth_sum.scatter_add_(0, pixel_idx, (weighted_opacity * z).to(dtype))  # (H*W,)
    weight_sum.scatter_add_(0, pixel_idx, weighted_opacity.to(dtype))       # (H*W,)
    alpha_sum.scatter_add_(0, pixel_idx, opacities_valid.to(dtype))         # (H*W,)
    
    # 8. 归一化
    depth = (depth_sum / (weight_sum + 1e-8)).reshape(H, W)  # (H, W)
    alpha = alpha_sum.clamp(0, 1).reshape(H, W)              # (H, W)
    
    return {'depth': depth, 'alpha': alpha}


def depth_to_normal(
    depth: torch.Tensor,       # (H, W)
    intrinsics: torch.Tensor,  # (3, 3)
    mask: Optional[torch.Tensor] = None,  # (H, W)
    use_soft_mask: bool = True,  # 是否使用软掩码（保持梯度）
) -> torch.Tensor:
    """
    从 depth 图估算相机空间法线。
    
    Args:
        depth: (H, W) 深度图
        intrinsics: (3, 3) 相机内参（归一化）
        mask: (H, W) 可选的掩码
        use_soft_mask: 是否使用软掩码（保持梯度）
    
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
    x = (x_grid / W - cx) * z / fx  # (H, W)
    y = (y_grid / H - cy) * z / fy  # (H, W)
    
    # 中心差分计算梯度（使用切片赋值保持梯度）
    # dx: 沿 x 方向的变化
    dx_x = (x[:, 2:] - x[:, :-2]) / 2  # (H, W-2)
    dx_y = (y[:, 2:] - y[:, :-2]) / 2  # (H, W-2)
    dx_z = (z[:, 2:] - z[:, :-2]) / 2  # (H, W-2)
    
    # dy: 沿 y 方向的变化
    dy_x = (x[2:, :] - x[:-2, :]) / 2  # (H-2, W)
    dy_y = (y[2:, :] - y[:-2, :]) / 2  # (H-2, W)
    dy_z = (z[2:, :] - z[:-2, :]) / 2  # (H-2, W)
    
    # 只在有效区域计算（中间 H-2 x W-2）
    dx = torch.stack([dx_x[1:-1, :], dx_y[1:-1, :], dx_z[1:-1, :]], dim=-1)  # (H-2, W-2, 3)
    dy = torch.stack([dy_x[:, 1:-1], dy_y[:, 1:-1], dy_z[:, 1:-1]], dim=-1)  # (H-2, W-2, 3)
    
    # 法线 = dy × dx
    normal_inner = torch.linalg.cross(dy, dx)  # (H-2, W-2, 3)
    normal_inner = F.normalize(normal_inner, dim=-1, eps=1e-6)  # (H-2, W-2, 3)
    
    # Pad 回原始大小
    normal = F.pad(normal_inner, (0, 0, 1, 1, 1, 1), mode='constant', value=0)  # (H, W, 3)
    
    # 确保法线朝向相机（z < 0）
    flip_mask = (normal[..., 2:3] > 0).float()  # (H, W, 1)
    normal = normal * (1 - 2 * flip_mask)  # 翻转 z > 0 的法线
    
    if mask is not None:
        mask = mask.reshape(H, W)
        if use_soft_mask:
            # 软掩码：保持梯度
            normal = normal * mask[..., None]
        else:
            # 硬掩码：可能阻断梯度
            normal = normal * mask[..., None].detach()
    
    return normal  # (H, W, 3)


class SoftVoxelRenderer:
    """
    纯 PyTorch 可微体素渲染器。
    
    输出维度:
        depth: (H, W)
        alpha: (H, W)
        normal: (H, W, 3)
        mask: (H, W)
    """
    
    def __init__(self, resolution: int = 512, temperature: float = 50.0):
        self.resolution = resolution
        self.temperature = temperature
        self.bg_color = torch.tensor([0.5, 0.5, 1.0])  # 中性法线背景
    
    def render(
        self,
        voxel_proxy,  # VoxelProxy 对象
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
    ) -> edict:
        """
        渲染 VoxelProxy。
        
        Args:
            voxel_proxy: VoxelProxy 对象
            extrinsics: (4, 4) W2C 相机外参
            intrinsics: (3, 3) 相机内参（归一化）
        
        Returns:
            edict: {depth, alpha, normal, mask}
        """
        H = W = self.resolution
        device = voxel_proxy.position.device
        bg = self.bg_color.to(device)
        
        # 过滤低不透明度体素
        visible_mask = voxel_proxy.opacities > 0.01  # (N,)
        positions = voxel_proxy.position[visible_mask]  # (M, 3)
        opacities = voxel_proxy.opacities[visible_mask]  # (M,)
        
        # 渲染
        out = soft_voxel_render(
            positions, opacities, extrinsics, intrinsics,
            H, W, self.temperature
        )
        
        depth = out['depth']  # (H, W)
        alpha = out['alpha']  # (H, W)
        mask = (alpha > 0.5).float()  # (H, W)
        
        # Depth → Normal
        normal_cam = depth_to_normal(depth, intrinsics, mask)  # (H, W, 3)
        normal_vis = (-normal_cam * 0.5 + 0.5) * mask[..., None] + bg * (1 - mask[..., None])  # (H, W, 3)
        
        return edict(
            depth=depth,
            alpha=alpha,
            normal=normal_vis,
            mask=mask,
        )


# ============ 多尺度 Occupancy 监督 ============

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
    
    # 8 个子 voxel 的偏移（相对于父 voxel，[0,1] 内）
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], device=device, dtype=torch.float32) * 0.5  # (8, 3)
    
    # AABB = [-0.5, 0.5]^3
    voxel_size = 1.0 / parent_resolution
    child_size = voxel_size / 2
    
    # 父 voxel 角点世界坐标
    parent_origin = parent_coords.float() * voxel_size - 0.5  # (N, 3)
    
    # 子 voxel 中心 = 父角点 + 偏移 * 父尺寸 + 子尺寸/2
    positions = (
        parent_origin.unsqueeze(1) +        # (N, 1, 3)
        offsets.unsqueeze(0) * voxel_size + # (1, 8, 3)
        child_size / 2
    ).reshape(-1, 3)  # (N*8, 3)
    
    occupancies = torch.sigmoid(sub_logits).reshape(-1)  # (N*8,)
    
    return positions, occupancies


def multiscale_occupancy_loss(
    subs: list,                   # List[SparseTensor]
    extrinsics: torch.Tensor,     # (4, 4)
    intrinsics: torch.Tensor,     # (3, 3)
    target_alpha: torch.Tensor,   # (H, W)
    base_resolution: int = 64,
    max_render_size: int = 256,
    temperature: float = 50.0,
) -> torch.Tensor:
    """
    多尺度 occupancy 监督（Soft Z-buffer）。
    
    Args:
        subs: Decoder 各层 subdivision，每个 sub.coords (N, 4), sub.feats (N, 8)
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
        # 该层分辨率
        parent_res = base_resolution * (2 ** i)
        render_size = min(parent_res * 2, max_render_size)
        
        # 展开 subdivision
        coords = sub.coords[:, 1:] if sub.coords.shape[1] == 4 else sub.coords  # 去 batch 索引
        positions, occupancies = expand_subdivision_to_voxels(coords, sub.feats, parent_res)
        
        # 渲染
        out = soft_voxel_render(positions, occupancies, extrinsics, intrinsics, render_size, render_size, temperature)
        
        # 下采样 target
        target_i = F.interpolate(
            target_alpha.unsqueeze(0).unsqueeze(0),
            size=(render_size, render_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        # 层 loss + 层权重
        layer_loss = F.mse_loss(out['alpha'], target_i)
        layer_weight = 2 ** i
        total_loss += layer_weight * layer_loss
        weight_sum += layer_weight
    
    return total_loss / weight_sum if weight_sum > 0 else total_loss


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试纯 PyTorch 可微体素渲染器")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 模拟输入 - 使用 leaf tensor
    # 生成一个密集的体素网格，确保渲染结果不稀疏
    N = 5000
    # 在屏幕中心生成体素，确保它们会被渲染到
    positions_raw = torch.randn(N, 3, device=device) * 0.2  # 更集中
    positions_raw.requires_grad_(True)
    
    opacities_logits = torch.randn(N, device=device)  # (N,)
    opacities_logits.requires_grad_(True)
    
    # 通过可微操作生成实际输入
    positions = positions_raw  # 直接使用 leaf tensor
    opacities = torch.sigmoid(opacities_logits)  # (N,)
    
    print(f"体素数量: {N}")
    print(f"positions shape: {positions.shape}, is_leaf: {positions.is_leaf}")
    print(f"opacities_logits shape: {opacities_logits.shape}, is_leaf: {opacities_logits.is_leaf}")
    
    # 相机参数
    extrinsics = torch.eye(4, device=device)
    extrinsics[2, 3] = 2.0  # 相机在 z=2 处看向原点
    
    intrinsics = torch.tensor([
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)
    
    print(f"\n相机外参:\n{extrinsics}")
    print(f"\n相机内参:\n{intrinsics}")
    
    # 渲染
    print("\n" + "-" * 40)
    print("渲染中...")
    out = soft_voxel_render(positions, opacities, extrinsics, intrinsics, H=128, W=128)
    
    print(f"depth shape: {out['depth'].shape}")
    print(f"alpha shape: {out['alpha'].shape}")
    print(f"depth 非零像素: {(out['depth'] > 0).sum().item()}")
    print(f"alpha 非零像素: {(out['alpha'] > 0).sum().item()}")
    
    # 测试梯度 - 使用 torch.autograd.grad
    print("\n" + "-" * 40)
    print("测试梯度回传...")
    loss = out['depth'].sum() + out['alpha'].sum()
    
    # 使用 autograd.grad 检查梯度
    grads = torch.autograd.grad(
        loss, 
        [positions_raw, opacities_logits], 
        retain_graph=True,
        allow_unused=True
    )
    grad_positions, grad_opacities = grads
    
    print(f"grad_positions is None: {grad_positions is None}")
    print(f"grad_opacities is None: {grad_opacities is None}")
    
    if grad_positions is not None:
        print(f"grad_positions norm: {grad_positions.norm().item():.6f}")
        print(f"grad_positions 非零元素: {(grad_positions.abs() > 1e-8).sum().item()}")
    
    if grad_opacities is not None:
        print(f"grad_opacities norm: {grad_opacities.norm().item():.6f}")
        print(f"grad_opacities 非零元素: {(grad_opacities.abs() > 1e-8).sum().item()}")
    
    # 测试 depth_to_normal
    print("\n" + "-" * 40)
    print("测试 depth_to_normal + 完整梯度链...")
    
    positions_raw2 = torch.randn(N, 3, device=device) * 0.3
    positions_raw2.requires_grad_(True)
    opacities_logits2 = torch.randn(N, device=device)
    opacities_logits2.requires_grad_(True)
    
    opacities2 = torch.sigmoid(opacities_logits2)
    
    out2 = soft_voxel_render(positions_raw2, opacities2, extrinsics, intrinsics, H=128, W=128)
    
    # 使用 soft mask（alpha 本身），而不是 hard mask（alpha > 0.5）
    # 这样可以保持梯度连续
    soft_mask = out2['alpha']  # 直接用 alpha 作为软掩码
    normal = depth_to_normal(out2['depth'], intrinsics, soft_mask, use_soft_mask=True)
    
    print(f"normal shape: {normal.shape}")
    print(f"normal 非零像素: {(normal.abs().sum(dim=-1) > 1e-6).sum().item()}")
    
    loss2 = normal.sum()
    grads2 = torch.autograd.grad(
        loss2,
        [positions_raw2, opacities_logits2],
        allow_unused=True
    )
    grad_pos2, grad_opa2 = grads2
    
    print(f"grad_positions2 is None: {grad_pos2 is None}")
    print(f"grad_opacities2 is None: {grad_opa2 is None}")
    
    if grad_pos2 is not None:
        print(f"grad_positions2 norm: {grad_pos2.norm().item():.6f}")
        print(f"grad_positions2 非零元素: {(grad_pos2.abs() > 1e-10).sum().item()}")
    if grad_opa2 is not None:
        print(f"grad_opacities2 norm: {grad_opa2.norm().item():.6f}")
        print(f"grad_opacities2 非零元素: {(grad_opa2.abs() > 1e-10).sum().item()}")
    
    # 额外测试：简化版梯度测试
    print("\n" + "-" * 40)
    print("简化版梯度测试（直接测试 z → depth）...")
    
    # 最简单的场景：只有几个点，确保都在屏幕内
    z_test = torch.tensor([1.5, 1.8, 2.0], device=device, requires_grad=True)  # 相机空间 z
    positions_test = torch.stack([
        torch.zeros(3, device=device),  # x
        torch.zeros(3, device=device),  # y
        z_test - 2.0,  # 世界空间 z = 相机空间 z - 2
    ], dim=-1)  # (3, 3)
    positions_test = positions_test.requires_grad_(True)
    opacities_test = torch.ones(3, device=device, requires_grad=True)
    
    out_test = soft_voxel_render(
        positions_test, opacities_test, extrinsics, intrinsics, 
        H=16, W=16, temperature=10.0
    )
    
    print(f"depth 非零: {(out_test['depth'] > 0).sum().item()}")
    print(f"depth max: {out_test['depth'].max().item():.4f}")
    
    loss_test = out_test['depth'].sum()
    grads_test = torch.autograd.grad(loss_test, [positions_test, opacities_test])
    
    print(f"positions_test grad norm: {grads_test[0].norm().item():.6f}")
    print(f"opacities_test grad norm: {grads_test[1].norm().item():.6f}")
    print(f"positions_test grad:\n{grads_test[0]}")
    
    # 总结
    print("\n" + "=" * 60)
    all_pass = True
    
    # 检查 opacities 梯度（这个应该一直工作）
    if grad_opacities is None or grad_opacities.norm() < 1e-10:
        print("❌ opacities 梯度失败")
        all_pass = False
    else:
        print("✅ opacities 梯度正常")
    
    # positions 梯度：由于使用整数索引，只有 z 分量有梯度
    if grads_test[0][:, 2].abs().sum() > 1e-6:
        print("✅ positions (z 分量) 梯度正常")
    else:
        print("❌ positions (z 分量) 梯度失败")
        all_pass = False
    
    # x, y 分量不应该有梯度（它们只影响 pixel_idx）
    if grads_test[0][:, :2].abs().sum() < 1e-6:
        print("ℹ️ positions (x, y 分量) 无梯度 - 这是预期行为")
    
    if all_pass:
        print("\n🎉 核心梯度测试通过！")
        print("注意：positions 的 x, y 分量没有梯度是预期的（只影响像素索引）")
        print("注意：depth_to_normal 需要密集的深度图才能有效工作")
    else:
        print("\n⚠️ 部分测试失败")
    print("=" * 60)

