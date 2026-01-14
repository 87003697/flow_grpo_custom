"""
测试 diff_voxel_normal 模块

测试内容：
1. voxel_id 输出正确性（与 alpha > 0 的像素一致）
2. FDG 模式梯度流向 dual_vertices 和 intersected_logits
3. Sub 模式梯度流向 sub_logits
4. Normal 方向正确性（Camera Space，朝向相机）

运行方式：
    cd /home/zhiyuan_ma/code/flow_grpo_custom
    python scripts/test_diff_voxel_normal.py
"""
import sys
import os

# 添加项目路径
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
sys.path.insert(0, repo_root)
sys.path.insert(0, trellis2_ref_root)

import torch
import torch.nn.functional as F
import numpy as np


def test_voxel_id_output():
    """测试 1: voxel_id 输出正确性"""
    print("\n" + "=" * 60)
    print("测试 1: voxel_id 输出正确性")
    print("=" * 60)
    
    import o_voxel
    
    # 创建简单的测试数据：8 个 voxel 组成的立方体
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.1],
        [0.0, 0.1, 0.1],
        [0.1, 0.1, 0.1],
    ], dtype=torch.float32, device='cuda')  # (8, 3)
    
    attrs = torch.ones(8, 3, device='cuda', dtype=torch.float32)  # (8, 3)
    voxel_size = 0.1
    
    # 相机参数（从正面看）
    extrinsics = torch.eye(4, device='cuda', dtype=torch.float32)
    extrinsics[2, 3] = 2.0  # 相机在 z=2 位置
    
    intrinsics = torch.tensor([
        [500.0, 0.0, 0.5],
        [0.0, 500.0, 0.5],
        [0.0, 0.0, 1.0],
    ], device='cuda', dtype=torch.float32)
    
    renderer = o_voxel.rasterize.VoxelRenderer({
        "resolution": 64,
        "near": 0.1,
        "far": 10.0,
        "ssaa": 1,
    })
    
    ret = renderer.render(positions, attrs, voxel_size, extrinsics, intrinsics)
    
    voxel_id = ret['voxel_id']  # (H, W)
    alpha = ret['alpha']        # (H, W)
    
    print(f"  voxel_id shape: {voxel_id.shape}")
    print(f"  alpha shape: {alpha.shape}")
    print(f"  voxel_id dtype: {voxel_id.dtype}")
    print(f"  voxel_id range: [{voxel_id.min().item()}, {voxel_id.max().item()}]")
    
    # 验证：alpha > 0 的像素应该有有效的 voxel_id
    has_alpha = alpha > 0.5
    has_valid_id = voxel_id >= 0
    
    match_rate = (has_alpha == has_valid_id).float().mean().item()
    print(f"  alpha > 0.5 与 voxel_id >= 0 匹配率: {match_rate * 100:.2f}%")
    
    assert match_rate > 0.99, f"匹配率过低: {match_rate}"
    print("  ✓ voxel_id 输出正确!")
    
    return True


def test_fdg_gradient_flow():
    """测试 2: FDG 模式梯度流"""
    print("\n" + "=" * 60)
    print("测试 2: FDG 模式梯度流")
    print("=" * 60)
    
    from edit4shape.renderers.diff_voxel_normal import (
        RenderConfig, render_normal_fdg, _compute_axis_face_normals
    )
    
    # 创建 5x5x5 立方体网格（125 个 voxel），确保可见的 voxel 有完整邻居
    # 相机从 +Z 方向看过来，所以 Z 最大的面是可见的
    # 我们需要确保 z=17 面的 voxel 有邻居（z=16, z=18 层需要存在）
    coords_list = []
    for x in range(13, 18):  # 5 层: 13,14,15,16,17
        for y in range(13, 18):
            for z in range(13, 18):
                coords_list.append([x, y, z])
    coords = torch.tensor(coords_list, device='cuda', dtype=torch.int32)  # (125, 3)
    
    N = coords.shape[0]
    # 使用小的非零初始值，确保梯度不为零
    dual_vertices = torch.randn(N, 3, device='cuda', dtype=torch.float32) * 0.1
    dual_vertices.requires_grad_(True)
    intersected_logits = torch.randn(N, 3, device='cuda', dtype=torch.float32) * 0.1
    intersected_logits.requires_grad_(True)
    
    # 渲染配置：相机看向原点
    config = RenderConfig(
        intrinsics=torch.tensor([
            [500.0, 0.0, 0.5],
            [0.0, 500.0, 0.5],
            [0.0, 0.0, 1.0],
        ], device='cuda'),
        extrinsics=torch.eye(4, device='cuda'),
        resolution=64,
        voxel_size=1.0 / 32,
        origin=torch.tensor([-0.5, -0.5, -0.5], device='cuda'),
        grid_size=torch.tensor([32, 32, 32], device='cuda'),
    )
    config.extrinsics[2, 3] = 1.5  # 相机距离更近
    
    # 前向传播
    normal, mask = render_normal_fdg(coords, dual_vertices, intersected_logits, config)  # (H, W, 3), (H, W)
    
    print(f"  normal shape: {normal.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  前景像素数: {mask.sum().item()}")
    
    # 计算 loss 并反向传播
    loss = normal[mask].sum()
    loss.backward()
    
    # 检查梯度
    dual_grad = dual_vertices.grad
    intersected_grad = intersected_logits.grad
    
    print(f"  dual_vertices.grad: {'有梯度' if dual_grad is not None and dual_grad.abs().sum() > 0 else '无梯度'}")
    print(f"  intersected_logits.grad: {'有梯度' if intersected_grad is not None and intersected_grad.abs().sum() > 0 else '无梯度'}")
    
    if dual_grad is not None:
        print(f"    dual_grad 范围: [{dual_grad.min().item():.6f}, {dual_grad.max().item():.6f}]")
    if intersected_grad is not None:
        print(f"    intersected_grad 范围: [{intersected_grad.min().item():.6f}, {intersected_grad.max().item():.6f}]")
    
    has_dual_grad = dual_grad is not None and dual_grad.abs().sum() > 0
    has_intersected_grad = intersected_grad is not None and intersected_grad.abs().sum() > 0
    
    if has_dual_grad and has_intersected_grad:
        print("  ✓ FDG 模式梯度流正确!")
        return True
    else:
        print("  ✗ FDG 模式梯度流失败")
        return False


def test_sub_gradient_flow():
    """测试 3: Sub 模式梯度流"""
    print("\n" + "=" * 60)
    print("测试 3: Sub 模式梯度流")
    print("=" * 60)
    
    from edit4shape.renderers.diff_voxel_normal import (
        RenderConfig, render_normal_sub, _compute_occupancy_gradient
    )
    
    # 创建测试数据：在视野中心附近的 voxel
    N = 100
    # 坐标在 6-10 范围内，在 16x16 网格中会在中心附近
    sub_coords = torch.randint(6, 10, (N, 3), device='cuda', dtype=torch.int32)  # (N, 3)
    sub_logits = torch.randn(N, 8, device='cuda', dtype=torch.float32, requires_grad=True)  # (N, 8)
    
    # 渲染配置
    config = RenderConfig(
        intrinsics=torch.tensor([
            [500.0, 0.0, 0.5],
            [0.0, 500.0, 0.5],
            [0.0, 0.0, 1.0],
        ], device='cuda'),
        extrinsics=torch.eye(4, device='cuda'),
        resolution=64,
        voxel_size=1.0 / 16,
        origin=torch.tensor([-0.5, -0.5, -0.5], device='cuda'),
        grid_size=torch.tensor([16, 16, 16], device='cuda'),
    )
    config.extrinsics[2, 3] = 1.5  # 相机距离更近
    
    # 前向传播
    normal, mask = render_normal_sub(sub_coords, sub_logits, config)  # (H, W, 3), (H, W)
    
    print(f"  normal shape: {normal.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  前景像素数: {mask.sum().item()}")
    
    # 计算 loss 并反向传播
    loss = normal[mask].sum()
    loss.backward()
    
    # 检查梯度
    sub_grad = sub_logits.grad
    
    print(f"  sub_logits.grad: {'有梯度' if sub_grad is not None and sub_grad.abs().sum() > 0 else '无梯度'}")
    
    if sub_grad is not None:
        print(f"    sub_grad 范围: [{sub_grad.min().item():.6f}, {sub_grad.max().item():.6f}]")
    
    has_sub_grad = sub_grad is not None and sub_grad.abs().sum() > 0
    
    if has_sub_grad:
        print("  ✓ Sub 模式梯度流正确!")
        return True
    else:
        print("  ✗ Sub 模式梯度流失败")
        return False


def test_normal_direction():
    """测试 4: Normal 方向正确性"""
    print("\n" + "=" * 60)
    print("测试 4: Normal 方向正确性（Camera Space）")
    print("=" * 60)
    
    from edit4shape.renderers.diff_voxel_normal import (
        _compute_occupancy_gradient, _flip_normals_to_camera
    )
    
    # 测试 occupancy 梯度计算
    # 创建一个简单的测试用例：内部高、外部低
    sub_logits = torch.tensor([
        [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # 只有索引 0 被占用
        [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # 只有索引 1 被占用
    ], device='cuda', dtype=torch.float32)
    
    normals = _compute_occupancy_gradient(sub_logits)  # (2, 3)
    
    print(f"  occupancy 梯度计算:")
    print(f"    case 0 (占用位置 0): normal = {normals[0].tolist()}")
    print(f"    case 1 (占用位置 1): normal = {normals[1].tolist()}")
    
    # 测试翻转逻辑
    voxel_normals = torch.tensor([
        [0.0, 0.0, 1.0],   # 指向 +Z
        [0.0, 0.0, -1.0],  # 指向 -Z
    ], device='cuda', dtype=torch.float32)
    
    surface_pos = torch.tensor([
        [0.0, 0.0, 1.0],   # 在 +Z 方向
        [0.0, 0.0, 1.0],   # 在 +Z 方向
    ], device='cuda', dtype=torch.float32)
    
    # 相机在原点，看向 +Z
    extrinsics = torch.eye(4, device='cuda', dtype=torch.float32)
    
    normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, extrinsics)  # (2, 3)
    
    print(f"  翻转测试:")
    print(f"    原始 normal [0, 0, 1], pos [0, 0, 1] → 翻转后: {normals_cam[0].tolist()}")
    print(f"    原始 normal [0, 0, -1], pos [0, 0, 1] → 翻转后: {normals_cam[1].tolist()}")
    
    # 验证：翻转后的法线应该朝向相机（即 dot(normal, pos) <= 0）
    dot0 = (normals_cam[0] * surface_pos[0]).sum().item()
    dot1 = (normals_cam[1] * surface_pos[1]).sum().item()
    
    print(f"    dot(normal_cam[0], pos[0]) = {dot0:.4f} (应 <= 0)")
    print(f"    dot(normal_cam[1], pos[1]) = {dot1:.4f} (应 <= 0)")
    
    if dot0 <= 0 and dot1 <= 0:
        print("  ✓ Normal 方向正确（朝向相机）!")
        return True
    else:
        print("  ✗ Normal 方向错误")
        return False


def main():
    print("=" * 60)
    print("diff_voxel_normal 模块测试")
    print("=" * 60)
    
    results = []
    
    # 测试 1: voxel_id 输出
    try:
        results.append(("voxel_id 输出", test_voxel_id_output()))
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("voxel_id 输出", False))
    
    # 测试 2: FDG 梯度流
    try:
        results.append(("FDG 梯度流", test_fdg_gradient_flow()))
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FDG 梯度流", False))
    
    # 测试 3: Sub 梯度流
    try:
        results.append(("Sub 梯度流", test_sub_gradient_flow()))
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Sub 梯度流", False))
    
    # 测试 4: Normal 方向
    try:
        results.append(("Normal 方向", test_normal_direction()))
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Normal 方向", False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败!")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
