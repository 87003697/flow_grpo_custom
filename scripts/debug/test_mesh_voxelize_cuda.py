#!/usr/bin/env python3
"""
测试 GPU mesh_to_voxel_indices_cuda 与 CPU mesh_to_flexible_dual_grid 的一致性和速度对比。

用法:
    CUDA_VISIBLE_DEVICES=7 python scripts/debug/test_mesh_voxelize_cuda.py
"""

import os, sys, time
import torch
import numpy as np

# ---------- 环境设置 ----------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, repo_root)

trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)


def make_test_mesh_sphere(n_subdivisions=4, device="cuda"):
    """创建一个 icosphere 作为测试 mesh。"""
    import math
    # 正二十面体
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ]
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # 归一化到单位球
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Loop subdivision
    for _ in range(n_subdivisions):
        edge_midpoint = {}
        new_verts = list(verts)
        new_faces = []
        for f in faces:
            mid_indices = []
            for i in range(3):
                e = tuple(sorted((f[i], f[(i+1)%3])))
                if e not in edge_midpoint:
                    mp = (verts[e[0]] + verts[e[1]]) / 2
                    mp = mp / np.linalg.norm(mp)  # 投影到球面
                    idx = len(new_verts)
                    new_verts.append(mp)
                    edge_midpoint[e] = idx
                mid_indices.append(edge_midpoint[e])
            a, b, c = f
            m0, m1, m2 = mid_indices
            new_faces.extend([
                [a, m0, m2], [b, m1, m0], [c, m2, m1], [m0, m1, m2],
            ])
        verts = np.array(new_verts, dtype=np.float32)
        faces = np.array(new_faces, dtype=np.int32)

    # 缩放到 [-0.4, 0.4] 区间（在 [-0.5, 0.5] AABB 内）
    verts *= 0.4

    V, F_count = verts.shape[0], faces.shape[0]
    verts_t = torch.from_numpy(verts).to(device)
    faces_t = torch.from_numpy(faces).to(device)
    return verts_t, faces_t, V, F_count


def run_cpu_version(vertices_cpu, faces_cpu, grid_size):
    """运行 CPU 版本 mesh_to_flexible_dual_grid。"""
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
    voxel_indices, _dual_vertices, _intersected = mesh_to_flexible_dual_grid(
        vertices_cpu.float(),
        faces_cpu.long(),
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    )
    return voxel_indices  # (N, 3) int on CPU


def run_gpu_version(vertices_cuda, faces_cuda, grid_size):
    """运行 GPU 版本 mesh_to_voxel_indices_cuda。"""
    from edit4shape.generators.trellis2.ops.mesh_voxelize import mesh_to_voxel_indices_cuda
    voxel_indices = mesh_to_voxel_indices_cuda(
        vertices_cuda.float(),
        faces_cuda.int(),
        grid_size=grid_size,
    )
    return voxel_indices  # (N, 3) int32 on CUDA


def voxel_set(indices_tensor):
    """将 (N, 3) tensor 转为 set of tuples 用于集合比较。"""
    arr = indices_tensor.cpu().numpy()
    return set(map(tuple, arr))


def main():
    device = "cuda"
    grid_size = 1024

    print("=" * 70)
    print("  GPU mesh_to_voxel_indices_cuda vs CPU mesh_to_flexible_dual_grid")
    print("=" * 70)

    # ---- 构建测试 mesh ----
    for subdiv in [3, 4, 5]:
        verts_cuda, faces_cuda, V, F_count = make_test_mesh_sphere(
            n_subdivisions=subdiv, device=device)
        verts_cpu = verts_cuda.cpu()
        faces_cpu = faces_cuda.cpu()

        print(f"\n{'─' * 60}")
        print(f"  Icosphere subdiv={subdiv}: V={V}, F={F_count}")
        print(f"{'─' * 60}")

        # ---- GPU 版本 ----
        # warmup (JIT compile + first run)
        _ = run_gpu_version(verts_cuda, faces_cuda, grid_size)
        torch.cuda.synchronize()

        n_runs = 10
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            gpu_result = run_gpu_version(verts_cuda, faces_cuda, grid_size)
            torch.cuda.synchronize()
        t_gpu = (time.perf_counter() - t0) / n_runs
        print(f"  GPU: {gpu_result.shape[0]:>8d} voxels, {t_gpu*1000:.2f} ms")

        # ---- CPU 版本 ----
        # warmup
        _ = run_cpu_version(verts_cpu, faces_cpu, grid_size)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            cpu_result = run_cpu_version(verts_cpu, faces_cpu, grid_size)
        t_cpu = (time.perf_counter() - t0) / n_runs
        print(f"  CPU: {cpu_result.shape[0]:>8d} voxels, {t_cpu*1000:.2f} ms")

        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        print(f"  Speedup: {speedup:.1f}x")

        # ---- 一致性检查 ----
        gpu_set = voxel_set(gpu_result)
        cpu_set = voxel_set(cpu_result)

        common = gpu_set & cpu_set
        gpu_only = gpu_set - cpu_set
        cpu_only = cpu_set - gpu_set

        print(f"\n  [一致性]")
        print(f"    GPU voxels:  {len(gpu_set)}")
        print(f"    CPU voxels:  {len(cpu_set)}")
        print(f"    共有 (交集):  {len(common)}")
        print(f"    仅 GPU:      {len(gpu_only)}")
        print(f"    仅 CPU:      {len(cpu_only)}")

        # GPU (AABB) 是 CPU (精确) 的超集或近似超集
        if len(cpu_only) == 0:
            coverage = "✓ CPU 结果完全被 GPU 覆盖 (GPU ⊇ CPU)"
        else:
            miss_rate = len(cpu_only) / len(cpu_set) * 100
            coverage = f"✗ GPU 遗漏了 {len(cpu_only)} 个 CPU voxel ({miss_rate:.2f}%)"
        print(f"    {coverage}")

        if len(gpu_only) > 0:
            extra_rate = len(gpu_only) / len(cpu_set) * 100
            print(f"    GPU AABB 多出 {len(gpu_only)} 个 voxel ({extra_rate:.1f}% overhead)")
        else:
            print(f"    GPU 没有多余 voxel (完全一致)")

    print(f"\n{'=' * 70}")
    print("  测试完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
