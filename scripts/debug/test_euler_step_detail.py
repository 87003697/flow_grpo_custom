#!/usr/bin/env python3
"""
详细检查 Euler 步进的 delta 计算差异
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
sys.path.insert(0, trellis2_ref_root)

import torch
import numpy as np


def main():
    device = torch.device("cuda")
    
    print("=" * 70)
    print("  Euler 步进 Delta 计算对比")
    print("=" * 70)
    
    steps = 12
    rescale_t = 3.0
    
    # ========== 参考实现的时间步序列 (numpy) ==========
    ref_t_seq = np.linspace(1, 0, steps + 1)
    ref_t_seq = rescale_t * ref_t_seq / (1 + (rescale_t - 1) * ref_t_seq)
    
    print("\n参考实现 (numpy):")
    for i in range(min(5, len(ref_t_seq))):
        print(f"  t[{i}] = {ref_t_seq[i]:.15f}")
    
    # ========== 用户实现的时间步序列 (torch) ==========
    from edit4shape.generators.trellis2.pipeline_adapter import FlowEulerScheduler
    
    your_scheduler = FlowEulerScheduler(rescale_t=rescale_t)
    your_scheduler.set_timesteps(steps, device=device)
    
    print("\n用户实现 (torch):")
    for i in range(min(5, len(your_scheduler.timesteps))):
        print(f"  t[{i}] = {your_scheduler.timesteps[i].item():.15f}")
    
    # ========== 对比 delta ==========
    print("\n对比 delta (t - t_prev):")
    for i in range(min(5, steps)):
        ref_t = ref_t_seq[i]
        ref_t_prev = ref_t_seq[i + 1]
        ref_delta = ref_t - ref_t_prev
        
        your_t = your_scheduler.timesteps[i].item()
        your_t_prev = your_scheduler.timesteps[i + 1].item()
        your_delta = your_t - your_t_prev
        
        delta_diff = abs(ref_delta - your_delta)
        
        print(f"\n  Step {i}:")
        print(f"    ref:  t={ref_t:.10f}, t_prev={ref_t_prev:.10f}, delta={ref_delta:.15f}")
        print(f"    your: t={your_t:.10f}, t_prev={your_t_prev:.10f}, delta={your_delta:.15f}")
        print(f"    delta_diff = {delta_diff:.2e}")
    
    # ========== 测试使用 numpy 时间步调用 scheduler ==========
    print("\n" + "=" * 70)
    print("  测试：使用 numpy 时间步调用 scheduler")
    print("=" * 70)
    
    # 模拟 Step 0
    t = ref_t_seq[0]  # numpy float64
    t_prev = ref_t_seq[1]
    
    print(f"\n使用 numpy t={t} 调用 scheduler.step():")
    
    # 看看 scheduler 能否正确找到 t_prev
    t_tensor = torch.tensor(t)
    match_idx = torch.isclose(
        your_scheduler.timesteps,
        torch.tensor(float(t), device=your_scheduler.timesteps.device, dtype=your_scheduler.timesteps.dtype)
    ).nonzero(as_tuple=False)
    
    if match_idx.numel() > 0:
        idx = int(match_idx[0])
        scheduler_t_prev = float(your_scheduler.timesteps[idx + 1].item())
        print(f"  scheduler 找到 t_prev = {scheduler_t_prev:.15f}")
        print(f"  numpy t_prev = {t_prev:.15f}")
        print(f"  差异 = {abs(scheduler_t_prev - t_prev):.2e}")
    else:
        print(f"  ❌ scheduler 未找到匹配的 t")
    
    # ========== 验证：如果强制使用相同的 delta 会怎样 ==========
    print("\n" + "=" * 70)
    print("  关键测试：验证是否是 delta 差异导致的问题")
    print("=" * 70)
    
    # 使用相同的 velocity 和 sample
    torch.manual_seed(42)
    sample = torch.randn(100, 32, device=device)
    velocity = torch.randn(100, 32, device=device)
    
    # 参考实现的 Euler 步进
    ref_delta = float(ref_t_seq[0] - ref_t_seq[1])
    ref_result = sample - ref_delta * velocity
    
    # 用户实现的 Euler 步进
    your_delta = float(your_scheduler.timesteps[0] - your_scheduler.timesteps[1])
    your_result = sample - your_delta * velocity
    
    diff = (ref_result - your_result).abs()
    print(f"\nEuler 步进结果对比:")
    print(f"  ref_delta = {ref_delta:.15f}")
    print(f"  your_delta = {your_delta:.15f}")
    print(f"  delta 差异 = {abs(ref_delta - your_delta):.2e}")
    print(f"  结果 max_diff = {diff.max().item():.6e}")
    print(f"  结果 mean_diff = {diff.mean().item():.6e}")
    
    # ========== 验证累积效应 ==========
    print("\n" + "=" * 70)
    print("  验证：delta 差异的累积效应")
    print("=" * 70)
    
    torch.manual_seed(42)
    ref_sample = torch.randn(100, 32, device=device)
    your_sample = ref_sample.clone()
    
    for step in range(steps):
        # 假设 velocity 在两边是相同的
        velocity = torch.randn(100, 32, device=device)
        
        ref_delta = float(ref_t_seq[step] - ref_t_seq[step + 1])
        your_delta = float(your_scheduler.timesteps[step] - your_scheduler.timesteps[step + 1])
        
        ref_sample = ref_sample - ref_delta * velocity
        your_sample = your_sample - your_delta * velocity
        
        diff = (ref_sample - your_sample).abs()
        print(f"  Step {step}: max_diff = {diff.max().item():.6e}, mean_diff = {diff.mean().item():.6e}")
    
    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    print("\n如果上面的累积差异很小，说明问题不在 delta 计算，")
    print("而是在模型本身对微小输入差异的敏感性（混沌效应）。")


if __name__ == "__main__":
    main()





