#!/usr/bin/env python3
import argparse
import math
from typing import List, Tuple

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
    Direct3DS2PipelineWithLogProb,
)


def _as_tensor_scalar(x: float, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(float(x), device=device, dtype=torch.float32)  # 形状: ()
    return t  # 形状: ()


def _make_scheduler(
    num_train_timesteps: int,
    num_inference_steps: int,
    device: torch.device,
    pipeline_path: str = "",
):
    """构建调度器：
    - 若提供 pipeline_path，则从预训练管线加载真实的 sparse 调度器；
    - 否则使用默认 FlowMatchEulerDiscreteScheduler。
    """
    if isinstance(pipeline_path, str) and len(pipeline_path) > 0:
        pipe = Direct3DS2PipelineWithLogProb.from_pretrained(pipeline_path)  # 形状: 管线对象
        scheduler = pipe.ref.sparse_scheduler_512  # 形状: 调度器
        scheduler.set_timesteps(int(num_inference_steps), device=device)
        return scheduler  # 形状: 调度器
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=int(num_train_timesteps))  # 形状: 调度器
    scheduler.set_timesteps(int(num_inference_steps), device=device)
    return scheduler  # 形状: 调度器


def check_timestep_index_mapping(
    num_train_timesteps: int,
    num_inference_steps: int,
    noise_level: float,
    eps_small: float,
    device: torch.device,
    pipeline_path: str = "",
) -> Tuple[int, int, int, int]:
    # 构造调度器并设置推理步数（可选从预训练管线加载）
    scheduler = _make_scheduler(
        num_train_timesteps=int(num_train_timesteps),  # 形状: 标量
        num_inference_steps=int(num_inference_steps),  # 形状: 标量
        device=device,  # 形状: () 设备
        pipeline_path=pipeline_path,  # 形状: 字符串
    )

    timesteps = scheduler.timesteps  # 形状: (T,)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)  # 形状: (S,)

    problems_dt_positive = 0  # 形状: 标量
    problems_dt_zero = 0  # 形状: 标量
    problems_same_index = 0  # 形状: 标量
    problems_sqrt_invalid = 0  # 形状: 标量

    one = torch.ones((), device=device, dtype=torch.float32)  # 形状: ()

    for i in range(len(timesteps)):
        t = float(timesteps[i].item())  # 形状: 标量
        t_prev = float(timesteps[i + 1].item()) if (i + 1) < len(timesteps) else float(t)  # 形状: 标量

        t_cur_tensor = _as_tensor_scalar(t, device)  # 形状: ()
        t_prev_tensor = _as_tensor_scalar(t_prev, device)  # 形状: ()

        step_index = int(scheduler.index_for_timestep(t_cur_tensor))  # 形状: 标量
        prev_step_index = int(scheduler.index_for_timestep(t_prev_tensor))  # 形状: 标量

        step_index = max(0, min(step_index, int(sigmas.shape[0]) - 1))  # 形状: 标量
        prev_step_index = max(0, min(prev_step_index, int(sigmas.shape[0]) - 1))  # 形状: 标量

        sigma = sigmas[step_index]  # 形状: ()
        sigma_prev = sigmas[prev_step_index]  # 形状: ()

        dt = sigma_prev - sigma  # 形状: ()

        # 统计潜在问题
        if float(dt.item()) > 0.0:
            problems_dt_positive += 1  # 形状: 标量
        if abs(float(dt.item())) <= eps_small:
            problems_dt_zero += 1  # 形状: 标量
        if step_index == prev_step_index:
            problems_same_index += 1  # 形状: 标量

        # 复现 direct3d_s2_sparse_tensor 中用于 step_std 的中间量，检查开方有效性
        # 注意：复用 sigma_cmp 逻辑，若 sigma≈1 则替换为 sigma_max
        sigma_max_index = 1 if int(sigmas.shape[0]) > 1 else 0  # 形状: 标量
        sigma_max = sigmas[sigma_max_index]  # 形状: ()
        ones_like_sigma = one  # 形状: ()
        sigma_cmp = torch.where(torch.isclose(sigma, ones_like_sigma), sigma_max, sigma)  # 形状: ()

        # std_dev_t 与 step_std 形状均为标量；仅用于校验是否会触发无效开方/除零
        denom = (one - sigma_cmp)  # 形状: ()
        # 分母过小会导致发散，这里仅做存在性检查
        if float(denom.item()) <= 0.0:
            problems_sqrt_invalid += 1  # 形状: 标量
        else:
            std_dev_t = torch.sqrt(torch.clamp(sigma / denom, min=0.0)) * float(noise_level)  # 形状: ()
            neg_dt = -dt  # 形状: ()
            if float(neg_dt.item()) < 0.0:
                problems_sqrt_invalid += 1  # 形状: 标量（-dt 应≥0）
            else:
                _ = torch.sqrt(torch.clamp(neg_dt, min=0.0)) * std_dev_t  # 形状: ()

        # 可选：打印详细调试（注释掉默认关闭）
        # print(f"i={i} t={t:.6f} t_prev={t_prev:.6f} idx=({step_index},{prev_step_index}) "
        #       f"sigma=({float(sigma):.6e},{float(sigma_prev):.6e}) dt={float(dt):.6e}")

    return problems_dt_positive, problems_dt_zero, problems_same_index, problems_sqrt_invalid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train_timesteps", type=int, default=1000, help="训练时间步（调度器内部表长）")
    ap.add_argument("--num_inference_steps", type=int, default=30, help="推理步数（外部循环步数）")
    ap.add_argument("--noise_level", type=float, default=0.7, help="用于 std_dev_t 的噪声缩放")
    ap.add_argument("--eps_small", type=float, default=1e-12, help="判断 dt≈0 的阈值")
    ap.add_argument("--device", type=str, default=None, help="设备（默认自动选择 cuda/cpu)")
    ap.add_argument(
        "--pipeline_path",
        type=str,
        default="",
        help="可选：从该目录加载 Direct3D-S2 预训练管线以使用真实 scheduler（如 pretrained_weights/direct3d_s2-v-1-1）",
    )
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 形状: () 设备
    else:
        device = torch.device(args.device)  # 形状: () 设备

    p_pos, p_zero, p_same, p_sqrt = check_timestep_index_mapping(
        num_train_timesteps=int(args.num_train_timesteps),  # 形状: 标量
        num_inference_steps=int(args.num_inference_steps),  # 形状: 标量
        noise_level=float(args.noise_level),  # 形状: 标量
        eps_small=float(args.eps_small),  # 形状: 标量
        device=device,  # 形状: () 设备
        pipeline_path=str(args.pipeline_path),  # 形状: 字符串
    )

    print("[Check] index_for_timestep 与 (t, t_prev) 对应关系：")
    print(f"  dt>0 次数: {p_pos}")
    print(f"  |dt|<=eps 次数: {p_zero}")
    print(f"  相同索引次数(step_index==prev_step_index): {p_same}")
    print(f"  潜在无效开方/分母<=0 次数: {p_sqrt}")

    # 非零问题数时以非零退出码提示
    total_problems = p_pos + p_zero + p_same + p_sqrt  # 形状: 标量
    if total_problems > 0:
        # 提示如何缓解：
        print("[Hint] 若存在问题，考虑：确保 dt<0、避免重复索引、末步跳过 logprob、在分母/开方前加 eps 裁剪。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


