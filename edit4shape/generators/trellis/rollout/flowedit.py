"""
FlowEdit Rollout - 差分双分支采样

在 3D latent 空间（Dense / Sparse）上运行 FlowEdit 算法：
- 双分支 CFG（src 反向 / tgt 正向）
- Aligned 噪声更新
- 差分 Euler 步

仅用于推理/评估。
"""

from typing import Optional, Any
import torch
from tqdm import tqdm
from accelerate import Accelerator

from trellis.modules.sparse import SparseTensor

from .base import (
    _predict_sparse_cond_velocity,
    _predict_dense_cond_velocity,
    _expand_cond_to_batch,
    prepare_embeddings,
)


TrellisState = Any
System = Any


# =====================================================================
# FlowEdit Sparse — Stage 2 (slat_flow_model)
# =====================================================================

def rollout_sparse_flowedit(
    state: TrellisState,
    cfg,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Sparse FlowEdit 差分采样（SLAT Stage 2）。

    核心流程:
      1. 标准 Euler rollout → x_src (teacher clean z₀)
      2. FlowEdit 差分循环：双分支 CFG + aligned noise → z_edit
      3. 反归一化 → 挂载 state.stage2.z0

    Side Effects:
        - state.stage2.z0: 挂载 FlowEdit 编辑后的 SparseTensor（反归一化）
    """
    pipeline = system.pipeline
    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = pipeline.sparse.get_runtime_params()

    cond_emb, uncond_emb = prepare_embeddings(state, device)
    assert state.coords is not None, "state.coords 缺失"
    assert generator is not None, "generator 必须由调用方提供"

    # ---- 1. 标准 Euler Rollout → x_src ----
    from .ode import rollout_sparse
    rollout_sparse(state, cfg, system, device, generator=generator, is_training=False)

    # 反向 normalize（ode rollout 输出的是 denorm 的，需要 re-normalize）
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    x_src_feats = (state.stage2.z0.feats - mean) / std  # (N, C) normalized
    x_src = state.stage2.z0.replace(x_src_feats)

    # ---- 2. FlowEdit 参数 ----
    fe_cfg = cfg.rollout.flowedit
    fe_steps = int(fe_cfg.steps)
    n_max = int(fe_cfg.n_max)
    cfg_scale_tgt = float(fe_cfg.cfg_scale_tgt)
    cfg_scale_src = float(fe_cfg.cfg_scale_src)

    # ---- 3. 初始化 FlowEdit 状态 ----
    z_edit_feats = x_src.feats.clone()  # (N, C)
    noise = torch.randn(
        x_src.feats.shape, generator=generator,
        device=device, dtype=x_src.feats.dtype,
    )  # (N, C)

    # ---- 4. Scheduler 配置 ----
    scheduler = pipeline.sparse.scheduler()
    scheduler.set_timesteps(fe_steps, device=device, rescale_t=slat_rescale_t)

    # ---- 5. FlowEdit 差分循环 ----
    steps = list(scheduler.timesteps)[:-1]
    num_steps = len(steps)
    B = cond_emb.shape[0]

    steps_iter = tqdm(steps, desc="FlowEdit-Sparse", leave=False,
                      disable=not Accelerator().is_main_process)

    with torch.no_grad():
        for i, t in enumerate(steps_iter):
            if num_steps - i > n_max:
                continue

            t_val = float(t.item())
            t_prev_val = float(scheduler.timesteps[i + 1].item())
            dt = t_prev_val - t_val  # < 0
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)

            # ---- Source Branch（反向 CFG）----
            latents_src_feats = (1 - t_val) * x_src.feats + t_val * noise  # (N, C)
            latents_src = x_src.replace(latents_src_feats)

            v_cond_src = _predict_sparse_cond_velocity(
                pipeline, latents_src, t_batch, cond_emb
            )
            v_uncond_src = _predict_sparse_cond_velocity(
                pipeline, latents_src, t_batch, uncond_emb
            )

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src_feats = (
                (1 + cfg_scale_src) * v_cond_src.feats - cfg_scale_src * v_uncond_src.feats
            )

            # ---- Target Branch（正向 CFG）----
            latents_tgt_feats = z_edit_feats + (latents_src_feats - x_src.feats)  # (N, C)
            latents_tgt = x_src.replace(latents_tgt_feats)

            v_cond_tgt = _predict_sparse_cond_velocity(
                pipeline, latents_tgt, t_batch, cond_emb
            )
            v_uncond_tgt = _predict_sparse_cond_velocity(
                pipeline, latents_tgt, t_batch, uncond_emb
            )

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt_feats = (
                (1 + cfg_scale_tgt) * v_cond_tgt.feats - cfg_scale_tgt * v_uncond_tgt.feats
            )

            # ---- 差分 Euler 步 ----
            v_delta = v_cfg_tgt_feats - v_cfg_src_feats  # (N, C)
            z_edit_feats = z_edit_feats + dt * v_delta

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt.feats - v_uncond_tgt.feats) * (1.0 - t_val)

    # ---- 6. 反归一化 + 挂载 ----
    denorm_feats = z_edit_feats * std + mean  # (N, C)
    state.stage2.z0 = x_src.replace(denorm_feats)


# =====================================================================
# FlowEdit Dense — Stage 1 (sparse_structure_flow_model)
# =====================================================================

def rollout_dense_flowedit(
    state: TrellisState,
    cfg,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Dense FlowEdit 差分采样（Stage 1 — sparse_structure_flow_model）。

    核心流程:
      1. 标准 Euler rollout → x_src (teacher clean z_s)
      2. FlowEdit 差分循环：双分支 CFG + aligned noise → z_edit
      3. 挂载 state.stage1.z0

    Stage 1 无 normalization，raw latent 直接操作。

    Side Effects:
        - state.stage1.z0: 挂载 FlowEdit 编辑后的 Dense Tensor (B, C, R, R, R)
    """
    pipeline = system.pipeline
    ss_steps, ss_guidance, ss_rescale_t, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

    cond_emb, uncond_emb = prepare_embeddings(state, device)
    assert generator is not None, "generator 必须由调用方提供"

    # ---- 1. 标准 Euler Rollout → x_src ----
    from .ode import rollout_dense
    rollout_dense(state, cfg, system, device, generator=generator)
    x_src = state.stage1.z0.clone()  # (B, C, R, R, R)

    # ---- 2. FlowEdit 参数 ----
    fe_cfg = cfg.rollout.flowedit
    fe_steps = int(fe_cfg.steps)
    n_max = int(fe_cfg.n_max)
    cfg_scale_tgt = float(fe_cfg.cfg_scale_tgt)
    cfg_scale_src = float(fe_cfg.cfg_scale_src)

    # ---- 3. 初始化 FlowEdit 状态 ----
    z_edit = x_src.clone()  # (B, C, R, R, R)
    # ODE init_latents 需要 CPU generator；FlowEdit noise 需要 CUDA generator。
    # 从传入的 generator 中提取种子重建 CUDA generator，避免设备类型冲突。
    _fe_seed = generator.initial_seed() + 1 if generator is not None else 0
    cuda_gen = torch.Generator(device=device).manual_seed(_fe_seed)
    noise = torch.randn(
        x_src.shape, generator=cuda_gen,
        device=device, dtype=x_src.dtype,
    )  # (B, C, R, R, R)

    # ---- 4. 时间步序列 ----
    _, t_pairs = pipeline.dense.scheduler(fe_steps, ss_rescale_t)
    num_steps = len(t_pairs)

    B = x_src.shape[0]

    # ---- 5. FlowEdit 差分循环 ----
    with torch.no_grad():
        for i, (t, t_prev) in enumerate(tqdm(
            t_pairs, desc="FlowEdit-Dense", leave=False,
            disable=not Accelerator().is_main_process
        )):
            if num_steps - i > n_max:
                continue

            t_val = float(t)
            dt = float(t_prev) - t_val  # < 0

            # ---- Source Branch（反向 CFG）----
            latents_src = (1 - t_val) * x_src + t_val * noise  # (B, C, R, R, R)

            cond_input = _expand_cond_to_batch(cond_emb, B)
            uncond_input = _expand_cond_to_batch(uncond_emb, B)

            v_cond_src = _predict_dense_cond_velocity(pipeline, latents_src, t_val, cond_input)
            v_uncond_src = _predict_dense_cond_velocity(pipeline, latents_src, t_val, uncond_input)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src = (1 + cfg_scale_src) * v_cond_src - cfg_scale_src * v_uncond_src

            # ---- Target Branch（正向 CFG）----
            latents_tgt = z_edit + (latents_src - x_src)  # (B, C, R, R, R)

            v_cond_tgt = _predict_dense_cond_velocity(pipeline, latents_tgt, t_val, cond_input)
            v_uncond_tgt = _predict_dense_cond_velocity(pipeline, latents_tgt, t_val, uncond_input)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt = (1 + cfg_scale_tgt) * v_cond_tgt - cfg_scale_tgt * v_uncond_tgt

            # ---- 差分 Euler 步 ----
            v_delta = v_cfg_tgt - v_cfg_src  # (B, C, R, R, R)
            z_edit = z_edit + dt * v_delta

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt - v_uncond_tgt) * (1.0 - t_val)

    # ---- 6. 挂载到 state（无 normalization）----
    state.stage1.z0 = z_edit
