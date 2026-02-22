
# =====================================================================
# 三阶段 Autograd — Phase 函数
#
# 纯计算函数，不含 main() 或编排逻辑。
# 被 stage_ops.py（同步模板）和 entries/*_async.py（异步流水线）共用。
# =====================================================================

from __future__ import annotations

from typing import Any, Dict

import torch

from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import Trellis2System
from edit4shape.generators.trellis2.rollout import (
    rollout_shape, rollout_tex, RolloutTracker,
)
from edit4shape.generators.trellis2.rollout.base import _predict_velocity
from edit4shape.systems.trellis2.forward import (
    decode_and_render_normal,
    decode_and_render_pbr,
    trellis2_shape_forward,
    _detach_shape_outputs,
)


def shape_phase1_rollout(
    state: Trellis2State,
    system: Trellis2System,
    gen_seed: int,
) -> RolloutTracker:
    """
    Shape Phase 1: 无梯度 rollout + 记录 proxy 轨迹。
    
    - 创建 RolloutTracker 并传入 rollout_shape()
    - rollout_shape 在每步记录 input/output proxy，用 proxy 推进 scheduler
    - 最终 state.features.shape_slat 含 proxy chain（不含模型计算图）
    
    Args:
        state: 已填充 coords 和条件编码的状态
        system: 训练系统
        gen_seed: rollout 随机种子
    
    Returns:
        RolloutTracker: 已填充 input_trajectory / output_trajectory
    """
    cfg = system.cfg
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    device = system.accelerator.device
    
    tracker = RolloutTracker()
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    
    # ⚠️ 不可包裹 torch.no_grad()：rollout_shape 内部 is_training=False
    #    已用 no_grad 做模型推理，但 tracker 的 proxy 需要 autograd 图
    #    （scheduler 用 proxy 推进 → slat 依赖 proxy chain）
    rollout_shape(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=gen,
        is_training=False,   # 模型推理 no_grad
        tracker=tracker,     # ★ 记录 proxy 轨迹
    )
    # state.features.shape_slat: SparseTensor（有 proxy chain，不含模型图）
    
    return tracker


def shape_phase2a_decode_render(
    state: Trellis2State,
    system: Trellis2System,
) -> torch.Tensor:
    """
    Shape Phase 2a: slat → decode → render → comp_rgb (normals)。
    
    直接使用带 proxy chain 的 slat（不创建 slat_proxy），
    decode/render 的 autograd 图连接到 proxy chain 上，
    后续 loss.backward() 一路反传到 output_trajectory[t].grad。
    
    Args:
        state: 已填充 shape_slat 的状态
        system: 训练系统
    
    Returns:
        comp_rgb: (B, V, H, W, 3) Normal 渲染图（有 autograd 图，连接到 proxy chain）
    """
    cfg = system.cfg
    pipeline = system.pipeline
    device = system.accelerator.device
    target_res = pipeline.target_resolution
    # ★ 直接使用带 proxy chain 的 slat — 无需 slat_proxy 中间层
    # loss.backward() 将一路反传：renderer → decoder → slat → scheduler → CFG → cond_proxy.grad
    render_out = decode_and_render_normal(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=target_res,
    )
    comp_rgb = render_out["color"]  # (B, V, H, W, 3)
    
    # 挂载可视化数据（detach 避免保留计算图）
    state.views_generated.shape_tensor = comp_rgb.detach()
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]
    
    return comp_rgb


def shape_phase2_guidance_and_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Phase 2（同步版）: guidance + reg 合并 backward → 填充 tracker 梯度 → 释放图。
    
    ★ 纯计算函数：只做 guidance 前向 + 合并 backward + 构建日志 + 释放计算图引用。
    不做任何 decode cache / spatial cache 的清理——由调用方通过
    state.prepare_for_shape_vjp() 统一管理 P2→P3 过渡的显存释放。
    
    reg_loss 在 Phase 1 的 rollout 中已计算并存于 state.regularization.reg_loss，
    其计算图连着 cond_proxy（通过 velocity → CFG → cond_proxy chain）。
    与 guidance_loss 合并后一次性 backward：
      cond_proxy.grad = ∂L_guid/∂cond_proxy + reg_weight * ∂L_reg/∂cond_proxy
    Phase 3 直接用 cond_proxy.grad 做 VJP，完全不感知 reg。
    
    流程:
    1. guidance = compute_guidance(comp_rgb, ...)
    2. total_loss = guidance.loss * weight + reg_weight * reg_loss
    3. accelerator.backward(total_loss)
       → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）
    4. 构建日志
    5. 释放计算图引用 + empty_cache()
    
    Returns:
        日志字典（包含 guidance loss、reg loss 等指标）
    """
    cfg = system.cfg
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = cfg.shape.train.loss.guidance
    reg_weight = cfg.shape.train.loss.reg
    
    # 1. Guidance 前向（同步阻塞）
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=cfg.shape.guidance,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)
    
    # 2. 合并 loss: guidance + reg（reg_loss 的图连着 cond_proxy，backward 自然传播）
    # comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    # reg_loss ← MSE ← velocity ← CFG ← cond_proxy
    # → 两路梯度汇聚到 cond_proxy.grad
    total_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    reg_loss = state.regularization.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()
    
    accelerator.backward(total_loss)
    
    # 3. 构建日志
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = (guidance_result.loss.to(device) * guidance_weight).item()
    if reg_loss is not None:
        guidance_log["loss/reg"] = reg_loss.item()
    
    # 4. 释放所有计算图引用（decode/render + proxy chain + reg 图一次性释放）
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None  # 图已释放，清空引用
    
    torch.cuda.empty_cache()
    
    return guidance_log


def shape_phase3_rollout_grad_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
) -> Dict[str, Any]:
    """
    Phase 3: 纯 VJP — 逐步重算 f_θ，用 cond_proxy.grad 做 VJP → θ.grad 累积。
    显存 O(1)，不随步数增长。
    
    ★ Phase 3 完全不感知 reg：
    reg_loss 已在 Phase 1 计算（连着 cond_proxy），Phase 2 合并 backward 后，
    cond_proxy.grad 已同时包含 guidance 梯度和 reg 梯度。
    Phase 3 只需做 VJP: (cond_proxy.grad)^T · ∂f_θ/∂θ。
    
    流程（每步 t）:
      1. t_val, x_t, v_grad ← tracker 直接读取
      2. cond_pred = f_θ(x_t, t) — 唯一需要重算的，有 θ 梯度
      3. (v_grad * cond_pred).sum().backward() — VJP，图立即释放
    
    Returns:
        空日志字典（reg 日志已由 Phase 2 提供）
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_config = pipeline.get_stage_config("shape")
    flow_res = stage_config["flow_resolution"]
    
    # ---- 条件编码（只需 cond，不需要 uncond） ----
    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    
    T = len(tracker.input_trajectory)
    B = cond_emb.shape[0]  # ()
    
    for i in range(T):
        # 1. 从 tracker 直接读取：t, x_t, v_grad
        t_val = tracker.timesteps[i]  # float64 精度
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
        
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.shape_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 2. 重算 cond_pred = f_θ(x_t, t)（仅对 θ 有梯度，x_t detached）
        # ★ 只需 cond forward，无需 uncond / CFG 混合 / reg 计算
        # v_grad = cond_proxy.grad 已包含 CFG 因子 + reg 梯度（Phase 2 一次性 backward 得到）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_batch, cond_emb,
            "shape", flow_res, None
        )  # SparseTensor
        
        # 3. VJP: (v_grad)^T · ∂f_θ/∂θ → θ.grad +=
        # v_grad 为 None 时说明 Phase 2 OOM，跳过（无梯度贡献）
        v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
        if v_grad is not None:
            (v_grad * cond_pred.feats).sum().backward()  # 图立即释放，显存 O(1)
    
    # ---- 释放 tracker 数据 ----
    del tracker.input_trajectory[:], tracker.output_trajectory[:]
    del tracker.timesteps[:]
    torch.cuda.empty_cache()
    
    return {}



def shape_frozen_prepare_no_grad(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
) -> None:
    """
    Shape 冻结前置：no_grad 下执行 Shape forward 获取几何条件（coords/meshes/subs）。
    
    完成后 detach 所有 Shape 产物，切断与 Shape 计算图的依赖。
    
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.features.shape_slat / shape_slat_norm: 挂载 shape latent
        - state.features.subs / meshes: 挂载几何中间结果（已 detach）
    """
    with torch.no_grad():
        # Tex 模式不需要 Normal 渲染，仅 decode 获取 subs/meshes
        trellis2_shape_forward(
            system, state, global_step,
            is_training=False,
            render_normal=False,
        )
    
    # ★ 彻底 detach Shape 产物，切断与 Shape 计算图的依赖
    _detach_shape_outputs(state)


def tex_phase1_rollout(
    state: Trellis2State,
    system: Trellis2System,
    gen_seed: int,
) -> RolloutTracker:
    """
    Tex Phase 1: 无梯度 rollout_tex + 记录 proxy 轨迹 + 计算 reg_grads。
    
    - 创建 RolloutTracker 并传入 rollout_tex()
    - rollout_tex 在每步记录 input/output proxy，用 proxy 推进 scheduler
    - reg_loss 在 rollout 内计算，reg_grads 提前通过 autograd.grad 存入 tracker
    - 最终 state.features.tex_slat 含 proxy chain（不含模型计算图）
    
    前置条件:
        - state.coords, state.features.shape_slat_norm, subs, meshes 已就绪（由 shape_frozen_prepare_no_grad 产出）
    
    Args:
        state: 已填充几何条件的状态
        system: 训练系统
        gen_seed: rollout 随机种子
    
    Returns:
        RolloutTracker: 已填充 input_trajectory / output_trajectory / timesteps [/ reg_grads]
    """
    cfg = system.cfg
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("tex")
    device = system.accelerator.device
    
    tracker = RolloutTracker()
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    
    # ⚠️ 不可包裹 torch.no_grad()：rollout_tex 内部 is_training=False
    #    已用 no_grad 做模型推理，但 tracker 的 proxy 需要 autograd 图
    #    （scheduler 用 proxy 推进 → slat 依赖 proxy chain）
    rollout_tex(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=gen,
        is_training=False,   # 模型推理 no_grad
        tracker=tracker,     # ★ 记录 proxy 轨迹 + 计算 reg_grads
    )
    # state.features.tex_slat: SparseTensor（有 proxy chain，不含模型图）
    # state.regularization.reg_loss: tensor（有图，连着 cond_proxy）或 None
    # tracker.reg_grads: 提前计算的 reg 梯度（纯数据）
    
    return tracker


def tex_phase2a_decode_render(
    state: Trellis2State,
    system: Trellis2System,
) -> torch.Tensor:
    """
    Tex Phase 2a: tex_slat(含 proxy chain) → decode_tex → PBR render → comp_rgb。
    
    直接使用带 proxy chain 的 tex_slat（不创建 slat_proxy），
    decode/render 的 autograd 图连接到 proxy chain 上，
    后续 loss.backward() 一路反传到 output_trajectory[t].grad。
    
    Args:
        state: 已填充 tex_slat 的状态
        system: 训练系统
    
    Returns:
        comp_rgb: (B, V, H, W, 3) PBR 渲染图（有 autograd 图，连接到 proxy chain）
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    target_res = pipeline.target_resolution
    
    # ★ 直接使用带 proxy chain 的 tex_slat — 无需 slat_proxy 中间层
    # loss.backward() 将一路反传：renderer → decoder → slat → scheduler → CFG → cond_proxy.grad
    render_out = decode_and_render_pbr(
        state.features.meshes,
        state.features.tex_slat,
        state.features.subs,
        state.cameras,
        pipeline,
        system.tex.renderer,
        device,
        resolution=target_res,
    )
    comp_rgb = render_out["color"]  # (B, V, H, W, 3)
    
    # 挂载可视化数据（detach 避免保留额外计算图引用）
    state.views_generated.pbr_tensor = comp_rgb.detach()
    
    return comp_rgb


def tex_phase2_guidance_and_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Phase 2（同步版）: guidance + reg 合并 backward → 填充 tracker 梯度 → 释放图。
    
    ★ 纯计算函数：只做 guidance 前向 + 合并 backward + 构建日志 + 释放计算图引用。
    不做任何 decode cache / spatial cache / subs / meshes 的清理——由调用方通过
    state.prepare_for_tex_vjp() 统一管理 P2→P3 过渡的显存释放。
    
    reg_loss 在 Phase 1 的 rollout 中已计算并存于 state.regularization.reg_loss，
    其计算图连着 cond_proxy（通过 MSE(cond_proxy, teacher)）。
    与 guidance_loss 合并后一次性 backward：
      cond_proxy.grad = ∂L_guid/∂cond_proxy + reg_weight * ∂L_reg/∂cond_proxy
    Phase 3 直接用 cond_proxy.grad 做 VJP，完全不感知 reg。
    
    流程:
    1. guidance = compute_guidance(comp_rgb, ...)
    2. total_loss = guidance.loss * weight + reg_weight * reg_loss
    3. accelerator.backward(total_loss)
       → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）
    4. 构建日志
    5. 释放计算图引用 + empty_cache()
    
    Returns:
        日志字典（包含 guidance loss、reg loss 等指标）
    """
    cfg = system.cfg
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = cfg.tex.train.loss.guidance
    reg_weight = cfg.tex.train.loss.reg
    
    # 1. Guidance 前向（同步阻塞）
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=cfg.tex.guidance,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)
    
    # 2. 合并 loss: guidance + reg（reg_loss 的图连着 cond_proxy，backward 自然传播）
    # comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    # reg_loss ← MSE ← cond_proxy
    # → 两路梯度汇聚到 cond_proxy.grad
    total_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    reg_loss = state.regularization.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()
    
    accelerator.backward(total_loss)
    
    # 3. 构建日志
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = (guidance_result.loss.to(device) * guidance_weight).item()
    if reg_loss is not None:
        guidance_log["loss/reg"] = reg_loss.item()
    
    # 4. 释放所有计算图引用（decode/render + proxy chain + reg 图一次性释放）
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None  # 图已释放，清空引用
    
    torch.cuda.empty_cache()
    
    return guidance_log


def tex_phase3_rollout_grad_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
) -> Dict[str, Any]:
    """
    Phase 3: 纯 VJP — 逐步重算 f_θ，用 cond_proxy.grad 做 VJP → θ.grad 累积。
    显存 O(1)，不随步数增长。
    
    ★ Phase 3 完全不感知 reg（与 shape_autograd 对齐）：
    reg_loss 已在 Phase 1 计算（连着 cond_proxy），Phase 2 合并 backward 后，
    cond_proxy.grad 已同时包含 guidance 梯度和 reg 梯度。
    Phase 3 只需做 VJP: (cond_proxy.grad)^T · ∂f_θ/∂θ。
    
    流程（每步 t）:
      1. t_val, x_t, v_grad ← tracker 直接读取
      2. cond_pred = f_θ(x_t, t, cond, shape_cond) — 唯一需要重算的，有 θ 梯度
      3. (v_grad * cond_pred).sum().backward() — VJP，图立即释放
    
    Returns:
        空日志字典（reg 日志已由 Phase 2 提供）
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_config = pipeline.get_stage_config("tex")
    flow_res = stage_config["flow_resolution"]
    
    # ---- 条件编码（只需 cond，不需要 uncond） ----
    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    
    # ---- shape 条件（tex flow model 的额外条件） ----
    shape_cond = state.features.shape_slat_norm
    
    T = len(tracker.input_trajectory)
    B = cond_emb.shape[0]  # ()
    
    for i in range(T):
        # 1. 从 tracker 直接读取：t, x_t, v_grad
        t_val = tracker.timesteps[i]  # float64 精度
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
        
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.tex_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 2. 重算 cond_pred = f_θ(x_t, t, cond, shape_cond)（仅对 θ 有梯度，x_t detached）
        # ★ 只需 cond forward，无需 uncond / CFG 混合 / reg 计算
        # v_grad = cond_proxy.grad 已包含 CFG 因子 + reg 梯度（Phase 2 一次性 backward 得到）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_batch, cond_emb,
            "tex", flow_res, shape_cond
        )  # SparseTensor
        
        # 3. VJP: (v_grad)^T · ∂f_θ/∂θ → θ.grad +=
        # v_grad 为 None 时说明 Phase 2 OOM，跳过（无梯度贡献）
        v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
        if v_grad is not None:
            (v_grad * cond_pred.feats).sum().backward()  # 图立即释放，显存 O(1)
    
    # ---- 释放 tracker 数据 ----
    del tracker.input_trajectory[:], tracker.output_trajectory[:]
    del tracker.timesteps[:]
    torch.cuda.empty_cache()
    
    return {}
