"""
Trellis2 Tex 训练系统 — 三阶段 Autograd 版本。

基于 trellis2_tex.py 的共享组件（Trellis2System, Trellis2State, build_system,
decode_and_render_pbr, trellis2_tex_forward, evaluate），
本模块仅实现 **Autograd 三阶段训练策略**：

核心流程：
  Phase 0: Shape 冻结前置（no_grad，获取几何条件）
  Phase 1: Tex Rollout（no_grad + RolloutTracker 记录 proxy 轨迹）
           + 计算 reg_loss → autograd.grad → reg_grads（纯数据，存 tracker）
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）→ 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)，完全不感知 reg

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ Phase 2 一次性 backward 算出                       ↑ Phase 3 逐步 VJP

与 trellis2_tex.py 的区别：
- trellis2_tex.py: 标准 forward → guidance → backward（端到端计算图）
- 本模块: 三阶段 Autograd（显存 O(1)，不随步数增长）

独有组件：
1. shape_frozen_prepare_no_grad: Shape 冻结前置
2. tex_phase1_rollout: Tex Phase 1（Rollout + Tracker + reg_grads）
3. tex_phase2a_decode_render: Tex Phase 2a（Decode + Render）
4. phase2_guidance_and_backward: Phase 2（Guidance + reg 合并 Backward）
5. phase3_rollout_grad_backward: Phase 3（纯 VJP — 逐步重算 + Backward）
6. three_phase_tex_step: 三阶段编排
7. main: 训练主循环（使用三阶段策略）
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 标准库 & 第三方库
# =====================================================================
import logging
from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from absl import app
from ml_collections import config_flags

# =====================================================================
# 从 trellis2_tex.py 导入共享组件（避免代码重复）
# 注意：trellis2_tex 的模块级 sys.path 设置会在 import 时自动执行，
#       之后即可直接 import trellis2.* 模块
# =====================================================================
from edit4shape.systems.trellis2_tex import (
    # 系统 & 状态
    Trellis2System,
    Trellis2State,
    # 构建 & 评估
    build_system,
    evaluate,
    # Tex 阶段核心函数
    decode_and_render_pbr,
    trellis2_tex_forward,
)

# =====================================================================
# trellis2.* 直接依赖（Phase 函数使用）
# =====================================================================
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh

# =====================================================================
# 项目内部导入（Phase 函数 & main 使用）
# =====================================================================
from edit4shape.systems.trellis2_shape import (
    trellis2_shape_forward,
)
from edit4shape.generators.trellis2.rollout import rollout_tex, RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, PhaseProfiler
from edit4shape.guidance import create_guidance

# =====================================================================
# absl 配置
# =====================================================================
# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。


# =====================================================================
# 三阶段 Autograd — Phase 函数
# =====================================================================

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
        # ★ Fix #1: trellis2_shape_forward 签名为 (system, state, global_step, is_training)
        #   不存在 render_normal 参数。Normal 渲染虽然多余但在 no_grad 下代价很小。
        trellis2_shape_forward(
            system, state, global_step,
            is_training=False,
        )
    
    # ★ 彻底 detach Shape 产物，切断与 Shape 计算图的依赖
    # 1. 条件嵌入（Shape/Tex 共用）
    if state.views_conditioned.cond_512_embed is not None:
        state.views_conditioned.cond_512_embed = state.views_conditioned.cond_512_embed.detach()  # (B, S, C)
    if state.views_conditioned.uncond_512_embed is not None:
        state.views_conditioned.uncond_512_embed = state.views_conditioned.uncond_512_embed.detach()  # (B, S, C)
    if state.views_conditioned.cond_1024_embed is not None:
        state.views_conditioned.cond_1024_embed = state.views_conditioned.cond_1024_embed.detach()  # (B, S, C)
    if state.views_conditioned.uncond_1024_embed is not None:
        state.views_conditioned.uncond_1024_embed = state.views_conditioned.uncond_1024_embed.detach()  # (B, S, C)
    
    # 2. coords
    state.coords = state.coords.detach().clone()  # (N, 4)
    
    # 3. shape_slat
    state.features.shape_slat = SparseTensor(
        coords=state.features.shape_slat.coords.detach(),
        feats=state.features.shape_slat.feats.detach()
    )
    
    # 4. subs
    state.features.subs = [
        SparseTensor(coords=sub.coords.detach(), feats=sub.feats.detach())
        for sub in state.features.subs
    ]
    
    # 5. meshes
    state.features.meshes = [
        Mesh(
            vertices=m.vertices.detach(),  # (V, 3)
            faces=m.faces,                 # (F, 3) 整数，不需要 detach
            vertex_attrs=m.vertex_attrs.detach() if m.vertex_attrs is not None else None
        )
        for m in state.features.meshes
    ]


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


def phase2_guidance_and_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Phase 2（同步版）: guidance + reg 合并 backward → 填充 tracker 梯度 → 释放图。
    
    ★ 与 shape_autograd 对齐：
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
    5. 释放所有计算图 + empty_cache()
    
    Returns:
        日志字典（包含 guidance loss、reg loss 等指标）
    """
    cfg = system.cfg
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = cfg.train.loss.guidance
    reg_weight = cfg.train.loss.reg
    
    # 1. Guidance 前向（同步阻塞）
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
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
    
    # 4. 释放所有计算图（decode/render + proxy chain + reg 图一次性释放）
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None  # 图已释放，清空引用
    
    # 5. 释放 Shape 解码中间产物（subs/meshes，Phase 3 不再需要）
    # ★ 不能调用 release_shape_decode_cache()：shape_slat_norm 仍需用于 Phase 3
    #   （作为 tex flow model 的条件输入）
    state.features.subs = None
    state.features.meshes = None
    
    torch.cuda.empty_cache()
    
    return guidance_log


def phase3_rollout_grad_backward(
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
    
    for i in range(T):
        # 1. 从 tracker 直接读取：t, x_t, v_grad
        t_val = tracker.timesteps[i]  # float64 精度
        
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.tex_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 2. 重算 cond_pred = f_θ(x_t, t, cond, shape_cond)（仅对 θ 有梯度，x_t detached）
        # ★ 只需 cond forward，无需 uncond / CFG 混合 / reg 计算
        # v_grad = cond_proxy.grad 已包含 CFG 因子 + reg 梯度（Phase 2 一次性 backward 得到）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_val, cond_emb,
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


# =====================================================================
# 三阶段 Autograd — 编排函数
# =====================================================================

def three_phase_tex_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler = None,
) -> Dict[str, Any]:
    """
    Tex-only 三阶段训练步（同步 Guidance 版本）。
    
    编排:
      shape_frozen_prepare (no_grad) → Tex Phase 1 (rollout + tracker + reg_grads)
      → Tex Phase 2a (decode PBR + render)
      → Phase 2 (guidance + reg 合并 backward，一路反传到 cond_proxy.grad)
      → Phase 3 (纯 VJP 逐步重算 + backward) → 返回日志
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler，用于测量各阶段耗时（enabled=False 时为空操作）
    
    Returns:
        合并的日志字典
    """
    cfg = system.cfg
    gen_seed = int(cfg.seed) + global_step + 1000  # +1000 避免与 shape seed 冲突
    
    # 公共准备：Shape 冻结前置（no_grad）
    profiler.tick("shape_frozen_prepare")
    shape_frozen_prepare_no_grad(state, system, global_step)
    
    # Tex Phase 1: Rollout no_grad + 记录 proxy 轨迹 + 计算 reg_grads
    profiler.tick("P1_rollout")
    tracker = tex_phase1_rollout(state, system, gen_seed)
    
    # Tex Phase 2a: Decode + Render PBR（直接连接 proxy chain，无 slat_proxy）
    # ★ OOM 降级（与 shape_autograd 对齐）：Phase 2a/2 OOM → Phase 3 降级零梯度贡献
    profiler.tick("P2a_decode_render")
    comp_rgb = None
    try:
        comp_rgb = tex_phase2a_decode_render(state, system)
        
        # Phase 2: Guidance + reg 合并 Backward（一路反传到 output_trajectory[t].grad）
        profiler.tick("P2_guidance_backward")
        guidance_log = phase2_guidance_and_backward(state, system, tracker, comp_rgb)
    except torch.cuda.OutOfMemoryError:
        # Phase 2a/2 OOM（decode/render 或 guidance backward 显存不足）
        # → Phase 3 降级，该 micro-batch 零梯度贡献。
        # 安全性：Phase 2a/2 不经过模型参数，不触发 DDP hooks，不会导致分布式死锁。
        logging.warning(f"[Step {global_step}] Phase 2a/2 OOM → Phase 3 降级 reg-only")
        del comp_rgb
        # 释放 Shape 解码中间产物（subs/meshes，Phase 3 不再需要）
        # ★ 保留 shape_slat_norm（Phase 3 需要作为 tex flow model 的条件输入）
        state.features.subs = None
        state.features.meshes = None
        torch.cuda.empty_cache()
        guidance_log = {}
    
    # Phase 3: 纯 VJP — 逐步重算 + 即时 Backward
    profiler.tick("P3_grad_backward")
    phase3_log = phase3_rollout_grad_backward(state, system, tracker)
    
    profiler.tick("end")
    
    # 合并日志 + 计算 loss/total（与 shape_autograd 对齐）
    # reg 日志在 guidance_log 里（Phase 2 合并 backward 时记录）
    merged = {**guidance_log, **phase3_log}
    total = guidance_log.get("loss/guidance", 0.0)
    if "loss/reg" in guidance_log:
        total += cfg.train.loss.reg * guidance_log["loss/reg"]
    merged["loss/total"] = total
    
    # Profiler: 收集计时并合入日志（外部无需感知 profiler）
    merged.update(profiler.collect(global_step, print_freq=int(cfg.freq.profiler)))
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd 版本）。
    
    只训练 Tex Flow Model，使用 PBR 渲染监督纹理。
    Shape 阶段使用冻结的模型生成几何。
    训练策略使用三阶段 Autograd（显存 O(1)）。
    
    流程: Dense Sampling → Shape Rollout (frozen) → Tex Rollout → PBR 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_tex_autograd --config=configs/trellis2_tex.py
    """
    del argv
    cfg = _CONFIG.value
    
    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)
    
    # =====================================================
    # Step 2: 初始化 Accelerator（含 wandb 日志）
    # =====================================================
    use_wandb = cfg.use_wandb
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )
    
    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    
    # 初始化 wandb trackers
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    # ★ Fix #3: 从 trellis2 导入 build_dataloaders（与 shape_autograd 对齐）
    from edit4shape.systems.trellis2 import build_dataloaders
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)
    
    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_optimizers(accelerator)
    
    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    # ★ Fix #2: 对齐 shape_autograd 签名 load(path, mode)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    
    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return
    
    # =====================================================
    # Step 8: 训练循环（三阶段 Autograd — cond-level proxy）
    # =====================================================
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    # ★ Fix #5: 添加 PhaseProfiler（与 shape_autograd 对齐）
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_log = three_phase_tex_step(state, system, global_step, profiler=profiler)
                
                # Optimizer Step
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # Logging & Visualization
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 PBR 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 残留引用
            del state, tex_log
            torch.cuda.empty_cache()

        # ---- 周期性评估（epoch 级别）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)
        
        # ---- 周期性保存检查点 ----
        # ★ Fix #2: 对齐 shape_autograd 签名 save(epoch, step)
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
