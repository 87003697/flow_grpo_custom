"""
Trellis2 Shape+Tex 双阶段训练系统 — 三阶段 Autograd 版本。

同时训练 Shape 和 Tex 两个 Flow Model，每个阶段分别使用三阶段 Autograd 策略，
将 rollout / decoder+renderer 的计算图隔离，任意时刻只有一个阶段的计算图驻留显存。

整体编排（每个训练步）：
  ┌─────────── Shape Training ───────────┐
  │ Dense Sampling (no_grad)             │
  │ P1: rollout + RolloutTracker         │
  │ P2a: decode + render Normal          │
  │ P2: guidance + backward → proxy.grad │
  │ P3: VJP → shape θ.grad              │
  │ optimizer_shape.step()               │
  └──────────────────────────────────────┘
         ↓ detach shape 产物
  ┌─────────── Tex Training ─────────────┐
  │ P1: tex rollout + RolloutTracker     │
  │ P2a: decode_tex + render PBR         │
  │ P2: guidance + backward → proxy.grad │
  │ P3: VJP → tex θ.grad                │
  │ optimizer_tex.step()                 │
  └──────────────────────────────────────┘

三阶段流程（与 shape_autograd / tex_autograd 对齐）：
  Phase 1: rollout no_grad → slat（proxy chain，不含模型图）+ reg_loss
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad → 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)

复用关系：
- Shape Phase 函数：从 trellis2_shape_autograd 导入
- Tex Phase 函数：从 trellis2_tex_autograd 导入
- 新增：shape_phase2 变体（不释放 decode cache）+ detach 转接 + 编排 + 训练循环

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
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
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from pathlib import Path

import torch
from accelerate import Accelerator
from absl import app
from ml_collections import config_flags

# =====================================================================
# 从 trellis2_tex.py 导入共享组件
# 注意：trellis2_tex 的模块级 sys.path 设置会在 import 时自动执行，
#       之后即可直接 import trellis2.* 模块
# =====================================================================
from edit4shape.systems.trellis2_tex import (
    # 扩展版 State（含 tex 字段：features.tex_slat, views_generated.pbr_tensor 等）
    Trellis2State,
    # Tex 阶段核心函数
    decode_and_render_pbr,
    # 评估函数（内部调用 shape_forward + tex_forward）
    evaluate,
)

# =====================================================================
# trellis2.* 直接依赖
# =====================================================================
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh

# =====================================================================
# 从 trellis2_shape_autograd 导入 Shape Phase 函数
# =====================================================================
from edit4shape.systems.trellis2_shape_autograd import (
    dense_sampling_no_grad,
    shape_phase1_rollout,
    shape_phase2a_decode_render,
    phase3_rollout_grad_backward as shape_phase3_vjp,
)

# =====================================================================
# 从 trellis2_tex_autograd 导入 Tex Phase 函数
# =====================================================================
from edit4shape.systems.trellis2_tex_autograd import (
    tex_phase1_rollout,
    tex_phase2a_decode_render,
    phase2_guidance_and_backward as tex_phase2_guidance_and_backward,
    phase3_rollout_grad_backward as tex_phase3_vjp,
)

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.systems.trellis2_shape import StageSystem
from edit4shape.generators.trellis2.rollout import RolloutTracker
from edit4shape.systems.base import TrainModeGuard, EvalModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import (
    Trellis2CheckpointIO, StageConfig,
)
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, PhaseProfiler
from edit4shape.guidance import create_guidance
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin

# =====================================================================
# 从 trellis2.py 导入基础 build_system 和 build_dataloaders
# =====================================================================
from edit4shape.systems.trellis2 import (
    build_system as _build_system_base,
    build_dataloaders,
)

# =====================================================================
# absl 配置
# =====================================================================
# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。


# =====================================================================
# Trellis2 双阶段系统（三阶段 Autograd 版本）
# =====================================================================

@dataclass
class Trellis2System:
    """
    Trellis2 Shape+Tex 双阶段训练系统（三阶段 Autograd 版本）。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（model, optimizer, renderer, config）
    - tex: Tex 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    - strategy: 训练策略（LoRA / Full / Frozen）
    - cfg / accelerator: 运行时上下文（Phase 函数通过 system 访问）
    
    渲染器配置：
    - shape.renderer: MeshPeeledRenderer (face_normal + intersect_logits 双路可微)
    - tex.renderer: MeshPeeledRenderer (PBR + IBL 着色，支持梯度)
    """
    
    pipeline: Any = None
    
    # 分阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    tex: StageSystem = field(default_factory=StageSystem)
    
    # 共享组件
    guidance: Any = None
    
    # 训练策略
    strategy: Any = None
    
    # 运行时上下文（Phase 函数只需 (state, system) 即可访问所有配置和组件）
    cfg: Any = None
    accelerator: Accelerator = None
    
    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """设置随机种子与确定性运行环境。"""
        import random
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def prepare_lora(self, cfg: Any, adapter: str = "base", **kwargs) -> "Trellis2System":
        """准备 LoRA 适配器"""
        for module in [self.pipeline, self.guidance]:
            if module is not None and hasattr(module, "set_adapter"):
                module.set_adapter(adapter)
        return self
    
    def prepare_optimizers(self, accelerator: Accelerator) -> "Trellis2System":
        """
        通过 strategy.prepare() 逐阶段做 DDP 包裹 + 回写 pipeline。
        
        对 shape 和 tex 分别调用 strategy.prepare(accelerator, stage, resolution, optimizer)：
        1. accelerator.prepare(model, optimizer) → DDP 包裹模型 + 注册优化器
        2. 回写 DDP 模型到 pipeline.pipe.models
        3. 返回 (model, optimizer) 赋值给对应 StageSystem
        """
        if self.strategy is not None:
            if self.shape.optimizer is not None:
                sc = self.shape.config
                self.shape.model, self.shape.optimizer = self.strategy.prepare(
                    accelerator, sc.model_stage, sc.flow_resolution, self.shape.optimizer
                )
            if self.tex.optimizer is not None:
                tc = self.tex.config
                self.tex.model, self.tex.optimizer = self.strategy.prepare(
                    accelerator, tc.model_stage, tc.flow_resolution, self.tex.optimizer
                )
        return self


# =====================================================================
# 构建函数 — 包裹 trellis2.build_system + 后处理
# =====================================================================

def build_system(cfg, accelerator, guidance_factory) -> Trellis2System:
    """
    构建双阶段 Trellis2 系统。
    
    基于 trellis2.py 的 build_system（双阶段 + strategy），
    额外添加 Autograd 所需的后处理：
    1. 注入 ChunkedDecoderMixin（自适应显存分块）
    2. 冻结 envmap（不优化环境光）
    3. 挂载 cfg / accelerator 到 system
    """
    base = _build_system_base(cfg, accelerator, guidance_factory)
    
    pipeline = base.pipeline
    
    # ---- 1. 注入 Chunked Decoder（强制启用自适应显存分块） ----
    ChunkedDecoderMixin.inject_to(pipeline.pipe.models['shape_slat_decoder'])
    ChunkedDecoderMixin.inject_to(pipeline.pipe.models['tex_slat_decoder'])
    logging.info("[Shape+Tex Autograd] Shape/Tex decoder 已启用 chunked forward")
    
    # ---- 2. 冻结 envmap（不优化环境光，只优化纹理） ----
    # EnvironmentLight 构造函数会强制 base 为 nn.Parameter(requires_grad=True)，
    # 关掉梯度后重建 mips，使 specular/diffuse 从源头就无梯度
    _envlight = base.tex.renderer.envmap._backend
    _envlight.base.requires_grad_(False)
    _envlight.build_mips()
    logging.info("[Shape+Tex Autograd] envmap 已冻结")
    
    # ---- 3. 构建带 cfg/accelerator 的 Trellis2System ----
    return Trellis2System(
        pipeline=base.pipeline,
        shape=base.shape,
        tex=base.tex,
        guidance=base.guidance,
        strategy=base.strategy,
        cfg=cfg,
        accelerator=accelerator,
    )


# =====================================================================
# Shape → Tex 转接：轻量 Detach
# =====================================================================

def detach_shape_outputs_for_tex(state: Trellis2State) -> None:
    """
    轻量 detach：切断 Shape 训练计算图残留，为 Tex 三阶段准备干净输入。
    
    Shape 三阶段结束后，backward 已释放计算图，但张量仍携带残留 grad_fn。
    本函数创建全新的张量 / SparseTensor / Mesh，确保 Tex 阶段无跨阶段图依赖。
    
    ★ 不需要重新执行 shape forward，直接 detach 现有数据。
    ★ 同时释放 shape decoder 累积的 _spatial_cache（neighbor maps，可达 20-40 GiB）。
    
    Side Effects:
        - state.views_conditioned.cond_*_embed: detach  (B, S, C)
        - state.coords: detach + clone  (N, 4)
        - state.features.shape_slat: 全新 SparseTensor
        - state.features.subs: 全新 List[SparseTensor]
        - state.features.meshes: 全新 List[Mesh]
        - shape_slat._spatial_cache: 清理
    """
    # 0. 释放 shape decoder 的 spatial cache（neighbor maps 等大缓存）
    state.release_shape_spatial_cache()
    
    # 1. 条件嵌入（Shape/Tex 共用，必须 detach）
    for attr in ('cond_512_embed', 'uncond_512_embed', 'cond_1024_embed', 'uncond_1024_embed'):
        emb = getattr(state.views_conditioned, attr, None)
        if emb is not None:
            setattr(state.views_conditioned, attr, emb.detach())  # (B, S, C)
    
    # 2. coords — 创建全新张量，避免 SparseTensor 缓存关联
    state.coords = state.coords.detach().clone()  # (N, 4)
    
    # 3. shape_slat — 全新 SparseTensor（断开 proxy chain）
    state.features.shape_slat = SparseTensor(
        coords=state.features.shape_slat.coords.detach(),
        feats=state.features.shape_slat.feats.detach(),
    )
    
    # 4. subs — 全新 List[SparseTensor]
    if state.features.subs is not None:
        state.features.subs = [
            SparseTensor(coords=s.coords.detach(), feats=s.feats.detach())
            for s in state.features.subs
        ]
    
    # 5. meshes — vertices/vertex_attrs 来自 shape decoder，需 detach
    if state.features.meshes is not None:
        state.features.meshes = [
            Mesh(
                vertices=m.vertices.detach(),  # (V, 3)
                faces=m.faces,                 # (F, 3) 整数，不需要 detach
                vertex_attrs=m.vertex_attrs.detach() if m.vertex_attrs is not None else None,
            )
            for m in state.features.meshes
        ]


# =====================================================================
# Shape Phase 2: guidance + backward（保留 decode cache）
# =====================================================================

def shape_phase2_guidance_and_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Shape Phase 2: guidance + reg 合并 backward → 填充 tracker 梯度。
    
    ★ 与 shape_autograd 的 phase2_guidance_and_backward 唯一区别：
    不调用 release_shape_decode_cache()，保留 subs/meshes/shape_slat_norm
    供后续 Tex 阶段使用。
    
    流程:
    1. guidance = compute_guidance(comp_rgb, ...)
    2. total_loss = guidance.loss * weight + reg_weight * reg_loss
    3. accelerator.backward(total_loss)
       → 一路反传到 output_trajectory[t].grad
    4. 构建日志
    5. 释放 comp_rgb / loss 计算图 + empty_cache()
    
    Returns:
        日志字典
    """
    cfg = system.cfg
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = cfg.shape.train.loss.guidance
    reg_weight = cfg.shape.train.loss.reg
    
    # 1. Guidance 前向
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=cfg.shape.guidance,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)
    
    # 2. 合并 loss: guidance + reg
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
    
    # 4. 释放 loss 计算图（但 ★ 不释放 subs/meshes/shape_slat_norm）
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None  # 图已释放，清空引用
    
    torch.cuda.empty_cache()
    
    return guidance_log


# =====================================================================
# Shape 三阶段编排
# =====================================================================

def three_phase_shape_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Shape 三阶段训练步。
    
    与 shape_autograd 的 three_phase_shape_step 区别：
    - 使用 shape_phase2_guidance_and_backward（不释放 decode cache）
    - 不调用 release_shape_decode_cache()
    
    编排:
      dense_sampling → P1 (rollout + tracker) → P2a (decode + render)
      → P2 (guidance backward，保留 decode cache) → P3 (VJP) → 返回日志
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler
    
    Returns:
        合并的日志字典（key 前缀 "shape/"）
    """
    gen_seed = int(system.cfg.seed) + global_step
    
    profiler.tick("shape_dense_sampling")
    dense_sampling_no_grad(state, system)
    
    profiler.tick("shape_P1_rollout")
    tracker = shape_phase1_rollout(state, system, gen_seed)
    
    profiler.tick("shape_P2a_decode_render")
    comp_rgb = None
    try:
        comp_rgb = shape_phase2a_decode_render(state, system)
        
        profiler.tick("shape_P2_guidance_backward")
        guidance_log = shape_phase2_guidance_and_backward(
            state, system, tracker, comp_rgb
        )
    except torch.cuda.OutOfMemoryError:
        # Shape P2a/P2 OOM → P3 降级（零梯度贡献，但仍运行以维持 DDP 同步）
        # ★ 此时 subs/meshes 可能不完整，后续 Tex P2a 也会失败
        logging.warning(f"[Step {global_step}] Shape P2a/P2 OOM → 降级")
        del comp_rgb
        # 不调用 release_shape_decode_cache()：让后续 Tex 判断 meshes 是否可用
        torch.cuda.empty_cache()
        guidance_log = {}
    
    profiler.tick("shape_P3_grad_backward")
    shape_phase3_vjp(state, system, tracker)
    
    profiler.tick("shape_end")
    
    # 合并日志 + loss/total（对齐 shape_autograd）
    merged = {f"shape/{k}": v for k, v in guidance_log.items()}
    total = guidance_log.get("loss/guidance", 0.0)
    if "loss/reg" in guidance_log:
        total += system.cfg.shape.train.loss.reg * guidance_log["loss/reg"]
    merged["shape/loss/total"] = total
    
    return merged


# =====================================================================
# Tex 三阶段编排（从已有 Shape 产物出发，跳过 shape forward）
# =====================================================================

def three_phase_tex_step_from_shape(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Tex 三阶段训练步（从已有 Shape 产物出发）。
    
    与 tex_autograd 的 three_phase_tex_step 区别：
    - 跳过 shape_frozen_prepare_no_grad（Shape 产物已由 shape 三阶段 + detach 提供）
    
    前置条件:
        - state.coords / features.shape_slat / subs / meshes 已就绪（detached）
    
    编排:
      P1 (tex rollout + tracker) → P2a (decode PBR + render)
      → P2 (guidance backward) → P3 (VJP) → 返回日志
    
    Args:
        state: 已 detach 的状态（含 Shape 产物）
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler
    
    Returns:
        合并的日志字典（key 前缀 "tex/"）
    """
    cfg = system.cfg
    gen_seed = int(cfg.seed) + global_step + 1000  # +1000 避免与 shape seed 冲突
    
    # Tex Phase 1: Rollout（不需要 meshes，只需 coords + shape_slat_norm）
    profiler.tick("tex_P1_rollout")
    tracker = tex_phase1_rollout(state, system, gen_seed)
    
    # Tex Phase 2a + P2: Decode PBR + Render + Guidance Backward
    profiler.tick("tex_P2a_decode_render")
    comp_rgb = None
    skip_phase3 = False
    try:
        # ★ 如果 shape P2a OOM 导致 meshes 为 None，此处会失败
        if state.features.meshes is None:
            raise RuntimeError("meshes 不可用（Shape P2a OOM），跳过 Tex P2a")
        
        comp_rgb = tex_phase2a_decode_render(state, system)
        
        profiler.tick("tex_P2_guidance_backward")
        guidance_log = tex_phase2_guidance_and_backward(
            state, system, tracker, comp_rgb
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Tex P2a/P2 OOM 或 meshes 不可用
        # → 跳过 P3（超大样本 VJP 可能导致 NCCL timeout）
        # 安全性：P2a/P2 不经过 tex flow model，不触发 DDP hooks
        logging.warning(f"[Step {global_step}] Tex P2a/P2 failed: {e} → 跳过 P3")
        skip_phase3 = True
        del comp_rgb
        state.features.subs = None
        state.features.meshes = None
        torch.cuda.empty_cache()
        guidance_log = {}
    
    # Tex Phase 3: VJP
    if not skip_phase3:
        profiler.tick("tex_P3_grad_backward")
        tex_phase3_vjp(state, system, tracker)
    else:
        profiler.tick("tex_P3_skip")
        # 仅清理 tracker 数据，不执行 VJP
        del tracker.input_trajectory[:], tracker.output_trajectory[:]
        del tracker.timesteps[:]
        torch.cuda.empty_cache()
    
    profiler.tick("tex_end")
    
    # 合并日志 + loss/total（对齐 tex_autograd）
    merged = {f"tex/{k}": v for k, v in guidance_log.items()}
    total = guidance_log.get("loss/guidance", 0.0)
    if "loss/reg" in guidance_log:
        total += cfg.tex.train.loss.reg * guidance_log["loss/reg"]
    merged["tex/loss/total"] = total
    
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd 版本）。
    
    同时训练 Shape 和 Tex 两个 Flow Model。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理
    训练策略使用三阶段 Autograd（显存 O(1)，不随步数增长）。
    
    流程: Dense Sampling → Shape 三阶段 → Detach → Tex 三阶段
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_shape+tex_autograd \\
            --config=configs/trellis2_shape+tex.py
    """
    del argv
    cfg = _CONFIG.value
    
    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)
    
    # =====================================================
    # Step 2: 初始化 Accelerator
    # =====================================================
    use_wandb = cfg.use_wandb
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )
    
    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape+tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(
        visuals_train_dir, target_h=cfg.renderer.resolution,
        vis_freq=vis_freq, accelerator=accelerator,
    )
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
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
    # Step 8: 训练循环（双阶段三阶段 Autograd）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(
                batch, pipeline=system.pipeline,
                resolution=system.tex.config.cond_resolution,
            )
            
            # ============================================
            # Shape 三阶段 Forward + Backward + Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_log = three_phase_shape_step(
                        state, system, global_step, profiler,
                    )
                
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # ★ 保存 Shape 可视化（在 Tex guidance 覆盖 views_edited 之前）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)
            
            # ============================================
            # Detach 转接：Shape → Tex
            # ============================================
            detach_shape_outputs_for_tex(state)
            
            # ============================================
            # Tex 三阶段 Forward + Backward + Update
            # ============================================
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_log = three_phase_tex_step_from_shape(
                        state, system, global_step, profiler,
                    )
                
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # ★ 保存 Tex 可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)
            
            # ============================================
            # Logging
            # ============================================
            # 收集 profiler 计时（在最后一个 tick 之后）
            profiler_log = profiler.collect(
                global_step, print_freq=int(cfg.freq.profiler),
            )
            shape_log.update(profiler_log)
            
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 释放当前 step 残留引用
            del state, shape_log, tex_log
            torch.cuda.empty_cache()
        
        # ---- 周期性评估（epoch 级别） ----
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
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
