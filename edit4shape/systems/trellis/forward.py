"""
Trellis 共享前向传播、渲染、评估函数。

从 system.py 提取的纯计算函数，供 standard / autograd / bilevel 入口复用。

主要组件：
1. decode_and_render_mesh: 解码 SparseTensor → Mesh → 渲染多视角图像
2. decode_and_render_gs:   解码 SparseTensor → Gaussian Splatting → 渲染多视角图像
3. trellis_forward:        共享前向传播（Dense Sampling → Rollout → Decode → Render）
4. evaluate:               评估循环（推理 + 可视化保存）
"""

from __future__ import annotations

import torch
import ml_collections
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from accelerate import Accelerator

from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import rollout_sparse, rollout_sparse_sde
from edit4shape.systems.base import EvalModeGuard
from edit4shape.systems.trellis.system import TrellisSystem
from edit4shape.systems.utils import VisualIO


# =====================================================================
# 渲染工具函数 - Mesh 渲染
# =====================================================================

def decode_and_render_mesh(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # TrellisMeshRasterizer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Mesh 并渲染多视角图像。

    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: Mesh 渲染器实例
        device: 运行设备

    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "normal": (B,V,H,W,3) 法线图
            - "depth": (B,V,H,W,1) 深度图
            - "meshes": list[len=B] of MeshExtractResult
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['mesh'])  # dict
    meshes = outputs['mesh']  # list[len=B] of MeshExtractResult

    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    batch_size, num_views = extr_all.shape[:2]  # (), ()

    # ---- 逐样本逐视角渲染 ----
    all_renders: Dict[str, List[torch.Tensor]] = {}

    for i, mesh in enumerate(meshes):
        view_renders: Dict[str, List[torch.Tensor]] = {}

        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)

            # Mesh 渲染器返回 RenderOutput
            render_out = renderer.render(mesh, ext_iv, intr_iv)  # RenderOutput
            render_dict = {
                "color": render_out.color,  # (H,W,3)
                "normal": render_out.normal,  # (H,W,3)
                "depth": render_out.depth,  # (H,W)
                "mask": render_out.mask,  # (H,W)
            }

            for k, val in render_dict.items():
                if val is None:
                    continue
                view_renders.setdefault(k, []).append(val)  # (H,W,C) or (H,W)

        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        for k, v_list in view_renders.items():
            stacked = torch.stack(v_list, dim=0)  # (V,H,W,C)
            all_renders.setdefault(k, []).append(stacked)

    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {}
    for k, b_list in all_renders.items():
        result[k] = torch.stack(b_list, dim=0)  # (B,V,H,W,C)

    result["meshes"] = meshes  # 保留 mesh 供导出
    return result


# =====================================================================
# 渲染工具函数 - Gaussian Splatting 渲染
# =====================================================================

def decode_and_render_gs(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # GaussianRenderer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Gaussian Splatting 并渲染多视角图像。

    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: GS 渲染器实例
        device: 运行设备

    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "gaussians": list[len=B] of Gaussian 对象
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['gaussian'])  # dict
    gaussians = outputs['gaussian']  # list[len=B] of Gaussian

    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    _, num_views = extr_all.shape[:2]  # (), ()

    # ---- 逐样本逐视角渲染 ----
    all_colors: List[torch.Tensor] = []

    for i, gs in enumerate(gaussians):
        view_colors: List[torch.Tensor] = []

        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)

            # GS 渲染器返回 RenderOutput
            render_out = renderer.render(gs, ext_iv, intr_iv)  # RenderOutput
            color = render_out.color  # (H,W,3)
            view_colors.append(color)

        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        stacked = torch.stack(view_colors, dim=0)  # (V,H,W,C)
        all_colors.append(stacked)

    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {
        "color": torch.stack(all_colors, dim=0),  # (B,V,H,W,C)
        "gaussians": gaussians,  # 保留 GS 供其他用途
    }
    return result


# =====================================================================
# 前向传播 - 共享的 Trellis 前向逻辑
# =====================================================================

def trellis_forward(
    system: TrellisSystem,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Trellis 前向传播：Dense Sampling → Rollout → Decode → Render

    抽取共享的前向逻辑，供训练、评估和流水线并行版本复用。

    注意：调用此函数时需要在外层使用 TrainModeGuard（训练时）或 EvalModeGuard（评估时）。

    Args:
        system: 系统组件（pipeline、renderer）
        state: TrellisState 状态对象（已挂载 batch 数据，含条件编码）
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数（用于随机种子）
        is_training: 是否为训练模式

    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B,V,H,W,C) 渲染图像
            - "meshes" 或 "gaussians": 3D 表示（用于导出）

    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.regularization: 挂载 reg_loss
        - state.views_generated.image_tensor: 挂载渲染图像
    """
    pipeline = system.pipeline

    # ---- 1. Dense Sampling（结构生成）----
    # 如果 state.coords 已经预计算（例如 teacher 复用 student 的 coords），则跳过
    if state.coords is None:
        ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()
        with torch.no_grad():
            cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
            coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
        state.coords = coords  # (N,4) - 挂载坐标供后续 rollout 使用

    # ---- 2. Rollout：执行稀疏特征采样（挂载 state.features.slat 和 state.regularization）----
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)

    # 根据配置选择 ODE 或 SDE rollout
    # 注意：推理时强制使用 ODE（确定性），训练时可选
    use_sde = is_training and cfg.rollout.type == "sde"

    if use_sde:
        rollout_sparse_sde(
            state, cfg, system, device,
            generator=generator,
            is_training=is_training,
            track_trajectory=False,
        )
    else:
        rollout_sparse(
            state, cfg, system, device,
            generator=generator,
            is_training=is_training,
        )
    latents = state.features.slat  # SparseTensor (挂载于 rollout)

    # 释放 rollout 阶段产生的显存碎片，为 decode 腾出空间
    torch.cuda.empty_cache()

    # ---- 3. 解码 & 渲染 ----
    renderer_type = cfg.renderer.type
    renderer = system.renderers[renderer_type]  # 从 renderers dict 查找

    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color": (B,V,H,W,C), "gaussians": list
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        render_out["color"] = render_out["normal"]

    state.views_generated.image_tensor = render_out["color"]  # (B,V,H,W,C) 挂载生成图用于可视化

    return render_out


# =====================================================================
# 评估 - 推理与可视化保存
# =====================================================================

@torch.no_grad()
def evaluate(
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """
    评估函数：执行推理并保存可视化结果。

    完整的评估流程：
    1. 从图像提取条件编码
    2. 执行 Dense Sampling 生成稀疏结构
    3. 执行 Sparse Sampling 生成特征
    4. 解码为 3D 表示（mesh 或 GS）
    5. 渲染多视角图像并保存
    6. 导出 mesh 文件

    注意：
        accelerator.prepare() 会为模型附加 autocast(bf16) 上下文，其中 nn.Linear
        （包括 SparseLinear）的输出会被提升为 bf16。而 spconv 在 eval 模式下走
        ops.implicit_gemm 推理路径，该路径的 ConvTunerSimple 无法为 bf16 输入
        找到合适的 GEMM 算法，导致 RuntimeError。
        因此评估前需要临时卸下 DDP/autocast 包装，使用原始模型推理。
        （参考 TRELLIS 原始代码：训练用 self.training_models，推理用 self.models）

    Args:
        system: 系统组件
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 可视化输出目录

    Returns:
        dict: 评估日志字典
    """
    if eval_loader is None:
        return {}

    pipeline = system.pipeline
    # 获取采样参数
    ss_steps, _, slat_steps, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()

    # ---- 创建 VisualIO 用于保存 ----
    visual_io = VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution, accelerator=accelerator)

    # =====================================================
    # 使用 EvalModeGuard 确保所有模型处于评估模式
    # =====================================================
    pipe_models = pipeline.pipe.models
    # ★ TRELLIS 风格：推理时换回原始模型（无 DDP / autocast(bf16)）
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models['slat_flow_model'],
        pipe_models['slat_decoder_mesh'],
        pipe_models['slat_decoder_gs'],
    ):
        # =====================================================
        # 遍历评估数据集
        # =====================================================
        for batch_idx, batch in enumerate(eval_loader):
            # 每个 batch 创建独立状态，避免跨 batch 残留
            state = TrellisState()

            # ---- 挂载 batch 数据 ----
            state.attach_batch(batch, pipeline=pipeline)  # 自动从 image_pils 生成条件编码并挂载

            # ---- 使用共享的 trellis_forward 执行前向传播 ----
            render_out = trellis_forward(
                system, state, cfg, accelerator.device, global_step, is_training=False
            )

            # ---- 保存结果（所有进程都保存各自处理的样本）----
            renderer_type = cfg.renderer.type
            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=(renderer_type != "gs"),
            )

    return {"eval_done": 1.0}
