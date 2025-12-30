"""
Trellis 单 renderer 版（适配 Gen2Turbo Trellis 逻辑）。

本模块实现了基于 TRELLIS 架构的 3D 生成系统，支持从单张图像生成 3D 模型。
核心流程：图像条件编码 -> 结构采样 (Dense Sampling) -> 特征采样 (Sparse Sampling) -> 解码 -> 渲染

特性：
- 单 renderer，训练/推理共用统一 rollout。
- 必需稠密结构 coords，若缺失直接报错。
- 统一步数 num_steps_sparse，训练/推理一致。
- 全程 CFG：每步都跑 cond/uncond，再 mix_cfg。

主要组件：
1. TrellisState: 存储生成状态（从 trellis.py 导入）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. evaluate: 评估循环，生成 mesh 并保存可视化结果
4. main: 流水线并行版训练主循环

依赖：
- TRELLIS 参考实现 (_reference_codes/TRELLIS)
- Accelerate 分布式训练库
- nvdiffrast (mesh 渲染) 或 Gaussian Splatting (GS 渲染)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# =====================================================================
# 第三方库导入
# =====================================================================
from PIL import Image
import numpy as np
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点，节省显存
from tqdm import tqdm

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.datasets.trellis import TrellisDataConfig, TrellisDataModule

# =====================================================================
# TRELLIS 参考实现路径设置
# 将 TRELLIS 参考代码目录加入 Python 路径，以便导入其模块
# =====================================================================
import os
import sys
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
# 包含 coords (坐标) 和 feats (特征) 两个主要属性
from trellis.modules.sparse import SparseTensor

# =====================================================================
# Guidance 模块（使用流水线并行版本）
# =====================================================================

from edit4shape.guidance.base import GuidanceResult


def create_guidance_pp(cfg, train_device):
    """
    创建流水线并行版本的 Guidance 实例。
    
    使用 local_pp.py 中支持异步接口的 LocalGuidance。
    传递完整配置，以便从 cfg.guidance.flowedit 读取算法参数，
    从 cfg.train.loss 读取 loss 权重。
    """
    from edit4shape.guidance.backends.local_pp import LocalGuidance
    return LocalGuidance(cfg, train_device)


# =====================================================================
# 从 trellis.py 和 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    mix_cfg,
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    System,
    CheckpointIO,
    build_run_paths,
)
from edit4shape.systems.trellis import (
    _CONFIG,
    TrellisState,
    trellis_forward,
    build_dataloaders,
    build_system,
    evaluate,
)
from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO, LossDict


# =====================================================================
# 注意：build_system 从 trellis.py 导入
# 在 main 中调用时传入 guidance_factory=create_guidance_pp
# =====================================================================


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    完整的训练/评估流程：
    1. 解析配置文件
    2. 设置环境与随机种子
    3. 初始化 Accelerator（分布式/混合精度）
    4. 创建运行目录
    5. 构建数据加载器
    6. 构建系统组件（pipeline, renderer, optimizer）
    7. 加载检查点（如有）
    8. 执行训练循环或评估
    
    配置文件示例：
        python -m edit4shape.systems.trellis --config=configs/trellis.py
    
    关键配置项：
        - cfg.eval_only: True 时仅执行评估
        - cfg.num_epochs: 训练总 epoch 数
        - cfg.eval_freq: 评估频率（每 N 个 epoch）
        - cfg.save_freq: 保存检查点频率
        - cfg.checkpoint: 恢复训练的检查点路径
    
    Args:
        argv: 命令行参数（由 absl.app.run 传入，本函数不使用）
    """
    del argv  # absl.app.run 会传入 argv；本函数不使用
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
    # 配置混合精度训练和梯度累积
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,  # "no", "fp16", "bf16"
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq)

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator, guidance_factory=create_guidance_pp)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式（仅评估不训练）
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator, 
            epoch=start_epoch, 
            global_step=global_step, 
            eval_loader=eval_loader, 
            visuals_eval_dir=visuals_eval_dir
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（流水线并行版）
    # =====================================================
    # 初始化训练日志记录器（自动处理梯度累积）
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    grad_accum_steps = cfg.train.gradient_accumulation_steps
    
    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models
    
    def _compute_loss_and_backward(state: TrellisState) -> Dict[str, Any]:
        """
        计算 loss 并反向传播。
        
        所有需要的数据都已挂载在 state 中：
        - state.guidance: 包含 loss_ssim, loss_lpips, loss_latent_mse
        - state.regularization: 包含 reg_loss, reg_metric
        """
        # ---- 统一 Loss 管理 ----
        losses = LossDict(device=accelerator.device)
        guidance_weights = system.guidance.get_loss_weights()
        
        losses.add("ssim", state.guidance.loss_ssim, weight=guidance_weights["ssim"])
        losses.add("lpips", state.guidance.loss_lpips, weight=guidance_weights["lpips"])
        losses.add("latent_mse", state.guidance.loss_latent_mse, weight=guidance_weights["latent_mse"])
        losses.add("reg", state.regularization.reg_loss, weight=cfg.train.loss.reg)
        
        # ---- 反向传播 ----
        total_loss = losses.total()
        accelerator.backward(total_loss / grad_accum_steps)  # 除以 grad_accum_steps 做平均
        
        # ---- 构建日志 ----
        logs = losses.to_logs()
        if state.regularization.reg_metric is not None:
            logs["loss/reg"] = state.regularization.reg_metric
        return logs

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        # 设置分布式采样器的 epoch（确保各进程数据不同）
        train_loader.sampler.set_epoch(epoch)
        
        # 将 dataloader 转为 iterator，便于手动控制 gradient accumulation
        data_iter = iter(train_loader)
        
        # 流水线主循环
        epoch_done = False
        
        while not epoch_done:
            n_micro = 0
            pending_states: List[TrellisState] = []
            
            with TrainModeGuard(
                pipe_models['slat_flow_model'],
                pipe_models['slat_decoder_mesh'],
                pipe_models['slat_decoder_gs'],
            ):
                for mb_idx in range(grad_accum_steps):
                    # ---- 1. 获取下一个 batch ----
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        epoch_done = True
                        break
                    
                    global_step += 1
                    n_micro += 1
                    
                    # ---- 2. 准备当前 micro-batch ----
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=system.pipeline)  # 自动从 image_pils 生成条件编码并挂载
                    pending_states.append(state)
                    
                    # ---- 3. Trellis 前向（在默认 stream 上）----
                    # 注意：trellis_forward 会自动将 reg_loss/reg_metric 挂载到 state.regularization
                    render_out = trellis_forward(
                        system, state, cfg, accelerator.device, global_step, is_training=True
                    )
                    comp_rgb = render_out["color"]  # (B,V,H,W,C)
                    
                    # ---- 4. 异步提交 guidance（不阻塞）----
                    # 使用 FIFO 队列，可安全地先提交再等待
                    system.guidance.submit_async(comp_rgb, state.views_conditioned.image_pils)
                    
                    # ---- 5. 等待并处理已完成的 guidance（流水线并行）----
                    # 当队列中有 2 个任务时，说明前一个已经有足够时间完成，可以取出处理
                    # 这样实现真正的双缓冲：当前 batch 的 guidance 在后台执行，
                    # 同时处理上一个 batch 的 backward
                    if mb_idx > 0:
                        result = system.guidance.wait_and_get()
                        pending_states[mb_idx - 1].attach_guidance_result(result)
                        train_log = _compute_loss_and_backward(pending_states[mb_idx - 1])
                        train_logger.accumulate(train_log, batch_size=1)
                
                # ---- 6. 清空队列：处理所有剩余的 pending guidance ----
                # 已处理的数量 = mb_idx（因为 mb_idx > 0 时处理了 mb_idx - 1）
                # 如果循环正常结束（没有 break），remaining_idx = n_micro - 1
                # 需要处理 pending_states[n_micro - queue_size : n_micro]
                processed_count = max(0, n_micro - 1) if n_micro > 0 else 0  # 已经处理了多少个
                while system.guidance.has_pending() and processed_count < len(pending_states):
                    result = system.guidance.wait_and_get()
                    pending_states[processed_count].attach_guidance_result(result)
                    train_log = _compute_loss_and_backward(pending_states[processed_count])
                    train_logger.accumulate(train_log, batch_size=1)
                    processed_count += 1
                
                # ---- 7. 优化器步进 ----
                if n_micro > 0:
                    system.optimizer.step()
                    system.optimizer.zero_grad()
            
            # ---- 8. 日志和可视化 ----
            if n_micro > 0:
                # 使用最后一个 state 做可视化
                if pending_states and accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                    visual_io.save_batch_train(
                        state=pending_states[-1],
                        epoch=epoch,
                        step=global_step,
                    )
                
                # flush 日志（分布式聚合 + 发射）
                train_logger.flush(global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator, 
                epoch=epoch, 
                global_step=global_step, 
                eval_loader=eval_loader, 
                visuals_eval_dir=visuals_eval_dir
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            # 使用最后一个 state 保存
            last_state = pending_states[-1] if pending_states else TrellisState()
            ckpt_io.save(system, last_state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
