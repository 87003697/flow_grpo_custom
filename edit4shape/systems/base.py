"""
edit4shape.systems.base
=======================

训练系统的通用基类与占位符。
提取多种模型训练代码的共性，供不同模型后端继承/扩展。
"""

# =====================================================================
# 标准库导入
# =====================================================================
import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, ClassVar

# =====================================================================
# 第三方库导入
# =====================================================================
import numpy as np
import yaml
import torch
from accelerate import Accelerator
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


# =====================================================================
# 设备分配工具 - 训练与 Guidance 的 GPU 映射
# =====================================================================

def compute_guidance_device(train_device: torch.device) -> torch.device:
    """
    根据训练设备计算 Guidance 模型设备。
    
    策略：每个节点内，前 N 张 GPU 给训练，后 N 张 GPU 给 Guidance
    - 单机 N=1: train=cuda:0, guidance=cuda:1
    - 单机 N=4: train=cuda:0-3, guidance=cuda:4-7
    - 多机每节点 N=4: 每节点 train=cuda:0-3, guidance=cuda:4-7
    
    公式: guidance_device = LOCAL_WORLD_SIZE + LOCAL_RANK
    
    环境变量说明：
    - LOCAL_WORLD_SIZE: 当前节点的进程数（即节点内训练使用的 GPU 数）
    - LOCAL_RANK: 当前进程在节点内的排名
    - WORLD_SIZE: 全局进程总数（多机时 = 节点数 * 每节点进程数）
    
    Args:
        train_device: 训练使用的设备（用于类型检查）
    
    Returns:
        torch.device: Guidance 模型设备
    
    Raises:
        ValueError: 如果训练设备不是 CUDA
        RuntimeError: 如果 GPU 数量不足
    
    Example:
        >>> # 假设 LOCAL_WORLD_SIZE=4, LOCAL_RANK=1
        >>> train_device = torch.device("cuda:1")
        >>> guidance_device = compute_guidance_device(train_device)
        >>> print(guidance_device)  # cuda:5
    """
    if train_device.type != "cuda":
        raise ValueError(f"训练设备必须是 CUDA，当前: {train_device}")
    
    # ---- 获取节点内进程数 ----
    # 优先使用 LOCAL_WORLD_SIZE（torchrun/accelerate launch 会设置）
    # 备选方案：WORLD_SIZE / NNODES 或单机回退到 WORLD_SIZE
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    
    if local_world_size is not None:
        n_train_gpus = int(local_world_size)
    else:
        # 回退方案：假设单机，或根据 NNODES 计算
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        nnodes = int(os.environ.get("NNODES", 1))
        n_train_gpus = world_size // nnodes
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Guidance 设备 = 节点内训练GPU数 + LOCAL_RANK
    guidance_idx = n_train_gpus + local_rank
    
    # 检查设备是否存在
    n_gpus = torch.cuda.device_count()
    if guidance_idx >= n_gpus:
        raise RuntimeError(
            f"Guidance 需要 cuda:{guidance_idx}，但本节点只有 {n_gpus} 个 GPU。\n"
            f"当前配置：节点内训练进程数={n_train_gpus}, LOCAL_RANK={local_rank}\n"
            f"需要每节点至少 {n_train_gpus * 2} 张 GPU（前 {n_train_gpus} 张训练，后 {n_train_gpus} 张 Guidance）。"
        )
    
    return torch.device(f"cuda:{guidance_idx}")


# NOTE: SpecifyGradient 已移至 edit4shape/guidance/base.py
# 统一使用: from edit4shape.guidance.base import SpecifyGradient


# =====================================================================
# 工具函数 - CFG 混合
# =====================================================================

def mix_cfg(
    cond_pred: torch.Tensor, 
    uncond_pred: torch.Tensor, 
    scale: float, 
    uncond_mode: str = "detach"
) -> torch.Tensor:
    """
    Classifier-Free Guidance (CFG) 混合函数。
    
    CFG 是一种在扩散模型中增强条件生成质量的技术。
    公式: output = cond_pred + scale * (cond_pred - uncond_pred)
    
    Args:
        cond_pred: 条件预测结果，形状 (B,T,C) 或 (N,C)
        uncond_pred: 无条件预测结果，形状与 cond_pred 相同，可为 None
        scale: CFG 缩放因子，通常 > 1.0 以增强条件效果
        uncond_mode: 梯度处理模式
            - "detach": 对 uncond_pred 断开梯度（默认）
            - "mirror": 对 cond_pred 断开梯度
            - "none": 保持两者梯度

    Returns:
        混合后的预测结果，形状与输入相同
    """
    if uncond_pred is None:
        return cond_pred  # (B,T,C)
    if uncond_mode == "detach":
        uncond_pred = uncond_pred.detach()  # (B,T,C)
    if uncond_mode == "mirror":
        cond_pred = cond_pred.detach()  # (B,T,C)
    return cond_pred + scale * (cond_pred - uncond_pred)  # (B,T,C)


# =====================================================================
# ModeGuard - 模块模式上下文管理器
# =====================================================================

class ModeGuard:
    """
    模块模式上下文管理器。
    
    用于临时将模块切换到指定模式（train 或 eval），并在退出时恢复原状态。
    支持同时管理多个模块，确保 BatchNorm、Dropout 等层在不同模式下行为正确。
    
    Usage:
        with ModeGuard(model1, model2, training=True):
            output = model1(input)  # train 模式
        # 退出后自动恢复原来的 training 状态
    """

    def __init__(self, *modules: Any, training: bool = False):
        """
        Args:
            *modules: 要管理的 nn.Module 实例（自动过滤 None）
            training: 目标模式，True 为训练模式，False 为评估模式
        """
        self.modules = [m for m in modules if m is not None]
        self.training = training
        self.states: List[bool] = []

    def __enter__(self):
        """进入上下文：保存状态并切换到目标模式。"""
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.train(self.training)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复原始训练状态。"""
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


def TrainModeGuard(*modules: Any) -> ModeGuard:
    """训练模式守卫。等价于 ModeGuard(..., training=True)。"""
    return ModeGuard(*modules, training=True)


def EvalModeGuard(*modules: Any) -> ModeGuard:
    """评估模式守卫。等价于 ModeGuard(..., training=False)。"""
    return ModeGuard(*modules, training=False)


# =====================================================================
# BaseConfig - 配置基类
# =====================================================================

@dataclass
class BaseConfig:
    """
    训练配置基类，包含所有模型通用的配置字段。
    子类（如 TrellisConfig）可继承并添加模型特定字段。
    """
    config_path: Optional[str] = None
    run_name: str = "run"
    logdir: str = "./logs"
    seed: int = 42
    eval_only: bool = False
    num_epochs: int = 100
    batch_size: int = 1
    eval_batch_size: int = 1
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    checkpoint: Optional[str] = None
    
    # 子配置字典（子类可覆盖具体结构）
    train: Dict[str, Any] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    renderer: Dict[str, Any] = field(default_factory=dict)
    guidance: Dict[str, Any] = field(default_factory=dict)
    freq: Dict[str, Any] = field(default_factory=dict)

    def save_yaml(self, path: str) -> None:
        """将当前配置保存为 YAML。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_from_file(cls, path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> "BaseConfig":
        """从文件构造配置（子类需实现具体逻辑）。"""
        raise NotImplementedError("load_from_file 需由子类实现。")


# =====================================================================
# BaseState - 状态基类
# =====================================================================

@dataclass
class BaseState:
    """
    生成过程的状态容器基类。
    
    存储整个生成流程中的所有中间状态，包括：
    - 稀疏结构坐标和特征
    - 相机参数（用于渲染）
    - 条件视角（含图像、路径、嵌入）
    - 生成/编辑的视角缓存
    
    子类可根据具体模型扩展字段。
    """

    @dataclass
    class Cameras:
        """相机参数容器。"""
        c2w: Any = None         # (B,V,4,4) camera-to-world
        w2c: Any = None         # (B,V,4,4) world-to-camera
        mvp: Any = None         # (B,V,4,4) MVP 矩阵
        positions: Any = None   # (B,V,3) 相机位置
        intrinsics: Any = None  # (B,V,3,3) 内参矩阵
        light_positions: Any = None  # (B,V,3) 光源位置

    @dataclass
    class ViewsGenerated:
        """生成视角缓存。存储从 3D 表示渲染出的多视角图像。"""
        image_tensor: Any = None  # (B,V,H,W,C) 或 (B,V,C,H,W)

    @dataclass
    class ViewsEdited:
        """编辑后视角缓存。存储经过编辑后的视角图像。"""
        image_tensor: Any = None      # (B,V,C,H,W)
        trackers: Any = None          # List[FlowEditStateTracker]，用于中间步可视化

    @dataclass
    class ViewsConditioned:
        """条件视角缓存。存储输入的条件图像及其嵌入。"""
        image_pils: Any = None   # list[len=B] of PIL.Image
        paths: Any = None        # list[len=B] of str
        cond_embed: Any = None   # (B,S,C) 条件嵌入
        uncond_embed: Any = None # (B,S,C) 无条件嵌入（用于 CFG）

    @dataclass
    class Guidance:
        """Guidance 缓存。存储用于监督的指导信号。"""

    # ============== 核心状态字段 ==============
    coords: Any = None  # 稀疏结构坐标
    feats: Any = None   # 稀疏特征
    
    # ============== 子状态容器 ==============
    cameras: Cameras = field(default_factory=Cameras)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)
    views_conditioned: ViewsConditioned = field(default_factory=ViewsConditioned)
    guidance: Guidance = field(default_factory=Guidance)
    
    # ============== 数据挂载字段 ==============
    guidances_data: Any = None
    
    # ============== 可卸载属性列表（子类覆盖） ==============
    _OFFLOADABLE: ClassVar[List[str]] = []

    def attach_batch(self, batch: Dict[str, Any]) -> "BaseState":
        """
        从数据批次中提取并挂载条件、相机等信息。
        子类应覆盖此方法以处理模型特定的 batch 结构。
        """
        raise NotImplementedError("attach_batch 需由子类实现。")

    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 views_conditioned 中提取条件和无条件嵌入。
        
        Returns:
            tuple: (cond_embed, uncond_embed)
        """
        cond = self.views_conditioned.cond_embed
        uncond = self.views_conditioned.uncond_embed
        
        if cond is None:
            raise ValueError("views_conditioned.cond_embed 为空，无法提取 embeddings。")
        
        # 处理 list 格式
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=0)  # (B,S,C)
        if isinstance(uncond, list):
            uncond = torch.cat(uncond, dim=0)  # (B,S,C)
        
        # 处理多余维度
        if isinstance(cond, torch.Tensor) and cond.dim() == 4 and cond.shape[1] == 1:
            cond = cond.squeeze(1)  # (B,S,C)
        if isinstance(uncond, torch.Tensor) and uncond.dim() == 4 and uncond.shape[1] == 1:
            uncond = uncond.squeeze(1)  # (B,S,C)
        
        return cond, uncond

    # ============== 设备搬运方法 ==============
    
    @staticmethod
    def _to_device(val: Any, device: torch.device) -> Any:
        """
        将值搬到目标设备（自动处理 None、Tensor、SparseTensor、List）。
        """
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.to(device)
        if isinstance(val, list):
            return [BaseState._to_device(v, device) for v in val]
        if hasattr(val, 'feats') and hasattr(val, 'replace'):  # SparseTensor
            return val.replace(val.feats.to(device))
        if hasattr(val, 'to'):  # Mesh 等
            return val.to(device)
        return val

    def _get_attr(self, path: str) -> Any:
        """获取嵌套属性，如 'features.subs'。"""
        obj = self
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _set_attr(self, path: str, value: Any) -> None:
        """设置嵌套属性，如 'features.subs'。"""
        parts = path.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def offload_features(self) -> "BaseState":
        """将 _OFFLOADABLE 中的属性搬到 CPU，节省显存。"""
        cpu = torch.device("cpu")
        for path in self._OFFLOADABLE:
            self._set_attr(path, self._to_device(self._get_attr(path), cpu))
        return self

    def reload_features(self, device: torch.device) -> "BaseState":
        """将 _OFFLOADABLE 中的属性搬回 GPU。"""
        for path in self._OFFLOADABLE:
            self._set_attr(path, self._to_device(self._get_attr(path), device))
        return self


# =====================================================================
# System - 系统组件容器
# =====================================================================

@dataclass
class System:
    """
    系统核心组件容器。
    
    封装了训练系统的四大核心组件：
    1. pipeline: 生成管道
    2. renderer: 渲染器
    3. guidance: 指导模块
    4. optimizer: 优化器
    """

    pipeline: Any = None
    renderer: Any = None
    guidance: Any = None
    optimizer: Any = None

    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """
        设置随机种子与确定性运行环境。
        
        确保实验可复现性，设置以下随机源的种子：
        - Python random
        - NumPy random
        - PyTorch CPU/CUDA
        - cuDNN 确定性模式
        """
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_lora(
        self,
        cfg: Any,
        adapter: str = "base",
        load_path: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> "System":
        """
        准备 LoRA 适配器（子类可覆盖以添加模型特定逻辑）。
        """
        target_modules = [m for m in [self.pipeline, self.guidance] if hasattr(m, "set_adapter")]
        for module in target_modules:
            if load_path and hasattr(module, "load_adapter"):
                module.load_adapter(load_path, adapter_name=adapter)
            module.set_adapter(adapter)
        return self

    def prepare_models_and_optimizers(self, cfg: Any, accelerator: Accelerator) -> "System":
        """
        使用 Accelerate 包装模型和优化器以支持分布式训练。
        """
        if accelerator is None:
            return self
        
        items = []
        if isinstance(self.pipeline, torch.nn.Module):
            items.append(("pipeline", self.pipeline))
        if self.optimizer is not None:
            items.append(("optimizer", self.optimizer))
            
        if not items:
            return self

        prepared = accelerator.prepare(*[obj for _, obj in items])
        
        if len(items) == 1:
            prepared = [prepared]
            
        for (name, _), wrapped in zip(items, prepared):
            setattr(self, name, wrapped)
        return self


# =====================================================================
# 构建函数占位 - 子类需实现
# =====================================================================

def build_system(cfg: Any, accelerator: Accelerator) -> System:
    """构建系统组件（子类需实现）。"""
    raise NotImplementedError("build_system 需由具体模型模块实现。")


def build_dataloaders(cfg: Any, accelerator: Accelerator) -> Tuple[Any, Any]:
    """构建 DataLoader（子类需实现）。"""
    raise NotImplementedError("build_dataloaders 需由具体模型模块实现。")


# =====================================================================
# 训练/评估函数占位
# =====================================================================

def train_step(
    system: System,
    state: BaseState,
    cfg: Any,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Dict[str, torch.Tensor]:
    """
    单步训练函数（占位，子类需实现）。
    
    Returns:
        dict: 训练日志字典，包含 loss 等指标
    """
    raise NotImplementedError("train_step 需由具体模型模块实现。")


@torch.no_grad()
def evaluate(
    system: System,
    cfg: Any,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """
    评估函数（占位，子类需实现）。
    
    Returns:
        dict: 评估日志字典
    """
    raise NotImplementedError("evaluate 需由具体模型模块实现。")


# =====================================================================
# 运行目录与 CSV 工具
# =====================================================================

def build_run_paths(cfg: Any, accelerator: Accelerator) -> Tuple[Path, Path, Path, Path]:
    """
    创建实验运行目录结构并保存配置。
    
    目录结构：
    {logdir}/{run_name}/
    ├── config.yaml
    ├── run_command.txt
    ├── logs/
    └── visualizations/
        ├── train/
        └── eval/
    """
    run_root = Path(cfg.logdir) / (cfg.run_name if cfg.run_name else "run")
    logs_dir = run_root / "logs"
    visuals_train_dir = run_root / "visualizations" / "train"
    visuals_eval_dir = run_root / "visualizations" / "eval"
    
    if accelerator.is_main_process:
        run_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        visuals_train_dir.mkdir(parents=True, exist_ok=True)
        visuals_eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置（使用 ml_collections.ConfigDict 的 to_dict 方法）
        with (run_root / "config.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(cfg.to_dict(), sort_keys=False))
        
        # 保存启动命令
        with (run_root / "run_command.txt").open("w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))
    
    return run_root, logs_dir, visuals_train_dir, visuals_eval_dir


# =====================================================================
# CheckpointIO - 检查点管理
# =====================================================================

@dataclass
class CheckpointIO:
    """
    检查点读写封装类。
    使用 Accelerate 的 save_state/load_state 进行分布式安全的检查点操作。
    """

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(self, system: System, state: BaseState, cfg: Any, epoch: int, global_step: int) -> None:
        """保存检查点。"""
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"
        target.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(target))
        
        if self.accelerator.is_main_process:
            meta = {"epoch": int(epoch), "global_step": int(global_step)}
            with (target / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        self.accelerator.wait_for_everyone()

    def load(self, path: str, mode: str = "train") -> int:
        """加载检查点。"""
        cp = path
        if not (isinstance(cp, str) and cp):
            self.start_epoch = 0
            return 0
        
        root = Path(cp)
        if not (root.is_dir() and (root / "state.json").exists() and root.name.startswith("checkpoint_")):
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(str(root))
        self.accelerator.wait_for_everyone()
        
        meta_path = root / "meta.json"
        assert meta_path.exists(), f"meta.json missing in {root}"
        meta = json.load(meta_path.open("r", encoding="utf-8")) or {}
        epoch_val = meta["epoch"]
        step_val = meta["global_step"]
        
        self.start_epoch = int(epoch_val) + 1 if mode == "train" else 0
        self.start_global_step = int(step_val)
        return self.start_epoch


# =====================================================================
# 从 utils 导入通用工具（避免重复定义）
# =====================================================================

from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO


# =====================================================================
# 主函数模板（供参考）
# =====================================================================

def main_template(
    cfg: Any,
    build_system_fn,
    build_dataloaders_fn,
    train_step_fn,
    evaluate_fn,
    state_cls,
) -> None:
    """
    通用主函数模板。
    
    Args:
        cfg: 配置对象
        build_system_fn: 系统构建函数
        build_dataloaders_fn: DataLoader 构建函数
        train_step_fn: 训练步骤函数
        evaluate_fn: 评估函数
        state_cls: 状态类（如 TrellisState）
    """
    # Step 1: 环境设置
    System.setup_env_and_seed(cfg)

    # Step 2: 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    # Step 3: 创建运行目录
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    # Step 4: 构建数据加载器
    train_loader, eval_loader = build_dataloaders_fn(cfg, accelerator)

    # Step 5: 构建系统组件
    system = build_system_fn(cfg, accelerator)
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # Step 6: 检查点管理
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # Step 7: 评估模式
    if cfg.eval_only:
        eval_log = evaluate_fn(
            system, cfg, accelerator, 
            epoch=start_epoch, 
            global_step=global_step, 
            eval_loader=eval_loader, 
            visuals_eval_dir=visuals_eval_dir
        )
        return

    # Step 8: 训练循环
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            
            # 子类需实现具体的状态初始化和训练步骤
            state = state_cls()
            # state = state.attach_batch(batch)
            # train_log = train_step_fn(system, state, cfg, accelerator, epoch, global_step)
            # train_logger.log_step(train_log, batch_size, global_step, epoch)

        # 周期性评估
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate_fn(
                system, cfg, accelerator, 
                epoch=epoch, 
                global_step=global_step, 
                eval_loader=eval_loader, 
                visuals_eval_dir=visuals_eval_dir
            )

        # 周期性保存
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)
