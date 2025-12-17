"""
Trellis 单 renderer 版（适配 Gen2Turbo Trellis 逻辑）。

特性：
- 单 renderer，训练/推理共用统一 rollout。
- 必需稠密结构 coords，若缺失直接报错。
- 统一步数 num_steps_sparse，训练/推理一致。
- 全程 CFG：每步都跑 cond/uncond，再 mix_cfg。
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

import torch
from accelerate import Accelerator


# === 实用函数 ===
def mix_cfg(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, scale: float, uncond_mode: str = "detach") -> torch.Tensor:
    """
    与参考实现一致的 CFG 混合。
    uncond_mode: detach/mirror/none。
    """
    if uncond_pred is None:
        return cond_pred  # (B,T,C)
    if uncond_mode == "detach":
        uncond_pred = uncond_pred.detach()  # (B,T,C)
    if uncond_mode == "mirror":
        cond_pred = cond_pred.detach()  # (B,T,C)
    return cond_pred + scale * (cond_pred - uncond_pred)  # (B,T,C)


def scheduler_step_at_index(scheduler: Any, t: torch.Tensor, latents: torch.Tensor, noise_pred: torch.Tensor) -> Any:
    """
    兼容参考实现的安全 step，若 scheduler 不支持 index_for_timestep，则直接 step。
    """
    if hasattr(scheduler, "index_for_timestep"):
        _ = scheduler.index_for_timestep(t, scheduler.timesteps)  # ()
    return scheduler.step(noise_pred, t, latents)  # (obj: prev_sample/pred_original_sample)


def stage2_rollout_step(
    pipeline: Any,
    scheduler: Any,
    latents: torch.Tensor,
    coords: torch.Tensor,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: Optional[torch.Tensor],
    step_index: int,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    单步 rollout：返回 next_feats、velocity_preds、final_feats_ft。
    """
    batch_size = latents.shape[0]  # 标量 ()，B
    t = scheduler.timesteps[step_index]  # 标量 ()
    t_expanded = t.expand(batch_size)  # (B,)

    cond_pred = pipeline.denoise(
        noisy_input=latents,  # (B,T,C)
        timesteps=t_expanded,  # (B,)
        cond_embeddings=cond_embeddings,  # (B,S,C)
        coords=coords,  # (B,T,4)
    )  # (B,T,C)

    uncond_pred = None  # (B,T,C) 或 None
    if uncond_embeddings is not None:
        uncond_pred = pipeline.denoise(
            noisy_input=latents,  # (B,T,C)
            timesteps=t_expanded,  # (B,)
            uncond_embeddings=uncond_embeddings,  # (B,S,C)
            coords=coords,  # (B,T,4)
        )  # (B,T,C)

    velocity_preds = mix_cfg(
        cond_pred=cond_pred,  # (B,T,C)
        uncond_pred=uncond_pred,  # (B,T,C) 或 None
        scale=float(cfg.guidance_scale),  # 标量 ()
        uncond_mode=cfg.uncond_mode_rollout,  # str
    )  # (B,T,C)

    step_out = scheduler_step_at_index(scheduler, t, latents, velocity_preds)  # (obj 包含 prev_sample/pred_original_sample)
    next_feats = step_out.prev_sample  # (B,T,C)
    final_feats_ft = getattr(step_out, "pred_original_sample", velocity_preds)  # (B,T,C)

    return next_feats, velocity_preds, final_feats_ft


def _zeros_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=value.device, dtype=value.dtype)  # ()


def compute_kl_step_regularization(
    scheduler: Any,
    batch_size: int,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_mode: str,
    latents_ori: torch.Tensor,
    t: torch.Tensor,
    final_pred_ft: torch.Tensor,
    pipeline: Any,
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    占位 KL 正则，需按项目替换。返回 (reg_scalar, grad_norm)。
    """
    reg_scalar = _zeros_like(final_pred_ft)  # ()
    grad_norm = _zeros_like(final_pred_ft)  # ()
    return reg_scalar, grad_norm


def compute_score_distillation_step_regularization(
    method: str,
    scheduler: Any,
    batch_size: int,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_mode: str,
    pipeline: Any,
    final_latent_ft: torch.Tensor,
    latents_x_t: torch.Tensor,
    t: torch.Tensor,
    weight_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    占位 SDS/CSD 正则，需按项目替换。返回 (reg_scalar, grad_norm)。
    """
    reg_scalar = _zeros_like(final_latent_ft)  # ()
    grad_norm = _zeros_like(final_latent_ft)  # ()
    return reg_scalar, grad_norm


# === 配置与状态 ===
@dataclass
class TrellisConfig:
    """Trellis 训练/推理配置。"""

    config_path: Optional[str]
    run_name: str
    logdir: str
    seed: int
    eval_only: bool
    num_epochs: int
    train: Dict[str, Any]
    sample: Dict[str, Any]
    renderer: Dict[str, Any]
    guidance: Dict[str, Any]
    optimizer: Dict[str, Any]
    loss: Dict[str, Any]
    exporter: Optional[Dict[str, Any]] = None
    checkpoint: Optional[str] = None
    gradient_accumulation_steps: int = 1
    num_steps_sparse: int = 50
    num_steps_dense: int = 25
    guidance_scale: float = 5.0
    uncond_mode_rollout: str = "detach"
    uncond_mode_reg: str = "detach"
    reg_type: str = "kl"
    lambda_reg: float = 0.0
    lambda_distill: float = 0.0
    mixed_precision: str = "fp16"
    use_lora: bool = True
    eval_freq: int = 0
    save_freq: int = 0

    def save_yaml(self, path: str) -> None:
        """将当前配置保存为 YAML。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def load_from_file(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> "TrellisConfig":
        """从文件/flags 构造 TrellisConfig。"""
        raise NotImplementedError("load_from_file 尚未实现。")


@dataclass
class TrellisState:
    """仅存储核心稀疏特征占位，并挂载空的视角/条件占位类。"""

    @dataclass
    class Conditions:
        """条件编码占位。"""

    @dataclass
    class Cameras:
        """相机参数占位。"""

    @dataclass
    class ViewsGenerated:
        """生成视角缓存占位。"""

    @dataclass
    class ViewsEdited:
        """编辑后视角缓存占位。"""

    @dataclass
    class Guidance:
        """guidance 缓存占位。"""

    coords: Any = None
    feats: Any = None
    cameras: Cameras = field(default_factory=Cameras)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)
    conditions: Conditions = field(default_factory=Conditions)
    guidance: Guidance = field(default_factory=Guidance)
    space_cache: Any = None
    conditions_data: Any = None  # 挂载 batch["Conditions"]
    guidances_data: Any = None  # 挂载 batch["Guidances"]

    def attach_batch(self, batch: Dict[str, Any]) -> "TrellisState":
        """从 batch 挂载条件与指导数据。"""
        self.conditions_data = batch.get("Conditions", None)
        self.guidances_data = batch.get("Guidances", None)
        return self

    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 conditions_data 中提取 cond/uncond embeddings。"""
        condition_utils = self.conditions_data
        if condition_utils is None:
            raise ValueError("TrellisState.conditions_data 为空，无法提取 embeddings。")
        cond_embeddings = condition_utils.image_embeddings  # list/Tensor
        if isinstance(cond_embeddings, list):
            cond_embeddings = torch.cat(cond_embeddings, dim=0)  # (B,S,C)
        if isinstance(cond_embeddings, torch.Tensor) and cond_embeddings.dim() == 4 and cond_embeddings.shape[1] == 1:
            cond_embeddings = cond_embeddings.squeeze(1)  # (B,S,C) 或 (B,C)

        uncond_embeddings = condition_utils.uncond_image_embeddings  # list/Tensor
        if isinstance(uncond_embeddings, list):
            uncond_embeddings = torch.cat(uncond_embeddings, dim=0)  # (B,S,C)
        if isinstance(uncond_embeddings, torch.Tensor) and uncond_embeddings.dim() == 4 and uncond_embeddings.shape[1] == 1:
            uncond_embeddings = uncond_embeddings.squeeze(1)  # (B,S,C) 或 (B,C)
        return cond_embeddings, uncond_embeddings


@dataclass
class System:
    """系统组件：pipeline(原 geometry) / renderer / guidance / optimizer。"""

    pipeline: Any = None
    renderer: Any = None
    guidance: Any = None
    optimizer: Any = None

    @staticmethod
    def setup_env_and_seed(cfg: TrellisConfig) -> None:
        """设置随机种子与确定性。"""
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_lora(
        self,
        cfg: TrellisConfig,
        adapter: str = "base",
        load_path: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> "System":
        """
        LoRA 适配占位：若组件支持 set_adapter/load_adapter 则调用。
        """
        target_modules = [m for m in [self.pipeline, self.guidance] if hasattr(m, "set_adapter")]
        for module in target_modules:
            if load_path and hasattr(module, "load_adapter"):
                module.load_adapter(load_path, adapter_name=adapter)
            module.set_adapter(adapter)
        return self

    def prepare_models_and_optimizers(self, cfg: TrellisConfig, accelerator: Accelerator) -> "System":
        """
        仅包装可训练模块：pipeline/optimizer。
        """
        if accelerator is None:
            return self
        items = [(name, obj) for name, obj in (("pipeline", self.pipeline), ("optimizer", self.optimizer)) if obj is not None]
        prepared = accelerator.prepare(*[obj for _, obj in items])
        for (name, _), wrapped in zip(items, prepared):
            setattr(self, name, wrapped)
        return self


# === 构建函数（需按项目实际替换） ===
def build_system(cfg: TrellisConfig, accelerator: Accelerator) -> System:
    """
    构建 geometry/renderer/guidance/optimizer。
    需根据项目自定义，此处为占位。
    """
    raise NotImplementedError("build_system 需按项目实现。")


def build_dataloaders(cfg: TrellisConfig, accelerator: Accelerator) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """构造训练与评估 DataLoader。"""
    raise NotImplementedError("build_dataloaders 需按项目实现。")


# === Rollout（训练/评估共用） ===
def rollout_sparse(
    state: TrellisState,
    cfg: TrellisConfig,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """
    稠密结构 + 稀疏去噪 rollout（训练/评估共用）。
    返回 {"latents": (B,T,C), "coords": (1,T,4)}。
    """
    cond_embeddings, uncond_embeddings = state.extract_embeddings()  # (B,S,C),(B,S,C)
    cond_embeddings = cond_embeddings.to(device)  # (B,S,C)
    uncond_embeddings = uncond_embeddings.to(device)  # (B,S,C)

    condition_utils = state.conditions_data
    coords = system.pipeline.generate_structure(condition_utils, steps=cfg.num_steps_dense)  # (T,4) or (1,T,4)
    if coords is None:
        raise ValueError("generate_structure 返回 None，无法继续。")
    if isinstance(coords, torch.Tensor) and coords.dim() == 2:
        coords = coords.unsqueeze(0)  # (1,T,4)
    coords = coords.to(device=device, dtype=torch.int32)  # (1,T,4)
    coords = coords.expand(cond_embeddings.shape[0], -1, -1)  # (B,T,4)

    batch_size = cond_embeddings.shape[0]  # ()
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    latents = system.pipeline.init_latents(batch_size=batch_size, coords=coords, generator=generator)  # (B,T,C)

    scheduler = system.pipeline.get_scheduler()
    scheduler.set_timesteps(cfg.num_steps_sparse, device=device)

    for step_idx in range(len(scheduler.timesteps) - 1):
        next_latents, velocity_preds, final_feats_ft = stage2_rollout_step(
            pipeline=system.pipeline,  # pipeline
            scheduler=scheduler,  # scheduler
            latents=latents,  # (B,T,C)
            coords=coords,  # (B,T,4)
            cond_embeddings=cond_embeddings,  # (B,S,C)
            uncond_embeddings=uncond_embeddings,  # (B,S,C)
            step_index=step_idx,  # 标量 ()
            cfg=cfg,  # cfg
        )  # (B,T,C),(B,T,C),(B,T,C)
        latents = next_latents  # (B,T,C)

    return {"latents": latents, "coords": coords}


# === Loss 与指导 ===
def compute_guidance(
    guidance_module: Any,
    out: Dict[str, Any],
    state: TrellisState,
    cfg: TrellisConfig,
    step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算 guidance_loss（单标量），内部按 loss_* 与对应 lambda_* 聚合。
    """
    guidance_rgb = out["comp_rgb"].permute(0, 3, 1, 2)  # (B,3,H,W) 或 (B,4,3,H,W) -> (B,3,H,W) 视 renderer 输出而定
    batch_extra = getattr(state, "batch_data", {}) or {}
    guidance_out = guidance_module(
        guidance_rgb,
        conditions=getattr(state, "guidances_data", None),
        **batch_extra,
    )
    guidance_loss = torch.zeros((), device=guidance_rgb.device, dtype=guidance_rgb.dtype)  # ()
    log_items: Dict[str, Any] = {}
    for name, value in guidance_out.items():
        log_items[f"guidance/{name}_{step}"] = value
        if name.startswith("loss_"):
            lambda_name = name.replace("loss_", "lambda_")
            weight = float(cfg.loss.get(lambda_name, 1.0))
            guidance_loss = guidance_loss + value * weight  # ()
    if cfg.lambda_distill > 0.0:
        distill_loss = guidance_out.get("loss_distill", None)
        if distill_loss is not None:
            guidance_loss = guidance_loss + cfg.lambda_distill * distill_loss  # ()
            log_items["loss/distill"] = distill_loss
    return guidance_loss, log_items


# === 训练 ===
def train_edit4shape(
    system: System,
    state: TrellisState,
    cfg: TrellisConfig,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Dict[str, torch.Tensor]:
    """单 renderer 训练步，含可选逐步正则。"""
    device = accelerator.device
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    with accelerator.accumulate(system.pipeline if system.pipeline is not None else system.renderer):
        # rollout（可带逐步正则）
        condition_utils = state.conditions_data
        cond_embeddings, uncond_embeddings = state.extract_embeddings()  # (B,S,C),(B,S,C)
        cond_embeddings = cond_embeddings.to(device)  # (B,S,C)
        uncond_embeddings = uncond_embeddings.to(device)  # (B,S,C)

        coords = system.pipeline.generate_structure(condition_utils, steps=cfg.num_steps_dense)  # (T,4) or (1,T,4)
        if coords is None:
            raise ValueError("训练阶段 generate_structure 返回 None。")
        coords = coords.to(device=device, dtype=torch.int32)  # (1,T,4) 或 (T,4)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # (1,T,4)
        coords = coords.expand(cond_embeddings.shape[0], -1, -1)  # (B,T,4)

        batch_size = cond_embeddings.shape[0]  # ()
        scheduler = system.pipeline.get_scheduler()
        scheduler.set_timesteps(cfg.num_steps_sparse, device=device)

        latents = system.pipeline.init_latents(batch_size=batch_size, coords=coords, generator=generator)  # (B,T,C)

        for step_idx in range(len(scheduler.timesteps) - 1):
            next_latents, velocity_preds, final_feats_ft = torch.utils.checkpoint.checkpoint(
                stage2_rollout_step,
                system.pipeline,  # pipeline
                scheduler,  # scheduler
                latents,  # (B,T,C)
                coords,  # (B,T,4)
                cond_embeddings,  # (B,S,C)
                uncond_embeddings,  # (B,S,C)
                step_idx,  # 标量 ()
                cfg,  # cfg
                use_reentrant=False,
            )  # (B,T,C),(B,T,C),(B,T,C)


            latents = next_latents  # (B,T,C)

        sparse_latent = (
            system.pipeline.backend.tokens_to_sparse(latents, coords)
        )  # (B,T,C) 或 Sparse
        render_batch = {
            "space_cache": system.pipeline.precompute_cache(sparse_latent),
            "Conditions": state.conditions_data,
            "Guidances": state.guidances_data,
        }
        state.space_cache = render_batch["space_cache"]
        state.coords = coords
        out = system.renderer(**render_batch)  # renderer 输出

        guidance_loss, guidance_logs = compute_guidance(system.guidance, out, state, cfg, step=global_step)
        loss = guidance_loss  # ()

        loss = loss / float(accelerator.gradient_accumulation_steps)  # ()
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            if system.optimizer is not None:
                system.optimizer.step()
                system.optimizer.zero_grad()
        train_log = {
            "loss_total": loss.detach() * float(accelerator.gradient_accumulation_steps),  # ()
            "loss_guidance": guidance_loss.detach(),  # ()
        }
        train_log.update(guidance_logs)
        return train_log


# === 评估 ===
@torch.no_grad()
def evaluate(
    system: System,
    state: TrellisState,
    cfg: TrellisConfig,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
) -> Dict[str, Any]:
    """
    评估：rollout -> renderer，返回基础日志占位。
    """
    if eval_loader is None:
        return {}
    logs: Dict[str, Any] = {}
    metrics = EvalMetricLogger()
    pipeline = system.pipeline
    renderer = system.renderer
    with EvalModeGuard(pipeline, renderer):
        for batch in eval_loader:
            # TODO: 编写评估前向与指标计算
            _ = batch  # 占位，避免未使用变量告警
            # 例如：out = system.forward(...); metrics.update(...); 可在此调用 metrics.distributed_mean
            pass
    log_dict = metrics.to_global_log_dict(accelerator)
    return log_dict or {}


# === 记录与工具 ===
def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """追加写入 CSV（若不存在则写表头）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_visualizations(visuals: Dict[str, Any], out_dir: Path, prefix: str) -> None:
    """
    保存可视化结果占位。
    """
    if not visuals:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in visuals.items():
        placeholder = out_dir / f"{prefix}_{name}.txt"
        with placeholder.open("w", encoding="utf-8") as f:
            f.write("TODO: save visualization content here.")


def build_run_paths(cfg: TrellisConfig, accelerator: Accelerator) -> Tuple[Path, Path, Path, Path]:
    """创建运行目录并保存配置/启动命令。"""
    run_root = Path(cfg.logdir) / (cfg.run_name if cfg.run_name else "trellis_run")
    logs_dir = run_root / "logs"
    visuals_train_dir = run_root / "visualizations" / "train"
    visuals_eval_dir = run_root / "visualizations" / "eval"
    if accelerator.is_main_process:
        run_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        visuals_train_dir.mkdir(parents=True, exist_ok=True)
        visuals_eval_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_yaml(str(run_root / "config.yaml"))
        with (run_root / "run_command.txt").open("w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))
    return run_root, logs_dir, visuals_train_dir, visuals_eval_dir


@dataclass
class CheckpointIO:
    """封装 checkpoint 读写。"""

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(self, system: System, state: TrellisState, cfg: TrellisConfig, epoch: int, global_step: int) -> None:
        """
        保存当前状态到 ckpt_dir/checkpoint_{epoch}_{global_step}。
        """
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
        """
        加载指定 checkpoint_XXXX 目录。
        """
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
        epoch_val = meta["epoch"]  # ()
        step_val = meta["global_step"]  # ()
        self.start_epoch = int(epoch_val) + 1 if mode == "train" else 0
        self.start_global_step = int(step_val)
        return self.start_epoch


class EvalModeGuard:
    """上下文管理：进入 eval，退出恢复原 training 状态。"""

    def __init__(self, *modules: Any):
        self.modules = [m for m in modules if m is not None]
        self.states = []

    def __enter__(self):
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


class MetricLoggerBase:
    """指标聚合基础类。"""

    @staticmethod
    def emit_logs(log_dict: Optional[Dict[str, Any]], accelerator: Accelerator, csv_path: Path, global_step: int, epoch: int) -> None:
        if not log_dict:
            return
        if accelerator.is_main_process:
            row = {"global_step": global_step, "epoch": epoch}
            row.update({k: float(v) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()})
            append_csv_row(csv_path, row)
        accelerator.log(log_dict, step=global_step)

    @staticmethod
    def distributed_mean(values_np: Any, accelerator: Accelerator) -> float:
        """分布式均值占位。"""
        raise NotImplementedError("MetricLoggerBase.distributed_mean 尚未实现。")


class TrainMetricLogger(MetricLoggerBase):
    """训练指标聚合。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum_total = 0.0
        self.count = 0.0
        self.extras: Dict[str, float] = {}

    def update(self, total_loss: torch.Tensor, batch_size: int, **kwargs: torch.Tensor) -> None:
        bs = float(batch_size)
        self.sum_total += float(total_loss.detach().item()) * bs
        self.count += bs
        for k, v in kwargs.items():
            self.extras.setdefault(k, 0.0)
            self.extras[k] += float(v.detach().item()) * bs

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        if self.count <= 0.0:
            return None
        base = {"loss/total": self.sum_total / self.count}
        for k, v in self.extras.items():
            base[f"loss/{k}"] = v / self.count
        return base


class EvalMetricLogger(MetricLoggerBase):
    """评估指标聚合。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def update(self, metrics: Dict[str, torch.Tensor], batch_size: int) -> None:
        bs = float(batch_size)
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.detach().item()) * bs
            self.counts[k] = self.counts.get(k, 0.0) + bs

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        if len(self.sums) == 0:
            return None
        out: Dict[str, float] = {}
        for k, v in self.sums.items():
            denom = self.counts.get(k, 0.0)
            if denom > 0.0:
                out[k] = v / denom
        return out if len(out) > 0 else None


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="配置路径/名称")
    parser.add_argument("--eval_only", action="store_true", help="仅评估")
    return parser.parse_args()


def main():
    """
    入口：解析配置 -> 环境 -> Accelerator -> 构建系统 -> 训练/评估。
    """
    args = parse_args()

    cfg = TrellisConfig.load_from_file(args.config, overrides=None)
    cfg.eval_only = bool(cfg.eval_only or args.eval_only)

    System.setup_env_and_seed(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    system = build_system(cfg, accelerator)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint if cfg.checkpoint else None, mode="train")
    global_step = int(ckpt_io.start_global_step)

    if cfg.eval_only:
        eval_log = evaluate(system, TrellisState(), cfg, accelerator, epoch=start_epoch, global_step=global_step, eval_loader=eval_loader)
        EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, start_epoch)
        return

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        state = TrellisState()
        for batch in train_loader:
            global_step += 1
            state = state.attach_batch(batch)
            train_log = train_edit4shape(system, state, cfg, accelerator, epoch, global_step)
            TrainMetricLogger.emit_logs(train_log, accelerator, logs_dir / "train.csv", global_step, epoch)

        if cfg.eval_freq and (epoch % int(cfg.eval_freq) == 0):
            eval_log = evaluate(system, state, cfg, accelerator, epoch=epoch, global_step=global_step, eval_loader=eval_loader)
            EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, epoch)

        if cfg.save_freq and (epoch % int(cfg.save_freq) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


if __name__ == "__main__":
    main()