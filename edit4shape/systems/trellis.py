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
import importlib.util
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from PIL import Image
import numpy as np
import yaml
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from edit4shape.datasets.trellis import TrellisDataConfig, TrellisDataModule

_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")

import os
import sys
# Add _reference_codes/TRELLIS to sys.path
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

from trellis.modules.sparse import SparseTensor

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
        """从 batch 挂载条件与指导数据（仅在提供时覆盖）。"""
        if "Conditions" in batch:
            cond_dict = batch["Conditions"] or {}
            cond = cond_dict.get("cond")
            if cond is None:
                raise ValueError("batch['Conditions'] 缺少 cond，无法构造条件。")
            neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))
            self.conditions_data = {"cond": cond, "neg_cond": neg_cond}
        elif self.conditions_data is None:
            # evaluate 路径会预先通过 pipeline.prepare_image_conditions 写入 self.conditions_data
            raise ValueError("batch['Conditions'] 为空且 state.conditions_data 未设置，无法构造条件。")

        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]

        if "mesh_c2w" in batch:
            # 假设 mesh_ 前缀的参数属于高分辨率相机，用于 mesh render
            self.cameras.mesh_c2w = batch["mesh_c2w"]
            self.cameras.mesh_w2c = batch["mesh_w2c"]
            self.cameras.mesh_mvp = batch["mesh_mvp_mtx"]
            self.cameras.mesh_positions = batch["mesh_camera_positions"]
            self.cameras.mesh_intrinsics = batch["mesh_intrinsics"]
        
        if "sdf_c2w" in batch:
             # 假设 sdf_ 前缀的参数属于低分辨率相机，用于 sdf render
            self.cameras.sdf_c2w = batch["sdf_c2w"]
            self.cameras.sdf_w2c = batch["sdf_w2c"]
            # self.cameras.sdf_rays_o = batch["sdf_rays_o"] # 如果需要
            # self.cameras.sdf_rays_d = batch["sdf_rays_d"]

        # 共享参数
        if "camera_positions" in batch:
            self.cameras.camera_positions = batch["camera_positions"]
        if "light_positions" in batch:
             self.cameras.light_positions = batch["light_positions"]
        
        return self

    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 conditions_data 中提取 cond/uncond embeddings。"""
        condition_utils = self.conditions_data
        if condition_utils is None:
            raise ValueError("TrellisState.conditions_data 为空，无法提取 embeddings。")
        cond_embeddings = condition_utils.get('cond')  # list/Tensor
        if isinstance(cond_embeddings, list):
            cond_embeddings = torch.cat(cond_embeddings, dim=0)  # (B,S,C)
        if isinstance(cond_embeddings, torch.Tensor) and cond_embeddings.dim() == 4 and cond_embeddings.shape[1] == 1:
            cond_embeddings = cond_embeddings.squeeze(1)  # (B,S,C) 或 (B,C)

        uncond_embeddings = condition_utils.get('neg_cond')  # list/Tensor
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
    def setup_env_and_seed(cfg: ml_collections.ConfigDict) -> None:
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
        cfg: ml_collections.ConfigDict,
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

    def prepare_models_and_optimizers(self, cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> "System":
        """
        仅包装可训练模块：pipeline/optimizer。
        """
        if accelerator is None:
            return self
        
        items = []
        # Only prepare pipeline if it is a nn.Module (TrellisRefAdapter is not)
        if isinstance(self.pipeline, torch.nn.Module):
            items.append(("pipeline", self.pipeline))
        if self.optimizer is not None:
            items.append(("optimizer", self.optimizer))
            
        if not items:
            return self

        prepared = accelerator.prepare(*[obj for _, obj in items])
        
        # accelerator.prepare returns a single object if only one arg is passed
        if len(items) == 1:
            prepared = [prepared]
            
        for (name, _), wrapped in zip(items, prepared):
            setattr(self, name, wrapped)
        return self


# === 构建函数（需按项目实际替换） ===
def build_system(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> System:
    """
    构建 geometry/renderer/guidance/optimizer。
    """
    # 1. Pipeline
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    pipeline = build_pipeline_from_reference(cfg, accelerator)

    # 2. Renderer: 根据 cfg.renderer.type 选择 mesh 或 gs
    cam = cfg.camera
    renderer_type = cfg.renderer.get("type", "mesh")  # 默认 mesh
    
    if renderer_type == "gs":
        # Gaussian Splatting Renderer
        from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
        rendering_options = {
            "resolution": cam.get("render_resolution", 512),  # 渲染分辨率 ()
            "near": cfg.renderer.get("near", 0.8),  # 近裁剪面 ()
            "far": cfg.renderer.get("far", 1.6),  # 远裁剪面 ()
            "ssaa": cfg.renderer.get("ssaa", 1),  # 超采样倍数 ()
            "bg_color": cfg.renderer.get("bg_color", "random"),  # 背景色
        }
        renderer = GaussianRenderer(rendering_options)
    else:
        # Mesh Rasterizer (nvdiffrast)
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
        renderer_cfg = TrellisRendererConfig(
            resolution=cam.get("render_resolution", 512),  # 渲染分辨率 ()
            ssaa=cfg.renderer.get("ssaa", 1),  # 超采样倍数 ()
            near=cfg.renderer.get("near", 0.8),  # 近裁剪面 ()
            far=cfg.renderer.get("far", 1.6),  # 远裁剪面 ()
        )
        renderer = TrellisMeshRasterizer(cfg=renderer_cfg, device=str(accelerator.device))

    # 3. Guidance & Optimizer
    guidance = None
    optimizer = None

    if not cfg.eval_only:
        from edit4shape.generators.trellis.training_adpter import build_optimizer_for_slat
        slat_model = pipeline.pipe.models["slat_flow_model"]
        optimizer = build_optimizer_for_slat(slat_model, cfg.train.optimizer)

    return System(pipeline=pipeline, renderer=renderer, guidance=guidance, optimizer=optimizer)


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """构造 DataLoader，直接复用 @edit4shape/datasets 的逻辑"""
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig
    
    cam = cfg.camera
    
    # 构建训练相机配置
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cam.train.n_view,
        yaw_range=list(cam.train.yaw_range),
        pitch_range=list(cam.train.pitch_range),
        r_range=list(cam.train.r_range),
        fov_range=list(cam.train.fov_range),
    )
    
    # 构建评估相机配置
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cam.eval.n_view,
        yaw=cam.eval.yaw,
        pitch=cam.eval.pitch,
        r=cam.eval.r,
        fov=cam.eval.fov,
    )
    
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        width=cam.render_resolution,
        height=cam.render_resolution,
        ray_height=cam.ray_height,
        ray_width=cam.ray_width,
        image_dataset_dir=cfg.train_data_dir if not cfg.eval_only else cfg.eval_data_dir,
        eval_image_path=cfg.eval_data_dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    dm = TrellisDataModule(dm_cfg, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    dm.setup()

    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader


# === Rollout（训练/评估共用） ===
def rollout_sparse(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    稠密结构 + 稀疏去噪 rollout（训练/评估共用）。
    支持 is_training=True 时开启梯度和 Checkpointing。
    返回 {"latents": SparseTensor, "coords": (B*T,4)}。
    """
    pipeline = system.pipeline
    ss_steps, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    
    cond_embeddings, uncond_embeddings = state.extract_embeddings()  # (B,S,C),(B,S,C)
    cond_embeddings = cond_embeddings.to(device)  # (B,S,C)
    if uncond_embeddings is not None:
        uncond_embeddings = uncond_embeddings.to(device)  # (B,S,C)

    # 1. 结构生成 (Structure Generation)
    # 优先复用 state 中已有的 coords，否则生成
    condition_utils = state.conditions_data
    if state.coords is not None:
        coords = state.coords  # 复用已有的 coords: (B*T,4)
    else:
        # 训练时 Stage 1 通常不需要梯度
        with torch.no_grad():
            coords = pipeline.dense_sampling(condition_utils, steps=ss_steps)  # (B*T,4)
        state.coords = coords
    
    batch_size = cond_embeddings.shape[0]  # ()
    if generator is None:
        # 训练模式下建议在外部根据 step 设置 generator
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    # 2. Latent 初始化
    in_channels = pipeline.pipe.models['slat_flow_model'].in_channels
    latents_sparse = pipeline.init_latents(coords=coords, in_channels=in_channels, generator=generator)  # SparseTensor

    # 提取 feats（模型参数有梯度，无需对输入 latent 开梯度）
    latents_feats = latents_sparse.feats

    # 3. Scheduler 设置
    scheduler = pipeline.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    slat_cfg_min, slat_cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]  # float

    # 4. 定义拆分后的去噪函数
    def _expand_t_to_batch(t_scalar, batch_size, device):
        """将标量 t 扩展为 (batch_size,) 形状，模型期望 t 形状为 (B,)。"""
        if torch.is_tensor(t_scalar):
            t_val = float(t_scalar.item()) if t_scalar.dim() == 0 else float(t_scalar)  # ()
        else:
            t_val = float(t_scalar)  # ()
        return torch.full((batch_size,), t_val, device=device, dtype=torch.float32)  # (B,)

    def get_cond_pred(current_feats, t_tensor, cond_emb):
        """仅计算条件预测（需要梯度/Checkpoint）"""
        x_t = SparseTensor(coords=coords, feats=current_feats)  # feats: (N,C)
        t_batch = _expand_t_to_batch(t_tensor, cond_emb.shape[0], current_feats.device)  # (B,)
        cond_out = pipeline.sparse_sampling_step(
            x_t, t_batch, cond_emb, uncond_embeddings=None, guidance_scale=0.0
        )  # cond_out.feats: (N,C)
        return cond_out.feats  # (N,C)

    def get_uncond_pred(current_feats, t_tensor, uncond_emb):
        """仅计算无条件预测（无需梯度）"""
        x_t = SparseTensor(coords=coords, feats=current_feats)  # feats: (N,C)
        t_batch = _expand_t_to_batch(t_tensor, uncond_emb.shape[0], current_feats.device)  # (B,)
        uncond_out = pipeline.sparse_sampling_step(
            x_t, t_batch, uncond_emb, uncond_embeddings=None, guidance_scale=0.0
        )  # uncond_out.feats: (N,C)
        return uncond_out.feats  # (N,C)

    # 5. 执行循环
    # 按照 Pipeline Adapter 逻辑，遍历 steps 次 (steps+1 个时间点，最后一个不用推)
    timesteps_list = list(scheduler.timesteps)
    steps_to_run = timesteps_list[:-1] if len(timesteps_list) > 1 else timesteps_list
    # 训练时显示进度条
    if is_training:
        steps_to_run = tqdm(steps_to_run, desc="Rollout", leave=False, disable=not Accelerator().is_main_process)

    # 仅在训练时启用 checkpointing
    use_ckpt = is_training

    for t in steps_to_run:
        t_val = float(t) if torch.is_tensor(t) else float(t)  # ()
        apply_cfg = slat_cfg_min <= t_val <= slat_cfg_max  # ()

        # 1. Cond Branch (Checkpoint if training, no_grad if inference)
        if use_ckpt:
            cond_pred = checkpoint(
                get_cond_pred,
                latents_feats,
                t,
                cond_embeddings,
                use_reentrant=False
            )  # (N,C)
        else:
            # 推理模式：使用 no_grad 减少内存占用
            with torch.no_grad():
                cond_pred = get_cond_pred(latents_feats, t, cond_embeddings)  # (N,C)

        # 2. Uncond Branch (Always no_grad per user request)
        uncond_pred = None
        if apply_cfg and uncond_embeddings is not None:
            with torch.no_grad():
                uncond_pred = get_uncond_pred(latents_feats, t, uncond_embeddings)  # (N,C)

        # 3. Mix CFG（仅在 cfg_interval 内生效）
        if apply_cfg:
            velocity_preds = mix_cfg(
                cond_pred=cond_pred,
                uncond_pred=uncond_pred,
                scale=float(slat_guidance),
                uncond_mode=True
            )  # (N,C)
        else:
            velocity_preds = cond_pred  # (N,C)

        # 4. Scheduler Step
        x_t_sparse = SparseTensor(coords=coords, feats=latents_feats)  # feats: (N,C)
        v_pred_sparse = SparseTensor(coords=coords, feats=velocity_preds)  # feats: (N,C)
        
        step_out = scheduler.step(v_pred_sparse, t, x_t_sparse)
        latents_feats = step_out.prev_sample.feats

    # 6. 应用 slat_normalization（与源代码 sample_slat 对齐）
    # 参考：_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:248-250
    slat_norm = pipeline.pipe.slat_normalization
    std = torch.tensor(slat_norm['std'])[None].to(latents_feats.device)  # (1, C)
    mean = torch.tensor(slat_norm['mean'])[None].to(latents_feats.device)  # (1, C)
    latents_feats = latents_feats * std + mean  # (N, C)

    # 7. 重组结果
    final_latents = SparseTensor(coords=coords, feats=latents_feats)
    return {"latents": final_latents, "coords": coords}


# === Loss 与指导 ===
def compute_guidance(
    guidance_module: Any,
    out: Dict[str, Any],
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算 guidance_loss（单标量），内部按 loss_* 与对应 lambda_* 聚合。
    """
    guidance_rgb = out["comp_rgb"].permute(0, 3, 1, 2)  # (B,3,H,W) 或 (B,4,3,H,W) -> (B,3,H,W) 视 renderer 输出而定
    # batch_extra = getattr(state, "batch_data", {}) or {}
    batch_extra = {} # TODO: replace batch_data logic
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
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Dict[str, torch.Tensor]:
    """
    核心训练循环：复用 rollout_sparse 进行 Flow Matching 训练
    """
    device = accelerator.device
    optimizer = system.optimizer

    # 1. 准备阶段
    optimizer.zero_grad()
    
    # 2. Rollout (Structure + Sparse Sampling w/ Checkpointing)
    # 训练时随机种子变化
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    
    rollout_out = rollout_sparse(
        state, cfg, system, device, 
        generator=generator, 
        is_training=True, 
    )
    
    # latents 包含完整的梯度图
    latents = rollout_out["latents"]
    
    # TODO: Decode & Render & Loss ...
    
    optimizer.zero_grad()
    return {}


# === 评估 ===
@torch.no_grad()
def evaluate(
    system: System,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """
    评估：rollout -> decoder -> save mesh
    """
    if eval_loader is None:
        return {}
    
    pipeline = system.pipeline
    ss_steps, _, slat_steps, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()
    save_dir = visuals_eval_dir / f"epoch_{epoch}"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logs: Dict[str, Any] = {}
    
    for batch_idx, batch in enumerate(eval_loader):
        # 每个 batch 独立状态，避免跨 batch 残留
        state = TrellisState()
        # 现在的 batch 是字典
        # batch['pixel_values'] 直接就是 [PIL.Image, ...]
        images = batch['pixel_values']  # list[len=B] of PIL.Image
        
        # 这里的 image_path 是 list[str]
        image_names = [os.path.basename(p) for p in batch['image_path']]  # list[len=B]
        
        # 1. Prepare conditions & State
        batch["Conditions"] = pipeline.prepare_image_conditions(images)  # dict with cond/neg_cond
        state.attach_batch(batch)  # 保存相机参数等以备后用
        
        # 2. Dense Sampling (moved out of rollout_sparse)
        coords = pipeline.dense_sampling(state.conditions_data, steps=ss_steps)  # (B*T,4)
        state.coords = coords  # (B*T,4)

        # 3. Rollout (Init + Sparse Sampling)
        rollout_out = rollout_sparse(state, cfg, system, accelerator.device)  # dict
        latents = rollout_out["latents"]  # SparseTensor

        # 4. Decode & Save
        renderer_type = cfg.renderer.get("type", "mesh")  # 获取 renderer 类型
        
        if renderer_type == "gs":
            # Gaussian Splatting 渲染分支
            outputs = pipeline.decode(latents, formats=['gaussian'])  # dict
            gaussians = outputs['gaussian']  # list[len=B] of Gaussian
            
            if accelerator.is_main_process:
                extr_all = state.cameras.mesh_w2c  # Tensor (B,V,4,4)
                intr_all = state.cameras.mesh_intrinsics  # Tensor (B,V,3,3)
                
                for i, gs in enumerate(gaussians):
                    ext_i = extr_all[i, 0].to(accelerator.device)  # (4,4)
                    intr_i = intr_all[i, 0].to(accelerator.device)  # (3,3)
                    render_out = system.renderer.render(gs, ext_i, intr_i)  # color: (3,H,W)
                    name = os.path.splitext(image_names[i])[0]
                    
                    # GS renderer 输出 color 形状为 (3,H,W)，需转换为 (H,W,3)
                    img_chw = render_out['color']  # (3,H,W)
                    img_hwc = img_chw.permute(1, 2, 0)  # (H,W,3)
                    img_np = (img_hwc.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,3)
                    
                    img_dir = save_dir / name
                    img_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img_np).save(str(img_dir / "color.png"))
        else:
            # Mesh Rasterizer 渲染分支
            outputs = pipeline.decode(latents, formats=['mesh'])  # dict
            meshes = outputs['mesh']  # list[len=B] of MeshExtractResult

            # 5. 渲染图片（相机参数固定为 (B,V,...)，默认渲染第 1 个视角）
            if accelerator.is_main_process:
                extr_all = state.cameras.mesh_w2c  # Tensor (B,V,4,4)
                intr_all = state.cameras.mesh_intrinsics  # Tensor (B,V,3,3)

                for i, mesh in enumerate(meshes):
                    ext_i = extr_all[i, 0].to(accelerator.device)  # (4,4)
                    intr_i = intr_all[i, 0].to(accelerator.device)  # (3,3)
                    render_out = system.renderer.render(mesh, ext_i, intr_i)  # dict of (H,W,C)
                    name = os.path.splitext(image_names[i])[0]
                    for k, v in render_out.items():
                        img_np = (v.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
                        if img_np.ndim == 3 and img_np.shape[-1] == 1:
                            img_np = img_np[..., 0]  # (H,W)
                        img_dir = save_dir / name
                        img_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img_np).save(str(img_dir / f"{k}.png"))

            if accelerator.is_main_process:
                for i, mesh in enumerate(meshes):
                    name = os.path.splitext(image_names[i])[0]  # 去掉 .png 扩展名
                    mesh_dir = save_dir / name
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    out_path = mesh_dir / "mesh.obj"
                    pipeline.export_mesh_obj(mesh, str(out_path))
                    print(f"Saved mesh to {out_path}")

    return {"eval_done": 1.0}


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


def build_run_paths(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[Path, Path, Path, Path]:
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
        # Save config using yaml
        with (run_root / "config.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(cfg.to_dict(), sort_keys=False))
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

    def save(self, system: System, state: TrellisState, cfg: ml_collections.ConfigDict, epoch: int, global_step: int) -> None:
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
        # Simple distributed mean implementation
        tensor = torch.tensor(values_np, device=accelerator.device)
        reduced = accelerator.gather(tensor)
        return float(reduced.mean().item())


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


def main(argv) -> None:
    """
    入口：解析配置 -> 环境 -> Accelerator -> 构建系统 -> 训练/评估。
    """
    del argv  # absl.app.run 会传入 argv；本函数不使用
    cfg = _CONFIG.value

    System.setup_env_and_seed(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )

    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    system = build_system(cfg, accelerator)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.get('checkpoint'), mode="train")
    global_step = int(ckpt_io.start_global_step)

    if cfg.eval_only:
        eval_log = evaluate(system, cfg, accelerator, epoch=start_epoch, global_step=global_step, eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir)
        EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, start_epoch)
        return

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            state = TrellisState()
            # 从 batch 提取图像并准备条件编码（与 evaluate 对齐）
            images = batch['pixel_values']  # list[len=B] of PIL.Image
            batch["Conditions"] = system.pipeline.prepare_image_conditions(images)  # dict with cond/neg_cond
            state = state.attach_batch(batch)
            train_log = train_edit4shape(system, state, cfg, accelerator, epoch, global_step)
            TrainMetricLogger.emit_logs(train_log, accelerator, logs_dir / "train.csv", global_step, epoch)

        if cfg.eval_freq and (epoch % int(cfg.eval_freq) == 0):
            eval_log = evaluate(system, cfg, accelerator, epoch=epoch, global_step=global_step, eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir)
            EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, epoch)

        if cfg.save_freq and (epoch % int(cfg.save_freq) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


if __name__ == "__main__":
    app.run(main)
