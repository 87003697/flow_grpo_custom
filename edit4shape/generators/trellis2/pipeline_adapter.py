"""
Trellis2 reference pipeline 适配器（统一使用 SparseTensor）。

仅依赖 _reference_codes/TRELLIS.2 下的 Trellis2ImageTo3DPipeline，
并对齐 edit4shape/systems/trellis2.py 期望的接口。

核心接口（使用 stage + resolution 参数统一）：
- prepare_image_conditions(images, resolution): 预处理图像并生成条件编码
- dense_sampling(cond_dict, steps, resolution): 生成稀疏结构 coords
- init_latents(coords, stage, resolution): 生成初始 latent
- scheduler(stage): 创建 Scheduler
- sampling_step(x_t, t, cond, stage, resolution, shape_cond): 单步采样
- decode(shape_slat, tex_slat, resolution): 解码为 Mesh/MeshWithVoxel
- normalize(slat, stage) / denormalize(slat, stage): 归一化/反归一化

Stage 类型：
- "shape": 几何生成阶段
- "tex": 纹理生成阶段

Resolution 类型：
- 512: 低分辨率模型
- 1024: 高分辨率模型

注意：
- 不使用 low_vram 模式
- 所有张量操作行均添加形状注释
"""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import trimesh

# =====================================================================
# TRELLIS.2 参考实现路径设置
# =====================================================================
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
from trellis2.representations import MeshWithVoxel

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape", "tex"]
Resolution = Literal[512, 1024]
PipelineType = Literal["512", "1024", "1024_cascade", "1536_cascade"]


# =====================================================================
# 数值稳定性工具函数
# =====================================================================

def safe_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    梯度安全的 Clamp：使用 straight-through estimator 保持边界处的梯度流动。
    
    与 torch.clamp 的区别：
    - torch.clamp 在边界处梯度为 0，可能导致训练停滞
    - safe_clamp 使用 detach 技巧，让梯度直接穿过边界
    
    Args:
        x: 输入张量
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        clamp 后的张量，但梯度不被截断
    """
    return x + (torch.clamp(x, min_val, max_val) - x).detach()

# =====================================================================
# Pipeline 配置字典（外部定义，便于维护）
# =====================================================================
# 每个 pipeline_type 对应的配置：
# - target_resolution: 最终输出分辨率
# - stages: 各阶段的分辨率配置
#   - ss_resolution: sparse structure 分辨率（用于 dense_sampling）
#   - cond_resolution: 条件编码分辨率（用于 prepare_image_conditions）
#   - flow_resolution: flow model 分辨率（用于 get_flow_model）
# - models_to_remove: 构建 pipeline 时需要删除的模型列表

PIPELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "512": {
        "target_resolution": 512,
        "stages": {
            "shape": {"ss_resolution": 32, "cond_resolution": 512, "flow_resolution": 512},
            "tex": {"ss_resolution": 32, "cond_resolution": 512, "flow_resolution": 512},
        },
        "models_to_remove": ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
    },
    "1024": {
        "target_resolution": 1024,
        "stages": {
            "shape": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
            "tex": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
        },
        "models_to_remove": ["shape_slat_flow_model_512", "tex_slat_flow_model_512"],
    },
    "1024_cascade": {
        "target_resolution": 1024,
        "stages": {
            # Stage 1: 512 分辨率生成粗糙形状
            "shape_stage1": {"ss_resolution": 32, "cond_resolution": 512, "flow_resolution": 512},
            # Stage 2: 1024 分辨率精细化形状
            "shape_stage2": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
            # Tex: 1024 分辨率
            "tex": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
        },
        "models_to_remove": ["tex_slat_flow_model_512"],
    },
    "1536_cascade": {
        "target_resolution": 1536,
        "stages": {
            "shape_stage1": {"ss_resolution": 32, "cond_resolution": 512, "flow_resolution": 512},
            "shape_stage2": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
            "tex": {"ss_resolution": 64, "cond_resolution": 1024, "flow_resolution": 1024},
        },
        "models_to_remove": ["tex_slat_flow_model_512"],
    },
}


# =====================================================================
# FlowEuler Scheduler（独立类）
# =====================================================================
class FlowEulerScheduler:
    """
    基于 FlowEuler 公式的 Scheduler。
    
    提供 set_timesteps() 和 step() 方法，用于去噪采样。
    """
    
    def __init__(self, rescale_t: float = 1.0):
        self.timesteps: torch.Tensor = torch.tensor([])
        self.rescale_t = rescale_t
    
    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        """
        设置时间步序列。
        
        Args:
            num_steps: 采样步数
            device: 目标设备
        
        timesteps: 递减序列 [1.0, ..., 0.0]，长度 num_steps + 1
        """
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # (steps+1,)
        self.timesteps = self.rescale_t * timesteps / (1 + (self.rescale_t - 1) * timesteps)  # (steps+1,)
    
    def step(
        self,
        velocity: SparseTensor,
        t: torch.Tensor,
        latents: SparseTensor,
    ) -> SimpleNamespace:
        """
        Euler 步进：x_{t-1} = x_t - (t - t_prev) * v
        
        Args:
            velocity: SparseTensor，velocity 预测
            t: 当前时间步（标量）
            latents: SparseTensor，当前 latent
        
        Returns:
            SimpleNamespace: 包含 prev_sample
        """
        t_val = float(t)
        
        # 查找 t_prev
        match_idx = torch.isclose(
            self.timesteps,
            torch.tensor(t_val, device=self.timesteps.device, dtype=self.timesteps.dtype)
        ).nonzero(as_tuple=False)
        
        assert match_idx.numel() > 0, f"t={t_val} 未匹配到 timesteps"
        idx = int(match_idx[0])
        assert idx + 1 < self.timesteps.numel(), f"t={t_val} 无后继步"
        
        t_prev = float(self.timesteps[idx + 1].item())
        delta = t_val - t_prev  # 标量
        
        # Euler 步进
        pred_feats = latents.feats - delta * velocity.feats  # (N, C)
        prev_sample = SparseTensor(coords=latents.coords, feats=pred_feats)
        
        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=None)


# =====================================================================
# Pipeline 构建函数
# =====================================================================
def build_pipeline_from_reference(
    cfg: Any,
    accelerator: Any,
    device: Optional[torch.device] = None
) -> "Trellis2RefAdapter":
    """
    构建参考 Trellis2 pipeline 的适配器实例。
    
    Args:
        cfg: 配置对象，需包含：
            - cfg.pretrained.model: 预训练模型路径 (如 "microsoft/TRELLIS.2-4B")
            - cfg.pretrained.dino_local_path: (可选) DINOv3 本地路径
        accelerator: Accelerate 加速器
        device: 可选，指定模型加载的设备
    
    Returns:
        Trellis2RefAdapter: 适配器实例
    """
    # 设置默认 CUDA 设备
    if device is None:
        device = accelerator.device
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    
    # 加载预训练 pipeline
    pipe_raw = Trellis2ImageTo3DPipeline.from_pretrained(
        cfg.pretrained.model,
        dino_local_path=cfg.pretrained.dino_local_path
    )
    
    # 不使用 low_vram 模式，直接将所有模型移动到设备
    pipe_raw.low_vram = False
    pipe_raw.to(device)
    pipe_raw.image_cond_model.to(device)
    if pipe_raw.rembg_model is not None:
        pipe_raw.rembg_model.to(device)
    
    os.environ["TRELLIS_VERBOSE"] = "1" if cfg.verbose else "0"
    
    # ========== 按需删除不需要的 Flow Model，节省显存 ==========
    pipeline_type: str = cfg.pipeline_type
    
    if pipeline_type not in PIPELINE_CONFIGS:
        raise ValueError(
            f"未知的 pipeline_type: {pipeline_type}，"
            f"可选值: {list(PIPELINE_CONFIGS.keys())}"
        )
    
    config = PIPELINE_CONFIGS[pipeline_type]
    models_to_remove = config["models_to_remove"]
    
    for model_key in models_to_remove:
        pipe_raw.models.pop(model_key, None)
    
    if models_to_remove:
        print(f"[Trellis2Adapter] pipeline_type={pipeline_type}, removed: {models_to_remove}")
    
    torch.cuda.empty_cache()
    
    return Trellis2RefAdapter(pipe_raw, pipeline_type=pipeline_type)


# =====================================================================
# 主适配器类
# =====================================================================
class Trellis2RefAdapter:
    """
    适配 _reference_codes/TRELLIS.2 的 Trellis2ImageTo3DPipeline。
    
    使用 (stage, resolution) 参数统一 Shape/Tex 和 512/1024 模型的接口。
    """

    def __init__(self, pipe_raw: Trellis2ImageTo3DPipeline, pipeline_type: str = "1024"):
        self.pipe = pipe_raw
        self.pipeline_type = pipeline_type
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取当前 pipeline 配置"""
        return PIPELINE_CONFIGS[self.pipeline_type]
    
    @property
    def is_cascade(self) -> bool:
        """是否为 cascade 模式"""
        return "cascade" in self.pipeline_type
    
    @property
    def target_resolution(self) -> int:
        """目标输出分辨率"""
        return self.config["target_resolution"]
    
    def get_stage_config(self, stage: Stage, cascade_stage: int = 1) -> Dict[str, int]:
        """
        获取指定阶段的配置。
        
        Args:
            stage: "shape" 或 "tex"
            cascade_stage: cascade 模式下的子阶段 (1 或 2)，非 cascade 时忽略
        
        Returns:
            dict: {"ss_resolution": int, "cond_resolution": int, "flow_resolution": int}
        """
        stages_config = self.config["stages"]
        
        if self.is_cascade and stage == "shape":
            key = f"shape_stage{cascade_stage}"
        else:
            key = stage
        
        if key not in stages_config:
            raise KeyError(f"配置中不存在 stage={key}，可用: {list(stages_config.keys())}")
        
        return stages_config[key]
    
    @property
    def device(self) -> torch.device:
        """获取 pipeline 设备"""
        return self.pipe.device
    
    # =========================================================================
    # 模型访问（统一接口）
    # =========================================================================
    
    def get_flow_model(self, stage: Stage, resolution: Resolution) -> nn.Module:
        """
        获取指定阶段和分辨率的 Flow Model。
        
        Args:
            stage: "shape" 或 "tex"
            resolution: 512 或 1024
        
        Returns:
            nn.Module: 对应的 Flow Model
        """
        key = f"{stage}_slat_flow_model_{resolution}"
        if key not in self.pipe.models:
            raise KeyError(f"模型 '{key}' 不存在，可用模型: {list(self.pipe.models.keys())}")
        return self.pipe.models[key]
    
    def get_in_channels(self, stage: Stage, resolution: Resolution) -> int:
        """
        获取 Flow Model 的输入通道数。
        
        Args:
            stage: "shape" 或 "tex"
            resolution: 512 或 1024
        
        Returns:
            int: 输入通道数（tex 阶段返回去除 shape concat 后的通道数）
        """
        model = self.get_flow_model(stage, resolution)
        if stage == "tex":
            # Tex 的实际输入通道 = 总通道 - shape 通道
            shape_channels = self.get_flow_model("shape", resolution).in_channels
            return model.in_channels - shape_channels
        return model.in_channels
    
    # =========================================================================
    # 采样参数（统一接口）
    # =========================================================================
    
    def get_sampler_params(self, stage: Stage) -> Dict[str, Any]:
        """
        获取指定阶段的采样参数。
        
        Args:
            stage: "shape" 或 "tex"
        
        Returns:
            dict: 包含 steps, cfg_strength, rescale_t, cfg_interval 等
        """
        if stage == "shape":
            return self.pipe.shape_slat_sampler_params
        else:
            return self.pipe.tex_slat_sampler_params
    
    def get_cfg_interval(self, stage: Stage) -> Tuple[float, float]:
        """
        获取 CFG 区间。
        
        Args:
            stage: "shape" 或 "tex"
        
        Returns:
            (min, max): CFG 生效的时间步区间
        """
        params = self.get_sampler_params(stage)
        interval = params["guidance_interval"]
        return (float(interval[0]), float(interval[1]))
    
    def get_ss_params(self) -> Dict[str, Any]:
        """获取 Sparse Structure 采样参数"""
        return self.pipe.sparse_structure_sampler_params
    
    # =========================================================================
    # Scheduler（统一接口）
    # =========================================================================
    
    def scheduler(self, stage: Stage) -> FlowEulerScheduler:
        """
        创建指定阶段的 Scheduler。
        
        Args:
            stage: "shape" 或 "tex"
        
        Returns:
            FlowEulerScheduler: 提供 set_timesteps() 和 step() 方法
        """
        params = self.get_sampler_params(stage)
        rescale_t = float(params["rescale_t"])
        return FlowEulerScheduler(rescale_t=rescale_t)

    # =========================================================================
    # 条件编码
    # =========================================================================
    
    def prepare_image_conditions(
        self,
        images: List[Any],
        resolution: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        预处理图像并生成 cond/neg_cond（使用 DINOv3）。
        
        Args:
            images: PIL Image 列表
            resolution: 图像编码分辨率 (512 或 1024)
        
        Returns:
            dict: {"cond": (B, S, C), "neg_cond": (B, S, C)}
        """
        # 预处理图像（去背、裁剪等）
        images_proc = [self.pipe.preprocess_image(img) for img in images]  # List[PIL]
        
        # 获取条件编码
        cond_dict = self.pipe.get_cond(images_proc, resolution=resolution)  # dict

        cond = cond_dict["cond"]  # (B, S, C)
        neg_cond = cond_dict["neg_cond"]  # (B, S, C)

        return {"cond": cond, "neg_cond": neg_cond}

    # =========================================================================
    # 稀疏结构采样
    # =========================================================================
    
    def dense_sampling(
        self,
        condition_utils: Dict[str, Any],
        steps: Optional[int] = None,
        resolution: int = 32,
    ) -> torch.Tensor:
        """
        生成稀疏结构 coords。
        
        Args:
            condition_utils: {"cond": (B, S, C), "neg_cond": (B, S, C)}
            steps: 采样步数（None 则使用默认）
            resolution: 稀疏结构分辨率（默认 32）
        
        Returns:
            coords: (B*T, 4) int32，coords[:, 0] 为 batch 索引
        """
        cond = condition_utils["cond"]
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=0)  # (B, S, C)
        batch_size = int(cond.shape[0])  # ()

        ss_params = self.get_ss_params()
        steps_val = int(ss_params["steps"] if steps is None else steps)
        
        sampler_params = {**ss_params, "steps": steps_val}
        
        # 调用原始 pipeline 的稀疏结构采样
        coords = self.pipe.sample_sparse_structure(
            cond=condition_utils,
            resolution=resolution,
            num_samples=1,
            sampler_params=sampler_params,
        )  # (T, 4)，coords[:, 0] 默认为 0
        
        coords = coords.to(device=self.device, dtype=torch.int32)  # (T, 4)

        # 为每个 batch 样本写入 batch 索引并拼接
        coords_list = []
        for b in range(batch_size):
            cb = coords.clone()  # (T, 4)
            cb[:, 0] = b  # 写入 batch 维
            coords_list.append(cb)
        
        coords_batched = torch.cat(coords_list, dim=0)  # (B*T, 4)
        return coords_batched

    # =========================================================================
    # Latent 初始化
    # =========================================================================
    
    def init_latents(
        self,
        coords: torch.Tensor,
        stage: Stage,
        resolution: Resolution,
        generator: Optional[torch.Generator] = None,
    ) -> SparseTensor:
        """
        生成初始 SparseTensor latent。
        
        Args:
            coords: (N, 4) 稀疏坐标
            stage: "shape" 或 "tex"
            resolution: 512 或 1024
            generator: 随机数生成器
        
        Returns:
            SparseTensor: feats 形状 (N, C)
        """
        in_channels = self.get_in_channels(stage, resolution)
        feats = torch.randn(
            coords.shape[0],
            in_channels,
            device=coords.device,
            dtype=torch.float32,
            generator=generator,
        )  # (N, C)
        return SparseTensor(coords=coords, feats=feats)

    # =========================================================================
    # 单步采样（统一接口）
    # =========================================================================
    
    def sampling_step(
        self,
        x_t: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        stage: Stage,
        resolution: Resolution,
        shape_cond: Optional[SparseTensor] = None,
    ) -> SparseTensor:
        """
        单步 velocity 预测（统一接口）。
        
        Args:
            x_t: SparseTensor，当前 latent
            t: (B,) 或标量，时间步（范围 [0, 1]）
            cond: (B, S, C) 条件嵌入
            stage: "shape" 或 "tex"
            resolution: 512 或 1024
            shape_cond: SparseTensor，tex 阶段需要的 shape 条件（已归一化）
        
        Returns:
            SparseTensor: velocity 预测
        """
        model = self.get_flow_model(stage, resolution)
        
        # 时间步缩放：模型期望 t * 1000
        t_scaled = self._scale_timesteps(t, cond.shape[0], x_t.device)  # (B,)
        
        if stage == "tex":
            if shape_cond is None:
                raise ValueError("tex 阶段需要提供 shape_cond")
            # 前向预测（concat_cond 会在模型内部与 x_t 拼接）
            pred_v = model(x_t, t_scaled, cond, concat_cond=shape_cond)  # SparseTensor
        else:
            pred_v = model(x_t, t_scaled, cond)  # SparseTensor
        
        return pred_v
    
    # =========================================================================
    # 归一化/反归一化（统一接口）
    # =========================================================================
    
    def _get_normalization(self, stage: Stage) -> Dict[str, List[float]]:
        """获取归一化参数"""
        if stage == "shape":
            return self.pipe.shape_slat_normalization
        else:
            return self.pipe.tex_slat_normalization
    
    def normalize(self, slat: SparseTensor, stage: Stage) -> SparseTensor:
        """
        归一化 latent（用于作为条件）。
        
        Args:
            slat: SparseTensor，反归一化后的特征
            stage: "shape" 或 "tex"
        
        Returns:
            SparseTensor: 归一化后的特征
        """
        norm = self._get_normalization(stage)
        std = torch.tensor(norm['std'])[None].to(slat.device)  # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(slat.device)  # (1, C)
        
        normalized_feats = (slat.feats - mean) / std  # (N, C)
        return slat.replace(feats=normalized_feats)
    
    def denormalize(self, slat: SparseTensor, stage: Stage) -> SparseTensor:
        """
        反归一化 latent（采样结束后）。
        
        Args:
            slat: SparseTensor，归一化的特征
            stage: "shape" 或 "tex"
        
        Returns:
            SparseTensor: 反归一化后的特征
        """
        norm = self._get_normalization(stage)
        std = torch.tensor(norm['std'])[None].to(slat.device)  # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(slat.device)  # (1, C)
        
        denormalized_feats = slat.feats * std + mean  # (N, C)
        return slat.replace(feats=denormalized_feats)
    
    # =========================================================================
    # 解码接口
    # =========================================================================
    
    def _set_decoder_checkpointing(self, decoder_name: str, enable: bool) -> None:
        """设置 decoder 的 gradient checkpointing 状态。"""
        decoder = self.pipe.models[decoder_name]
        for res in decoder.blocks:
            for block in res:
                if hasattr(block, 'use_checkpoint'):
                    block.use_checkpoint = enable
    
    def decode_shape(
        self,
        shape_slat: SparseTensor,
        resolution: int = 1024,
    ) -> Dict[str, Any]:
        """
        Shape 解码接口（支持梯度传播）。
        
        Args:
            shape_slat: SparseTensor，shape 特征（已反归一化）
            resolution: 输出分辨率
            use_checkpointing: 是否使用 gradient checkpointing 减少显存
        
        Returns:
            dict: {
                "meshes": List[Mesh],
                "subs": List[SparseTensor],  # 中间结果，供 decode_tex 使用
            }
        """
        meshes, subs = self.pipe.decode_shape_slat(shape_slat, resolution)
        return {"meshes": meshes, "subs": subs}
    
    def decode_tex(
        self,
        tex_slat: SparseTensor,
        meshes: List[Any],
        subs: List[SparseTensor],
        resolution: int = 1024,
    ) -> Dict[str, Any]:
        """
        Tex 解码接口（支持梯度传播）。
        
        Args:
            tex_slat: SparseTensor，tex 特征（已反归一化）
            meshes: List[Mesh]，由 decode_shape 返回
            subs: List[SparseTensor]，由 decode_shape 返回
            resolution: 输出分辨率
            use_checkpointing: 是否使用 gradient checkpointing 减少显存
        
        Returns:
            dict: {
                "tex_voxels": SparseTensor,
                "mesh_with_voxel": List[MeshWithVoxel],
            }
        """
        tex_voxels = self.pipe.decode_tex_slat(tex_slat, subs)
        
        # 构建 MeshWithVoxel（保持梯度连接）
        # ★ 数值保护：只对 base_color 通道使用 safe_clamp
        # PBR 渲染中只有 basecolor ** 2.2 需要非负输入，否则会产生 NaN
        # 属性布局: base_color(0-2), metallic(3), roughness(4), alpha(5)
        EPS = 1e-4
        mesh_with_voxel = []
        for m, v in zip(meshes, tex_voxels):
            # 避免 inplace 操作：用 torch.cat 拼接 clamped base_color 和其他通道
            clamped_rgb = torch.clamp(v.feats[:, :3], EPS, 1.0 - EPS)  # (N, 3) base_color
            attrs = torch.cat([clamped_rgb, v.feats[:, 3:]], dim=1)  # (N, 6)
            
            mesh_with_voxel.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=attrs,  # 使用保护后的 attrs
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pipe.pbr_attr_layout
                )
            )
        
        return {"tex_voxels": tex_voxels, "mesh_with_voxel": mesh_with_voxel}
    
    def decode(
        self,
        shape_slat: SparseTensor,
        tex_slat: Optional[SparseTensor] = None,
        resolution: int = 1024,
    ) -> Dict[str, Any]:
        """
        统一解码接口（兼容旧代码）。
        
        Args:
            shape_slat: SparseTensor，shape 特征（已反归一化）
            tex_slat: SparseTensor，tex 特征（可选）
            resolution: 输出分辨率
            use_checkpointing: 是否使用 gradient checkpointing
        
        Returns:
            dict: 包含 meshes, subs, tex_voxels, mesh_with_voxel（视参数而定）
        """
        result = self.decode_shape(shape_slat, resolution)
        
        if tex_slat is not None:
            tex_result = self.decode_tex(
                tex_slat, result["meshes"], result["subs"], resolution
            )
            result.update(tex_result)
        
        return result
    
    # =========================================================================
    # Mesh 导出
    # =========================================================================
    
    def export_mesh_obj(self, mesh: Any, out_path: str) -> None:
        """
        导出 Mesh 为 OBJ 文件。
        
        Args:
            mesh: Mesh 或 MeshWithVoxel 对象
            out_path: 输出路径
        """
        if mesh is None:
            return
        
        vertices = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        
        mesh_np = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh_np.export(out_path)
    
    # =========================================================================
    # LoRA 控制
    # =========================================================================
    
    @contextmanager
    def disable_lora_context(self, stage: Stage, resolution: Resolution):
        """
        临时禁用 LoRA 适配器的上下文管理器。
        
        Args:
            stage: "shape" 或 "tex"
            resolution: 512 或 1024
        
        Yields:
            None
        """
        model = self.get_flow_model(stage, resolution)
        
        if hasattr(model, 'disable_adapters'):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            yield
    
    # =========================================================================
    # PBR 属性
    # =========================================================================
    
    def get_pbr_attr_layout(self) -> Dict[str, slice]:
        """获取 PBR 属性布局"""
        return self.pipe.pbr_attr_layout
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _scale_timesteps(
        self,
        timesteps: Union[torch.Tensor, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        缩放时间步到模型期望的范围 [0, 1000]。
        
        Args:
            timesteps: 标量或 (B,) 张量，范围 [0, 1]
            batch_size: batch 大小
            device: 目标设备
        
        Returns:
            (B,) 张量，范围 [0, 1000]
        """
        if torch.is_tensor(timesteps):
            if timesteps.dim() == 0:
                # 标量 tensor，扩展为 batch
                t_scaled = torch.full(
                    (batch_size,),
                    float(timesteps.item()) * 1000,
                    device=device,
                    dtype=torch.float32
                )  # (B,)
            else:
                t_scaled = timesteps.to(device) * 1000  # (B,)
        else:
            t_scaled = torch.full(
                (batch_size,),
                float(timesteps) * 1000,
                device=device,
                dtype=torch.float32
            )  # (B,)
        
        return t_scaled
