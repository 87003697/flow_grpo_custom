#!/usr/bin/env python3
"""
TRELLIS Stage 2训练管道包装类
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Union, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# 在导入TRELLIS模块之前设置环境变量，避免flash_attn与远程下载
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 添加模块路径 - 支持TRELLIS导入
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入TRELLIS模块 - 按照开发规则使用官方路径
reference_path = Path(__file__).parent.parent.parent / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))
import trellis.modules.sparse as sp
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline

class TrellisStage2Pipeline:
    """TRELLIS Stage 2训练管道包装类
    
    简化架构设计：
    - Stage 1 (稀疏结构): 预训练权重固定，GRPO训练中在线推理
    - Stage 2 (SLAT生成): 使用GRPO进行强化学习训练
    - 只训练SLatFlowModel，冻结其他组件
    
    对应 SD3 的整体结构参考：
    - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:12-462`（管线级采样+logprob）
    - `scripts/train_sd3.py:198-231`（训练时按样本重算对数概率）
    """
    
    def __init__(self, model_path='./pretrained_weights/TRELLIS-image-large', verbose: bool = False):
        """初始化pipeline，加载预训练模型，Stage 1固定推理
        
        Args:
            model_path (str): TRELLIS预训练模型路径，默认使用本地下载的TRELLIS-image-large (1.2B参数)
            
        环境要求:
            - 需要设置 ATTN_BACKEND=xformers (避免flash_attn编译问题)
            - 需要设置 HF_HUB_OFFLINE=1 (使用本地模型)
            - 推荐使用脚本: ./scripts/run_trellis.sh python your_script.py
        """
        self.verbose = bool(verbose)
        if self.verbose:
            print(f"🔄 正在加载TRELLIS模型: {model_path}")
        
        # 加载TRELLIS官方pipeline
        self.core_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
        
        # 遵循官方模式：不在初始化时自动转换设备
        # 用户需要手动调用 pipeline.cuda() 或 pipeline.to(device)
        
        # 冻结Stage 1相关模型
        self._freeze_stage1()
        
        if self.verbose:
            print("✅ TRELLIS Stage 2 Pipeline初始化成功")
            print(f"📍 设备: {self.device}")
            print("🔒 Stage 1模型已冻结，仅Stage 2(SLatFlowModel)可训练")
            print("💡 提示: 请手动调用 pipeline.cuda() 将模型移动到GPU")
    
    def _freeze_stage1(self):
        """冻结Stage 1相关模型，只训练SLatFlowModel"""
        # 冻结稀疏结构相关模型 (Stage 1)
        models_to_freeze = [
            'sparse_structure_flow_model',    # 稀疏结构流模型
            'sparse_structure_encoder',       # 稀疏结构编码器  
            'sparse_structure_decoder',       # 稀疏结构解码器
            'image_cond_model'               # 图像条件模型(DINOv2)
        ]
        
        frozen_count = 0
        for model_name in models_to_freeze:
            if model_name in self.core_pipeline.models:
                model = self.core_pipeline.models[model_name]
                model.requires_grad_(False)
                model.eval()
                frozen_count += 1
                if self.verbose:
                    print(f"🔒 已冻结: {model_name}")
        
        # 确保SLatFlowModel可训练 (Stage 2)
        if 'slat_flow_model' in self.core_pipeline.models:
            slat_model = self.core_pipeline.models['slat_flow_model']
            slat_model.requires_grad_(True)
            slat_model.train()
            if self.verbose:
                print(f"🎯 Stage 2可训练: slat_flow_model")
        
        if self.verbose:
            print(f"📊 冻结模型数量: {frozen_count}")
    
    def get_trainable_model(self) -> nn.Module:
        """获取可训练的Stage 2模型 (SLatFlowModel)
        
        Returns:
            nn.Module: SLatFlowModel用于LoRA训练
        """
        if 'slat_flow_model' not in self.core_pipeline.models:
            raise ValueError("未找到slat_flow_model，无法进行Stage 2训练")
        
        return self.core_pipeline.models['slat_flow_model']
    
    @property
    def device(self) -> torch.device:
        """获取当前设备，遵循TRELLIS官方实现"""
        return self.core_pipeline.device
    
    def to(self, device: torch.device) -> None:
        """将所有模型移动到指定设备，遵循TRELLIS官方接口"""
        self.core_pipeline.to(device)
        # 确保内部 device 属性同步
        try:
            self.core_pipeline.device = device
        except Exception:
            pass
    
    def cuda(self) -> None:
        """将所有模型移动到CUDA设备，遵循TRELLIS官方接口"""
        self.core_pipeline.cuda()
        try:
            self.core_pipeline.device = torch.device('cuda')
        except Exception:
            pass
    
    def cpu(self) -> None:
        """将所有模型移动到CPU设备，遵循TRELLIS官方接口"""
        self.core_pipeline.cpu()

    @property
    def dtype(self) -> torch.dtype:
        """返回当前训练模型的默认 dtype
        优先从 Stage 2 可训练模型的参数推断；若不可用则回退到 float32。
        """
        slat_model = self.get_trainable_model() if 'slat_flow_model' in self.core_pipeline.models else None
        if slat_model is not None:
            for p in slat_model.parameters():
                return p.dtype
        return torch.float32
    
    def prepare_image_conditions(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """准备TRELLIS图像条件，使用DINOv2特征提取
        
        参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:145-160 (get_cond)
        参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:119-143 (encode_image)
        
        Args:
            images (List[Image.Image]): 输入图像列表
            
        Returns:
            Dict[str, torch.Tensor]: 包含cond和neg_cond的条件字典
                - cond: shape (B, N_patches, C) 其中B是batch size, N_patches是patch数量, C是特征维度
                - neg_cond: shape (B, N_patches, C) 全零张量
        """
        with torch.no_grad():
            # 使用TRELLIS官方的图像条件编码
            cond_dict = self.core_pipeline.get_cond(images)  # cond: (B, N_patches, C), neg_cond: (B, N_patches, C)
        
        return cond_dict
    
    def forward_stage1(self, image_cond: Dict[str, torch.Tensor], **sampler_params) -> torch.Tensor:
        """Stage 1在线推理生成稀疏结构坐标
        
        参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:162-193 (sample_sparse_structure)
        
        Args:
            image_cond (Dict[str, torch.Tensor]): 图像条件
                - cond: shape (B, N_patches, C)
                - neg_cond: shape (B, N_patches, C)
            sampler_params: 采样器参数（如 max_points 等）
            
        Returns:
            torch.Tensor: 稀疏结构坐标 [N, 4] (batch_idx, x, y, z)
        """
        # 提取仅在本函数内使用的控制参数，避免传入 sampler/model
        max_points = sampler_params.pop('max_points', None)
        num_samples = sampler_params.pop('num_samples', 1)

        with torch.no_grad():
            # 临时清理内部 sampler 默认参数中的不受支持键（如 max_points）
            original_default_params = self.core_pipeline.sparse_structure_sampler_params
            cleaned_default_params = {}
            for k, v in original_default_params.items():
                if k != 'max_points':
                    cleaned_default_params[k] = v
            self.core_pipeline.sparse_structure_sampler_params = cleaned_default_params

            # Stage 1推理：生成稀疏结构坐标
            coords = self.core_pipeline.sample_sparse_structure(  # output shape: (N, 4)
                cond=image_cond,
                num_samples=num_samples,
                sampler_params=sampler_params  # 仅保留 sampler 可识别的参数
            )

            # 恢复默认参数
            self.core_pipeline.sparse_structure_sampler_params = original_default_params
        
        # 可选：对坐标进行点数下采样，限制显存占用
        if isinstance(max_points, int) and max_points > 0 and coords.shape[0] > max_points:
            device = coords.device
            N = coords.shape[0]
            # 随机下采样前 max_points 个索引
            perm = torch.randperm(N, device=device)
            keep = perm[:max_points]
            coords = coords.index_select(0, keep)
        
        return coords  # shape: (N, 4)
    
    def forward_stage2_with_logprob(self, 
                                   coords: torch.Tensor, 
                                   image_cond: Dict[str, torch.Tensor], 
                                   **kwargs) -> Tuple[sp.SparseTensor, List[torch.Tensor], List[torch.Tensor]]:
        """Stage 2推理+LogProb计算，基于在线生成的稀疏结构
        
        参考:
        - TRELLIS: `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:219-252` (sample_slat)
        - Hunyuan3D: `flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py`
        - SD3 对应: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:12-462`
        
        注意: LogProb计算逻辑在 `flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py` 中实现
        """
        # Stage 2推理：SLAT采样
        # 注意：这里需要特殊处理来计算LogProb，将在对应的patch文件中实现
        slat_output = self.core_pipeline.sample_slat(  # output feats shape: (N, slat_channels)
            cond=image_cond,
            coords=coords,  # shape: (N, 4)
            sampler_params=kwargs.get('slat_sampler_params', {})
        )
        
        # 临时返回，LogProb计算将在trellis_stage2_with_logprob.py中实现
        return slat_output, [], []
    
    def decode_slat_to_mesh(self, slat: sp.SparseTensor) -> List:
        """将SLAT解码为mesh格式
        
        参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:195-217 (decode_slat)
        
        Args:
            slat (sp.SparseTensor): SLAT稀疏张量，feats shape: (N, slat_channels)
            
        Returns:
            List: mesh对象列表，每个mesh包含vertices和faces
        """
        with torch.no_grad():
            # 使用TRELLIS官方解码器解码为mesh
            decoded_outputs = self.core_pipeline.decode_slat(
                slat=slat,  # feats shape: (N, slat_channels)
                formats=['mesh']
            )
        
        return decoded_outputs['mesh']  # List of mesh objects
    
    @contextmanager 
    def train_mode(self):
        """训练模式上下文管理器，确保只有Stage 2处于训练状态"""
        # 保存当前状态
        original_states = {}
        for name, model in self.core_pipeline.models.items():
            original_states[name] = model.training
        
        # 设置训练状态：只有slat_flow_model训练，其他评估
        for name, model in self.core_pipeline.models.items():
            if name == 'slat_flow_model':
                model.train()
            else:
                model.eval()
        yield
        # 恢复原始状态
        for name, model in self.core_pipeline.models.items():
            model.train(original_states[name])
    
    @contextmanager
    def eval_mode(self):
        """评估模式上下文管理器"""
        # 保存当前状态  
        original_states = {}
        for name, model in self.core_pipeline.models.items():
            original_states[name] = model.training
            
        # 全部设为评估模式
        for model in self.core_pipeline.models.values():
            model.eval()
        yield
        # 恢复原始状态
        for name, model in self.core_pipeline.models.items():
            model.train(original_states[name]) 