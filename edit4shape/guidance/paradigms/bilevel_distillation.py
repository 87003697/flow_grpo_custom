"""
双层蒸馏 Guidance 模块（VSD - Variational Score Distillation）。

教师-学生双层优化：
    教师（基础模型 / LoRA 关闭）：提供真实分布的速度估计
    学生（LoRA 开启）：拟合当前渲染图分布的速度场

两层 Loss：
    外层 VSD Loss（优化 3D 模型，通过 compute_guidance 返回）：
        复用 CSD 体系 → x0_pos = x0_teacher（吸引），x0_neg = x0_student（排斥）
        loss_vsd = csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))
    
    内层 Student Loss（优化 LoRA，在 compute_guidance 内部完成）：
        loss_student = lambda_sup * MSE(v_student, noise - clean_latents)

数据流：
    1. 格式转换（父类）
    2. 编码到 latent（父类，有梯度）
    3. 教师/学生双前向（Pipeline，无梯度）
    4. 外层 VSD Loss（通过 Tracker.loss()，梯度流向 3D 模型）
    5. 学生带梯度前向 + LoRA 更新（gradient_checkpoint）
    6. 返回外层 loss（训练循环 backward 更新 3D 模型）
"""

import logging
from typing import List, Any, Dict, Tuple, Optional

import torch
from PIL import Image

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipelines.qwen_image_edit.bilevel_distillation import (
    QwenImageBilevelDistillationPipeline,
    BilevelDistillationOutput,
)


logger = logging.getLogger(__name__)


# =============================================================================
# BilevelDistillationGuidance
# =============================================================================

class BilevelDistillationGuidance(BaseGuidance):
    """
    双层蒸馏 Guidance（VSD）。
    
    教师-学生双层优化：
    - 外层 VSD Loss（优化 3D 模型）：
        复用 CSD 体系 → loss_vsd = csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))
    - 内层 Student Loss（优化 LoRA）：
        loss_student = lambda_sup * MSE(v_student, noise - clean_latents)
    
    LoRA 注入、优化器、更新步 均在 Guidance 内部管理，
    对外接口与普通 Distillation 完全一致。
    """
    
    # 类属性：用于 loss_dict 的 key 名称
    loss_key = "bilevel_distillation"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 Bilevel Distillation Guidance。
        
        Args:
            cfg: 完整配置对象（cfg.guidance.bilevel_distillation）
            train_device: 训练使用的设备
        """
        super().__init__(cfg, train_device)
        
        # ======== 蒸馏基础配置 ========
        self.bilevel_distillation_cfg = cfg.guidance.bilevel_distillation
        self.min_step_percent = self.bilevel_distillation_cfg.min_step_percent
        self.max_step_percent = self.bilevel_distillation_cfg.max_step_percent
        self.true_cfg_scale = self.bilevel_distillation_cfg.true_cfg_scale
        self.target_prompt = self.bilevel_distillation_cfg.target_prompt
        self.negative_prompt = self.bilevel_distillation_cfg.negative_prompt
        self.seed = self.bilevel_distillation_cfg.seed
        
        # Loss 权重（外层 VSD Loss）
        self.mse_weight = self.bilevel_distillation_cfg.mse_weight
        self.csd_weight = self.bilevel_distillation_cfg.csd_weight
        
        # 梯度归一化
        self.ada_normalize = self.bilevel_distillation_cfg.get("ada_normalize", True)
        self.ada_eps = self.bilevel_distillation_cfg.get("ada_eps", 1e-2)
        
        # MTS（多时间步采样）
        self.num_timesteps = self.bilevel_distillation_cfg.get("num_timesteps", 1)
        self.reduce_mode = self.bilevel_distillation_cfg.get("reduce_mode", "mean")
        
        # 噪声模式
        self.noise_mode = self.bilevel_distillation_cfg.get("noise_mode", "fixed")
        
        # ======== VSD 专属配置 ========
        self.lambda_sup = self.bilevel_distillation_cfg.get("lambda_sup", 1.0)
        
        # LoRA 配置
        self.lora_rank = self.bilevel_distillation_cfg.get("lora_rank", 64)
        self.lora_alpha = self.bilevel_distillation_cfg.get("lora_alpha", 64)
        self.lora_dropout = self.bilevel_distillation_cfg.get("lora_dropout", 0.1)
        self.lora_target_modules = self.bilevel_distillation_cfg.get(
            "lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]
        )
        self.lora_lr = self.bilevel_distillation_cfg.get("lora_lr", 1e-4)
        
        # ======== 加载 Pipeline ========
        model_path = cfg.guidance.model_path
        
        logger.info(f"[BilevelGuidance] Loading pipeline on {self.device}...")
        logger.info(f"[BilevelGuidance] Model: {model_path}")
        logger.info(f"[BilevelGuidance] VSD config: lambda_sup={self.lambda_sup}, "
                     f"lora_rank={self.lora_rank}, lora_lr={self.lora_lr}")
        logger.info(f"[BilevelGuidance] Outer loss: mse_weight={self.mse_weight}, csd_weight={self.csd_weight}")
        logger.info(f"[BilevelGuidance] MTS: num_timesteps={self.num_timesteps}, reduce_mode={self.reduce_mode}")
        
        self.pipe = QwenImageBilevelDistillationPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # ======== 注入 LoRA ========
        self.pipe.init_lora(
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
        )
        
        # ======== 创建 LoRA 优化器 ========
        lora_params = self.pipe.get_lora_trainable_parameters()
        self.lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.lora_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        
        logger.info(f"[BilevelGuidance] LoRA optimizer: AdamW, lr={self.lora_lr}, "
                     f"params={sum(p.numel() for p in lora_params):,}")
        logger.info(f"[BilevelGuidance] Params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
                     f"ada={self.ada_normalize}, cfg={self.true_cfg_scale}, noise_mode={self.noise_mode}")
    
    # =========================================================================
    # Pipeline 调用（实现抽象方法）
    # =========================================================================
    
    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        B: int,
        V: int,
    ) -> BilevelDistillationOutput:
        """
        调用 Bilevel Distillation Pipeline（教师/学生双前向）。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            BilevelDistillationOutput: tracker + student_loss_context
        """
        rendered_pil = self.tensor_to_pil(comp_rgb[0].cpu())
        condition_pil = composite_alpha_to_white(condition_images[0])
        image_list = [rendered_pil, condition_pil]
        
        return self.pipe(
            image=image_list,
            prompt=self.target_prompt,
            negative_prompt=self.negative_prompt,
            src_latent=src_latent.to(torch.bfloat16),
            height=self.edit_resolution,
            width=self.edit_resolution,
            min_step_percent=self.min_step_percent,
            max_step_percent=self.max_step_percent,
            true_cfg_scale=self.true_cfg_scale,
            num_timesteps=self.num_timesteps,
            noise_mode=self.noise_mode,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        )
    
    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================
    
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: BilevelDistillationOutput,
        comp_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算外层 VSD Loss（通过 Tracker.loss()，复用 CSD 体系）。
        
        VSD Loss 结构：
            x0_pos = x0_teacher（吸引），x0_neg = x0_student（排斥）
            loss = csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: Pipeline 输出
            comp_rgb: [N, C, H, W] 渲染图（未使用）
        
        Returns:
            (loss, loss_dict)
        """
        tracker = pipeline_output.tracker
        
        loss = tracker.loss(
            src=src_latent,
            mse_weight=self.mse_weight,
            csd_weight=self.csd_weight,
            ada=self.ada_normalize,
            eps=self.ada_eps,
            reduce=self.reduce_mode,
        )  # scalar
        
        return loss, {}
    
    # =========================================================================
    # 主入口（重写：加入学生 LoRA 更新步）
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> GuidanceResult:
        """
        计算 Guidance loss（双层优化）。
        
        完整流程：
            1. 格式转换
            2. 编码到 latent（有梯度）
            3. 教师/学生双前向（Pipeline，无梯度）
            4. 外层 VSD Loss（梯度流向 3D 模型）
            5. 学生带梯度前向 → LoRA backward + step
            6. 返回外层 loss
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C) 或 (B,V,C,H,W)
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 额外参数
        
        Returns:
            GuidanceResult: 包含外层 VSD loss
        """
        # 1. 格式转换
        comp_rgb, B, V, C, H, W, source_device = self._reshape_input(comp_rgb)
        
        # 2. 编码到 latent（有梯度，流向 3D 模型）
        src_latent = self.encode_to_latent(comp_rgb)  # [N, seq, C_lat]
        
        # 3. 教师/学生双前向（Pipeline，无梯度）
        with torch.no_grad():
            pipeline_output = self._run_pipeline(
                comp_rgb,
                condition_images,
                src_latent=src_latent.detach(),
                B=B, V=V,
            )
        
        # 4. 外层 VSD Loss（梯度流向 3D 模型 → src_latent → VAE → comp_rgb）
        loss_vsd, loss_dict = self._compute_loss(
            src_latent,
            pipeline_output,
            comp_rgb,
        )
        
        # 5. 学生带梯度前向 + LoRA 更新（梯度仅流向 LoRA 参数）
        student_context = pipeline_output.student_loss_context
        if student_context is not None:
            loss_student = self.pipe.compute_student_loss(
                student_context,
                lambda_sup=self.lambda_sup,
            )  # scalar, float32
            
            # LoRA backward + step
            self.lora_optimizer.zero_grad()
            loss_student.backward()
            self.lora_optimizer.step()
            
            # 记录学生 loss（用于日志）
            loss_dict["student_loss"] = loss_student.detach()
            loss_dict["lora_stats"] = self.pipe.log_lora_param_ranges()
        
        # 6. 移动到训练设备并返回
        loss_vsd = loss_vsd.to(self.train_device)
        
        return self._build_result(
            loss_vsd, loss_dict, pipeline_output, B, V, C, H, W, source_device
        )
    
    # =========================================================================
    # LoRA 管理接口（对外暴露）
    # =========================================================================
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """导出 LoRA 权重，用于保存 checkpoint。"""
        return self.pipe.get_lora_state_dict()
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载 LoRA 权重，用于恢复 checkpoint。"""
        current = dict(self.pipe.transformer.named_parameters())
        for name, param in state_dict.items():
            if name in current:
                current[name].data.copy_(param)
            else:
                logger.warning(f"[BilevelGuidance] LoRA key not found: {name}")
    
    def cleanup(self) -> None:
        """释放资源（含 LoRA 优化器）。"""
        if hasattr(self, 'lora_optimizer'):
            del self.lora_optimizer
        super().cleanup()
