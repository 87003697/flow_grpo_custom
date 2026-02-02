"""Trellis Flow Matching Scheduler"""
import torch
from types import SimpleNamespace
from typing import Optional, Union

from trellis.modules.sparse import SparseTensor

from edit4shape.generators.base.scheduler import FlowScheduler
from edit4shape.generators.base.rollout import sde_step_with_logprob_sparse


class TrellisFlowScheduler(FlowScheduler):
    """
    Trellis 专用 Flow Matching 调度器。
    
    继承基类的 set_timesteps、sigmas、index_for_timestep，
    额外提供 SparseTensor 的 ODE step 和 SDE step 方法。
    """
    
    def step(self, noise_pred: SparseTensor, t: Union[float, torch.Tensor], latents: SparseTensor) -> SimpleNamespace:
        """
        Euler ODE 步进：x_{t-1} = x_t - (t - t_prev) * v
        
        Args:
            noise_pred: 速度场预测 v，SparseTensor
            t: 当前时间步，标量
            latents: 当前样本 x_t，SparseTensor
            
        Returns:
            SimpleNamespace(prev_sample=SparseTensor, pred_original_sample=None)
        """
        t_val = float(t)
        idx = self.index_for_timestep(t_val)
        t_prev = float(self.timesteps[idx + 1].item())
        delta = t_val - t_prev  # 时间差
        
        pred_feats = latents.feats - delta * noise_pred.feats  # (N, C)
        prev_sample = SparseTensor(coords=latents.coords, feats=pred_feats)
        
        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=None)

    def sde_step(
        self,
        noise_pred: SparseTensor,
        t: Union[float, torch.Tensor],
        latents: SparseTensor,
        noise_level: float = 0.7,
        prev_sample: Optional[SparseTensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_type: str = 'sde',
        return_sqrt_dt: bool = False,
    ):
        """
        SDE 步进，调用 rollout.sde_step_with_logprob_sparse。
        
        Args:
            noise_pred: 速度场预测 v，SparseTensor
            t: 当前时间步，标量 float
            latents: 当前样本 x_t，SparseTensor
            noise_level: 噪声强度 (默认 0.7)
            prev_sample: 如果提供，使用此样本计算 log_prob
            generator: 随机数生成器
            sde_type: 'sde' 或 'cps'
            return_sqrt_dt: 是否返回 sqrt_dt
            
        Returns:
            prev_sample: SparseTensor
            log_prob: Tensor (B,)
            prev_sample_mean: SparseTensor
            std_dev_t: Tensor
            (可选) sqrt_dt: Tensor
        """
        return sde_step_with_logprob_sparse(
            self,
            model_output=noise_pred,
            timestep=float(t),
            sample=latents,
            noise_level=noise_level,
            prev_sample=prev_sample,
            generator=generator,
            sde_type=sde_type,
            return_sqrt_dt=return_sqrt_dt,
        )
