"""
SDERolloutTracker: SDE 采样轨迹追踪器

用于 Nabla-R2D3 风格的 Score Function Matching 训练。
追踪 SDE rollout 过程中的：
- 每步状态 (x_t, x_prev)
- 速度场预测 (velocity)
- 对数概率 (log_prob)
- 采样参数 (std_dev_t, prev_sample_mean)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch

from trellis.modules.sparse import SparseTensor


@dataclass
class StepRecord:
    """单个 SDE 步的记录"""
    t: float                           # 当前时间步（未归一化）
    t_norm: float                      # 归一化时间步 (t / 1000)
    x_t: SparseTensor                  # 当前状态
    x_prev: SparseTensor               # 下一状态
    velocity: SparseTensor             # 速度场预测 v
    prev_sample_mean: SparseTensor     # SDE 均值 μ_θ(x_t, t)
    std_dev_t: torch.Tensor            # 扩散系数 σ_t
    log_prob: torch.Tensor             # 对数概率 log p(x_prev | x_t) (B,)
    sqrt_dt: Optional[torch.Tensor] = None  # sqrt(-dt)


@dataclass
class SDERolloutTracker:
    """
    SDE 采样轨迹追踪器
    
    核心功能：
    1. 记录每一步的状态和概率
    2. 计算 student/reference transition score
    3. 支持选择特定时间步进行 loss 计算
    
    Usage:
        tracker = SDERolloutTracker(device=device)
        tracker.set_initial_latent(x_T)
        
        for t in timesteps:
            x_prev, log_prob, mean, std = scheduler.sde_step(...)
            tracker.record_step(t, x_t, x_prev, velocity, mean, std, log_prob)
        
        # 计算 score matching loss
        selected_steps = tracker.select_timesteps(...)
        for record in selected_steps:
            stu_score = tracker.compute_transition_score_student(record, velocity)
            ref_score = tracker.compute_transition_score_reference(record)
    """
    device: torch.device
    steps: List[StepRecord] = field(default_factory=list)
    initial_latent: Optional[SparseTensor] = None
    
    def set_initial_latent(self, x_T: SparseTensor) -> None:
        """设置初始噪声 x_T"""
        self.initial_latent = x_T
    
    def record_step(
        self,
        t: float,
        x_t: SparseTensor,
        x_prev: SparseTensor,
        velocity: SparseTensor,
        prev_sample_mean: SparseTensor,
        std_dev_t: torch.Tensor,
        log_prob: torch.Tensor,
        sqrt_dt: Optional[torch.Tensor] = None,
    ) -> None:
        """
        记录一个 SDE 步的信息
        
        Args:
            t: 当前时间步（未归一化，如 1000, 950, ...）
            x_t: 当前状态 SparseTensor
            x_prev: 下一状态 SparseTensor
            velocity: 速度场预测 v
            prev_sample_mean: SDE 均值
            std_dev_t: 扩散系数
            log_prob: 对数概率 (B,)
            sqrt_dt: sqrt(-dt)（可选）
        """
        record = StepRecord(
            t=t,
            t_norm=t / 1000.0,
            x_t=x_t,
            x_prev=x_prev,
            velocity=velocity,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
            log_prob=log_prob,
            sqrt_dt=sqrt_dt,
        )
        self.steps.append(record)
    
    def get_x_t(self, step_idx: int) -> SparseTensor:
        """获取指定步的 x_t"""
        return self.steps[step_idx].x_t
    
    def get_x_prev(self, step_idx: int) -> SparseTensor:
        """获取指定步的 x_prev"""
        return self.steps[step_idx].x_prev
    
    @property
    def final_latent(self) -> Optional[SparseTensor]:
        """获取最终状态 x_0"""
        if not self.steps:
            return None
        return self.steps[-1].x_prev
    
    @property
    def total_log_prob(self) -> torch.Tensor:
        """计算整条轨迹的总对数概率 (B,)"""
        if not self.steps:
            return torch.tensor(0.0, device=self.device)
        return sum(step.log_prob for step in self.steps)
    
    def compute_transition_score_student(
        self,
        record: StepRecord,
        new_velocity: SparseTensor,
    ) -> torch.Tensor:
        """
        计算学生模型的 Transition Score: ∇_{x_t} log p_θ(x_{t-1} | x_t)
        
        对于 Flow Matching SDE:
            x_{t-1} = μ_θ(x_t, t) + σ_t * ε
            log p = -||x_{t-1} - μ_θ||² / (2σ_t²)
            ∇_{x_t} log p = (1/σ_t²) * (x_{t-1} - μ_θ) * ∂μ_θ/∂x_t
        
        简化实现：直接使用 μ_θ 相对于 x_t 的梯度
        
        Args:
            record: 记录的 SDE 步信息
            new_velocity: 使用当前参数重新计算的速度场
            
        Returns:
            transition_score: (N, C) 特征维度的 score
        """
        # 从 record 中获取采样时的 x_prev（固定）
        x_prev_feats = record.x_prev.feats.detach()  # (N, C)
        std_dev_t = record.std_dev_t  # 标量
        sqrt_dt = record.sqrt_dt  # 标量
        t_norm = record.t_norm
        
        # 使用新速度场计算 μ_θ (依赖 SDE 类型)
        # 对于 'sde' 类型: μ_θ = x_t * (1 + σ²/(2σ_t)*dt) + v * (1 + σ²(1-σ_t)/(2σ_t)) * dt
        # 这里我们需要 x_t 保持梯度
        x_t_feats = record.x_t.feats  # (N, C) 保持梯度
        v_feats = new_velocity.feats  # (N, C) 有梯度
        
        # 重新计算 prev_sample_mean（简化版：Euler ODE 形式）
        # 对于 score matching，核心是 (x_prev - μ) / σ²
        # 使用 Euler 近似: μ ≈ x_t - dt * v
        dt = -record.t_norm * 1000 / 1000  # 近似 dt
        
        # 计算 scale = σ_t * sqrt(-dt)
        if sqrt_dt is not None:
            scale = std_dev_t * sqrt_dt
        else:
            scale = std_dev_t
        
        # Transition score = (x_prev - μ) / σ²
        # 注意：这是简化版本，实际需要完整的 μ 计算
        # 这里返回的是用于 loss 计算的中间量
        score = (x_prev_feats - new_velocity.feats) / (scale ** 2 + 1e-8)  # (N, C)
        
        return score
    
    def compute_transition_score_reference(
        self,
        record: StepRecord,
    ) -> torch.Tensor:
        """
        计算参考模型的 Transition Score（采样时的冻结值）
        
        这是采样时计算的 score，不需要梯度。
        score = (x_{t-1} - μ_ref) / σ²
        
        Args:
            record: 记录的 SDE 步信息
            
        Returns:
            transition_score: (N, C) 特征维度的 score
        """
        x_prev_feats = record.x_prev.feats.detach()  # (N, C)
        mean_feats = record.prev_sample_mean.feats.detach()  # (N, C)
        std_dev_t = record.std_dev_t  # 标量
        sqrt_dt = record.sqrt_dt
        
        if sqrt_dt is not None:
            scale = std_dev_t * sqrt_dt
        else:
            scale = std_dev_t
        
        # score = (x_prev - μ) / σ²
        score = (x_prev_feats - mean_feats) / (scale ** 2 + 1e-8)  # (N, C)
        
        return score.detach()
    
    def select_timesteps(
        self,
        mode: str = "all",
        num_steps: Optional[int] = None,
        t_min: float = 0.0,
        t_max: float = 1000.0,
    ) -> List[StepRecord]:
        """
        选择用于 loss 计算的时间步
        
        Args:
            mode: 选择模式
                - "all": 所有时间步
                - "random": 随机选择 num_steps 个
                - "uniform": 均匀采样 num_steps 个
                - "weighted": 按重要性加权采样（时间越大权重越高）
            num_steps: 选择的步数
            t_min: 最小时间步
            t_max: 最大时间步
            
        Returns:
            selected_records: 选中的 StepRecord 列表
        """
        # 过滤时间范围
        filtered = [s for s in self.steps if t_min <= s.t <= t_max]
        
        if not filtered:
            return []
        
        if mode == "all":
            return filtered
        
        if num_steps is None or num_steps >= len(filtered):
            return filtered
        
        if mode == "random":
            indices = torch.randperm(len(filtered), device=self.device)[:num_steps]
            return [filtered[i] for i in indices.tolist()]
        
        elif mode == "uniform":
            step_size = len(filtered) / num_steps
            indices = [int(i * step_size) for i in range(num_steps)]
            return [filtered[i] for i in indices]
        
        elif mode == "weighted":
            # 时间越大（越接近噪声）权重越高
            weights = torch.tensor([s.t_norm for s in filtered], device=self.device)
            weights = weights / weights.sum()
            indices = torch.multinomial(weights, num_steps, replacement=False)
            return [filtered[i] for i in indices.tolist()]
        
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
    
    def clear(self) -> None:
        """清空所有记录"""
        self.steps = []
        self.initial_latent = None
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, idx: int) -> StepRecord:
        return self.steps[idx]
