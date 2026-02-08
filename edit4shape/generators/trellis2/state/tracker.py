# =====================================================================
# Imports
# =====================================================================
from typing import Any, Dict, List

import torch

# =====================================================================
# Debug Tracker - 极简调试跟踪器
# =====================================================================

class DebugTracker:
    """
    极简调试跟踪器，用于 rollout 过程中的中间变量跟踪。
    
    只需在循环中添加一行 log() 调用，调试完删除即可。
    自动对 Tensor 进行 detach + cpu 处理以节省显存。
    
    使用示例:
        tracker = DebugTracker()
        for t in timesteps:
            ...
            tracker.log(t=t_val, latents=latents, velocity=velocity)
            ...
        
        # 分析
        print(tracker["latents"])  # 所有步的 latents 列表
        print(tracker["velocity"])  # 所有步的 velocity 列表
        print(len(tracker))  # 总步数
    """
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    def log(self, **kwargs) -> None:
        """
        记录任意 key-value。Tensor 会自动 detach + cpu。
        
        Args:
            **kwargs: 任意键值对，如 t=0.5, latents=latents, velocity=velocity
        """
        processed = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.detach().cpu()
            else:
                processed[k] = v
        self.data.append(processed)
    
    def __getitem__(self, key: str) -> List[Any]:
        """获取所有步中某个 key 的值列表，如 tracker["latents"]"""
        return [d.get(key) for d in self.data if key in d]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def clear(self) -> None:
        """清空所有记录"""
        self.data = []
    
    def __repr__(self) -> str:
        keys = set()
        for d in self.data:
            keys.update(d.keys())
        return f"DebugTracker(steps={len(self.data)}, keys={keys})"

