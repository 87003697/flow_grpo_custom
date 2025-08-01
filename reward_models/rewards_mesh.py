"""
3D Mesh 奖励函数 - Hunyuan3D 专用 (类实现版)
用于计算生成的3D网格的质量评分
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from kiui.mesh import Mesh

class MeshScorer:
    """Mesh质量评分器 - 一次初始化，重复使用"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        print(f"🔧 初始化MeshScorer: {self.device}")
        
        # 一次性加载所有模型
        from reward_models.uni3d_scorer.simple_uni3d import SimpleUni3DScorer
        self.uni3d_scorer = SimpleUni3DScorer(self.device)
        print(f"✅ MeshScorer初始化完成: {self.device}")
    
    def score(self, meshes, images, metadata, score_fns_cfg):
        """计算mesh评分"""
        if "uni3d" in score_fns_cfg and score_fns_cfg["uni3d"] > 0:
            scores = self.uni3d_scorer.compute_scores(meshes, images)
            weighted_scores = np.array(scores) * score_fns_cfg["uni3d"]
        else:
            weighted_scores = np.ones(len(meshes)) * 0.5
        
        return {"avg": weighted_scores}, {}

# 向后兼容的接口 - 但不推荐使用，应该直接用MeshScorer类
def multi_mesh_score(meshes, images, metadata, score_fns_cfg):
    """向后兼容的接口 - 每次都创建新实例，不高效"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = MeshScorer(device)  # 每次都创建新实例
    return scorer.score(meshes, images, metadata, score_fns_cfg)

def preload_scorers(score_fns_cfg: Dict[str, float], device: torch.device):
    """预加载占位函数 - 实际初始化在MeshScorer.__init__中"""
    print(f"✅ 预加载占位完成: {device}") 