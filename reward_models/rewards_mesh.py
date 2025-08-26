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
    
    def __init__(self, device="cuda", verbose: bool = False):
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        if self.verbose:
            print(f"🔧 初始化MeshScorer: {self.device}")
        
        # 一次性加载所有模型
        from reward_models.uni3d_scorer.simple_uni3d import SimpleUni3DScorer
        self.uni3d_scorer = SimpleUni3DScorer(self.device, verbose=self.verbose)
        # 懒加载 camera_normal_scorer（按需）
        self._camera_normal_scorer = None
        self._mesh_renderer = None
        if self.verbose:
            print(f"✅ MeshScorer初始化完成: {self.device}")
    
    def score(self, meshes, images, metadata, score_fns_cfg):
        """计算mesh评分"""
        weighted = np.zeros(len(meshes), dtype=np.float32)
        details = {}

        # uni3d
        if "uni3d" in score_fns_cfg and score_fns_cfg["uni3d"] > 0:
            scores = self.uni3d_scorer.compute_scores(meshes, images)
            details["uni3d"] = scores
            weighted += np.array(scores, dtype=np.float32) * float(score_fns_cfg["uni3d"])  # 形状: (K,)

        # camera_normal
        if "camera_normal" in score_fns_cfg and score_fns_cfg["camera_normal"] > 0:
            if self._camera_normal_scorer is None:
                from reward_models.camera_normal_scorer import CameraNormalScorer
                # 从环境读取配置（训练脚本会把 config.camera_normal 注入）
                cfg = getattr(self, "camera_normal_cfg", None)
                if cfg is None:
                    raise ValueError("camera_normal 配置未设置到 MeshScorer.camera_normal_cfg")
                self._camera_normal_scorer = CameraNormalScorer(self.device, cfg)
            if self._mesh_renderer is None:
                from generators.trellis.renderers.renderers.mesh_renderer import MeshRenderer
                # 采用白底仅 normal 渲染，设置 R/near/far/ssaa
                R = int(self._camera_normal_scorer.resolution)
                self._mesh_renderer = MeshRenderer(
                    rendering_options={"resolution": R, "near": 0.1, "far": 10.0, "ssaa": 2},
                    device=str(self.device)
                )

            # metadata 需包含 image_path 或 image_name
            scores_cn = self._camera_normal_scorer.compute_scores(
                meshes=meshes,
                images=images,
                metadata=metadata,
                renderer=self._mesh_renderer,
            )
            details["camera_normal"] = scores_cn
            weighted += np.array(scores_cn, dtype=np.float32) * float(score_fns_cfg["camera_normal"])  # 形状: (K,)

        # 若无任何项，默认 0.5
        if len(details) == 0:
            weighted = np.ones(len(meshes), dtype=np.float32) * 0.5
        
        return {"avg": weighted, **details}, {}

# 向后兼容的接口 - 但不推荐使用，应该直接用MeshScorer类
def multi_mesh_score(meshes, images, metadata, score_fns_cfg):
    """向后兼容的接口 - 每次都创建新实例，不高效"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = MeshScorer(device, verbose=False)  # 每次都创建新实例
    return scorer.score(meshes, images, metadata, score_fns_cfg)

def preload_scorers(score_fns_cfg: Dict[str, float], device: torch.device, verbose: bool = False):
    """预加载占位函数 - 实际初始化在MeshScorer.__init__中"""
    if bool(verbose):
        print(f"✅ 预加载占位完成: {device}") 