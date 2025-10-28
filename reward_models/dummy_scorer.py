"""
轻量 DummyScorer：用于低显存调试的极简奖励器。

特性：
- 不依赖外部大模型与权重，CPU/GPU 皆可运行；
- 仅基于 mesh 顶点的简单几何启发式产生分数，保证分组内(K个候选)有可区分性；
- 接口对齐其它 scorer：提供 compute_scores(meshes, images, metadata)。
"""

from typing import Any, Dict, List

import torch


class DummyScorer:
    """基于几何启发式的超轻量评分器。

    思路：
    - 分数越高表示几何越“居中且紧凑”。
    - 使用顶点的质心范数与包围盒对角线长度作为两项指标，组合成 [0,1] 的分数。
    - 该启发式对不同候选通常产生差异，便于调试优势计算与训练流程。
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def _get_vertices(self, mesh: Any) -> torch.Tensor:
        """从多种 mesh 表达中提取顶点张量到 device。

        返回: (V,3) float32 张量。
        """
        v = getattr(mesh, "v", None)
        if v is None and hasattr(mesh, "vertices"):
            v = mesh.vertices  # (V,3)
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float32)  # (V,3)
        return v.to(self.device, dtype=torch.float32)  # (V,3)

    @torch.no_grad()
    def compute_scores(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """为每个 mesh 计算一个 [0,1] 的启发式分数。

        返回: (scores_list, grouped_meta)
        - scores_list: 长度 K 的 float 列表
        - grouped_meta: 为空列表（保留接口一致性）
        """
        scores: List[float] = []

        for mesh in meshes:
            v = self._get_vertices(mesh)  # 形状: (V,3)
            if v.numel() == 0:
                scores.append(0.5)
                continue

            centroid = v.mean(dim=0)  # 形状: (3,)
            center_norm = centroid.norm(p=2)  # 形状: ()

            v_min = v.min(dim=0).values  # 形状: (3,)
            v_max = v.max(dim=0).values  # 形状: (3,)
            diag = (v_max - v_min).norm(p=2)  # 形状: ()

            # 将两项映射到 [0,1] 并组合：
            # - 中心越接近原点越好：score_center = 1/(1+||c||)
            # - 尺度不过大（<=2 近似标准化单位球）越好：score_scale = 1/(1+max(0, diag-2))
            score_center = 1.0 / (1.0 + float(center_norm.item()))  # 形状: 标量
            scale_excess = max(0.0, float(diag.item()) - 2.0)  # 形状: 标量
            score_scale = 1.0 / (1.0 + scale_excess)  # 形状: 标量

            score = 0.6 * score_center + 0.4 * score_scale  # 形状: 标量
            # 裁剪到 [0,1]
            score = float(max(0.0, min(1.0, score)))  # 形状: 标量
            scores.append(score)

        grouped_meta: List[Dict[str, Any]] = []  # 形状: 长度 G 的列表（此处空）
        return scores, grouped_meta


