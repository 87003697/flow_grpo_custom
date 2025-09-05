"""
默认 Mesh 奖励器（来自 V4，模块化、无懒加载、无回退）。
详见类与方法注释。
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch


class MeshScorer:
    """默认 Mesh 评分器（V4 版）。

    功能：
        - 初始化阶段一次性构建启用项的组件。
        - 调用 score 时按传入权重执行对应评分并加权汇总。

    输入参数：
        - score_fns_cfg: Dict[str, float]
        - device: str | torch.device
        - verbose: bool
        - camera_normal_cfg: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        score_fns_cfg: Dict[str, float],
        device: str | torch.device = "cuda",
        verbose: bool = False,
        camera_normal_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        self.score_fns_cfg: Dict[str, float] = dict(score_fns_cfg)
        self.camera_normal_cfg: Optional[Dict[str, Any]] = (
            dict(camera_normal_cfg) if camera_normal_cfg is not None else None
        )

        if self.verbose:
            print(f"🔧 初始化默认 MeshScorer: {self.device}")

        self._build_components(self.score_fns_cfg)

        if self.verbose:
            print("✅ 默认 MeshScorer 初始化完成")

    # 旋转：封装为函数（仅依赖 camera_normal.source_front，目标前向固定为 +z）

    def _rotate_by_source_front(self, meshes: List[Any]) -> None:
        """按 kiui front_dir 规则将源朝向旋到 +z（就地修改，不恢复）。"""
        if len(meshes) == 0 or self.camera_normal_cfg is None:
            return
        src = str(self.camera_normal_cfg.get("source_front", "+z"))  # 形状: 字符串
        if src == "+z":
            return

        v0 = getattr(meshes[0], 'v', None)
        assert isinstance(v0, torch.Tensor), "mesh.v 必须为 torch.Tensor"
        device, dtype = v0.device, v0.dtype  # 形状: 标量, 标量

        k = 0  # 形状: 标量
        if src.endswith('1'): k = 1
        elif src.endswith('2'): k = 2
        elif src.endswith('3'): k = 3
        base = src[:-1] if k > 0 else src  # 形状: 字符串

        if base == "-z":
            T = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], device=device, dtype=dtype)  # 形状: (3,3)
        elif base == "+x":
            T = torch.tensor([[0,0,1],[0,1,0],[1,0,0]], device=device, dtype=dtype)  # 形状: (3,3)
        elif base == "-x":
            T = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]], device=device, dtype=dtype)  # 形状: (3,3)
        elif base == "+y":
            T = torch.tensor([[1,0,0],[0,0,1],[0,1,0]], device=device, dtype=dtype)  # 形状: (3,3)
        elif base == "-y":
            T = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], device=device, dtype=dtype)  # 形状: (3,3)
        else:
            T = torch.eye(3, device=device, dtype=dtype)  # 形状: (3,3)

        if k == 1:
            T = T @ torch.tensor([[0,-1,0],[1,0,0],[0,0,1]], device=device, dtype=dtype)  # 形状: (3,3)
        elif k == 2:
            T = T @ torch.tensor([[1,0,0],[0,-1,0],[0,0,1]], device=device, dtype=dtype)  # 形状: (3,3)
        elif k == 3:
            T = T @ torch.tensor([[0,1,0],[-1,0,0],[0,0,1]], device=device, dtype=dtype)  # 形状: (3,3)

        for m in meshes:
            m.v = m.v @ T  # 形状: (N,3)

    def _build_components(self, weights: Dict[str, float]) -> None:
        """根据初始化期权重字典构建评分组件。"""
        self._uni3d = None
        self._camera_normal = None

        if ("uni3d" in weights) and (float(weights["uni3d"]) > 0.0):
            if self.verbose:
                print("⏳ 构建 SimpleUni3DScorer ...")
            from reward_models.uni3d_scorer.simple_uni3d import SimpleUni3DScorer
            self._uni3d = SimpleUni3DScorer(self.device, verbose=self.verbose)

        if ("camera_normal" in weights) and (float(weights["camera_normal"]) > 0.0):
            if self.camera_normal_cfg is None:
                raise ValueError("启用 camera_normal 时必须提供 camera_normal_cfg")
            if self.verbose:
                print("⏳ 构建 CameraNormalScorer ...")
            from reward_models.camera_normal_scorer import CameraNormalScorer
            self._camera_normal = CameraNormalScorer(self.device, dict(self.camera_normal_cfg))

    def _score_uni3d(self, meshes: List[Any], images: List[Any]) -> np.ndarray:
        """计算 Uni3D 评分，返回 (K,) 数组。"""
        scores_u = self._uni3d.compute_scores(meshes, images)  # 形状: 长度 K 的列表
        arr_u = np.array(scores_u, dtype=np.float32)  # 形状: (K,)
        return arr_u  # 形状: (K,)

    def _score_camera_normal(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """计算 CameraNormal 评分，返回 (K,) 数组与每图最佳/最差配对元数据列表。

        在评分前根据 source_front 将 mesh 前向对齐到 +z。
        """
        self._rotate_by_source_front(meshes)
        scores_cn, grouped_meta = self._camera_normal.compute_scores(  # 形状: 长度 K 的列表, 长度 G 的分组meta
            meshes=meshes,
            images=images,
            metadata=metadata,
        )
        arr_cn = np.array(scores_cn, dtype=np.float32)  # 形状: (K,)
        # 从分组元数据中选出每组分数最高与最低的候选，并展平成配对记录
        filtered_meta_best: List[Dict[str, Any]] = []  # 形状: 长度 G 的列表
        filtered_meta_worst: List[Dict[str, Any]] = []  # 形状: 长度 G 的列表
        for grp in grouped_meta:
            image_path = grp.get("image_path", "")  # 形状: 字符串
            img_pil = grp.get("image_normal_pil", None)  # 形状: PIL(R,R,3)
            cands = grp.get("candidates", [])  # 形状: 长度 K 的列表
            if len(cands) == 0:
                continue
            # 选出分数最高与最低的候选
            best = cands[0]
            worst = cands[0]
            for cand in cands[1:]:
                score_c = float(cand.get("score", -1.0))
                if score_c > float(best.get("score", -1.0)):
                    best = cand
                if score_c < float(worst.get("score", 1e9)):
                    worst = cand
            # 组装展平配对（含图像与渲染法线）
            filtered_meta_best.append({
                "image_path": image_path,                               # 形状: 字符串
                "image_normal_pil": img_pil,                            # 形状: PIL(R,R,3)
                "rendered_normal_pil": best.get("rendered_normal_pil"),# 形状: PIL(R,R,3)
                "mesh_index": int(best.get("mesh_index", -1)),         # 形状: 标量
                "score": float(best.get("score", 0.0)),                # 形状: 标量
            })
            filtered_meta_worst.append({
                "image_path": image_path,                                 # 形状: 字符串
                "image_normal_pil": img_pil,                              # 形状: PIL(R,R,3)
                "rendered_normal_pil": worst.get("rendered_normal_pil"),# 形状: PIL(R,R,3)
                "mesh_index": int(worst.get("mesh_index", -1)),         # 形状: 标量
                "score": float(worst.get("score", 0.0)),                # 形状: 标量
            })
        return arr_cn, filtered_meta_best, filtered_meta_worst  # 形状: (K,), 长度 G 的列表

    def _aggregate_scores(
        self,
        num: int,
        enabled: List[str],
        parts: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """按权重聚合各项评分，返回 (K,) 数组。"""
        weighted = np.zeros(num, dtype=np.float32)  # 形状: (K,)
        for k in enabled:
            weighted += float(weights[k]) * parts[k]  # 形状: (K,)
        return weighted  # 形状: (K,)

    def score(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]],
        score_fns_cfg: Dict[str, float],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """执行评分流程并返回各项与加权总分。"""
        num = len(meshes)  # 形状: 标量
        # 输入长度一致性校验，避免错位
        assert num == len(images) == len(metadata)

        enabled: List[str] = [k for k, v in score_fns_cfg.items() if float(v) > 0.0]  # 形状: 长度 M 的列表
        if len(enabled) == 0:
            raise ValueError("score_fns_cfg 未启用任何评分项")

        if ("uni3d" in enabled) and (self._uni3d is None):
            raise RuntimeError("uni3d 权重>0，但组件未构建。")
        if ("camera_normal" in enabled) and (self._camera_normal is None):
            raise RuntimeError("camera_normal 权重>0，但组件未构建。")

        parts: Dict[str, np.ndarray] = {}  # 形状: 字典
        meta_out: Dict[str, Any] = {}  # 形状: 字典
        if "uni3d" in enabled:
            parts["uni3d"] = self._score_uni3d(meshes, images)  # 形状: (K,)
        if "camera_normal" in enabled:
            arr_cn, meta_cn_best, meta_cn_worst = self._score_camera_normal(meshes, images, metadata)
            parts["camera_normal"] = arr_cn  # 形状: (K,)
            # 同时输出最佳与最差配对，供上层可视化/记录
            meta_out["camera_normal_pairs_best"] = meta_cn_best  # 形状: 长度 G 的列表
            meta_out["camera_normal_pairs_worst"] = meta_cn_worst  # 形状: 长度 G 的列表

        weighted = self._aggregate_scores(num, enabled, parts, score_fns_cfg)  # 形状: (K,)
        details: Dict[str, np.ndarray] = {**parts, "avg": weighted}  # 形状: 字典
        return details, meta_out  # 形状: (字典, 字典)


def preload_scorers(score_fns_cfg: Dict[str, float], device: torch.device, verbose: bool = False):
    """默认实现不需要显式预加载，保留占位以兼容。"""
    if bool(verbose):
        print(f"✅ 预加载占位完成: {device}")