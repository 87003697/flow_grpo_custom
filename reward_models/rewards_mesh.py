"""
默认 Mesh 奖励器（来自 V4，模块化、无懒加载、无回退）。
详见类与方法注释。
"""

from typing import Any, Dict, List, Optional
from PIL import Image
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

        # 兼容不同 mesh 类型：
        # - 旧流程/自定义 Mesh: 使用属性 `v` (torch.Tensor)
        # - Direct3D / Trimesh: 返回 `trimesh.Trimesh`，顶点属性为 `vertices` (np.ndarray / torch.Tensor)
        # 这里统一访问并在需要时转换为 torch.Tensor 放入临时字段 `v`，避免后续代码分支。
        m0 = meshes[0]
        v0 = getattr(m0, 'v', None)
        if v0 is None and hasattr(m0, 'vertices'):
            verts = m0.vertices
            # trimesh.Trimesh.vertices 可能是 (N,3) 的 numpy.ndarray
            if not isinstance(verts, torch.Tensor):
                verts = torch.from_numpy(verts).to(self.device)
            # 缓存到对象上，后续旋转直接原地修改 (不修改 m0.vertices 以免破坏 trimesh 内部缓存)
            setattr(m0, 'v', verts)
            v0 = verts
        assert isinstance(v0, torch.Tensor), "mesh.v 必须为 torch.Tensor 或能从 vertices 转换"
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
            mv = getattr(m, 'v', None)
            if mv is None and hasattr(m, 'vertices'):
                verts = m.vertices
                if not isinstance(verts, torch.Tensor):
                    verts = torch.from_numpy(verts).to(device=device, dtype=dtype)
                setattr(m, 'v', verts)
                mv = verts
            if isinstance(mv, torch.Tensor):
                mv_rot = mv @ T  # 形状: (N,3)
                # 回写：如果原始对象具有 vertices 且不是 torch.Tensor，需要同步 numpy 以便后续可能使用
                setattr(m, 'v', mv_rot)
                if hasattr(m, 'vertices'):
                    try:
                        # 尝试同步回 trimesh 的顶点 (trimesh.Trimesh.vertices 是 np.ndarray)
                        import numpy as _np
                        m.vertices = mv_rot.detach().cpu().to(torch.float32).numpy().astype(_np.float32)
                    except Exception:
                        pass

    def _build_components(self, weights: Dict[str, float]) -> None:
        """根据初始化期权重字典构建评分组件。"""
        self._uni3d = None
        self._camera_normal = None
        self._dummy = None

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

        if ("dummy" in weights) and (float(weights["dummy"]) > 0.0):
            if self.verbose:
                print("⏳ 构建 DummyScorer ...")
            from reward_models.dummy_scorer import DummyScorer
            self._dummy = DummyScorer(self.device)

    def _score_uni3d(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """计算 Uni3D 评分，并返回 (K,) 数组与每图最佳/最差配对元数据列表。"""
        scores_u, grouped_meta = self._uni3d.compute_scores(meshes, images, metadata)  # 形状: 长度 K 的列表, 长度 G 的分组meta
        arr_u = np.array(scores_u, dtype=np.float32)  # 形状: (K,)
        # 由 Uni3D scorer 内部构建 best/worst 配对
        pairs_best: List[Dict[str, Any]] = []  # 形状: 列表
        pairs_worst: List[Dict[str, Any]] = []  # 形状: 列表
        pairs_best, pairs_worst = self._uni3d.build_best_worst_pairs(
            meshes, images, grouped_meta, arr_u, R=256
        )  # 形状: 列表, 列表
        return arr_u, pairs_best, pairs_worst  # 形状: (K,), 长度 G 的列表, 长度 G 的列表


    def _score_camera_normal(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """计算 CameraNormal 评分，返回 (K,) 数组与每图最佳/最差配对元数据列表。

        在评分前根据 source_front 将 mesh 前向对齐到 +z。
        """
        # 若缺失 normal_pil 元数据，返回零评分以保持流程不中断（最小可用阶段）。
        if len(metadata) == 0 or (('normal_pil' not in metadata[0]) or (metadata[0].get('normal_pil') is None)):
            arr_cn = np.zeros(len(meshes), dtype=np.float32)
            return arr_cn, [], []
        self._rotate_by_source_front(meshes)
        scores_cn, grouped_meta = self._camera_normal.compute_scores(  # 形状: 长度 K 的列表, 长度 G 的分组meta
            meshes=meshes,
            images=images,
            metadata=metadata,
        )
        arr_cn = np.array(scores_cn, dtype=np.float32)  # 形状: (K,)
        # 交给 camera_normal_scorer 内部的方法构造 best/worst 列表（保持职责内聚）
        filtered_meta_best, filtered_meta_worst = self._camera_normal.build_best_worst_pairs(grouped_meta)
        return arr_cn, filtered_meta_best, filtered_meta_worst  # 形状: (K,), 长度 G 的列表

    def _score_dummy(
        self,
        meshes: List[Any],
        images: List[Any],
        metadata: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """计算 Dummy 评分，返回 (K,) 数组与空配对列表。"""
        scores_d, _ = self._dummy.compute_scores(meshes, images, metadata)  # 形状: 长度 K 的列表
        arr_d = np.array(scores_d, dtype=np.float32)  # 形状: (K,)
        return arr_d, [], []

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

    # 统一对 images 做 RGBA→白底 RGB 的 alpha 合成，确保下游（如 Uni3D/CLIP）三通道输入
    @staticmethod
    def alpha_composite_white(images: List[Any]) -> List[Any]:
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                if img.mode == 'RGBA':
                    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    out.append(Image.alpha_composite(bg, img).convert('RGB'))
                else:
                    out.append(img.convert('RGB'))
            else:
                out.append(img)
        return out

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
        if ("dummy" in enabled) and (self._dummy is None):
            raise RuntimeError("dummy 权重>0，但组件未构建。")

        images_proc = self.alpha_composite_white(images)

        parts: Dict[str, np.ndarray] = {}  # 形状: 字典
        meta_out: Dict[str, Any] = {}  # 形状: 字典
        if "uni3d" in enabled:
            arr_u, meta_u_best, meta_u_worst = self._score_uni3d(meshes, images_proc, metadata)  # 形状: (K,), 列表, 列表
            parts["uni3d"] = arr_u  # 形状: (K,)
            meta_out["uni3d_pairs_best"] = meta_u_best  # 形状: 长度 G 的列表
            meta_out["uni3d_pairs_worst"] = meta_u_worst  # 形状: 长度 G 的列表
        if "camera_normal" in enabled:
            arr_cn, meta_cn_best, meta_cn_worst = self._score_camera_normal(meshes, images_proc, metadata)
            parts["camera_normal"] = arr_cn  # 形状: (K,)
            # 同时输出最佳与最差配对，供上层可视化/记录
            meta_out["camera_normal_pairs_best"] = meta_cn_best  # 形状: 长度 G 的列表
            meta_out["camera_normal_pairs_worst"] = meta_cn_worst  # 形状: 长度 G 的列表
        if "dummy" in enabled:
            arr_d, _, _ = self._score_dummy(meshes, images_proc, metadata)
            parts["dummy"] = arr_d  # 形状: (K,)

        weighted = self._aggregate_scores(num, enabled, parts, score_fns_cfg)  # 形状: (K,)
        details: Dict[str, np.ndarray] = {**parts, "avg": weighted}  # 形状: 字典
        return details, meta_out  # 形状: (字典, 字典)


def preload_scorers(score_fns_cfg: Dict[str, float], device: torch.device, verbose: bool = False):
    """默认实现不需要显式预加载，保留占位以兼容。"""
    if bool(verbose):
        print(f"✅ 预加载占位完成: {device}")