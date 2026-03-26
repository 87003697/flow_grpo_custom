"""
Chunked Forward 核心数据结构。

用于将大的 SparseTensor 按空间坐标轴切分成多个小块进行处理，以降低显存峰值。

核心类：
- ChunkMeta: 切分元信息
- Chunk: 单个 chunk 的封装
- ChunkableSparseTensor: 主要的分块处理器
- MemoryMonitor: 显存监控和 chunk_size 估算

限制：
- Chunked Forward 仅支持 batch_size=1

原理：
- 将 SparseTensor 按某轴（如 z）的坐标范围切分
- 每块向外扩展 halo 区域以保证边界处卷积计算正确
- 处理完后丢弃 halo 部分的输出，合并各块结果
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Iterator, Tuple
import torch
import torch.nn as nn

# 使用绝对导入（适配移动到 edit4shape/generators/trellis2/）
from trellis2.modules.sparse import SparseTensor


@dataclass
class ChunkMeta:
    """切分元信息"""
    start: int           # 原始坐标起点
    end: int             # 原始坐标终点
    actual_halo: int     # 实际 halo 偏移（边界处可能被截断）
    original_scale: tuple  # SparseTensor._scale
    valid_mask: torch.Tensor  # 有效区域掩码 [n_chunk_points]，True 表示非 halo


class Chunk:
    """
    单个 chunk，由 ChunkableSparseTensor.chunks() 生成。
    
    提供 5 个核心方法：
    - tensor: 主 SparseTensor
    - get(name): 获取附加 SparseTensor
    - get_indexed_cache(key): 获取有效区域的 indexed cache（已过滤 halo）
    - set_result(tensor): 设置处理结果
    - set_attached_result(name, tensor): 设置附加 tensor 的结果
    """
    
    def __init__(
        self, 
        tensor: SparseTensor, 
        attached: Dict[str, SparseTensor],
        indexed_cache: Dict[str, torch.Tensor],
        meta: ChunkMeta, 
        axis: int, 
        coord_scale: int
    ):
        self._tensor = tensor
        self._attached = attached
        self._indexed_cache = indexed_cache
        self._meta = meta
        self._axis = axis
        self._coord_scale = coord_scale
        self._result: Optional[SparseTensor] = None
        self._result_attached: Dict[str, SparseTensor] = {}
    
    @property
    def tensor(self) -> SparseTensor:
        """主 tensor"""
        return self._tensor
    
    @property
    def meta(self) -> ChunkMeta:
        """切分元信息"""
        return self._meta
    
    def get(self, name: str) -> Optional[SparseTensor]:
        """获取附加的 SparseTensor"""
        return self._attached.get(name)
    
    def get_indexed_cache(self, key: str) -> Optional[torch.Tensor]:
        """获取有效区域的 indexed cache（已过滤 halo）"""
        cache = self._indexed_cache.get(key)
        if cache is None:
            return None
        return cache[self._meta.valid_mask]  # (N_valid, ...)
    
    def get_valid_feats(self, tensor: SparseTensor = None) -> torch.Tensor:
        """获取有效区域的 feats（已过滤 halo）"""
        t = tensor if tensor is not None else self._tensor
        return t.feats[self._meta.valid_mask]  # (N_valid, C)
    
    def set_result(self, tensor: SparseTensor) -> None:
        """设置主 tensor 的处理结果"""
        self._result = tensor
    
    def set_attached_result(self, name: str, tensor: SparseTensor) -> None:
        """设置附加 SparseTensor 的处理结果"""
        self._result_attached[name] = tensor


class ChunkableSparseTensor:
    """
    支持空间分块的 SparseTensor 包装器。
    
    Features:
    - 自动管理切分/合并逻辑
    - 支持携带关联的 SparseTensor（如 skip connection）
    - 支持携带按点索引的 spatial_cache（如 subdivision）
    - 保留 SparseTensor 的 scale 等属性
    
    Usage:
        # 简单用法
        result = ChunkableSparseTensor(h, axis=3, chunk_size=64, halo=8).apply(
            lambda chunk: block(chunk.tensor)
        )
        
        # 携带附加 tensor
        chunked = ChunkableSparseTensor(h, axis=3, chunk_size=64, halo=8)
        chunked.attach("skip", skip_tensor)
        for chunk in chunked.chunks():
            out = process(chunk.tensor, chunk.get("skip"))
            chunk.set_result(out)
        merged = chunked.merge()
        
        # 手动收集 indexed cache（如 subdivision）
        chunked = ChunkableSparseTensor(h, axis=3, chunk_size=64, halo=8)
        subdiv_list = []
        for chunk in chunked.chunks():
            subdiv_list.append(chunk.get_indexed_cache("subdivision"))
            # ...
        subdiv_gt = torch.cat(subdiv_list, dim=0)
    """
    
    def __init__(
        self,
        tensor: SparseTensor,
        axis: int = 3,
        chunk_size: int = 64,
        halo: int = 8,
        coord_scale: int = 1,
        indexed_cache_keys: Optional[List[str]] = None
    ):
        """
        Args:
            tensor: 主 SparseTensor
            axis: 切分轴 (1=x, 2=y, 3=z)
            chunk_size: chunk 大小
            halo: 边界扩展大小
            coord_scale: 合并时的坐标缩放因子（upsample 后为 2）
            indexed_cache_keys: 需要按点索引切分的 spatial_cache key 列表。
                               默认为 ['subdivision']。
        
        Raises:
            ValueError: 如果 batch_size > 1
        """
        self._tensor = tensor
        self._axis = axis
        self._chunk_size = chunk_size
        self._halo = halo
        self._coord_scale = coord_scale
        self._attached: Dict[str, SparseTensor] = {}
        self._chunks: Optional[List[Chunk]] = None
        self._indexed_cache_keys = indexed_cache_keys if indexed_cache_keys is not None else ['subdivision']
        
        # 验证 batch_size
        if tensor.coords.shape[0] > 0 and tensor.coords[:, 0].max() > 0:
            raise ValueError("ChunkableSparseTensor only supports batch_size=1")
    
    @property
    def tensor(self) -> SparseTensor:
        """获取原始/合并后的 tensor"""
        return self._tensor
    
    def attach(self, name: str, tensor: SparseTensor) -> 'ChunkableSparseTensor':
        """附加关联 SparseTensor，随主 tensor 一起切分。返回 self 支持链式调用。"""
        self._attached[name] = tensor
        return self
    
    def get_attached(self, name: str) -> Optional[SparseTensor]:
        """获取合并后的附加 SparseTensor"""
        return self._attached.get(name)
    
    def chunks(self) -> Iterator[Chunk]:
        """生成切分后的 chunks"""
        if self._chunks is None:
            self._chunks = self._split()
        for chunk in self._chunks:
            yield chunk
    
    def _split(self) -> List[Chunk]:
        """执行切分"""
        coords = self._tensor.coords  # (N, 4)
        
        # 处理空 coords
        if coords.shape[0] == 0:
            return []
        
        max_coord = coords[:, self._axis].max().item() + 1  # scalar
        
        chunks = []
        for start in range(0, max_coord, self._chunk_size):
            end = min(start + self._chunk_size, max_coord)
            halo_start = max(0, start - self._halo)
            halo_end = min(max_coord, end + self._halo)
            
            # halo 区域的点
            mask = (coords[:, self._axis] >= halo_start) & \
                   (coords[:, self._axis] < halo_end)  # (N,)
            
            # 跳过没有点的空 chunk（避免后续稀疏卷积对空张量调用 max() 报错）
            if mask.sum().item() == 0:
                continue
            
            # 计算有效区域掩码（非 halo 区域）
            valid_in_original = (coords[:, self._axis] >= start) & \
                                (coords[:, self._axis] < end)  # (N,)
            valid_mask = valid_in_original[mask]  # (N_chunk,)
            
            # 切分主 tensor 和 indexed cache
            chunk_tensor, chunk_indexed_cache = self._slice_with_cache(
                self._tensor, mask, halo_start
            )
            
            # 切分附加 SparseTensor
            chunk_attached = {}
            for name, t in self._attached.items():
                chunk_coords = t.coords[mask].clone()  # (N_chunk, 4)
                chunk_feats = t.feats[mask]  # (N_chunk, C)
                chunk_coords[:, self._axis] -= halo_start  # (N_chunk, 4)
                chunk_attached[name] = SparseTensor(chunk_feats, chunk_coords, scale=t._scale)
            
            meta = ChunkMeta(
                start=start,
                end=end,
                actual_halo=start - halo_start,
                original_scale=self._tensor._scale,
                valid_mask=valid_mask
            )
            
            chunks.append(Chunk(
                chunk_tensor, chunk_attached, chunk_indexed_cache,
                meta, self._axis, self._coord_scale
            ))
        
        return chunks
    
    def _slice_with_cache(
        self, 
        tensor: SparseTensor, 
        mask: torch.Tensor,
        offset: int
    ) -> Tuple[SparseTensor, Dict[str, torch.Tensor]]:
        """切分 tensor 并提取 indexed cache"""
        # 切分 tensor
        chunk_coords = tensor.coords[mask].clone()  # (N_chunk, 4)
        chunk_feats = tensor.feats[mask]  # (N_chunk, C)
        chunk_coords[:, self._axis] -= offset  # (N_chunk, 4)
        chunk_tensor = SparseTensor(chunk_feats, chunk_coords, scale=tensor._scale)  # SparseTensor feats: (N_chunk, C)
        
        # 切分 indexed cache
        chunk_indexed_cache = {}
        for key in self._indexed_cache_keys:
            cache = tensor.get_spatial_cache(key)
            if cache is not None and isinstance(cache, torch.Tensor):
                chunk_indexed_cache[key] = cache[mask]  # (N_chunk, ...)
        
        return chunk_tensor, chunk_indexed_cache
    
    def merge(self) -> SparseTensor:
        """
        合并所有 chunks 的结果。
        
        ★ 类型契约：始终返回 SparseTensor，退化时返回 0 点空张量。
        """
        if self._chunks is None:
            return self._tensor
        
        # 合并主 tensor
        self._tensor = self._merge_tensors(
            [(c._result, c._meta) for c in self._chunks if c._result is not None]
        )
        
        # 合并附加 SparseTensor
        attached_names = set()
        for chunk in self._chunks:
            attached_names.update(chunk._result_attached.keys())
        
        for name in attached_names:
            tensors = [(c._result_attached[name], c._meta) 
                       for c in self._chunks if name in c._result_attached]
            self._attached[name] = self._merge_tensors(tensors)
        
        for chunk in self._chunks:
            chunk._result = None
            chunk._result_attached.clear()
        
        return self._tensor
    
    def _merge_tensors(
        self, 
        tensors: List[Tuple[SparseTensor, ChunkMeta]]
    ) -> SparseTensor:
        """
        合并多个 chunk 结果，丢弃 halo 区域，按坐标规范排序。
        
        ★ 类型契约：始终返回有效的 SparseTensor（可能是 0 点的空张量），
          不返回 None。退化 latent 时返回 0 点 SparseTensor，
          由上层 ``h.feats.shape[0] == 0`` 自然触发 StageSkipError。
        """
        # ────────────────────────────────────────────────────
        # 退化保护：无 chunk 结果（输入即为空、或所有 chunk 被跳过）
        # ────────────────────────────────────────────────────
        if not tensors:
            logging.warning(
                "_merge_tensors: no chunk results (degenerate latent) "
                "→ returning 0-point SparseTensor"
            )
            device = self._tensor.feats.device
            dtype = self._tensor.feats.dtype
            C = self._tensor.feats.shape[1]  # 保留输入通道数，避免下游线性层形状不匹配
            return SparseTensor(
                torch.empty(0, C, device=device, dtype=dtype),              # (0, C)
                torch.zeros(0, 4, dtype=torch.int32, device=device),        # (0, 4)
                scale=self._tensor._scale,
            )
        
        # ────────────────────────────────────────────────────
        # 主逻辑：halo 过滤 + 全局坐标恢复 + 规范排序
        # ────────────────────────────────────────────────────
        torch.cuda.empty_cache()  # 回收碎片显存，为 merge + sort 的临时张量留空间
        
        all_coords, all_feats = [], []
        merged_scale = None
        
        for i in range(len(tensors)):
            tensor, meta = tensors[i]
            tensors[i] = None  # 释放列表对 chunk tensor 的引用
            
            if merged_scale is None:
                merged_scale = tensor._scale
            
            # 计算有效区域边界
            local_start = meta.actual_halo * self._coord_scale
            local_end = (meta.actual_halo + meta.end - meta.start) * self._coord_scale
            
            valid = (tensor.coords[:, self._axis] >= local_start) & \
                    (tensor.coords[:, self._axis] < local_end)  # (N_chunk,)
            
            # 恢复全局坐标
            valid_coords = tensor.coords[valid].clone()  # (N_valid, 4)
            valid_coords[:, self._axis] = \
                valid_coords[:, self._axis] - local_start + meta.start * self._coord_scale  # (N_valid, 4)
            
            all_coords.append(valid_coords)
            all_feats.append(tensor.feats[valid])  # (N_valid, C)
            del tensor, valid
        
        merged_coords = torch.cat(all_coords)  # (N, 4)
        merged_feats = torch.cat(all_feats)    # (N, C)
        del all_coords, all_feats              # 释放列表引用，让 chunk 碎片可回收
        
        # ────────────────────────────────────────────────────
        # 退化保护：halo 过滤后无有效点
        # ────────────────────────────────────────────────────
        if merged_coords.shape[0] == 0:
            logging.warning(
                "_merge_tensors: all chunks empty after halo filtering "
                "(degenerate latent) → returning 0-point SparseTensor"
            )
            return SparseTensor(merged_feats, merged_coords, scale=merged_scale)
        
        # ★ 按坐标规范排序，消除 chunk 边界对点顺序的影响。
        # 这保证了不同 chunk_size 产生相同的输出顺序，
        # 使 gradient checkpoint recompute 时 grad_output 与 recomputed output 对齐。
        D = merged_coords[:, 1:].max().item() + 1  # 坐标维度上界
        sort_key = (merged_coords[:, 0] * (D ** 3) +
                    merged_coords[:, 1] * (D ** 2) +
                    merged_coords[:, 2] * D +
                    merged_coords[:, 3])            # (N,)
        sort_idx = sort_key.argsort()               # (N,)
        del sort_key                                # 释放 sort_key（N,）int64

        # 分步排序：先排 coords（小），释放旧 coords，再排 feats（大），
        # 避免 old_feats + new_feats 同时存在时的峰值显存
        merged_coords = merged_coords[sort_idx]     # (N, 4)
        old_feats = merged_feats
        del merged_feats
        torch.cuda.empty_cache()                    # 回收旧 merged_feats + 碎片
        merged_feats = old_feats[sort_idx]          # (N, C)
        del old_feats, sort_idx
        
        return SparseTensor(merged_feats, merged_coords, scale=merged_scale)  # SparseTensor feats: (N, C)
    
    def apply(self, func: Callable[[Chunk], SparseTensor]) -> SparseTensor:
        """对每个 chunk 应用函数并合并结果（最常用的高层接口）"""
        for chunk in self.chunks():
            result = func(chunk)
            chunk.set_result(result)
        return self.merge()


class MemoryMonitor:
    """
    显存监控器，用于自动估算合适的 chunk_size。
    """
    def __init__(self, target_usage_ratio: float = 0.8, min_chunk_size: int = 32):
        self.target_usage_ratio = target_usage_ratio
        self.min_chunk_size = min_chunk_size
    
    def get_available_memory(self) -> int:
        """获取当前可用显存（字节）"""
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved()
        return int((total - reserved) * self.target_usage_ratio)
    
    def estimate_chunk_size(
        self, 
        num_points: int, 
        coord_range: int, 
        bytes_per_point: int = 4096
    ) -> int:
        """
        估算合适的 chunk_size。
        
        Args:
            num_points: 输入点数
            coord_range: 坐标范围（如 resolution）
            bytes_per_point: 每点显存消耗估计
        
        Returns:
            chunk_size: 建议的 chunk 大小，显存充足时返回 coord_range（即不分块）
        """
        available = self.get_available_memory()
        max_points = available // bytes_per_point
        if num_points <= max_points:
            return coord_range  # 显存充足，无需分块，返回完整坐标范围
        num_chunks = (num_points + max_points - 1) // max_points
        return max(coord_range // num_chunks, self.min_chunk_size)


__all__ = [
    'ChunkMeta',
    'Chunk', 
    'ChunkableSparseTensor',
    'MemoryMonitor',
]
