# TRELLIS.2 VAE Decoder Chunked Forward 方案

## 背景

TRELLIS.2 的 VAE Decoder 处理大规模稀疏张量时显存消耗高。当前实现一次性处理整个输入，峰值显存与输入规模成正比。Chunked Forward 方案通过空间分块处理，将峰值显存控制在固定范围内。

## 原理

将大的 SparseTensor 按空间坐标轴（x/y/z）切分成多个小块，每块单独 forward 后合并结果。

### 空间分块

SparseTensor 包含 `coords [N, 4]` 和 `feats [N, C]`。其中 `coords[:, 0]` 是 batch index，`coords[:, 1:4]` 是 x/y/z 坐标。按某轴（如 z）的坐标范围切分，每块只包含该范围内的点。处理完后合并各块输出。

**限制**：Chunked Forward **仅支持 batch_size=1**。

### Halo Region

稀疏卷积（3×3×3 kernel）需要访问邻居点。切分边界处的点会丢失另一块的邻居，导致计算错误。解决方案：切分时向外扩展 halo 区域，包含边界外的邻居点。处理完后丢弃 halo 部分的输出。

### Halo 大小计算

`halo_size = num_conv_layers × (kernel_size - 1) / 2`

对于 3×3×3 kernel，每个卷积层贡献 halo = 1。

### 分块层级选择

| 方案 | 分块位置 | halo 大小 | 复杂度 |
|------|---------|----------|--------|
| A. 外层分块 | 整个 Decoder 外 | 总层数累积（如 8） | 低 |
| B. 每层分块 | 每个 Block 内 | 1 | 高 |
| C. 按分辨率分块 | 每个分辨率层级 | 该层级层数（如 4） | 中 |

**采用方案 C**：在每个分辨率层级内做分块处理。

## 模型架构分析

### Decoder 结构

5 个分辨率层级，每层级包含若干 ConvNeXtBlock 和 1 个 Upsample：

- Level 0: 4 blocks + upsample (1024 ch)
- Level 1: 16 blocks + upsample (512 ch)  ← 卷积最多
- Level 2: 8 blocks + upsample (256 ch)
- Level 3: 4 blocks + upsample (128 ch)
- Level 4: 0 blocks (64 ch)

### Upsample Block 结构

`SparseResBlockC2S3d` 的执行顺序：

```
conv1 (原坐标系) → Channel2Spatial (坐标×2) → conv2 (2x坐标系)
```

关键点：conv1 在原坐标系执行，conv2 在 2x 坐标系执行。这意味着需要**两阶段分块处理**。

### Halo 计算

每层级的 halo 分两阶段：

| 阶段 | 操作 | 坐标系 | Halo 贡献 |
|------|------|--------|-----------|
| Stage 1 | ConvNeXt blocks + upsample.conv1 | 原始 | `num_blocks + 1` |
| Stage 2 | upsample.conv2 | 2x | 1 |

## 设计方案：ChunkableSparseTensor

### 核心理念

1. **类封装**：创建 `ChunkableSparseTensor` 类包装 SparseTensor，提供分块相关接口
2. **附加 SparseTensor 机制**：支持携带关联 SparseTensor（如 skip connection），随主 tensor 一起切分/合并
3. **Indexed Cache 机制**：支持按点索引切分 `_spatial_cache` 中的 tensor（如 subdivision）
4. **自动元信息管理**：切分时记录 scale、halo 等信息，合并时自动恢复
5. **显式数据流**：禁止通过对象属性传递状态，所有状态显式管理

### 数据结构

```python
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Iterator, Tuple
import torch

@dataclass
class ChunkMeta:
    """切分元信息"""
    start: int           # 原始坐标起点
    end: int             # 原始坐标终点
    actual_halo: int     # 实际 halo 偏移（边界处可能被截断）
    original_scale: tuple  # SparseTensor._scale
    valid_mask: torch.Tensor  # 有效区域掩码 [n_chunk_points]，True 表示非 halo
```

### ChunkableSparseTensor 类

```python
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
        indexed_cache_keys: List[str] = None
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
        if tensor.coords[:, 0].max() > 0:
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
    
    def chunks(self) -> Iterator['Chunk']:
        """生成切分后的 chunks"""
        if self._chunks is None:
            self._chunks = self._split()
        for chunk in self._chunks:
            yield chunk
    
    def _split(self) -> List['Chunk']:
        """执行切分"""
        coords = self._tensor.coords
        
        # 处理空 coords
        if coords.shape[0] == 0:
            return []
        
        max_coord = coords[:, self._axis].max().item() + 1
        
        chunks = []
        for start in range(0, max_coord, self._chunk_size):
            end = min(start + self._chunk_size, max_coord)
            halo_start = max(0, start - self._halo)
            halo_end = min(max_coord, end + self._halo)
            
            # halo 区域的点
            mask = (coords[:, self._axis] >= halo_start) & \
                   (coords[:, self._axis] < halo_end)
            
            # 计算有效区域掩码（非 halo 区域）
            valid_in_original = (coords[:, self._axis] >= start) & \
                                (coords[:, self._axis] < end)
            valid_mask = valid_in_original[mask]
            
            # 切分主 tensor 和 indexed cache
            chunk_tensor, chunk_indexed_cache = self._slice_with_cache(
                self._tensor, mask, halo_start
            )
            
            # 切分附加 SparseTensor
            chunk_attached = {}
            for name, t in self._attached.items():
                chunk_coords = t.coords[mask].clone()
                chunk_feats = t.feats[mask]
                chunk_coords[:, self._axis] -= halo_start
                chunk_attached[name] = SparseTensor(chunk_feats, chunk_coords, scale=t._scale)
            
            meta = ChunkMeta(
                start=start,
                end=end,
                actual_halo=start - halo_start,
                original_scale=self._tensor._scale,
                valid_mask=valid_mask
            )
            
            chunks.append(Chunk(chunk_tensor, chunk_attached, chunk_indexed_cache,
                               meta, self._axis, self._coord_scale))
        
        return chunks
    
    def _slice_with_cache(self, tensor: SparseTensor, mask: torch.Tensor,
                          offset: int) -> Tuple[SparseTensor, Dict[str, torch.Tensor]]:
        """切分 tensor 并提取 indexed cache"""
        # 切分 tensor
        chunk_coords = tensor.coords[mask].clone()
        chunk_feats = tensor.feats[mask]
        chunk_coords[:, self._axis] -= offset
        chunk_tensor = SparseTensor(chunk_feats, chunk_coords, scale=tensor._scale)
        
        # 切分 indexed cache
        chunk_indexed_cache = {}
        for key in self._indexed_cache_keys:
            cache = tensor.get_spatial_cache(key)
            if cache is not None and isinstance(cache, torch.Tensor):
                chunk_indexed_cache[key] = cache[mask]
        
        return chunk_tensor, chunk_indexed_cache
    
    def merge(self) -> SparseTensor:
        """合并所有 chunks 的结果"""
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
        
        return self._tensor
    
    def _merge_tensors(self, tensors: List[Tuple[SparseTensor, ChunkMeta]]) -> Optional[SparseTensor]:
        """合并多个 tensor，丢弃 halo 区域"""
        if not tensors:
            return None
        
        all_coords, all_feats = [], []
        merged_scale = None
        
        for tensor, meta in tensors:
            if merged_scale is None:
                merged_scale = tensor._scale
            
            # 计算有效区域边界
            local_start = meta.actual_halo * self._coord_scale
            local_end = (meta.actual_halo + meta.end - meta.start) * self._coord_scale
            
            valid = (tensor.coords[:, self._axis] >= local_start) & \
                    (tensor.coords[:, self._axis] < local_end)
            
            # 恢复全局坐标
            valid_coords = tensor.coords[valid].clone()
            valid_coords[:, self._axis] = \
                valid_coords[:, self._axis] - local_start + meta.start * self._coord_scale
            
            all_coords.append(valid_coords)
            all_feats.append(tensor.feats[valid])
        
        return SparseTensor(torch.cat(all_feats), torch.cat(all_coords), scale=merged_scale)
    
    def apply(self, func: Callable[['Chunk'], SparseTensor]) -> SparseTensor:
        """对每个 chunk 应用函数并合并结果（最常用的高层接口）"""
        for chunk in self.chunks():
            result = func(chunk)
            chunk.set_result(result)
        return self.merge()
```

### Chunk 类

```python
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
    
    def __init__(self, tensor: SparseTensor, attached: Dict[str, SparseTensor],
                 indexed_cache: Dict[str, torch.Tensor],
                 meta: ChunkMeta, axis: int, coord_scale: int):
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
    
    def get(self, name: str) -> Optional[SparseTensor]:
        """获取附加的 SparseTensor"""
        return self._attached.get(name)
    
    def get_indexed_cache(self, key: str) -> Optional[torch.Tensor]:
        """获取有效区域的 indexed cache（已过滤 halo）"""
        cache = self._indexed_cache.get(key)
        if cache is None:
            return None
        return cache[self._meta.valid_mask]
    
    def get_valid_feats(self, tensor: SparseTensor = None) -> torch.Tensor:
        """获取有效区域的 feats（已过滤 halo）"""
        t = tensor if tensor is not None else self._tensor
        return t.feats[self._meta.valid_mask]
    
    def set_result(self, tensor: SparseTensor) -> None:
        """设置主 tensor 的处理结果"""
        self._result = tensor
    
    def set_attached_result(self, name: str, tensor: SparseTensor) -> None:
        """设置附加 SparseTensor 的处理结果"""
        self._result_attached[name] = tensor
```

### Upsample 分阶段执行

```python
def execute_upsample_stage1(upsample_block, x: SparseTensor):
    """
    执行 SparseResBlockC2S3d 的第一阶段。
    
    Returns:
        output: conv1 + updown 后的结果（2x 坐标系）
        skip: updown 后的 x（2x 坐标系，用于 skip connection）
        subdiv: subdivision 预测（原坐标系）
    """
    if upsample_block.pred_subdiv:
        subdiv = upsample_block.to_subdiv(x)
    else:
        subdiv = None
    
    h = x.replace(upsample_block.norm1(x.feats))
    h = h.replace(F.silu(h.feats))
    h = upsample_block.conv1(h)
    
    subdiv_bin = subdiv.replace(subdiv.feats > 0) if subdiv else None
    h = upsample_block.updown(h, subdiv_bin)      # h 变成 2x 坐标
    skip = upsample_block.updown(x, subdiv_bin)   # x 也变成 2x 坐标
    
    return h, skip, subdiv


def execute_upsample_stage2(upsample_block, h: SparseTensor, 
                            skip: SparseTensor) -> SparseTensor:
    """执行 SparseResBlockC2S3d 的第二阶段，显式传入 skip tensor"""
    h = h.replace(upsample_block.norm2(h.feats))
    h = h.replace(F.silu(h.feats))
    h = upsample_block.conv2(h)
    return h + upsample_block.skip_connection(skip)
```

### 层级处理

```python
def process_level(
    h: SparseTensor,
    conv_blocks: nn.ModuleList,
    upsample_block: Optional[nn.Module],
    axis: int,
    chunk_size: int,
    collect_subdiv: bool = False
) -> Tuple[SparseTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    处理一个分辨率层级，采用两阶段分块策略。
    
    Returns:
        output: 处理后的 SparseTensor
        subdiv: 预测的 subdivision feats（训练时），shape [N, 8]
        subdiv_gt: ground truth subdivision（训练时），shape [N, 8]
    """
    has_upsample = upsample_block is not None
    halo_s1 = len(conv_blocks) + (1 if has_upsample else 0)
    
    # ======== Stage 1 ========
    # indexed_cache_keys 默认包含 'subdivision'，会自动按点切分
    chunked_s1 = ChunkableSparseTensor(
        h, axis=axis, chunk_size=chunk_size, 
        halo=halo_s1, coord_scale=2 if has_upsample else 1
    )
    
    subdiv_chunks = []
    subdiv_gt_chunks = []
    
    for chunk in chunked_s1.chunks():
        x = chunk.tensor
        
        # 获取有效区域的 subdiv_gt（已自动过滤 halo）
        if collect_subdiv:
            chunk_subdiv_gt = chunk.get_indexed_cache("subdivision")
            if chunk_subdiv_gt is not None:
                subdiv_gt_chunks.append(chunk_subdiv_gt)
        
        # ConvNeXt blocks
        for block in conv_blocks:
            x = block(x)
        
        if has_upsample:
            output, skip, subdiv = execute_upsample_stage1(upsample_block, x)
            chunk.set_result(output)
            chunk.set_attached_result("skip", skip)
            if subdiv is not None:
                # 使用 get_valid_feats 过滤 halo 区域的预测
                subdiv_chunks.append(chunk.get_valid_feats(subdiv))
        else:
            chunk.set_result(x)
    
    merged_s1 = chunked_s1.merge()
    merged_skip = chunked_s1.get_attached("skip")
    
    # 合并 subdivision 预测和 GT（已经过滤 halo，直接拼接）
    subdiv = torch.cat(subdiv_chunks, dim=0) if subdiv_chunks else None
    subdiv_gt = torch.cat(subdiv_gt_chunks, dim=0) if subdiv_gt_chunks else None
    
    # ======== Stage 2 ========
    if has_upsample and merged_skip is not None:
        chunked_s2 = ChunkableSparseTensor(
            merged_s1, axis=axis, chunk_size=chunk_size * 2, halo=1,
            indexed_cache_keys=[]  # Stage 2 不需要处理 indexed cache
        )
        chunked_s2.attach("skip", merged_skip)
        
        for chunk in chunked_s2.chunks():
            result = execute_upsample_stage2(
                upsample_block, chunk.tensor, chunk.get("skip")
            )
            chunk.set_result(result)
        
        final_output = chunked_s2.merge()
    else:
        final_output = merged_s1
    
    return final_output, subdiv, subdiv_gt
```

### Decoder Forward

```python
def forward(self, x, chunk_size=None, axis=3):
    """
    Chunked forward pass.
    
    Args:
        x: 输入 SparseTensor
        chunk_size: 基础 chunk 大小。None 表示不分块。
        axis: 切分轴 (1=x, 2=y, 3=z)
    """
    if chunk_size is None:
        return self._forward_original(x)
    
    h = self.from_latent(x)
    h = h.type(self.dtype)
    
    current_chunk_size = chunk_size
    collect_subdiv = self.training and self.pred_subdiv
    
    all_subs, all_subs_gt = [], []
    
    for i, level_blocks in enumerate(self.blocks):
        # 分离 conv blocks 和 upsample block
        # 最后一层没有 upsample，其他层的最后一个 block 是 upsample
        if i < len(self.blocks) - 1:
            conv_blocks = level_blocks[:-1]
            upsample_block = level_blocks[-1]
        else:
            conv_blocks = level_blocks
            upsample_block = None
        
        h, subdiv, subdiv_gt = process_level(
            h, conv_blocks, upsample_block, axis, current_chunk_size, collect_subdiv
        )
        
        if subdiv is not None:
            all_subs.append(subdiv)
        if subdiv_gt is not None:
            all_subs_gt.append(subdiv_gt)
        
        if upsample_block is not None:
            current_chunk_size *= 2
    
    h = h.type(x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.output_layer(h)
    
    if self.training and self.pred_subdiv:
        return h, all_subs_gt, all_subs
    return h
```

## 显存自适应

### MemoryMonitor

```python
class MemoryMonitor:
    """
    显存监控器，用于自动估算合适的 chunk_size。
    """
    def __init__(self, target_usage_ratio=0.8, min_chunk_size=32):
        self.target_usage_ratio = target_usage_ratio
        self.min_chunk_size = min_chunk_size
    
    def get_available_memory(self) -> int:
        """获取当前可用显存（字节）"""
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved()
        return int((total - reserved) * self.target_usage_ratio)
    
    def estimate_chunk_size(self, num_points, coord_range, bytes_per_point=4096):
        """
        估算合适的 chunk_size。
        
        Args:
            num_points: 输入点数
            coord_range: 坐标范围（如 resolution）
            bytes_per_point: 每点显存消耗估计
        
        Returns:
            chunk_size: 建议的 chunk 大小，None 表示无需分块
        """
        available = self.get_available_memory()
        max_points = available // bytes_per_point
        if num_points <= max_points:
            return None
        num_chunks = (num_points + max_points - 1) // max_points
        return max(coord_range // num_chunks, self.min_chunk_size)
```

### 调用示例

```python
# 在 decode 调用前
memory_monitor = MemoryMonitor()
chunk_size = memory_monitor.estimate_chunk_size(num_points, resolution)
h = decoder.forward(x, chunk_size=chunk_size)
```

## 代码改动计划

### 新建文件

| 文件 | 说明 |
|------|------|
| `trellis2/modules/sparse/spatial/chunked.py` | ChunkableSparseTensor, Chunk, ChunkMeta, MemoryMonitor |

### 修改文件

| 文件 | 改动 |
|------|------|
| `trellis2/modules/sparse/spatial/__init__.py` | 导出 chunked 模块 |
| `trellis2/models/sc_vaes/sparse_unet_vae.py` | 添加 chunked forward 方法 |

### 实现顺序

1. `chunked.py`: 实现 ChunkableSparseTensor 类
2. 单元测试：验证切分/合并的正确性（coords 和 feats 完整性）
3. 集成到 Decoder：添加 `forward_chunked` 方法
4. 端到端测试：对比分块与非分块输出一致性

## 注意事项

1. **使用限制**：
   - Chunked Forward **仅支持 batch_size=1**
   - 训练和推理均可使用
   - 传 `chunk_size=None` 时使用原始 forward（不分块）

2. **Skip Connection 处理**：
   - Stage 1 通过 `chunk.set_attached_result("skip", skip)` 保存
   - Stage 2 通过 `chunk.get("skip")` 获取
   - 合并后通过 `chunked.get_attached("skip")` 获取
   - **禁止**使用 `tensor._skip` 等对象属性传递

3. **SparseTensor 属性**：
   - `_scale`：通过 `ChunkMeta.original_scale` 保留，合并时正确传递
   - `_spatial_cache`：除 `indexed_cache_keys` 指定的 key 外，切分后失效，让后续操作惰性重算

4. **Indexed Cache 处理**（如 `subdivision`）：
   - 通过 `indexed_cache_keys` 参数指定需要按点索引切分的缓存 key
   - 默认包含 `['subdivision']`
   - 切分时：`cache[mask]` 按相同 mask 切分
   - 在 chunk 内通过 `chunk.get_indexed_cache(key)` 获取（**已自动过滤 halo**）
   - **需用户手动收集并拼接**（不自动合并）

5. **有效区域过滤**：
   - 每个 chunk 内部自动维护 `valid_mask`，标记非 halo 区域的点
   - `chunk.get_indexed_cache(key)` 自动返回有效区域的缓存
   - `chunk.get_valid_feats(tensor)` 返回有效区域的 feats

6. **边界处理**：
   - 空 coords 时 `_split()` 返回空列表
   - 第一个 chunk 的 `actual_halo` 可能为 0（左边界无法扩展）
   - 最后一个 chunk 的 `end` 可能小于 `start + chunk_size`
   - 合并时根据 `actual_halo` 计算有效区域

7. **Gradient Checkpointing**：
   - 可在 `apply()` 内添加 checkpoint 支持
   - 对每个 chunk 的处理使用 `torch.utils.checkpoint`
   - 仅在 `requires_grad=True` 时启用

## 附录：数据流图

```
原始输入 h (携带 subdivision 缓存)
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 1: ChunkableSparseTensor(h, coord_scale=2)          │
│          indexed_cache_keys=['subdivision'] (默认)        │
│                                                           │
│   切分 ──► [chunk_0, chunk_1, chunk_2, ...]              │
│              │         │         │                        │
│              │ 每个 chunk 包含:                           │
│              │ - tensor (切分后的 SparseTensor)           │
│              │ - indexed_cache["subdivision"] (切分后)    │
│              ▼         ▼         ▼                        │
│   收集 ◄── get_indexed_cache("subdivision")              │
│              │  (自动过滤 halo 区域)                      │
│              │         │         │                        │
│              ▼         ▼         ▼                        │
│          ConvNeXt   ConvNeXt   ConvNeXt                   │
│              │         │         │                        │
│              ▼         ▼         ▼                        │
│          Upsample  Upsample  Upsample                     │
│          Stage1    Stage1    Stage1                       │
│           │ │       │ │       │ │                         │
│           │ └─skip  │ └─skip  │ └─skip                    │
│           │         │         │                           │
│   收集 ◄── get_valid_feats(subdiv) ──────────────────────│
│           ▼         ▼         ▼                           │
│   合并 ◄── [out_0,   out_1,   out_2, ...]                │
│              + 附加: [skip_0, skip_1, skip_2, ...]        │
└───────────────────────────────────────────────────────────┘
    │
    ▼ merged_s1, merged_skip
    │ subdiv = cat([valid_subdiv_0, valid_subdiv_1, ...])
    │ subdiv_gt = cat([valid_gt_0, valid_gt_1, ...])
    │
┌───────────────────────────────────────────────────────────┐
│ Stage 2: ChunkableSparseTensor(merged_s1)                 │
│          .attach("skip", merged_skip)                     │
│          indexed_cache_keys=[] (不需要处理缓存)           │
│                                                           │
│   切分 ──► [(out_0, skip_0), (out_1, skip_1), ...]       │
│                   │              │                        │
│                   ▼              ▼                        │
│              Upsample       Upsample                      │
│              Stage2         Stage2                        │
│                   │              │                        │
│                   ▼              ▼                        │
│   合并 ◄── [result_0, result_1, ...]                     │
└───────────────────────────────────────────────────────────┘
    │
    ▼
最终输出, subdiv, subdiv_gt
```
