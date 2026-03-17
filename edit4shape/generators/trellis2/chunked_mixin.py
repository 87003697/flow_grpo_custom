"""
Chunked Forward Mixin for SparseVAE Decoder.

提供逐层自适应 chunked forward 能力，通过 Mixin 方式注入到 SparseVAE 类中，
实现零侵入性的显存优化。

每层根据实时显存余量、当前点数和通道数自动估算 chunk_size，无需外部手动配置。

返回格式与原始 SparseUnetVaeDecoder.forward() 完全兼容：
    - 训练时 (pred_subdiv=True): (h, subs_gt, subs)
    - 推理时 (return_subs=True): (h, subs)
    - 推理时 (return_subs=False): h

方法调用层次：
    forward_chunked              公共入口：遍历层 + output_layer
      └→ _process_level          单层入口：估算 chunk_size + 可选 checkpoint 包裹
          └→ _run_chunked_stages 纯计算：Stage 1/2 分块循环（不做任何估算）

★ gradient checkpoint 兼容性：
    chunk_size 估算使用 torch.cuda.memory_reserved()（非确定性函数），
    因此必须在 checkpoint 边界外完成。_process_level 负责估算并固化 chunk_size，
    _run_chunked_stages 只接收固化后的值，保证 forward 和 recompute 一致。

Usage:
    # 方式一：继承组合（推荐）
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseVAE
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    
    class ChunkedSparseVAE(ChunkedDecoderMixin, SparseVAE):
        pass
    
    decoder = ChunkedSparseVAE(...)
    output = decoder.forward_chunked(x)
    
    # 方式二：动态注入（不修改类定义，推荐用于预训练模型）
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    decoder = load_pretrained_decoder()
    ChunkedDecoderMixin.inject_to(decoder)
    output = decoder.forward_chunked(x)
    
    # 推理时获取 subdivision（用于后续纹理解码）
    h, subs = decoder.forward_chunked(x, return_subs=True)
"""
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

# 使用绝对导入（适配移动到 edit4shape/generators/trellis2/）
from trellis2.modules.sparse import SparseTensor
from .chunked import ChunkableSparseTensor


class ChunkedDecoderMixin:
    """
    为 SparseVAE Decoder 提供逐层自适应 chunked forward 能力的 Mixin。
    
    每层进入前实时查询 GPU 显存，根据当前层的点数和通道数估算 chunk_size，
    比外部一次性估算更准确（因为上采样后点数/通道数/显存余量都会变化）。
    
    要求宿主类具有以下属性：
    - self.blocks: nn.ModuleList
    - self.from_latent: nn.Module
    - self.output_layer: nn.Module
    - self.dtype: torch.dtype
    - self.pred_subdiv: bool
    - self.training: bool
    """
    
    # =====================================================================
    # 坐标对齐工具
    # =====================================================================
    
    @staticmethod
    def _align_guide_sub(guide_sub: SparseTensor, h: SparseTensor) -> SparseTensor:
        """
        将 guide_sub 的 feats 按 h 的坐标顺序重排。

        支持两种情况：
        1. N_h == N_g（原始 bijection 路径）：两者坐标集合相同但顺序不同。
        2. N_h != N_g（size mismatch 路径）：guide_sub 可能含有 h 中不存在的
           额外坐标（来自 _build_subs_from_coords 的 merge），也可能缺少 h 中
           新增的坐标（来自前一层额外展开的 octant）。
           此时使用 searchsorted 做精确坐标匹配：
           - h 中有对应 guide 坐标的点：取 guide feats
           - h 中无对应 guide 坐标的点：feats 置零（不展开）

        Args:
            guide_sub: subdivision SparseTensor（可能比 h 多或少坐标）
            h: 当前层输入 SparseTensor

        Returns:
            重排后的 guide_sub，coords 与 h 完全一致
        """
        c_h = h.coords      # (N_h, 4)
        c_g = guide_sub.coords  # (N_g, 4)

        # 快速路径：坐标已经一致
        if torch.equal(c_h, c_g):
            return guide_sub

        N_h = c_h.shape[0]
        N_g = c_g.shape[0]
        D = max(c_h[:, 1:].max().item(), c_g[:, 1:].max().item()) + 1
        key_h = c_h[:, 0] * (D ** 3) + c_h[:, 1] * (D ** 2) + c_h[:, 2] * D + c_h[:, 3]
        key_g = c_g[:, 0] * (D ** 3) + c_g[:, 1] * (D ** 2) + c_g[:, 2] * D + c_g[:, 3]

        if N_h == N_g:
            # ---- 快速 bijection 路径（原有逻辑） ----
            h_sort = key_h.argsort()
            g_sort = key_g.argsort()
            h_inv = h_sort.argsort()
            reorder = g_sort[h_inv]
            return SparseTensor(
                guide_sub.feats[reorder],
                c_h.clone(),
                scale=guide_sub._scale,
            )

        # ---- 通用路径：searchsorted 精确匹配 ----
        g_sorted_keys, g_argsort = key_g.sort()
        pos = torch.searchsorted(g_sorted_keys, key_h)
        pos = pos.clamp(0, N_g - 1)
        matched = (g_sorted_keys[pos] == key_h)

        result_feats = torch.zeros(
            N_h, guide_sub.feats.shape[1],
            device=guide_sub.feats.device, dtype=guide_sub.feats.dtype)
        result_feats[matched] = guide_sub.feats[g_argsort[pos[matched]]]

        return SparseTensor(
            result_feats,
            c_h.clone(),
            scale=guide_sub._scale,
        )
    
    # =====================================================================
    # 动态注入
    # =====================================================================
    
    @classmethod
    def inject_to(cls, instance) -> None:
        """
        动态注入 chunked forward 方法到已有实例。
        
        Args:
            instance: SparseVAE 实例
            
        Usage:
            decoder = load_pretrained_decoder()
            ChunkedDecoderMixin.inject_to(decoder)
            output = decoder.forward_chunked(x)
        """
        from types import MethodType
        
        instance.forward_chunked = MethodType(cls.forward_chunked, instance)
        instance._process_level = MethodType(cls._process_level, instance)
        instance._run_chunked_stages = MethodType(cls._run_chunked_stages, instance)
        instance._estimate_chunk_size = cls._estimate_chunk_size       # staticmethod
        instance._align_guide_sub = cls._align_guide_sub               # staticmethod
        instance._execute_upsample_stage1 = cls._execute_upsample_stage1  # staticmethod
        instance._execute_upsample_stage2 = cls._execute_upsample_stage2  # staticmethod
    
    # =====================================================================
    # 公共入口
    # =====================================================================
    
    def forward_chunked(
        self, 
        x: SparseTensor, 
        axis: int = 3,
        return_subs: bool = False,
        guide_subs: Optional[List[SparseTensor]] = None,
        chunk_size_override: Optional[int] = None,
        use_checkpoint: bool = False,
    ) -> SparseTensor:
        """
        逐层自适应 Chunked forward pass。
        
        每层根据实时显存余量、当前点数和通道数自动估算 chunk_size，
        无需外部手动配置。仅支持 batch_size=1。
        
        同时支持两种 decoder 模式：
        - pred_subdiv=True (Shape Decoder): 自己预测 subdivision
        - pred_subdiv=False (Tex Decoder): 使用外部传入的 guide_subs 指导上采样
        
        Args:
            x: 输入 SparseTensor
            axis: 切分轴 (1=x, 2=y, 3=z)
            return_subs: 推理时是否返回 subdivision 预测（用于后续纹理解码）
            guide_subs: 外部提供的 subdivision 列表（Tex Decoder 使用），
                每层一个 SparseTensor，共 len(self.blocks)-1 个。
                为 None 时由 decoder 自己预测（Shape Decoder）。
            chunk_size_override: 强制指定 chunk_size，用于调试/测试。
                为 None 时使用自动估算。
            use_checkpoint: 是否启用 level-level gradient checkpoint。
                启用时每层的中间激活在 forward 后释放，backward 时按需重算，
                峰值显存从 5 层同时驻留降低到 1 层。
            
        Returns:
            与原始 forward() 相同的返回格式：
            - 训练时 (pred_subdiv=True): (h, subs_gt, subs)
            - 推理时 (return_subs=True): (h, subs)
            - 推理时 (return_subs=False): h
        
        退化保护:
            退化 latent 可能导致中间某层 merge 后无有效点。此时 h 为
            0 点的空 SparseTensor（``h.feats.shape[0] == 0``），会正常
            通过 output_layer 返回。上层调用方（如 forward.py）通过
            检查 ``h.feats.shape[0] == 0`` 触发 StageSkipError，
            无需本方法做额外处理。
        """
        assert return_subs == False or self.pred_subdiv == True, \
            "Only decoders with pred_subdiv=True can be used with return_subs"
        # guide_subs 传入时不再要求 pred_subdiv=False：
        # _execute_upsample_stage1 已改为 guide_sub 优先，
        # 因此无论 pred_subdiv 值如何，guide_sub 都会被正确使用。
        # 这避免了 pred_subdiv_override context manager 在有梯度 forward
        # 中使用导致 backward 重算时 pred_subdiv 状态不一致的问题。
        
        h = self.from_latent(x)  # SparseTensor feats: (N, C_latent)
        h = h.type(self.dtype)   # SparseTensor feats: (N, C_latent)
        
        collect_subdiv = (self.training and self.pred_subdiv) or return_subs
        all_subs, all_subs_gt = [], []
        
        for i, level_blocks in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                conv_blocks = level_blocks[:-1]
                upsample_block = level_blocks[-1]
            else:
                conv_blocks = level_blocks
                upsample_block = None
            
            # 获取当前层对应的 guide_sub（仅有 upsample 的层需要）
            guide_sub_i = None
            if guide_subs is not None and i < len(self.blocks) - 1:
                guide_sub_i = self._align_guide_sub(guide_subs[i], h)
            
            h, subdiv, subdiv_gt = self._process_level(
                h, conv_blocks, upsample_block, axis,
                chunk_size_override=chunk_size_override,
                collect_subdiv=collect_subdiv,
                use_checkpoint=use_checkpoint,
                level_idx=i,
                guide_sub=guide_sub_i,
            )  # h: SparseTensor feats (N, C); subdiv: SparseTensor or None; subdiv_gt: Tensor or None
            
            if subdiv is not None:
                all_subs.append(subdiv)
            if subdiv_gt is not None:
                all_subs_gt.append(subdiv_gt)
        
        # ────────────────────────────────────────────────────
        # 退化保护：0 点空张量的通道数可能不匹配 output_layer
        # （conv/upsample 未实际执行，通道映射缺失），直接返回。
        # 上层通过 h.feats.shape[0]==0 触发 StageSkipError。
        # ────────────────────────────────────────────────────
        if h.feats.shape[0] == 0:
            if self.training and self.pred_subdiv:
                return h, all_subs_gt, all_subs
            elif return_subs:
                return h, all_subs
            else:
                return h
        
        # output_layer + layer_norm：当 use_checkpoint 时也放进 checkpoint，
        # 释放保存的输入张量（15.5M × 64 × 2 ≈ 2 GB × 2 = ~4 GB）
        if use_checkpoint:
            # ★ 用 _h_snap 快照打断闭包循环引用。
            #   如果直接在 _output_fn 中引用变量 h，h = h.replace(ckpt(...)) 赋值后
            #   闭包看到的是 NEW h，而 NEW h.data['feats'].grad_fn → CheckpointBackward
            #   → ctx → _output_fn → 闭包 → h → NEW h，形成穿越 C++ grad_fn 的循环
            #   引用，Python gc 无法打破，导致整条 checkpoint 链 ~5 GiB 永远无法释放。
            _h_snap = h  # 快照 OLD h，_output_fn 闭包捕获 _h_snap 而非 h
            def _output_fn(feats):
                feats = feats.to(x.dtype)                              # (N, C)
                feats = F.layer_norm(feats, feats.shape[-1:])          # (N, C)
                return self.output_layer(_h_snap.replace(feats)).feats  # (N, C_out)
            h = h.replace(ckpt(_output_fn, h.feats, use_reentrant=False))  # SparseTensor feats: (N, C_out)
            # 注意：不能 del _h_snap！backward recompute 时 _output_fn 闭包仍需访问它。
            # _h_snap 会随 _output_fn 闭包的生命周期自然释放。
        else:
            h = h.type(x.dtype)  # SparseTensor feats: (N, C_latent)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))  # SparseTensor feats: (N, C_latent)
            h = self.output_layer(h)  # SparseTensor feats: (N, C_out)
        
        # 返回格式与原始 forward() 完全兼容
        if self.training and self.pred_subdiv:
            return h, all_subs_gt, all_subs
        elif return_subs:
            return h, all_subs
        else:
            return h
    
    # =====================================================================
    # 单层入口：估算 chunk_size + 可选 checkpoint
    # =====================================================================
    
    def _process_level(
        self,
        h: SparseTensor,
        conv_blocks: list,
        upsample_block,
        axis: int,
        *,
        chunk_size_override: Optional[int],
        collect_subdiv: bool,
        use_checkpoint: bool,
        level_idx: int,
        guide_sub: Optional[SparseTensor] = None,
    ) -> Tuple[SparseTensor, Optional[SparseTensor], Optional[torch.Tensor]]:
        """
        处理单个分辨率层级。
        
        职责：
        1. 估算 chunk_size（★ 在 checkpoint 边界外，保证 forward/recompute 一致性）
        2. 可选地用 gradient checkpoint 包裹实际计算
        
        Args:
            h: 当前层输入 SparseTensor
            conv_blocks: SparseConvNeXtBlock3d 列表
            upsample_block: SparseResBlockC2S3d 或 None（最后一层无上采样）
            axis: 切分轴 (1=x, 2=y, 3=z)
            chunk_size_override: 强制 chunk_size（None 时自动估算）
            collect_subdiv: 是否收集 subdivision 预测和 GT
            use_checkpoint: 是否启用 level-level gradient checkpoint
            level_idx: 层序号（仅用于日志）
            guide_sub: 外部提供的 subdivision（Tex Decoder 使用），
                SparseTensor 或 None。与 h 有相同的坐标空间。
            
        Returns:
            h_out: 处理后的 SparseTensor（上采样层坐标 ×2）
            subdiv: subdivision 预测 SparseTensor 或 None
            subdiv_gt: subdivision GT Tensor 或 None
        """
        # ---- 1. chunk_size 估算（checkpoint 边界外，确保确定性） ----
        # ★ _estimate_chunk_size 依赖 torch.cuda.memory_reserved()（非确定性函数），
        #   必须在 checkpoint 外调用，否则 forward/recompute 的 chunk_size 不一致，
        #   导致中间张量 shape 不同，触发 CheckpointError。
        if chunk_size_override is not None:
            chunk_size = chunk_size_override
        else:
            chunk_size = self._estimate_chunk_size(h, axis)
        
        # logging.info(
        #     f"[Decoder L{level_idx}] coord_range={h.coords[:, axis].max().item() + 1}, "
        #     f"points={h.coords.shape[0]}, ch={h.feats.shape[1]}, chunk_size={chunk_size}"
        # )
        
        # ---- 2. 不使用 checkpoint：直接计算 ----
        if not use_checkpoint:
            return self._run_chunked_stages(
                h, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv,
                guide_sub=guide_sub,
            )
        
        # ---- 3. 使用 checkpoint：包裹 _run_chunked_stages ----
        # 实现要点：
        # - h.feats（需要梯度追踪的标准 Tensor）和 h（SparseTensor，含
        #   coords/scale/spatial_cache 元数据）分别作为 ckpt 的 args 传入，
        #   保证被 tuple 捕获，不受外层循环变量覆盖影响。
        # - ckpt 函数只返回 h_out.feats（标准 Tensor），满足
        #   use_reentrant=False 的输出约束。
        # - subdiv / subdiv_gt 等非梯度输出通过 _captured dict 侧信道带出；
        #   每次调用创建独立的 dict，各层互不干扰。
        # - guide_sub（SparseTensor 或 None）作为 ckpt 的 *args 传入，
        #   确保 forward/recompute 使用相同的 guide_sub。
        _captured = {}
        
        def _ckpt_fn(h_feats, h_sparse, conv_blocks, upsample_block,
                     axis, chunk_size, collect_subdiv, guide_sub):
            h_rebuilt = h_sparse.replace(h_feats)
            h_out, subdiv, subdiv_gt = self._run_chunked_stages(
                h_rebuilt, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv,
                guide_sub=guide_sub,
            )
            _captured['h_out'] = h_out
            _captured['subdiv'] = subdiv
            _captured['subdiv_gt'] = subdiv_gt
            return h_out.feats  # (N_out, C_out) — 唯一需要梯度追踪的输出
        
        # 所有会随循环迭代变化的变量都走 ckpt 的 *args（被 tuple 捕获）
        h_out_feats = ckpt(
            _ckpt_fn,
            h.feats, h, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv,
            guide_sub,
            use_reentrant=False,
        )  # (N_out, C_out)
        
        # 用 forward 时捕获的 SparseTensor 元数据 + checkpoint 返回的 feats 重建输出
        h_out_sp = _captured['h_out']
        subdiv = _captured['subdiv']
        subdiv_gt = _captured['subdiv_gt']
        # ★ 立即清空 _captured，断开闭包 → forward-pass SparseTensor 的引用。
        #   _ckpt_fn 被 checkpoint ctx 持有，backward recompute 时会重新填充 _captured，
        #   但此时 forward-pass 的 SparseTensor 已不再被 _captured 引用，可以正常被
        #   gc 回收。不清空的话，_captured 会同时持有 forward 和 recompute 两份。
        _captured.clear()
        h_out = h_out_sp.replace(h_out_feats)
        del h_out_sp  # 释放 forward-pass SparseTensor
        return h_out, subdiv, subdiv_gt
    
    # =====================================================================
    # 显存自适应 chunk_size 估算
    # =====================================================================
    
    @staticmethod
    def _estimate_chunk_size(
        h: SparseTensor,
        axis: int,
        target_ratio: float = 0.5,
        bytes_per_element: int = 256,
        min_chunk: int = 32,
    ) -> int:
        """
        根据当前显存余量和 tensor 状态，估算本层的 chunk_size。
        
        核心公式:
            cost_per_point = channels × bytes_per_element
            max_points = available_mem / cost_per_point
            chunk_size = coord_range × (max_points / num_points)
        
        Args:
            h: 当前层的 SparseTensor
            axis: 切分轴
            target_ratio: 可用显存使用比例。设为 0.5 以留足 merge 阶段的
                2× 峰值开销（chunk 结果 + 合并后的新 tensor 同时存在）
            bytes_per_element: 每通道每点的显存消耗估算（含激活值、梯度、中间变量）
                fp16 训练 + MLP 4× 膨胀 + 稀疏卷积 gather/scatter + 梯度图
                经验值 ≈ 256 bytes
            min_chunk: 最小 chunk_size 下限
            
        Returns:
            chunk_size: 坐标空间的分块大小，显存充足时返回 coord_range（不分块）
        """
        # ── 退化保护：0 点输入 ──
        num_points = h.coords.shape[0]
        if num_points == 0:
            return min_chunk  # 无意义值，空 tensor 不会产生任何 chunk
        
        device = h.feats.device
        total = torch.cuda.get_device_properties(device).total_memory  # bytes
        reserved = torch.cuda.memory_reserved(device)  # bytes
        available = int((total - reserved) * target_ratio)  # bytes
        
        channels = h.feats.shape[1]
        coord_range = h.coords[:, axis].max().item() + 1
        
        cost_per_point = channels * bytes_per_element  # bytes/point
        max_points = max(available // cost_per_point, 1)
        
        if num_points <= max_points:
            return coord_range  # 显存充足，不分块
        
        # chunk_size ∝ (max_points / num_points) × coord_range
        return max(coord_range * max_points // num_points, min_chunk)
    
    # =====================================================================
    # 纯计算：两阶段分块处理
    # =====================================================================
    
    def _run_chunked_stages(
        self,
        h: SparseTensor,
        conv_blocks: list,
        upsample_block,
        axis: int,
        chunk_size: int,
        collect_subdiv: bool,
        guide_sub: Optional[SparseTensor] = None,
    ) -> Tuple[SparseTensor, Optional[SparseTensor], Optional[torch.Tensor]]:
        """
        处理一个分辨率层级的纯计算逻辑，采用两阶段分块策略。
        
        Stage 1: ConvNeXt blocks + upsample.conv1 + updown (原坐标系 → 2x坐标系)
        Stage 2: upsample.conv2 + skip_connection (2x坐标系内)
        
        ★ 本方法不做 chunk_size 估算，直接使用传入的 chunk_size。
          当被 gradient checkpoint 包裹时，这保证了 forward/recompute 的确定性。
        
        Args:
            h: 输入 SparseTensor
            conv_blocks: ConvNeXt blocks 列表
            upsample_block: Upsample block（最后一层为 None）
            axis: 切分轴
            chunk_size: 已固化的 chunk 大小（由 _process_level 在 ckpt 外估算）
            collect_subdiv: 是否收集 subdivision 预测
            guide_sub: 外部提供的 subdivision（Tex Decoder 使用），
                SparseTensor 或 None。随 h 一起按空间轴切分。
            
        Returns:
            output: 处理后的 SparseTensor（退化时为 0 点空张量）
            subdiv: subdivision 预测（SparseTensor），None 如果不收集
            subdiv_gt: subdivision ground truth（Tensor），None 如果不收集
        """
        has_upsample = upsample_block is not None
        halo_s1 = len(conv_blocks) + (1 if has_upsample else 0)
        
        # ======== Stage 1: conv + upsample_stage1 ========
        chunked_s1 = ChunkableSparseTensor(
            h, axis=axis, chunk_size=chunk_size, 
            halo=halo_s1, coord_scale=2 if has_upsample else 1
        )
        
        # 如果有外部 guide_sub，附加到 ChunkableSparseTensor 随主 tensor 自动切分
        if guide_sub is not None:
            chunked_s1.attach("guide_sub", guide_sub)
        
        subdiv_chunks = []
        subdiv_gt_chunks = []
        
        for chunk in chunked_s1.chunks():
            x = chunk.tensor  # SparseTensor feats: (N_chunk, C)
            
            # 获取有效区域的 subdiv_gt（已自动过滤 halo）
            if collect_subdiv:
                chunk_subdiv_gt = chunk.get_indexed_cache("subdivision")  # (N_chunk, 3) or None
                if chunk_subdiv_gt is not None:
                    subdiv_gt_chunks.append(chunk_subdiv_gt)
            
            # ConvNeXt blocks
            for block in conv_blocks:
                x = block(x)  # SparseTensor feats: (N_chunk, C)
            
            if has_upsample:
                # 获取当前 chunk 对应的 guide_sub 切片
                chunk_guide_sub = chunk.get("guide_sub")  # SparseTensor or None
                output, skip, subdiv = self._execute_upsample_stage1(
                    upsample_block, x, guide_sub=chunk_guide_sub
                )
                chunk.set_result(output)
                chunk.set_attached_result("skip", skip)
                if subdiv is not None:
                    # 使用 get_valid_feats 过滤 halo 区域的预测
                    subdiv_chunks.append(chunk.get_valid_feats(subdiv))  # (N_valid, 3)
            else:
                chunk.set_result(x)
        
        torch.cuda.empty_cache()  # 释放 chunk 处理中的碎片显存，缓解 merge 阶段 OOM
        merged_s1 = chunked_s1.merge()  # SparseTensor feats: (N, C)  ★ 退化时为 0 点空张量
        merged_skip = chunked_s1.get_attached("skip")  # SparseTensor feats: (N, C) or None
        
        # 合并 subdivision 预测和 GT（已经过滤 halo，直接拼接）
        subdiv = SparseTensor(
            torch.cat(subdiv_chunks, dim=0),  # (N, 3)
            h.coords.clone(),  # (N, 4) 使用原始坐标
            scale=h._scale
        ) if subdiv_chunks else None
        subdiv_gt = torch.cat(subdiv_gt_chunks, dim=0) if subdiv_gt_chunks else None  # (N, 3) or None
        
        # ======== Stage 2: upsample_stage2 ========
        if has_upsample and merged_skip is not None:
            # upsample 后坐标 ×2，chunk_size 等比放大（确定性公式，不依赖 runtime 显存）
            s2_chunk = chunk_size * 2
            chunked_s2 = ChunkableSparseTensor(
                merged_s1, axis=axis, chunk_size=s2_chunk, halo=1,
                indexed_cache_keys=[]  # Stage 2 不需要处理 indexed cache
            )
            chunked_s2.attach("skip", merged_skip)
            
            for chunk in chunked_s2.chunks():
                result = self._execute_upsample_stage2(
                    upsample_block, chunk.tensor, chunk.get("skip")
                )  # SparseTensor feats: (N_chunk, C)
                chunk.set_result(result)
            
            torch.cuda.empty_cache()  # 释放 chunk 处理中的碎片显存，缓解 merge 阶段 OOM
            final_output = chunked_s2.merge()  # SparseTensor feats: (N, C)  ★ 退化时为 0 点空张量
        else:
            final_output = merged_s1
        
        return final_output, subdiv, subdiv_gt
    
    # =====================================================================
    # Upsample 子步骤（静态方法）
    # =====================================================================
    
    @staticmethod
    def _execute_upsample_stage1(
        upsample_block, 
        x: SparseTensor,
        guide_sub: Optional[SparseTensor] = None,
    ) -> Tuple[SparseTensor, SparseTensor, Optional[SparseTensor]]:
        """
        执行 SparseResBlockC2S3d 的第一阶段。
        
        执行顺序：
        1. 获取 subdivision（自己预测 或 使用外部 guide_sub）
        2. norm1 + silu + conv1
        3. updown（坐标 ×2）
        
        Args:
            upsample_block: SparseResBlockC2S3d 实例
            x: 输入 SparseTensor
            guide_sub: 外部提供的 subdivision（Tex Decoder 使用），
                SparseTensor 或 None。pred_subdiv=False 时使用。
            
        Returns:
            output: conv1 + updown 后的结果（2x 坐标系）
            skip: updown 后的 x（2x 坐标系，用于 skip connection）
            subdiv: subdivision 预测（原坐标系），None 如果不预测且无 guide_sub
        """
        # 获取 subdivision：外部 guide_sub 优先（避免 pred_subdiv_override context
        # manager 退出后 backward 重计算使用错误的 self-predict 路径），
        # 否则自己预测（Shape Decoder）。
        if guide_sub is not None:
            subdiv = guide_sub  # SparseTensor feats: (N, 8)，来自 Shape Decoder 或 _build_subs_from_coords
        elif upsample_block.pred_subdiv:
            subdiv = upsample_block.to_subdiv(x)  # SparseTensor feats: (N, 8)
        else:
            subdiv = None
        
        # norm1 + silu + conv1
        h = x.replace(upsample_block.norm1(x.feats))  # SparseTensor feats: (N, C)
        h = h.replace(F.silu(h.feats))  # SparseTensor feats: (N, C)
        h = upsample_block.conv1(h)  # SparseTensor feats: (N, C)
        
        # updown（坐标 ×2）
        subdiv_bin = subdiv.replace(subdiv.feats > 0) if subdiv else None  # SparseTensor feats: (N, 3) or None
        h = upsample_block.updown(h, subdiv_bin)  # SparseTensor feats: (N, C) 2x 坐标
        skip = upsample_block.updown(x, subdiv_bin)  # SparseTensor feats: (N, C) 2x 坐标
        
        return h, skip, subdiv
    
    @staticmethod
    def _execute_upsample_stage2(
        upsample_block, 
        h: SparseTensor, 
        skip: SparseTensor
    ) -> SparseTensor:
        """
        执行 SparseResBlockC2S3d 的第二阶段。
        
        执行顺序：
        1. norm2 + silu + conv2
        2. skip_connection + residual
        
        Args:
            upsample_block: SparseResBlockC2S3d 实例
            h: Stage 1 输出（2x 坐标系）
            skip: Stage 1 的 skip tensor（2x 坐标系）
            
        Returns:
            output: 完整 upsample 后的结果
        """
        h = h.replace(upsample_block.norm2(h.feats))  # SparseTensor feats: (N, C)
        h = h.replace(F.silu(h.feats))  # SparseTensor feats: (N, C)
        h = upsample_block.conv2(h)  # SparseTensor feats: (N, C)
        return h + upsample_block.skip_connection(skip)  # SparseTensor feats: (N, C)


__all__ = ['ChunkedDecoderMixin']
