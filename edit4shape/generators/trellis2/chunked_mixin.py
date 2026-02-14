"""
Chunked Forward Mixin for SparseVAE Decoder.

提供逐层自适应 chunked forward 能力，通过 Mixin 方式注入到 SparseVAE 类中，
实现零侵入性的显存优化。

每层根据实时显存余量、当前点数和通道数自动估算 chunk_size，无需外部手动配置。

返回格式与原始 SparseUnetVaeDecoder.forward() 完全兼容：
    - 训练时 (pred_subdiv=True): (h, subs_gt, subs)
    - 推理时 (return_subs=True): (h, subs)
    - 推理时 (return_subs=False): h

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
import logging
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
    
    # =========== 类方法：动态注入 ===========
    
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
        
        # 注入所有需要的方法
        instance.forward_chunked = MethodType(cls.forward_chunked, instance)
        instance._process_level_chunked = MethodType(cls._process_level_chunked, instance)
        instance._process_level_chunked_ckpt = MethodType(cls._process_level_chunked_ckpt, instance)
        instance._estimate_level_chunk_size = cls._estimate_level_chunk_size  # 静态方法
        instance._execute_upsample_stage1 = cls._execute_upsample_stage1  # 静态方法
        instance._execute_upsample_stage2 = cls._execute_upsample_stage2  # 静态方法
    
    # =========== 公共接口 ===========
    
    def forward_chunked(
        self, 
        x: SparseTensor, 
        axis: int = 3,
        return_subs: bool = False,
        chunk_size_override: Optional[int] = None,
        use_checkpoint: bool = False,
    ) -> SparseTensor:
        """
        逐层自适应 Chunked forward pass。
        
        每层根据实时显存余量、当前点数和通道数自动估算 chunk_size，
        无需外部手动配置。仅支持 batch_size=1。
        
        Args:
            x: 输入 SparseTensor
            axis: 切分轴 (1=x, 2=y, 3=z)
            return_subs: 推理时是否返回 subdivision 预测（用于后续纹理解码）
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
        """
        assert return_subs == False or self.pred_subdiv == True, \
            "Only decoders with pred_subdiv=True can be used with return_subs"
        
        h = self.from_latent(x)  # SparseTensor feats: (N, C_latent)
        h = h.type(self.dtype)  # SparseTensor feats: (N, C_latent)
        
        collect_subdiv = (self.training and self.pred_subdiv) or return_subs
        all_subs, all_subs_gt = [], []
        
        for i, level_blocks in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                conv_blocks = level_blocks[:-1]
                upsample_block = level_blocks[-1]
            else:
                conv_blocks = level_blocks
                upsample_block = None
            
            # ★ 逐层确定 chunk_size：优先使用 override，否则自动估算
            if chunk_size_override is not None:
                chunk_size = chunk_size_override
            else:
                chunk_size = self._estimate_level_chunk_size(h, axis)
            coord_range = h.coords[:, axis].max().item() + 1
            logging.info(
                f"[Decoder L{i}] chunk={chunk_size}, coord_range={coord_range}, "
                f"points={h.coords.shape[0]}, ch={h.feats.shape[1]}, "
                f"chunked={'YES' if chunk_size < coord_range else 'NO'}"
            )
            
            h, subdiv, subdiv_gt = self._process_level_chunked_ckpt(
                h, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv,
                use_checkpoint=use_checkpoint,
            )  # h: SparseTensor feats (N, C); subdiv: SparseTensor or None; subdiv_gt: Tensor or None
            
            if subdiv is not None:
                all_subs.append(subdiv)
            if subdiv_gt is not None:
                all_subs_gt.append(subdiv_gt)
        
        # output_layer + layer_norm：当 use_checkpoint 时也放进 checkpoint，
        # 释放保存的输入张量（15.5M × 64 × 2 ≈ 2 GB × 2 = ~4 GB）
        if use_checkpoint:
            def _output_fn(feats):
                feats = feats.to(x.dtype)                         # (N, C)
                feats = F.layer_norm(feats, feats.shape[-1:])     # (N, C)
                return self.output_layer(h.replace(feats)).feats  # (N, C_out)
            h = h.replace(ckpt(_output_fn, h.feats, use_reentrant=False))  # SparseTensor feats: (N, C_out)
        else:
            h = h.type(x.dtype)  # SparseTensor feats: (N, C_latent)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))  # SparseTensor feats: (N, C_latent)
            h = self.output_layer(h)  # SparseTensor feats: (N, C_out)
        
        # 返回格式与原始 forward() 完全兼容
        if self.training and self.pred_subdiv:
            return h, all_subs_gt, all_subs
        else:
            if return_subs:
                return h, all_subs
            else:
                return h
    
    # =========== 逐层显存估算 ===========
    
    @staticmethod
    def _estimate_level_chunk_size(
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
        device = h.feats.device
        total = torch.cuda.get_device_properties(device).total_memory  # bytes
        reserved = torch.cuda.memory_reserved(device)  # bytes
        available = int((total - reserved) * target_ratio)  # bytes
        
        num_points = h.coords.shape[0]
        channels = h.feats.shape[1]
        coord_range = h.coords[:, axis].max().item() + 1
        
        cost_per_point = channels * bytes_per_element  # bytes/point
        max_points = max(available // cost_per_point, 1)
        
        if num_points <= max_points:
            return coord_range  # 显存充足，不分块
        
        # chunk_size ∝ (max_points / num_points) × coord_range
        return max(coord_range * max_points // num_points, min_chunk)
    
    # =========== 内部方法 ===========
    
    def _process_level_chunked_ckpt(
        self,
        h: SparseTensor,
        conv_blocks: list,
        upsample_block,
        axis: int,
        chunk_size: int,
        collect_subdiv: bool,
        use_checkpoint: bool = False,
    ) -> Tuple[SparseTensor, Optional[SparseTensor], Optional[torch.Tensor]]:
        """
        带可选 level-level gradient checkpoint 的单层 chunked 处理。
        
        当 use_checkpoint=True 时，使用 torch.utils.checkpoint 包裹
        _process_level_chunked 调用，释放本层中间激活（conv/upsample 的
        前向缓存），backward 时按需重算。内部的 block-level checkpoint
        在重算时也会正常触发，两级 checkpoint 互不冲突。
        
        实现要点：
        - h.feats（需要梯度追踪的标准 Tensor）和 h（SparseTensor，含
          coords/scale/spatial_cache 元数据）分别作为 checkpoint 的 args
          传入，保证被 tuple 捕获，不受外层循环变量覆盖影响。
        - checkpoint 函数只返回 h_out.feats（标准 Tensor），满足
          use_reentrant=False 的输出约束。
        - subdiv / subdiv_gt 等非梯度输出通过 _captured dict 侧信道带出；
          每次调用创建独立的 dict，各层互不干扰。
        
        Args:
            h: 当前层输入 SparseTensor
            conv_blocks: 当前层的 SparseConvNeXtBlock3d 列表
            upsample_block: SparseResBlockC2S3d 或 None（最后一层无上采样）
            axis: 切分轴 (1=x, 2=y, 3=z)
            chunk_size: 坐标空间的分块大小
            collect_subdiv: 是否收集 subdivision 预测和 GT
            use_checkpoint: 是否启用 level-level gradient checkpoint
            
        Returns:
            h_out: 处理后的 SparseTensor（上采样层坐标 ×2）
            subdiv: subdivision 预测 SparseTensor 或 None
            subdiv_gt: subdivision GT Tensor 或 None
        """
        if not use_checkpoint:
            return self._process_level_chunked(
                h, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv
            )

        # 侧信道：捕获 checkpoint 函数内部的非 Tensor 输出
        # 每次调用新建独立 dict，self / _captured 通过闭包捕获（不在循环内，安全）
        _captured = {}

        def _forward_level(
            h_feats,          # Tensor (N, C) — checkpoint 追踪梯度
            h_sparse,         # SparseTensor  — 携带 coords / scale / spatial_cache
            conv_blocks,      # List[Module]
            upsample_block,   # Module or None
            axis,             # int
            chunk_size,       # int
            collect_subdiv,   # bool
        ):
            # 用 checkpoint 保存的 SparseTensor 元数据 + 梯度追踪的 feats 重建输入
            h_rebuilt = h_sparse.replace(h_feats)
            h_out, subdiv, subdiv_gt = self._process_level_chunked(
                h_rebuilt, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv
            )
            _captured['h_out'] = h_out          # SparseTensor（含新 coords）
            _captured['subdiv'] = subdiv        # SparseTensor or None
            _captured['subdiv_gt'] = subdiv_gt  # Tensor or None
            return h_out.feats  # (N_out, C_out) — 唯一需要梯度追踪的输出

        # 所有会随循环迭代变化的变量都走 checkpoint 的 *args（被 tuple 捕获）
        h_out_feats = ckpt(
            _forward_level,
            h.feats, h, conv_blocks, upsample_block, axis, chunk_size, collect_subdiv,
            use_reentrant=False,
        )  # (N_out, C_out)

        # 用 forward 时捕获的 SparseTensor 元数据 + checkpoint 返回的 feats 重建输出
        h_out = _captured['h_out'].replace(h_out_feats)
        return h_out, _captured['subdiv'], _captured['subdiv_gt']
    
    def _process_level_chunked(
        self,
        h: SparseTensor,
        conv_blocks: list,
        upsample_block: Optional[object],
        axis: int,
        chunk_size: int,
        collect_subdiv: bool,
    ) -> Tuple[SparseTensor, Optional[SparseTensor], Optional[torch.Tensor]]:
        """
        处理一个分辨率层级，采用两阶段分块策略。
        
        Stage 1: ConvNeXt blocks + upsample.conv1 + updown (原坐标系 → 2x坐标系)
        Stage 2: upsample.conv2 + skip_connection (2x坐标系内)
        
        Args:
            h: 输入 SparseTensor
            conv_blocks: ConvNeXt blocks 列表
            upsample_block: Upsample block（最后一层为 None）
            axis: 切分轴
            chunk_size: chunk 大小
            collect_subdiv: 是否收集 subdivision 预测
            
        Returns:
            output: 处理后的 SparseTensor
            subdiv: subdivision 预测（SparseTensor），None 如果不收集
            subdiv_gt: subdivision ground truth（Tensor），None 如果不收集
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
                output, skip, subdiv = self._execute_upsample_stage1(upsample_block, x)  # SparseTensor feats: (N_chunk, C)
                chunk.set_result(output)
                chunk.set_attached_result("skip", skip)
                if subdiv is not None:
                    # 使用 get_valid_feats 过滤 halo 区域的预测
                    subdiv_chunks.append(chunk.get_valid_feats(subdiv))  # (N_valid, 3)
            else:
                chunk.set_result(x)
        
        torch.cuda.empty_cache()  # 释放 chunk 处理中的碎片显存，缓解 merge 阶段 OOM
        merged_s1 = chunked_s1.merge()  # SparseTensor feats: (N, C)
        merged_skip = chunked_s1.get_attached("skip")  # SparseTensor feats: (N, C) or None
        
        # 合并 subdivision 预测和 GT（已经过滤 halo，直接拼接）
        subdiv = SparseTensor(
            torch.cat(subdiv_chunks, dim=0),  # (N, 3)
            h.coords.clone(),  # (N, 4) 使用原始坐标
            scale=h._scale
        ) if subdiv_chunks else None
        subdiv_gt = torch.cat(subdiv_gt_chunks, dim=0) if subdiv_gt_chunks else None  # (N, 3) or None
        
        # ======== Stage 2 ========
        if has_upsample and merged_skip is not None:
            chunked_s2 = ChunkableSparseTensor(
                merged_s1, axis=axis, chunk_size=chunk_size * 2, halo=1,
                indexed_cache_keys=[]  # Stage 2 不需要处理 indexed cache
            )
            chunked_s2.attach("skip", merged_skip)
            
            for chunk in chunked_s2.chunks():
                result = self._execute_upsample_stage2(
                    upsample_block, chunk.tensor, chunk.get("skip")
                )  # SparseTensor feats: (N_chunk, C)
                chunk.set_result(result)
            
            torch.cuda.empty_cache()  # 释放 chunk 处理中的碎片显存，缓解 merge 阶段 OOM
            final_output = chunked_s2.merge()  # SparseTensor feats: (N, C)
        else:
            final_output = merged_s1
        
        return final_output, subdiv, subdiv_gt
    
    @staticmethod
    def _execute_upsample_stage1(
        upsample_block, 
        x: SparseTensor
    ) -> Tuple[SparseTensor, SparseTensor, Optional[SparseTensor]]:
        """
        执行 SparseResBlockC2S3d 的第一阶段。
        
        执行顺序：
        1. 预测 subdivision（如果 pred_subdiv=True）
        2. norm1 + silu + conv1
        3. updown（坐标 ×2）
        
        Args:
            upsample_block: SparseResBlockC2S3d 实例
            x: 输入 SparseTensor
            
        Returns:
            output: conv1 + updown 后的结果（2x 坐标系）
            skip: updown 后的 x（2x 坐标系，用于 skip connection）
            subdiv: subdivision 预测（原坐标系），None 如果不预测
        """
        # 预测 subdivision
        if upsample_block.pred_subdiv:
            subdiv = upsample_block.to_subdiv(x)  # SparseTensor feats: (N, 3)
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
