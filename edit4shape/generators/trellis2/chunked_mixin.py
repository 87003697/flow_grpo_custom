"""
Chunked Forward Mixin for SparseVAE Decoder.

提供 chunked forward 能力，通过 Mixin 方式注入到 SparseVAE 类中，
实现零侵入性的显存优化。

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
    output = decoder.forward_chunked(x, chunk_size=64)
    
    # 方式二：动态注入（不修改类定义，推荐用于预训练模型）
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    decoder = load_pretrained_decoder()
    ChunkedDecoderMixin.inject_to(decoder)
    output = decoder.forward_chunked(x, chunk_size=64)
    
    # 推理时获取 subdivision（用于后续纹理解码）
    h, subs = decoder.forward_chunked(x, chunk_size=64, return_subs=True)
"""
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

# 使用绝对导入（适配移动到 edit4shape/generators/trellis2/）
from trellis2.modules.sparse import SparseTensor
from .chunked import ChunkableSparseTensor


class ChunkedDecoderMixin:
    """
    为 SparseVAE Decoder 提供 chunked forward 能力的 Mixin。
    
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
            output = decoder.forward_chunked(x, chunk_size=64)
        """
        from types import MethodType
        
        # 注入所有需要的方法
        instance.forward_chunked = MethodType(cls.forward_chunked, instance)
        instance._process_level_chunked = MethodType(cls._process_level_chunked, instance)
        instance._execute_upsample_stage1 = cls._execute_upsample_stage1  # 静态方法
        instance._execute_upsample_stage2 = cls._execute_upsample_stage2  # 静态方法
    
    # =========== 公共接口 ===========
    
    def forward_chunked(
        self, 
        x: SparseTensor, 
        chunk_size: int = 64,
        axis: int = 3,
        return_subs: bool = False,
    ) -> SparseTensor:
        """
        Chunked forward pass，分块处理以降低显存峰值。
        
        仅支持 batch_size=1。
        
        Args:
            x: 输入 SparseTensor
            chunk_size: 基础 chunk 大小
            axis: 切分轴 (1=x, 2=y, 3=z)
            return_subs: 推理时是否返回 subdivision 预测（用于后续纹理解码）
            
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
        
        current_chunk_size = chunk_size
        # 训练时收集 subdiv 和 subdiv_gt；推理时如果 return_subs=True 也需要收集 subdiv
        collect_subdiv = (self.training and self.pred_subdiv) or return_subs
        
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
            
            h, subdiv, subdiv_gt = self._process_level_chunked(
                h, conv_blocks, upsample_block, axis, current_chunk_size, collect_subdiv
            )  # h: SparseTensor feats (N, C); subdiv: SparseTensor or None; subdiv_gt: Tensor or None
            
            if subdiv is not None:
                all_subs.append(subdiv)
            if subdiv_gt is not None:
                all_subs_gt.append(subdiv_gt)
            
            if upsample_block is not None:
                current_chunk_size *= 2
        
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
    
    # =========== 内部方法 ===========
    
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
