# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder, PointCrossAttentionEncoder
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from .volume_decoders import VanillaVolumeDecoder, HierarchicalVolumeDecoding, FlashVDMVolumeDecoding
from ...utils import logger, synchronize_timer, smart_load_model


def create_default_sphere_mesh(radius=0.5, subdivisions=2):
    """
    生成默认的球形mesh作为fallback
    
    Args:
        radius: 球的半径，默认0.5
        subdivisions: 细分级别，越高越平滑
    
    Returns:
        包含球形mesh的列表，格式与正常mesh输出一致
    """
    import math
    
    # 生成icosphere的顶点和面
    # 基于正二十面体细分算法
    
    # 黄金比例
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    
    # 初始顶点（正二十面体的12个顶点）
    vertices = []
    
    # 添加顶点
    a = radius / math.sqrt(phi * phi + 1.0)
    b = a * phi
    
    # 生成12个初始顶点
    vertices.extend([
        [-a, b, 0], [a, b, 0], [-a, -b, 0], [a, -b, 0],
        [0, -a, b], [0, a, b], [0, -a, -b], [0, a, -b],
        [b, 0, -a], [b, 0, a], [-b, 0, -a], [-b, 0, a]
    ])
    
    # 初始面（正二十面体的20个面）
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # 简单细分（仅做一次以保持mesh不太复杂）
    if subdivisions > 0:
        new_vertices = list(vertices)
        new_faces = []
        vertex_cache = {}
        
        def get_middle_point(v1, v2):
            """获取两点中点并归一化到球面"""
            key = (min(v1, v2), max(v1, v2))
            if key in vertex_cache:
                return vertex_cache[key]
            
            # 计算中点
            p1, p2 = new_vertices[v1], new_vertices[v2]
            middle = [(p1[i] + p2[i]) / 2.0 for i in range(3)]
            
            # 归一化到球面
            length = math.sqrt(sum(x*x for x in middle))
            if length > 0:
                middle = [x * radius / length for x in middle]
            
            # 添加新顶点
            index = len(new_vertices)
            new_vertices.append(middle)
            vertex_cache[key] = index
            return index
        
        # 对每个面进行细分
        for face in faces:
            v1, v2, v3 = face
            a = get_middle_point(v1, v2)
            b = get_middle_point(v2, v3) 
            c = get_middle_point(v3, v1)
            
            new_faces.extend([
                [v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]
            ])
        
        vertices = new_vertices
        faces = new_faces
    
    # 转换为numpy数组（Hunyuan3D mesh格式）
    vertices_np = np.array(vertices, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)
    
    # 创建mesh对象（模拟Hunyuan3D的mesh输出格式）
    class SimpleMesh:
        def __init__(self, vertices, faces):
            self.v = vertices
            self.f = faces
            self.vc = None  # 默认无颜色
    
    sphere_mesh = SimpleMesh(vertices_np, faces_np)
    
    logger.warning(f"🟡 生成默认球形mesh: {len(vertices)} 顶点, {len(faces)} 面")
    return [sphere_mesh]  # 返回列表格式，与正常输出保持一致


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        """
        Initialize a diagonal Gaussian distribution with mean and log-variance parameters.

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]): 
                Either a single tensor containing concatenated mean and log-variance along `feat_dim`,
                or a list of two tensors [mean, logvar].
            deterministic (bool, optional): If True, the distribution is deterministic (zero variance). 
                Default is False. feat_dim (int, optional): Dimension along which mean and logvar are 
                concatenated if parameters is a single tensor. Default is 1.
        """
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        """
        Sample from the diagonal Gaussian distribution.

        Returns:
            torch.Tensor: A sample tensor with the same shape as the mean.
        """
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2, 3)):
        """
        Compute the Kullback-Leibler (KL) divergence between this distribution and another.

        If `other` is None, compute KL divergence to a standard normal distribution N(0, I).

        Args:
            other (DiagonalGaussianDistribution, optional): Another diagonal Gaussian distribution.
            dims (tuple, optional): Dimensions along which to compute the mean KL divergence. 
                Default is (1, 2, 3).

        Returns:
            torch.Tensor: The mean KL divergence value.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class VectsetVAE(nn.Module):

    @classmethod
    @synchronize_timer('VectsetVAE Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        model_kwargs = config['params']
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-vae-v2-1',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )
        
    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def __init__(
        self,
        volume_decoder=None,
        surface_extractor=None
    ):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        """
        将latents转换为mesh，包含异常处理和默认球形mesh fallback
        """
        # 🔧 修复：FlashVDM只能处理单样本，需要用for循环处理批次
        if isinstance(self.volume_decoder, FlashVDMVolumeDecoding) and latents.shape[0] > 1:
            # 对于FlashVDM，逐个处理每个样本
            all_outputs = []
            for i in range(latents.shape[0]):
                try:
                    single_latents = latents[i:i+1]  # 保持batch维度
                    
                    # 🔧 关键修复：为每个样本创建独立的processor，避免状态污染
                    from .volume_decoders import FlashVDMCrossAttentionProcessor
                    fresh_processor = FlashVDMCrossAttentionProcessor()
                    
                    # 保存原始processor
                    original_processor = self.geo_decoder.cross_attn_decoder.attn.attention.attn_processor
                    
                    # 设置新的processor
                    self.geo_decoder.set_cross_attention_processor(fresh_processor)
                    
                    with synchronize_timer(f'Volume decoding (sample {i+1}/{latents.shape[0]})'):
                        grid_logits = self.volume_decoder(single_latents, self.geo_decoder, **kwargs)

                    with synchronize_timer(f'Surface extraction (sample {i+1}/{latents.shape[0]})'):
                        outputs = self.surface_extractor(grid_logits, **kwargs)
                    all_outputs.extend(outputs)  # outputs是一个列表，需要extend
                    
                    # 恢复原始processor
                    self.geo_decoder.set_cross_attention_processor(original_processor)
                    
                except Exception as e:
                    logger.warning(f"单个样本 {i+1} mesh生成失败: {e}, 使用默认球形mesh")
                    # 恢复原始processor（如果出现异常）
                    if 'original_processor' in locals():
                        self.geo_decoder.set_cross_attention_processor(original_processor)
                    # 添加默认球形mesh
                    default_meshes = create_default_sphere_mesh()
                    all_outputs.extend(default_meshes)
                    
            return all_outputs
        else:
            # 单样本或非FlashVDM情况，使用原始逻辑
            with synchronize_timer('Volume decoding'):
                grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
                
            with synchronize_timer('Surface extraction'):
                outputs = self.surface_extractor(grid_logits, **kwargs)
            return outputs


    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
    ):
        if enabled:
            if adaptive_kv_selection:
                self.volume_decoder = FlashVDMVolumeDecoding(topk_mode)
            else:
                self.volume_decoder = HierarchicalVolumeDecoding()
            if mc_algo not in SurfaceExtractors.keys():
                raise ValueError(f'Unsupported mc_algo {mc_algo}, available:{list(SurfaceExtractors.keys())}')
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class ShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        ckpt_path = None
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=self.downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, latents):
        # 🔧 修复：确保输入数据类型与模型权重一致
        model_dtype = next(self.parameters()).dtype
        latents = latents.to(dtype=model_dtype)
        
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def encode(self, surface, sample_posterior=True):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, _ = self.encoder(pc, feats)
        # print(latents.shape, self.pre_kl.weight.shape)
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            latents = posterior.sample()
        else:
            latents = posterior.mode()
        return latents

    def decode(self, latents):
        # 🔧 修复：确保输入数据类型与模型权重一致
        model_dtype = next(self.parameters()).dtype
        latents = latents.to(dtype=model_dtype)
        
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents
