#!/usr/bin/env python3
"""
SparseTensor处理工具函数
用于GRPO训练中的稀疏张量操作和批处理

参考:
- TRELLIS 官方实现: `_reference_codes/TRELLIS/trellis/modules/sparse/basic.py:420-444` (sparse_cat)
- SD3 对应逻辑: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`（正/负分支再线性合并）
"""
from typing import List
import torch

from generators.trellis import sparse as sp

def sparse_tensor_cat(tensors: List[sp.SparseTensor]) -> sp.SparseTensor:
    """SparseTensor的批量拼接操作，用于CFG处理
    
    参考: _reference_codes/TRELLIS/trellis/modules/sparse/basic.py:420-444 (sparse_cat)
    基于TRELLIS官方sparse_cat实现，dim=0时的逻辑
    
    Args:
        tensors (List[sp.SparseTensor]): 要拼接的稀疏张量列表
        
    Returns:
        sp.SparseTensor: 拼接后的稀疏张量
    """
    if not tensors:
        raise ValueError("输入张量列表为空")
    
    if len(tensors) == 1:
        return tensors[0]
    
    # 按照源代码逻辑进行batch维度拼接
    start = 0
    coords = []
    for input_tensor in tensors:
        coords.append(input_tensor.coords.clone().to(torch.int32))  # 形状 (N_i, 4) 确保为int32类型
        coords[-1][:, 0] += start  # 形状 (N_i, 4) 调整 batch 索引
        start += input_tensor.shape[0]  # 标量，更新下一段起始 batch 索引
    
    # 拼接坐标和特征
    combined_coords = torch.cat(coords, dim=0)  # 形状 (sum(N_i), 4)
    combined_feats = torch.cat([input_tensor.feats for input_tensor in tensors], dim=0)  # 形状 (sum(N_i), C)
    
    # 创建新的SparseTensor
    output = sp.SparseTensor(
        coords=combined_coords,  # 形状 (sum(N_i), 4)
        feats=combined_feats,    # 形状 (sum(N_i), C)
    )
    
    return output 