#!/usr/bin/env python3
"""
PyTorch RMSNorm 兼容性补丁
为PyTorch 2.2.1等老版本提供RMSNorm支持
"""
import torch
import torch.nn as nn
import math

def apply_rmsnorm_patch():
    """为老版本PyTorch添加RMSNorm支持"""
    
    # 检查是否已经有RMSNorm
    if hasattr(torch.nn, 'RMSNorm'):
        print("✅ PyTorch已有原生RMSNorm，无需补丁")
        return
    
    print("🔧 为PyTorch添加RMSNorm兼容性补丁...")
    
    class RMSNorm(nn.Module):
        """
        Root Mean Square Layer Normalization的兼容实现
        参考: https://arxiv.org/abs/1910.07467
        """
        def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            
            if eps is None:
                eps = torch.finfo(torch.float32).eps
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter('weight', None)
                
        def forward(self, x):
            # 计算RMS: sqrt(mean(x^2) + eps)
            norm_dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
            rms = torch.sqrt(torch.mean(x.pow(2), dim=norm_dims, keepdim=True) + self.eps)
            
            # 归一化
            x_normed = x / rms
            
            # 如果有可学习参数，应用缩放
            if self.elementwise_affine:
                x_normed = x_normed * self.weight
                
            return x_normed
        
        def extra_repr(self):
            return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
        
        def reset_parameters(self):
            if self.elementwise_affine:
                nn.init.ones_(self.weight)
    
    # 将RMSNorm添加到torch.nn
    torch.nn.RMSNorm = RMSNorm
    
    # 同时添加到torch.nn.modules.normalization（如果存在的话）
    if hasattr(torch.nn, 'modules') and hasattr(torch.nn.modules, 'normalization'):
        torch.nn.modules.normalization.RMSNorm = RMSNorm
    
    print("✅ RMSNorm补丁应用成功！")

if __name__ == "__main__":
    # 测试补丁
    apply_rmsnorm_patch()
    
    # 简单测试
    try:
        rms_norm = torch.nn.RMSNorm([2, 3])
        input_tensor = torch.randn(2, 2, 3)
        output = rms_norm(input_tensor)
        print(f"🎯 测试成功！输入shape: {input_tensor.shape}, 输出shape: {output.shape}")
    except Exception as e:
        print(f"❌ 测试失败: {e}") 