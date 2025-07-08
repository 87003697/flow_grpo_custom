#!/usr/bin/env python3
"""
下载EVA模型权重到pretrained_weights目录
"""

import os
import sys
import timm
import open_clip
import torch
from pathlib import Path

def download_eva_giant():
    """下载EVA Giant模型权重"""
    print("🔄 正在下载EVA Giant模型权重...")
    
    # 创建模型以触发下载
    model = timm.create_model('eva_giant_patch14_560', pretrained=True)
    
    # 获取权重
    state_dict = model.state_dict()
    
    # 保存到pretrained_weights目录
    project_root = Path(__file__).parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    
    eva_giant_path = weights_dir / "eva_giant_patch14_560.pt"
    torch.save(state_dict, eva_giant_path)
    
    print(f"✅ EVA Giant权重已保存到: {eva_giant_path}")
    return eva_giant_path

def download_eva02_clip():
    """下载EVA02 CLIP模型权重"""
    print("🔄 正在下载EVA02 CLIP模型权重...")
    
    # 创建模型以触发下载
    model, _, preprocess = open_clip.create_model_and_transforms(
        'EVA02-E-14-plus', 
        pretrained='laion2b_s9b_b144k'
    )
    
    # 获取权重
    state_dict = model.state_dict()
    
    # 保存到pretrained_weights目录
    project_root = Path(__file__).parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    
    eva02_path = weights_dir / "eva02_e_14_plus_laion2b_s9b_b144k.pt"
    torch.save(state_dict, eva02_path)
    
    print(f"✅ EVA02 CLIP权重已保存到: {eva02_path}")
    return eva02_path

def main():
    print("🚀 开始下载EVA模型权重...")
    
    try:
        # 下载EVA Giant权重
        eva_giant_path = download_eva_giant()
        
        # 下载EVA02 CLIP权重
        eva02_path = download_eva02_clip()
        
        print("\n✅ 所有EVA模型权重下载完成！")
        print(f"EVA Giant: {eva_giant_path}")
        print(f"EVA02 CLIP: {eva02_path}")
        
        # 显示文件大小
        eva_giant_size = eva_giant_path.stat().st_size / (1024*1024)
        eva02_size = eva02_path.stat().st_size / (1024*1024)
        
        print(f"\n📊 文件大小:")
        print(f"EVA Giant: {eva_giant_size:.1f} MB")
        print(f"EVA02 CLIP: {eva02_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ 下载过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 