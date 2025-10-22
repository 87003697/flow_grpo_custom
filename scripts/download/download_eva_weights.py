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
import urllib.request

UNI3D_G_URL = "https://huggingface.co/BAAI/Uni3D/resolve/main/modelzoo/uni3d-g/model.pt?download=true"

def download_eva_giant():
    """下载EVA Giant模型权重"""
    print("🔄 正在下载EVA Giant模型权重...")
    
    # 创建模型以触发下载
    model = timm.create_model('eva_giant_patch14_560', pretrained=True)
    
    # 获取权重
    state_dict = model.state_dict()
    
    # 保存到pretrained_weights目录
    # 修正：保存到仓库根目录下的 pretrained_weights/
    project_root = Path(__file__).resolve().parent.parent.parent
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
    # 修正：保存到仓库根目录下的 pretrained_weights/
    project_root = Path(__file__).resolve().parent.parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    
    eva02_path = weights_dir / "eva02_e_14_plus_laion2b_s9b_b144k.pt"
    torch.save(state_dict, eva02_path)
    
    print(f"✅ EVA02 CLIP权重已保存到: {eva02_path}")
    return eva02_path

def download_uni3d_g():
    """下载 Uni3D-g 权重并保存为 uni3d-g.pt（带存在性检查）。"""
    print("🔄 正在下载Uni3D-g权重...")
    project_root = Path(__file__).resolve().parent.parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    out_path = weights_dir / "uni3d-g.pt"
    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        print(f"✅ 发现已存在的Uni3D-g权重: {out_path}")
        return out_path
    tmp_path = out_path.with_suffix(".tmp")
    urllib.request.urlretrieve(UNI3D_G_URL, tmp_path)
    Path(tmp_path).rename(out_path)
    print(f"✅ Uni3D-g权重已保存到: {out_path}")
    return out_path

def main():
    print("🚀 开始下载EVA模型权重...")
    # 下载EVA Giant权重
    eva_giant_path = download_eva_giant()
    
    # 下载EVA02 CLIP权重
    eva02_path = download_eva02_clip()
    
    # 下载Uni3D-g权重
    uni3d_path = download_uni3d_g()
    
    print("\n✅ 所有EVA模型权重下载完成！")
    print(f"EVA Giant: {eva_giant_path}")
    print(f"EVA02 CLIP: {eva02_path}")
    print(f"Uni3D-g: {uni3d_path}")
    
    # 显示文件大小
    eva_giant_size = eva_giant_path.stat().st_size / (1024*1024)
    eva02_size = eva02_path.stat().st_size / (1024*1024)
    uni3d_size = uni3d_path.stat().st_size / (1024*1024)
    
    print(f"\n📊 文件大小:")
    print(f"EVA Giant: {eva_giant_size:.1f} MB")
    print(f"EVA02 CLIP: {eva02_size:.1f} MB")
    print(f"Uni3D-g: {uni3d_size:.1f} MB")

if __name__ == "__main__":
    main() 