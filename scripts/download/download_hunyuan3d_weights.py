#!/usr/bin/env python3
"""
下载HunyuanD模型权重到pretrained_weights目录
"""

import os
import sys
import shutil
from pathlib import Path

def download_hunyuan3d_weights(model_name='tencent/Hunyuan3D-2.1'):
    """下载HunyuanD模型权重"""
    print(f"🚀 开始下载HunyuanD权重: {model_name}")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 需要安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)
    
    # 设置目录路径 - 按照HunyuanD期望的结构
    project_root = Path(__file__).parent.parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    
    # HunyuanD期望的路径结构: pretrained_weights/tencent/Hunyuan3D-2.1/
    hunyuan_dir = weights_dir / model_name  # tencent/Hunyuan3D-2.1
    hunyuan_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    # 下载子模块列表
    subfolders = [
        "hunyuan3d-dit-v2-1",     # DiT模型
        "hunyuan3d-vae-v2-1",     # VAE模型
    ]
    
    for subfolder in subfolders:
        print(f"\n🔄 正在下载 {subfolder}...")
        
        try:
            # 直接下载到目标位置
            downloaded_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=hunyuan_dir
            )
            
            target_path = hunyuan_dir / subfolder
            if target_path.exists():
                downloaded_files.append(target_path)
                print(f"✅ {subfolder} 下载完成")
            else:
                print(f"⚠️ {subfolder} 下载可能不完整")
                
        except Exception as e:
            print(f"❌ 下载 {subfolder} 失败: {e}")
            return False
    
    # 验证下载结果
    print(f"\n📊 下载结果验证:")
    total_size = 0
    for file_path in downloaded_files:
        if file_path.exists():
            size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            print(f"  ✅ {file_path.name}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {file_path.name}: 不存在")
            
    print(f"\n💾 总下载大小: {total_size:.1f} MB")
    print(f"📁 权重保存位置: {hunyuan_dir}")
    
    return True

def cleanup_old_downloads():
    """清理旧的下载目录"""
    project_root = Path(__file__).parent.parent.parent
    weights_dir = project_root / "pretrained_weights"
    
    # 清理旧的hunyuan3d目录
    old_hunyuan_dir = weights_dir / "hunyuan3d"
    if old_hunyuan_dir.exists():
        print(f"🧹 清理旧的下载目录: {old_hunyuan_dir}")
        shutil.rmtree(old_hunyuan_dir)
        print("✅ 旧目录清理完成")

def main():
    print("🚀 开始下载HunyuanD模型权重...")
    
    try:
        # 清理旧的下载
        cleanup_old_downloads()
        
        # 下载权重
        success = download_hunyuan3d_weights()
        
        if success:
            print("\n✅ HunyuanD权重下载完成！")
            
            print("\n🎯 接下来的步骤:")
            print("1. 测试: python scripts/test_hunyuan3d.py")
            print("2. 权重位置: pretrained_weights/tencent/Hunyuan3D-2.1/")
            print("3. 模型将自动从本地权重加载")
            
        else:
            print("❌ HunyuanD权重下载失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 下载过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 