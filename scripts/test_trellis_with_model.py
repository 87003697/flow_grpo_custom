#!/usr/bin/env python3
"""
TRELLIS模型加载和基础推理测试
验证下载的TRELLIS-image-large模型是否能正常工作
"""

import sys
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """测试模型加载"""
    print("🧪 测试TRELLIS-image-large模型加载...")
    
    try:
        from generators.trellis.pipeline import TrellisStage2Pipeline
        
        print("🔄 正在加载模型...")
        pipeline = TrellisStage2Pipeline()
        
        # 遵循TRELLIS官方模式：手动将模型移动到GPU
        if torch.cuda.is_available():
            print("🚀 将模型移动到GPU...")
            pipeline.cuda()
        
        print("✅ 模型加载成功")
        print(f"📍 设备: {pipeline.device}")
        
        # 检查可训练模型
        trainable_model = pipeline.get_trainable_model()
        print(f"🎯 可训练模型: {type(trainable_model).__name__}")
        
        # 检查模型参数数量
        total_params = sum(p.numel() for p in pipeline.core_pipeline.models.values() for p in p.parameters())
        trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
        
        print(f"📊 总参数量: {total_params:,}")
        print(f"🎯 可训练参数量: {trainable_params:,}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_conditions(pipeline):
    """测试图像条件编码"""
    print("\n🧪 测试图像条件编码...")
    
    try:
        from generators.trellis.utils import trellis_preprocess_image
        
        # 创建测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        print(f"原始图像: {test_image.size}")
        
        # 预处理
        preprocessed = trellis_preprocess_image(test_image)
        print(f"预处理后: {preprocessed.size}")
        
        # 图像条件编码
        images = [preprocessed]
        image_conds = pipeline.prepare_image_conditions(images)
        
        print("✅ 图像条件编码成功:")
        for key, tensor in image_conds.items():
            print(f"   {key}: {tensor.shape} {tensor.dtype} (设备: {tensor.device})")
        
        return image_conds
        
    except Exception as e:
        print(f"❌ 图像条件编码失败: {e}")
        return None

def test_stage1_inference(pipeline, image_conds):
    """测试Stage 1推理"""
    print("\n🧪 测试Stage 1推理 (稀疏结构生成)...")
    
    try:
        coords = pipeline.forward_stage1(image_conds)
        print(f"✅ Stage 1推理成功:")
        print(f"   稀疏坐标shape: {coords.shape}")
        print(f"   稀疏点数量: {coords.shape[0]}")
        print(f"   坐标范围: [{coords.min().item():.2f}, {coords.max().item():.2f}]")
        
        return coords
        
    except Exception as e:
        print(f"❌ Stage 1推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_stage2_inference(pipeline, coords, image_conds):
    """测试Stage 2推理"""
    print("\n🧪 测试Stage 2推理 (SLAT生成)...")
    
    try:
        slat_output, all_latents, all_log_probs = pipeline.forward_stage2_with_logprob(coords, image_conds)
        
        print(f"✅ Stage 2推理成功:")
        print(f"   SLAT coords shape: {slat_output.coords.shape}")
        print(f"   SLAT feats shape: {slat_output.feats.shape}")
        print(f"   特征维度: {slat_output.feats.shape[1]}")
        print(f"   注意: LogProb功能待实现 (当前返回空列表)")
        
        return slat_output
        
    except Exception as e:
        print(f"❌ Stage 2推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mesh_decoding(pipeline, slat_output):
    """测试mesh解码"""
    print("\n🧪 测试mesh解码...")
    
    try:
        meshes = pipeline.decode_slat_to_mesh(slat_output)
        
        print(f"✅ mesh解码成功:")
        print(f"   生成mesh数量: {len(meshes)}")
        
        if meshes:
            mesh = meshes[0]
            print(f"   第一个mesh:")
            print(f"     类型: {type(mesh).__name__}")
            if hasattr(mesh, 'vertices'):
                print(f"     vertices: {len(mesh.vertices)}")
            if hasattr(mesh, 'faces'):
                print(f"     faces: {len(mesh.faces)}")
            if hasattr(mesh, 'is_watertight'):
                print(f"     是否watertight: {mesh.is_watertight}")
            elif hasattr(mesh, 'mesh') and hasattr(mesh.mesh, 'is_watertight'):
                print(f"     是否watertight: {mesh.mesh.is_watertight}")
        
        return meshes
        
    except Exception as e:
        print(f"❌ mesh解码失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_usage():
    """测试内存使用"""
    print("\n🧪 测试内存使用...")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        
        print(f"   GPU内存已分配: {memory_allocated:.2f} GB")
        print(f"   GPU内存已保留: {memory_reserved:.2f} GB")
        
        if memory_allocated > 12.0:
            print("⚠️  内存使用较高 (>12GB)")
        else:
            print("✅ 内存使用正常")
    else:
        print("   跳过GPU内存检查 (CUDA不可用)")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 TRELLIS-image-large 模型测试套件")
    print("=" * 60)
    
    # 1. 模型加载测试
    pipeline = test_model_loading()
    if pipeline is None:
        print("❌ 模型加载失败，退出测试")
        return False
    
    # 2. 图像条件编码测试
    image_conds = test_image_conditions(pipeline)
    if image_conds is None:
        print("❌ 图像条件编码失败，退出测试") 
        return False
    
    # 3. Stage 1推理测试
    coords = test_stage1_inference(pipeline, image_conds)
    if coords is None:
        print("❌ Stage 1推理失败，退出测试")
        return False
    
    # 4. Stage 2推理测试
    slat_output = test_stage2_inference(pipeline, coords, image_conds)
    if slat_output is None:
        print("❌ Stage 2推理失败，退出测试")
        return False
    
    # 5. Mesh解码测试
    meshes = test_mesh_decoding(pipeline, slat_output)
    if meshes is None:
        print("❌ Mesh解码失败")
        return False
    
    # 6. 内存使用测试
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！TRELLIS-image-large模型工作正常")
    print("📋 模型验证完成，可以进行GRPO训练开发")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 