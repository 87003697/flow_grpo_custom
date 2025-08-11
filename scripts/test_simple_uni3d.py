#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from pathlib import Path
import trimesh
from PIL import Image

# 添加项目路径
sys.path.append('.')

from reward_models.uni3d_scorer.simple_uni3d import SimpleUni3DScorer
from kiui.mesh import Mesh

def load_glb_as_mesh(mesh_path):
    """加载GLB文件为Mesh对象"""
    scene = trimesh.load(mesh_path, force='scene')
    geometry = list(scene.geometry.values())[0]
    
    mesh = Mesh()
    mesh.v = geometry.vertices
    mesh.f = geometry.faces
    if hasattr(geometry.visual, 'vertex_colors'):
        mesh.vc = geometry.visual.vertex_colors[:, :3] / 255.0
    else:
        mesh.vc = None
    
    return mesh

def test_simple_uni3d_recall():
    """测试简化版uni3d_scorer在5个特定样本上的recall@1"""
    
    # 测试数据
    samples = [
        "dancing_patrick_star",
        "flying_ironman", 
        "scaring_skull",
        "walking_siamese_cat",
        "firing_pistol"
    ]
    
    print("🧪 测试简化版SimpleUni3DScorer - Recall@1")
    print("=" * 50)
    
    # 初始化评分器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    scorer = SimpleUni3DScorer(device)
    
    # 准备数据
    images = []
    meshes = []
    
    for sample in samples:
        image_path = f"dataset/eval3d/images/{sample}.png"
        mesh_path = f"dataset/eval3d/meshes/{sample}_textured_frame_000000.glb"
        
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            continue
        if not os.path.exists(mesh_path):
            print(f"❌ Mesh文件不存在: {mesh_path}")
            continue
            
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        
        # 加载mesh
        mesh = load_glb_as_mesh(mesh_path)
        meshes.append(mesh)
        print(f"✅ 加载样本: {sample}")
    
    if len(images) != len(samples):
        print(f"⚠️ 只找到 {len(images)}/{len(samples)} 个样本")
    
    if len(images) == 0:
        print("❌ 没有找到任何有效样本，退出测试")
        return
    
    # 批量评分
    print(f"\n🚀 开始批量评分 {len(images)} 个样本...")
    scores = scorer.compute_scores(meshes, images)
    
    print(f"\n📊 评分结果:")
    for i, (sample, score) in enumerate(zip(samples[:len(scores)], scores)):
        print(f"  {sample}: {score:.4f}")
    
    # 计算相似度矩阵
    print(f"\n🔄 计算完整相似度矩阵...")
    
    # 所有图像vs所有mesh的相似度矩阵
    all_scores = []
    for i, image in enumerate(images):
        row_scores = scorer.compute_scores(meshes, [image] * len(meshes))
        all_scores.append(row_scores)
        print(f"  图像 {i+1} vs 所有mesh: {[f'{s:.3f}' for s in row_scores]}")
    
    similarity_matrix = np.array(all_scores)
    
    # 计算Recall@1
    correct_matches = 0
    total_queries = len(images)
    
    print(f"\n🎯 Recall@1 分析:")
    for i in range(total_queries):
        # 每行找最高分的索引
        predicted_idx = np.argmax(similarity_matrix[i])
        ground_truth_idx = i  # 对角线为正确匹配
        
        is_correct = predicted_idx == ground_truth_idx
        if is_correct:
            correct_matches += 1
            
        print(f"  查询 {i+1} ({samples[i]}): "
              f"预测={predicted_idx+1} ({'✅' if is_correct else '❌'})")
    
    recall_at_1 = correct_matches / total_queries
    print(f"\n🏆 最终结果:")
    print(f"  正确匹配: {correct_matches}/{total_queries}")
    print(f"  Recall@1: {recall_at_1:.1%}")
    
    if recall_at_1 == 1.0:
        print("🎉 完美！SimpleUni3DScorer达到100%性能！")
    elif recall_at_1 > 0.8:
        print("✅ 性能良好，SimpleUni3DScorer效果不错")
    else:
        print("⚠️ 性能较低，需要检查配置或数据")

if __name__ == "__main__":
    test_simple_uni3d_recall() 