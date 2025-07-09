#!/usr/bin/env python3
"""
测试统一的3D网格评分系统
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from glob import glob

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_individual_scores():
    """测试各个评分函数"""
    print("🧪 测试各个评分函数...")
    
    from reward_models.rewards_mesh import (
        vertex_face_ratio_score,
        area_distribution_score,
        edge_distribution_score,
        complexity_score,
        uni3d_score,
        geometric_quality_score,
        multi_mesh_score
    )
    
    # 创建测试mesh
    test_mesh = create_test_mesh()
    test_prompt = "A simple 3D test object"
    
    # 测试各个函数
    functions = [
        ("vertex_face_ratio_score", vertex_face_ratio_score),
        ("area_distribution_score", area_distribution_score),
        ("edge_distribution_score", edge_distribution_score),
        ("complexity_score", complexity_score),
        ("geometric_quality_score", geometric_quality_score),
        ("uni3d_score", uni3d_score),
    ]
    
    for name, func in functions:
        try:
            score_fn = func("cpu")
            scores, metadata = score_fn([test_mesh], [test_prompt], {})
            score = scores[0] if isinstance(scores, list) else scores
            print(f"✅ {name}: {score:.4f}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    # 测试组合评分
    print("\n🔧 测试组合评分...")
    score_dict = {
        "vertex_face_ratio": 0.2,
        "area_distribution": 0.2,
        "edge_distribution": 0.2,
        "complexity": 0.1,
        "uni3d": 0.3,
    }
    
    try:
        multi_score_fn = multi_mesh_score("cpu", score_dict)
        scores, metadata = multi_score_fn([test_mesh], [test_prompt], {})
        print(f"✅ multi_mesh_score: {scores}")
        print(f"📊 metadata: {metadata}")
    except Exception as e:
        print(f"❌ multi_mesh_score: {str(e)}")

def create_test_mesh():
    """创建测试用的mesh"""
    from kiui.mesh import Mesh
    
    # 创建一个简单的立方体
    vertices = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 7, 6], [4, 6, 5],  # 顶面
        [0, 4, 5], [0, 5, 1],  # 前面
        [2, 6, 7], [2, 7, 3],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 5, 6], [1, 6, 2]   # 右面
    ], dtype=torch.long)
    
    return Mesh(v=vertices, f=faces)

if __name__ == "__main__":
    print("🧪 测试统一的3D网格评分系统")
    print("="*60)
    
    test_individual_scores()
    
    print("\n🎉 测试完成！") 