#!/usr/bin/env python3
"""
测试 Uni3D 评分器的有效性
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from glob import glob
import trimesh
from kiui.mesh import Mesh

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_glb_as_kiui(glb_path):
    """将 .glb 文件加载为 kiui mesh"""
    try:
        # 用 trimesh 加载
        trimesh_obj = trimesh.load(glb_path)
        
        # 如果是 Scene，提取主要的 mesh
        if isinstance(trimesh_obj, trimesh.Scene):
            meshes = []
            for name, geom in trimesh_obj.geometry.items():
                if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                    meshes.append(geom)
            
            if not meshes:
                return None
            
            # 选择顶点数最多的 mesh
            trimesh_obj = max(meshes, key=lambda m: len(m.vertices))
        
        # 转换为 kiui mesh
        vertices = torch.tensor(trimesh_obj.vertices, dtype=torch.float32)
        faces = torch.tensor(trimesh_obj.faces, dtype=torch.long)
        
        return Mesh(v=vertices, f=faces)
        
    except Exception as e:
        print(f"❌ 加载 {glb_path} 失败: {e}")
        return None

def test_uni3d_scorer():
    """测试 Uni3D 评分器"""
    print("🚀 开始测试 Uni3D 评分器...")
    
    try:
        from reward_models.uni3d_scorer import Uni3DScorerSimple
        print("✅ Uni3DScorerSimple 导入成功")
    except ImportError as e:
        print(f"❌ Uni3DScorerSimple 导入失败: {e}")
        return False
    
    # 初始化评分器
    print("🔄 正在初始化 Uni3D 评分器...")
    try:
        scorer = Uni3DScorerSimple(device="cuda", dtype=torch.float32)
        print("✅ Uni3D 评分器初始化成功")
    except Exception as e:
        print(f"❌ Uni3D 评分器初始化失败: {e}")
        return False
    
    # 查看可用的模板和标签
    print(f"📋 可用模板: {scorer.get_available_templates()}")
    print(f"📋 ModelNet40 标签: {scorer.get_labels('modelnet40_openshape')[:10]}...")
    
    # 加载测试数据
    mesh_dir = "dataset/eval3d/meshes"
    glb_files = glob(os.path.join(mesh_dir, "*.glb"))[:5]  # 只测试前5个
    print(f"📁 找到 {len(glb_files)} 个测试文件")
    
    # 测试1: 单个 mesh 与文本提示的评分
    print("\n🔍 测试1: 单个 mesh 与文本提示的语义相似度...")
    test_mesh = load_glb_as_kiui(glb_files[0])
    if test_mesh is None:
        print("❌ 无法加载测试 mesh")
        return False
    
    # 测试不同类型的提示
    prompts = [
        "a chair",
        "a table", 
        "a car",
        "a sofa",
        "a airplane"
    ]
    
    print(f"📝 测试提示: {prompts}")
    
    try:
        # 分别计算每个提示的评分
        for prompt in prompts:
            score = scorer(test_mesh, prompt)
            print(f"  '{prompt}': {score.item():.4f}")
            
        print("✅ 单个 mesh 语义评分成功")
    except Exception as e:
        print(f"❌ 语义评分失败: {e}")
        return False
    
    # 测试2: 批量 mesh 评分
    print("\n🔍 测试2: 批量 mesh 评分...")
    meshes = []
    mesh_names = []
    
    for glb_file in glb_files:
        mesh = load_glb_as_kiui(glb_file)
        if mesh is not None:
            meshes.append(mesh)
            mesh_names.append(os.path.basename(glb_file))
    
    if len(meshes) == 0:
        print("❌ 没有成功加载的 mesh")
        return False
    
    # 使用统一的提示
    batch_prompt = "a 3D object"
    
    try:
        batch_scores = scorer(meshes, batch_prompt)
        print(f"✅ 批量评分成功，处理了 {len(meshes)} 个 mesh")
        
        # 显示结果
        print("\n📊 批量评分结果:")
        for name, score in zip(mesh_names, batch_scores):
            print(f"  {name}: {score.item():.4f}")
        
        print(f"\n📈 统计信息:")
        print(f"  平均评分: {batch_scores.mean().item():.4f}")
        print(f"  标准差: {batch_scores.std().item():.4f}")
        print(f"  最高评分: {batch_scores.max().item():.4f}")
        print(f"  最低评分: {batch_scores.min().item():.4f}")
        
    except Exception as e:
        print(f"❌ 批量评分失败: {e}")
        return False
    
    # 测试3: 使用模板评分
    print("\n🔍 测试3: 使用预定义模板评分...")
    try:
        # 使用 ModelNet40 的一些类别
        test_classes = ["chair", "table", "airplane", "car", "sofa"]
        
        for class_name in test_classes:
            template_scores = scorer.score_with_templates(
                meshes[0], 
                class_name, 
                template_key="modelnet40_64"
            )
            print(f"  {class_name}: {template_scores.item():.4f}")
            
    except Exception as e:
        print(f"❌ 模板评分失败: {e}")
        return False
    
    print("\n🎉 Uni3D 评分器测试完成！")
    return True

if __name__ == "__main__":
    success = test_uni3d_scorer()
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1) 