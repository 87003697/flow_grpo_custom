#!/usr/bin/env python3
"""
批量评分 mesh 数据集测试
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

def batch_score_meshes():
    """批量评分 mesh 数据集"""
    print("🚀 开始批量评分 mesh 数据集...")
    
    # 初始化评分器
    from reward_models.mesh_basic_scorer import MeshBasicScorer
    scorer = MeshBasicScorer(device="cuda", dtype=torch.float32)
    print("✅ 评分器初始化成功")
    
    # 加载所有 mesh 文件
    mesh_dir = "dataset/eval3d/meshes"
    glb_files = glob(os.path.join(mesh_dir, "*.glb"))
    print(f"📁 找到 {len(glb_files)} 个 .glb 文件")
    
    # 批量加载为 kiui mesh
    print("🔄 批量加载 mesh...")
    meshes = []
    names = []
    
    for glb_file in glb_files:
        mesh_name = os.path.basename(glb_file).replace('_textured_frame_000000.glb', '')
        kiui_mesh = load_glb_as_kiui(glb_file)
        
        if kiui_mesh is not None:
            meshes.append(kiui_mesh)
            names.append(mesh_name)
            print(f"  ✅ {mesh_name}")
        else:
            print(f"  ❌ {mesh_name}")
    
    print(f"✅ 成功加载 {len(meshes)} 个 mesh")
    
    # 批量评分
    print("\n🎯 批量评分中...")
    scores = scorer(meshes)
    print("✅ 批量评分完成")
    
    # 显示结果
    print("\n📊 评分结果:")
    print("="*60)
    
    results = []
    for i, (name, mesh) in enumerate(zip(names, meshes)):
        score = scores[i].item()
        n_vertices = mesh.v.shape[0]
        n_faces = mesh.f.shape[0]
        
        results.append({
            'name': name,
            'score': score,
            'vertices': n_vertices,
            'faces': n_faces
        })
        
        print(f"{name:<25} 评分: {score:.4f} 顶点: {n_vertices:,} 面: {n_faces:,}")
    
    # 按评分排序
    print("\n🏆 按评分排序:")
    print("="*60)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results):
        print(f"{i+1:2d}. {result['name']:<25} {result['score']:.4f}")
    
    # 统计信息
    scores_list = [r['score'] for r in results]
    print(f"\n📈 统计信息:")
    print(f"评分范围: {min(scores_list):.4f} - {max(scores_list):.4f}")
    print(f"平均评分: {np.mean(scores_list):.4f}")
    print(f"标准差: {np.std(scores_list):.4f}")
    
    print(f"\n🎉 批量评分完成！")

def load_glb_as_kiui(glb_path):
    """将 .glb 文件加载为 kiui mesh"""
    import trimesh
    from kiui.mesh import Mesh
    
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

if __name__ == "__main__":
    batch_score_meshes() 