"""
3D Mesh Reward Functions - 统一的3D网格评分系统
"""
import torch
import numpy as np
from typing import List, Union, Optional, Dict
from pathlib import Path
from kiui.mesh import Mesh


def vertex_face_ratio_score(device="cuda"):
    """顶点-面比例评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            try:
                n_vertices = mesh.v.shape[0] if hasattr(mesh, 'v') and mesh.v is not None else 0
                n_faces = mesh.f.shape[0] if hasattr(mesh, 'f') and mesh.f is not None else 0
                
                if n_faces == 0 or n_vertices == 0:
                    scores.append(0.0)
                    continue
                
                # 理想比例约为 2:1 (顶点:面)
                ratio = n_vertices / n_faces
                ideal_ratio = 2.0
                
                # 计算偏差评分
                deviation = abs(ratio - ideal_ratio) / ideal_ratio
                score = 1.0 / (1.0 + deviation)
                
                # 🔧 修复：检查NaN值
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
                    
                scores.append(score)
            except Exception as e:
                print(f"⚠️ vertex_face_ratio_score计算失败: {e}")
                scores.append(0.0)
                
        return scores, {}
    
    return _fn


def area_distribution_score(device="cuda"):
    """面积分布一致性评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            try:
                if not hasattr(mesh, 'v') or not hasattr(mesh, 'f') or mesh.v is None or mesh.f is None:
                    scores.append(0.0)
                    continue
                    
                vertices = mesh.v.cpu().numpy() if hasattr(mesh.v, 'cpu') else mesh.v
                faces = mesh.f.cpu().numpy() if hasattr(mesh.f, 'cpu') else mesh.f
                
                if len(faces) == 0 or len(vertices) == 0:
                    scores.append(0.0)
                    continue
                
                # 计算面积
                v0 = vertices[faces[:, 0]]
                v1 = vertices[faces[:, 1]]
                v2 = vertices[faces[:, 2]]
                cross = np.cross(v1 - v0, v2 - v0)
                areas = 0.5 * np.linalg.norm(cross, axis=1)
                
                if len(areas) == 0 or np.all(areas == 0):
                    scores.append(0.0)
                    continue
                    
                # 一致性评分
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                cv = std_area / (mean_area + 1e-8)
                area_score = 1.0 / (1.0 + cv)
                
                # 🔧 修复：检查NaN值
                if np.isnan(area_score) or np.isinf(area_score):
                    area_score = 0.0
                    
                scores.append(area_score)
            except Exception as e:
                print(f"⚠️ area_distribution_score计算失败: {e}")
                scores.append(0.0)
                
        return scores, {}
    
    return _fn


def edge_distribution_score(device="cuda"):
    """边长分布一致性评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            try:
                if not hasattr(mesh, 'v') or not hasattr(mesh, 'f') or mesh.v is None or mesh.f is None:
                    scores.append(0.0)
                    continue
                    
                vertices = mesh.v.cpu().numpy() if hasattr(mesh.v, 'cpu') else mesh.v
                faces = mesh.f.cpu().numpy() if hasattr(mesh.f, 'cpu') else mesh.f
                
                if len(faces) == 0 or len(vertices) == 0:
                    scores.append(0.0)
                    continue
                
                # 计算边长
                edges = []
                for i in range(3):
                    j = (i + 1) % 3
                    edge_lengths = np.linalg.norm(
                        vertices[faces[:, i]] - vertices[faces[:, j]], axis=1
                    )
                    edges.extend(edge_lengths)
                
                edges = np.array(edges)
                if len(edges) == 0 or np.all(edges == 0):
                    scores.append(0.0)
                    continue
                    
                # 一致性评分
                mean_edge = np.mean(edges)
                std_edge = np.std(edges)
                cv = std_edge / (mean_edge + 1e-8)
                edge_score = 1.0 / (1.0 + cv)
                
                # 🔧 修复：检查NaN值
                if np.isnan(edge_score) or np.isinf(edge_score):
                    edge_score = 0.0
                    
                scores.append(edge_score)
            except Exception as e:
                print(f"⚠️ edge_distribution_score计算失败: {e}")
                scores.append(0.0)
                
        return scores, {}
    
    return _fn


def complexity_score(device="cuda"):
    """几何复杂度评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            try:
                if not hasattr(mesh, 'v') or mesh.v is None:
                    scores.append(0.0)
                    continue
                    
                n_vertices = mesh.v.shape[0]
                
                # 期望范围：1k-100k顶点
                if n_vertices < 1000:
                    score = n_vertices / 1000.0
                elif n_vertices > 100000:
                    score = 1.0 - (n_vertices - 100000) / 100000.0
                    score = max(0.0, score)
                else:
                    score = 1.0
                
                # 🔧 修复：检查NaN值
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
                    
                scores.append(score)
            except Exception as e:
                print(f"⚠️ complexity_score计算失败: {e}")
                scores.append(0.0)
                
        return scores, {}
    
    return _fn


def uni3d_score(device="cuda", use_image=True):
    """基于Uni3D的语义对齐评分函数 - 支持图像输入"""
    from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
    
    # 使用现有的Uni3DScorer，它知道如何正确加载本地权重
    scorer = Uni3DScorer(device=device)
    
    @torch.no_grad()
    def _fn(meshes, prompts, metadata, images=None):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        
        if use_image and images is not None:
            # 🔧 使用图像模式
            if isinstance(images, (str, Path)):
                images = [images]
            
            if len(meshes) != len(images):
                if len(images) == 1:
                    images = images * len(meshes)
                else:
                    raise ValueError(f"Mesh数量与图像数量不匹配: {len(meshes)} vs {len(images)}")
            
            for mesh, image_path in zip(meshes, images):
                try:
                    # 加载和预处理图像
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    image = Image.open(image_path).convert("RGB")
                    preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = preprocess(image)
                    
                    # 🔧 使用图像语义评分
                    score = scorer._compute_image_semantic_score(mesh, image_tensor, num_points=10000)
                    scores.append(score)
                    
                except Exception as e:
                    print(f"⚠️ 图像语义评分失败 ({image_path}): {e}")
                    scores.append(0.5)  # 默认分数
        else:
            # 🔧 回退到文本模式（如果需要）
            if isinstance(prompts, str):
                prompts = [prompts]
                
            if len(meshes) != len(prompts):
                if len(prompts) == 1:
                    prompts = prompts * len(meshes)
                else:
                    raise ValueError(f"Mesh数量与prompt数量不匹配")
                    
            for mesh, prompt in zip(meshes, prompts):
                # 使用现有的scorer来计算评分
                score = scorer.score(mesh, prompt)
                scores.append(score)
                
        return scores, {}
    
    return _fn


def geometric_quality_score(device="cuda"):
    """几何质量综合评分函数"""
    vertex_face_fn = vertex_face_ratio_score(device)
    area_dist_fn = area_distribution_score(device)
    edge_dist_fn = edge_distribution_score(device)
    complexity_fn = complexity_score(device)
    
    def _fn(meshes, prompts, metadata):
        vertex_face_scores, _ = vertex_face_fn(meshes, prompts, metadata)
        area_dist_scores, _ = area_dist_fn(meshes, prompts, metadata)
        edge_dist_scores, _ = edge_dist_fn(meshes, prompts, metadata)
        complexity_scores, _ = complexity_fn(meshes, prompts, metadata)
        
        total_scores = []
        for vf, ad, ed, c in zip(vertex_face_scores, area_dist_scores, 
                               edge_dist_scores, complexity_scores):
            # 🔧 修复：处理NaN值，确保总是返回有效分数
            score = (vf + ad + ed + c) / 4
            if np.isnan(score) or np.isinf(score):
                score = 0.0  # 如果计算结果是NaN或Inf，返回0分
            total_scores.append(score)
        
        return total_scores, {}
    
    return _fn


def multi_mesh_score(device, score_dict):
    """多维度mesh评分函数"""
    score_functions = {
        "vertex_face_ratio": vertex_face_ratio_score,
        "area_distribution": area_distribution_score,
        "edge_distribution": edge_distribution_score,
        "complexity": complexity_score,
        "uni3d": lambda device: uni3d_score(device, use_image=True),  # 🔧 启用图像模式
        "geometric_quality": geometric_quality_score,
    }
    
    score_fns = {}
    # 🚀 显存优化：只加载权重不为0的评分函数，避免加载不需要的大型模型
    for score_name, weight in score_dict.items():
        if weight > 0:  # 只加载权重大于0的评分函数
            print(f"🔄 加载评分函数: {score_name} (权重: {weight})")
            score_fns[score_name] = score_functions[score_name](device)
        else:
            print(f"⏭️  跳过评分函数: {score_name} (权重: {weight}，已禁用)")
    
    def _fn(meshes, prompts, metadata, images=None):  # 🔧 新增 images 参数
        total_scores = []
        score_details = {}
        
        # 🚀 显存优化：只计算权重大于0的评分
        for score_name, weight in score_dict.items():
            if weight > 0 and score_name in score_fns:
                # 🔧 传递 images 参数
                if score_name == "uni3d":
                    # uni3d_score 需要 images 参数
                    scores, _ = score_fns[score_name](meshes, prompts, metadata, images)
                else:
                    # 其他评分函数不需要 images 参数
                    scores, _ = score_fns[score_name](meshes, prompts, metadata)
                
                score_details[score_name] = scores
                weighted_scores = [weight * score for score in scores]
                
                if not total_scores:
                    total_scores = weighted_scores
                else:
                    total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
            elif weight == 0:
                # 权重为0的评分，设置为0分
                score_details[score_name] = [0.0] * len(meshes)
        
        # 🔧 修复：确保返回格式与trainer期望一致
        score_details['avg'] = total_scores
        return score_details, {}
    
    return _fn


def main():
    """测试函数"""
    import trimesh
    from kiui.mesh import Mesh
    
    # 创建测试mesh
    mesh_trimesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = Mesh(v=mesh_trimesh.vertices, f=mesh_trimesh.faces)
    
    # 测试配置
    score_dict = {
        "geometric_quality": 0.3,
        "uni3d": 0.7
    }
    
    # 测试评分
    device = "cuda"
    scoring_fn = multi_mesh_score(device, score_dict)
    scores, _ = scoring_fn([mesh], ["a cube"], {}, images="path/to/image.jpg") # 🔧 提供图像路径
    
    print("Scores:", scores)


if __name__ == "__main__":
    main() 