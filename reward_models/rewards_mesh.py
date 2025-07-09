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
            n_vertices = mesh.v.shape[0]
            n_faces = mesh.f.shape[0]
            
            if n_faces == 0:
                scores.append(0.0)
                continue
            
            # 理想比例约为 2:1 (顶点:面)
            ratio = n_vertices / n_faces
            ideal_ratio = 2.0
            
            # 计算偏差评分
            deviation = abs(ratio - ideal_ratio) / ideal_ratio
            score = 1.0 / (1.0 + deviation)
            scores.append(score)
                
        return scores, {}
    
    return _fn


def area_distribution_score(device="cuda"):
    """面积分布一致性评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            vertices = mesh.v.cpu().numpy()
            faces = mesh.f.cpu().numpy()
            
            if len(faces) == 0:
                scores.append(0.0)
                continue
            
            # 计算面积
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            if len(areas) == 0:
                scores.append(0.0)
                continue
                
            # 一致性评分
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv = std_area / (mean_area + 1e-8)
            area_score = 1.0 / (1.0 + cv)
            scores.append(area_score)
                
        return scores, {}
    
    return _fn


def edge_distribution_score(device="cuda"):
    """边长分布一致性评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            vertices = mesh.v.cpu().numpy()
            faces = mesh.f.cpu().numpy()
            
            if len(faces) == 0:
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
            if len(edges) == 0:
                scores.append(0.0)
                continue
                
            # 一致性评分
            mean_edge = np.mean(edges)
            std_edge = np.std(edges)
            cv = std_edge / (mean_edge + 1e-8)
            edge_score = 1.0 / (1.0 + cv)
            scores.append(edge_score)
                
        return scores, {}
    
    return _fn


def complexity_score(device="cuda"):
    """几何复杂度评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            n_vertices = mesh.v.shape[0]
            
            # 期望范围：1k-100k顶点
            if n_vertices < 1000:
                score = n_vertices / 1000.0
            elif n_vertices > 100000:
                score = 1.0 - (n_vertices - 100000) / 100000.0
                score = max(0.0, score)
            else:
                score = 1.0
                
            scores.append(score)
                
        return scores, {}
    
    return _fn


def uni3d_score(device="cuda"):
    """基于Uni3D的语义对齐评分函数"""
    from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
    
    # 使用现有的Uni3DScorer，它知道如何正确加载本地权重
    scorer = Uni3DScorer(device=device)
    
    @torch.no_grad()
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if len(meshes) != len(prompts):
            if len(prompts) == 1:
                prompts = prompts * len(meshes)
            else:
                raise ValueError(f"Mesh数量与prompt数量不匹配")
                
        scores = []
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
        
        total_scores = [
            (vf + ad + ed + c) / 4 
            for vf, ad, ed, c in zip(vertex_face_scores, area_dist_scores, 
                                   edge_dist_scores, complexity_scores)
        ]
        
        return total_scores, {}
    
    return _fn


def multi_mesh_score(device, score_dict):
    """多维度mesh评分函数"""
    score_functions = {
        "vertex_face_ratio": vertex_face_ratio_score,
        "area_distribution": area_distribution_score,
        "edge_distribution": edge_distribution_score,
        "complexity": complexity_score,
        "uni3d": uni3d_score,
        "geometric_quality": geometric_quality_score,
    }
    
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device)
    
    def _fn(meshes, prompts, metadata):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](meshes, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
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
    scores, _ = scoring_fn([mesh], ["a cube"], {})
    
    print("Scores:", scores)


if __name__ == "__main__":
    main() 