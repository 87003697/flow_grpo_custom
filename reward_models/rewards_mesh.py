"""
3D Mesh 奖励函数 - Hunyuan3D 专用
用于计算生成的3D网格的质量评分
"""

import torch
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
import time
from kiui.mesh import Mesh
from PIL import Image
import torchvision.transforms as transforms

# 导入评分函数
from .uni3d_scorer.uni3d_scorer import Uni3DScorer

# 🚀 全局单例模式: 创建一个模块级的全局缓存来存储评分器实例
# 这样可以确保在整个训练过程中，模型只被加载一次
_CACHED_SCORERS = {}

def vertex_face_ratio_score(device="cuda"):
    """顶点-面比例评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for mesh in meshes:
            n_vertices = mesh.v.shape[0]
            n_faces = mesh.f.shape[0]
            
            # 理想比例约为 2:1 (顶点:面)
            ratio = n_vertices / n_faces
            ideal_ratio = 2.0
            
            # 计算偏差评分
            deviation = abs(ratio - ideal_ratio) / ideal_ratio
            score = 1.0 / (1.0 + deviation)
            
            scores.append(score)
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
        return scores, {}
    
    return _fn


def area_distribution_score(device="cuda"):
    """面积分布一致性评分函数"""
    def _fn(meshes, prompts, metadata):
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        
        scores = []
        for i, mesh in enumerate(meshes):
            vertices = mesh.v.cpu().numpy()
            faces = mesh.f.cpu().numpy()
            
            # 计算面积
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            # 一致性评分
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv = std_area / (mean_area + 1e-8)
            area_score = 1.0 / (1.0 + cv)
            
            scores.append(area_score)
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
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
            
            # 计算边长
            edges = []
            for i in range(3):
                j = (i + 1) % 3
                edge_lengths = np.linalg.norm(
                    vertices[faces[:, i]] - vertices[faces[:, j]], axis=1
                )
                edges.extend(edge_lengths)
            
            edges = np.array(edges)
            
            # 一致性评分
            mean_edge = np.mean(edges)
            std_edge = np.std(edges)
            cv = std_edge / (mean_edge + 1e-8)
            edge_score = 1.0 / (1.0 + cv)
            
            scores.append(edge_score)
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
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
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
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
            if isinstance(images, (str, os.PathLike)):
                images = [images]
            

            
            for mesh, image_path in zip(meshes, images):
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
        else:
            # 文本模式
            if isinstance(prompts, str):
                prompts = [prompts]
                
            for mesh, prompt in zip(meshes, prompts):
                # 使用现有的scorer来计算评分
                score = scorer.score(mesh, prompt)
                scores.append(score)
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
        return scores, {}
    
    return _fn


def geometric_quality_score(device="cuda"):
    """几何质量综合评分函数"""
    vertex_face_fn = vertex_face_ratio_score(device)
    area_dist_fn = area_distribution_score(device)
    edge_dist_fn = edge_distribution_score(device)
    complexity_fn = complexity_score(device)
    
    def _fn(meshes, prompts, metadata):
        # 🔧 适配新的元组返回格式
        vertex_face_scores, _ = vertex_face_fn(meshes, prompts, metadata)
        area_dist_scores, _ = area_dist_fn(meshes, prompts, metadata)
        edge_dist_scores, _ = edge_dist_fn(meshes, prompts, metadata)
        complexity_scores, _ = complexity_fn(meshes, prompts, metadata)
        
        # 计算平均分
        total_scores = []
        for vf, ad, ed, c in zip(vertex_face_scores, area_dist_scores, 
                               edge_dist_scores, complexity_scores):
            score = (vf + ad + ed + c) / 4
            total_scores.append(score)
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (scores, metadata) 元组
        return total_scores, {}
    
    return _fn


def multi_mesh_score(meshes, images, metadata, score_fns_cfg):
    """计算多个评分函数的加权和 - 🚀 超高效版本，只支持图像模式"""
    
    if len(score_fns_cfg) == 0:
        return {"avg": np.zeros(len(meshes))}, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    score_fns = {}
    
    for score_name, weight in score_fns_cfg.items():
        if weight == 0.0:
            continue
        
        # 从缓存获取评分器，如果没有就报错
        if score_name not in _CACHED_SCORERS:
            raise RuntimeError(
                f"评分器 '{score_name}' 未找到! 请先调用 preload_scorers 初始化。"
            )
        
        score_fns[score_name] = _CACHED_SCORERS[score_name]

    # 计算评分
    score_dict = {}
    debug_info = {}
    
    for score_name, score_fn in score_fns.items():
        weight = score_fns_cfg[score_name]
        if weight == 0.0:
            continue
            
        # 移除 try/except，让错误直接抛出以进行调试
        if score_name == "uni3d":
            scores, dbg = score_fn(meshes, images, metadata)
        else:
            # 其他评分函数暂不支持图像模式，返回默认分数
            scores = [0.5] * len(meshes)
            dbg = {"warning": f"{score_name} 暂不支持图像模式"}
            
        score_dict[score_name] = np.array(scores) * weight
        debug_info[score_name] = dbg
    
    # 计算加权平均
    if score_dict:
        avg_scores = sum(score_dict.values())
    else:
        avg_scores = np.zeros(len(meshes))
    
    # 添加平均分
    score_dict["avg"] = avg_scores
    
    return score_dict, debug_info


def preload_scorers(score_fns_cfg: Dict[str, float], device: torch.device):
    """预加载所有评分器到指定设备"""
    for score_name, weight in score_fns_cfg.items():
        if weight == 0.0:
            continue
        
        if score_name not in _CACHED_SCORERS:
            if score_name == "uni3d":
                if "uni3d" not in _CACHED_SCORERS:
                    print(f"🔧 正在预加载 Uni3D 评分器到 {device} (常驻模式)...")
                    # 🔧 修改：禁用动态卸载，让模型常驻GPU
                    _CACHED_SCORERS[score_name] = Uni3DScorer(device=device, enable_dynamic_offload=False)

# 🔧 废弃：这个函数不再需要，因为模型将常驻GPU
# def set_scorers_phase(phase: str, accelerator=None):
#     """设置所有评分器的设备位置: 'gpu' 或 'cpu'"""
#     for scorer in _CACHED_SCORERS.values():
#         if hasattr(scorer, 'set_phase'):
#             scorer.set_phase(phase, accelerator)


def main():
    """测试函数"""
    import trimesh
    from kiui.mesh import Mesh
    
    # 创建测试mesh
    mesh_trimesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = Mesh(v=mesh_trimesh.vertices, f=mesh_trimesh.faces)
    
    # 测试配置
    score_dict = {
        "uni3d": 1.0
    }
    
    # 第一次调用
    print("\n--- 第一次调用 ---")
    scores1, _ = multi_mesh_score([mesh], ["path/to/image.jpg"], {}, score_dict)
    print("Scores 1:", scores1)

    # 第二次调用，应该复用缓存
    print("\n--- 第二次调用 ---")
    scores2, _ = multi_mesh_score([mesh], ["path/to/image.jpg"], {}, score_dict)
    print("Scores 2:", scores2)


if __name__ == "__main__":
    main() 