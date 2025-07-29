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


def multi_mesh_score(device, score_dict: dict):
    """
    多维度网格评分函数 - 支持动态内存管理
    
    支持的评分函数：
    - geometric_quality: 几何质量评分 (顶点/面比例, 面积分布, 边长分布, 复杂度)
    - uni3d: Uni3D语义一致性评分
    - complexity: 网格复杂度评分
    """
    
    score_functions = {
        "geometric_quality": geometric_quality_score,
        "uni3d": uni3d_score,
        "complexity": complexity_score,
    }
    
    score_fns = {}
    
    # 🚀 显存优化：只加载权重不为0的评分函数，避免加载不需要的大型模型
    for score_name, weight in score_dict.items():
        if weight > 0:  # 只加载权重大于0的评分函数
            print(f"🔄 加载评分函数: {score_name} (权重: {weight})")
            if score_name == "uni3d":
                # 🔧 FIX: 为避免初始化时OOM，先在CPU上创建Uni3DScorer对象
                # 直接创建 Uni3DScorer 对象，强制在 CPU 上初始化
                base_scorer = Uni3DScorer(device="cpu")  # 关键修改：先在CPU上初始化
                score_fns[score_name] = DynamicGPUOffloadWrapper(base_scorer, device)
            else:
                score_fns[score_name] = score_functions[score_name](device)
        else:
            print(f"⏭️  跳过评分函数: {score_name} (权重: {weight}，已禁用)")
    
    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(meshes, prompts, metadata, images=None):  # 🔧 新增 images 参数
        total_scores = []
        score_details = {}
        
        # 只遍历已加载（权重>0）的评分函数
        for score_name, weight in score_dict.items():
            if score_name not in score_fns:
                continue # 跳过未加载的函数
            
            # 🔧 适配新的元组返回格式
            if score_name == "uni3d":
                # uni3d 评分器支持 images 参数
                scores, _ = score_fns[score_name](meshes, prompts, metadata, images=images)
            else:
                # 其他评分函数不需要 images 参数
                scores, _ = score_fns[score_name](meshes, prompts, metadata)
            
            score_details[score_name] = scores
            
            # 加权求和
            if total_scores == []:
                total_scores = [s * weight for s in scores]
            else:
                for i in range(len(scores)):
                    total_scores[i] += scores[i] * weight
        
        # 如果没有任何评分函数，返回零分
        if total_scores == []:
            total_scores = [0.0] * len(meshes)
        
        # 添加平均分
        score_details["avg"] = total_scores
        
        # 🔧 对齐 SD3 train_sd3.py：返回 (score_details, metadata) 元组
        return score_details, {}
    
    return _fn


class DynamicGPUOffloadWrapper:
    """
    动态 GPU/CPU 内存管理包装器
    
    工作原理：
    1. 初始时模型在 CPU 上
    2. 调用时自动移到 GPU
    3. 完成后立即 offload 回 CPU
    """
    
    def __init__(self, scorer, target_device):
        self.scorer = scorer
        self.target_device = target_device
        self.cpu_device = torch.device("cpu")
        
        # 🔧 更新：由于 Uni3DScorer 现在已在 CPU 上初始化，无需再次 offload
        print(f"✅ Uni3D 模型已在 CPU 上初始化，动态内存管理已就绪")
        
    def _offload_to_cpu(self):
        """将模型移动到 CPU"""
        # 🔧 安全检查：只有当模型不在 CPU 上时才移动
        if next(self.scorer.uni3d_model.parameters()).device != self.cpu_device:
            self.scorer.uni3d_model = self.scorer.uni3d_model.to(self.cpu_device)
        if next(self.scorer.clip_model.parameters()).device != self.cpu_device:
            self.scorer.clip_model = self.scorer.clip_model.to(self.cpu_device)
        
        # 🔧 FIX: 同步更新 scorer 的 device 属性
        self.scorer.device = self.cpu_device
        
    def _load_to_gpu(self):
        """将模型移动到 GPU"""
        print(f"🔄 将 Uni3D 模型加载到 GPU 进行评分...")
        # 🔧 安全检查：只有当模型不在目标设备上时才移动
        if next(self.scorer.uni3d_model.parameters()).device != self.target_device:
            self.scorer.uni3d_model = self.scorer.uni3d_model.to(self.target_device)
        if next(self.scorer.clip_model.parameters()).device != self.target_device:
            self.scorer.clip_model = self.scorer.clip_model.to(self.target_device)
        
        # 🔧 FIX: 更新 scorer 的 device 属性，确保内部数据移动使用正确的设备
        self.scorer.device = self.target_device
        
    def __call__(self, meshes, prompts, metadata, images=None):
        """
        执行评分时的动态内存管理
        """
        try:
            # 1. 加载到 GPU
            self._load_to_gpu()
            
            # 2. 执行评分 - 适配 Uni3DScorer 对象的接口
            if isinstance(meshes, Mesh):
                meshes = [meshes]
            
            scores = []
            
            # 使用图像模式评分
            if images is not None:
                if isinstance(images, (str, os.PathLike)):
                    images = [images]
                    
                for mesh, image_path in zip(meshes, images):
                    # 加载和预处理图像
                    image = Image.open(image_path).convert("RGB")
                    preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = preprocess(image)
                    
                    # 使用图像语义评分（设备移动在方法内部处理）
                    score = self.scorer._compute_image_semantic_score(mesh, image_tensor, num_points=10000)
                    scores.append(score)
            else:
                # 文本模式
                if isinstance(prompts, str):
                    prompts = [prompts]
                    
                for mesh, prompt in zip(meshes, prompts):
                    score = self.scorer.score(mesh, prompt)
                    scores.append(score)
            
            return scores, {}
            
        finally:
            # 3. 无论成功失败，都要 offload 回 CPU
            print(f"🔄 评分完成，将 Uni3D 模型 offload 回 CPU...")
            self._offload_to_cpu()
            
            # 4. 强制清理 GPU 缓存
            torch.cuda.empty_cache()
            
            # 🔧 NEW: 增强稳定性措施
            # 强制同步 CUDA 操作，确保所有操作完成
            torch.cuda.synchronize()
            
            # 更激进的内存清理
            import gc
            gc.collect()
            
            print(f"✅ Uni3D 模型已 offload 回 CPU，GPU 内存已释放")


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