import torch
import numpy as np
from typing import List, Union
from kiui.mesh import Mesh


class MeshBasicScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
    def score(self, mesh: Mesh) -> float:
        """
        计算单个 kiui mesh 的基础几何质量评分
        
        Args:
            mesh: kiui mesh 对象
            
        Returns:
            float: 评分结果，范围 [0, 1]
        """
        return self._evaluate_mesh(mesh)
    
    @torch.no_grad()
    def __call__(self, meshes: Union[Mesh, List[Mesh]]) -> torch.Tensor:
        """
        计算kiui mesh的基础几何质量评分
        
        Args:
            meshes: 单个或多个 kiui mesh 对象
            
        Returns:
            torch.Tensor: 评分结果，范围 [0, 1]
        """
        if isinstance(meshes, Mesh):
            meshes = [meshes]
            
        scores = []
        for mesh in meshes:
            score = self._evaluate_mesh(mesh)
            scores.append(score)
            
        return torch.tensor(scores, device=self.device, dtype=self.dtype)
    
    def _evaluate_mesh(self, mesh: Mesh) -> float:
        """评估单个mesh的质量"""
        # 获取基础统计信息
        n_vertices = mesh.v.shape[0]
        n_faces = mesh.f.shape[0]
        
        if n_vertices == 0 or n_faces == 0:
            return 0.0
            
        # 1. 顶点面数比例评分 (期望比例约为 1:2)
        vertex_face_ratio = n_faces / n_vertices
        ratio_score = 1.0 - abs(vertex_face_ratio - 2.0) / 2.0
        ratio_score = max(0.0, min(1.0, ratio_score))
        
        # 2. 面积密度评分 (检查面积分布的一致性)
        area_score = self._compute_area_score(mesh)
        
        # 3. 边长分布评分 (检查边长的一致性)
        edge_score = self._compute_edge_score(mesh)
        
        # 4. 几何复杂度评分 (适中复杂度最佳)
        complexity_score = self._compute_complexity_score(mesh)
        
        # 加权平均
        total_score = (
            ratio_score * 0.25 +
            area_score * 0.25 +
            edge_score * 0.25 +
            complexity_score * 0.25
        )
        
        return total_score
    
    def _compute_area_score(self, mesh: Mesh) -> float:
        """计算面积分布评分"""
        try:
            # 获取所有面的面积
            vertices = mesh.v.cpu().numpy()
            faces = mesh.f.cpu().numpy()
            
            # 计算每个面的面积
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            
            # 叉积计算面积
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            if len(areas) == 0:
                return 0.0
                
            # 面积分布的一致性评分
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            # 标准差与平均值的比例，越小越好
            cv = std_area / (mean_area + 1e-8)
            area_score = 1.0 / (1.0 + cv)
            
            return area_score
            
        except Exception:
            return 0.5  # 默认分数
    
    def _compute_edge_score(self, mesh: Mesh) -> float:
        """计算边长分布评分"""
        try:
            vertices = mesh.v.cpu().numpy()
            faces = mesh.f.cpu().numpy()
            
            # 计算所有边的长度
            edges = []
            for i in range(3):
                j = (i + 1) % 3
                edge_lengths = np.linalg.norm(
                    vertices[faces[:, i]] - vertices[faces[:, j]], axis=1
                )
                edges.extend(edge_lengths)
            
            edges = np.array(edges)
            
            if len(edges) == 0:
                return 0.0
                
            # 边长分布的一致性评分
            mean_edge = np.mean(edges)
            std_edge = np.std(edges)
            
            cv = std_edge / (mean_edge + 1e-8)
            edge_score = 1.0 / (1.0 + cv)
            
            return edge_score
            
        except Exception:
            return 0.5  # 默认分数
    
    def _compute_complexity_score(self, mesh: Mesh) -> float:
        """计算几何复杂度评分"""
        try:
            n_vertices = mesh.v.shape[0]
            n_faces = mesh.f.shape[0]
            
            # 基于顶点数的复杂度评分
            # 期望范围：1k-100k顶点
            if n_vertices < 1000:
                complexity_score = n_vertices / 1000.0
            elif n_vertices > 100000:
                complexity_score = 1.0 - (n_vertices - 100000) / 100000.0
                complexity_score = max(0.0, complexity_score)
            else:
                complexity_score = 1.0
                
            return complexity_score
            
        except Exception:
            return 0.5  # 默认分数


# 测试函数
def main():
    # 创建评分器
    scorer = MeshBasicScorer(device="cuda", dtype=torch.float32)
    
    print("✅ MeshBasicScorer 初始化成功")
    print("📋 使用方法:")
    print("  - 单个 mesh: score = scorer.score(mesh)")
    print("  - 批量 mesh: scores = scorer([mesh1, mesh2, ...])")
    print("要测试评分器，请传入 kiui mesh 对象")


if __name__ == "__main__":
    main()
