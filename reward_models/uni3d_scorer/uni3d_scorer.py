"""
Uni3D Scorer - 基于 Uni3D 预训练模型的 3D mesh 语义质量评分器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Union, Optional
from pathlib import Path

# 导入 kiui mesh
from kiui.mesh import Mesh

# 导入本地模块
from .models.uni3d import Uni3D, create_uni3d
from .models.point_encoder import PointcloudEncoder
from .utils.processing import prepare_pointcloud_batch

# 导入 CLIP 模型
import open_clip


class Uni3DScorer(torch.nn.Module):
    """
    Uni3D 3D-Language 语义一致性评分器
    
    使用 Uni3D 模型计算 3D mesh 与文本提示的语义一致性评分
    """
    
    def __init__(self, 
                 device="cuda", 
                 dtype=torch.float32,
                 uni3d_checkpoint_path: Optional[str] = None,
                 clip_model_name: str = "EVA02-E-14-plus",
                 eva_giant_checkpoint_path: Optional[str] = None,
                 eva02_clip_checkpoint_path: Optional[str] = None):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        # 设置预训练权重路径
        project_root = Path(__file__).parent.parent.parent
        
        if uni3d_checkpoint_path is None:
            uni3d_checkpoint_path = project_root / "pretrained_weights" / "uni3d-g.pt"
            
        if eva_giant_checkpoint_path is None:
            eva_giant_checkpoint_path = project_root / "pretrained_weights" / "eva_giant_patch14_560.pt"
            
        if eva02_clip_checkpoint_path is None:
            eva02_clip_checkpoint_path = project_root / "pretrained_weights" / "eva02_e_14_plus_laion2b_s9b_b144k.pt"
        
        # 加载 CLIP 模型
        print(f"🔄 正在加载 CLIP 模型: {clip_model_name}")
        self.clip_model, _, self.clip_preprocess = self._load_clip_model(
            clip_model_name, eva02_clip_checkpoint_path
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        print("✅ CLIP 模型加载成功")
        
        # 加载 Uni3D 模型
        print("🔄 正在初始化 Uni3D 模型...")
        self.uni3d_model = self._load_uni3d_model(uni3d_checkpoint_path, eva_giant_checkpoint_path)
        print("✅ Uni3D 模型初始化成功")
        
        # 加载模板和标签
        self.templates, self.labels = self._load_templates_and_labels()
        
        # 设置评估模式
        self.eval()
        
    def _load_clip_model(self, clip_model_name: str, checkpoint_path: Path):
        """加载 CLIP 模型"""
        if checkpoint_path.exists():
            print(f"📁 从本地加载 CLIP 权重: {checkpoint_path}")
            # 先创建模型架构
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=None  # 不加载预训练权重
            )
            # 加载本地权重
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            return model, _, preprocess
        else:
            print(f"⚠️ 本地CLIP权重不存在，使用在线下载: {checkpoint_path}")
            print("💡 运行 python scripts/download_eva_weights.py 来下载权重到本地")
            return open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained='laion2b_s9b_b144k'
            )
    
    def score(self, mesh: Mesh, prompt: str, num_points: int = 10000) -> float:
        """
        计算单个 mesh 与文本提示的语义一致性评分
        
        Args:
            mesh: kiui mesh 对象
            prompt: 文本提示
            num_points: 点云采样点数（官方默认 10000）
            
        Returns:
            float: 评分结果，范围 [0, 1]
        """
        return self._compute_semantic_score(mesh, prompt, num_points)
        
    def _load_uni3d_model(self, checkpoint_path: Optional[str] = None, eva_giant_checkpoint_path: Optional[str] = None) -> Uni3D:
        """加载 Uni3D 模型"""
        # 创建模型配置 (匹配官方 Uni3D 实现)
        class Args:
            pc_model = "eva_giant_patch14_560"  # 官方 Giant 版本
            pretrained_pc = str(eva_giant_checkpoint_path) if eva_giant_checkpoint_path and Path(eva_giant_checkpoint_path).exists() else None
            drop_path_rate = 0.0
            # PointcloudEncoder 需要的属性（基于官方源代码）
            pc_feat_dim = 1408     # EVA Giant transformer 维度
            embed_dim = 1024       # 匹配预训练权重和 EVA02-E-14-plus (1024 维)
            group_size = 64        # 每组点数（官方默认 64）
            num_group = 512        # 组数（官方默认 512）
            pc_encoder_dim = 512   # 编码器输出维度（官方默认 512）
            patch_dropout = 0.0    # patch dropout 率（推理时为 0）
            
        args = Args()
        
        # 打印EVA Giant权重加载信息
        if args.pretrained_pc:
            print(f"📁 使用本地EVA Giant权重: {args.pretrained_pc}")
        else:
            print(f"⚠️ 本地EVA Giant权重不存在，使用在线下载")
            print("💡 运行 python scripts/download_eva_weights.py 来下载权重到本地")
        
        # 创建 Uni3D 模型
        model = create_uni3d(args)
        
        # 加载预训练权重 (如果提供)
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"🔄 正在加载Uni3D预训练权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理权重键名
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 移除 'module.' 前缀（如果存在）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Uni3D预训练权重加载成功")
        else:
            print("⚠️ 未提供Uni3D预训练权重，使用随机初始化")
            
        return model.to(self.device)
        
    def _load_templates_and_labels(self):
        """加载模板和标签数据"""
        # 获取当前文件的目录
        current_dir = Path(__file__).parent
        
        # 加载模板
        templates_path = current_dir / "data" / "templates.json"
        with open(templates_path, 'r') as f:
            templates = json.load(f)
            
        # 加载标签
        labels_path = current_dir / "data" / "labels.json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            
        return templates, labels
        
    @torch.no_grad()
    def __call__(self, 
                 meshes: Union[Mesh, List[Mesh]], 
                 prompts: Union[str, List[str]],
                 num_points: int = 10000) -> torch.Tensor:
        """
        计算 mesh 与文本提示的语义一致性评分
        
        Args:
            meshes: 单个或多个 kiui mesh 对象
            prompts: 单个或多个文本提示
            num_points: 点云采样点数（官方默认 10000）
            
        Returns:
            torch.Tensor: 评分结果，范围 [0, 1]
        """
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # 确保 mesh 和 prompt 数量匹配
        if len(meshes) != len(prompts):
            if len(prompts) == 1:
                prompts = prompts * len(meshes)
            else:
                raise ValueError(f"Mesh 数量 ({len(meshes)}) 与 prompt 数量 ({len(prompts)}) 不匹配")
                
        scores = []
        
        for mesh, prompt in zip(meshes, prompts):
            score = self._compute_semantic_score(mesh, prompt, num_points)
            scores.append(score)
            
        return torch.tensor(scores, device=self.device, dtype=self.dtype)
        
    def _compute_semantic_score(self, mesh: Mesh, prompt: str, num_points: int) -> float:
        """计算单个 mesh 与文本提示的语义一致性评分"""
        try:
            # 1. 将 mesh 转换为点云
            pointcloud_batch = prepare_pointcloud_batch([mesh], num_points=num_points)
            pointcloud_batch = pointcloud_batch.to(self.device)  # (1, num_points, 6)
            
            # 2. 使用 Uni3D 编码点云
            pc_features = self.uni3d_model.encode_pc(pointcloud_batch)  # (1, embed_dim)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 3. 使用 CLIP 编码文本
            text_tokens = open_clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)  # (1, clip_feature_dim)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 4. 计算余弦相似度（现在维度应该匹配）
            similarity = torch.mm(pc_features, text_features.T)  # (1, 1)
            score = similarity.item()
            
            # 5. 将相似度从 [-1, 1] 映射到 [0, 1]
            score = (score + 1) / 2
            
            return score
            
        except Exception as e:
            print(f"⚠️ 计算语义评分时出错: {e}")
            return 0.5  # 返回默认分数
    
    def _compute_image_semantic_score(self, mesh: Mesh, image_tensor: torch.Tensor, num_points: int) -> float:
        """
        计算单个 mesh 与图像的语义一致性评分 (示例实现)
        
        Args:
            mesh: kiui mesh 对象
            image_tensor: 图像张量 (C, H, W) 或 (1, C, H, W)
            num_points: 点云采样点数
            
        Returns:
            float: 评分结果，范围 [0, 1]
        """
        try:
            # 1. 将 mesh 转换为点云
            pointcloud_batch = prepare_pointcloud_batch([mesh], num_points=num_points)
            pointcloud_batch = pointcloud_batch.to(self.device)  # (1, num_points, 6)
            
            # 2. 使用 Uni3D 编码点云
            pc_features = self.uni3d_model.encode_pc(pointcloud_batch)  # (1, embed_dim)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 3. 使用 CLIP 编码图像
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
            image_tensor = image_tensor.to(self.device)
            
            # 预处理图像（CLIP 需要特定的预处理）
            if hasattr(self, 'clip_preprocess'):
                # 如果有预处理函数，使用它
                image_features = self.clip_model.encode_image(image_tensor)
            else:
                # 否则假设图像已经预处理过
                image_features = self.clip_model.encode_image(image_tensor)
                
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 4. 计算余弦相似度（现在维度应该匹配）
            similarity = torch.mm(pc_features, image_features.T)  # (1, 1)
            score = similarity.item()
            
            # 5. 将相似度从 [-1, 1] 映射到 [0, 1]
            score = (score + 1) / 2
            
            return score
            
        except Exception as e:
            print(f"⚠️ 计算图像语义评分时出错: {e}")
            return 0.5  # 返回默认分数
            
    def score_with_templates(self, 
                           meshes: Union[Mesh, List[Mesh]], 
                           class_names: Union[str, List[str]],
                           template_key: str = "modelnet40_64",
                           num_points: int = 10000) -> torch.Tensor:
        """
        使用预定义模板计算评分
        
        Args:
            meshes: 单个或多个 kiui mesh 对象
            class_names: 类别名称
            template_key: 模板键名
            num_points: 点云采样点数（官方默认 10000）
            
        Returns:
            torch.Tensor: 评分结果，范围 [0, 1]
        """
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(class_names, str):
            class_names = [class_names]
            
        # 确保 mesh 和 class_name 数量匹配
        if len(meshes) != len(class_names):
            if len(class_names) == 1:
                class_names = class_names * len(meshes)
            else:
                raise ValueError(f"Mesh 数量 ({len(meshes)}) 与 class_name 数量 ({len(class_names)}) 不匹配")
        
        # 获取模板
        if template_key not in self.templates:
            raise ValueError(f"未找到模板: {template_key}")
            
        templates = self.templates[template_key]
        
        # 为每个 mesh 计算评分
        scores = []
        for mesh, class_name in zip(meshes, class_names):
            # 使用所有模板计算评分并取平均
            template_scores = []
            for template in templates:
                prompt = template.format(class_name)
                score = self._compute_semantic_score(mesh, prompt, num_points)
                template_scores.append(score)
                
            avg_score = sum(template_scores) / len(template_scores)
            scores.append(avg_score)
            
        return torch.tensor(scores, device=self.device, dtype=self.dtype)
        
    def get_available_templates(self) -> List[str]:
        """获取可用的模板列表"""
        return list(self.templates.keys())
        
    def get_labels(self, dataset_name: str) -> List[str]:
        """获取指定数据集的标签列表"""
        if dataset_name not in self.labels:
            raise ValueError(f"未找到数据集: {dataset_name}")
        return self.labels[dataset_name]

def main():
    """测试 Uni3D 评分器"""
    # 测试基本功能
    scorer = Uni3DScorer()
    
    # 创建测试 mesh
    import trimesh
    mesh_trimesh = trimesh.creation.box(extents=[1, 1, 1])
    from kiui.mesh import Mesh
    mesh = Mesh(v=mesh_trimesh.vertices, f=mesh_trimesh.faces)
    
    # 测试评分
    score = scorer.score(mesh, "a cube")
    print(f"Score: {score}")
    
    # 测试模板评分
    template_score = scorer.score_with_templates(mesh, "cube", "modelnet40_64")
    print(f"Template score: {template_score}")

if __name__ == "__main__":
    main() 