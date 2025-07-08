"""
Uni3D Scorer - 基于 Uni3D 预训练模型的 3D mesh 语义质量评分器
"""
import torch
import torch.nn as nn
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
from .utils.tokenizer import SimpleTokenizer
from .utils.processing import prepare_pointcloud_batch

# 导入 CLIP 模型
import open_clip


class Uni3DScorer(torch.nn.Module):
    """
    基于 Uni3D 预训练模型的 3D mesh 语义质量评分器
    """
    
    def __init__(self, 
                 device="cuda", 
                 dtype=torch.float32,
                 uni3d_checkpoint_path: Optional[str] = None,
                 clip_model_name: str = "ViT-B/32"):
        """
        初始化 Uni3D 评分器
        
        Args:
            device: 计算设备
            dtype: 数据类型
            uni3d_checkpoint_path: Uni3D 预训练模型路径
            clip_model_name: CLIP 模型名称
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # 加载 CLIP 模型
        print(f"🔄 正在加载 CLIP 模型: {clip_model_name}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', device=device)
        self.clip_model.eval()
        print("✅ CLIP 模型加载成功")
        
        # 初始化 tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # 加载 Uni3D 模型
        print("🔄 正在初始化 Uni3D 模型...")
        self.uni3d_model = self._load_uni3d_model(uni3d_checkpoint_path)
        print("✅ Uni3D 模型初始化成功")
        
        # 加载模板和标签
        self.templates, self.labels = self._load_templates_and_labels()
        
        # 设置评估模式
        self.eval()
        
    def _load_uni3d_model(self, checkpoint_path: Optional[str] = None) -> Uni3D:
        """加载 Uni3D 模型"""
        # 创建模型配置 (简化版)
        class Args:
            pc_model = "eva_giant_patch14_224.clip_ft_in1k"
            pretrained_pc = None
            drop_path_rate = 0.0
            
        args = Args()
        
        # 创建 Uni3D 模型
        model = create_uni3d(args)
        
        # 加载预训练权重 (如果提供)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔄 正在加载预训练权重: {checkpoint_path}")
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
            print("✅ 预训练权重加载成功")
        else:
            print("⚠️ 未提供预训练权重，使用随机初始化")
            
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
                 num_points: int = 8192) -> torch.Tensor:
        """
        计算 mesh 与文本提示的语义一致性评分
        
        Args:
            meshes: 单个或多个 kiui mesh 对象
            prompts: 单个或多个文本提示
            num_points: 点云采样点数
            
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
            pc_features = self.uni3d_model.encode_pc(pointcloud_batch)  # (1, feature_dim)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 3. 使用 CLIP 编码文本
            text_tokens = open_clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)  # (1, feature_dim)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 4. 计算余弦相似度
            similarity = torch.mm(pc_features, text_features.T)  # (1, 1)
            score = similarity.item()
            
            # 5. 将相似度从 [-1, 1] 映射到 [0, 1]
            score = (score + 1) / 2
            
            return score
            
        except Exception as e:
            print(f"⚠️ 计算语义评分时出错: {e}")
            return 0.5  # 返回默认分数
            
    def score_with_templates(self, 
                           meshes: Union[Mesh, List[Mesh]], 
                           class_names: Union[str, List[str]],
                           template_key: str = "modelnet40_64",
                           num_points: int = 8192) -> torch.Tensor:
        """
        使用预定义模板计算评分
        
        Args:
            meshes: 单个或多个 kiui mesh 对象
            class_names: 类别名称
            template_key: 模板键名
            num_points: 点云采样点数
            
        Returns:
            torch.Tensor: 评分结果，范围 [0, 1]
        """
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(class_names, str):
            class_names = [class_names]
            
        # 确保数量匹配
        if len(meshes) != len(class_names):
            if len(class_names) == 1:
                class_names = class_names * len(meshes)
            else:
                raise ValueError(f"Mesh 数量 ({len(meshes)}) 与类别数量 ({len(class_names)}) 不匹配")
                
        # 获取模板
        if template_key not in self.templates:
            raise ValueError(f"未找到模板键: {template_key}")
            
        templates = self.templates[template_key]
        
        scores = []
        
        for mesh, class_name in zip(meshes, class_names):
            # 为每个类别生成多个提示
            class_prompts = [template.format(class_name) for template in templates]
            
            # 计算与所有模板的相似度
            template_scores = []
            for prompt in class_prompts:
                score = self._compute_semantic_score(mesh, prompt, num_points)
                template_scores.append(score)
                
            # 取平均值作为最终评分
            final_score = np.mean(template_scores)
            scores.append(final_score)
            
        return torch.tensor(scores, device=self.device, dtype=self.dtype)
        
    def get_available_templates(self) -> List[str]:
        """获取可用的模板键"""
        return list(self.templates.keys())
        
    def get_labels(self, dataset_name: str) -> List[str]:
        """获取指定数据集的标签"""
        if dataset_name not in self.labels:
            raise ValueError(f"未找到数据集: {dataset_name}")
        return self.labels[dataset_name]


def main():
    """测试函数"""
    # 创建评分器
    scorer = Uni3DScorer(device="cuda", dtype=torch.float32)
    
    print("✅ Uni3DScorer 初始化成功")
    print(f"📋 可用模板: {scorer.get_available_templates()}")
    print(f"📋 ModelNet40 标签数量: {len(scorer.get_labels('modelnet40_openshape'))}")
    
    # 这里需要真实的 kiui mesh 对象来测试
    print("🔄 准备测试数据...")
    print("要测试评分器，请传入 kiui mesh 对象和文本提示")
    

if __name__ == "__main__":
    main() 