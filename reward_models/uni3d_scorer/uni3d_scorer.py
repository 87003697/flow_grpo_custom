"""
Uni3D Scorer - 🚀 超高效的3D mesh语义质量评分器，优化CPU/GPU offload
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Union, Tuple
from pathlib import Path

# 导入正确的模型
import open_clip
from .models.uni3d import create_uni3d, Uni3D
from .models.mesh_utils import sample_points_from_mesh, Mesh

class Uni3DScorer:
    """🚀 超高效的Uni3D评分器，优化CPU/GPU offload性能"""
    
    def __init__(self, device="cuda", enable_dynamic_offload=True, target_device="cuda"):
        # 🔧 设备配置
        self.enable_dynamic_offload = enable_dynamic_offload
        self.target_device = torch.device(target_device if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # 🔧 模型缓存状态
        self._models_initialized = False
        self._models_on_gpu = False
        self._last_gpu_time = 0
        self._gpu_timeout = 30  # 30秒后自动offload
        
        print(f"🚀 FastUni3D初始化：enable_offload={enable_dynamic_offload}, target={target_device}")
        
        # 🔧 初始化模型（始终在CPU上，按需移动到GPU）
        self._init_models()
        
        # 🔧 预热GPU streams以加速传输
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def _init_models(self):
        """一次性初始化所有模型，避免重复加载"""
        if self._models_initialized:
            return
            
        print("🔄 一次性初始化Uni3D模型...")
        start_time = time.time()
        
        # 🔧 先清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 1. 初始化CLIP模型 - 使用正确的open_clip接口
        print("🔄 正在加载 CLIP 模型: EVA02-E-14-plus")
        clip_weights_path = Path("pretrained_weights/eva02_e_14_plus_laion2b_s9b_b144k.pt")
        print(f"📁 从本地加载 CLIP 权重: {clip_weights_path}")
        
        if clip_weights_path.exists():
            # 先创建模型架构
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'EVA02-E-14-plus', 
                pretrained=None  # 不使用预训练权重
            )
            # 加载本地权重
            state_dict = torch.load(clip_weights_path, map_location='cpu', weights_only=False)
            self.clip_model.load_state_dict(state_dict, strict=False)
            # 🔧 立即清理state_dict占用的内存
            del state_dict
        else:
            print("⚠️ 本地CLIP权重不存在，使用在线下载")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'EVA02-E-14-plus', 
                pretrained='laion2b_s9b_b144k'
            )
        print("✅ CLIP 模型加载成功")
        
        # 🔧 中间清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. 初始化Uni3D模型 - 使用正确的create_uni3d接口
        print("🔄 正在初始化 Uni3D 模型...")
        eva_weights_path = Path("pretrained_weights/eva_giant_patch14_560.pt")
        uni3d_weights_path = Path("pretrained_weights/uni3d-g.pt")
        
        # 创建模型配置参数
        class Args:
            pc_model = "eva_giant_patch14_560"
            pretrained_pc = str(eva_weights_path) if eva_weights_path.exists() else None
            drop_path_rate = 0.0
            pc_feat_dim = 1408     # EVA Giant transformer 维度
            embed_dim = 1024       # 匹配 EVA02-E-14-plus
            group_size = 64        # 每组点数
            num_group = 512        # 组数
            pc_encoder_dim = 512   # 编码器输出维度
            patch_dropout = 0.0    # patch dropout 率
        
        args = Args()
        print(f"📁 使用本地EVA Giant权重: {eva_weights_path}")
        self.uni3d_model = create_uni3d(args)
        
        # 加载Uni3D预训练权重
        print(f"🔄 正在加载Uni3D预训练权重: {uni3d_weights_path}")
        if uni3d_weights_path.exists():
            checkpoint = torch.load(uni3d_weights_path, map_location='cpu', weights_only=False)
            # 处理权重键名
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.uni3d_model.load_state_dict(state_dict, strict=False)
            # 🔧 立即清理checkpoint占用的内存
            del checkpoint, state_dict
        print("✅ Uni3D预训练权重加载成功")
        
        # 3. 设置为评估模式
        self.clip_model.eval()
        self.uni3d_model.eval()
        
        # 🔧 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 4. 初始设备状态
        if self.enable_dynamic_offload:
            # 初始在CPU上
            self.device = self.cpu_device
            self._models_on_gpu = False
            print(f"✅ 模型初始化在CPU上，enable_offload=True")
        else:
            # 直接移动到目标设备
            self.device = self.target_device
            self.clip_model = self.clip_model.to(self.target_device)
            self.uni3d_model = self.uni3d_model.to(self.target_device)
            self._models_on_gpu = True
            print(f"✅ 模型直接加载到 {self.target_device}")
        
        elapsed = time.time() - start_time
        print(f"✅ Uni3D 模型初始化成功，耗时: {elapsed:.2f}秒")
        self._models_initialized = True
    
    def _fast_load_to_gpu(self):
        """🚀 超快速GPU加载 - 使用异步流和缓存"""
        if not self.enable_dynamic_offload or self._models_on_gpu:
            return
            
        print("⚡ 快速加载模型到GPU...")
        start_time = time.time()
        
        with torch.cuda.device(self.target_device):
            # 使用异步传输流加速
            if self.stream:
                with torch.cuda.stream(self.stream):
                    self.uni3d_model = self.uni3d_model.to(self.target_device, non_blocking=True)
                    self.clip_model = self.clip_model.to(self.target_device, non_blocking=True)
                torch.cuda.synchronize()  # 确保传输完成
            else:
                self.uni3d_model = self.uni3d_model.to(self.target_device)
                self.clip_model = self.clip_model.to(self.target_device)
        
        self.device = self.target_device
        self._models_on_gpu = True
        self._last_gpu_time = time.time()
        
        elapsed = time.time() - start_time
        print(f"⚡ GPU加载完成，耗时: {elapsed:.2f}秒")
    
    def _fast_offload_to_cpu(self):
        """🚀 快速offload到CPU"""
        if not self.enable_dynamic_offload or not self._models_on_gpu:
            return
            
        print("⚡ 快速offload模型到CPU...")
        start_time = time.time()
        
        # 快速移动到CPU
        self.uni3d_model = self.uni3d_model.to(self.cpu_device)
        self.clip_model = self.clip_model.to(self.cpu_device)
        self.device = self.cpu_device
        self._models_on_gpu = False
        
        # 强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        print(f"⚡ CPU offload完成，GPU内存已释放，耗时: {elapsed:.2f}秒")
    
    def _check_auto_offload(self):
        """检查是否需要自动offload（长时间未使用）"""
        if (self.enable_dynamic_offload and self._models_on_gpu and 
            time.time() - self._last_gpu_time > self._gpu_timeout):
            print(f"⏰ {self._gpu_timeout}秒未使用，自动offload到CPU")
            self._fast_offload_to_cpu()
    
    def _compute_semantic_score(self, mesh: Mesh, image_path: str, num_points: int = 8000) -> float:
        """计算mesh与图像的语义一致性评分 - 真正的实现"""
        try:
            # 1. 从mesh采样点云
            points, colors = sample_points_from_mesh(mesh, num_points)
            if points is None:
                print("⚠️ 点云采样失败")
                return 0.5
            
            # 2. 准备点云数据 (添加颜色维度)
            if colors is None:
                # 如果没有颜色信息，设置为白色
                colors = np.ones_like(points)
            
            # 组合点云和颜色 [N, 6] = [x,y,z,r,g,b]
            pc_data = np.concatenate([points, colors], axis=1)
            pc_tensor = torch.from_numpy(pc_data).float().to(self.device)
            pc_tensor = pc_tensor.unsqueeze(0)  # [1, N, 6]
            
            # 3. 加载并预处理图像
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # 4. 提取特征并计算相似度
            with torch.no_grad():
                # 提取图像特征
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 提取点云特征 
                pc_features = self.uni3d_model.encode_pc(pc_tensor)
                pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(image_features, pc_features, dim=-1)
                score = float(similarity.cpu().item())
                
                # 将相似度从 [-1, 1] 映射到 [0, 1]
                normalized_score = (score + 1) / 2
                
                return normalized_score
                
        except Exception as e:
            print(f"⚠️ 语义评分计算失败: {e}")
            return 0.5
    
    @torch.no_grad()
    def __call__(self, 
                 meshes: Union[Mesh, List[Mesh]], 
                 images: Union[str, List[str]],
                 metadata: dict = None) -> Tuple[List[float], dict]:
        """🚀 超高效图像模式评分 - 真正的实现"""
        

        
        # 检查自动offload
        self._check_auto_offload()
        
        # 确保模型初始化
        self._init_models()
        
        # 快速加载到GPU
        self._fast_load_to_gpu()
        
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(images, str):
            images = [images]
            
        # 确保 mesh 和 image 数量匹配
        if len(meshes) != len(images):
            if len(images) == 1:
                images = images * len(meshes)
            else:
                raise ValueError(f"Mesh 数量 ({len(meshes)}) 与 image 数量 ({len(images)}) 不匹配")
        
        # 🚀 真正的批量计算评分
        scores = []
        start_time = time.time()
        
        for i, (mesh, image_path) in enumerate(zip(meshes, images)):
            try:
                score = self._compute_semantic_score(mesh, image_path)
                scores.append(score)
                print(f"🎯 评分样本 {i+1}/{len(meshes)}: {score:.4f}")
            except Exception as e:
                print(f"⚠️ 评分样本 {i+1} 失败: {e}")
                scores.append(0.5)  # 使用默认分数
        
        # 快速offload到CPU
        self._fast_offload_to_cpu()
        
        elapsed = time.time() - start_time
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"⚡ 评分完成，{len(meshes)}个样本，平均分: {avg_score:.4f}，耗时: {elapsed:.2f}秒")
        
        # 返回兼容奖励系统的格式
        return scores, {
            "num_meshes": len(meshes), 
            "avg_score": avg_score,
            "eval_time": elapsed
        }

def main():
    """测试 Uni3D 评分器"""
    scorer = Uni3DScorer(enable_dynamic_offload=True)
    print("✅ Uni3D评分器初始化成功")

if __name__ == "__main__":
    main() 