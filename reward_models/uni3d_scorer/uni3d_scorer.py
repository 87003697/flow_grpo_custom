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
from .models.mesh_utils import Mesh

# 헬퍼 함수: Farthest Point Sampling (FPS) - PyTorch 구현
def _fps_pytorch(xyz, npoint):
    """
    Furthest Point Sampling using PyTorch - Official Uni3D Logic
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    
    return centroids

def _gather_pytorch(points, idx):
    """
    Gather operation using PyTorch - Official Uni3D Logic
    """
    B, N, C = points.shape
    idx = idx.unsqueeze(-1).expand(-1, -1, C)
    new_points = torch.gather(points, 1, idx)
    return new_points

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
        
        # 1. 初始化CLIP模型 - 强制本地权重
        print("🔄 正在加载 CLIP 模型: EVA02-E-14-plus")
        clip_weights_path = Path("pretrained_weights/eva02_e_14_plus_laion2b_s9b_b144k.pt")
        if not clip_weights_path.exists():
            raise FileNotFoundError(
                f"🔥 错误: CLIP权重文件未找到! 请确保 '{clip_weights_path}' 存在。"
            )
        print(f"📁 从本地加载 CLIP 权重: {clip_weights_path}")
        
        # 先创建模型架构
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'EVA02-E-14-plus', 
            pretrained=None  # 强制不使用在线预训练权重
        )
        # 加载本地权重
        state_dict = torch.load(clip_weights_path, map_location='cpu', weights_only=False)
        
        # 严格加载CLIP权重
        try:
            missing_keys, unexpected_keys = self.clip_model.load_state_dict(state_dict, strict=True)
            print("✅ CLIP权重严格加载成功")
        except Exception as e:
            print(f"❌ CLIP严格加载失败: {e}")
            print("🔄 尝试非严格加载...")
            missing_keys, unexpected_keys = self.clip_model.load_state_dict(state_dict, strict=False)
            print(f"⚠️ 缺失的键: {len(missing_keys)} 个")
            print(f"⚠️ 多余的键: {len(unexpected_keys)} 个")
            if missing_keys:
                print(f"缺失键示例: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"多余键示例: {unexpected_keys[:5]}")
                
        del state_dict # 立即清理内存
        
        # 🔧 中间清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. 初始化Uni3D模型 - 强制本地权重
        print("🔄 正在初始化 Uni3D 模型...")
        eva_weights_path = Path("pretrained_weights/eva_giant_patch14_560.pt")
        uni3d_weights_path = Path("pretrained_weights/uni3d-g.pt")

        if not eva_weights_path.exists():
            raise FileNotFoundError(
                f"🔥 错误: EVA权重文件未找到! 请确保 '{eva_weights_path}' 存在。"
            )
        if not uni3d_weights_path.exists():
            raise FileNotFoundError(
                f"🔥 错误: Uni3D权重文件未找到! 请确保 '{uni3d_weights_path}' 存在。"
            )
        
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
        checkpoint = torch.load(uni3d_weights_path, map_location='cpu', weights_only=False)
        
        # 按照官方代码的方式处理权重键名
        if 'module' in checkpoint:
            print("✅ 使用 'module' 键加载权重（官方方式）")
            state_dict = checkpoint['module']
        elif 'model' in checkpoint:
            print("✅ 使用 'model' 键加载权重")
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            print("✅ 使用 'state_dict' 键加载权重")
            state_dict = checkpoint['state_dict']
        else:
            print("✅ 直接使用根级别权重")
            state_dict = checkpoint
            
        # 严格加载权重，不允许不匹配
        try:
            missing_keys, unexpected_keys = self.uni3d_model.load_state_dict(state_dict, strict=True)
            print("✅ Uni3D预训练权重严格加载成功")
        except Exception as e:
            print(f"❌ 严格加载失败: {e}")
            print("🔄 尝试非严格加载...")
            missing_keys, unexpected_keys = self.uni3d_model.load_state_dict(state_dict, strict=False)
            print(f"⚠️ 缺失的键: {len(missing_keys)} 个")
            print(f"⚠️ 多余的键: {len(unexpected_keys)} 个")
            if missing_keys:
                print(f"缺失键示例: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"多余键示例: {unexpected_keys[:5]}")
                
        del checkpoint, state_dict # 立即清理内存
        
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
    
    def _meshes_to_pointclouds_torch(self, meshes: List[Mesh], num_points: int = 10000) -> torch.Tensor:
        """
        🔥 批量并行地将多个Mesh对象高效转换为一个点云张量。
        """
        # 使用列表推导式并行处理每个mesh
        pointclouds = [self._mesh_to_pointcloud_torch(mesh, num_points) for mesh in meshes]
        
        # 将所有点云张量堆叠成一个批次
        return torch.stack(pointclouds)

    def _mesh_to_pointcloud_torch(self, mesh: Mesh, num_points: int = 10000) -> torch.Tensor:
        """
        ⚡ 高效的torch版本mesh到点云转换，严格遵循Uni3D官方规范
        1. 面积加权过采样 -> 2. 最远点采样 (FPS) -> 3. 标准化
        """
        # 1. 严格检查mesh数据
        if not (hasattr(mesh, 'v') and hasattr(mesh, 'f') and
                mesh.v is not None and mesh.f is not None and
                len(mesh.v) > 0 and len(mesh.f) > 0):
            raise ValueError(
                f"🔥 错误: 无效的mesh数据! 顶点数: {len(mesh.v) if hasattr(mesh, 'v') and mesh.v is not None else 'N/A'}, "
                f"面数: {len(mesh.f) if hasattr(mesh, 'f') and mesh.f is not None else 'N/A'}"
            )

        vertices = mesh.v if torch.is_tensor(mesh.v) else torch.from_numpy(mesh.v).float()
        faces = mesh.f if torch.is_tensor(mesh.f) else torch.from_numpy(mesh.f).long()
        
        # 2. 处理颜色信息
        if hasattr(mesh, 'vc') and mesh.vc is not None:
            vertex_colors = mesh.vc if torch.is_tensor(mesh.vc) else torch.from_numpy(mesh.vc).float()
            # 确保颜色值在 [0, 1] 范围内（官方Uni3D规范）
            if vertex_colors.max() > 1.0:
                vertex_colors = vertex_colors / 255.0
        else:
            # 使用默认颜色 0.4 (完全按照官方Uni3D实现)
            vertex_colors = torch.ones_like(vertices) * 0.4
        
        # 3. ⚡ 高效的面积加权采样 (全torch操作)
        face_vertices = vertices[faces]  # (F, 3, 3)
        v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
        
        # 计算面积
        cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_areas = 0.5 * torch.norm(cross_product, dim=1)
        face_probs = face_areas / face_areas.sum()
        
        # 采样足够多的点
        initial_num_points = max(num_points * 2, 4096)
        sampled_face_indices = torch.multinomial(face_probs, initial_num_points, replacement=True)
        
        # 在采样面片上重心坐标采样
        sampled_faces = face_vertices[sampled_face_indices]  # (initial_num_points, 3, 3)
        
        # 重心坐标采样
        r1 = torch.rand(initial_num_points, 1)
        r2 = torch.rand(initial_num_points, 1)
        
        # 确保 r1 + r2 <= 1
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]  
        r2[mask] = 1 - r2[mask]
        r3 = 1 - r1 - r2
        
        # 计算采样点
        sampled_points = (
            r1 * sampled_faces[:, 0] + 
            r2 * sampled_faces[:, 1] + 
            r3 * sampled_faces[:, 2]
        )  # (initial_num_points, 3)
        
        # 为采样点计算颜色（重心坐标插值）
        sampled_face_colors = vertex_colors[faces[sampled_face_indices]]  # (initial_num_points, 3, 3)
        sampled_colors = (
            r1 * sampled_face_colors[:, 0] + 
            r2 * sampled_face_colors[:, 1] + 
            r3 * sampled_face_colors[:, 2]
        )  # (initial_num_points, 3)
        
        # 合并坐标和颜色
        initial_pointcloud = torch.cat([sampled_points, sampled_colors], dim=1)  # (initial_num_points, 6)
        
        # 4. 🔥 关键修正: 使用最远点采样 (FPS) 替代随机采样
        if initial_pointcloud.shape[0] > num_points:
            # FPS需要 [B, N, 3] 格式的输入
            xyz_for_fps = initial_pointcloud[:, :3].unsqueeze(0)  # 增加batch维度
            fps_indices = _fps_pytorch(xyz_for_fps, num_points) # [1, npoint]
            
            # 使用gather操作根据索引选择点
            pointcloud = _gather_pytorch(initial_pointcloud.unsqueeze(0), fps_indices)[0] # 移除batch维度
        else:
            # 如果点数不足，直接使用 (这种情况很少见)
            pointcloud = initial_pointcloud
        
        # 5. 🔧 关键：使用官方pc_normalize标准化
        xyz = pointcloud[:, :3]  # (num_points, 3)
        colors = pointcloud[:, 3:]  # (num_points, 3)
        
        # 官方pc_normalize实现 (torch版本)
        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
        xyz = xyz / m
        
        # 重新组合标准化后的数据
        normalized_pointcloud = torch.cat([xyz, colors], dim=1)  # (num_points, 6)
        
        return normalized_pointcloud

    def _check_auto_offload(self):
        """检查是否需要自动offload（长时间未使用）"""
        if (self.enable_dynamic_offload and self._models_on_gpu and 
            time.time() - self._last_gpu_time > self._gpu_timeout):
            print(f"⏰ {self._gpu_timeout}秒未使用，自动offload到CPU")
            self._fast_offload_to_cpu()
    
    @torch.no_grad()
    def __call__(self, 
                 meshes: Union[Mesh, List[Mesh]], 
                 images: Union[str, List[str]],
                 metadata: dict = None,
                 openshape_setting: bool = False) -> Tuple[List[float], dict]:
        """🚀 使用官方Uni3D流程的图像-3D评分器"""
        
        # 检查自动offload和初始化
        self._check_auto_offload()
        self._init_models()
        self._fast_load_to_gpu()
        
        start_time = time.time()
        
        # 统一输入格式
        if isinstance(meshes, Mesh):
            meshes = [meshes]
        if isinstance(images, str):
            images = [images]
            
        # 确保数量匹配
        if len(meshes) != len(images):
            if len(images) == 1:
                images = images * len(meshes)
            else:
                raise ValueError(f"Mesh 数量 ({len(meshes)}) 与 image 数量 ({len(images)}) 不匹配")
        
        # 使用官方流程处理点云
        pc_tensor = prepare_pointcloud_batch(meshes, num_points=10000, 
                                           openshape_setting=openshape_setting)
        pc_tensor = pc_tensor.to(self.device)
 
        # 批量处理图像
        from PIL import Image
        image_tensors = torch.stack(
            [self.clip_preprocess(Image.open(p).convert('RGB')) for p in images]
        ).to(self.device)
 
        # 批量推理
        with torch.no_grad():
            # 提取特征
            image_features = self.clip_model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            pc_features = self.uni3d_model.encode_pc(pc_tensor)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            
            # 调试信息
            print(f"🔍 调试信息:")
            print(f"   image_features.shape: {image_features.shape}")
            print(f"   pc_features.shape: {pc_features.shape}")
            print(f"   image_features 范围: [{image_features.min():.6f}, {image_features.max():.6f}]")
            print(f"   pc_features 范围: [{pc_features.min():.6f}, {pc_features.max():.6f}]")
            print(f"   image_features 均值: {image_features.mean():.6f}")
            print(f"   pc_features 均值: {pc_features.mean():.6f}")
            print(f"   image_features L2范数: {image_features.norm(dim=-1).mean():.6f}")
            print(f"   pc_features L2范数: {pc_features.norm(dim=-1).mean():.6f}")
            
            # 计算相似度（模拟官方main.py:556的直接矩阵乘法）
            dot_product = (image_features * pc_features).sum(dim=-1)
            print(f"   点积结果: {dot_product.cpu().tolist()}")
            
            similarity = torch.cosine_similarity(image_features, pc_features, dim=-1)
            print(f"   余弦相似度: {similarity.cpu().tolist()}")
            scores = similarity.cpu().tolist()
        
        # 清理
        self._fast_offload_to_cpu()
        
        elapsed = time.time() - start_time
        avg_score = sum(scores) / len(scores) if scores else 0.0
        for i, score in enumerate(scores):
            print(f"⚡ 样本 {i+1} 分数: {score:.4f}")
        print(f"⏱️ 评分耗时: {elapsed:.2f}秒")
        
        return scores, {
            "num_meshes": len(meshes), 
            "avg_score": avg_score,
            "eval_time": elapsed
        }

def prepare_pointcloud_batch(meshes: List[Mesh], num_points: int = 10000, 
                          openshape_setting: bool = False) -> torch.Tensor:
    """
    将多个 mesh 转换为批量点云数据，完全按照官方 Uni3D 流程
    
    Args:
        meshes: kiui mesh 对象列表
        num_points: 每个点云的采样点数（官方默认 10000）
        openshape_setting: 是否使用OpenShape设置（Y-Z轴翻转）
        
    Returns:
        torch.Tensor: 批量点云数据，形状 (batch_size, num_points, 6)
    """
    pointclouds = []
    
    for mesh in meshes:
        # 按照官方流程处理每个mesh
        pointcloud = _sample_points_from_mesh_official(mesh, num_points, openshape_setting)
        pointclouds.append(pointcloud)
    
    return torch.stack(pointclouds, dim=0)

def _sample_points_from_mesh_official(mesh: Mesh, num_points: int = 10000, 
                                   openshape_setting: bool = False) -> torch.Tensor:
    """
    从单个mesh采样点云，完全按照官方Uni3D流程
    """
    vertices = mesh.v if torch.is_tensor(mesh.v) else torch.from_numpy(mesh.v).float()
    faces = mesh.f if torch.is_tensor(mesh.f) else torch.from_numpy(mesh.f).long()
    
    initial_num_points = max(num_points * 3, 30000)
    
    # 处理颜色信息（官方方式）
    if hasattr(mesh, 'vc') and mesh.vc is not None:
        vertex_colors = mesh.vc if torch.is_tensor(mesh.vc) else torch.from_numpy(mesh.vc).float()
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
    else:
        vertex_colors = torch.ones_like(vertices) * 0.4
    
    # 面积加权采样
    face_vertices = vertices[faces]
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = 0.5 * torch.norm(cross_product, dim=1)
    face_probs = face_areas / face_areas.sum()
    
    selected_faces = torch.multinomial(face_probs, initial_num_points, replacement=True)
    selected_face_vertices = face_vertices[selected_faces]
    
    # 重心坐标采样
    u = torch.rand(initial_num_points, device=vertices.device)
    v = torch.rand(initial_num_points, device=vertices.device)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    w = 1.0 - u - v
    
    sampled_points = (
        w.unsqueeze(-1) * selected_face_vertices[:, 0] +
        u.unsqueeze(-1) * selected_face_vertices[:, 1] +
        v.unsqueeze(-1) * selected_face_vertices[:, 2]
    )
    
    selected_face_colors = vertex_colors[faces[selected_faces]]
    sampled_colors = (
        w.unsqueeze(-1) * selected_face_colors[:, 0] +
        u.unsqueeze(-1) * selected_face_colors[:, 1] +
        v.unsqueeze(-1) * selected_face_colors[:, 2]
    )
    
    # FPS下采样
    if sampled_points.shape[0] > num_points:
        xyz_for_fps = sampled_points.unsqueeze(0)
        fps_indices = _fps_pytorch(xyz_for_fps, num_points)
        sampled_points = _gather_pytorch(sampled_points.unsqueeze(0), fps_indices)[0]
        sampled_colors = _gather_pytorch(sampled_colors.unsqueeze(0), fps_indices)[0]
    
    # 官方坐标处理流程
    xyz = sampled_points
    rgb = sampled_colors
    
    if openshape_setting:
        xyz[:, [1, 2]] = xyz[:, [2, 1]]  # Y-Z轴交换
        # normalize_pc
        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
        xyz = xyz / m
    else:
        # pc_normalize  
        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
        xyz = xyz / m
    
    # 最终拼接
    pointcloud = torch.cat([xyz, rgb], dim=1)
    return pointcloud

def main():
    """测试 Uni3D 评分器"""
    scorer = Uni3DScorer(enable_dynamic_offload=True)
    print("✅ Uni3D评分器初始化成功")

if __name__ == "__main__":
    main() 