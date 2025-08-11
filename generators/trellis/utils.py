#!/usr/bin/env python3
"""
TRELLIS工具函数
包含图像预处理和输出转换功能
"""
import sys
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
import torch
import trimesh
from PIL import Image

# 添加TRELLIS模块路径
reference_path = Path(__file__).parent.parent.parent / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))

import trellis.modules.sparse as sp

# 添加 KIUI 工具路径（用于 Mesh 对象）
kiui_path = Path(__file__).parent.parent.parent / "_reference_codes" / "kiuikit"
sys.path.insert(0, str(kiui_path))
from kiui.mesh import Mesh as KiuiMesh

def trellis_preprocess_image(image: Image.Image) -> Image.Image:
    """TRELLIS图像预处理，包含背景移除等
    
    参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:82-116 (preprocess_image)
    
    Args:
        image (Image.Image): 输入图像
        
    Returns:
        Image.Image: 预处理后的图像，输出尺寸为518x518
    """
    # 检查是否有alpha通道 - 完全按照源代码逻辑
    has_alpha = False
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]  # shape: (H, W)
        if not np.all(alpha == 255):
            has_alpha = True
    
    if has_alpha:
        # 已有透明通道，直接使用
        output = image
    else:
        # 需要移除背景
        image = image.convert('RGB')
        
        # 缩放处理 - 限制最大尺寸为1024
        max_size = max(image.size)  # max(W, H)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 使用rembg移除背景
        import rembg
        # 创建背景移除会话 - 使用u2net模型
        rembg_session = rembg.new_session('u2net')
        output = rembg.remove(image, session=rembg_session)  # shape: (H, W, 4) with alpha
        print("✅ 背景移除成功")
    
    # 按照源代码进行bbox裁剪和调整 - 源代码第103-116行
    output_np = np.array(output)  # shape: (H, W, 4)
    alpha = output_np[:, :, 3]  # shape: (H, W)
    bbox = np.argwhere(alpha > 0.8 * 255)  # shape: (N, 2) where N是有效像素数
    
    if len(bbox) > 0:
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])  # (x_min, y_min, x_max, y_max)
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # (center_x, center_y)
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # max(width, height)
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)
    
    # 调整到518x518尺寸
    output = output.resize((518, 518), Image.Resampling.LANCZOS)  # shape: (518, 518, 4)
    
    # 预乘alpha通道处理 - 源代码第113-115行
    output = np.array(output).astype(np.float32) / 255  # shape: (518, 518, 4), range: [0, 1]
    output = output[:, :, :3] * output[:, :, 3:4]  # shape: (518, 518, 3), 预乘alpha
    output = Image.fromarray((output * 255).astype(np.uint8))  # shape: (518, 518, 3)
    
    return output

def convert_trellis_to_trimesh(slat_outputs: Union[sp.SparseTensor, Dict, List]) -> List[trimesh.Trimesh]:
    """将TRELLIS输出转换为trimesh格式，用于奖励计算
    
    参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:195-217 (decode_slat)
    注意: 这个函数处理decode_slat的输出，不是直接对应源代码中的某个函数
    此函数为自定义实现，用于GRPO训练中的mesh数据处理
    
    Args:
        slat_outputs: TRELLIS SLAT解码输出，可能是SparseTensor或解码后的mesh
        
    Returns:
        List[trimesh.Trimesh]: trimesh对象列表
    """
    meshes = []
    
    # 处理不同类型的输入
    if isinstance(slat_outputs, dict):
        # 如果是字典格式的解码输出
        if 'mesh' in slat_outputs:
            mesh_data = slat_outputs['mesh']
            if isinstance(mesh_data, list):
                meshes.extend(mesh_data)
            else:
                meshes.append(mesh_data)
        else:
            print("⚠️ 解码输出中未找到mesh字段")
            # 返回空mesh而非异常
            empty_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))  # vertices: (0, 3), faces: (0, 3)
            return [empty_mesh]
    
    elif isinstance(slat_outputs, list):
        # 如果是mesh列表
        meshes.extend(slat_outputs)
    
    elif isinstance(slat_outputs, sp.SparseTensor):
        # 如果是SparseTensor，需要先解码
        print("⚠️ 收到SparseTensor，需要先通过pipeline解码为mesh")
        # 返回空mesh而非异常
        empty_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))  # vertices: (0, 3), faces: (0, 3)
        return [empty_mesh]
    
    else:
        # 单个mesh对象
        meshes.append(slat_outputs)
    
    # 转换为trimesh格式
    trimesh_objects = []
    for mesh in meshes:
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # 已经是mesh对象，直接转换
            if isinstance(mesh, trimesh.Trimesh):
                trimesh_objects.append(mesh)
            else:
                # 转换为trimesh
                vertices = mesh.vertices  # shape: (V, 3) where V是顶点数
                faces = mesh.faces  # shape: (F, 3) where F是面数
                
                # 确保tensor转换为numpy数组
                if torch.is_tensor(vertices):
                    vertices = vertices.cpu().numpy()
                if torch.is_tensor(faces):
                    faces = faces.cpu().numpy()
                
                trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                trimesh_objects.append(trimesh_obj)
        
        elif isinstance(mesh, dict):
            # 字典格式的mesh数据
            vertices = mesh.get('vertices', mesh.get('verts'))  # shape: (V, 3)
            faces = mesh.get('faces', mesh.get('faces'))  # shape: (F, 3)
            
            if vertices is not None and faces is not None:
                # 转换numpy数组
                if torch.is_tensor(vertices):
                    vertices = vertices.cpu().numpy()  # shape: (V, 3)
                if torch.is_tensor(faces):
                    faces = faces.cpu().numpy()  # shape: (F, 3)
                
                trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                trimesh_objects.append(trimesh_obj)
            else:
                print(f"⚠️ mesh字典缺少vertices或faces字段: {mesh.keys()}")
                continue
        
        else:
            print(f"⚠️ 未知的mesh格式: {type(mesh)}")
            continue
    
    if not trimesh_objects:
        print("⚠️ 未能转换任何有效的mesh对象")
        # 返回一个空的mesh作为fallback
        empty_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))  # vertices: (0, 3), faces: (0, 3)
        trimesh_objects.append(empty_mesh)
    
    print(f"✅ 成功转换 {len(trimesh_objects)} 个mesh对象")
    return trimesh_objects


def convert_trellis_to_kiuimesh(
    decoded: Union[Dict, List, trimesh.Trimesh]
) -> List[KiuiMesh]:
    """
    将 TRELLIS 解码输出转换为 kiui.mesh.Mesh（包含 .v/.f），便于统一下游打分与渲染接口。

    支持输入：
    - dict: 期望包含 'mesh' 键；值可以是 trimesh.Trimesh 或列表
    - list: mesh 列表
    - trimesh.Trimesh: 单个网格
    """
    kiui_meshes: List[KiuiMesh] = []

    meshes: List[trimesh.Trimesh] = []
    if isinstance(decoded, dict):
        if 'mesh' in decoded:
            mesh_data = decoded['mesh']
            if isinstance(mesh_data, list):
                for m in mesh_data:
                    if isinstance(m, trimesh.Trimesh):
                        meshes.append(m)
            elif isinstance(mesh_data, trimesh.Trimesh):
                meshes.append(mesh_data)
            else:
                print("⚠️ 未识别的 decoded['mesh'] 类型，跳过")
        else:
            print("⚠️ 解码输出缺少 'mesh' 键，返回空列表")
            return []
    elif isinstance(decoded, list):
        for m in decoded:
            if isinstance(m, trimesh.Trimesh):
                meshes.append(m)
    elif isinstance(decoded, trimesh.Trimesh):
        meshes.append(decoded)
    else:
        print(f"⚠️ 未知的 decoded 类型: {type(decoded)}，返回空列表")
        return []

    for m in meshes:
        v = torch.tensor(m.vertices, dtype=torch.float32)
        f = torch.tensor(m.faces, dtype=torch.int32)
        kiui_meshes.append(KiuiMesh(v=v, f=f, device=v.device))

    return kiui_meshes

def normalize_slat_tensor(slat: sp.SparseTensor, 
                         normalization: Dict[str, List[float]]) -> sp.SparseTensor:
    """对SLAT张量进行标准化
    
    参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:248-250 (sample_slat中的标准化)
    
    Args:
        slat (sp.SparseTensor): 原始SLAT张量，feats shape: (N, C)
        normalization (Dict): 包含mean和std的标准化参数
        
    Returns:
        sp.SparseTensor: 标准化后的SLAT张量，feats shape: (N, C)
    """
    # 按照源代码的标准化逻辑
    std = torch.tensor(normalization['std'])[None].to(slat.device)  # shape: (1, C)
    mean = torch.tensor(normalization['mean'])[None].to(slat.device)  # shape: (1, C)
    
    # 创建标准化后的SparseTensor - 使用源代码的公式
    normalized_feats = slat.feats * std + mean  # shape: (N, C) = (N, C) * (1, C) + (1, C)
    normalized_slat = sp.SparseTensor(
        feats=normalized_feats,  # shape: (N, C)
        coords=slat.coords  # shape: (N, 4)
    )
    
    return normalized_slat

def create_noise_sparse_tensor(coords: torch.Tensor, 
                              num_channels: int,
                              device: torch.device) -> sp.SparseTensor:
    """创建噪声稀疏张量用于采样
    
    参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:235-238 (sample_slat中的噪声创建)
    
    Args:
        coords (torch.Tensor): 稀疏结构坐标，shape: (N, 4)
        num_channels (int): 特征通道数
        device (torch.device): 设备
        
    Returns:
        sp.SparseTensor: 噪声稀疏张量，feats shape: (N, num_channels)
    """
    # 按照源代码创建噪声稀疏张量
    noise_feats = torch.randn(coords.shape[0], num_channels).to(device)  # shape: (N, num_channels)
    noise_tensor = sp.SparseTensor(
        feats=noise_feats,  # shape: (N, num_channels)
        coords=coords  # shape: (N, 4)
    )
    
    return noise_tensor

def validate_sparse_tensor(tensor: sp.SparseTensor) -> bool:
    """验证稀疏张量的有效性
    
    注意: 此函数为自定义实现，用于GRPO训练中的基础数据验证
    
    Args:
        tensor (sp.SparseTensor): 要验证的稀疏张量
        
    Returns:
        bool: 是否有效
    """
    if not isinstance(tensor, sp.SparseTensor):
        return False
    
    # 检查基本属性存在
    if tensor.coords is None or tensor.feats is None:
        return False
    
    # 检查维度匹配：coords和feats的第一维应该相等
    if tensor.coords.shape[0] != tensor.feats.shape[0]:  # coords: (N, 4), feats: (N, C) -> N should match
        return False
    
    return True 