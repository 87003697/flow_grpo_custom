import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def save_rgb_visualization(
    images_batched: torch.Tensor,  # 形状: (K,S,3,H,W)
    rgb_img: torch.Tensor,         # 形状: (3,R,R) 输入 RGB [0,1]
    rgb_mesh: torch.Tensor,        # 形状: (3,R,R) 渲染 RGB [0,1]
    vis_dir: str,
    tag: str
) -> None:
    """保存 RGB 相机搜索可视化结果，包含 support cameras、input RGB 和 rendered RGB。

    关键差异（相对于 camera_normal_scorer）:
        - 输入和渲染的都是 RGB 图像（值域 [0,1]），而非法线（[-1,1]）
        - 可视化时无需映射，直接显示

    输入:
        images_batched: (K,S,3,H,W) 包含 support 视角和 query 视角的图像批次
        rgb_img: (3,R,R) 输入 RGB 图像 [0,1]
        rgb_mesh: (3,R,R) 预测视角渲染的 RGB 图像 [0,1]
        vis_dir: 输出目录
        tag: 文件名标识
    输出:
        None（生成综合可视化图片: camera_rgb_vis_{tag}.png）
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # 只处理第一个样本（K=1 的情况）
    images = images_batched[0]  # 形状: (S,3,H,W)
    S = images.shape[0]  # S 个视角（前 S-1 个是 support，最后 1 个是 query）
    
    # 计算子图布局：支持相机数量 + 输入 RGB + 渲染 RGB
    num_support = S - 1  # 支持相机数量
    total_cols = num_support + 2  # support cameras + input RGB + rendered RGB
    
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
    if total_cols == 1:
        axes = [axes]
    
    # 1. 显示所有 support cameras
    for i in range(num_support):
        support_img = images[i]  # 形状: (3,H,W)
        # 转换为可显示格式 [0,1] -> RGB
        support_img_np = support_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # 形状: (H,W,3)
        
        axes[i].imshow(support_img_np)
        axes[i].set_title(f"Support Camera {i+1}")
        axes[i].axis('off')
    
    # 2. 显示 input RGB（关键差异：直接使用 [0,1]，无需映射）
    rgb_img_display = rgb_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
    axes[num_support].imshow(rgb_img_display)
    axes[num_support].set_title("Input RGB")
    axes[num_support].axis('off')
    
    # 3. 显示 rendered RGB（关键差异：直接使用 [0,1]，无需映射）
    rgb_mesh_display = rgb_mesh.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
    axes[num_support + 1].imshow(rgb_mesh_display)
    axes[num_support + 1].set_title("Rendered RGB")
    axes[num_support + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"camera_rgb_vis_{tag}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_rgb_comparison(rgb_img: torch.Tensor, rgb_mesh: torch.Tensor, vis_dir: str, tag: str) -> None:
    """保存简化版 RGB 对比图（仅输入 RGB vs 渲染 RGB）。

    输入:
        rgb_img: 输入 RGB (3,R,R) [0,1]
        rgb_mesh: 渲染 RGB (3,R,R) [0,1]
        vis_dir: 输出目录
        tag: 文件名标识
    输出:
        None（生成两张 PNG: input_rgb_{tag}.png, rendered_rgb_{tag}.png）
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # 保存输入 RGB
    Image.fromarray(
        (rgb_img.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    ).save(os.path.join(vis_dir, f"input_rgb_{tag}.png"))
    
    # 保存渲染 RGB
    Image.fromarray(
        (rgb_mesh.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    ).save(os.path.join(vis_dir, f"rendered_rgb_{tag}.png"))
