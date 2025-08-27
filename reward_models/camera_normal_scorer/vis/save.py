import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def save_similarity_inputs(n_img: torch.Tensor, n_mesh: torch.Tensor, vis_dir: str, tag: str) -> None:
    """保存用于可视化的法线对比图。

    输入:
        n_img: 预测/图像侧法线 (3,R,R) [-1,1]
        n_mesh: 渲染法线 (3,R,R) [-1,1]
        vis_dir: 输出目录。
        tag: 文件名标识。
    输出:
        None（生成两张 PNG: pred_normal_{tag}.png, render_normal_{tag}.png）
    """
    os.makedirs(vis_dir, exist_ok=True)
    Image.fromarray(((n_img.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(vis_dir, f"pred_normal_{tag}.png")
    )
    Image.fromarray(((n_mesh.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(vis_dir, f"render_normal_{tag}.png")
    )


def save_camera_search_visualization(
    images_batched: torch.Tensor,  # 形状: (K,S,3,H,W)
    n_img: torch.Tensor,          # 形状: (3,R,R) 输入法线
    n_mesh: torch.Tensor,         # 形状: (3,R,R) 预测视角渲染法线
    vis_dir: str,
    tag: str
) -> None:
    """保存完整的相机搜索可视化结果，包含support cameras、input normal和predicted view。

    输入:
        images_batched: (K,S,3,H,W) 包含support视角和query视角的图像批次
        n_img: (3,R,R) 输入法线图 [-1,1]
        n_mesh: (3,R,R) 预测视角渲染的法线图 [-1,1]
        vis_dir: 输出目录
        tag: 文件名标识
    输出:
        None（生成综合可视化图片: camera_search_vis_{tag}.png）
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # 只处理第一个样本（K=1的情况）
    images = images_batched[0]  # 形状: (S,3,H,W)
    S = images.shape[0]  # S个视角（前S-1个是support，最后1个是query）
    
    # 计算子图布局：支持相机数量 + 输入法线 + 预测视角
    num_support = S - 1  # 支持相机数量
    total_cols = num_support + 2  # support cameras + input normal + predicted view
    
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
    if total_cols == 1:
        axes = [axes]
    
    # 1. 显示所有support cameras
    for i in range(num_support):
        support_img = images[i]  # 形状: (3,H,W)
        # 转换为可显示格式 [0,1] -> RGB
        support_img_np = support_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # 形状: (H,W,3)
        
        axes[i].imshow(support_img_np)
        axes[i].set_title(f"Support Camera {i+1}")
        axes[i].axis('off')
    
    # 2. 显示input normal
    n_img_display = ((n_img.clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
    axes[num_support].imshow(n_img_display)
    axes[num_support].set_title("Input Normal")
    axes[num_support].axis('off')
    
    # 3. 显示predicted view (渲染的法线)
    n_mesh_display = ((n_mesh.clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
    axes[num_support + 1].imshow(n_mesh_display)
    axes[num_support + 1].set_title("Predicted View")
    axes[num_support + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"camera_search_vis_{tag}.png"), dpi=150, bbox_inches='tight')
    plt.close()



