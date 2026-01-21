"""VisualIO - 训练/评估可视化保存"""
import os
import numpy as np
from PIL import Image
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict

from .mixins import WandbMixin


# =====================================================================
# 通用图像处理工具
# =====================================================================

def composite_alpha_to_black(img: Image.Image) -> Image.Image:
    """
    将带有 Alpha 通道的图像合成到黑色背景上，并转为 RGB。
    如果图像没有 Alpha 通道，直接转为 RGB。
    
    与 TRELLIS preprocess_image 的 Alpha 预乘处理保持一致：
    output = output[:, :, :3] * output[:, :, 3:4]
    """
    if img.mode == 'RGBA':
        background = Image.new('RGBA', img.size, (0, 0, 0, 255))  # 黑色不透明背景
        combined = Image.alpha_composite(background, img)
        return combined.convert('RGB')
    else:
        return img.convert('RGB')


def composite_alpha_to_white(img: Image.Image) -> Image.Image:
    """
    将带有 Alpha 通道的图像合成到白色背景上，并转为 RGB。
    如果图像没有 Alpha 通道，直接转为 RGB。
    """
    if img.mode == 'RGBA':
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 白色不透明背景
        combined = Image.alpha_composite(background, img)
        return combined.convert('RGB')
    else:
        return img.convert('RGB')


# =====================================================================
# VisualIO - 训练/评估可视化保存（使用 WandbMixin）
# =====================================================================


class VisualIO(WandbMixin):
    """
    统一的可视化保存工具（训练/评估共用，支持 Wandb）。
    
    继承自 WandbMixin，提供本地文件保存 + Wandb 图像上传。
    """

    def __init__(
        self, 
        root: Path, 
        target_h: int = 512, 
        vis_freq: int = 100, 
        accelerator: Accelerator = None,
        max_wandb_samples: int = 4,
    ):
        self.root = root
        self.target_h = target_h
        self.vis_freq = vis_freq
        self.accelerator = accelerator
        self.max_wandb_samples = max_wandb_samples

    # ===== 工具方法 =====

    @staticmethod
    def to_pil(x) -> Image.Image:
        """Tensor/ndarray -> PIL"""
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return Image.fromarray((x * 255).clip(0, 255).astype(np.uint8))

    def resize(self, img: Image.Image) -> Image.Image:
        """按 target_h 等比缩放"""
        w, h = img.size
        scale = self.target_h / max(1, h)
        return img.resize((max(1, int(w * scale)), self.target_h), Image.Resampling.LANCZOS)

    @staticmethod
    def get_names(state) -> list:
        """从 state 提取图像名列表"""
        return [os.path.splitext(os.path.basename(p))[0] for p in state.views_conditioned.paths]

    def make_grid(self, images: list) -> Image.Image:
        """将 PIL 列表拼成水平网格"""
        imgs = [self.resize(im) for im in images]
        margin = 12
        total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
        total_h = max(im.height for im in imgs) + margin * 2
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x = margin
        for im in imgs:
            canvas.paste(im, (x, margin))
            x += im.width + margin
        return canvas

    def save_pil(self, pil: Image.Image, path: Path) -> None:
        """保存 PIL 到文件（自动创建目录）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(path)

    # ===== 主方法 =====

    def save_batch_train(self, state, epoch: int, step: int, pipe=None, n_progress_samples: int = 0) -> None:
        """
        训练模式：保存 [cond | gen | edit | progress] 网格。
        
        目录结构: root/epoch_{N}/step_{M}/{name}.png
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        
        conds = state.views_conditioned.image_pils
        gens = state.views_generated.image_tensor
        edits = state.views_edited.image_tensor
        trackers = state.views_edited.trackers if n_progress_samples > 0 else None
        
        wandb_images = {}
        for b, name in enumerate(names):
            # 构建图像列表
            imgs = [
                composite_alpha_to_white(conds[b]),
                self.to_pil(gens[b, 0]),
            ]
            if edits is not None:
                imgs.append(self.to_pil(edits[b, 0].permute(1, 2, 0)))
            if trackers and pipe:
                imgs.append(trackers[b].get_progress_grid(pipe, n_progress_samples))
            
            # 保存网格
            grid = self.make_grid(imgs)
            self.save_pil(grid, out_dir / f"{name}.png")
            
            if b < self.max_wandb_samples:
                wandb_images[f"train/{name}"] = grid
        
        self.log_images(wandb_images, step=step, prefix="")

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        评估模式：保存渲染图 + 可选 mesh 导出。
        
        目录结构: root/epoch_{N}/{name}/color.png, mesh.obj
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        gens = state.views_generated.image_tensor
        meshes = (render_out or {}).get("meshes", [])
        
        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name
            
            # 保存 color
            color_pil = self.to_pil(gens[b, 0])
            self.save_pil(color_pil, sample_dir / "color.png")
            if b < self.max_wandb_samples:
                wandb_images[f"eval/{name}"] = color_pil
            
            # 导出 mesh
            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))
        
        self.log_images(wandb_images, step=epoch, prefix="")


class Trellis2VisualIO(VisualIO):
    """
    Trellis2 专用的可视化保存工具。
    
    支持 Shape + Tex 双阶段输出：
    - 训练模式：分别保存 shape / pbr 网格图
    - 评估模式：保存 cond、shape_normal、pbr_color 和可选 mesh
    """

    def _build_grid(self, images: list) -> Image.Image:
        imgs = [im for im in images if im is not None]
        if not imgs:
            return None
        return self.make_grid(imgs)

    def save_batch_train(
        self,
        state,
        epoch: int,
        step: int,
        pipe: Any = None,
        n_progress_samples: int = 0,
    ) -> None:
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        conds = state.views_conditioned.image_pils
        shape_tensor = getattr(state.views_generated, "shape_tensor", None)
        pbr_tensor = getattr(state.views_generated, "pbr_tensor", None)
        edits = state.views_edited.image_tensor
        trackers = state.views_edited.trackers if n_progress_samples > 0 else None

        wandb_images: Dict[str, Image.Image] = {}

        for b, name in enumerate(names):
            cond_img = composite_alpha_to_white(conds[b]) if conds else None
            edit_img = None
            if edits is not None:
                edit_img = self.to_pil(edits[b, 0].permute(1, 2, 0))

            progress_grid = None
            if trackers is not None and pipe is not None:
                progress_grid = trackers[b].get_progress_grid(pipe, n_progress_samples)

            if shape_tensor is not None:
                shape_img = self.to_pil(shape_tensor[b, 0])
                grid = self._build_grid([cond_img, shape_img, edit_img, progress_grid])
                if grid is not None:
                    self.save_pil(grid, out_dir / f"{name}_shape.png")
                    if b < self.max_wandb_samples:
                        wandb_images[f"train/{name}_shape"] = grid

            if pbr_tensor is not None:
                pbr_img = self.to_pil(pbr_tensor[b, 0])
                grid = self._build_grid([cond_img, pbr_img, edit_img, progress_grid])
                if grid is not None:
                    self.save_pil(grid, out_dir / f"{name}_pbr.png")
                    if b < self.max_wandb_samples:
                        wandb_images[f"train/{name}_pbr"] = grid

        self.log_images(wandb_images, step=step, prefix="")

    def save_batch_eval(
        self,
        state,
        epoch: int,
        render_out: dict = None,
        pipeline: Any = None,
        export_mesh: bool = False,
    ) -> None:
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"

        conds = state.views_conditioned.image_pils
        shape_tensor = getattr(state.views_generated, "shape_tensor", None)
        pbr_tensor = getattr(state.views_generated, "pbr_tensor", None)
        meshes = (render_out or {}).get("meshes", [])

        wandb_images: Dict[str, Image.Image] = {}

        for b, name in enumerate(names):
            sample_dir = out_dir / name
            sample_dir.mkdir(parents=True, exist_ok=True)

            if conds and b < len(conds):
                cond_img = composite_alpha_to_white(conds[b])
                self.save_pil(cond_img, sample_dir / "cond.png")
                if b < self.max_wandb_samples:
                    wandb_images[f"eval/{name}_cond"] = cond_img

            if shape_tensor is not None:
                shape_img = self.to_pil(shape_tensor[b, 0])
                self.save_pil(shape_img, sample_dir / "shape_normal.png")
                if b < self.max_wandb_samples:
                    wandb_images[f"eval/{name}_shape"] = shape_img

            if pbr_tensor is not None:
                pbr_img = self.to_pil(pbr_tensor[b, 0])
                self.save_pil(pbr_img, sample_dir / "pbr_color.png")
                if b < self.max_wandb_samples:
                    wandb_images[f"eval/{name}_pbr"] = pbr_img

            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))

        self.log_images(wandb_images, step=epoch, prefix="")

