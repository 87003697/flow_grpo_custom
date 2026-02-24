"""VisualIO - 训练/评估可视化保存"""
import os
import numpy as np
from PIL import Image
from accelerate import Accelerator
from pathlib import Path

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
    可视化工具基类（state 无关）。

    提供通用工具方法（to_pil / make_grid / save_pil / resize / get_names）
    和通用的 save_batch_eval。

    ★ 不包含 save_batch_train：训练可视化依赖具体 State 的字段结构，
      由子类（TrellisVisualIO / Trellis2VisualIO）实现。
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

    # ===== 评估方法 =====

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        评估模式：保存渲染图 + 可选 mesh 导出。
        
        目录结构: root/epoch_{N}/{name}/color_v0.png, color_v1.png, ..., mesh.obj
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        gens = state.views_generated.image_tensor  # (B, V, H, W, C)
        meshes = (render_out or {}).get("meshes", [])
        num_views = gens.shape[1]  # 视角数 V
        
        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name
            
            # 保存所有视角的 color
            for v in range(num_views):
                color_pil = self.to_pil(gens[b, v])  # (H, W, C)
                self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                if b < self.max_wandb_samples and v == 0:
                    wandb_images[f"eval/{name}"] = color_pil
            
            # 导出 mesh
            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))
        
        self.log_images(wandb_images, step=epoch, prefix="")


# =====================================================================
# TrellisVisualIO - Trellis 专用可视化（支持 Hybrid 双路渲染）
# =====================================================================


class TrellisVisualIO(VisualIO):
    """
    Trellis 专用可视化保存（TrellisState 感知）。

    适用场景：
    - 单路训练（save_batch_train）: [cond | gen | edit | progress] → {name}.png
    - Hybrid 双路训练:
      - save_normal_train: [cond | normal | edit | progress] → {name}_normal.png
      - save_color_train:  [cond | color | edit | progress]  → {name}_color.png

    TrellisState 字段映射：
    - views_generated: image_tensor（GS Color）、normal_tensor（Mesh Normal）
    - views_edited:    color_tensor / color_trackers、normal_tensor / normal_trackers

    评估模式（save_batch_eval 覆写）：
    - normal.png（normal_tensor 有值时）
    - color.png（image_tensor 有值时）
    """

    def _save_stage_train(
        self, state, epoch: int, step: int,
        render_tensor,       # (B, V, H, W, C) 渲染结果
        suffix: str,         # 文件后缀，如 "_normal" 或 "_color"
        edit_tensor=None,    # (B, V, C, H, W) edit 结果（显式传入）
        trackers=None,       # List[StateTracker]（显式传入）
        pipe=None,           # guidance pipeline（用于生成 progress grid）
        n_progress_samples: int = 0,
        wandb_prefix: str = "train",
    ) -> None:
        """
        内部方法：保存单个渲染路的训练可视化网格。

        网格内容: [cond | render | edit | progress]
          - edit / progress 仅在对应参数有值时显示
        目录结构: root/epoch_{N}/step_{M}/{name}_{suffix}.png
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        conds = state.views_conditioned.image_pils

        wandb_images = {}
        for b, name in enumerate(names):
            imgs = [composite_alpha_to_white(conds[b])]

            if render_tensor is not None:
                imgs.append(self.to_pil(render_tensor[b, 0]))

            if edit_tensor is not None:
                imgs.append(self.to_pil(edit_tensor[b, 0].permute(1, 2, 0)))

            if trackers and pipe and n_progress_samples > 0:
                imgs.append(trackers[b].get_progress_grid(pipe, n_progress_samples))

            grid = self.make_grid(imgs)
            self.save_pil(grid, out_dir / f"{name}{suffix}.png")

            if b < self.max_wandb_samples:
                wandb_images[f"{wandb_prefix}/{name}{suffix}"] = grid

        self.log_images(wandb_images, step=step, prefix="")

    # ===== 单路训练 =====

    def save_batch_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """
        单路训练可视化: [cond | gen | edit | progress] → {name}.png

        读取 views_generated.image_tensor 和 views_edited.color_tensor。
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.image_tensor,
            edit_tensor=state.views_edited.color_tensor,
            trackers=state.views_edited.color_trackers,
            suffix="",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    # ===== Hybrid 双路训练 =====

    def save_normal_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """Mesh Normal 渲染可视化: [cond | normal | edit | progress] → {name}_normal.png"""
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.normal_tensor,
            edit_tensor=state.views_edited.normal_tensor,
            trackers=state.views_edited.normal_trackers,
            suffix="_normal",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    def save_color_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """GS Color 渲染可视化: [cond | color | edit | progress] → {name}_color.png"""
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.image_tensor,
            edit_tensor=state.views_edited.color_tensor,
            trackers=state.views_edited.color_trackers,
            suffix="_color",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        评估模式：按渲染器分别保存（支持多视角）。

        目录结构:
            root/epoch_{N}/{name}/
            ├── normal_v0.png, normal_v1.png, ...   # normal_tensor 有值时保存
            ├── color_v0.png, color_v1.png, ...     # image_tensor 有值时保存
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        vg = state.views_generated

        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name

            if vg.normal_tensor is not None:
                num_views = vg.normal_tensor.shape[1]  # V
                for v in range(num_views):
                    normal_pil = self.to_pil(vg.normal_tensor[b, v])  # (H, W, C)
                    self.save_pil(normal_pil, sample_dir / f"normal_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/normal"] = normal_pil

            if vg.image_tensor is not None:
                num_views = vg.image_tensor.shape[1]  # V
                for v in range(num_views):
                    color_pil = self.to_pil(vg.image_tensor[b, v])  # (H, W, C)
                    self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/color"] = color_pil

        self.log_images(wandb_images, step=epoch, prefix="")


# =====================================================================
# Trellis2VisualIO - Trellis2 专用可视化
# =====================================================================


class Trellis2VisualIO(VisualIO):
    """
    Trellis2 专用可视化保存（自动适配 Shape / Tex / Shape+Tex）。
    
    与 VisualIO 的区别：
    - Trellis2 的 ViewsGenerated 包含 shape_tensor（法线）和 pbr_tensor（RGB），
      而非单一的 image_tensor。
    - 训练可视化按阶段分别保存：
      - save_shape_train: [cond | normal | edited_normal]  → {name}_shape.png
      - save_tex_train:   [cond | rgb   | edited_rgb]      → {name}_tex.png
    - 评估可视化分文件保存：
      - normal.png（shape_tensor 有值时）
      - color.png（pbr_tensor 有值时）
      - mesh.obj（export_mesh=True 时）
    
    使用方式：
        # shape-only:
        visual_io.save_shape_train(state, epoch, step)
        
        # tex-only:
        visual_io.save_tex_train(state, epoch, step)
        
        # shape+tex:
        visual_io.save_shape_train(state, epoch, step)  # shape guidance 之后
        # ... tex forward + tex guidance ...
        visual_io.save_tex_train(state, epoch, step)     # tex guidance 之后
    """

    def _save_stage_train(
        self, state, epoch: int, step: int,
        render_tensor,  # (B, V, H, W, C) 渲染结果
        suffix: str,    # 文件后缀，如 "_shape" 或 "_tex"
        wandb_prefix: str = "train",
    ) -> None:
        """
        内部方法：保存单个阶段的训练可视化网格。
        
        网格内容: [cond | render | edit]
        目录结构: root/epoch_{N}/step_{M}/{name}_{suffix}.png
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        
        conds = state.views_conditioned.image_pils
        edits = state.views_edited.color_tensor  # 当前阶段的 edit（调用时机决定内容）
        
        wandb_images = {}
        for b, name in enumerate(names):
            imgs = [composite_alpha_to_white(conds[b])]
            
            # 渲染结果
            if render_tensor is not None:
                imgs.append(self.to_pil(render_tensor[b, 0]))
            
            # 编辑后图像（guidance edit）
            if edits is not None:
                imgs.append(self.to_pil(edits[b, 0].permute(1, 2, 0)))
            
            grid = self.make_grid(imgs)
            self.save_pil(grid, out_dir / f"{name}{suffix}.png")
            
            if b < self.max_wandb_samples:
                wandb_images[f"{wandb_prefix}/{name}{suffix}"] = grid
        
        self.log_images(wandb_images, step=step, prefix="")

    def save_shape_train(self, state, epoch: int, step: int) -> None:
        """
        Shape 阶段训练可视化: [cond | normal | edited_normal]
        
        必须在 shape guidance 之后、tex forward 之前调用，
        否则 state.views_edited 会被 tex guidance 覆盖。
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.shape_tensor,
            suffix="_shape",
        )

    def save_tex_train(self, state, epoch: int, step: int) -> None:
        """
        Tex 阶段训练可视化: [cond | rgb | edited_rgb]
        
        在 tex guidance 之后调用。
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.pbr_tensor,
            suffix="_tex",
        )

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        评估模式：按阶段分别保存渲染图 + 可选 mesh 导出（支持多视角）。
        
        目录结构:
            root/epoch_{N}/{name}/
            ├── normal_v0.png, normal_v1.png, ...   # shape_tensor 有值时保存
            ├── color_v0.png, color_v1.png, ...     # pbr_tensor 有值时保存
            └── mesh.obj                            # export_mesh=True 时保存
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        vg = state.views_generated
        meshes = (render_out or {}).get("meshes", [])
        
        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name
            
            # Normal 图（Shape 阶段）
            if vg.shape_tensor is not None:
                num_views = vg.shape_tensor.shape[1]  # V
                for v in range(num_views):
                    normal_pil = self.to_pil(vg.shape_tensor[b, v])  # (H, W, C)
                    self.save_pil(normal_pil, sample_dir / f"normal_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/normal"] = normal_pil
            
            # RGB 图（Tex 阶段）
            if vg.pbr_tensor is not None:
                num_views = vg.pbr_tensor.shape[1]  # V
                for v in range(num_views):
                    color_pil = self.to_pil(vg.pbr_tensor[b, v])  # (H, W, C)
                    self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/color"] = color_pil
            
            # Mesh 导出
            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))
        
        self.log_images(wandb_images, step=epoch, prefix="")

