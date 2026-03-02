"""TRELLIS.2 Shape+Tex 双阶段联合蒸馏训练配置。

对应模块: edit4shape.systems.trellis2.entries.shape_tex_*

配置内容（mode="shape_tex"）:
    cfg.render_base    → 共享渲染基础（resolution, ssaa, near, far）
    cfg.guidance_init  → Guidance 初始化（model_path, flowedit.{steps, n_max, ...}）
    cfg.shape.renderer → Shape 专有（type, grad_checkpoint）
    cfg.shape.train    → Shape 训练超参（optimizer, loss）
    cfg.shape.guidance → Shape Guidance 运行时（prompt, loss 权重, ...）
    cfg.tex.renderer   → Tex 专有（envmap_path）
    cfg.tex.train      → Tex 训练超参（optimizer, loss）
    cfg.tex.guidance   → Tex Guidance 运行时（prompt, loss 权重, ...）

★ Shape 和 Tex 各自拥有独立的 train / guidance 配置，
  支持不同的学习率、loss 权重、Guidance prompt 等。
  Guidance 模型只加载一次（cfg.guidance_init），运行时参数 per-stage 传入。
"""
from config.trellis2_base import get_default_config


def get_config():
    cfg = get_default_config(mode="shape_tex")
    cfg.run_name = "trellis2_shape_tex_distill"

    # # ★ Shape / Tex 独立训练超参覆盖示例：
    # # Shape 阶段
    # cfg.shape.train.optimizer.lr = 1e-4
    # cfg.shape.train.loss.guidance = 1.0
    # cfg.shape.train.loss.reg = 1e-4

    # # Tex 阶段
    # cfg.tex.train.optimizer.lr = 1e-4
    # cfg.tex.train.loss.guidance = 0.01
    # cfg.tex.train.loss.reg = 1e-4

    return cfg
