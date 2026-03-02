"""TRELLIS.2 Shape 阶段蒸馏训练配置（仅训练 Shape Flow Model）。

对应模块: edit4shape.systems.trellis2.entries.shape_*

配置内容（mode="shape"）:
    cfg.render_base    → 共享渲染基础（resolution, ssaa, near, far）
    cfg.guidance_init  → Guidance 初始化（model_path, flowedit.{steps, n_max, ...}）
    cfg.shape.renderer → Shape 专有（type, grad_checkpoint）
    cfg.shape.train    → Shape 训练超参（optimizer, loss）
    cfg.shape.guidance → Shape Guidance 运行时（prompt, loss 权重, ...）

★ 不含 cfg.tex（shape-only 模式不需要 Tex 阶段配置）。
"""
from config.trellis2_base import get_default_config


def get_config():
    cfg = get_default_config(mode="shape")
    cfg.run_name = "trellis2_shape_distill"
    return cfg
