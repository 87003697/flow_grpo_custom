"""TRELLIS.2 Tex 阶段蒸馏训练配置（Shape 冻结，只训练 Tex）。

对应模块: edit4shape.systems.trellis2.entries.tex_*

配置内容（mode="tex"）:
    cfg.render_base    → 共享渲染基础（resolution, ssaa, near, far, peel_layers）
    cfg.guidance_init  → Guidance 初始化（model_path, flowedit.{steps, n_max, ...}）
    cfg.shape.renderer → Shape 专有（冻结渲染器仍需要 type, bg_color 等）
    cfg.tex.renderer   → Tex 专有（envmap_path, peel_layers）
    cfg.tex.train      → Tex 训练超参（optimizer, loss）
    cfg.tex.guidance   → Tex Guidance 运行时（prompt, loss 权重, ...）

★ Shape 阶段完全冻结，cfg.shape 仅提供渲染器参数。
"""
from config.trellis2_base import get_default_config


def get_config():
    cfg = get_default_config(mode="tex")
    cfg.run_name = "trellis2_tex_distill"
    return cfg
