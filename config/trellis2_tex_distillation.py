"""TRELLIS.2 Tex 阶段蒸馏训练配置（Shape 冻结，只训练 Tex）。

对应模块: edit4shape.systems.trellis2_tex

配置结构:
    cfg.guidance     → Guidance 初始化（model_path, flowedit.pipeline_type, ...）
    cfg.renderer     → 共享渲染基础（resolution, ssaa, near, far, peel_layers）
    cfg.tex.renderer → Tex 专有（envmap_path, peel_layers）
    cfg.tex.train    → Tex 训练超参（optimizer, loss）
    cfg.tex.guidance → Tex Guidance 运行时（prompt, loss 权重, ...）

★ Shape 阶段完全冻结，使用共享 renderer 默认值（MeshPeeledRenderer + peel_layers），
  不需要提供 cfg.shape。
"""
from config.trellis2_base import (
    get_base_config_general,
    get_base_config_data,
    get_base_config_pretrained,
    get_base_config_renderer,
    get_base_config_guidance,
    get_base_config_tex_stage,
)


def get_config():
    # 组装全局共享配置
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.guidance = get_base_config_guidance()

    # Tex 阶段配置（可训练）
    cfg.tex = get_base_config_tex_stage()

    # Tex 专用覆盖
    cfg.run_name = "trellis2_tex_distill"

    return cfg
