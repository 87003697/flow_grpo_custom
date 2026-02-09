"""TRELLIS.2 Tex 阶段蒸馏训练配置（Shape 冻结，只训练 Tex）。

对应模块: edit4shape.systems.trellis2_tex

本文件仅覆盖与 trellis2_base 默认值不同的字段。
"""
from config.trellis2_base import (
    get_base_config_general,
    get_base_config_data,
    get_base_config_pretrained,
    get_base_config_renderer,
    get_base_config_train,
    get_base_config_reg,
    get_base_config_guidance,
)


def get_config():
    # === 基础配置（从 trellis2_base 组装）===
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.train = get_base_config_train()
    cfg.reg = get_base_config_reg()
    cfg.guidance = get_base_config_guidance()

    # === General ===
    cfg.run_name = "trellis2_tex_distill"

    # === Renderer ===
    cfg.renderer.envmap_path = "_reference_codes/TRELLIS.2/assets/hdri/forest.exr"

    return cfg
