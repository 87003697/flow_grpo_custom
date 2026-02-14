"""TRELLIS.2 Tex 阶段蒸馏训练配置（Shape 冻结，只训练 Tex）。

对应模块: edit4shape.systems.trellis2_tex
"""
from config.trellis2_base import (
    get_base_config_general,
    get_base_config_data,
    get_base_config_pretrained,
    get_base_config_renderer,
    get_base_config_train,
    get_base_config_guidance,
)


def get_config():
    # 组装基础配置
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.train = get_base_config_train()
    cfg.guidance = get_base_config_guidance()

    # Tex 专用配置
    cfg.run_name = "trellis2_tex_distill"
    cfg.guidance.type = "flowedit"

    # Tex 阶段使用 PBR 渲染，需要环境贴图
    cfg.renderer.envmap_path = "_reference_codes/TRELLIS.2/assets/hdri/forest.exr"

    return cfg
