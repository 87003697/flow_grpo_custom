"""TRELLIS.2 Shape 阶段蒸馏训练配置（仅训练 Shape Flow Model）。

对应模块: edit4shape.systems.trellis2_shape
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


    
    # Shape 专用配置
    cfg.run_name = "trellis2_shape_distill"
    
    # 切换到 512 分辨率 pipeline
    cfg.pipeline_type = "1024"
    
    # 使用 26 邻居 soft occupancy 可微 Normal 渲染
    cfg.renderer.normal_mode = "hybrid26" # "mesh" "hybrid26"
    
    # Guidance 专用配置
    cfg.guidance.flowedit.target_prompt = "Move the camera. Convert to normal map."
    cfg.guidance.flowedit.source_prompt = cfg.guidance.flowedit.target_prompt


    return cfg
