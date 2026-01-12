"""TRELLIS.2 Shape 阶段蒸馏训练配置（仅训练 Shape Flow Model）。

对应模块: edit4shape.systems.trellis2_shape
"""
import ml_collections
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
    # 组装基础配置
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.train = get_base_config_train()
    cfg.reg = get_base_config_reg()
    cfg.guidance = get_base_config_guidance()
    
    # Shape 专用配置
    cfg.run_name = "trellis2_shape_distill"
    
    # 自适应相机距离（Shape 专用）
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9
    
    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9

    # Guidance 专用配置
    cfg.guidance.flowedit.steps = 40
    cfg.guidance.flowedit.n_max = 25
    cfg.guidance.flowedit.n_min = 2
    cfg.guidance.flowedit.cfg_rescale = True
    cfg.guidance.flowedit.shared_noise = True
    cfg.guidance.flowedit.target_prompt = "Move the camera. Convert to normal map."
    cfg.guidance.flowedit.true_cfg_scale_tgt = 12.0

    return cfg
