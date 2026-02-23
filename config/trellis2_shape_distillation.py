"""TRELLIS.2 Shape 阶段蒸馏训练配置（仅训练 Shape Flow Model）。

对应模块: edit4shape.systems.trellis2.shape

配置结构:
    cfg.guidance       → Guidance 初始化（model_path, flowedit.{steps, n_max, ...}）
    cfg.renderer       → 共享渲染基础（resolution, ssaa, near, far, peel_layers）
    cfg.shape.renderer → Shape 专有（peel_layers, grad_checkpoint）
    cfg.shape.train    → Shape 训练超参（optimizer, loss）
    cfg.shape.guidance → Shape Guidance 运行时（prompt, loss 权重, ...）
"""
from config.trellis2_base import (
    get_base_config_general,
    get_base_config_data,
    get_base_config_pretrained,
    get_base_config_renderer,
    get_base_config_guidance,
    get_base_config_shape_stage,
)


def get_config():
    # 组装全局共享配置
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.guidance = get_base_config_guidance()

    # Shape 阶段独立配置
    cfg.shape = get_base_config_shape_stage()

    # Shape 专用覆盖
    cfg.run_name = "trellis2_shape_distill"
    cfg.pipeline_type = "1024"

    return cfg
