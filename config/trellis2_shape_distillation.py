"""TRELLIS.2 Shape 阶段蒸馏训练配置（仅训练 Shape Flow Model）。

对应模块: edit4shape.systems.trellis2_shape

本文件仅覆盖与 trellis2_base 默认值不同的字段。
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
    # === 基础配置（从 trellis2_base 组装）===
    cfg = get_base_config_general()
    cfg.data = get_base_config_data()
    cfg.pretrained = get_base_config_pretrained()
    cfg.renderer = get_base_config_renderer()
    cfg.train = get_base_config_train()
    cfg.reg = get_base_config_reg()
    cfg.guidance = get_base_config_guidance()

    # === General ===
    cfg.run_name = "trellis2_shape_distill"

    # === Pipeline ===
    cfg.pipeline_type = "1024"  # Shape 阶段使用 512 分辨率

    # === Renderer ===
    cfg.renderer.normal_mode = "hybrid26"  # 26-neighbor occupancy + grid_sample_3d（subs 可微）

    # === 数据（Shape 专用：自适应相机距离）===
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9

    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9

    # === Guidance: FlowEdit 覆盖项 ===
    cfg.guidance.flowedit.target_prompt = "Move the camera. Convert to normal map."
    cfg.guidance.flowedit.true_cfg_scale_tgt = 20.0

    return cfg
