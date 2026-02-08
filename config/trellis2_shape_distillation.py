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
    cfg.guidance.type = "flowedit"


    
    # Shape 专用配置
    cfg.run_name = "trellis2_shape_distill"
    
    # 切换到 512 分辨率 pipeline
    cfg.pipeline_type = "512"
    
    # 使用 26 邻居 soft occupancy 可微 Normal 渲染
    cfg.renderer.normal_mode = "neighbor26_soft"

    # 自适应相机距离（Shape 专用）
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9
    
    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9

    """训练超参（optimizer, loss）。"""
    cfg.train.gradient_accumulation_steps = 1

    cfg.train.optimizer.type = "adam"
    cfg.train.optimizer.lr = 3e-5
    
    cfg.train.loss.ssim = 0.0
    cfg.train.loss.lpips = 0.0
    cfg.train.loss.latent_mse = 1.0
    cfg.train.loss.dino = 0.0
    cfg.train.loss.guidance = 1.0
    cfg.train.loss.reg = 1.0
    cfg.train.loss.latent_mse_mode = "weighted"   # "final" | "mean" | "weighted" | "ada" | "ada_position"

    # Guidance 专用配置
    cfg.guidance.flowedit.steps = 40
    cfg.guidance.flowedit.n_max = 25
    cfg.guidance.flowedit.target_prompt = "Move the camera. Convert to normal map."
    cfg.guidance.flowedit.true_cfg_scale_tgt = 20.0
    cfg.guidance.flowedit.fixed_noise = True
    cfg.guidance.flowedit.noise_mode = "fixed"
    cfg.guidance.flowedit.update_mode = "tgt"
    cfg.guidance.flowedit.latent_mse_mode = cfg.train.loss.latent_mse_mode
    cfg.guidance.flowedit.reduce_mode = "mean"
    cfg.guidance.flowedit.ada_normalize = True
    cfg.guidance.flowedit.ada_eps = 1e-4
    
    cfg.guidance.flowedit.loss = ml_collections.ConfigDict()
    cfg.guidance.flowedit.loss.ssim = cfg.train.loss.ssim
    cfg.guidance.flowedit.loss.lpips = cfg.train.loss.lpips
    cfg.guidance.flowedit.loss.latent_mse = cfg.train.loss.latent_mse
    cfg.guidance.flowedit.loss.latent_csd = 0.0
    cfg.guidance.flowedit.loss.latent_delta = 0.0
    cfg.guidance.flowedit.loss.dino = cfg.train.loss.dino

    return cfg
