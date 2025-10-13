from dataclasses import dataclass


@dataclass
class ScorerConfig:
    """Camera RGB Scorer 配置。
    
    关键差异：
        - 不需要 query_input, normal_weights_dir, normal_version（法线预测仍由 metadata 提供）
        - rgb_resolution: RGB 渲染分辨率
    """
    rgb_resolution: int  # RGB 渲染分辨率
    cache_dir: str
    encoder: str = "dino_v2"
    dino_v2_path: str = "pretrained_weights/dinov2-giant"
    dino_v3_path: str = "pretrained_weights/dinov3-vitb14"
    save_vis: bool = False
    vis_dir: str = "logs/dino_vis_rgb"
    cam_batch_size: int = 64
    render_batch_size: int = 32
    dino_batch_size: int = 64
    
    # VGGT Camera Search 设定（与 normal scorer 相同）
    # 仅支持 VGGTObj 风格："module.path:function_name"
    camera_config_py: str = "training.config.camera_search_seven_view_fixed:get_camera_search_seven_view_config"
    use_mesh_support: bool = True
    camera_param_dim: int = 9
    img_size: int = 518
    camera_ckpt: str = ""
    
    # Mesh 前向方向（与 kiui front_dir 语义一致），用于上游旋转到 +z
    source_front: str = "+z"
