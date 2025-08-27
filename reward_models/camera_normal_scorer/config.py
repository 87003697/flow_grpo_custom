from dataclasses import dataclass


@dataclass
class ScorerConfig:
    normal_resolution: int
    cache_dir: str
    encoder: str = "dino_v2"
    dino_v2_path: str = "pretrained_weights/dinov2-base"
    dino_v3_path: str = "pretrained_weights/dinov3-vitb14"
    save_vis: bool = False
    vis_dir: str = "logs/dino_vis"
    cam_batch_size: int = 64
    render_batch_size: int = 32
    dino_batch_size: int = 64
    # 新增：VGGT Camera Search 设定
    camera_config_py: str = "_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py"
    use_mesh_support: bool = True
    camera_param_dim: int = 9
    img_size: int = 518
    camera_ckpt: str = ""

    # Query 输入对齐参考脚本
    query_input: str = "rgb"  # 可选: "rgb", "normal_pred", "normal_image"
    normal_weights_dir: str = "./pretrained_weights"
    normal_version: str = "yoso-normal-v1-8-1"

    # Mesh 前向方向（与 kiui front_dir 语义一致），用于上游旋转到 +z
    source_front: str = "+z"


