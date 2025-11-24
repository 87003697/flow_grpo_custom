from dataclasses import dataclass


@dataclass
class ScorerConfig:
    normal_resolution: int
    cache_dir: str
    encoder: str = "dino_v2"
    dino_v2_path: str = "pretrained_weights/dinov2-giant"
    dino_v3_path: str = "pretrained_weights/dinov3-vitb14"
    # PickScore 图像编码器（可选）
    pickscore_model_id: str = "pretrained_weights/pickscore/PickScore_v1"
    pickscore_processor_id: str = "pretrained_weights/pickscore/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # CLIP 图像编码器（可选）
    clip_model_id: str = "pretrained_weights/clip/clip-vit-large-patch14"
    clip_processor_id: str = "pretrained_weights/clip/clip-vit-large-patch14"
    # HPSv2 图像编码器（可选）
    hpsv2_ckpt_path: str = "pretrained_weights/hpsv2/HPS_v2.1_compressed.pt"
    # 新增：DINO 相似度模式与 dense-match 分块（可选："cls" / "dense" / "dense_all" / "match_gird2pixel" / "match_pixel"）
    dino_similarity_type: str = "match_pixel"
    dense_match_chunk_size: int = 4096
    save_vis: bool = False
    vis_dir: str = "logs/dino_vis"
    cam_batch_size: int = 64
    render_batch_size: int = 32
    encoding_batch_size: int = 64
    # 新增：VGGT Camera Search 设定
    camera_config_py: str = "_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py"
    use_mesh_support: bool = True
    camera_param_dim: int = 9
    img_size: int = 518
    camera_ckpt: str = ""
    camera_type: str = "search"  # 可选: "search", "fixed_v1"

    # Query 输入对齐参考脚本
    query_input: str = "rgb"  # 可选: "rgb", "normal_pred", "normal_image"
    normal_weights_dir: str = "./pretrained_weights"
    normal_version: str = "yoso-normal-v1-8-1"

    # Mesh 前向方向（与 kiui front_dir 语义一致），用于上游旋转到 +z
    source_front: str = "+z"

    # 是否对同一图像分组内 K 个候选的相机估计结果做均值，并在渲染中复用
    avg_camera_per_group: bool = False

    # 是否使用 RGB 组进行比较（默认使用法线组）
    use_RGB_for_comparison: bool = False

    # VLM (Gemini) 相关默认参数
    vlm_api_key_env: str = "GEMINI_API_KEY"
    vlm_api_key: str = ""
    vlm_model: str = "gemini-2.5-flash-internal"
    vlm_prompt_template: str = (
        "You are a mesh normal evaluator. You receive two normal maps: the first is the "
        "reference target, and the second comes from a generated mesh. Compare alignment "
        "quality (structure, fine details, lighting consistency) and output only "
        "\"Final Score: <float between 0 and 1>\"."
    )
    vlm_score_min: float = 0.0
    vlm_score_max: float = 1.0


