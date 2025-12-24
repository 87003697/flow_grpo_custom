import ml_collections


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.save_freq = 5
    cfg.eval_freq = 5
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False

    # === LoRA 配置 ===
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # === 数据路径 ===
    cfg.train_data_dir = "dataset/alphaimages_1k/train/images"
    cfg.eval_data_dir = "dataset/alphaimages_1k/test/images"

    # === 预训练权重 ===
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === 数据批次 ===
    cfg.batch_size = 1
    cfg.eval_batch_size = 1

    # === 相机与渲染配置 (TRELLIS 风格: yaw/pitch/r/fov) ===
    cfg.camera = cam = ml_collections.ConfigDict()
    cam.render_resolution = 256
    cam.ray_height = 256
    cam.ray_width = 256

    # 训练时相机参数
    cam.train = ml_collections.ConfigDict()
    cam.train.n_view = 1                      # 训练时视角数
    cam.train.yaw_range = [180.0, 180.0]        # yaw 采样范围 (度)
    cam.train.pitch_range = [-15.0, 45.0]     # pitch 采样范围 (度)
    cam.train.r_range = [2.0, 2.0]            # 相机距离范围
    cam.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)

    # 评估时相机参数
    cam.eval = ml_collections.ConfigDict()
    cam.eval.n_view = 4                       # 评估时视角数
    cam.eval.yaw = 180                        # 评估时固定 yaw (度)
    cam.eval.pitch = 0.0                     # 评估时固定 pitch (度)
    cam.eval.r = 2.0                          # 评估时相机距离
    cam.eval.fov = 40.0                       # 评估时视场角 (度)

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "lion"
    tr.optimizer.lr = 3e-4
    tr.optimizer.beta1 = 0.9
    tr.optimizer.beta2 = 0.999
    tr.optimizer.weight_decay = 1e-4
    tr.optimizer.eps = 1e-8

    # === Guidance 占位（当前 trellis.py 中的 compute_guidance 使用）===
    cfg.lambda_distill = 0.0
    cfg.loss = ml_collections.ConfigDict()

    return cfg
