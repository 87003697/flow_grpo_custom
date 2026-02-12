import ml_collections


def _flowedit_config(g: ml_collections.ConfigDict, loss_cfg: ml_collections.ConfigDict) -> None:
    """FlowEdit 专用配置"""
    g.flowedit = ml_collections.ConfigDict()
    
    # Pipeline 类型: "simple" | "full"
    # - "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
    # - "full": FlowEditPipeline，双分支都使用模型推理（效果更好）
    g.flowedit.pipeline_type = "simple"
    
    g.flowedit.seed = 0
    g.flowedit.n_max = 10
    g.flowedit.steps = 20
    g.flowedit.noise_mode = "fixed"
    g.flowedit.use_mts_sampling = True
    
    g.flowedit.true_cfg_scale_tgt = 8.0
    g.flowedit.target_prompt = "Move the camera"
    g.flowedit.negative_prompt_tgt = " "
    
    # "full" 模式专用参数（仅当 pipeline_type="full" 时生效）
    g.flowedit.true_cfg_scale_src = 4.0              # source branch CFG scale
    g.flowedit.source_prompt = "Reconstruct the image"                    # 描述原图的 prompt
    g.flowedit.negative_prompt_src = " "
    
    # reduce_mode: 聚合方式
    g.flowedit.reduce_mode = "mean"
    # ada_normalize: 是否使用自适应归一化
    g.flowedit.ada_normalize = True
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    g.flowedit.ada_eps = 1e-4
    
    # FlowEdit 专属 loss 权重（仅对 flowedit 类型有效）
    g.flowedit.loss = ml_collections.ConfigDict()
    g.flowedit.loss.ssim = loss_cfg.ssim
    g.flowedit.loss.lpips = loss_cfg.lpips
    g.flowedit.loss.latent_mse = loss_cfg.latent_mse
    g.flowedit.loss.latent_csd = 0.0
    g.flowedit.loss.dino = loss_cfg.dino


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    
    # === 频率控制 ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 2  # 训练可视化保存步频
    cfg.freq.save.ckpt = 5    # ckpt 保存频率（epoch）
    cfg.freq.eval = 5         # 评估频率（epoch）

    # === LoRA 配置 ===
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()
    
    # 训练数据配置
    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v2/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1                      # 训练时视角数
    cfg.data.train.yaw_range = [180.0, 180.0]      # yaw 采样范围 (度)
    cfg.data.train.pitch_range = [-15.0, 45.0]     # pitch 采样范围 (度)
    cfg.data.train.r_range = [2.0, 2.0]            # 相机距离范围
    cfg.data.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)
    
    # 评估数据配置
    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v2/test"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 4                       # 评估时视角数
    cfg.data.eval.yaw = 180                        # 评估时固定 yaw (度)
    cfg.data.eval.pitch = 0.0                      # 评估时固定 pitch (度)
    cfg.data.eval.r = 2.0                          # 评估时相机距离
    cfg.data.eval.fov = 40.0                       # 评估时视场角 (度)

    # === 预训练权重 ===
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS.2-4B"  # TRELLIS.2 预训练模型（本地路径）
    pretrained.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"  # DINOv3 本地路径
    
    # === 详细日志 ===
    cfg.verbose = False  # 是否打印详细日志
    
    # === Pipeline 类型 ===
    # TRELLIS.2 支持多种模式：
    #   - "512": 仅 512 分辨率
    #   - "1024": 仅 1024 分辨率（非 cascade，推荐）
    #   - "1024_cascade": 512 → 1024 cascade
    #   - "1536_cascade": 512 → 1536 cascade
    cfg.pipeline_type = "1024"

    # === Renderer 配置 ===
    cfg.renderer = renderer = ml_collections.ConfigDict()
    renderer.resolution = 1024  # 渲染分辨率，FlowEdit 要求 1024×1024
    renderer.type = "mesh"  # 可选: mesh / voxel（TRELLIS.2 使用 mesh 或 PBR voxel）
    renderer.ssaa = 1  # 超采样倍数
    renderer.bg_color = [1.0, 1.0, 1.0]
    renderer.near = 1.0  # 近裁剪面（与参考实现 render_utils.py 默认值一致）
    renderer.far = 100.0 # 远裁剪面
    # 环境贴图路径（相对于项目根目录，指向 TRELLIS.2 参考代码中的 HDRI）
    renderer.envmap_path = "_reference_codes/TRELLIS.2/assets/hdri/forest.exr"

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    tr.gradient_accumulation_steps = 1  # 临时设为 1 测试
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "sgd"
    tr.optimizer.lr = 3e-5
    tr.optimizer.weight_decay = 1e-4
    tr.optimizer.eps = 1e-4
    
    # Loss 权重配置（Shape 和 Tex 阶段统一使用）
    tr.loss = ml_collections.ConfigDict()
    tr.loss.ssim = 1.0          # SSIM loss 权重
    tr.loss.lpips = 0.0         # LPIPS loss 权重
    tr.loss.latent_mse = 1.0    # Latent MSE loss 权重
    tr.loss.dino = 0.0          # DINO loss 权重
    tr.loss.guidance = 1.0      # Guidance loss 权重（统一控制 flowedit/sds/csd/csd_rev）
    tr.loss.reg = 1.0           # 正则化 loss 权重（DMD/KL）

    # === 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    cfg.reg = reg = ml_collections.ConfigDict()
    reg.type = "x0"  # 正则化类型: "none" | "x0" | "v"
                      # - "none": 不使用正则化
                      # - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
                      # - "v": MSE(v_stu, v_tea)，梯度仅当前步

    # === Guidance 配置 ===
    # FlowEdit 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → FlowEdit 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    g.type = "flowedit"
    
    # 模型路径（HuggingFace ID 或本地路径）
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    
    # 工作分辨率
    g.edit_resolution = 1024
    
    
    # 加载对应的专用配置
    _flowedit_config(g, tr.loss)

    return cfg
