"""TRELLIS.2 蒸馏训练基础配置（按模块拆分）。

每个字段都经过验证，确保在 edit4shape/ 代码中被实际读取。
未使用的字段已清理（详见 git log）。
"""
import ml_collections


def get_base_config_general():
    """通用配置（seed, epochs, 频率等）。"""
    cfg = ml_collections.ConfigDict()
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    cfg.verbose = False
    cfg.pipeline_type = "1024"
    cfg.use_wandb = False  # 是否启用 wandb 日志

    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1
    cfg.freq.save.ckpt = 1
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数）
    cfg.freq.eval = 1
    cfg.freq.profiler = 1 # PhaseProfiler 汇总打印频率（每 N 步打印一次平均值）

    # 正则化配置
    # - "none": 不使用正则化
    # - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
    # - "v": MSE(v_stu, v_tea)，梯度仅当前步
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "x0"    # none | x0 | v
    return cfg


def get_base_config_data():
    """数据配置（训练/评估）。"""
    cfg = ml_collections.ConfigDict()

    cfg.train = ml_collections.ConfigDict()
    cfg.train.dir = "dataset/alphaimages_v2/train"
    cfg.train.batch_size = 1
    cfg.train.n_view = 1
    cfg.train.yaw_range = [180.0, 180.0]
    cfg.train.pitch_range = [0.0, 0.0]  # 固定 pitch 角度
    cfg.train.r_range = [2.0, 2.0]
    cfg.train.fov_range = [40.0, 40.0]
    cfg.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.train.adaptive_distance.enabled = True
    cfg.train.adaptive_distance.fill_ratio = 0.9

    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.dir = "dataset/alphaimages_v2/test"
    cfg.eval.batch_size = 1
    cfg.eval.n_view = 4
    cfg.eval.yaw = 180
    cfg.eval.pitch = 0.0
    cfg.eval.r = 2.0
    cfg.eval.fov = 40.0
    cfg.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.eval.adaptive_distance.enabled = True
    cfg.eval.adaptive_distance.fill_ratio = 0.9
    return cfg


def get_base_config_pretrained():
    """预训练权重路径。"""
    cfg = ml_collections.ConfigDict()
    cfg.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


def get_base_config_renderer():
    """渲染器配置。

    注意: renderer.type 已移至各系统自己的配置文件（trellis2_shape 不使用，
    仅 trellis.py / trellis_nabla.py 老版系统需要）。
    """
    cfg = ml_collections.ConfigDict()
    cfg.resolution = 1024
    cfg.ssaa = 1
    cfg.bg_color = [1.0, 1.0, 1.0]
    cfg.near = 1.0
    cfg.far = 100.0

    # Normal 渲染模式：
    # - "mesh": Mesh Normal（nvdiffrast，dual_vertices 可微，intersected detach）
    # - "mesh_peeled": face_normal + DepthPeeler alpha（dual_vertices + intersect_logits 均可微）
    cfg.normal_mode = "mesh"

    # DepthPeeler 参数（mesh_peeled 使用）
    cfg.peel_layers = 8             # DepthPeeler 剥离层数
    cfg.grad_checkpoint = True      # per-layer gradient checkpoint
    return cfg


def get_base_config_train():
    """训练超参（optimizer + loss 总权重）。

    注意: 细分 loss 权重（ssim/lpips/latent_mse/dino）由各 Guidance 子配置管理，
    train.loss 只保留训练循环实际读取的总权重。
    """
    cfg = ml_collections.ConfigDict()

    # 训练模式: "lora" | "full"
    # - "lora": LoRA 微调（默认，显存友好）
    # - "full": 全参微调（需要更多显存，加载独立教师模型）
    cfg.mode = "full"
    if cfg.mode == "lora":
        cfg.lora = ml_collections.ConfigDict()
        cfg.lora.lora_rank = 32

    cfg.gradient_accumulation_steps = 4

    cfg.optimizer = ml_collections.ConfigDict()
    cfg.optimizer.type = "sgd"
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 0

    # Loss 总权重（训练循环中统一乘以各 guidance/reg loss）
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.guidance = 1.0  # Guidance loss 总权重
    cfg.loss.reg = 0.1       # 正则化 loss 总权重
    return cfg


def _flowedit_config(g: ml_collections.ConfigDict) -> None:
    """FlowEdit 专用配置。

    所有字段均在 edit4shape/guidance/paradigms/flowedit.py
    或 edit4shape/guidance/pipelines/adapters.py 中被读取。
    """
    g.flowedit = ml_collections.ConfigDict()

    # 随机种子（FlowEdit Pipeline 的 generator 种子）
    g.flowedit.seed = 42

    # Pipeline 类型: "simple" | "full"
    # - "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
    # - "full": FlowEditFullPipeline，双分支都使用模型推理（效果更好）
    g.flowedit.pipeline_type = "full"

    # num_inference_steps: 总时间步数
    g.flowedit.steps = 12
    # 实际执行的最后 n_max 步
    g.flowedit.n_max = 9

    # 噪声模式: "random" | "fixed" | "aligned" | "traj_*"
    # - random: 每步随机噪声
    # - fixed: 固定噪声（所有 step 共用）
    # - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    # - traj_*: 轨迹对齐噪声更新（traj_cond / traj_uncond / traj_cfg）
    g.flowedit.noise_mode = "aligned"
    # 是否启用 MTS 时间步采样（simple/full 均可用）
    g.flowedit.use_mts_sampling = True

    # Target 分支参数
    g.flowedit.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.flowedit.negative_prompt_tgt = " "  # target 分支的 negative prompt
    g.flowedit.true_cfg_scale_tgt = 4.0
    # Source 分支参数（full 模式需要；simple 模式下不会读取）
    g.flowedit.true_cfg_scale_src = -1 * g.flowedit.true_cfg_scale_tgt
    g.flowedit.source_prompt = g.flowedit.target_prompt
    g.flowedit.negative_prompt_src = g.flowedit.negative_prompt_tgt

    # 多步 Loss 配置（分离聚合方式和归一化方式）
    # reduce_mode: 聚合方式
    #   - "final": 只用最后一步
    #   - "mean": 均匀加权
    #   - "weighted": 1/k 加权（前期大）
    #   - "inv_weighted": k/K 加权（后期大）
    g.flowedit.reduce_mode = "mean"
    # ada_normalize: 是否使用自适应归一化
    #   - True: 梯度归一化（稳定训练）
    #   - False: 标准 MSE
    g.flowedit.ada_normalize = True
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    g.flowedit.ada_eps = 1e-1

    # ========== Loss 权重配置 ==========
    # FlowEdit 专属 loss 权重（仅对 flowedit 类型有效）
    g.flowedit.loss = ml_collections.ConfigDict()
    # 核心蒸馏 loss（latent space）
    g.flowedit.loss.latent_mse = 0.0    # MSE: MSE(src, z_edit)
    g.flowedit.loss.latent_csd = 1.0    # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
    # 辅助 loss（pixel / feature space）
    g.flowedit.loss.ssim = 0.0          # SSIM loss（像素级结构）
    g.flowedit.loss.lpips = 0.0         # LPIPS loss（感知特征）
    g.flowedit.loss.dino = 0.0          # DINO loss（语义特征）


def get_base_config_guidance():
    """Guidance 配置。

    当前支持的 Guidance 类型:
    - "flowedit": FlowEdit（编辑图像 → 计算相似度 loss）
    - "distillation": 蒸馏（单步/多步，SDS/CSD 变体）

    注意: 老版 SDS/CSD/CSD-Rev 实现已废弃，create_guidance() 不再支持。
    """
    cfg = ml_collections.ConfigDict()

    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    cfg.type = "flowedit"

    # 模型路径（HuggingFace ID 或本地路径）
    cfg.model_path = "Qwen/Qwen-Image-Edit-2511"
    # 工作分辨率
    cfg.edit_resolution = 1024

    _flowedit_config(cfg)

    return cfg
