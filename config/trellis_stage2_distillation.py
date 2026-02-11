import ml_collections



def _flowedit_config(g: ml_collections.ConfigDict):
    """FlowEdit 专用配置"""
    g.flowedit = ml_collections.ConfigDict()
    
    # Pipeline 类型: "simple" | "full"
    # - "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
    # - "full": FlowEditFullPipeline，双分支都使用模型推理（效果更好）
    g.flowedit.pipeline_type = "full"
    
    g.flowedit.seed = 0
    g.flowedit.steps = 40   # num_inference_steps: 总时间步数
    g.flowedit.n_max = 20   # 实际执行的最后 n_max 步
    
    # 噪声模式
    # pipeline_type="simple" 支持:
    #   - random / fixed / aligned
    #   - traj_cond / traj_uncond / traj_cfg: DNAEdit 轨迹对齐
    # pipeline_type="full" 支持:
    #   - random: 每步随机噪声
    #   - fixed: 固定噪声（所有 step 共用）
    #   - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    g.flowedit.noise_mode = "aligned"
    
    
    # MTS 采样: 是否使用均匀分区随机采样（与 Distillation 一致）
    # - False: 使用 scheduler 的固定时间步序列
    # - True: 在 [0.02, 0.98] 范围内均匀分区随机采样 steps 个时间步，执行后 n_max 步
    g.flowedit.use_mts_sampling = True

    g.flowedit.true_cfg_scale_tgt = 4
    g.flowedit.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.flowedit.negative_prompt_tgt = " "  # target 分支的 negative prompt

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
    g.flowedit.ada_eps = 1e-2
    
    # ========== Loss 权重配置 ==========
    g.flowedit.loss = ml_collections.ConfigDict()
    
    # 核心蒸馏 loss（latent space，支持多步聚合 + ada normalize）
    g.flowedit.loss.latent_mse = 0.0   # MSE: MSE(src, z_edit)
    g.flowedit.loss.latent_csd = 1.0   # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
    g.flowedit.loss.latent_delta = 0.0 # Delta: MSE(src, delta_pos) - MSE(src, delta_neg)，速度分解对比
    
    # 辅助 loss（pixel / feature space）
    g.flowedit.loss.ssim = 0.0         # SSIM loss（像素级结构）
    g.flowedit.loss.lpips = 0.0        # LPIPS loss（感知特征）
    g.flowedit.loss.dino = 0.0         # DINO loss（语义特征）
    
    # "full" 模式专用参数（仅当 pipeline_type="full" 时生效）
    g.flowedit.true_cfg_scale_src = -1 * g.flowedit.true_cfg_scale_tgt
    g.flowedit.source_prompt = g.flowedit.target_prompt
    g.flowedit.negative_prompt_src = g.flowedit.negative_prompt_tgt



def _distillation_config(g: ml_collections.ConfigDict):
    """蒸馏配置（支持 MTS 多时间步采样）
    
    x0 预测定义：
        - x0_pos: 纯 cond 预测 (v_cond)，CSD 正样本
        - x0_neg: 纯 uncond 预测 (v_uncond)，CSD 负样本
        - x0_cfg: CFG 后预测 (v_cfg = v_uncond + scale * (v_cond - v_uncond))
    
    通过 mse_weight 和 csd_weight 控制 loss 类型：
        - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_cfg)
        - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
        - mse_weight=1, csd_weight=1 → 混合模式
    
    CSD 相比 MSE 的优势：
        - 更稳定：避免了噪声带来的方差问题
        - 更好的信号：利用对比差分捕捉"增强方向"
    """
    g.distillation = ml_collections.ConfigDict()
    
    g.distillation.seed = 0
    g.distillation.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    g.distillation.max_step_percent = 0.50   # 最大时间步百分比（0.50 = t=500）
    
    g.distillation.true_cfg_scale = 12        # CFG scale（高 CFG 分支强度）
    
    # Loss 权重（控制 MSE/CSD 模式）
    g.distillation.mse_weight = 0.0          # MSE loss 权重: MSE(src, x0_cfg) — 蒸馏到 CFG 后预测
    g.distillation.csd_weight = 1.0          # CSD loss 权重: MSE(src, x0_pos) - MSE(src, x0_neg) — 对比纯 cond vs uncond
    
    # MTS（多时间步采样）配置
    g.distillation.num_timesteps = 20        # 采样时间步数量（1=单步，>1=MTS 多时间步）
    g.distillation.reduce_mode = "mean"      # 多步 loss 聚合方式: "final" | "mean" | "weighted" | "inv_weighted"
    
    # 梯度归一化配置（与 flowedit 一致）
    g.distillation.ada_normalize = True      # 是否使用自适应梯度归一化（稳定训练）
    g.distillation.ada_eps = 1e-2            # 归一化 epsilon（防止除零）
    
    # 噪声模式配置
    # - "random": 每次随机噪声
    # - "fixed": 固定噪声
    # - "aligned": DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    # - "inversion_cond/uncond/cfg": Naive Inversion（Euler 积分反演）
    # - "traj_cond/uncond/cfg": DNAEdit 轨迹对齐 ε -= (v_theoretical - v_model) * t
    #     其中 v_theoretical = noise - x_src
    g.distillation.noise_mode = "fixed"
    
    # Prompt 配置
    g.distillation.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.distillation.negative_prompt = " "


def _lora_config(cfg: ml_collections.ConfigDict):
    """LoRA 配置（仅在非 full 模式下使用）。"""
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.lora.lora_alpha = 32  # LoRA alpha（通常与 rank 相同）
    cfg.lora.lora_dropout = 0.0  # LoRA dropout
    cfg.lora.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]  # 目标模块


def _sde_rollout_config(cfg: ml_collections.ConfigDict):
    """SDE Rollout 专用配置（仅当 rollout.type == "sde" 时使用）。"""
    cfg.rollout.noise_level = 0.7  # 噪声水平 (0~1)
    cfg.rollout.sde_type = "cps"   # SDE 类型: "sde" | "cps"


def _adaptive_distance_config(cfg: ml_collections.ConfigDict):
    """统一添加 adaptive_distance 配置（假设 cfg.data.train / cfg.data.eval 已创建）。"""
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9

    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
    cfg.use_wandb = False  # 是否启用 wandb 日志
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    
    # === 频率控制 ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1  # 训练可视化保存步频
    cfg.freq.save.ckpt = 10000    # ckpt 保存频率（epoch）
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数：4, 9, 16...）
    cfg.freq.eval = 1         # 评估频率（epoch）

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()
    
    # 训练数据配置
    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v2/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1                      # 训练时视角数
    cfg.data.train.yaw_range = [180.0, 180.0]      # yaw 采样范围 (度)
    cfg.data.train.pitch_range = [0.0, 0.0]     # pitch 采样范围 (度)
    cfg.data.train.r_range = [2.0, 2.0]            # 相机距离范围
    cfg.data.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)
    
    # 评估数据配置
    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v2/test"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 1                       # 评估时视角数
    cfg.data.eval.yaw = 180                        # 评估时固定 yaw (度)
    cfg.data.eval.pitch = 0.0                      # 评估时固定 pitch (度)
    cfg.data.eval.r = 2.0                          # 评估时相机距离
    cfg.data.eval.fov = 40.0                       # 评估时视场角 (度)
    _adaptive_distance_config(cfg)

    # === 预训练权重 ===
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置 ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024  # 渲染分辨率，FlowEdit 要求 1024×1024
    cfg.renderer.type = "gs"  # 可选: mesh / gs
    cfg.renderer.ssaa = 1  # 超采样倍数
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]
    if cfg.renderer.type == "mesh":
        cfg.renderer.near, cfg.renderer.far = 1.0, 100.0
    else:
        cfg.renderer.near, cfg.renderer.far = 0.8, 1.6

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    
    # 训练模式: "full" | "lora" | "frozen"
    # - "full": 全参微调，教师模型为冻结副本（放在 guidance 设备）
    # - "lora": LoRA 微调，教师模型为禁用 adapter 后的原始权重
    # - "frozen": 冻结模式（仅推理，不训练）
    tr.mode = "full"

    # LoRA 配置：仅在非 full 模式下添加
    if tr.mode != "full":
        _lora_config(cfg)
    
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "sgd"
    tr.optimizer.lr = 5e-3
    tr.optimizer.weight_decay = 0.0

    # === 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    # - "none": 不使用正则化
    # - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
    # - "v": MSE(v_stu, v_tea)，梯度仅当前步
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "none"  # 正则化类型: "none" | "x0" | "v"

    # === Rollout 配置 ===
    # 控制采样方式：ODE（确定性）或 SDE（随机）
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"  # "ode" | "sde"
    
    # SDE 专用配置（仅当 rollout.type == "sde" 时生效）
    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # === Guidance 配置（通用）===
    # Guidance 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → Guidance 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    # - "flowedit": FlowEdit 编辑式蒸馏（多步，生成编辑图像）
    # - "distillation": 单步蒸馏（SDS/CSD，通过权重控制）
    g.type = "flowedit"
    
    # 模型路径（HuggingFace ID 或本地路径）
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    
    # 工作分辨率
    g.edit_resolution = 1024

    # 根据 type 加载对应的专用配置
    if g.type == "flowedit":
        _flowedit_config(g)
    elif g.type == "distillation":
        _distillation_config(g)
    else:
        raise ValueError(f"Unknown guidance type: {g.type}. Choose from: flowedit, distillation")

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0     # Guidance loss 权重（统一控制 flowedit/sds/csd/csd_rev）
    tr.loss.reg = 0.           # 正则化 loss 权重（DMD/KL）
    
    return cfg
