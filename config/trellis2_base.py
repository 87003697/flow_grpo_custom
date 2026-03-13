"""TRELLIS.2 蒸馏训练基础配置。

唯一公开 API：get_default_config(mode)
    mode: "shape" | "tex" | "shape_tex"

返回完整的、开箱即用的默认配置。实验配置只需 import 后覆盖差异字段。

配置层级结构：
    cfg:
        # ===== 全局 =====
        seed, logdir, run_name, num_epochs, mixed_precision, pipeline_type, ...
        gradient_accumulation_steps
        freq: {save, eval, profiler}
        reg: {type}
        data: {train, eval}
        pretrained: {model, dino_local_path}

        # ===== 渲染基础（Shape/Tex 共享） =====
        render_base:
            resolution, ssaa, near, far, bg_color, peel_layers

        # ===== Guidance 初始化（全阶段共享，只加载模型） =====
        guidance_init:
            type, model_path, edit_resolution, bg_color
            flowedit: {steps, n_max, noise_mode, ...}

        # ===== Shape 阶段（所有 mode 都包含） =====
        shape:
            renderer: {type, grad_checkpoint, bg_color, grad_shrink_scale}
            train:    {mode, optimizer, loss}
            guidance: {seed, target_prompt, ..., loss: {...}}

        # ===== Tex 阶段（仅 mode="tex" / "shape_tex"） =====
        tex:
            renderer: {envmap_path, bg_color, grad_shrink_scale}
            train:    {mode, optimizer, loss}
            guidance: {seed, target_prompt, ..., loss: {...}}

★ Guidance 配置分两层：
  - cfg.guidance_init: 初始化配置（model_path 等），只加载一次模型
  - cfg.{stage}.guidance: 运行时配置（prompt, loss 权重等），每次调用 compute_guidance 传入
"""
import ml_collections
from typing import Literal


# =====================================================================
# 唯一公开 API
# =====================================================================

def get_default_config(mode: Literal["shape", "tex", "shape_tex"] = "shape_tex"):
    """返回完整的 Trellis2 默认配置。

    Args:
        mode: 训练模式
            - "shape":     包含 cfg.shape，不含 cfg.tex
            - "tex":       包含 cfg.shape（冻结渲染器需要）+ cfg.tex
            - "shape_tex": 包含 cfg.shape + cfg.tex
    """
    cfg = ml_collections.ConfigDict()

    # ── 全局参数 ──
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.run_name = "trellis2"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    cfg.verbose = False
    cfg.pipeline_type = "1024"
    cfg.use_wandb = False
    cfg.gradient_accumulation_steps = 4

    cfg.freq = _build_freq()
    cfg.reg = _build_reg()

    # ── 数据 ──
    cfg.data = _build_data()

    # ── 预训练权重 ──
    cfg.pretrained = _build_pretrained()

    # ── 渲染基础（Shape/Tex 共享） ──
    cfg.render_base = _build_render_base()

    # ── Guidance 初始化（模型加载，全阶段共享） ──
    cfg.guidance_init = _build_guidance_init()

    # ── Shape 阶段（所有 mode 都需要，tex-only 也需要冻结的 shape 渲染器） ──
    cfg.shape = _build_shape_stage()

    # ── Tex 阶段（仅 tex / shape_tex 模式） ──
    if mode in ("tex", "shape_tex"):
        cfg.tex = _build_tex_stage()

    return cfg


# =====================================================================
# 以下均为私有 helper，不应被外部直接 import
# =====================================================================

def _build_freq():
    """频率配置。"""
    cfg = ml_collections.ConfigDict()
    cfg.save = ml_collections.ConfigDict()
    cfg.save.visual = 1
    cfg.save.ckpt = 1
    cfg.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数）
    cfg.eval = 1
    cfg.profiler = 1  # PhaseProfiler 汇总打印频率（每 N 步打印一次平均值）
    return cfg


def _build_reg():
    """正则化配置。

    - "none": 不使用正则化
    - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
    - "x1": MSE(x0_stu, x0_tea)，不除 t²，小 t 时权重不被放大
    - "v": MSE(v_stu, v_tea)，梯度仅当前步
    """
    cfg = ml_collections.ConfigDict()
    cfg.type = "v"  # none | x0 | x1 | v
    return cfg


def _build_data():
    """数据配置（训练/评估）。"""
    cfg = ml_collections.ConfigDict()

    cfg.train = ml_collections.ConfigDict()
    cfg.train.dir = "dataset/alphaimages_v3/train"
    cfg.train.batch_size = 1
    cfg.train.n_view = 1
    cfg.train.yaw_range = [0.0, 360.0]
    cfg.train.pitch_range = [0.0, 0.0]
    cfg.train.r_range = [2.0, 2.0]
    cfg.train.fov_range = [40.0, 40.0]
    cfg.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.train.adaptive_distance.enabled = True
    cfg.train.adaptive_distance.fill_ratio = 0.9

    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.dir = "dataset/alphaimages_v3/test"
    cfg.eval.batch_size = 1
    cfg.eval.n_view = 6
    cfg.eval.yaw_range = [0.0, 360.0]
    cfg.eval.pitch_range = [0.0, 0.0]
    cfg.eval.r_range = [2.0, 2.0]
    cfg.eval.fov_range = [40.0, 40.0]
    cfg.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.eval.adaptive_distance.enabled = True
    cfg.eval.adaptive_distance.fill_ratio = 0.9
    return cfg


def _build_pretrained():
    """预训练权重路径。"""
    cfg = ml_collections.ConfigDict()
    cfg.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


def _build_render_base():
    """共享渲染基础参数（cfg.render_base）。

    所有渲染器（MeshRenderer / MeshPeeledRenderer）共用的参数。
    阶段专有参数见 cfg.shape.renderer / cfg.tex.renderer。
    """
    cfg = ml_collections.ConfigDict()
    cfg.resolution = 1024
    cfg.ssaa = 1
    cfg.bg_color = [1.0, 1.0, 1.0]
    cfg.near = 1.0
    cfg.far = 100.0
    # MeshPeeledRenderer 默认剥离层数（tex-only 模式冻结 Shape 渲染器的 fallback）
    cfg.peel_layers = 8
    return cfg


def _build_guidance_init():
    """Guidance 初始化配置（cfg.guidance_init，模型加载参数，全阶段共享）。

    ★ 仅包含模型加载所需参数。运行时参数（prompt / loss 权重 / 聚合策略）
    在 per-stage 的 cfg.shape.guidance / cfg.tex.guidance 中配置。

    当前支持的 Guidance 类型:
    - "flowedit": FlowEdit（编辑图像 → 计算相似度 loss）
    - "distillation": 蒸馏（单步/多步，SDS/CSD 变体）
    """
    cfg = ml_collections.ConfigDict()

    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    cfg.type = "flowedit"

    # 模型路径（HuggingFace ID 或本地路径）
    cfg.model_path = "Qwen/Qwen-Image-Edit-2511"
    # 工作分辨率（VAE encode 时使用）
    cfg.edit_resolution = 1024

    # 条件图背景色 float [0,1]，应与 cfg.render_base.bg_color 保持一致
    cfg.bg_color = [1.0, 1.0, 1.0]

    # FlowEdit 专属 init 参数
    _apply_flowedit_init(cfg)

    return cfg


def _apply_flowedit_init(g: ml_collections.ConfigDict):
    """FlowEdit 专属 init 参数（写入 cfg.guidance_init.flowedit）。

    这些参数在训练过程中不会变化，仅在构造 Pipeline 时读取一次。
    """
    g.flowedit = ml_collections.ConfigDict()

    # 采样步数
    g.flowedit.steps = 12   # num_inference_steps: 总时间步数
    g.flowedit.n_max = 9    # 实际执行的最后 n_max 步

    # 噪声模式:
    #   - random: 每步随机噪声
    #   - fixed: 固定噪声（所有 step 共用）
    #   - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    #   - delta: 双分支差分补偿 ε -= (v_cfg_tgt - v_cfg_src) * (1 - t)
    g.flowedit.noise_mode = "aligned"

    # CSD 正/负样本来源
    # pos: "cond" (纯条件,CFG=1) | "cfg" (原始CFG) | "cfg_rescale" (CFG+L2归一化)
    # neg: "uncond" (纯无条件) | "cond" (纯条件)
    g.flowedit.csd_pos_mode = "cfg"       # 默认: 原始CFG预测
    g.flowedit.csd_neg_mode = "uncond"    # 默认: 纯无条件预测

    # 是否用 src 分支的 x0_neg 替换 tgt 分支的 x0_neg
    g.flowedit.remove_tgt_neg = True


# =====================================================================
# Per-stage 阶段配置
# =====================================================================

def _build_flowedit_runtime():
    """FlowEdit 运行时参数（per-stage 调用时传入 compute_guidance）。

    包含 prompt、CFG scales、loss 权重、聚合策略等，
    不同阶段（Shape / Tex）可使用不同值。

    ★ 采样结构参数（steps / n_max / noise_mode / csd_pos_mode / csd_neg_mode）
      在 cfg.guidance_init.flowedit（init 配置）中设置，全阶段共享。
    """
    cfg = ml_collections.ConfigDict()

    # 随机种子（FlowEdit Pipeline 的 generator 种子）
    cfg.seed = 42

    # Target 分支参数
    cfg.target_prompt = "Rotate the camera."
    cfg.negative_prompt_tgt = " "  # target 分支的 negative prompt
    cfg.true_cfg_scale_tgt = 4.0
    # Source 分支参数（full 模式需要；simple 模式下不会读取）
    cfg.true_cfg_scale_src = -1 * cfg.true_cfg_scale_tgt
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # 多步 Loss 配置（分离聚合方式和归一化方式）
    # reduce_mode: 聚合方式
    #   - "final": 只用最后一步
    #   - "mean": 均匀加权
    #   - "weighted": 1/k 加权（前期大）
    #   - "inv_weighted": k/K 加权（后期大）
    cfg.reduce_mode = "final"
    # ada_normalize: 是否使用自适应归一化
    #   - True: 梯度归一化（稳定训练）
    #   - False: 标准 MSE
    cfg.ada_normalize = False
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    cfg.ada_eps = 1e-4

    # ========== Loss 权重配置 ==========
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.latent_mse = 1.0    # MSE: MSE(src, z_edit)
    cfg.loss.latent_csd = 0.0    # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)

    # 分支权重（> 0 时启用对应 tracker 并计算 loss）
    cfg.loss.tgt_branch = 1.0   # target 分支权重
    cfg.loss.src_branch = 0.0   # source 分支权重（= 0 不启用）

    return cfg


def _build_stage_train():
    """阶段训练超参的公共默认值。"""
    cfg = ml_collections.ConfigDict()

    # 训练模式: "lora" | "full" | "frozen"
    cfg.mode = "full"

    cfg.optimizer = ml_collections.ConfigDict()
    cfg.optimizer.type = "adan"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.weight_decay = 0
    if cfg.optimizer.type != "sgd":
        cfg.optimizer.eps = 1e-4

    # Loss 总权重（训练循环中乘以 guidance/reg loss）
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.guidance = 1.0  # Guidance loss 总权重
    cfg.loss.reg = 1e0       # 正则化 loss 总权重
    cfg.loss.guidance_grad_max_norm = 1.0  # per-timestep guidance grad 最大 L2 范数（≤0=不裁剪）

    # Onestep 噪声采样范围（sample_timestep 使用）
    cfg.noise = ml_collections.ConfigDict()
    cfg.noise.t_min = 0.02   # 最小时间步
    cfg.noise.t_max = 0.98   # 最大时间步
    return cfg


def _build_shape_stage():
    """Shape 阶段独立配置（renderer + train + guidance 运行时）。"""
    cfg = ml_collections.ConfigDict()

    # --- Shape 渲染器专有参数 ---
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.type = "hybrid26_peeled"        # "mesh_peeled" | "hybrid26_peeled"
    cfg.renderer.grad_checkpoint = True      # gradient checkpoint（省显存）
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]  # Normal map 背景色（灰色）
    cfg.renderer.grad_shrink_scale = 1.0  # 渲染梯度缩放（< 1.0 抑制梯度，1.0 = 不缩放）

    # --- Shape 训练超参 ---
    cfg.train = _build_stage_train()

    # --- Shape Guidance 运行时配置 ---
    cfg.guidance = _build_flowedit_runtime()
    # Shape 阶段默认使用 Normal map prompt
    cfg.guidance.target_prompt = "Rotate the camera. Convert to normal map."
    cfg.guidance.source_prompt = cfg.guidance.target_prompt

    cfg.train.loss.reg = 1e-0
    return cfg


def _build_tex_stage():
    """Tex 阶段独立配置（renderer + train + guidance 运行时）。"""
    cfg = ml_collections.ConfigDict()

    # --- Tex 渲染器专有参数 ---
    cfg.renderer = ml_collections.ConfigDict()
    # 环境贴图路径（PBR 渲染需要）
    cfg.renderer.envmap_path = "_reference_codes/TRELLIS.2/assets/hdri/forest.exr"
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]  # PBR 背景色（灰色）
    cfg.renderer.grad_shrink_scale = 1.0  # 渲染梯度缩放（< 1.0 抑制梯度，1.0 = 不缩放）

    # --- Tex 训练超参 ---
    cfg.train = _build_stage_train()

    # --- Tex Guidance 运行时配置 ---
    cfg.guidance = _build_flowedit_runtime()
    # Tex 阶段默认使用 RGB prompt
    cfg.guidance.target_prompt = "Rotate the camera."
    cfg.guidance.source_prompt = cfg.guidance.target_prompt

    cfg.train.loss.reg = 1e-1

    return cfg
