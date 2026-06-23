# Session Handoff: Distillation+GAN 实验 → 放弃 → flowedit_denoise 消融

## 对话 Transcript
`~/.claude-internal/projects/-Users-zhiyuanma-Desktop-codes-flow-grpo-custom-v2/d1af7e47-0d7c-4c46-a173-d0d54a6f2e49.jsonl`

## 前序 Session
- `.agents/sessions/2026-06-22-gan-bce-alignment.md` — BCE loss 对齐 + DDP 修复实验提交

## 相关 Plan
- `.agents/plans/distillation-gan.md` — Distillation + GAN 子方案设计（已废弃）

## 任务目的
1. 重构 DiscriminatorHelper，删除 accelerator 依赖
2. 创建全轨迹蒸馏 + GAN 变体并提交实验
3. 提交 flowedit_denoise 的 reg_type=v 消融实验
4. 清理 mixed_precision 代码（全局改 "no"，简化 inference context）

## 执行内容
1. 删除 `ericzyma-bce-r1-w01-ddpfix` 和 `ericzyma-bce-r1-w001-ddpfix` 两个旧实验
2. 以 baseline 为基础提交 reg_type="v" 消融实验（`ericzyma-bce-w001-regv`）
3. 提取 `DiscriminatorHelper` 类（重构 flowedit_gan.py，删除 accelerator 依赖）
4. 新建 `config/trellis_stage2_distillation_gan.py` + 对应启动脚本
5. 在 `standard.py` 补充 guidance checkpoint save/load 调用
6. 提交 distillation+GAN 8-GPU 实验 → 可视化崩坏 → 调低 gan=0.01 + reg=v → **仍无效，放弃**
7. 提交新 flowedit_denoise 实验：gan=0.1, reg=v（`ericzyma-bce-w01-regv`）
8. 全局 mixed_precision="no" 清理 + strategy.py inference context 简化

## 代码改动

### Commits（已推送到 `trellis_distill`）
| commit | message | 关键文件 | 说明 |
|--------|---------|---------|------|
| 785026c | fix: switch GAN loss from hinge to BCE + lower D lr | discriminator.py, flowedit_gan.py | 上一 session |
| ad7c497 | feat: add DINOv3-S GAN loss for FlowEdit distillation | discriminator.py, flowedit_gan.py | 上一 session |
| 5200662 | refactor: extract DiscriminatorHelper + remove accelerator dependency | discriminator.py, flowedit_gan.py | 本 session |
| c632204 | feat: add distillation+GAN config and training scripts | config, standard.py, launch scripts | 本 session |

### 文件详情（未 commit 改动）

**全局 mixed_precision="no" 清理**（8 config + 6 entry + 3 eval = 17 文件，每文件 1-2 行）
- 所有 config 删除 `cfg.mixed_precision = "bf16"` 行
- 所有 entry 的 `Accelerator(mixed_precision=cfg.mixed_precision)` → `Accelerator(mixed_precision="no")`
- 原因：bf16 autocast 导致 spconv eval 路径的 GEMM 算法查找失败，之前用 forward-patching 绕过，现在直接禁用 mixed_precision 从根源解决

**`edit4shape/systems/utils/strategy.py`**（-26 行）— 大幅简化 `SpconvInferenceMixin`
- 删除：`_original_forward` 字段、`prepare_sparse()` 中保存 forward 的逻辑、inference_context 中的 forward 恢复
- 保留：只做 DDP unwrap（`accelerator.unwrap_model`），不再 patch forward
- 原因：mixed_precision="no" 后不再需要绕过 bf16 autocast，只需剥离 DDP 包装

**`edit4shape/systems/trellis/entries/flowedit_autograd.py`**（-1/+2 行）
- 删除 L135: `system.guidance.set_accelerator(accelerator)`（DiscriminatorHelper 重构后不再需要）
- mixed_precision 改 "no"
- TrainModeGuard 加注释

**`edit4shape/systems/trellis/forward.py`**（-7/+2 行）— evaluate() 注释简化
- 删除 bf16/spconv/GEMM 相关的长注释段，改为简洁的 "剥离 DDP 包装" 描述

**`scripts/eval/eval_guidance_metrics.py`**（+3/-1 行）— 补 inference_context
- 新增 `inference_ctx = system.strategy.inference_context()` 包裹评估循环
- mixed_precision 改 "no"

**`scripts/eval/eval_trellis.py`、`eval_trellis_normal.py`**（各 2 行）
- mixed_precision 改 "no"、注释简化

**`config/trellis_stage2_distillation_gan.py`**（+1/-1 行）— gan 0.1→0.01, 加 reg.type="v"
- 这是已放弃的蒸馏+GAN 实验配置，保留以备参考

**`config/trellis_stage2_flowedit_gan.py`**（+1 行）— 新增 `gan_r1_gamma = 0.1`

## 调试经验

- **Distillation+GAN 不可行**：全轨迹蒸馏配合 GAN 即使调低 weight (0.01) + 改 reg=v，可视化仍崩坏。原因可能是 distillation 的全程 detach 使得 GAN 信号无法有效传导。flowedit_denoise 单步去噪路径更适合 GAN。
- **mixed_precision bf16 是 spconv eval 问题的根源**：之前用 forward-patching hack 绕过（保存原始 forward，推理时换回），现在直接禁用 mixed_precision 从根源解决，代码大幅简化。
- **GAN weight 敏感度**：flowedit_denoise 用 0.01 稳定训练；distillation 用 0.1 秒崩。新实验尝试 flowedit_denoise + 0.1 看是否能承受更大 GAN 信号。

## 运行中实验

| 实验名 | 配置 | 已运行 | 关注点 |
|--------|------|--------|--------|
| `ericzyma-bce-w001-ddpfix-normal-20260622-221117` | flowedit_denoise, gan=0.01, reg=x1 | ~22h | baseline |
| `ericzyma-bce-w001-regv-normal-20260623-131406` | flowedit_denoise, gan=0.01, reg=v | ~7h | reg_type=v 消融 |
| `ericzyma-bce-w01-regv-normal-20260623-190403` | flowedit_denoise, gan=0.1, reg=v | ~1.5h | **高 GAN weight 消融** |

S3 日志：`s3://arcwm-code-us-west-2/ericzyma/.koala-logs/<job-name>/`
Quota 到期：2026-06-23（剩余约 5h）

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `edit4shape/systems/trellis/autograd_template.py` | L76-135 | reg_type "v"/"x0"/"x1" 实现 |
| `edit4shape/guidance/discriminator.py` | DiscriminatorHelper | D 全生命周期封装 |
| `edit4shape/guidance/paradigms/flowedit_gan.py` | _compute_pixel_loss | GAN loss 集成点 |

## 最终方案

- **Distillation+GAN 放弃**：全轨迹蒸馏 + GAN 实验效果差，不再追求
- **flowedit_denoise + GAN 为主线**：三组消融进行中（w=0.01 × reg=x1/v, w=0.1 × reg=v）
- **mixed_precision 全局禁用**：从根源解决 spconv bf16 问题，删除 forward-patching hack

## 下一步任务

1. 三个实验跑完后对比 wandb 曲线：
   - baseline (w=0.01, reg=x1) vs reg=v (w=0.01, reg=v)：看 reg_type 影响
   - reg=v (w=0.01) vs high-w (w=0.1)：看 GAN weight 能否提高
2. 如果 reg=v 明显优于 x1，后续统一切换
3. 如果 w=0.1 稳定且效果好，考虑进一步提高 GAN weight

## 初步方案

- 先看 wandb dashboard 的 loss 曲线和可视化样例
- 重点关注：GAN d_loss 是否收敛、G 生成质量是否提升、reg loss 是否稳定
- baseline 跑了 22h 应该能看到明确趋势
- 如果三组结果接近，优先选 w=0.01 + reg=v（稳定 + 理论更优的 reg）
