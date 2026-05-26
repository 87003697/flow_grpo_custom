# Session Handoff: 8 卡训练成功

## 前序 Session
- `.agents/session/2025-05-26-8gpu-debug.md` — 排查失败原因，优化 setup 速度

## 任务目的
排查 8 卡 normal 训练反复 fail 的根因，修复并成功启动训练。

## 执行内容
- 排查 CUDA 扩展编译超时 → 改用预编译 site-packages tar（3s 恢复 vs 6-8 min rebuild）
- 排查训练启动后 crash → **根因是 HF_TOKEN 缺失**，TRELLIS.2 pipeline 加载 BiRefNet 时需要从 briaai/RMBG-2.0 下载模型（gated repo）
- setup 脚本新增 `export HF_TOKEN`
- 提交成功：`ericzyma-job-normal-20260526-135314`，Step 1 完成（157s），峰值显存 27 GiB/卡

## 调试经验
- **HF_TOKEN 是 8 卡训练失败的唯一原因**：所有之前的 normal pod 失败（~20 次）都是同一原因
- **Koala 不自动注入 HF_TOKEN**：只注入 AWS_* 和 CURSOR_API_KEY，HF_TOKEN 需要在脚本中显式 export
- **预编译 site-packages tar 必须包含顶层 .so**：`_nvdiffrast_c.cpython-312-x86_64-linux-gnu.so` 在 site-packages 根目录
- **8 卡分离模式显存极低**：峰值仅 27 GiB/训练卡（vs 共享模式 115 GiB），H200 (143 GiB) 非常充裕

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `scripts/setup_koala.sh` | `export HF_TOKEN=...` | 根因修复 |
| `scripts/setup_koala.sh` | `[2/6] CUDA_SP_TAR` | 预编译恢复路径 |

## 当前状态
- **任务名**：`ericzyma-job-normal-20260526-135314`
- **状态**：Running，Step 1+ 完成
- **WandB**：https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/trellis2-shape-distillation/runs/3e2lbzlq
- **时长上限**：48h

## 监控命令
```bash
koala logs ericzyma-job-normal-20260526-135314
koala list
# wandb: https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/trellis2-shape-distillation/runs/3e2lbzlq
```

## 下一步
- 监控训练是否稳定运行（观察 wandb loss 曲线）
- 如需调参，改 `config/trellis2_shape_distillation.py` 或训练脚本中的 RUN_NAME
- 后续可能需要：多机训练（-n 2+）、Tex 阶段训练、对比学习配置
