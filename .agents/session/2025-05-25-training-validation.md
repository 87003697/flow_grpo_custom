# Session Handoff: 8 卡训练验证与提交

## 前序 Session
- `.agents/session/2025-05-25-koala-env-setup.md` — 环境配置完成，创建 setup_koala.sh，S3 tar 缓存就绪

## 任务目的
在 debug pod 上验证 flow_grpo_custom 训练能跑通，修复代码 bug，提交 8 卡正式训练。

## 执行内容
- 在 debug pod（1x H200）上试跑 Trellis2 Shape 蒸馏训练
- 发现并修复 ninja 不在 PATH 的问题（setup 脚本加 `export PATH=/tmp/uv-venv/bin:$PATH`）
- 发现并修复代码 bug：`s.regularization.reg_loss` → `s.shape.reg_loss`（AttributeError）
- 训练成功跑到 Step 6，checkpoint 正常保存，GPU 峰值显存 115.4 GiB / 143 GiB
- 删除 debug pod
- 修改训练脚本默认参数为 8 卡（`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`）
- 提交 8 卡 normal 训练：`ericzyma-job-normal-20260525-211904`

## 调试经验
- **ninja 必须在 PATH 里**：mesh_voxelize 等自定义 op 需要 JIT 编译，torch cpp_extension 调用 ninja。`uv pip install ninja` 装在 `/tmp/uv-venv/bin/`，但默认不在 PATH
- **共享模式自动回退**：`compute_guidance_device()` 在 GPU 不够时会回退到训练同设备。1 卡 H200 (143 GiB) 也能跑完训练（峰值 115 GiB）
- **Qwen-Image-Edit-2511 首次下载约 2 分钟**：32 个文件从 HF 下载，模型加载 ~20s。已缓存到 `/local-ssd/hf_cache`（但未 tar 到 S3）
- **Step 耗时差异大**：Step 3 ~97s, Step 4 ~10s, Step 5 ~278s（因为不同 sample 的 sparse structure 大小不同）

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `scripts/setup_koala.sh` | `export PATH=...` (line 57) | 新增 PATH export |
| `edit4shape/systems/trellis2/entries/shape_autograd_async.py` | line 304 | 修复 `s.shape.reg_loss` |
| `scripts/multi_node/main_trellis2_shape_distilation_async.sh` | line 20-21 | 改为 8 卡默认 |
| `edit4shape/systems/base.py` | `compute_guidance_device()` | GPU 分配逻辑（含共享回退） |

## 最终方案
- setup 脚本加 `export PATH=/tmp/uv-venv/bin:$PATH` 确保 ninja 可用
- 修复 `s.regularization.reg_loss` → `s.shape.reg_loss`（Trellis2State 没有 regularization 属性，reg_loss 在 s.shape 里）
- 训练脚本默认 8 卡，RUN_NAME 可通过环境变量覆盖

## 当前状态
- **任务名**：`ericzyma-job-normal-20260525-211904`
- **资源**：8 GPU (H200), 160 CPU, 1760 Gi
- **时长上限**：48h
- **命令**：`cd /data/work/run_codes && . scripts/setup_koala.sh --fast && bash scripts/multi_node/main_trellis2_shape_distilation_async.sh --config.use_wandb=true`

## 下一步任务
- 监控训练是否正常启动：`koala logs ericzyma-job-normal-20260525-211904`
- 如果 setup `--fast` 阶段 CUDA 扩展编译超时导致 pod fail，需要改成缓存编译好的 wheel（而非源码）
- 训练稳定后观察 wandb 曲线，调整超参

## 监控命令
```bash
koala logs ericzyma-job-normal-20260525-211904       # 查看日志
koala logs ericzyma-job-normal-20260525-211904 -f    # 实时跟踪（可能超时）
koala list                                           # 查看任务状态
```
