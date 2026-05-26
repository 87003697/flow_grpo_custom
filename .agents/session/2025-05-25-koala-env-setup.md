![1779714990539](image/2025-05-25-koala-env-setup/1779714990539.png)# Session Handoff: Koala 环境配置完成

## 前序 Session
无（首份 handoff）

## 任务目的
在 Koala 集群上为 flow_grpo_custom 项目创建完整的环境初始化脚本 `scripts/setup_koala.sh`，使项目能一键恢复环境并运行 Trellis2 Shape 蒸馏训练。

## 执行内容
- 调研了 prime-rl / procfunc / infinigen / Code-as-Room 四个项目的 setup 脚本模式
- 编写 `scripts/setup_koala.sh`，支持 `--download`（首次）和 `--fast`（日常恢复）两种模式
- 在 debug pod (H200) 上实测：安装 torch 2.6.0+cu124 + 133 packages + flash-attn 2.7.3
- 编译 CUDA 扩展：nvdiffrast v0.4.0, nvdiffrec(renderutils), CuMesh, FlexGEMM, o-voxel
- 从 HuggingFace 下载 TRELLIS.2-4B 权重（microsoft/TRELLIS.2-4B）
- 从 ModelScope 下载 DINOv3（facebook HF gated，走 ModelScope 镜像）
- 从 HF 下载 alphaimages_v3 数据集（2396 张图片）
- clone https://github.com/87003697/TRELLIS.2.git 作为 `_reference_codes/TRELLIS.2`
- 所有产物打 tar 缓存到 S3，验证 `--fast` 恢复全流程通过
- 最终 import 验证全部通过：trellis2 + edit4shape + config 加载 + 路径检查

## 调试经验
- **flash-attn 编译需要 wheel 包**：`--no-build-isolation` 模式下 setup.py 依赖 wheel 但未声明。需在之前 `uv pip install wheel setuptools`
- **DINOv3 在 HuggingFace 上是 gated repo**：HF token 被拒，改用 ModelScope 下载。项目已有 `download_dinov3_trellis2_modelscope.py` 脚本
- **TRELLIS.2 仓库**：不是 `microsoft/TRELLIS`（那是 v1），正确地址是 `https://github.com/87003697/TRELLIS.2.git`（私人 fork，含 `trellis2/` 包 + `o-voxel/`）
- **S3 FUSE 写入 tar 前需要目录存在**：`tar cf /threed-code/.../file.tar` 会报 "No such file"，需 `mkdir -p` 先创建
- **Python 版本漂移**：Koala 镜像默认 uv venv 可能选 3.11 或 3.12，已在脚本中锁定 `--python 3.12`
- **o-voxel 缺 zstandard 依赖**：`import o_voxel` 需要 zstandard 包但未在 o-voxel 的 setup.py 中声明

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `scripts/setup_koala.sh` | 全文 | Koala 环境初始化脚本（本次核心产出） |
| `scripts/multi_node/main_trellis2_shape_distilation_async.sh` | GPU 分配逻辑 | 8 卡训练入口（4 训练 + 4 Guidance） |
| `config/trellis2_shape_distillation.py` | `get_config()` | Shape 蒸馏配置 |
| `config/trellis2_base.py` | `_build_pretrained()` | 权重路径、数据路径等默认值 |
| `edit4shape/systems/trellis2/entries/shape_autograd_async.py` | 训练入口 | 三阶段 Autograd + 异步 Guidance |
| `.agents/plans/koala-env-setup.md` | S3 布局 | tar 路径和使用方式记录 |

## 最终方案
采用 **uv pip install + S3 tar 缓存** 模式（与 prime-rl/infinigen 一致）：
- 首次 `--download`：从 HF/ModelScope 下载 + 编译 CUDA 扩展 + tar 到 S3
- 日常 `--fast`：从 S3 tar 恢复到 `/local-ssd/`，软链接回项目目录
- Python 3.12 + torch 2.6.0+cu124 + flash-attn 2.7.3（与 venus/init_env_trellis2.sh 对齐）
- 凭证 HF_TOKEN / WANDB_API_KEY 已在 `~/.zshrc` 配置，koala 自动注入容器

## S3 布局
```
/threed-code/ericzyma/
├── data/flow_grpo/
│   ├── pretrained_weights.tar      (21 GB — TRELLIS.2-4B + DINOv3 + TRELLIS-image-large)
│   ├── alphaimages_v3.tar          (474 MB — 训练数据集 2396 张)
│   └── trellis2_reference.tar      (361 MB — _reference_codes/TRELLIS.2)
└── tools/
    └── flow_grpo_cuda_ext.tar      (640 MB — nvdiffrast/CuMesh/FlexGEMM)
```

## 下一步任务
正式运行 8 卡训练（Trellis2 Shape 蒸馏）。

## 初步方案
1. **提交 8 卡 normal pod**：
   ```bash
   aws s3 sync . "s3://arcwm-code-us-west-2/ericzyma/flow_grpo_custom/" --exclude ...
   koala submit -m normal -g 8 --code "s3://arcwm-code-us-west-2/ericzyma/flow_grpo_custom:/data/work/run_codes" \
       -c "cd /data/work/run_codes && . scripts/setup_koala.sh --fast && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/multi_node/main_trellis2_shape_distilation_async.sh"
   ```

2. **训练脚本需调整**：当前 `main_trellis2_shape_distilation_async.sh` 默认 `CUDA_VISIBLE_DEVICES=4,5,6,7`（4 卡），8 卡需改为 `0,1,2,3,4,5,6,7`，结果是 4 卡训练 DDP + 4 卡 Guidance

3. **wandb 配置**：训练脚本需设 `--config.use_wandb=true`，确保 WANDB_API_KEY 已注入（已确认）

4. **关键风险**：
   - Qwen-Image-Edit-2511 Guidance 模型尚未缓存到 S3（首次训练会自动从 HF 下载到 `/local-ssd/hf_cache`）
   - normal pod 无法 SSH，出错只能看 `koala logs`
   - `--fast` 模式下 CUDA 扩展 rebuild 约 6-8 min（Python 版本匹配后应缩短）
   
5. **建议**：先在当前 debug pod 上跑一次 4 卡确认训练能启动（验证 Qwen 模型下载 + 完整训练 loop），再提交 8 卡 normal
