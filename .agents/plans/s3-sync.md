# Plan: 训练结果后台 S3 同步

## 目标
在 setup_koala.sh 中添加后台 S3 sync，定期将训练产出（logs/checkpoints/visualizations）同步到 S3 持久化存储，避免 pod crash 后数据丢失。

## 关键发现（Explore 阶段）

### 训练输出路径
- 基于 `cfg.logdir` (默认 "logs") + `cfg.run_name` 构成完整路径
- 训练脚本 cwd = `/data/work/run_codes`
- 输出结构：`logs/{run_name}/checkpoints/`、`logs/{run_name}/logs/`、`logs/{run_name}/visualizations/`

### prime-rl 参考实现
- `setup_s3_sync()` 函数：每 5 分钟 `aws s3 sync` 到 S3 bucket
- 用 `trap EXIT` 确保退出时做最后一次同步
- 排除 `broadcasts/*` 和 `*.bin`（optimizer 状态太大）

### 约束
- Normal pod 无法 SSH，日志只能通过 S3 或 `koala logs` 查看
- S3 FUSE 不适合写入 → 必须用 `aws s3 sync`（直接走 S3 API）
- Pod crash 时 trap 不一定能触发（SIGKILL 无法 trap）
- `aws s3 sync` 凭证已自动注入容器

### S3 目标路径
按 `.agents/koala/storage.md` 规范：
```
/threed-code/ericzyma/experiments/flow_grpo/{run_name}/
├── checkpoints/
├── logs/
└── visualizations/
```

## 相关代码
| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `scripts/setup_koala.sh` | 末尾 | 需要新增 sync 逻辑 |
| `prime-rl/scripts/setup_kaola.sh` | `setup_s3_sync()` | 参考实现 |
| `config/trellis2_base.py` | `cfg.logdir = "logs"` | 训练输出根目录 |
| `edit4shape/systems/base.py` | `build_run_paths()` | 构建输出目录结构 |

## 实现步骤
- [ ] Step 1: 在 setup_koala.sh 末尾（"完成" 之前）加入后台 S3 sync 函数
- [ ] Step 2: 上传到 S3 并验证

## 代码变更预览

在 `setup_koala.sh` 的 `# 完成` 之前插入：

```bash
# ============================================================================
# [7/7] 后台 S3 同步（训练产出持久化）
# ============================================================================
echo "=== [7/7] Background S3 sync ==="
LOGS_LOCAL="${PROJECT_DIR}/logs"
LOGS_S3="${S3_BUCKET}/experiments/flow_grpo"

sync_logs() {
    if [ -d "${LOGS_LOCAL}" ]; then
        aws s3 sync "${LOGS_LOCAL}/" "${LOGS_S3}/" \
            --exclude '*.bin' \
            --quiet >> /tmp/s3_sync.log 2>&1 || true
    fi
}

(while true; do sleep 300; sync_logs; done) &
SYNC_PID=$!
trap "kill ${SYNC_PID} 2>/dev/null || true; sync_logs" EXIT

echo "  PID: ${SYNC_PID} (every 5 min)"
echo "  ${LOGS_LOCAL}/ -> ${LOGS_S3}/"
echo "  Mac 查看: ~/threed-code/ericzyma/experiments/flow_grpo/{run_name}/"
```

## 方案对比
| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 后台 aws s3 sync（本方案） | 简单可靠，和 prime-rl 一致，不改训练代码 | SIGKILL 时最多丢 5 min 数据 |
| B: 改 cfg.logdir 直接写 S3 FUSE | 实时持久化 | FUSE rename 不支持，checkpoint 会崩 |
| C: 训练代码中 callback 同步 | 精准控制时机 | 需改训练代码，侵入性大 |

## 状态
**当前阶段**: Planning
