#!/usr/bin/env bash
set -euo pipefail

# 激活 Conda 环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
else
  echo "未找到 conda.sh，请检查 Conda 安装路径" >&2
  exit 1
fi

conda activate grpo3d

cd /home/zhiyuan_ma/code2/flow_grpo_custom

python scripts/debug/test_trellis_suite.py \
  --quick false \
  --steps 10 \
  --num-candidates 2 \
  "$@"

