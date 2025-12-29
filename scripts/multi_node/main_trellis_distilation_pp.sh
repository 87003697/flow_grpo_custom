#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本 (多机多卡 Pipeline 并行版)
#
# 设备分配规则：每个 worker 占用 2 张连续的卡
# - LOCAL_RANK 0: train=cuda:0, guidance=cuda:1
# - LOCAL_RANK 1: train=cuda:2, guidance=cuda:3
# - LOCAL_RANK 2: train=cuda:4, guidance=cuda:5
# - LOCAL_RANK 3: train=cuda:6, guidance=cuda:7
#
# 使用方法：
# 1. 在节点 0 上运行: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> ./main_trellis_distilation_pp.sh
# 2. 在节点 1 上运行: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> ./main_trellis_distilation_pp.sh

# === 可配置参数（通过环境变量设置）===
NNODES=${NNODES:-1}                          # 节点数，默认 1
NODE_RANK=${NODE_RANK:-0}                    # 当前节点排名，默认 0
MASTER_ADDR=${MASTER_ADDR:-"localhost"}      # 主节点 IP，默认 localhost
MASTER_PORT=${MASTER_PORT:-29500}            # 主节点端口，默认 29500
NPROC_PER_NODE=${NPROC_PER_NODE:-4}          # 每节点 worker 数，默认 4 (使用 8 张卡)

echo "========================================"
echo "多机多卡训练启动配置"
echo "========================================"
echo "节点数: $NNODES"
echo "当前节点: $NODE_RANK"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"
echo "每节点 worker 数: $NPROC_PER_NODE"
echo "每节点显卡需求: $((NPROC_PER_NODE * 2)) 张"
echo "========================================"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m edit4shape.systems.trellis_pp \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=False

