#!/bin/bash
set -euo pipefail

# 用法:
#   bash scripts/eval/pull_ckpt.sh <server_name> <remote_ckpt_path> [local_root] [ssh_port]
#
# 说明:
#   - server_name: s3 | s9 | s10
#   - remote_ckpt_path:
#       1) .../checkpoints
#       2) .../checkpoints/checkpoint_0_574
#   - local_root: 本地 logs 根目录（默认 ./logs_for_eval）
#
# 目录规则（自动解析 run_name，且统一保留 checkpoints 层级）:
#   - remote 为 .../checkpoints
#       => local 为 <local_root>/<run_name>/checkpoints/...
#   - remote 为 .../checkpoints/checkpoint_*
#       => local 为 <local_root>/<run_name>/checkpoints/checkpoint_*

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <server_name: s3|s9|s10> <remote_ckpt_path> [local_root] [ssh_port]"
  exit 1
fi

SERVER_NAME="$1"
REMOTE_PATH_RAW="$2"
LOCAL_ROOT="${3:-./logs_for_eval}"
SSH_PORT="${4:-22}"

USER_NAME="zhiyuan_ma"

case "$SERVER_NAME" in
  s3) SERVER_IP="10.21.21.174" ;;
  s9) SERVER_IP="10.21.21.196" ;;
  s10) SERVER_IP="10.21.21.197" ;;
  c6) SERVER_IP="10.21.21.185" ;;
  *)
    echo "Unknown server_name: $SERVER_NAME"
    echo "Supported: s3, s9, s10"
    exit 1
    ;;
esac

# 统一去掉尾部斜杠，避免 rsync 的目录语义差异。
REMOTE_PATH="${REMOTE_PATH_RAW%/}"
REMOTE_BASE="$(basename "$REMOTE_PATH")"
REMOTE_HOST="${USER_NAME}@${SERVER_IP}"

if [[ "$REMOTE_BASE" == "checkpoints" ]]; then
  RUN_NAME="$(basename "$(dirname "$REMOTE_PATH")")"
  LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"
  LOCAL_DEST="${LOCAL_RUN_DIR}/"
elif [[ "$REMOTE_BASE" == checkpoint_* ]]; then
  CHECKPOINTS_DIR="$(dirname "$REMOTE_PATH")"
  if [ "$(basename "$CHECKPOINTS_DIR")" != "checkpoints" ]; then
    echo "Invalid remote_ckpt_path: $REMOTE_PATH_RAW"
    echo "checkpoint_* must be directly under a 'checkpoints' directory."
    exit 1
  fi
  RUN_NAME="$(basename "$(dirname "$CHECKPOINTS_DIR")")"
  LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"
  LOCAL_DEST="${LOCAL_RUN_DIR}/checkpoints/"
else
  echo "Invalid remote_ckpt_path: $REMOTE_PATH_RAW"
  echo "Path must end with 'checkpoints' or 'checkpoint_*'."
  exit 1
fi

mkdir -p "$LOCAL_DEST"

echo "========================================"
echo "Start pulling checkpoint"
echo "SERVER_NAME : $SERVER_NAME"
echo "REMOTE_HOST : $REMOTE_HOST"
echo "REMOTE_PATH : $REMOTE_PATH"
echo "RUN_NAME    : $RUN_NAME"
echo "LOCAL_DEST  : $LOCAL_DEST"
echo "SSH_PORT    : $SSH_PORT"
echo "========================================"

rsync -avh --progress --partial --append-verify \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_DEST}"

echo "Done."
