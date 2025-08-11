#!/usr/bin/env bash

# 一键下载 TRELLIS-image-large 到 pretrained_weights/TRELLIS-image-large
# - 依赖: python + huggingface_hub
# - 默认下载到项目根目录的 pretrained_weights/TRELLIS-image-large
# - 可传入自定义目标目录作为第一个参数

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO_ID="microsoft/TRELLIS-image-large"
DEST_DIR="${1:-${PROJECT_ROOT}/pretrained_weights/TRELLIS-image-large}"

echo "Repo: ${REPO_ID}"
echo "Dest: ${DEST_DIR}"

mkdir -p "${DEST_DIR}"

# 使用 huggingface_hub 的 snapshot_download 保留仓库目录结构
HF_HUB_ENABLE_HF_TRANSFER=1 \
REPO_ID="${REPO_ID}" DEST_DIR="${DEST_DIR}" \
python - <<'PY'
from huggingface_hub import snapshot_download
import os

repo_id = os.environ.get("REPO_ID", "microsoft/TRELLIS-image-large")
dest_dir = os.environ["DEST_DIR"]

snapshot_download(
    repo_id=repo_id,
    local_dir=dest_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"✅ Downloaded {repo_id} to {dest_dir}")
PY

echo "Done. You can now set config.pretrained.model to: ${DEST_DIR}"


