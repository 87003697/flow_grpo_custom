#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/download/download_dinov3_from_meta.sh <DIRECT_URL> [DEST_DIR]
# 例：
#   bash scripts/download/download_dinov3_from_meta.sh "https://ai.meta.com/.../dinov3_vitb14.pth" pretrained_weights/dinov3-vitb14

URL="${1:-}"         # 直链 URL（来自 https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/）
DEST="${2:-pretrained_weights/dinov3-vitb14}"

if [[ -z "$URL" ]]; then
  echo "[用法] bash $0 <DIRECT_URL> [DEST_DIR]"
  echo "[提示] 请先在浏览器访问 dinov3 官方下载页，同意协议后复制具体 checkpoint 的直链 URL。"
  exit 1
fi

mkdir -p "$DEST"
FNAME=$(basename "$URL")
OUT="$DEST/$FNAME"

echo "[INFO] 下载: $URL"
echo "[INFO] 保存到: $OUT"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 5 --retry-delay 2 -o "$OUT" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
else
  echo "需要 curl 或 wget"
  exit 1
fi

echo "[OK] 下载完成: $OUT"


