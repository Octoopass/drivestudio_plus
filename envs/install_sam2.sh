#!/usr/bin/env bash
# Cài SAM2.1 cho công cụ vẽ mask thân xe của Studio (drive2sim/studio/api/ego_mask_tool.py):
#   1. venv models/sam2-venv bằng uv, dựng trên Python của môi trường drive2sim
#      (--system-site-packages: dùng lại PyTorch, OpenCV) và cài transformers đủ mới để có
#      Sam2Model; môi trường drive2sim giữ nguyên transformers cũ mà DriveStudio cần;
#   2. tải facebook/sam2.1-hiera-large (khoảng 900 MB) vào models/sam2.1-hiera-large.
# models/ nằm trong .gitignore.
#
#   bash envs/install_sam2.sh [--python PY]
#
# PY mặc định là python của PATH (chạy trong môi trường drive2sim).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="python"
[[ "${1:-}" == "--python" ]] && PY="$2"
VENV="$REPO/models/sam2-venv"
MODEL="$REPO/models/sam2.1-hiera-large"

UV="$(command -v uv || true)"
[[ -z "$UV" ]] && { "$PY" -m pip install -q uv; UV="$PY -m uv"; }
$UV venv -q --system-site-packages -p "$("$PY" -c 'import sys; print(sys.executable)')" "$VENV"
$UV pip install -q -p "$VENV/bin/python" "transformers==4.57.1"
"$VENV/bin/python" -c "from transformers import Sam2Model; import torch; print('torch', torch.__version__)"

if [[ ! -f "$MODEL/model.safetensors" ]]; then
  HF_HUB_DISABLE_TELEMETRY=1 "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam2.1-hiera-large', local_dir='$MODEL', allow_patterns=[
    'config.json', 'preprocessor_config.json', 'processor_config.json', 'model.safetensors'])"
fi
echo "SAM2 sẵn sàng: $VENV, $MODEL"
