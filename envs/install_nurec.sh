#!/usr/bin/env bash
# Cài Instant NuRec (codebase ngoài, ARCHITECTURE mục 2.9) vào D2S_NUREC_DIR
# (mặc định ~/.cache/drive2sim/instant-nurec), ba bước chạy tiếp được:
#   1. mã nguồn NVIDIA/instant-nurec đúng commit chốt;
#   2. venv riêng (.venv, Python 3.11, có nvidia-ncore) bằng `uv sync --frozen`;
#   3. checkpoint từ Hugging Face (nvidia/instant-nurec, revision chốt), tải tiếp được.
#
#   bash envs/install_nurec.sh [--models pa-multiview,pa-front,pq-front] [--from 2]
#
# Studio chạy script này từ nút cài (drive2sim/engine/env_install.py), truyền thêm --python là
# Python đang chạy Studio (chỉ dùng để chạy script cài; venv NuRec do uv dựng riêng). Cần git, uv.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${D2S_NUREC_DIR:-$HOME/.cache/drive2sim/instant-nurec}"
PY="python3"
FROM=1
MODELS="pa-multiview"
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) echo "không hiểu tham số $1" >&2; exit 2 ;;
  esac
done
SETUP="$REPO/drive2sim/external/nurec/tools/nurec_setup.py"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
echo "Instant NuRec vào $DIR (đặt D2S_NUREC_DIR để đổi)"

if [ "$FROM" -le 1 ]; then
  step "1/3 mã nguồn Instant NuRec"
  "$PY" -u "$SETUP" source --repo "$DIR"
fi
if [ "$FROM" -le 2 ]; then
  step "2/3 môi trường Python 3.11 (uv sync)"
  "$PY" -u "$SETUP" env --repo "$DIR"
fi
if [ "$FROM" -le 3 ]; then
  step "3/3 checkpoint ($MODELS)"
  "$PY" -u "$SETUP" weights --repo "$DIR" --models "$MODELS"
fi

step "kiểm tra"
"$DIR/.venv/bin/python" -c "import sys, torch, ncore; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
