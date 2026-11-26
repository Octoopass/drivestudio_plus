#!/usr/bin/env bash
# Cài XSIM (codebase ngoài, ARCHITECTURE mục 2.9) vào D2S_XSIM_DIR (mặc định ~/.cache/drive2sim/xsim),
# bốn bước chạy tiếp được:
#   1. mã nguồn avision-team/XSIM đúng commit chốt, submodule pytoast và xsimgs theo gitlink, áp
#      drive2sim/external/xsim/patches/runtime.patch;
#   2. venv riêng (.venv, Python 3.11) bằng uv, PyTorch 2.8 CUDA 12.8;
#   3. build pytoast rồi xsimgs (extension CUDA), cài requirements của XSIM với các gói git đã chốt
#      commit trong envs/xsim-constraints.txt;
#   4. cài XSIM và drive2sim của repo này dạng editable (worker luôn chạy đúng mã đang checkout),
#      rồi kiểm import và GPU.
#
#   bash envs/install_xsim.sh [--python PY] [--from N]
#
# Studio chạy script này từ nút cài (drive2sim/engine/env_install.py), truyền --python là Python
# đang chạy Studio; Python đó chỉ dùng để cài uv nếu thiếu, venv XSIM do uv dựng riêng.
# nvcc phải cùng major với CUDA của PyTorch (12.x): khác thì cài CUDA 12.8 riêng bằng
# envs/install_cuda.sh vào D2S_CUDA_PREFIX, như envs/install_main.sh.
set -euo pipefail
# Bước kiểm import nạp matplotlib.pyplot (qua xsim); kernel Jupyter của Kaggle, Colab đặt
# MPLBACKEND trỏ tới backend inline mà venv XSIM không có, matplotlib sẽ báo lỗi. Như worker
# (drive2sim.external.xsim.WORKER_ENV), dùng Agg.
export MPLBACKEND=Agg

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${D2S_XSIM_DIR:-$HOME/.cache/drive2sim/xsim}"
SRC="$DIR/XSIM"
VENV="$DIR/.venv"
PY="python3"
FROM=1
REVISION=5c86a33a036f3163f78eff5dcd8d3fc75c05d2a3
PATCH="$REPO/drive2sim/external/xsim/patches/runtime.patch"
CONSTRAINTS="$REPO/envs/xsim-constraints.txt"
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "không hiểu tham số $1" >&2; exit 2 ;;
  esac
done
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
echo "XSIM vào $DIR (đặt D2S_XSIM_DIR để đổi)"
mkdir -p "$DIR"

if [ "$FROM" -le 1 ]; then
  step "1/4 mã nguồn XSIM"
  if [ ! -d "$SRC/.git" ]; then
    git clone --quiet --no-checkout https://github.com/avision-team/XSIM.git "$SRC"
  fi
  git -C "$SRC" fetch --quiet origin "$REVISION" || true
  # Thư mục dành riêng cho tool: bỏ thay đổi ở file được theo dõi (kể cả bản vá cũ) trước khi áp
  # bản vá hiện tại, để cài lại sau khi bản vá đổi không bị kẹt.
  git -C "$SRC" checkout --quiet --force --detach "$REVISION"
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$REVISION" ] || { echo "revision XSIM không khớp" >&2; exit 1; }
  git -C "$SRC" submodule update --quiet --init --recursive
  git -C "$SRC" apply "$PATCH"
fi

if [ "$FROM" -le 2 ]; then
  step "2/4 môi trường Python 3.11 (uv)"
  if command -v uv >/dev/null; then
    UV=(uv)
  else
    "$PY" -m pip install --quiet uv || {
      echo "không cài được uv bằng $PY: cài uv (https://docs.astral.sh/uv/) rồi chạy lại" >&2
      exit 1
    }
    UV=("$PY" -m uv)
  fi
  [ -x "$VENV/bin/python" ] || "${UV[@]}" venv --quiet --seed --python 3.11 "$VENV"
  "$VENV/bin/python" -m pip install --quiet 'torch==2.8.0' 'torchvision==0.23.0' \
    --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/python" -m pip install --quiet setuptools wheel ninja 'pybind11>=3.0.4'
fi

if [ "$FROM" -le 3 ]; then
  step "3/4 build pytoast, xsimgs và cài requirements của XSIM"
  [ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
  nvcc_major="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  if [ "$nvcc_major" != "12" ]; then
    echo "nvcc ${nvcc_major:-không có} khác CUDA 12 của PyTorch: cài CUDA 12.8 riêng"
    CUDA_HOME="$(bash "$REPO/envs/install_cuda.sh" "${D2S_CUDA_PREFIX:-$HOME/.cache/drive2sim/cuda-12.8}")"
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
  fi
  if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi >/dev/null; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')"
    export TORCH_CUDA_ARCH_LIST
  fi
  echo "nvcc: $(command -v nvcc || echo 'không có'); TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}"
  cd "$SRC"
  "$VENV/bin/python" -m pip install --no-build-isolation ./third_party/pytoast
  "$VENV/bin/python" -m pip install --no-build-isolation ./third_party/xsimgs
  # Gói git trong requirements.txt của XSIM không chốt commit: cài bản đã chốt trước, rồi cài phần
  # còn lại của requirements.txt (bỏ các dòng git đó).
  "$VENV/bin/python" -m pip install --no-build-isolation -r "$CONSTRAINTS"
  grep -v '@git+' requirements.txt > "$DIR/requirements-nogit.txt"
  "$VENV/bin/python" -m pip install --no-build-isolation -r "$DIR/requirements-nogit.txt"
  "$VENV/bin/python" -m pip install scikit-image imageio-ffmpeg
fi

if [ "$FROM" -le 4 ]; then
  step "4/4 cài XSIM và drive2sim (editable), kiểm import và GPU"
  "$VENV/bin/python" -m pip install -e "$SRC" -e "$REPO"
  "$VENV/bin/python" -c 'import torch, toast, xsimgs._C, xsim, drive2sim; assert torch.cuda.is_available(), "worker XSIM chưa thấy GPU CUDA"; print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))'
fi
echo "xong: Python của worker là $VENV/bin/python"
