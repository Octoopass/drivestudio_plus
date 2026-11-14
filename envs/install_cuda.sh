#!/usr/bin/env bash
# Cài CUDA toolkit 12.8 (nvcc, header, thư viện dev) vào một thư mục riêng, không cần quyền root, rồi in
# đường dẫn đó (dòng cuối) để đặt CUDA_HOME. install_main.sh gọi script này khi nvcc của máy khác major
# với CUDA của PyTorch: image Kaggle từ 11/2026 chỉ có CUDA 13.0, còn torch 2.4.1+cu121 cần nvcc 12.x
# để build gsplat-geer, nvdiffrast, pytorch3d... Bản 12.8 là bản máy cá nhân của dự án dùng (chạy được
# với gcc 13).
#
#   bash envs/install_cuda.sh [thư mục]   # mặc định ~/.cache/drive2sim/cuda-12.8
set -euo pipefail

PREFIX="$(realpath -m "${1:-$HOME/.cache/drive2sim/cuda-12.8}")"
CHANNEL="nvidia/label/cuda-12.8.1"
MAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/download/2.0.5-0/micromamba-linux-64"
# Đủ cho torch.utils.cpp_extension: header của ATen/cuda gọi cublas, cusparse, cusolver, curand.
PACKAGES=(cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev libcusolver-dev
  libcurand-dev cuda-nvrtc-dev cuda-nvtx-dev cuda-profiler-api)

if [ ! -x "$PREFIX/bin/nvcc" ]; then
  mkdir -p "$(dirname "$PREFIX")"
  if command -v conda >/dev/null; then
    conda create --yes --quiet --prefix "$PREFIX" --override-channels -c "$CHANNEL" -c conda-forge "${PACKAGES[@]}" >&2
  else
    # Không có conda (Kaggle): dùng micromamba, một file chạy được, bản chốt.
    MAMBA="$(dirname "$PREFIX")/micromamba"
    [ -x "$MAMBA" ] || { curl -fsSL -o "$MAMBA" "$MAMBA_URL"; chmod +x "$MAMBA"; }
    MAMBA_ROOT_PREFIX="$(dirname "$PREFIX")/mamba" "$MAMBA" create --yes --quiet --prefix "$PREFIX" \
      --override-channels -c "$CHANNEL" -c conda-forge "${PACKAGES[@]}" >&2
  fi
fi
# Gói conda để header ở targets/x86_64-linux/include; torch.utils.cpp_extension chỉ thêm
# $CUDA_HOME/include khi build phần C++, nên nối các header đó vào include/ (symlink, không ghi đè).
cp -rsn "$PREFIX/targets/x86_64-linux/include/." "$PREFIX/include/"
"$PREFIX/bin/nvcc" --version | tail -1 >&2
echo "$PREFIX"
