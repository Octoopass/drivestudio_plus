#!/usr/bin/env bash
# Cài worker vào interpreter riêng trên máy thực thi.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Cách dùng: bash envs/install_xsim.sh <python-env-xsim> <thư-mục-nguồn>" >&2
  exit 2
fi
# Giữ symlink của interpreter để pip cài vào đúng venv.
python_bin=$(realpath -s "$1")
source_dir=$(realpath -m "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
revision=5c86a33a036f3163f78eff5dcd8d3fc75c05d2a3
patch="$root/drive2sim/external/xsim/patches/runtime.patch"
"$python_bin" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "XSIM cần Python 3.11 hoặc 3.12"'
if [[ ! -e "$source_dir" ]]; then
  git clone --no-checkout https://github.com/avision-team/XSIM.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
cd "$source_dir"
[[ "$(git rev-parse HEAD)" == "$revision" ]] || { echo "Revision XSIM không khớp" >&2; exit 1; }
git submodule update --init --recursive
if git apply --check "$patch" 2>/dev/null; then
  git apply "$patch"
else
  git apply --reverse --check "$patch"
fi
"$python_bin" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install setuptools wheel ninja 'pybind11>=3.0.4'
"$python_bin" -m pip install --no-build-isolation ./third_party/pytoast
"$python_bin" -m pip install --no-build-isolation ./third_party/xsimgs
"$python_bin" -m pip install --no-build-isolation -r requirements.txt
"$python_bin" -m pip install -e . "$root"
"$python_bin" -c 'import torch, toast, xsimgs._C, xsim; assert torch.cuda.is_available(), "Worker chưa thấy GPU CUDA"'
