#!/usr/bin/env bash
# Cài môi trường chính của drive2sim (ARCHITECTURE mục 2.9) vào một Python có sẵn.
# Dùng chung cho máy cá nhân (sau khi tạo conda env bằng envs/main.yml) và notebook Kaggle, để mọi
# run chạy cùng một bộ phiên bản.
#
#   conda env create -f envs/main.yml && conda activate drive2sim && bash envs/install_main.sh
#   # Kaggle (notebook do d2s kaggle push sinh): bash envs/install_main.sh --python python
#
# Phiên bản chốt ở đây và trong requirements.txt; drive2sim/engine/environment.py (EXPECTED) phải
# khớp (có test kiểm). Các gói build từ mã nguồn (gsplat-geer, nvdiffrast, pytorch3d) cần CUDA
# toolkit (nvcc) và mất hàng chục phút (fused-ssim cũng build từ mã nguồn); đặt
# TORCH_CUDA_ARCH_LIST (ví dụ "7.5") để build nhanh hơn.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="python"
FROM=1  # chạy tiếp từ bước này (--from 4), khi các bước trước đã xong
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "không hiểu tham số $1" >&2; exit 2 ;;
  esac
done

TORCH_INDEX="https://download.pytorch.org/whl/cu121"
TORCH="torch==2.4.1+cu121"
TORCHVISION="torchvision==0.19.1+cu121"
GSPLAT_REPO="https://github.com/boschresearch/3dgeer.git"
GSPLAT_BRANCH="gsplat-geer"
GSPLAT_COMMIT="f15cee2"
GSPLAT_VERSION=1.5.3
NVDIFFRAST="git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
PYTORCH3D="git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
# SSIM fused cho Faster-GS (docs/methods/fastergs.md); commit đã kiểm khớp pytorch_msssim.
FUSED_SSIM="git+https://github.com/rahul-goel/fused-ssim.git@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
# gsplat-geer clone ngoài repo, để không lẫn vào git của repo.
GEER_DIR="${D2S_GEER_DIR:-$HOME/.cache/drive2sim/3dgeer}"

pip() { "$PY" -m pip "$@"; }
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
want() { [ "$FROM" -le "$1" ]; }  # bước $1 có chạy không

if want 1; then
  step "1/7 công cụ build"
  # setuptools < 81: torchmetrics 0.10.3 (đo metric) còn import pkg_resources, đã bị bỏ từ 81.
  pip install --upgrade pip "setuptools<81" wheel ninja
fi

if want 2; then
  step "2/7 PyTorch ($TORCH)"
  pip install "$TORCH" "$TORCHVISION" --index-url "$TORCH_INDEX"
fi

if want 3; then
  step "3/7 requirements.txt"
  # chumpy (cần cho SMPL) có setup.py import pip, nên phải build không cô lập, trước các gói khác.
  pip install chumpy --no-build-isolation
  pip install -r "$REPO/requirements.txt"
fi

if want 4; then
  step "4/7 gsplat-geer $GSPLAT_VERSION (commit $GSPLAT_COMMIT)"
  if [ ! -d "$GEER_DIR/.git" ]; then
    git clone --recursive --branch "$GSPLAT_BRANCH" "$GSPLAT_REPO" "$GEER_DIR"
  fi
  git -C "$GEER_DIR" fetch --quiet origin "$GSPLAT_BRANCH" || true
  git -C "$GEER_DIR" checkout --quiet "$GSPLAT_COMMIT"
  git -C "$GEER_DIR" submodule update --init --recursive
  # Ở commit này submodule glm khai báo trong .gitmodules nhưng không có trong cây, nên thư mục trống
  # và build lỗi "glm/gtc/type_ptr.hpp: No such file": tự lấy glm 1.0.1 vào đúng chỗ.
  GLM_DIR="$GEER_DIR/gsplat/cuda/csrc/third_party/glm"
  if [ ! -f "$GLM_DIR/glm/gtc/type_ptr.hpp" ]; then
    rm -rf "$GLM_DIR"
    git clone --quiet --depth 1 --branch 1.0.1 https://github.com/g-truc/glm.git "$GLM_DIR"
  fi
  pip install "$GEER_DIR" --no-build-isolation
fi

if want 5; then
  step "5/7 nvdiffrast, fused-ssim, smplx"
  pip install "$NVDIFFRAST" --no-build-isolation
  # Build từ mã nguồn, cần PyTorch lúc build nên không để trong requirements.txt (bước 3 build cô lập).
  pip install "$FUSED_SSIM" --no-build-isolation
  pip install -e "$REPO/third_party/smplx"
fi

if want 6; then
  step "6/7 drive2sim"
  # Không cài extra "kaggle": Kaggle CLI 2.x cần Python 3.11+, và tool gọi lệnh kaggle như chương
  # trình riêng (cài ở đâu cũng được, ví dụ pipx), notebook Kaggle cũng không cần nó.
  pip install -e "$REPO[studio]"
fi

if want 7; then
  # pytorch3d: models, geometry và mọi loader đều import, nên luôn cần.
  step "7/7 pytorch3d (build lâu nhất)"
  pip install "$PYTORCH3D" --no-build-isolation
fi

step "kiểm tra"
"$PY" - <<'PYEOF'
import torch, gsplat
print("torch", torch.__version__, "cuda", torch.version.cuda, "GPU", torch.cuda.is_available())
print("gsplat", gsplat.__version__)
PYEOF
