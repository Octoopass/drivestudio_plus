#!/usr/bin/env bash
# Cài môi trường chính của drive2sim (ARCHITECTURE mục 2.9) vào một Python có sẵn.
# Dùng chung cho máy cá nhân (sau khi tạo conda env bằng envs/main.yml) và notebook Kaggle, để mọi
# run chạy cùng một bộ phiên bản.
#
#   conda env create -f envs/main.yml && conda activate drive2sim && bash envs/install_main.sh
#   # Kaggle (notebook do d2s kaggle push sinh): bash envs/install_main.sh --python /kaggle/tmp/venv/bin/python (venv Python 3.10)
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
# prep.smpl (OmniRe dựng người): 4D-Humans (submodule third_party/Humans4D, gói hmr2) và PHALP với hai
# gói build CUDA nó cần. Chốt commit để mọi phiên cài cùng một bản.
PHALP="git+https://github.com/brjathu/PHALP.git@96f7e6c09fb858ec3f597d59246c151ab4394bc3"
DETECTRON2="git+https://github.com/facebookresearch/detectron2.git@247691a1c292654867deb35cabb27f18d75a1363"
NMR="git+https://github.com/shubham-goel/NMR.git@e990b3c70f48d39231f607c79d76ce3db4bf7483"
# gsplat-geer clone ngoài repo, để không lẫn vào git của repo.
GEER_DIR="${D2S_GEER_DIR:-$HOME/.cache/drive2sim/3dgeer}"

pip() { "$PY" -m pip "$@"; }
# Gói build từ mã nguồn (CUDA): pip giấu output của bước build ("No available output"), nên ghi đủ
# log ra file và in phần cuối khi lỗi.
build() {
  local log
  log="$(mktemp "${TMPDIR:-/tmp}/d2s-build.XXXXXX")"
  if ! "$PY" -m pip install -v "$@" >"$log" 2>&1; then
    echo "build lỗi: pip install $*; 80 dòng cuối của $log:"
    tail -n 80 "$log"
    return 1
  fi
}
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
want() { [ "$FROM" -le "$1" ]; }  # bước $1 có chạy không

if want 1; then
  step "1/8 công cụ build"
  # setuptools < 81: torchmetrics 0.10.3 (đo metric) còn import pkg_resources, đã bị bỏ từ 81.
  pip install --upgrade pip "setuptools<81" wheel ninja
fi

if want 2; then
  step "2/8 PyTorch ($TORCH)"
  pip install "$TORCH" "$TORCHVISION" --index-url "$TORCH_INDEX"
fi

if want 3; then
  step "3/8 requirements.txt"
  # chumpy (cần cho SMPL) có setup.py import pip, nên phải build không cô lập, trước các gói khác.
  pip install chumpy --no-build-isolation
  pip install -r "$REPO/requirements.txt"
fi

if want 4; then
  # Bước 4, 5, 7 build CUDA: nvcc phải cùng major với CUDA của PyTorch (cu121 → 12.x).
  echo "nvcc: $(command -v nvcc || echo 'không có') $(nvcc --version 2>/dev/null | tail -1); CUDA_HOME=${CUDA_HOME:-}"
  # Bản fork của dự án (submodule third_party/gsplat-geer, docs/methods/fastergs.md): gsplat-geer
  # của Bosch cộng kỹ thuật Faster-GS, mặc định tắt nên khi không bật cho kết quả như bản gốc.
  # Repo clone kèm --recurse-submodules (notebook Kaggle) thì dùng luôn; không có thì clone bản Bosch.
  SUB_DIR="$REPO/third_party/gsplat-geer"
  if [ -f "$SUB_DIR/setup.py" ]; then
    step "4/8 gsplat-geer $GSPLAT_VERSION (fork, submodule $(git -C "$SUB_DIR" rev-parse --short HEAD))"
    git -C "$REPO" submodule update --init --recursive third_party/gsplat-geer
    build "$SUB_DIR" --no-build-isolation
  else
  step "4/8 gsplat-geer $GSPLAT_VERSION (commit $GSPLAT_COMMIT)"
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
  build "$GEER_DIR" --no-build-isolation
  fi
fi

if want 5; then
  step "5/8 nvdiffrast, fused-ssim, smplx"
  build "$NVDIFFRAST" --no-build-isolation
  # Build từ mã nguồn, cần PyTorch lúc build nên không để trong requirements.txt (bước 3 build cô lập).
  build "$FUSED_SSIM" --no-build-isolation
  pip install -e "$REPO/third_party/smplx"
fi

if want 6; then
  step "6/8 drive2sim"
  # Không cài extra "kaggle": Kaggle CLI 2.x cần Python 3.11+, và tool gọi lệnh kaggle như chương
  # trình riêng (cài ở đâu cũng được, ví dụ pipx), notebook Kaggle cũng không cần nó.
  pip install -e "$REPO[studio]"
fi

if want 7; then
  # pytorch3d: models, geometry và mọi loader đều import, nên luôn cần.
  step "7/8 pytorch3d (build lâu nhất)"
  build "$PYTORCH3D" --no-build-isolation
fi

if want 8; then
  step "8/8 4D-Humans, PHALP (prep.smpl)"
  git -C "$REPO" submodule update --init --depth 1 third_party/Humans4D
  build "$DETECTRON2" --no-build-isolation
  build "$NMR" --no-build-isolation
  # Ghi rõ numpy để pip không nâng numpy khi giải phụ thuộc (requirements.txt chốt 1.23.5).
  pip install -e "$REPO/third_party/Humans4D" --no-build-isolation "numpy==1.23.5"
  # PHALP khai phụ thuộc bằng URL git không chốt (detectron2, NMR, chumpy...); cài nó không kèm
  # phụ thuộc, rồi cài các gói PyPI nó dùng.
  pip install --no-deps "$PHALP"
  pip install joblib scikit-learn pyrender dill rich einops "scenedetect[opencv]" hydra-core timm av \
    pytube "numpy==1.23.5"
fi

step "kiểm tra"
"$PY" - <<'PYEOF'
import torch, gsplat
print("torch", torch.__version__, "cuda", torch.version.cuda, "GPU", torch.cuda.is_available())
print("gsplat", gsplat.__version__)
PYEOF
