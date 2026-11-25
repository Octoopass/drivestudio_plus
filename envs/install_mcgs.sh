#!/usr/bin/env bash
# Cài MCGS-SLAM (codebase ngoài, ARCHITECTURE mục 2.19) vào D2S_MCGS_DIR
# (mặc định ~/.cache/drive2sim/mcgs-slam): mã nguồn ở đó, env Python ở $DIR/env, wheel đã build
# ở $DIR/wheels (cache, dùng lại khi dựng lại máy).
#
#   bash envs/install_mcgs.sh [--from 3]
#
# Bước:
#   1. mã nguồn mcgs-slam/mcgs-slam đúng commit chốt, kèm submodule (lietorch, simple-knn, glm, eigen);
#   2. env micromamba riêng (Python 3.10, CUDA 11.8 toolkit, gcc 11);
#   3. gói Python: torch 2.1.2+cu118 (pip), numpy 1.26.4, torch-scatter, mmcv 2.1.0 từ chỉ mục dựng
#      sẵn của OpenMMLab, PyGLM...;
#   4. build droid_backends, lietorch, simple-knn, diff-gaussian-rasterization thành wheel rồi cài.
#
# Theo script đã chạy thật của nhóm trên Kaggle T4 (Phase 3/kaggle/setup_mcgs_env.sh, run 17,4
# phút, đỉnh 8,8 GB VRAM), không theo README của MCGS-SLAM từng chữ, vì ba chỗ README không chạy được
# nguyên văn:
#   - `mim install mmcv` build từ mã nguồn rất lâu: dùng wheel dựng sẵn (cu118, torch 2.1);
#   - setup.py của repo gọi setup() hai lần (droid_backends, lietorch); `bdist_wheel` thường xoá
#     thư mục build sau lần đầu nên lần hai lỗi, phải dùng `bdist_wheel -k`;
#   - `import glm` cần PyGLM.
# setup.py chỉ biên dịch sm_60..sm_86: bước 4 thêm sm_89, sm_90 khi nvidia-smi báo GPU đó (RTX 40xx,
# H100). Mặc định TORCH_CUDA_ARCH_LIST theo GPU thật, "7.5" (T4) khi không đọc được.
# Chưa chạy trên máy của dự án (mới kiểm cú pháp); script gốc đã chạy trên Kaggle (ARCHITECTURE 7.1).
# Cần git, curl, bash. Không cần conda: micromamba được tải về $DIR/mamba.
set -euo pipefail

DIR="${D2S_MCGS_DIR:-$HOME/.cache/drive2sim/mcgs-slam}"
REPO_URL="https://github.com/mcgs-slam/mcgs-slam.git"
REVISION="147eb3ac66684bed347cf3412b3e393d7c6c8171"  # khớp drive2sim/external/mcgs/__init__.py
ENV="$DIR/env"
WHEELS="$DIR/wheels"
PY="$ENV/bin/python"
export MAMBA_ROOT_PREFIX="$DIR/mamba"
MM="$MAMBA_ROOT_PREFIX/bin/micromamba"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
SCATTER_WHL="https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp310-cp310-linux_x86_64.whl"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html"
FROM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --python) shift 2 ;;  # Studio truyền, không dùng: env do micromamba dựng riêng
    *) echo "không hiểu tham số $1" >&2; exit 2 ;;
  esac
done
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
echo "MCGS-SLAM vào $DIR (đặt D2S_MCGS_DIR để đổi)"

# Compute capability của GPU (ví dụ 8.9) → "8.9"; không có nvidia-smi thì dùng T4 của Kaggle.
CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' || true)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CAPS:-7.5}}"
export MAX_JOBS="${MAX_JOBS:-4}" MPLBACKEND=Agg PYOPENGL_PLATFORM=egl

if [ "$FROM" -le 1 ]; then
  step "1/4 mã nguồn MCGS-SLAM @ ${REVISION:0:10}"
  if [ ! -d "$DIR/.git" ]; then
    mkdir -p "$DIR"
    git clone "$REPO_URL" "$DIR"
  fi
  git -C "$DIR" fetch --quiet origin
  git -C "$DIR" checkout --quiet "$REVISION"
  git -C "$DIR" submodule update --init --recursive
fi

if [ "$FROM" -le 2 ]; then
  step "2/4 env micromamba (Python 3.10, CUDA 11.8, gcc 11)"
  if [ ! -x "$MM" ]; then
    mkdir -p "$MAMBA_ROOT_PREFIX"
    (cd "$MAMBA_ROOT_PREFIX" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba)
  fi
  if [ ! -x "$PY" ]; then
    "$MM" create -y -p "$ENV" --strict-channel-priority -c nvidia/label/cuda-11.8.0 -c conda-forge \
      python=3.10 pip suitesparse cuda-toolkit gcc_linux-64=11 gxx_linux-64=11 ninja git
  fi
fi

if [ "$FROM" -le 3 ]; then
  step "3/4 gói Python (torch 2.1.2+cu118, mmcv 2.1.0, PyGLM...)"
  "$PY" -m pip install -q --upgrade pip "setuptools==69.5.1" wheel
  "$PY" -m pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url "$TORCH_INDEX"
  printf '%s\n' torch==2.1.2 torchvision==0.16.2 numpy==1.26.4 > "$DIR/constraints.txt"
  "$PY" -m pip install -q -c "$DIR/constraints.txt" numpy==1.26.4 scipy opencv-python tqdm matplotlib \
    pyyaml lightning timm open3d evo munch plyfile sophuspy torchmetrics pyrender imgviz rich glfw PyGLM
  "$PY" -m pip install -q "$SCATTER_WHL"
  "$PY" -m pip install -q -c "$DIR/constraints.txt" --no-deps \
    "git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git" \
    || echo "cảnh báo: evaluate_3d_reconstruction_lib lỗi (chỉ dùng để đánh giá mesh)"
  "$PY" -m pip install -q -c "$DIR/constraints.txt" mmengine "mmcv==2.1.0" -f "$MMCV_INDEX" \
    || echo "cảnh báo: mmcv lỗi"
fi

if [ "$FROM" -le 4 ]; then
  step "4/4 build phần mở rộng CUDA (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  export CUDA_HOME="$ENV"
  [ -e "$ENV/lib64" ] || ln -s "$ENV/lib" "$ENV/lib64"
  export CPATH="$ENV/targets/x86_64-linux/include:${CPATH:-}"
  export LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
  export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
  export PATH="$ENV/bin:$PATH" FORCE_CUDA=1
  mkdir -p "$WHEELS"
  "$ENV/bin/nvcc" --version | tail -2
  cd "$DIR"
  # setup.py chỉ có sm_60..86: thêm GPU mới hơn (idempotent)
  for cc in $(echo "$TORCH_CUDA_ARCH_LIST" | tr ';' ' ' | tr -d '.'); do
    if [ "$cc" -gt 86 ] && ! grep -q "compute_$cc" setup.py; then
      sed -i "s#^\( *\)'-gencode=arch=compute_86,code=sm_86',#&\n\1'-gencode=arch=compute_$cc,code=sm_$cc',#" setup.py
      echo "thêm sm_$cc vào setup.py"
    fi
  done
  build() {  # build <mẫu wheel> <thư mục>
    if ! ls "$WHEELS"/$1 >/dev/null 2>&1; then
      "$PY" -m pip wheel -v --no-deps --no-build-isolation -w "$WHEELS" "$2"
    fi
  }
  build 'simple_knn-*.whl' "$DIR/thirdparty/simple-knn"
  build 'diff_gaussian_rasterization-*.whl' "$DIR/thirdparty/diff-gaussian-rasterization"
  if ! ls "$WHEELS"/droid_backends-*.whl "$WHEELS"/lietorch-*.whl >/dev/null 2>&1; then
    # -k giữ build/bdist.*: wheel lietorch mang cả .so của droid_backends (cùng nguồn)
    rm -rf build/bdist.*
    "$PY" setup.py bdist_wheel -k -d "$WHEELS"
    if ! ls "$WHEELS"/lietorch-*.whl >/dev/null 2>&1; then
      "$PY" -m pip wheel -v --no-deps --no-build-isolation -w "$WHEELS" "$DIR/thirdparty/lietorch"
    fi
  fi
  git rev-parse HEAD > "$WHEELS/mcgs_commit.txt"
  "$PY" -m pip install -q --no-deps --force-reinstall "$WHEELS"/*.whl
fi

step "kiểm tra"
cd "$DIR"
"$PY" - <<'PYEOF'
import importlib, sys, torch
gpu = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda:", gpu)
bad = []
for name in ["droid_backends", "lietorch", "simple_knn._C", "diff_gaussian_rasterization",
             "torch_scatter", "mmcv", "mmengine", "evo", "glm"]:
    try:
        importlib.import_module(name)
        print("  ok  ", name)
    except Exception as error:  # noqa: BLE001
        print("  LỖI ", name, type(error).__name__, error)
        bad.append(name)
if gpu:
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
required = {"droid_backends", "lietorch", "simple_knn._C", "diff_gaussian_rasterization", "torch_scatter"}
if gpu and required & set(bad):
    sys.exit(f"thiếu bắt buộc: {sorted(required & set(bad))}")
PYEOF
