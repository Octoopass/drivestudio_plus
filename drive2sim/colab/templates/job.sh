#!/usr/bin/env bash
# Job chạy một run trên VM Colab (CONTRACTS mục 9.3). drive2sim/colab/executor.py chép file này lên
# VM rồi chạy nó tách khỏi kết nối (log ra file, mã thoát ghi vào file .exit), nên rớt kết nối
# "colab exec" không làm dừng job. Các bước giống notebook Kaggle (drive2sim/kaggle/templates/
# run.ipynb): clone đúng commit, cài môi trường bằng envs/install_main.sh, chạy "d2s kaggle run"
# với gói job. Chưa chạy thử trên Colab thật.
#
#   bash job.sh <thư mục gốc> <url repo> <nhánh> <commit> <thư mục gói job> <thư mục cảnh>
set -euo pipefail

ROOT="$1"        # /content/d2s
REPO_URL="$2"
BRANCH="$3"
COMMIT="$4"
BUNDLE="$5"      # gói job (run.json, split.json, scene_ref.json, resume.json)
SCENE_INPUT="$6" # nơi đã giải nén cảnh; d2s kaggle run tìm cảnh theo manifest.yaml dưới đây
REPO="$ROOT/repo"
VENV="$ROOT/venv"
WORK="$ROOT/work"

step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }

step "GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

step "code: $REPO_URL @ ${COMMIT:0:7}"
# Repo giữ lại giữa các lần chạy trong cùng phiên; lần sau chỉ fetch và checkout commit mới.
if [ ! -d "$REPO/.git" ]; then
  git clone --quiet --recurse-submodules --branch "$BRANCH" "$REPO_URL" "$REPO"
fi
git -C "$REPO" fetch --quiet origin "$BRANCH"
git -C "$REPO" checkout --quiet --force "$COMMIT"
git -C "$REPO" submodule update --quiet --init --recursive

step "môi trường"
# Python của Colab (3.12 trở lên) mới hơn bản requirements.txt hỗ trợ (numpy 1.23.5 chỉ tới 3.11),
# nên cài vào một venv Python 3.10 như envs/main.yml. Môi trường giữ lại trong phiên: chỉ cài lại
# khi envs/install_main.sh hoặc requirements.txt đổi.
cd "$REPO"
ENV_KEY="$(cat envs/install_main.sh requirements.txt | sha256sum | cut -c1-16)"
if [ "$(cat "$ROOT/env.done" 2>/dev/null || true)" != "$ENV_KEY" ]; then
  command -v uv >/dev/null || python3 -m pip install --quiet uv
  [ -x "$VENV/bin/python" ] || uv venv --quiet --seed --python 3.10 "$VENV"
  [ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
  # Build cho đúng GPU của phiên này (compute capability, ví dụ 7.5 cho T4, 8.9 cho L4).
  TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')"
  export TORCH_CUDA_ARCH_LIST
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  bash envs/install_main.sh --python "$VENV/bin/python"
  echo "$ENV_KEY" > "$ROOT/env.done"
else
  echo "đã cài ($ENV_KEY)"
fi

step "run"
# Chạy rồi luôn xoá symlink trong runs/ và cache/ (cảnh ảo trỏ vào thư mục cảnh), kể cả khi run lỗi,
# như notebook Kaggle: output tải về chỉ còn kết quả. Mã lỗi của run được trả lại sau khi dọn.
rc=0
"$VENV/bin/d2s" kaggle run "$BUNDLE" --workdir "$WORK" --input-root "$SCENE_INPUT" || rc=$?
find "$WORK/runs" "$WORK/cache" -type l -delete 2>/dev/null || true
step "xong, mã thoát $rc"
exit "$rc"
