#!/usr/bin/env bash
# Cài drive2sim bằng một lệnh trên Linux và macOS (docs/install.md).
#
#   bash envs/install.sh                 # Linux có GPU NVIDIA: cài đủ; máy khác: chỉ Studio
#   bash envs/install.sh --studio-only   # chỉ Studio + CLI (chạy run trên Kaggle, Colab)
#   bash envs/install.sh --full --start  # cài đủ rồi bật Studio
#
# Hai chế độ:
# - full (Linux, GPU NVIDIA): venv Python 3.10, drive2sim, rồi `d2s libraries install --all`
#   (môi trường chính bằng envs/install_main.sh, Instant NuRec, SDK dataset, công cụ), như nút
#   "Cài tất cả" của Studio. Mất cỡ 1,5 giờ (build gsplat-geer, nvdiffrast, pytorch3d).
# - studio (macOS, Linux không GPU): venv Python 3.11, drive2sim với Studio, Kaggle CLI, Colab CLI;
#   quản lý cảnh, chạy run trên GPU của Kaggle, Colab, xem kết quả. Không train trên máy.
#
# Mọi thứ nằm trong D2S_HOME (mặc định ~/.drive2sim): venv/, bin/drive2sim-studio. Chạy lại an
# toàn: bước đã xong được dùng lại, môi trường chính cài tiếp từ bước lỗi.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D2S_HOME="${D2S_HOME:-$HOME/.drive2sim}"
VENV="$D2S_HOME/venv"
WORKDIR="${D2S_WORKDIR:-$HOME/drive2sim-work}"
MODE=auto
PORT=8765
START=0
SKIP=""
TOOLS=1
LINK=1

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<EOF
Tuỳ chọn:
  --full            cài đủ (Linux, GPU NVIDIA)
  --studio-only     chỉ Studio + CLI
  --workdir DIR     thư mục làm việc (mặc định $WORKDIR)
  --port N          cổng Studio (mặc định $PORT)
  --no-nurec        không cài Instant NuRec (chế độ full)
  --start           bật Studio sau khi cài
  --skip-tools      không cài, nâng cấp Kaggle CLI, Colab CLI
  --no-link         không tạo lệnh drive2sim-studio, d2s trong ~/.local/bin
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --full) MODE=full; shift ;;
    --studio-only) MODE=studio; shift ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-nurec) SKIP="env-nurec"; shift ;;
    --start) START=1; shift ;;
    --skip-tools) TOOLS=0; shift ;;
    --no-link) LINK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "không hiểu tham số $1 (xem --help)" >&2; exit 2 ;;
  esac
done

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
OS="$(uname -s)"
has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
if [ "$MODE" = auto ]; then
  if [ "$OS" = Linux ] && has_gpu; then MODE=full; else MODE=studio; fi
fi
if [ "$MODE" = full ] && [ "$OS" != Linux ]; then
  echo "chế độ full chỉ chạy trên Linux có GPU NVIDIA; trên $OS dùng --studio-only" >&2
  exit 2
fi
if [ "$MODE" = full ] && ! has_gpu; then
  echo "không thấy GPU NVIDIA (nvidia-smi): cài full xong vẫn không train được; dùng --studio-only" >&2
  exit 2
fi
PYVER=3.11
[ "$MODE" = full ] && PYVER=3.10  # môi trường chính chốt Python 3.10 (envs/main.yml)
echo "drive2sim: chế độ $MODE, mã nguồn $SRC, venv $VENV (Python $PYVER), thư mục làm việc $WORKDIR"

step "1 uv (quản lý Python)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
  else
    echo "cần curl hoặc python3 để cài uv (https://docs.astral.sh/uv/)" >&2
    exit 1
  fi
fi
uv --version

step "2 venv Python $PYVER"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import sys; sys.exit(sys.version_info[:2] != tuple(map(int, '$PYVER'.split('.'))))"; then
  echo "dùng lại $VENV"
else
  rm -rf "$VENV"
  mkdir -p "$D2S_HOME"
  uv venv --seed --python "$PYVER" "$VENV"  # --seed: có pip (install_main.sh, bảng Thư viện)
fi
PY="$VENV/bin/python"

step "3 drive2sim + Studio"
uv pip install --python "$PY" -e "$SRC[studio]"

step "4 Kaggle CLI, Colab CLI"
if [ "$TOOLS" = 1 ]; then
  uv tool install --upgrade kaggle || echo "chưa cài được Kaggle CLI: cài sau bằng uv tool install kaggle"
  uv tool install --upgrade google-colab-cli --with keyrings.alt \
    || echo "chưa cài được Colab CLI: cài sau ở trang Thư viện của Studio"
else
  echo "bỏ qua (--skip-tools)"
fi

mkdir -p "$WORKDIR"
if [ "$MODE" = full ]; then
  step "5 môi trường train, Instant NuRec, SDK dataset (d2s libraries install --all)"
  ARGS=(libraries install --all --workdir "$WORKDIR")
  [ -n "$SKIP" ] && ARGS+=(--skip "$SKIP")
  "$VENV/bin/d2s" "${ARGS[@]}" || {
    echo "một số mục cài lỗi: xem log ở $WORKDIR/logs/, sửa nguyên nhân rồi chạy lại lệnh này" >&2
    exit 1
  }
fi

step "6 lệnh bật Studio"
mkdir -p "$D2S_HOME/bin"
LAUNCHER="$D2S_HOME/bin/drive2sim-studio"
cat >"$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Bật drive2sim Studio (do envs/install.sh tạo). Tham số thêm chuyển cho d2s studio.
export PATH="\$HOME/.local/bin:\$PATH"
exec "$VENV/bin/d2s" studio --workdir "\${D2S_WORKDIR:-$WORKDIR}" --port "\${D2S_PORT:-$PORT}" "\$@"
EOF
chmod +x "$LAUNCHER"
if [ "$LINK" = 1 ] && { [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; }; then
  ln -sf "$LAUNCHER" "$HOME/.local/bin/drive2sim-studio"
  ln -sf "$VENV/bin/d2s" "$HOME/.local/bin/d2s"
fi

cat <<EOF

Xong (chế độ $MODE).
  Bật Studio:   drive2sim-studio          (hoặc $LAUNCHER)
  Mở:           http://127.0.0.1:$PORT
  Máy chủ không màn hình: Studio chỉ nghe 127.0.0.1; từ máy của bạn chạy
                ssh -L $PORT:127.0.0.1:$PORT $(id -un)@$(hostname)
                rồi mở http://127.0.0.1:$PORT
  Lệnh CLI:     d2s --help
EOF
[ "$MODE" = studio ] && echo "  Chế độ studio: chạy run trên Kaggle hoặc Colab (Chạy mới → Nơi chạy); train trên máy cần Linux có GPU NVIDIA."

if [ "$START" = 1 ]; then
  exec "$LAUNCHER"
fi
