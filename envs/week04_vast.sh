#!/usr/bin/env bash
# Dựng một máy vast.ai cho benchmark tuần 04, từ thuê máy tới lúc chạy được đợt 1
# (docs/experiments/benchmark_week04.md mục 3). Chạy trên máy mình, ở gốc repo, nhánh
# feat/bench-week04; việc trên máy thuê do envs/week04_remote.sh làm.
#
#   bash envs/week04_vast.sh rent <offer_id>   # thuê máy (vastai CLI), chờ SSH, ghi .vast.env
#   bash envs/week04_vast.sh bootstrap         # rsync, git, uv, Kaggle CLI, gcloud trên máy thuê
#   bash envs/week04_vast.sh setup             # env drive2sim (nền), rồi MCGS, NuRec, venv Waymo (nền)
#   bash envs/week04_vast.sh creds             # chép token Kaggle và SMPL_NEUTRAL.pkl lên máy thuê
#   bash envs/week04_vast.sh data-kitti        # tải raw KITTI-360 0003 từ Kaggle, ingest A2 và cảnh dài (nền)
#   bash envs/week04_vast.sh waymo-auth        # gcloud auth login (tương tác, mở link trong trình duyệt)
#   bash envs/week04_vast.sh data-waymo        # tải segment W1, ingest (nền)
#   bash envs/week04_vast.sh finish            # split, lidar-check, in hàng đợi đợt 1 (chạy trước mặt)
#   bash envs/week04_vast.sh smoke             # run --quick OmniRe trên W1 để bắt lỗi GPU (~10 phút)
#   bash envs/week04_vast.sh status            # các mốc đã xong và dòng cuối của mỗi log
#   bash envs/week04_vast.sh destroy           # kéo runs/, cache/ về rồi huỷ máy thuê (hỏi lại)
# Sau `finish` và `smoke`: bash envs/vast.sh wave 1 (mục 3 của tài liệu trên).
#
# Cần trên máy mình: vastai (pip install vastai; vastai set api-key ...), khoá SSH công khai đã thêm ở
# console.vast.ai → Keys, ssh, rsync; token Kaggle (~/.kaggle/kaggle.json hoặc ~/.kaggle/access_token)
# của tài khoản đọc được notebook kascnte/d2s-raw-kitti360-0003; file SMPL (đặt SMPL_PKL=<đường dẫn>);
# tài khoản Google đã nhận điều khoản Waymo Open Dataset.
#
# Các bước chạy nền ghi mốc *_DONE vào log trên máy thuê; `status` cho biết bước nào xong. Chạy lại
# một bước là an toàn: phần đã có (env, dữ liệu tải về, cảnh đã ingest) được bỏ qua.
# Chưa chạy thật trên vast.ai: lần đầu nên theo dõi từng bước bằng `status`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO/.vast.env"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-200}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
VAST_DIR="${VAST_DIR:-/workspace/d2s}"
export VAST_DIR

need_host() {
  [ -n "${VAST_HOST:-}" ] || { echo "chưa có máy: chạy 'rent <offer_id>' hoặc đặt VAST_HOST, VAST_PORT" >&2; exit 2; }
  export VAST_HOST VAST_PORT="${VAST_PORT:-22}"
}

ssh_opts() { echo "-p ${VAST_PORT:-22} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"; }

raw_ssh() {  # chạy lệnh trên máy thuê, chưa cần conda env (bước bootstrap)
  # shellcheck disable=SC2046
  ssh $(ssh_opts) "$VAST_HOST" "$@"
}

vast() { bash "$REPO/envs/vast.sh" "$@"; }

remote_bg() {  # remote_bg <bước> <tên log>: chạy envs/week04_remote.sh <bước> ở nền, log ở logs/
  vast exec "mkdir -p logs; nohup bash envs/week04_remote.sh $1 > logs/$2.log 2>&1 & echo 'đang chạy nền: logs/$2.log'"
}

cmd_rent() {
  local offer="${1:?cần offer id (cột id của vastai search offers)}" out id url
  command -v vastai > /dev/null || { echo "cần vastai CLI: pip install vastai" >&2; exit 2; }
  out="$(vastai create instance "$offer" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw)"
  id="$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
  echo "đã thuê instance $id (offer $offer); chờ máy chạy..."
  for _ in $(seq 1 60); do
    url="$(vastai ssh-url "$id" 2>/dev/null || true)"
    if [[ "$url" =~ ^ssh://([^@]+)@([^:]+):([0-9]+) ]]; then
      VAST_HOST="${BASH_REMATCH[1]}@${BASH_REMATCH[2]}"
      VAST_PORT="${BASH_REMATCH[3]}"
      if ssh -p "$VAST_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        -o BatchMode=yes "$VAST_HOST" true 2> /dev/null; then
        printf 'VAST_INSTANCE_ID=%s\nVAST_HOST=%s\nVAST_PORT=%s\n' "$id" "$VAST_HOST" "$VAST_PORT" > "$ENV_FILE"
        echo "SSH được: $VAST_HOST cổng $VAST_PORT (ghi ở .vast.env; tiền thuê tính từ bây giờ)"
        return 0
      fi
    fi
    sleep 20
  done
  echo "sau 20 phút vẫn chưa SSH được instance $id: xem 'vastai show instance $id'; nhớ huỷ nếu bỏ" >&2
  exit 1
}

cmd_bootstrap() {
  need_host
  raw_ssh 'bash -s' << 'EOF'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq rsync git curl tmux unzip > /dev/null
python3 -m pip install -q uv
# kaggle trên PyPI không khai báo phụ thuộc six, nên lệnh kaggle báo ModuleNotFoundError nếu không thêm.
uv tool install -q --with six kaggle || uv tool upgrade -q --with six kaggle
if [ ! -x /opt/google-cloud-sdk/bin/gcloud ]; then
  curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt > /dev/null
fi
# ~/.profile vì vast.sh chạy lệnh bằng login shell không tương tác (~/.bashrc thường thoát sớm)
grep -q 'week04 PATH' ~/.profile 2> /dev/null ||
  echo 'export PATH="$HOME/.local/bin:/opt/google-cloud-sdk/bin:$PATH"  # week04 PATH' >> ~/.profile
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
df -h /workspace | tail -1
echo BOOTSTRAP_DONE
EOF
  vast check
}

cmd_setup() {
  need_host
  vast setup
  remote_bg extras setup-extras
}

cmd_creds() {
  need_host
  local sent=0 file
  raw_ssh 'mkdir -p ~/.kaggle && chmod 700 ~/.kaggle'
  for file in ~/.kaggle/kaggle.json ~/.kaggle/access_token; do
    if [ -f "$file" ]; then
      # shellcheck disable=SC2046
      scp -q -P "$VAST_PORT" -o StrictHostKeyChecking=accept-new "$file" "$VAST_HOST:.kaggle/"
      sent=1
    fi
  done
  raw_ssh 'chmod 600 ~/.kaggle/* 2>/dev/null || true'
  [ "$sent" = 1 ] && echo "đã chép token Kaggle" || echo "CẢNH BÁO: không thấy ~/.kaggle/kaggle.json hoặc access_token" >&2
  if [ -n "${SMPL_PKL:-}" ] && [ -f "$SMPL_PKL" ]; then
    raw_ssh "mkdir -p '$VAST_DIR/smpl_models'"
    scp -q -P "$VAST_PORT" "$SMPL_PKL" "$VAST_HOST:$VAST_DIR/smpl_models/SMPL_NEUTRAL.pkl"
    echo "đã chép SMPL vào $VAST_DIR/smpl_models/ (tool tự tìm ở đó)"
  else
    echo "CẢNH BÁO: chưa chép SMPL (đặt SMPL_PKL=<SMPL_NEUTRAL.pkl>): OmniRe trên W1 (#1) cần nó" >&2
  fi
}

cmd_waymo_auth() {
  need_host
  echo "mở link hiện ra trong trình duyệt, đăng nhập tài khoản Google đã nhận điều khoản Waymo, dán mã lại"
  # shellcheck disable=SC2046
  ssh -t $(ssh_opts) "$VAST_HOST" '/opt/google-cloud-sdk/bin/gcloud auth login --no-launch-browser'
}

cmd_finish() {
  need_host
  vast exec 'bash envs/week04_remote.sh finish'
}

cmd_smoke() {
  need_host
  vast run omnire --scene waymo_segment-10455472356147194054_1560_000_1580_000_with_camera_labels \
    --split every8_5cams --set fastgs=true --quick
}

cmd_status() {
  need_host
  vast exec 'for f in setup.log logs/setup-extras.log logs/data-kitti.log logs/data-waymo.log; do
      [ -f "$f" ] || { echo "== $f: chưa chạy"; continue; }
      echo "== $f: $(grep -ho "[A-Z_]*_DONE\|[A-Z_]*_FAILED" "$f" | tr "\n" " ")"; tail -n 3 "$f"; done
    echo "== cảnh:"; ls data 2>/dev/null; df -h . | tail -1'
}

cmd_destroy() {
  need_host
  vast pull
  read -r -p "đã kéo runs/, cache/ về. Huỷ instance ${VAST_INSTANCE_ID:-?} (mất mọi dữ liệu trên máy)? [y/N] " answer
  if [ "$answer" = y ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    vastai destroy instance "$VAST_INSTANCE_ID" && rm -f "$ENV_FILE"
  else
    echo "chưa huỷ"
  fi
}

command="${1:-}"; shift || true
case "$command" in
  rent) cmd_rent "$@" ;;
  bootstrap) cmd_bootstrap ;;
  setup) cmd_setup ;;
  creds) cmd_creds ;;
  data-kitti) need_host; remote_bg kitti data-kitti ;;
  waymo-auth) cmd_waymo_auth ;;
  data-waymo) need_host; remote_bg waymo data-waymo ;;
  finish) cmd_finish ;;
  smoke) cmd_smoke ;;
  status) cmd_status ;;
  destroy) cmd_destroy ;;
  *) sed -n 2,26p "${BASH_SOURCE[0]}"; exit 2 ;;
esac
