#!/usr/bin/env bash
# Phần chạy trên máy thuê của envs/week04_vast.sh (benchmark tuần 04). Chạy trong thư mục làm việc
# (VAST_DIR, mặc định /workspace/d2s), conda env drive2sim đã bật (envs/vast.sh exec lo việc đó).
#
#   bash envs/week04_remote.sh extras   # chờ env drive2sim xong, cài MCGS-SLAM, Instant NuRec, venv Waymo
#   bash envs/week04_remote.sh kitti    # tải raw KITTI-360 0003 (output notebook Kaggle), ingest 2 cảnh
#   bash envs/week04_remote.sh waymo    # tải segment W1 bằng gsutil, ingest bằng venv Waymo
#   bash envs/week04_remote.sh finish   # split, lidar-check, in hàng đợi đợt 1
#
# Mỗi bước in <BƯỚC>_DONE khi xong, <BƯỚC>_FAILED khi lỗi; chạy lại thì bỏ qua phần đã có.
set -euo pipefail
export PATH="$HOME/.local/bin:/opt/google-cloud-sdk/bin:$PATH"

RAW="${D2S_RAW_DIR:-/workspace/raw}"
KITTI_SOURCE="${KITTI_SOURCE:-kascnte/d2s-raw-kitti360-0003}"   # notebook giữ raw chuỗi 0003
SEQUENCE="2013_05_28_drive_0003_sync"
SHORT_SCENE="kitti360_$SEQUENCE"
LONG_SCENE="kitti360_0003_long"
# frame_ids.txt của cảnh 200 khung đã dùng (ingest trên Kaggle 25/11, khung gốc 1–220): ingest lại
# ở đây phải ra đúng cảnh đó, không thì split và số liệu cũ không còn so được.
SHORT_FRAME_IDS_MD5="46efb9e5c7cdcc1f9747903cf68b2fd2"
WAYMO_SEGMENT="segment-10455472356147194054_1560_000_1580_000_with_camera_labels"
WAYMO_SCENE_INDEX=23   # chỉ số trong waymo_train_list.txt
WAYMO_VENV="${WAYMO_VENV:-$PWD/.venv-waymo}"

step="${1:-}"
fail() { echo "${step^^}_FAILED: $*"; exit 1; }
trap 'echo "${step^^}_FAILED: lệnh lỗi ở dòng $LINENO"' ERR

has_scene() { [ -f "data/$1/manifest.yaml" ]; }

extras() {
  echo "chờ env drive2sim (setup.log có SETUP_DONE)..."
  until grep -q SETUP_DONE setup.log 2> /dev/null; do
    if grep -q "^ERROR\|Traceback" setup.log 2> /dev/null && ! pgrep -f install_main.sh > /dev/null; then
      fail "cài env drive2sim lỗi, xem setup.log"
    fi
    sleep 60
  done
  bash envs/install_mcgs.sh
  bash envs/install_nurec.sh
  # Tiền xử lý Waymo cần tensorflow 2.11 + waymo-open-dataset, không hợp với env chính: venv riêng
  # (Python 3.10) có drive2sim, các gói ingest như notebook ingest Kaggle, và devkit Waymo.
  if [ ! -x "$WAYMO_VENV/bin/d2s" ]; then
    uv venv --quiet --seed --python 3.10 "$WAYMO_VENV"
    uv pip install --quiet --python "$WAYMO_VENV/bin/python" \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
      -e . "torch==2.4.1" "waymo-open-dataset-tf-2-11-0==1.6.0" \
      "opencv-python-headless==4.11.0.86" "pyquaternion==0.9.9" "matplotlib<3.9" tqdm pillow
  fi
  "$WAYMO_VENV/bin/python" -c "import tensorflow, waymo_open_dataset; print('waymo venv ok, tf', tensorflow.__version__)"
  echo EXTRAS_DONE
}

kitti() {
  local dest="$RAW/kitti360-0003" root
  mkdir -p "$dest"
  if ! find "$dest" -path '*/calibration/perspective.txt' -print -quit | grep -q .; then
    echo "tải output notebook $KITTI_SOURCE (~8 GB)..."
    kaggle kernels output "$KITTI_SOURCE" -p "$dest" -q || fail "tải $KITTI_SOURCE lỗi (token Kaggle? quyền đọc notebook?)"
    find "$dest" -maxdepth 3 -name '*.zip' -exec unzip -q -n {} -d "$dest" \;
  fi
  root="$(dirname "$(dirname "$(find "$dest" -path '*/calibration/perspective.txt' -print -quit)")")"
  [ -d "$root/calibration" ] || fail "không thấy thư mục raw KITTI-360 (có calibration/) dưới $dest"
  echo "raw KITTI-360: $root"
  if ! has_scene "$SHORT_SCENE"; then
    d2s scene ingest "$root" --sequence "$SEQUENCE" --num-frames 200 --scene "$SHORT_SCENE"
  fi
  local got
  got="$(md5sum < "data/$SHORT_SCENE/frame_ids.txt" | cut -d' ' -f1)"
  [ "$got" = "$SHORT_FRAME_IDS_MD5" ] || fail "cảnh $SHORT_SCENE khác cảnh đã dùng (frame_ids.txt md5 $got)"
  if ! has_scene "$LONG_SCENE"; then
    d2s scene ingest "$root" --sequence "$SEQUENCE" --start-frame 167 --scene "$LONG_SCENE"
  fi
  echo "cảnh dài: $(wc -l < "data/$LONG_SCENE/frame_ids.txt") khung (mong đợi 864)"
  echo KITTI_DONE
}

waymo() {
  local dest="$RAW/waymo"
  [ -x "$WAYMO_VENV/bin/d2s" ] || fail "chưa có venv Waymo: chạy bước extras trước"
  gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . ||
    fail "gcloud chưa đăng nhập: chạy 'bash envs/week04_vast.sh waymo-auth' trên máy mình"
  if [ ! -f "$dest/$WAYMO_SEGMENT.tfrecord" ]; then
    "$WAYMO_VENV/bin/python" -m drive2sim.recon.preprocess.waymo.download \
      --scene_ids "$WAYMO_SCENE_INDEX" --target_dir "$dest"
  fi
  [ -f "$dest/$WAYMO_SEGMENT.tfrecord" ] || fail "không thấy $dest/$WAYMO_SEGMENT.tfrecord sau khi tải"
  if ! has_scene "waymo_$WAYMO_SEGMENT"; then
    # d2s của venv Waymo: ingest gọi script tiền xử lý bằng chính Python đó (cần tensorflow)
    "$WAYMO_VENV/bin/d2s" scene ingest "$dest" --sequence "$WAYMO_SEGMENT" --workdir "$PWD"
  fi
  echo WAYMO_DONE
}

finish() {
  python -m drive2sim.scripts.setup_benchmark_week04 splits
  local scene
  for scene in "$LONG_SCENE" "waymo_$WAYMO_SEGMENT"; do
    has_scene "$scene" && d2s scene lidar-check "$scene" | tail -n 3
  done
  echo "--- hàng đợi đợt 1 (bench_queue.sh kiểm từng dòng bằng approach check trước khi chạy):"
  python -m drive2sim.scripts.setup_benchmark_week04 queue --wave 1
  [ -f smpl_models/SMPL_NEUTRAL.pkl ] || echo "CẢNH BÁO: chưa có smpl_models/SMPL_NEUTRAL.pkl (#1 cần)"
  echo FINISH_DONE
}

case "$step" in
  extras) extras ;;
  kitti) kitti ;;
  waymo) waymo ;;
  finish) finish ;;
  *) sed -n 2,10p "${BASH_SOURCE[0]}"; exit 2 ;;
esac
