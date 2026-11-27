#!/usr/bin/env bash
# Kéo từng run về máy mình ngay khi nó kết thúc trên máy thuê (vast.ai), không đợi cả đợt xong.
#
#   bash envs/vast_autopull.sh                 # lặp mỗi 300 giây tới khi có QUEUE_DONE hoặc bị dừng
#   AUTOPULL_INTERVAL=120 bash envs/vast_autopull.sh
#
# Mỗi vòng: liệt kê run có status succeeded hoặc failed trên máy thuê (đọc run.json), run nào chưa
# có trong file trạng thái thì gọi ``envs/vast.sh pull <run_id>``. Run failed cũng kéo về để đọc log.
# Kéo thành công mới ghi vào file trạng thái, nên lần lỗi mạng sẽ được thử lại ở vòng sau.
# Dùng ``envs/vast.sh``, nên cùng cấu hình máy thuê (``.vast.env`` hay ``VAST_HOST``).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${AUTOPULL_INTERVAL:-300}"
PULLED_FILE="${AUTOPULL_STATE:-$REPO/.vast_pulled}"       # run đã kéo đủ
LIGHT_FILE="${PULLED_FILE}.light"                          # run đã kéo bản nhẹ (không *.pth, scene.ply)
touch "$PULLED_FILE" "$LIGHT_FILE"

list_finished() {
  # in "<run_id> <status>" cho mỗi run đã kết thúc; run.json nằm ở runs/<cảnh>/<approach>/<run_id>/d2s/
  bash "$REPO/envs/vast.sh" exec 'for f in runs/*/*/*/d2s/run.json; do [ -f "$f" ] || continue;
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"run_id\"], d[\"status\"])" "$f" || true; done' \
    2>/dev/null | grep -E '^[0-9]{8}-[0-9]{6}-[a-z0-9]+-[0-9a-f]{4} (succeeded|failed)$' || true
}

while true; do
  finished="$(list_finished)"
  # tầng 1: run nào chưa có bản nhẹ thì kéo bản nhẹ ngay (số liệu, ảnh, video, log; vài chục MB)
  while read -r run_id status; do
    [ -n "$run_id" ] || continue
    grep -qxF "$run_id" "$PULLED_FILE" "$LIGHT_FILE" && continue
    echo "$(date +%T) kéo bản nhẹ $run_id ($status)"
    if D2S_PULL_LIGHT=1 bash "$REPO/envs/vast.sh" pull "$run_id" >/dev/null 2>&1; then
      echo "$run_id" >> "$LIGHT_FILE"
    else
      echo "$(date +%T) kéo bản nhẹ $run_id lỗi, thử lại vòng sau"
    fi
  done <<< "$finished"
  # tầng 2: mỗi vòng kéo đủ một run (checkpoint, PLY lớn) sau khi mọi bản nhẹ đã xong
  while read -r run_id status; do
    [ -n "$run_id" ] || continue
    grep -qxF "$run_id" "$PULLED_FILE" && continue
    echo "$(date +%T) kéo đủ $run_id"
    if bash "$REPO/envs/vast.sh" pull "$run_id" >/dev/null 2>&1; then
      echo "$run_id" >> "$PULLED_FILE"
    else
      echo "$(date +%T) kéo đủ $run_id lỗi, thử lại vòng sau"
    fi
    break
  done <<< "$finished"
  # dừng khi hàng đợi đã xong và mọi run đã kết thúc đều được kéo đủ
  if bash "$REPO/envs/vast.sh" exec 'grep -lE "QUEUE_DONE" logs/wave*.out logs/retry*.out 2>/dev/null' >/dev/null 2>&1; then
    pending="$(while read -r run_id status; do
      [ -n "$run_id" ] && ! grep -qxF "$run_id" "$PULLED_FILE" && echo "$run_id"
    done <<< "$finished")"
    if [ -z "$pending" ]; then echo "$(date +%T) QUEUE_DONE và đã kéo đủ; dừng"; break; fi
  fi
  sleep "$INTERVAL"
done
