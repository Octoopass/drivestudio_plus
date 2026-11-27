#!/usr/bin/env bash
# Chỉ cho một tiến trình train chạy tại một thời điểm trên máy thuê, các tiến trình train khác bị
# tạm dừng (SIGSTOP) và tiếp tục (SIGCONT) khi tới lượt, theo thứ tự tiến trình cũ nhất trước.
#
#   nohup bash envs/bench_serial.sh benchmarks/queues/wave1.tsv > logs/serial.log 2>&1 &
#
# Vì sao: container thuê trên vast.ai bị giới hạn CPU (cgroup cpu.max, khoảng 61 lõi) trong khi mỗi
# tiến trình train mở khoảng 320 luồng. Hai run chạy song song (bench_queue.sh, mỗi GPU một worker) bị
# throttle liên tục: OmniRe đo được 0,55 giây/bước khi chạy cùng PVG và 0,135 giây/bước khi PVG bị
# dừng (cùng cảnh, cùng GPU), nên chạy lần lượt xong sớm hơn hẳn chạy song song, dù GPU thứ hai rảnh.
# Đây là đo trên một cặp run (OmniRe + PVG, cảnh Waymo W1), chưa kiểm trên cặp khác.
#
# Lưu ý cho kết quả: thời gian train ghi trong log của run bị dừng gồm cả thời gian chờ; log của script
# này (mỗi dòng giờ, hành động, pid) cho biết các khoảng dừng để trừ ra khi so thời gian train.
#
# Script thoát khi file đợt (tham số 1, tên file .out cùng đợt: logs/wave<N>.out) có QUEUE_DONE và không
# còn tiến trình train nào; khi thoát luôn SIGCONT mọi tiến trình để không để lại tiến trình bị dừng.
set -u

DONE_FILE="${1:?cần file log của đợt, ví dụ logs/wave1.out}"
# Neo đầu chuỗi: chỉ khớp tiến trình train chính, không khớp lệnh ssh hay chính script này. Worker của
# DataLoader (fork) có cùng dòng lệnh nhưng cha là một tiến trình train, nên bị loại ở bước lọc ppid.
PATTERN='^/opt/conda/envs/drive2sim/bin/python -m drive2sim.recon.tools.train'

log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*"; }

main_train_pids() {  # in pid của tiến trình train chính, cũ nhất trước
  local all pid ppid
  all="$(pgrep -f "$PATTERN" || true)"
  [ -n "$all" ] || return 0
  for pid in $all; do
    ppid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
    echo "$all" | grep -qx "$ppid" && continue
    echo "$(ps -o etimes= -p "$pid" | tr -d ' ') $pid"
  done | sort -rn | awk '{print $2}'
}

state_of() { ps -o stat= -p "$1" | cut -c1; }

release_all() {
  local pid
  for pid in $(main_train_pids); do kill -CONT "$pid" 2> /dev/null || true; done
  log "thoát, đã tiếp tục mọi tiến trình"
}
trap release_all EXIT

log "bắt đầu, theo dõi $DONE_FILE"
while true; do
  first=1
  for pid in $(main_train_pids); do
    st="$(state_of "$pid")"
    if [ "$first" = 1 ]; then
      first=0
      if [ "$st" = "T" ]; then kill -CONT "$pid" && log "tiếp tục pid $pid"; fi
    else
      if [ "$st" != "T" ]; then kill -STOP "$pid" && log "tạm dừng pid $pid"; fi
    fi
  done
  if [ "$first" = 1 ] && grep -q QUEUE_DONE "$DONE_FILE" 2> /dev/null; then
    break
  fi
  sleep 15
done
