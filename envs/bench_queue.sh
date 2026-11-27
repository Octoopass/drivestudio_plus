#!/usr/bin/env bash
# Chạy một hàng đợi run benchmark trên máy nhiều GPU: mỗi GPU một worker, worker rảnh lấy dòng kế
# tiếp của hàng đợi và chạy `d2s run start` với CUDA_VISIBLE_DEVICES của nó. Chạy trên máy thuê,
# trong thư mục làm việc (envs/vast.sh wave gọi script này).
#
#   bash envs/bench_queue.sh <hàng đợi.tsv> [--gpus 0,1] [--no-check] [--restart]
#
# Hàng đợi: mỗi dòng `<số run><tab><tham số của d2s run start>`, do
# `python -m drive2sim.scripts.setup_benchmark_week04 queue --wave N --out <file>` ghi, run dài trước.
# Trạng thái ở logs/queue-<tên file>/:
#   pending   dòng chưa chạy (worker lấy dòng đầu, có khoá flock)
#   done      <giờ><tab><số run><tab><GPU><tab>ok|fail<tab><log>
#   failed    dòng của run lỗi, để chạy lại: bash envs/bench_queue.sh logs/queue-<tên>/failed
#   run-*.log log của từng run
# Chạy lại cùng hàng đợi thì tiếp từ `pending` (run đang chạy dở lúc máy dừng phải tự thêm lại);
# --restart chép lại hàng đợi từ đầu.
#
# Trước khi chạy, mọi dòng được kiểm bằng `d2s approach check` (không cần GPU) để lỗi cấu hình, thiếu
# split hoặc cảnh hiện ngay chứ không giữa đêm; --no-check bỏ bước này.
# Hai run cùng cần một mục cache (mask, SMPL của cùng cảnh) thì run sau chờ run trước ghi xong
# (D2S_CACHE_WAIT_S, mặc định ở đây 3 giờ) thay vì lỗi.
set -uo pipefail

QUEUE="${1:?cần file hàng đợi}"; shift
GPUS=""
CHECK=1
RESTART=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --no-check) CHECK=0; shift ;;
    --restart) RESTART=1; shift ;;
    *) echo "không hiểu tham số $1" >&2; exit 2 ;;
  esac
done
[ -f "$QUEUE" ] || { echo "không có file $QUEUE" >&2; exit 2; }
if [ -z "$GPUS" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
[ -n "$GPUS" ] || { echo "không thấy GPU (đặt --gpus)" >&2; exit 2; }
export D2S_CACHE_WAIT_S="${D2S_CACHE_WAIT_S:-10800}"

name="$(basename "$QUEUE")"; name="${name%.*}"
STATE="logs/queue-$name"
mkdir -p "$STATE"
if [ "$RESTART" = 1 ] || [ ! -f "$STATE/pending" ]; then
  grep -v '^[[:space:]]*\(#\|$\)' "$QUEUE" > "$STATE/pending" || true
fi
touch "$STATE/done" "$STATE/failed"

# Tách một dòng thành số run và mảng tham số (args do setup_benchmark_week04 ghi bằng shlex.join).
split_line() {  # split_line <dòng>  -> đặt NUM và ARGV
  NUM="${1%%$'\t'*}"
  local rest="${1#*$'\t'}"
  eval "ARGV=($rest)"
}

if [ "$CHECK" = 1 ]; then
  bad=0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    split_line "$line"
    if ! d2s approach check "${ARGV[@]}" > "$STATE/check-$NUM.log" 2>&1; then
      echo "#$NUM không qua approach check: $STATE/check-$NUM.log" >&2
      tail -n 5 "$STATE/check-$NUM.log" >&2
      bad=1
    fi
  done < "$STATE/pending"
  [ "$bad" = 0 ] || { echo "dừng: sửa lỗi trên rồi chạy lại (hoặc --no-check)" >&2; exit 3; }
  echo "approach check: mọi run qua"
fi

pop() {  # in dòng đầu của pending và xoá nó; rỗng khi hết
  local line=""
  {
    flock 9
    line="$(head -n 1 "$STATE/pending")"
    [ -n "$line" ] && sed -i '1d' "$STATE/pending"
  } 9> "$STATE/lock"
  printf '%s' "$line"
}

worker() {
  local gpu="$1" line log status
  while true; do
    line="$(pop)"
    [ -n "$line" ] || break
    split_line "$line"
    local split="" i
    for ((i = 0; i < ${#ARGV[@]} - 1; i++)); do
      [ "${ARGV[i]}" = --split ] && split="${ARGV[i + 1]}"
    done
    log="$STATE/run-$NUM-$split-gpu$gpu-$(date +%m%d-%H%M%S).log"
    echo "$(date '+%F %T') GPU $gpu bắt đầu #$NUM: d2s run start ${ARGV[*]}"
    if CUDA_VISIBLE_DEVICES="$gpu" d2s run start "${ARGV[@]}" > "$log" 2>&1; then
      status=ok
    else
      status=fail
      printf '%s\n' "$line" >> "$STATE/failed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$NUM" "$gpu" "$status" "$log" >> "$STATE/done"
    echo "$(date '+%F %T') GPU $gpu xong #$NUM: $status ($log)"
  done
}

echo "hàng đợi $QUEUE: $(wc -l < "$STATE/pending") run, GPU $GPUS, trạng thái ở $STATE"
IFS=',' read -r -a gpu_list <<< "$GPUS"
for gpu in "${gpu_list[@]}"; do
  worker "$gpu" &
done
wait
fails="$(grep -c $'\tfail\t' "$STATE/done" || true)"
echo "QUEUE_DONE $name: $(wc -l < "$STATE/done") run đã chạy, $fails lỗi (dòng chạy lại ở $STATE/failed)"
