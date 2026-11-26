#!/usr/bin/env bash
# Chạy drive2sim trên máy GPU thuê (vast.ai) qua SSH + rsync, không cần executor riêng:
# run tạo ngay trên máy thuê với ``--executor local``, rồi kéo kết quả về thư mục làm việc.
# ARCHITECTURE mục 2.17.
#
#   export VAST_HOST=root@<ip>  VAST_PORT=<cổng SSH>     # lấy ở nút Connect của máy thuê
#   bash envs/vast.sh check                 # GPU, nvcc, dung lượng đĩa; cảnh báo GPU không chạy được
#   bash envs/vast.sh setup                 # chép code, tạo conda env, cài môi trường (nền, có log)
#   bash envs/vast.sh push-scene <scene_id> # chép data/<scene_id>/ (gồm splits/) và file SMPL
#   bash envs/vast.sh run <đối số của d2s run start>   # chạy nền, ví dụ: run omnire --scene ... --split ...
#   bash envs/vast.sh status                # log mới nhất và nvidia-smi
#   bash envs/vast.sh pull [<run_id>]       # kéo runs/ và cache/ về; chạy giữa chừng được
#   bash envs/vast.sh exec <lệnh>           # chạy lệnh bất kỳ trong thư mục làm việc trên máy thuê
#   bash envs/vast.sh setup-extra           # cài thêm MCGS-SLAM và Instant NuRec (nền, có log)
#   bash envs/vast.sh wave <1|2|3> [tuỳ chọn của setup_benchmark_week04 queue]
#                                           # benchmark tuần 04: dựng hàng đợi một đợt rồi chạy nền,
#                                           # mỗi GPU một worker (envs/bench_queue.sh); BENCH_GPUS=0,1
#   bash envs/vast.sh wave-status [<1|2|3>] # run đã xong, đang chờ của đợt, nvidia-smi
#
# Image nên có CUDA toolkit (nvcc) 12.x, ví dụ pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel, vì
# envs/install_main.sh build gsplat-geer, nvdiffrast, pytorch3d từ mã nguồn với torch 2.4.1+cu121.
# GPU Blackwell (RTX 50xx, compute capability 12.x) không chạy được với cu121: dùng RTX 3090, 4090,
# A100 hoặc H100. Dữ liệu lái xe có license riêng: giữ máy thuê ở chế độ riêng tư, xoá khi xong.
# Chưa chạy thật trên vast.ai (docs/drive2sim/ARCHITECTURE.md mục 7.1).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# .vast.env: máy do envs/week04_vast.sh rent thuê (VAST_HOST, VAST_PORT); biến đặt sẵn được ưu tiên
if [ -z "${VAST_HOST:-}" ] && [ -f "$REPO/.vast.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.vast.env"
fi
: "${VAST_HOST:?đặt VAST_HOST=root@<ip> (hoặc chạy envs/week04_vast.sh rent)}"
VAST_PORT="${VAST_PORT:-22}"
VAST_DIR="${VAST_DIR:-/workspace/d2s}"     # thư mục code và thư mục làm việc trên máy thuê
LOCAL_WORKDIR="${D2S_WORKDIR:-$REPO}"       # thư mục làm việc trên máy này
SSH_OPTS=(-p "$VAST_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
# VAST_DRY_RUN=1: chỉ in lệnh sẽ chạy (kiểm tra script mà không cần máy thuê).
DRY="${VAST_DRY_RUN:-0}"

remote() {
  # chạy lệnh trong login shell trên máy thuê, đã vào thư mục làm việc và bật conda env drive2sim
  local cmd="$1"
  local prelude='source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true; '
  prelude+="cd '$VAST_DIR'; export D2S_WORKDIR='$VAST_DIR'; "
  prelude+='conda activate drive2sim 2>/dev/null || true; '
  if [ "$DRY" = 1 ]; then echo "ssh ${SSH_OPTS[*]} $VAST_HOST bash -lc <<< $cmd"; return 0; fi
  ssh "${SSH_OPTS[@]}" "$VAST_HOST" bash -lc "$(printf '%q' "$prelude$cmd")"
}

sync_to() {  # rsync <nguồn> <đích trên máy thuê> [tuỳ chọn rsync...]
  local src="$1" dst="$2"; shift 2
  if [ "$DRY" = 1 ]; then echo "rsync -az $* $src $VAST_HOST:$dst"; return 0; fi
  rsync -az --info=progress2 "$@" -e "ssh ${SSH_OPTS[*]}" "$src" "$VAST_HOST:$dst"
}

sync_from() {  # rsync <nguồn trên máy thuê> <đích> [tuỳ chọn rsync...]
  local src="$1" dst="$2"; shift 2
  if [ "$DRY" = 1 ]; then echo "rsync -az $* $VAST_HOST:$src $dst"; return 0; fi
  rsync -az --info=progress2 "$@" -e "ssh ${SSH_OPTS[*]}" "$VAST_HOST:$src" "$dst"
}

cmd_check() {
  remote 'nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader;
          (nvcc --version | tail -1) || echo "không có nvcc: dùng image *-devel";
          df -h . | tail -1;
          nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F. "\$1>=12{print \"CẢNH BÁO: GPU Blackwell (cc \" \$0 \"), cu121 không chạy được\"}"'
}

cmd_setup() {
  remote "mkdir -p '$VAST_DIR'"
  # Chép cả file chưa commit (máy phát triển là nơi sửa code), ghi commit và cờ "bẩn" để truy vết.
  # Mẫu có "/" đầu chỉ khớp ở gốc: "data" không gốc sẽ loại cả drive2sim/recon/data (mã nguồn).
  sync_to "$REPO/" "$VAST_DIR/" --exclude .git --exclude .venv --exclude node_modules \
    --exclude /data --exclude /runs --exclude /cache --exclude /.demo --exclude /.d2s \
    --exclude '__pycache__' --exclude '*.pyc' --exclude /drive2sim/studio/frontend/src
  local commit dirty
  commit="$(git -C "$REPO" rev-parse HEAD)"
  dirty="$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)"
  remote "echo '$commit dirty_files=$dirty' > COMMIT"
  # Cài nền, có log: build gsplat-geer, pytorch3d... mất hàng chục phút. TORCH_CUDA_ARCH_LIST theo
  # GPU thật để build nhanh hơn; wheel đã build giữ ở wheels/ cho lần dựng lại máy sau.
  remote 'nohup bash -c "
    set -e
    conda env list | grep -q \"^drive2sim \" || conda env create -f envs/main.yml
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate drive2sim
    export TORCH_CUDA_ARCH_LIST=\"\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd\";\")\"
    export D2S_WHEEL_CACHE=\"$PWD/wheels\" D2S_WHEEL_SOURCES=\"$PWD/wheels\"
    bash envs/install_main.sh
    echo SETUP_DONE
  " > setup.log 2>&1 &'
  echo "đang cài ở nền; xem 'bash envs/vast.sh status', xong khi log có SETUP_DONE"
}

cmd_push_scene() {
  local scene="${1:?cần scene_id}"
  remote "mkdir -p data/'$scene'"
  sync_to "$LOCAL_WORKDIR/data/$scene/" "$VAST_DIR/data/$scene/"
  # File mô hình SMPL (OmniRe dựng người) nằm ngoài data/; chép nếu có (paths.SMPL_MODEL_ENV).
  if [ -n "${D2S_SMPL_MODEL:-}" ] && [ -f "${D2S_SMPL_MODEL}" ]; then
    remote "mkdir -p smpl_models"
    sync_to "$D2S_SMPL_MODEL" "$VAST_DIR/smpl_models/"
    echo "đã chép SMPL; trên máy thuê đặt D2S_SMPL_MODEL=$VAST_DIR/smpl_models/$(basename "$D2S_SMPL_MODEL")"
  fi
}

cmd_run() {
  local stamp args
  stamp="$(date +%Y%m%d-%H%M%S)"
  args="$(printf '%q ' "$@")"
  remote "mkdir -p logs; nohup d2s run start $args > logs/run-$stamp.log 2>&1 &
          echo \"đã chạy nền, log: $VAST_DIR/logs/run-$stamp.log\""
}

cmd_status() {
  remote 'tail -n 5 setup.log 2>/dev/null; echo ---; tail -n 20 "$(ls -t logs/run-*.log 2>/dev/null | head -1)" 2>/dev/null;
          echo ---; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader;
          d2s run list 2>/dev/null | head -20'
}

cmd_pull() {
  local run_id="${1:-}"
  mkdir -p "$LOCAL_WORKDIR/runs" "$LOCAL_WORKDIR/cache"
  if [ -n "$run_id" ]; then
    # run_id đi vào đường dẫn, nên chỉ nhận đúng định dạng (như engine/workdir.py::find_run)
    [[ "$run_id" =~ ^[0-9]{8}-[0-9]{6}-[a-z0-9]+-[0-9a-f]{4}$ ]] || { echo "run_id không hợp lệ"; exit 2; }
    sync_from "$VAST_DIR/runs/" "$LOCAL_WORKDIR/runs/" --include "*/" --include "*/*/$run_id/***" --exclude "*"
  else
    sync_from "$VAST_DIR/runs/" "$LOCAL_WORKDIR/runs/"
  fi
  # cache: chỉ lấy mục máy này chưa có, bỏ thư mục ghi dở
  sync_from "$VAST_DIR/cache/" "$LOCAL_WORKDIR/cache/" --ignore-existing --exclude '*.partial'
}

cmd_exec() { remote "$*"; }

cmd_setup_extra() {
  # MCGS-SLAM (#8) và Instant NuRec (#9) dùng env riêng ngoài conda env drive2sim.
  remote 'nohup bash -c "
    set -e
    bash envs/install_mcgs.sh
    bash envs/install_nurec.sh
    echo SETUP_EXTRA_DONE
  " > setup-extra.log 2>&1 &'
  echo "đang cài ở nền; xong khi setup-extra.log có SETUP_EXTRA_DONE (vast.sh exec tail setup-extra.log)"
}

cmd_wave() {
  # Benchmark tuần 04 (docs/experiments/benchmark_week04.md): queue của setup_benchmark_week04 ghi
  # hàng đợi trên máy thuê (cần cảnh đã ingest ở đó để biết số chunk), rồi bench_queue.sh chạy nền.
  local wave="${1:?cần số đợt 1, 2 hoặc 3}"; shift
  local opts="" gpus=""
  # Không có tham số thì để opts rỗng: printf '%q ' không đối số in ra '' (một đối số rỗng), và
  # setup_benchmark_week04.py từ chối nó ("unrecognized arguments").
  [ $# -gt 0 ] && opts="$(printf '%q ' "$@")"
  [ -n "${BENCH_GPUS:-}" ] && gpus="--gpus $(printf '%q' "$BENCH_GPUS")"
  remote "mkdir -p logs benchmarks/queues &&
    python -m drive2sim.scripts.setup_benchmark_week04 queue --wave $wave $opts --out benchmarks/queues/wave$wave.tsv &&
    nohup bash envs/bench_queue.sh benchmarks/queues/wave$wave.tsv $gpus > logs/wave$wave.out 2>&1 &
    sleep 2; cat logs/wave$wave.out"
  echo "đợt $wave chạy nền; theo dõi: bash envs/vast.sh wave-status $wave"
}

cmd_wave_status() {
  local wave="${1:-}" pattern='logs/queue-wave*'
  [ -n "$wave" ] && pattern="logs/queue-wave$wave"
  remote "for d in $pattern; do [ -d \"\$d\" ] || continue; echo \"== \$d\";
            echo \"xong:\"; cut -f1-4 \"\$d/done\"; echo \"chờ: \$(wc -l < \"\$d/pending\") run\"; done;
          tail -n 3 logs/wave*.out 2>/dev/null; echo ---;
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
}

command="${1:-}"; shift || true
case "$command" in
  check) cmd_check ;;
  setup) cmd_setup ;;
  push-scene) cmd_push_scene "$@" ;;
  run) cmd_run "$@" ;;
  status) cmd_status ;;
  pull) cmd_pull "$@" ;;
  exec) cmd_exec "$@" ;;
  setup-extra) cmd_setup_extra ;;
  wave) cmd_wave "$@" ;;
  wave-status) cmd_wave_status "$@" ;;
  *) sed -n 2,24p "${BASH_SOURCE[0]}"; exit 2 ;;
esac
