# Plan cho Claude Code: chạy benchmark tuần 04 trên vast.ai

Plan này để giao cho Claude Code chạy trên máy của K (có mạng, có `vastai`, `ssh`, khoá SSH). Claude Code đọc hết file này và `AGENTS.md` trước khi làm, rồi làm theo từng giai đoạn; mỗi giai đoạn có điều kiện xong và điểm phải dừng hỏi K.

Tài liệu gốc: `docs/experiments/benchmark_week04.md` (ma trận, kế hoạch 3 đợt, lệnh). Script: `envs/week04_vast.sh` (máy mình), `envs/week04_remote.sh` (máy thuê), `envs/vast.sh` (SSH + rsync, đợt chạy), `envs/bench_queue.sh` (hàng đợi mỗi GPU một worker), `drive2sim/scripts/setup_benchmark_week04.py` (split, hàng đợi, file định nghĩa).

## 0. Quy tắc cho Claude Code

- **Tiền:** máy thuê tính tiền từ lúc `rent` tới lúc `destroy`. Không thuê quá 1 máy; không để máy chạy không quá 1 giờ (báo K khi một đợt xong). Trần chi phí: 25 USD; gần chạm thì dừng hỏi.
- **Hỏi K trước khi:** thuê máy (`rent`, nêu offer, giá/giờ), chọn run thắng giữa các đợt, huỷ máy (`destroy`), sửa code trong `drive2sim/` (theo AGENTS.md mục 2.1), push.
- **Bí mật:** không in, không ghi vào file trong repo, không commit token Kaggle, khoá SSH, file SMPL. `.vast.env` đã trong `.gitignore` (chỉ có host, cổng, id). Token Kaggle chỉ nằm ở `~/.kaggle/access_token`.
- **Dữ liệu có license riêng** (KITTI-360, Waymo, SMPL): chỉ trên máy thuê ở chế độ riêng tư, xoá máy khi xong.
- **Script chưa chạy thật lần nào:** lỗi trong script là bình thường. Sửa trên nhánh `feat/bench-week04`, chạy `ruff` và test liên quan, commit với mô tả lỗi thật đã gặp; không lách lỗi bằng lệnh tay mà không sửa script.
- Theo dõi bước chạy nền bằng `bash envs/week04_vast.sh status` và `bash envs/vast.sh wave-status <N>`, cách nhau 10–20 phút; không poll liên tục.

## 1. Trạng thái lúc giao (26/11/2026)

- Repo `~/Projects/Phase 3/drivestudio_plus`, nhánh `feat/bench-week04` (worktree `drivestudio_plus-week04` cũ đã bỏ). Nhánh **đi trước origin** các commit `df19a83` (hàng đợi 3 đợt), `e93192b` (script dựng máy vast) và commit thêm file này. `AGENTS.md` có thay đổi chưa commit của K, `.claude/`, `docs/superpowers/` chưa track: không đụng, không commit.
- Ma trận ưu tiên 1 (Sheet "Ma trận chạy benchmark tuần 04", tab Ma trận, #1–#14):
  - Đợt 1: #1 OmniRe 3DGS+FastGS W1, #2 PVG W1, #3 DeformableGS W1, #5 OmniRe 3DGUT+FastGS A2, #6 OmniRe 3DGEER+FastGS A2, #8 MCGS-SLAM B2.
  - Đợt 2 (cần thắng A-L giữa #2, #3): #4 thắng A-L tắt LiDAR W1, #7 thắng A-L + 3DGUT A2.
  - Đợt 3 (cần cấu hình thắng A-LB sau #1–#7): #10 A3, #11 6 chunk B2.
  - Ưu tiên 2: #12 (đợt 2, `--p2`), #13 BezierGS (đợt 1, `--p2`, chỉ khi approach `beziergs` qua `approach check`), #14 SplatAD chưa có approach.
  - #9 NuRec chưa vào hàng đợi: chưa chọn cách chia đoạn (cảnh dài 864 khung, tool giới hạn 8 đoạn × 18 khung). Không chạy #9 trừ khi K quyết.
- Ước lượng: ~25 GPU-giờ RTX 4090, ~17 giờ thuê một máy 2× RTX 4090, ~14–17 USD.

## 2. Chuẩn bị trên máy mình (K làm, Claude Code kiểm)

Claude Code kiểm từng mục, mục nào thiếu thì dừng và chỉ K cách làm:

| Mục | Kiểm bằng |
|---|---|
| Đẩy commit | `git status -sb` trong `drivestudio_plus` không còn `ahead`; nếu còn: `git push` (hỏi K) |
| vastai CLI có key | `vastai show user` chạy được |
| Khoá SSH trên vast.ai | `vastai show ssh-keys` có khoá trùng `~/.ssh/id_ed25519.pub` |
| Token Kaggle | `test -s ~/.kaggle/access_token` và `stat -c %a` là 600 (không in nội dung) |
| SMPL | `find ~/Downloads -iname '*neutral*.pkl'` ra đúng một file; đặt `SMPL_PKL` bằng đường dẫn đó |
| Tài khoản Google | khanh.sylg@gmail.com đã nhận điều khoản Waymo (bucket `gs://waymo_open_dataset_scene_flow/train`) |

## 3. Giai đoạn A: thuê và dựng máy (~2 giờ, phần lớn chạy nền)

1. Tìm máy:
   ```bash
   vastai search offers 'gpu_name=RTX_4090 num_gpus=2 verified=true rentable=true reliability>0.98 cpu_ram>=64 disk_space>=200 inet_down>=500 cuda_max_good>=12.1' -o dph_total
   ```
   Lọc tiếp: bỏ máy ở Trung Quốc (khó tải GitHub, Hugging Face, Kaggle); ưu tiên rẻ nhất có độ tin cậy ≥ 0,99. **Dừng: đưa K 2–3 offer (id, nơi, giá/giờ, RAM, độ tin cậy) và chờ K chọn.**
2. `bash envs/week04_vast.sh rent <offer id>` → `.vast.env`. Lỗi SSH: `vastai show instance <id>`, kiểm khoá SSH; không thuê máy thứ hai khi máy đầu chưa huỷ.
3. `bash envs/week04_vast.sh bootstrap` → in 2 GPU RTX 4090, `BOOTSTRAP_DONE`, `vast.sh check` không có cảnh báo Blackwell.
4. `bash envs/week04_vast.sh setup` (nền). Xong khi `status` có `SETUP_DONE` (setup.log) và `EXTRAS_DONE` (setup-extras.log).
5. Song song với bước 4: `SMPL_PKL=<file> bash envs/week04_vast.sh creds`, rồi `bash envs/week04_vast.sh data-kitti` (nền, cần env drive2sim xong mới chạy được `d2s`: nếu log báo không có `d2s`, đợi `SETUP_DONE` rồi chạy lại). Xong khi có `KITTI_DONE`, cảnh dài 864 khung.
6. `bash envs/week04_vast.sh waymo-auth` (tương tác: K mở link, đăng nhập khanh.sylg@gmail.com, dán mã). Sau `EXTRAS_DONE`: `bash envs/week04_vast.sh data-waymo` → `WAYMO_DONE`.
7. `bash envs/week04_vast.sh finish` → split, lidar-check, hàng đợi đợt 1 in đủ 6 dòng; `FINISH_DONE`. Đọc độ đổi hướng của A3 mà `splits` in: dưới 30 độ thì báo K (Sheet tab Scene, A3).
8. `bash envs/week04_vast.sh smoke`: chờ run `--quick` xong (`bash envs/vast.sh status`), kiểm trạng thái `succeeded`, ghi VRAM đỉnh.

Rủi ro đã biết, xử lý nếu gặp:
- venv Waymo: `waymo-open-dataset-tf-2-11-0==1.6.0` có thể xung đột numpy/protobuf với `-e .` → thử bỏ `-e .`, cài `--no-deps` drive2sim rồi các gói ingest; ghi lại cách đã làm vào `envs/week04_remote.sh`.
- `kaggle kernels output` chậm hoặc dừng giữa chừng: chạy lại `data-kitti` (bỏ qua file đã có).
- `frame_ids.txt` của cảnh 200 khung khác md5 đã ghi: **dừng, báo K** (split và số liệu cũ không còn so được).
- `install_main.sh` lỗi build (gsplat-geer, pytorch3d): đọc `setup.log`, kiểm `TORCH_CUDA_ARCH_LIST` có `8.9`.

## 4. Giai đoạn B: ba đợt chạy (~14 giờ)

1. **Đợt 1:** `bash envs/vast.sh wave 1` (thêm `--p2` nếu K muốn #13). `bench_queue.sh` chạy `approach check` mọi dòng trước; lỗi thì sửa rồi chạy lại. Theo dõi `wave-status 1`; xong khi `logs/wave1.out` có `QUEUE_DONE`. Run lỗi: đọc log trong `logs/queue-wave1/`, sửa nếu là lỗi script/cấu hình, chạy lại dòng trong `failed`.
   Sau #1: so thời gian train và VRAM đỉnh thật với ước lượng (3,0 GPU-giờ); lệch quá 50% thì báo K ước lượng lại tổng.
2. **Chọn thắng A-L:**
   ```bash
   bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml'
   ```
   Đưa K bảng nhóm A-L (W1): PSNR, R3 nếu có, thời gian train, theo luật hoà R2 → R3 → thời gian (ngưỡng hoà 0,3 dB). **Dừng: chờ K chọn `pvg` hay `deformablegs`.**
3. **Đợt 2:** `bash envs/vast.sh wave 2 --al-winner <pvg|deformablegs>` (`--p2` thêm #12).
4. **Chọn cấu hình thắng A-LB:** chạy lại `definition` + `d2s benchmark`; đưa K kết quả #1, #5, #6 (và #4, #7 để tham khảo). A3 chỉ train `image_00` (pinhole) nên mặc định đề xuất cấu hình renderer `3dgs` của #1. **Dừng: chờ K chốt chuỗi `--alb-winner`.**
5. **Đợt 3:** `bash envs/vast.sh wave 3 --alb-winner "omnire renderer=3dgs fastgs=true"` (theo chuỗi K chốt).
6. Sau mỗi đợt: `bash envs/vast.sh pull` để có bản sao kết quả trên máy mình.

## 5. Giai đoạn C: thu kết quả và trả máy

1. `bash envs/vast.sh pull`; trên máy mình: `python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml --out benchmarks/week04.json`.
2. Kiểm đủ: mọi run ưu tiên 1 (trừ #9) `succeeded`, có trong `runs/`. **Hỏi K rồi** `bash envs/week04_vast.sh destroy`; xác nhận bằng `vastai show instances` không còn máy.
3. Cập nhật:
   - `docs/experiments/run_plan.md`: thêm các run (run ID, cấu hình, PSNR, SSIM, LPIPS, thời gian train, VRAM đỉnh, GPU RTX 4090 vast.ai).
   - `docs/experiments/benchmark_week04.md`: GPU-giờ thật, chi phí thật, lỗi đã gặp và cách sửa.
   - Sheet (K làm, hoặc Claude Code nếu có connector Sheet): cột Trạng thái, Run ID, GPU-giờ thật.
4. Commit trên `feat/bench-week04` (không commit `runs/`, `cache/`, `data/`, `.vast.env`); **hỏi K trước khi push.**

## 6. Việc ngoài plan này (chỉ ghi nhận, không làm khi chưa được giao)

- #9 NuRec: chọn giữa nâng giới hạn số đoạn (giảm `n_gaussians` mỗi đoạn) và chạy theo 6 chunk của B2.
- Nhóm "chunked" trong `d2s benchmark` để #11 vào bảng xếp hạng (ARCHITECTURE mục 2.17).
- PVG vẫn trong `TIME_VARYING_METHODS` (R3 bị bỏ qua): kiểm xuất nền tĩnh trên checkpoint #2 rồi mới bỏ.
- Merge `feat/studio-fisheye-lab`: K đã dặn chưa merge.
