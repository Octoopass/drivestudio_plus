# Plan cho Claude Code: đợt 2 benchmark tuần 04 trên vast.ai

Plan này để giao cho Claude Code chạy trên máy của K (có mạng, `vastai`, `ssh`, khoá SSH). Claude Code đọc hết file này và `AGENTS.md` trước khi làm, rồi làm theo từng giai đoạn; mỗi giai đoạn có điều kiện xong và điểm phải dừng hỏi K. Plan đợt 1 (`docs/experiments/benchmark_week04_claude_code.md`) chỉ dùng để tra chi tiết; khi hai file khác nhau, file này đúng.

Tài liệu liên quan: `docs/experiments/benchmark_week04.md` (ma trận, split, lệnh), project doc `claude/benchmark-week04-wave1-review.md` (kết quả và lỗi đợt 1). Script: `envs/week04_vast.sh` (máy mình), `envs/week04_remote.sh` (máy thuê), `envs/vast.sh` (SSH, rsync, đợt chạy), `envs/bench_queue.sh` (hàng đợi), `drive2sim/scripts/setup_benchmark_week04.py` (split, hàng đợi, file định nghĩa).

## 0. Quy tắc

- **Tiền:** máy tính tiền từ `rent` tới `destroy`. Chỉ thuê 1 máy; không để máy chạy không quá 1 giờ (báo K khi đợt xong). Trần cả tuần 04 là 25 USD: hỏi K đợt 1 đã tốn bao nhiêu trước khi thuê, ghi trần còn lại; gần chạm thì dừng hỏi.
- **Hỏi K trước khi:** thuê máy (nêu offer, giá/giờ), chọn cấu hình thắng A-LB cho đợt 3, huỷ máy, sửa code trong `drive2sim/` (AGENTS.md mục 2.1), push.
- **Bí mật:** không in, không ghi vào repo, không commit token Kaggle, khoá SSH, file SMPL. `.vast.env` đã trong `.gitignore`. Token Kaggle chỉ ở `~/.kaggle/access_token`.
- **Dữ liệu có license riêng** (KITTI-360, Waymo, SMPL): chỉ trên máy thuê ở chế độ riêng tư, xoá máy khi xong.
- **Lỗi script:** sửa trên nhánh `feat/bench-week04`, chạy `ruff` và test liên quan, commit kèm mô tả lỗi thật đã gặp; không lách bằng lệnh tay mà không sửa script. Không commit `runs/`, `cache/`, `data/`, `.vast.env`; không đụng `AGENTS.md` (thay đổi chưa commit của K), `.claude/`, `docs/superpowers/`.
- **Theo dõi bước nền** bằng `bash envs/week04_vast.sh status` và `bash envs/vast.sh wave-status 2`, cách nhau 10–20 phút; không poll liên tục.

## 1. Trạng thái lúc giao (27/11/2026)

- Repo `~/Projects/Phase 3/drivestudio_plus`, nhánh `feat/bench-week04` @ `28b1f07`, đã push.
- Đợt 1 xong: #1 `2e85`, #2 `c40a`, #3 `8355`, #5 `f4be`, #6 `393f` thành công; #8 MCGS hỏng. Kết quả đã kéo về `runs/`. Máy đợt 1 có lẽ đã huỷ (`.vast.env` không còn), chưa xác nhận.
- Quyết định của K:
  - **Thắng A-L trên W1: PVG** (29,79 dB, hơn DeformableGS 2,85 dB).
  - **Renderer fisheye: 3DGEER.** #7 chạy `renderer=3dgeer campose=false`; đợt 3 dự kiến `omnire renderer=3dgeer fastgs=true`, chốt sau khi có #6 mới.
  - **Chạy lại #6 và #8** sau hai bản sửa:
    - #6: 3DGEER không vẽ viền trên và phải của ảnh pinhole KITTI-360 (điểm chính lệch tâm). Sửa ở `drive2sim/recon/utils/geer_fov.py` (`6d217af`), không sửa submodule gsplat-geer. Chưa chạy thật trên GPU.
    - #8: MCGS-SLAM hỏng với đúng 2 thư mục ảnh (`FactorGraph.update(use_scaling=...)`). Sửa ở `rig_layout`: bố cục `[image_00, image_01, image_01]`, `prep.mcgs` lên 0.1.1 (`f9770da`).
- Hàng đợi đợt 2 (`setup_benchmark_week04 queue --wave 2 --al-winner pvg`):

| # | Approach | Cảnh, split | `--set` | Ô |
|---|---|---|---|---|
| 4 | pvg | W1, `every8_5cams` | `train.drivestudio.lidar=false` | W1 A-0 |
| 6 | omnire | A2 (`kitti360_2013_05_28_drive_0003_sync`), `every8_stereo_fisheye2` | `renderer=3dgeer fastgs=true` | A2 A-LB (chạy lại; `definition` lấy run mới nhất nên `393f` tự bị thay) |
| 7 | pvg | A2, `every8_stereo_fisheye2` | `renderer=3dgeer campose=false` | A2 A-L (PVG + 3DGEER chưa chạy thật) |
| 8 | mcgs | B2 (`kitti360_0003_long`), `every8_pinhole2` | | B (chạy lại) |

  #12 (ưu tiên 2, `--p2`) chỉ thêm khi K muốn. #9 NuRec vẫn ngoài plan.
- **Chạy một GPU, một worker.** Ở đợt 1 container vast.ai bị giới hạn khoảng 61 lõi CPU; hai run song song bị throttle (OmniRe 0,55 s/bước khi chạy cùng PVG, 0,135 s/bước khi chạy một mình), nên `bench_serial.sh` phải tạm dừng (SIGSTOP) bớt một run, GPU thứ hai gần như rảnh, và thời gian bị dừng lẫn vào thời gian train của #1–#3. Một worker thì các run chạy lần lượt, thời gian train trong log là thời gian thật (R8). Không chạy `bench_serial.sh` ở đợt này.
- Ước lượng: #6 0,7 GPU-giờ (đo ở đợt 1); #4, #7 mỗi run 1–2 giờ (PVG một mình 0,08 s/bước, khoảng 40 phút cho 30.000 bước, cộng init và eval); #8 chưa có số (giờ chạy thêm một camera). Tổng khoảng 4–5 GPU-giờ, cộng khoảng 2 giờ dựng máy mới: khoảng 7 giờ thuê, khoảng 3–4 USD ở 0,5 USD/giờ.

## 2. Chuẩn bị trên máy mình (K làm, Claude Code kiểm)

Mục nào thiếu thì dừng và chỉ K cách làm:

| Mục | Kiểm bằng |
|---|---|
| Code mới nhất | `git status -sb` trên `feat/bench-week04` không có `ahead`/`behind`; `git log -1` là `28b1f07` hoặc mới hơn |
| Máy cũ | `vastai show instances`: rỗng thì thuê mới (mục 3); còn máy đợt 1 thì báo K và hỏi dùng lại hay huỷ (dùng lại: đặt `VAST_HOST`, `VAST_PORT` hoặc ghi lại `.vast.env`, bỏ qua bước 3.1–3.2) |
| vastai CLI có key | `vastai show user` chạy được |
| Khoá SSH trên vast.ai | `vastai show ssh-keys` có khoá trùng `~/.ssh/id_ed25519.pub` |
| Token Kaggle | `test -s ~/.kaggle/access_token` và `stat -c %a` là 600 (không in nội dung) |
| SMPL | `find ~/Downloads -iname '*neutral*.pkl'` ra đúng một file; đặt `SMPL_PKL` |
| Tài khoản Google | khanh.sylg@gmail.com đã nhận điều khoản Waymo (cần cho W1, #4) |
| Chi phí đợt 1 | hỏi K; ghi trần còn lại = 25 USD − chi phí đợt 1 |

## 3. Giai đoạn A: thuê và dựng máy (~2 giờ, phần lớn chạy nền)

1. Tìm máy **1 GPU**:
   ```bash
   vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true reliability>0.98 cpu_ram>=64 cpu_cores_effective>=16 disk_space>=200 inet_down>=500 cuda_max_good>=12.1' -o dph_total
   ```
   Bỏ máy ở Trung Quốc; ưu tiên rẻ nhất có độ tin cậy ≥ 0,99. **Dừng: đưa K 2–3 offer (id, nơi, giá/giờ, RAM, số lõi, độ tin cậy) và chờ K chọn.**
2. `bash envs/week04_vast.sh rent <offer id>` → `.vast.env`. Lỗi SSH: `vastai show instance <id>`, kiểm khoá; không thuê máy thứ hai khi máy đầu chưa huỷ.
3. `bash envs/week04_vast.sh bootstrap` → in 1 GPU RTX 4090, `BOOTSTRAP_DONE`. (Bước này rsync code của máy mình, nên phải chạy sau khi đã ở `28b1f07`.)
4. `bash envs/week04_vast.sh setup` (nền). Xong khi `status` có `SETUP_DONE` và `EXTRAS_DONE`. MCGS-SLAM cài trong bước extras (`install_mcgs.sh`, có `libusb` từ `79a3b07`).
5. Song song với bước 4: `SMPL_PKL=<file> bash envs/week04_vast.sh creds`, rồi `bash envs/week04_vast.sh data-kitti` (nền; nếu log báo không có `d2s` thì đợi `SETUP_DONE` rồi chạy lại). Xong khi có `KITTI_DONE`, cảnh dài 864 khung. `frame_ids.txt` của cảnh 200 khung khác md5 đã ghi: **dừng, báo K.**
6. `bash envs/week04_vast.sh waymo-auth` (tương tác: K mở link, đăng nhập, dán mã). Sau `EXTRAS_DONE`: `bash envs/week04_vast.sh data-waymo` → `WAYMO_DONE`.
7. `bash envs/week04_vast.sh finish` → split, lidar-check, `FINISH_DONE`. (Lệnh này in hàng đợi đợt 1; bỏ qua, không chạy đợt 1.)
8. **Chạy thử thay cho `smoke`:** PVG + 3DGEER chưa chạy thật, và đây là lần kiểm thật đầu tiên của bản sửa 3DGEER:
   ```bash
   bash envs/vast.sh run pvg --scene kitti360_2013_05_28_drive_0003_sync --split every8_stereo_fisheye2 --set renderer=3dgeer --set campose=false --quick
   bash envs/vast.sh status
   ```
   Xong khi run `succeeded`. Kéo run về (`bash envs/vast.sh pull <run_id>`) và mở ảnh `train.drivestudio/images/step_<bước cuối>_rgbs.png`: hàng trên cùng là hai ảnh pinhole 188×704; dải khoảng 25 hàng trên cùng và 18 cột bên phải của chúng phải có nội dung như `step_<bước cuối>_gt_rgbs.png`, không đen hay nhiễu. Còn dải hỏng: **dừng, báo K** (kiểm trong log trainer xem `geer_fov.install()` có chạy không).

Rủi ro đã biết (từ đợt 1, đã sửa trong script; gặp lại thì đọc log, sửa, commit):
- thiếu thư viện hệ thống (EGL cho `prep.smpl`, `libusb` cho open3d của MCGS): thêm vào `bootstrap`;
- `kaggle kernels output` chậm hoặc dừng giữa chừng: chạy lại `data-kitti` (bỏ qua file đã có);
- `install_main.sh` lỗi build (gsplat-geer, pytorch3d): đọc `setup.log`, kiểm `TORCH_CUDA_ARCH_LIST` có `8.9`.

## 4. Giai đoạn B: đợt 2 (~4–5 giờ)

1. Chạy:
   ```bash
   BENCH_GPUS=0 bash envs/vast.sh wave 2 --al-winner pvg
   ```
   Hàng đợi in ra phải đúng 4 dòng như bảng mục 1 (#6 approach `omnire`, #8 `mcgs`; nếu #6 hay #8 hiện `pvg` là script cũ: dừng, kiểm code đã rsync). `bench_queue.sh` chạy `approach check` mọi dòng trước; lỗi thì sửa rồi chạy lại.
2. Theo dõi `bash envs/vast.sh wave-status 2`; xong khi `logs/wave2.out` có `QUEUE_DONE`. Run lỗi: đọc log trong `logs/queue-wave2/`, sửa nếu là lỗi script hay cấu hình, chạy lại các dòng trong `failed`:
   ```bash
   bash envs/vast.sh exec 'nohup bash envs/bench_queue.sh logs/queue-wave2/failed --gpus 0 > logs/retry2.out 2>&1 &'
   ```
   Lỗi mới của MCGS-SLAM (#8) hoặc của PVG + 3DGEER (#7) mà không phải lỗi môi trường: **dừng, báo K kèm traceback**; không tự đổi #7 sang 3DGUT.
3. Sau run đầu tiên xong: so thời gian train, VRAM đỉnh với ước lượng mục 1; tổng dự kiến vượt trần còn lại thì báo K.
4. Kiểm từng run (đọc `eval.standard/summary.json`, `eval.ply/ply_gap.json`, `train.drivestudio/train_stats.json`):

| # | Đạt khi | Nếu không đạt |
|---|---|---|
| 6 | `per_camera.image_00.psnr` không kém #5 (`f4be`, 25,04 dB) quá 0,3 dB; ảnh render không còn dải hỏng trên và phải | báo K (ảnh hưởng chọn renderer, mục 5) |
| 6 | ghi PSNR `image_02`, `image_03` so với #5 (đợt 1: 3DGEER kém 1,6 dB ở `image_02`) | chỉ ghi nhận |
| 6 | ghi R3 nhóm pinhole từ `ply_gap.json`, so với #5 (đợt 1: #5 gap 8,26 dB, chưa rõ lý do) | chỉ ghi nhận, không sửa trong đợt này |
| 7 | `succeeded`; `summary.json` có `shifted` (R2 trên `image_01`) | báo K |
| 8 | `succeeded`; `mcgs_prep.json` có `layout` 3 phần tử `[image_00, image_01, image_01]`; có quỹ đạo và PLY; có ATE | báo K kèm traceback |
| 4 | `succeeded`; `run.json` có `lidar=false` ở `train.drivestudio` | báo K |

5. `bash envs/vast.sh pull` để có bản sao kết quả trên máy mình.

## 5. Chọn cấu hình A-LB cho đợt 3

```bash
bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml'
```
Đưa K hai bảng:
- **A2 A-LB:** #5 (3DGUT) so với #6 mới (3DGEER): PSNR, SSIM, LPIPS theo camera; R2 (`shifted`, bản sau chỉnh màu); R3 (`ply_psnr`); thời gian train. Ghi rõ thời gian train của #5 lấy ở đợt 1 (có thể lẫn thời gian bị dừng).
- **A2 A-L:** #7; **W1 A-0:** #4 so với #2 (tham khảo).

**Dừng: chờ K chốt chuỗi `--alb-winner`.** K đã chọn 3DGEER; nếu #6 mới vẫn thua #5 quá ngưỡng hoà, nói rõ khi hỏi. Mặc định đề xuất `omnire renderer=3dgeer fastgs=true`.

## 6. Đợt 3 (khi K chốt)

```bash
BENCH_GPUS=0 bash envs/vast.sh wave 3 --alb-winner "omnire renderer=3dgeer fastgs=true"
```
#10 (A3) và 6 chunk của #11 (B2), khoảng 6 GPU-giờ. A3 chỉ train `image_00`, nên 3DGEER ở đây đi qua bản sửa pinhole. Theo dõi, sửa lỗi, kiểm `succeeded` như mục 4; `pull` khi xong.

## 7. Giai đoạn C: thu kết quả và trả máy

1. `bash envs/vast.sh pull`; trên máy mình:
   ```bash
   python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml --out benchmarks/week04.json
   ```
2. Kiểm đủ: #4, #6, #7, #8 (và #10, #11 nếu đã chạy đợt 3) `succeeded`, có trong `runs/`. **Hỏi K rồi** `bash envs/week04_vast.sh destroy`; xác nhận `vastai show instances` rỗng.
3. Cập nhật:
   - `docs/experiments/run_plan.md`: thêm các run (run ID, cấu hình, PSNR, SSIM, LPIPS, thời gian train, VRAM đỉnh, GPU RTX 4090 vast.ai).
   - `docs/experiments/benchmark_week04.md`: GPU-giờ thật, chi phí thật, lỗi đã gặp và cách sửa (gồm hai lỗi của đợt 1 đã sửa: MCGS 2 camera, 3DGEER điểm chính lệch tâm).
   - Project doc `claude/benchmark-week04-wave1-review.md` hoặc doc mới cho đợt 2: kết quả #6 mới so với #5, kết luận về renderer.
   - Sheet (K làm, hoặc Claude Code nếu có connector Sheet): cột Trạng thái, Run ID, GPU-giờ thật.
4. Commit trên `feat/bench-week04`; **hỏi K trước khi push.**

## 8. Ngoài plan (chỉ ghi nhận)

- R3 pinhole của #5 (gap 8,26 dB) và gap khoảng 5 dB trên W1 cho mọi method (`.ply` không mang `Affine` theo camera): cần điều tra riêng trước khi dùng R3 phân xử A2.
- #9 NuRec: chọn cách chia đoạn (nâng giới hạn số đoạn hoặc chạy theo 6 chunk của B2).
- Nhóm "chunked" trong `d2s benchmark` để #11 vào bảng xếp hạng.
- PVG vẫn trong `TIME_VARYING_METHODS`: kiểm xuất nền tĩnh trên checkpoint #2 rồi mới bỏ.
- Merge `feat/studio-fisheye-lab`: K đã dặn chưa merge.
