# Plan đợt 2 benchmark tuần 04 (27/11/2026)

Tiếp theo `docs/experiments/benchmark_week04_claude_code.md` (quy tắc, các bước dựng máy) và `docs/experiments/benchmark_week04.md` (ma trận, lệnh). Claude Code đọc cả ba file và `AGENTS.md` trước khi làm. Quy tắc mục 0 của plan đợt 1 giữ nguyên (trần 25 USD tính cả đợt 1, hỏi K trước khi thuê, chọn run thắng, huỷ máy, sửa `drive2sim/`, push).

## 1. Quyết định của K (27/11)

- **Thắng A-L trên W1: PVG** (`c40a`, 29,79 dB, hơn DeformableGS `8355` 2,85 dB; không hoà).
- **Renderer fisheye: 3DGEER.** #7 (PVG trên A2) chạy với `renderer=3dgeer campose=false` thay cho 3DGUT; ô A2 A-L của `definition` nhận run 3DGEER. Đợt 3 dự kiến `--alb-winner "omnire renderer=3dgeer fastgs=true"`, **chốt lại sau khi có #6 mới** (mục 4).
- **Chạy lại #6 và #8** trong đợt 2, sau hai bản sửa:
  - #6 OmniRe 3DGEER (A2): 3DGEER cắt viền ảnh pinhole có điểm chính lệch tâm; sửa ở `drive2sim/recon/utils/geer_fov.py` (6d217af), không sửa submodule.
  - #8 MCGS-SLAM (B2): MCGS-SLAM hỏng với rig 2 camera; sửa ở `rig_layout` (f9770da), `prep.mcgs` lên 0.1.1.
  - Chi tiết: project doc `claude/benchmark-week04-wave1-review.md`.

## 2. Hàng đợi đợt 2

`python -m drive2sim.scripts.setup_benchmark_week04 queue --wave 2 --al-winner pvg`:

| # | Approach | Cảnh, split | `--set` | Ô | Ghi chú |
|---|---|---|---|---|---|
| 4 | pvg | W1, `every8_5cams` | `train.drivestudio.lidar=false` | W1 A-0 | |
| 6 | omnire | A2, `every8_stereo_fisheye2` | `renderer=3dgeer fastgs=true` | A2 A-LB | chạy lại; `definition` lấy run mới nhất nên `393f` tự bị thay |
| 7 | pvg | A2, `every8_stereo_fisheye2` | `renderer=3dgeer campose=false` | A2 A-L | PVG + 3DGEER chưa chạy thật lần nào |
| 8 | mcgs | B2 (`kitti360_0003_long`), `every8_pinhole2` | | B | chạy lại; bố cục `[image_00, image_01, image_01]` |

#12 (ưu tiên 2, `--p2`): chỉ thêm nếu K muốn.

## 3. Các bước

### 3.1 Trước khi thuê (K làm, Claude Code kiểm)

| Mục | Kiểm bằng |
|---|---|
| Đã push | `git status -sb` không còn `ahead` trên `feat/bench-week04` |
| Máy cũ đã huỷ | `vastai show instances` rỗng; còn máy thì dùng lại (bỏ qua 3.2 bước 1–2, chạy `bootstrap` để rsync code mới) |
| Các mục còn lại | như bảng mục 2 của plan đợt 1 (vastai key, khoá SSH, token Kaggle, SMPL, tài khoản Waymo) |

### 3.2 Máy

**Thuê 1× RTX 4090, không phải 2×.** Ở đợt 1, container vast.ai bị giới hạn khoảng 61 lõi CPU (cgroup), hai run song song bị throttle (OmniRe 0,55 s/bước khi chạy cùng PVG, 0,135 s/bước khi chạy một mình), nên `bench_serial.sh` phải dừng bớt một run và GPU thứ hai gần như rảnh. Một GPU, một worker thì các run chạy lần lượt mà không cần SIGSTOP, và thời gian train ghi trong log là thời gian thật (R8).

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true reliability>0.98 cpu_ram>=64 cpu_cores_effective>=16 disk_space>=200 inet_down>=500 cuda_max_good>=12.1' -o dph_total
```
Bỏ máy ở Trung Quốc. **Dừng: đưa K 2–3 offer và chờ chọn.** Sau đó làm như mục 3 của plan đợt 1, bước 2–8 (`rent`, `bootstrap`, `setup`, `creds`, `data-kitti`, `waymo-auth`, `data-waymo`, `finish`, `smoke`). Máy mới thì phải ingest lại cả ba cảnh (W1 cho #4, A2 cho #6, #7, cảnh dài cho #8).

### 3.3 Chạy thử trước đợt

PVG + 3DGEER chưa chạy thật, nên chạy một run `--quick` trước:
```bash
bash envs/vast.sh run pvg --scene kitti360_2013_05_28_drive_0003_sync --split every8_stereo_fisheye2 --set renderer=3dgeer --set campose=false --quick
```
Xong khi `status: succeeded`. Mở `train.drivestudio/images/step_*_rgbs.png`: dải trên cùng và cột phải của ảnh pinhole (hai ô 188×704 ở hàng đầu) phải có nội dung như ảnh thật, không đen hay nhiễu. Đây cũng là kiểm thật đầu tiên của bản sửa 3DGEER (test của nó không cần GPU). Lỗi thì đọc log, sửa, ghi lỗi thật vào commit.

### 3.4 Đợt 2

```bash
BENCH_GPUS=0 bash envs/vast.sh wave 2 --al-winner pvg
bash envs/vast.sh wave-status 2        # cách nhau 10–20 phút; xong khi logs/wave2.out có QUEUE_DONE
```
`approach check` chạy mọi dòng trước khi train. Không chạy `bench_serial.sh` (chỉ có một worker). Run lỗi: đọc `logs/queue-wave2/`, sửa nếu là lỗi script hay cấu hình, chạy lại dòng trong `failed`.

Ước lượng: #6 0,7 GPU-giờ (đo ở đợt 1); #4, #7 mỗi run khoảng 1–2 giờ (PVG chạy một mình 0,08 s/bước, tức khoảng 40 phút cho 30.000 bước, cộng init, eval); #8 chưa có số (MCGS giờ chạy 2 camera + 1 camera thêm, nặng hơn). Tổng khoảng 4–5 GPU-giờ, cộng khoảng 2 giờ dựng máy nếu thuê mới: khoảng 7 giờ thuê, khoảng 3–4 USD ở 0,5 USD/giờ.

### 3.5 Kiểm kết quả từng run

| # | Đạt khi | Nếu không đạt |
|---|---|---|
| 6 | `image_00` không còn kém #5 (`f4be`, 25,04 dB) quá ngưỡng hoà; ảnh render không còn dải hỏng ở trên và bên phải | báo K; kiểm `geer_fov.install()` có chạy (log trainer, `gsplat.geer._preprocess.get_camera_tanfov` là bản bọc) |
| 6 | Ghi riêng PSNR fisheye (`image_02`, `image_03`) so với #5: đợt 1 3DGEER kém 1,6 dB ở `image_02` | chỉ ghi nhận; ảnh hưởng chọn renderer (mục 4) |
| 6 | R3 nhóm pinhole: so `ply_gap.json` với #5 (đợt 1: #5 gap 8,26 dB, chưa rõ lý do) | ghi nhận, không sửa trong đợt này |
| 7 | `succeeded`, có `shifted` (R2 trên `image_01`) | lỗi renderer: báo K trước khi đổi sang 3DGUT |
| 8 | `succeeded`; `mcgs_prep.json` có `layout` 3 phần tử; có quỹ đạo, PLY; ATE ghi trong kết quả | lỗi mới của MCGS: dừng, báo K kèm traceback |
| 4 | `succeeded`; so với #2 chỉ khác `lidar=false` | |

Thời gian train của #1–#3 ở đợt 1 có lẫn thời gian bị SIGSTOP của `bench_serial.sh` (log `logs/serial.log` trên máy cũ, chưa kéo về). Khi so R8 với đợt 2, chỉ so các run cùng điều kiện, hoặc trừ khoảng dừng nếu còn log.

## 4. Sau đợt 2: chọn cấu hình A-LB cho đợt 3

```bash
bash envs/vast.sh pull
bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml'
```
Đưa K bảng A2 A-LB (#5 3DGUT so với #6 3DGEER mới: PSNR theo camera, R2 sau chỉnh màu, R3, thời gian train) và A2 A-L (#7). K đã chọn 3DGEER; **dừng, xác nhận lại với K** nếu #6 mới vẫn thua #5 quá ngưỡng hoà. Sau khi chốt:
```bash
BENCH_GPUS=0 bash envs/vast.sh wave 3 --alb-winner "omnire renderer=3dgeer fastgs=true"
```
(Đợt 3 có #10 A3 và 6 chunk của #11, khoảng 6 GPU-giờ; A3 chỉ train `image_00`, nên 3DGEER ở đây đi qua bản sửa pinhole.)

Không còn đợt nào cần máy thì: `pull`, **hỏi K rồi** `bash envs/week04_vast.sh destroy`, xác nhận `vastai show instances` rỗng. Cập nhật `run_plan.md`, `benchmark_week04.md` (GPU-giờ, chi phí thật, lỗi gặp) và Sheet như mục 5 của plan đợt 1.
