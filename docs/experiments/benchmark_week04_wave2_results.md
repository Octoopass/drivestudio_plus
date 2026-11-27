# Benchmark tuần 04, đợt 2: kết quả (27/11/2026)

Đợt 2 chạy trên một máy vast.ai 1× RTX 4090 (offer `62286745`, California), một worker, mỗi lúc một run, nên thời gian train trong log là thời gian thật (R8). Plan: `benchmark_week04_wave2.md`. Số liệu lấy từ `runs/<cảnh>/<approach>/<run_id>/d2s/stages/` (`eval.standard/summary.json`, `eval.ply/ply_gap.json`). Mọi run chạy một lần, một seed: chênh dưới khoảng 0,3 dB (ngưỡng hoà của benchmark) chưa đủ để kết luận.

## 1. Trạng thái các run

| # | Ô | Run | Trạng thái | PSNR | SSIM | LPIPS | Thời gian train | VRAM đỉnh |
|---|---|---|---|---|---|---|---|---|
| 4 | W1 A-0 (PVG, `lidar=false`) | `20261127-075005-pvg-76e9` | thành công (đủ 30.000 bước lần 2, sau bản sửa) | 29,02 | 0,909 | 0,131 | 1172 s | 2,4 GB |
| 6 | A2 A-LB (OmniRe 3DGEER + FastGS) | `20261127-045255-omnire-7a64` | thành công | 22,61 | 0,793 | 0,165 | 1639 s | 3,3 GB |
| 7 | A2 A-L (PVG 3DGEER, `campose=false`) | `20261127-033252-pvg-fa62` | thành công, kết quả thấp | 19,24 | 0,703 | 0,303 | 4250 s | 6,9 GB |
| 8 | B2 (MCGS-SLAM, 864 khung) | `20261127-072153-mcgs-3902` | thành công (lần chạy thứ 4, sau hai bản sửa), kết quả rất thấp | 10,25 | 0,388 | 0,638 | 1539 s | không đo (`peak_vram_mb` 176 là của bước eval) |

Run hỏng và run chẩn đoán (không vào bảng xếp hạng): #4 `7d86` (SIGFPE, trong hàng đợi), `8dbb` (`--quick`, `faulthandler`, hỏng), `97ae` (`--quick`, bản vá thử, chạy được); #8 `2578` (hỏng ở global BA, trong hàng đợi), `6cf6` (chạy lại với `CUDA_LAUNCH_BLOCKING=1`, hỏng ở cùng chỗ, cho ra dòng lỗi chính xác), `7d1e` (`--buffer` lớn, hỏng ở `process_global_track_data`); `669d` (`--quick`, kiểm viền 3DGEER). Tất cả nằm trên Drive cùng các run thành công.

## 2. Hai bản sửa đợt trước, kiểm trên GPU lần đầu

- **3DGEER điểm chính lệch tâm (`6d217af`): đạt.** Ở #6, dải 25 hàng trên và 18 cột phải của hai ảnh pinhole 188×704 có nội dung: tỉ lệ pixel đen 0,0; sai số tuyệt đối trung bình so với ảnh gốc ở dải trên 5,0–5,2 so với 4,4–5,6 ở phần còn lại. `image_00` 25,37 dB, hơn #5 (25,04 dB).
- **MCGS với 2 thư mục ảnh (`f9770da`): đạt một phần.** Bố cục `[image_00, image_01, image_01]` chạy qua `prep.mcgs`, tracking và bước tinh chỉnh toàn cục (mục 4 liệt kê ba lỗi khác phải sửa mới chạy hết).

## 3. A2 A-LB: #5 (3DGUT) so với #6 (3DGEER), OmniRe + FastGS

| | #5 `f4be` (3DGUT, đợt 1) | #6 `7a64` (3DGEER, mới) |
|---|---|---|
| PSNR / SSIM / LPIPS tổng | 23,17 / 0,792 / 0,191 | 22,61 / 0,793 / 0,165 |
| `image_00` (pinhole) PSNR / SSIM / LPIPS | 25,04 / 0,845 / 0,157 | **25,37** / 0,863 / 0,113 |
| `image_01` (pinhole) | 23,77 / 0,823 / 0,167 | 23,65 / 0,838 / 0,128 |
| `image_02` (mắt cá) | **23,51** / 0,786 / 0,175 | 21,73 / 0,766 / 0,179 |
| `image_03` (mắt cá) | **20,96** / 0,745 / 0,241 | 20,72 / 0,749 / 0,202 |
| R2 `shifted` (`image_01`, camera chỉ test) PSNR / SSIM / LPIPS | 23,77 / 0,823 / 0,167 | 23,65 / 0,838 / 0,128 |
| R2 sau chỉnh màu PSNR / LPIPS | 24,65 / 0,168 | 24,59 / 0,130 |
| R3 `ply_psnr` tổng (gap) | 18,90 (4,24) | 18,95 (4,21) |
| R3 nhóm pinhole (gap) | 17,66 (8,26) | 19,53 (6,91) |
| R3 nhóm mắt cá (gap) | 19,52 (2,22) | 18,65 (2,86) |
| Gaussian | 284k | 256k |
| Thời gian train | 1851 s (đợt 1, có thể lẫn thời gian bị dừng) | 1639 s |
| VRAM đỉnh | 3,0 GB | 3,3 GB |

Đọc kết quả:
- Trên **pinhole** 3DGEER bằng hoặc hơn 3DGUT (`image_00` +0,33 dB, LPIPS tốt hơn 0,03–0,04 ở cả hai camera; `image_01` kém 0,12 dB, trong ngưỡng hoà). Chênh LPIPS lớn so với PSNR cho thấy 3DGEER vẽ chi tiết tốt hơn mà PSNR chưa phản ánh.
- Trên **mắt cá** 3DGEER kém ở `image_02` 1,78 dB (đợt 1 cũng kém 1,6 dB), `image_03` kém 0,24 dB (trong ngưỡng hoà). PSNR tổng 22,61 so với 23,17 vì hai camera mắt cá chiếm nửa số ảnh test.
- **Tiêu chí plan:** `image_00` của #6 không kém #5 quá 0,3 dB (hơn 0,33 dB); không còn dải hỏng. Đạt.
- R3 pinhole của #5 vẫn gap lớn (8,26 dB) và #6 vẫn 6,91 dB: chưa rõ lý do (`.ply` không mang `Affine` theo camera); không sửa trong đợt này, không dùng R3 pinhole để phân xử A2.
- Thời gian train #6 ngắn hơn #5 11% nhưng #5 lấy từ đợt 1 chạy hai run song song nên không so được.

**Đề xuất cho đợt 3:** giữ `omnire renderer=3dgeer fastgs=true` như K đã chọn, kèm lưu ý: A3 chỉ train `image_00`, nơi 3DGEER ngang hoặc hơn 3DGUT nhờ bản sửa pinhole; chênh ở mắt cá không ảnh hưởng A3. **K chốt** chuỗi `--alb-winner`.

## 4. A2 A-L và W1 A-0

| | Run | PSNR / SSIM / LPIPS |
|---|---|---|
| A2 A-L, PVG + 3DGEER (#7) | `fa62` | 19,24 / 0,703 / 0,303 |
| A2 A-LB, OmniRe + 3DGEER (#6) | `7a64` | 22,61 / 0,793 / 0,165 |
| W1 A-L, PVG có LiDAR (#2, đợt 1) | `c40a` | 29,79 / 0,911 / 0,126 |
| W1 A-0, PVG không LiDAR (#4) | `76e9` | 29,02 / 0,909 / 0,131 |

- **#7 thấp bất thường:** 19,24 dB, kém #6 3,4 dB, dùng gấp 10 lần Gaussian (2,73 triệu) và VRAM gấp đôi; R3 `ply_psnr` 11,89 dB với gap 9,53 dB. PSNR train in ra trong log chỉ 8–9 dB ở nhiều bước. Run chạy xong không lỗi nên plan không bắt dừng, nhưng nên coi PVG + 3DGEER trên KITTI-360 chưa dùng được cho so sánh (chưa điều tra nguyên nhân; ứng viên: PVG chia thời gian cho nền tĩnh, `campose=false`, hoặc tương tác với renderer 3DGEER trên mắt cá). `shifted` (R2) có: 20,13 dB trên `image_01`.
- **#4:** `run.json` có `lidar=false` ở `train.drivestudio` và `lidar_usage` rỗng. PSNR kém #2 0,77 dB: bỏ LiDAR chỉ mất ít trên W1 với PVG. Thời gian train 1172 s và #2 là 8279 s không so được (#2 chạy chồng nhiều run).

## 5. Ba lỗi phải sửa mới chạy hết #4 và #8

| Lỗi | Nguyên nhân gốc (bằng chứng) | Sửa |
|---|---|---|
| #4 SIGFPE ở bước train đầu | `lidar=false`: mọi điểm nền là điểm ngẫu nhiên có thời gian 0; PVG chỉ giữ Gaussian có `marginal_t > 0.05` quanh khung hiện tại (`pvg.py:390`), nên ở khung xa 0 không còn Gaussian nào (in `gs.means.shape` → N = 0 ở bước train đầu, N = 95.000 ở eval bước 0) và gsplat chia cho 0 ở phía host (`faulthandler`: `fully_fused_projection`) | `recon/trainers/single.py`: điểm ngẫu nhiên có thời gian ngẫu nhiên đều trong [0, 1] khi không có LiDAR (có LiDAR giữ nguyên, thời gian 0). Kiểm: `--quick` 3000 bước PSNR train 26,6 dB; run đủ 30.000 bước 29,02 dB |
| #8 `index out of bounds` ở global BA | `demo.py --buffer` mặc định 200 (bộ đệm video 215); 757 keyframe nên bộ đệm bị đẩy (`release_buffer`), rồi global BA đánh chỉ số mọi keyframe vào bộ đệm video (`video.nets[ii, ...]`, chạy lại với `CUDA_LAUNCH_BLOCKING=1` cho đúng dòng `factor_graph.py:96`) | `stages/train/mcgs.py` 0.1.1: `--buffer` = số khung + 1. VRAM lúc tracking khoảng 9,4 GB |
| #8 `AssertionError` ở `process_global_track_data` | MCGS-SLAM đặt mốc thời gian camera `i` là `tstamp + 500*i` (`mcgs_slam/mcgs.py:93,133`); chuỗi 864 khung có mốc của hai camera trùng (khung 600 camera 0 = khung 100 camera 1), nên khớp gaussian với khung ra nhiều hơn số gaussian | `envs/install_mcgs.sh` sửa `500` → `100000` sau khi checkout (idempotent) |

## 6. #8: chạy xong nhưng kết quả rất thấp

PSNR 10,25 dB, SSIM 0,388, LPIPS 0,638, `coverage` 0,79; quỹ đạo lệch ATE 21,10 m, tỉ lệ 0,909 trên cảnh dài 1289 m; PSNR từng khung 9–13 dB. Ảnh render mờ nhoè dọc theo hướng đi, tức bản đồ lệch pose: tracking của MCGS-SLAM trôi trên chuỗi dài KITTI-360 (và bố cục 3 thư mục là mẹo cho nhánh nhiều camera, không phải cấu hình gốc của KITTI). Kết quả là số thật của cấu hình này, không phải lỗi script; chưa điều tra nguyên nhân (ứng viên: trôi quỹ đạo vì không có loop closure, `--stride 1`, `baseline` stereo, độ sâu).

`mcgs_prep.json` có `layout` `[image_00, image_01, image_01]`; có quỹ đạo, PLY, ATE.

## 7. Chi phí và vận hành

- Máy thuê khoảng 6,5 giờ tính tới lúc #4 xong (chưa huỷ), giá thật 0,575 USD/giờ (offer ghi 0,43 USD/giờ chưa tính phí lưu trữ và mạng); credit từ 4,64 USD còn 0,58 USD khi #4 xong, tức đã tiêu khoảng 4,1 USD. Ước lượng của plan (3–4 USD) thấp hơn do ba run chẩn đoán và hai lần chạy lại #8.
- GPU-giờ (ước lượng từ thời gian run): #7 1,2; #6 0,6; #4 0,35 ở lần cuối; #8 0,45 ở lần cuối; phần còn lại (ba run #8 hỏng hoặc chẩn đoán, các run #4 hỏng và `--quick`) khoảng 1,8.
- **Đường truyền về máy mình chỉ khoảng 300 KB/s**: một run 1,6 GB mất hơn 1,5 giờ. Kết quả được up thẳng từ máy thuê lên Google Drive (`rclone`, scope `drive.file`) vào `d2s-bench-week04/runs/`; `envs/vast.sh pull` thêm `D2S_PULL_LIGHT=1` bỏ `*.pth` và `scene.ply` (hơn 1 GB mỗi run), và `envs/vast_autopull.sh` kéo bản nhẹ của run vừa xong.
- `week04_vast.sh waymo-auth` không chạy qua `!` hoặc nơi không có terminal (`ssh -t` cần tty); cách làm được: chạy `gcloud auth login --no-launch-browser` nền trên máy thuê đọc mã từ một FIFO, người dùng mở link rồi gửi mã.
- Test: `tests/drive2sim/recon/trainers/test_single_init_times.py` (2 test, chạy trên GPU env), `test_build_command_buffer_holds_every_frame` (thêm vào `test_mcgs_stages.py`). 9 test khác đang hỏng sẵn (colab, rig, gpu_viewer, test_box_free_approaches) không do thay đổi này.

## 8. Việc còn lại (ngoài plan)

- Điều tra #7 (PVG + 3DGEER kém bất thường) và #8 (ATE 21 m) trước khi dùng vào kết luận.
- R3 pinhole của #5, #6 (gap 7–8 dB) và gap khoảng 5 dB trên W1.
- #9 NuRec, nhóm "chunked" trong `d2s benchmark`, PVG trong `TIME_VARYING_METHODS`.
- Đợt 3 (#10, 6 chunk #11): chờ K chốt `--alb-winner`; chưa chạy, và credit hiện còn 0,58 USD nên cần nạp thêm (ước tính khoảng 6 GPU-giờ, khoảng 3,5 USD).
