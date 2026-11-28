# Benchmark tuần 04, đợt 3: kết quả (27–28/11/2026)

Đợt 3 chạy trên một máy vast.ai 1× RTX 4090 (offer `60452650`, California, instance `62862145`), một worker, mỗi lúc một run, nên thời gian train trong log là thời gian thật (R8). Plan: `benchmark_week04_wave3.md`. Số liệu lấy từ `runs/<cảnh>/<approach>/<run_id>/d2s/stages/` (`eval.standard/summary.json`, `train.drivestudio/train_stats.json`) và từ bảng `d2s benchmark` (`benchmarks/week04.json`, `benchmarks/benchmark_final.txt`). Mọi run dưới đây đã có ở máy mình (`runs/`, bản nhẹ không có `*.pth`, `scene.ply`; bản đầy đủ kéo qua `envs/vast_autopull.sh`).

## 1. Trạng thái các run

| # | Ô | Run | Trạng thái | PSNR | SSIM | LPIPS | Train | VRAM đỉnh |
|---|---|---|---|---|---|---|---|---|
| 1 | W1 A-LB (OmniRe `fastgs=true`) | `20261127-151558-omnire-182d` | thành công | 25,92 | 0,799 | 0,381 | 1120 s | 2,8 GB |
| 2 | W1 A-L (PVG) | `20261127-155423-pvg-c88f` | thành công | 24,01 | 0,755 | 0,470 | 1741 s | 6,1 GB |
| 3 | W1 A-L (DeformableGS) | `20261127-164106-deformablegs-95e6` | thành công | 24,04 | 0,764 | 0,442 | 4910 s | 22,1 GB |
| 4 | W1 A-0 (PVG, `lidar=false`) | `20261127-215609-pvg-5267` | thành công | 25,02 | 0,767 | 0,402 | 1620 s | 5,6 GB |
| 8 | W1 B (MCGS-SLAM, 3 camera trước) | `20261127-150833-mcgs-9b77` | thành công, dưới ngưỡng plan | 13,23 | 0,648 | 0,771 | 6,8 phút | 0,4 GB |
| 10 | A3 A-LB (OmniRe 3DGEER + FastGS) | `20261127-191729-omnire-67b7` | thành công | 27,45 | 0,852 | 0,126 | 2954 s | 2,5 GB |
| 11 | B2 một run cả cảnh dài (864 khung) | `20261127-225543-omnire-938c` | thành công ở lần chạy thứ 3 (sau hai bản sửa), `RigidNodes` về 0 | 20,74 | 0,651 | 0,417 | 1484 s | 2,7 GB |
| 15 | thử: một khối 320 khung | `20261127-181810-omnire-85bd` | thành công | 25,03 | 0,804 | 0,199 | 3265 s | 2,6 GB |
| 16 | thử: chunk 1 (khung 0–175) | `20261127-201027-omnire-2f64` | thành công | 28,00 | 0,861 | 0,093 | 2330 s | 2,5 GB |
| 16 | thử: chunk 2 (khung 144–319) | `20261127-205220-omnire-f7f0` | thành công | 23,73 | 0,798 | 0,159 | 3638 s | 2,5 GB |

Run hỏng (không vào bảng xếp hạng): #11 `20261127-140512-omnire-95ca` (bước 8400) và `20261127-223742-omnire-7ac1` (bước 9810), nguyên nhân ở mục 5.

Tiêu chí mục 5.5 của plan: #1, #2, #3, #4 `run.json` ổn (#4 có `"lidar": false` ở `train.drivestudio`, `"lidar_usage": []`); #1 có PSNR vùng xe (23,75 dB), cảnh W1 không có người; #10 có `shifted` và PSNR vùng người 24,60 dB; #11, #15 có `shifted`; hai chunk #16 đều thành công.

## 2. W1 đã chọn

Chỉ số **83** (`waymo_segment-11839652018869852123_2565_000_2585_000_with_camera_labels`), 198 khung. `data-waymo` tải và ingest cả 8 ứng viên; sau khi chọn, 7 segment còn lại bị xoá khỏi máy thuê để giảm đĩa (đĩa 200 GB).

| Chỉ số | Khung | Quãng đường | Tốc độ | Đổi hướng | Xe | Người | Vật chạy |
|---|---|---|---|---|---|---|---|
| **83** | 198 | **575 m** | 29,2 m/s | 0° | 13 | 0 | 13 |
| 41 | 199 | 568 m | 28,7 m/s | 8° | 10 | 0 | 9 |
| 46 | 199 | 567 m | 28,6 m/s | 0° | 79 | 0 | 76 |
| 62 | 199 | 373 m | 18,9 m/s | 0° | 35 | 0 | 16 |
| 2 | 199 | 332 m | 16,8 m/s | 1° | 30 | 4 | 12 |
| 71 | 194 | 315 m | 16,3 m/s | 5° | 101 | 1 | 44 |
| 82 | 199 | 138 m | 6,9 m/s | 1° | 107 | 4 | 23 |
| 73 | 198 | 121 m | 6,1 m/s | 1° | 111 | 36 | 95 |

Lý do: dài nhất, có xe chạy (13), ảnh đầu trời sáng rõ. 41 bị lóa nắng ngược ở khung đầu nên loại; 46 hơn 83 nhiều vật chạy (76) nhưng ngắn hơn 8 m (dưới 2%) và ảnh hơi tối. Không ứng viên nào vừa dài vừa có người, nên W1 mới không đo PSNR vùng người. Rủi ro nếu chọn sai: so sánh động học của A-LB trên 83 (13 vật chạy) yếu hơn trên 46. Người chọn là Claude Code theo tiêu chí của plan, K uỷ quyền khi vắng mặt; đổi ý thì `export D2S_W1_INDEX=46` và chạy lại các run W1 (khoảng 3,8 GPU-giờ cho #1–#4).

`lidar-check` trên W1: chiếu LiDAR lên 3 khung, 0 cảnh báo. Ở lần ingest đầu tiên (segment 2), `scene_rig` và `lidar_overlay` của venv Waymo báo "không tạo sơ đồ camera được", "không chiếu LiDAR lên ảnh được" (cảnh báo, không chặn); chưa tìm nguyên nhân, và `lidar-check` riêng trên W1 chạy được.

## 3. W1 ba ô (cảnh dài 575 m, `every8_5cams`)

| Ô | Run | PSNR / SSIM / LPIPS | PSNR vùng xe | Gaussian | Train | VRAM | R3 `.ply` (gap) |
|---|---|---|---|---|---|---|---|
| A-LB, OmniRe + FastGS (#1) | `182d` | **25,92** / 0,799 / 0,381 | **23,75** | 209k (xe 18,4k) | 1120 s | 2,8 GB | 22,99 (2,61) |
| A-L, PVG (#2) | `c88f` | 24,01 / 0,755 / 0,470 | 15,01 | 1,92 triệu | 1741 s | 6,1 GB | 21,49† (2,87) |
| A-L, DeformableGS (#3) | `95e6` | 24,04 / 0,764 / 0,442 | 14,97 | 418k | 4910 s | 22,1 GB | 17,67† (6,59) |
| A-0, PVG không LiDAR (#4) | `5267` | 25,02 / 0,767 / 0,402 | 18,10 | chưa đọc | 1620 s | 5,6 GB | 22,47† (2,75) |

† `.ply` chưa phải nền tĩnh của cảnh đã train (PVG, DeformableGS), R3 không phân xử.

- OmniRe + FastGS hơn PVG 1,9 dB tổng và 8,7 dB vùng xe (23,75 so với 15,01); chỉ OmniRe dựng riêng từng xe. PVG và DeformableGS ngang nhau (24,01 và 24,04, trong ngưỡng hoà 0,3 dB); `d2s benchmark` chọn PVG vì nhanh hơn (29 phút so với 82 phút) và nhẹ VRAM hơn (6,1 so với 22,1 GB). DeformableGS dùng 22 GB, sát giới hạn 24 GB của RTX 4090.
- **A-0 (PVG không LiDAR) cao hơn A-L (PVG có LiDAR) 1,0 dB** (25,02 so với 24,01), ngược với W1 cũ (29,02 so với 29,79). Chỉ một lần chạy mỗi cấu hình và không có ma trận hiệu chỉnh seed, nên chưa kết luận được là LiDAR có hại trên cảnh này. Chênh lệch 1,0 dB lớn hơn ngưỡng hoà 0,3 dB. AbsRel của #4 là 18,1 (so với 3,05 của #2): bỏ LiDAR làm hình học sai nhiều.
- Thời gian train của OmniRe (1120 s) ngắn hơn ước lượng của plan (1,5 GPU-giờ cho cả run) nhiều: số bước cố định và cảnh W1 chỉ có 5 camera, 199 khung.

## 4. #8 MCGS-SLAM trên W1

Run chạy xong trong 6,8 phút (VRAM đỉnh 402 MB, ở 1920×1280, không hết bộ nhớ) và đạt các điều kiện cấu trúc của plan: `prep.mcgs/mcgs_prep.json` có `layout` `[FRONT, FRONT_RIGHT, FRONT_LEFT, FRONT_RIGHT]` (4 phần tử, bắt đầu bằng `FRONT`, camera đối tác lặp ở cuối), có quỹ đạo và `mcgs_eval.json`. Chất lượng **dưới ngưỡng của plan** (ATE > 5 m, PSNR < 15 dB):

| Chỉ số | Giá trị |
|---|---|
| ATE | 78,57 m (57 keyframe), tỉ lệ căn 31,65, `coverage` 0,997 |
| PSNR / SSIM / LPIPS | 13,23 / 0,648 / 0,771 |
| PSNR từng khung (120 ảnh test) | thấp nhất 5,2; trung vị 13,6; cao nhất 18,7 |
| PSNR theo camera | FRONT 12,90; FRONT_LEFT 12,34; FRONT_RIGHT 12,79; SIDE_LEFT 14,69 |

Kiểm lại từ `traj_mcgs.txt` (57 pose keyframe, chỉ đọc file đã kéo về, chưa chạy lại): quỹ đạo ước lượng **dài 17,8 m, trong khi đoạn xe thật dài 575 m**, tức nhỏ hơn khoảng 32 lần, khớp với tỉ lệ căn 31,65. Hướng đi đúng (tăng gần như thuần theo trục z: 15,5 m, lệch ngang 0,6 m, lệch dọc 0,8 m) và quỹ đạo mượt, nên lỗi là **mất thang đo mét**, không phải pose quay loạn. Bước đầu (khung 0 → 3) mới nhỏ hơn thật khoảng 9 lần (0,99 m so với khoảng 8,7 m), về sau tăng tới khoảng 34 lần: thang đo trôi dần về 0. Ảnh render (`FRONT_1.png`) là một mảng nhoè, vì pose ở khung test (nội suy theo quỹ đạo căn lại bằng một hệ số) lệch hàng chục mét. Bố cục thư mục `[FRONT, FRONT_RIGHT, FRONT_LEFT, FRONT_RIGHT]` giống mẫu 100613 trong spec, nên không phải lỗi bố cục. Ảnh đã được undistort (log hiện `Adapt calib` với hệ số méo 0) và thu từ 1920×1280 về 536×360.

Ba giả thuyết chưa kiểm, xếp theo khả năng: (1) tốc độ 29 m/s, khung kế tiếp cách nhau 2,9 m nên cặp stereo (baseline khoảng 1 m, hai camera lệch hướng khoảng 45°) ít thị sai trên đường cao tốc trống, thang đo từ baseline và Metric3D không giữ được; (2) chỉ 57 keyframe trên 174 khung, thưa so với tốc độ; (3) `calib.yml` sai chiều hoặc sai đơn vị ở `baseline` (chưa đọc được file này vì `cache/` chỉ còn trên máy thuê đang dừng). Cách phân biệt: chạy MCGS-SLAM (7 phút, khoảng 0,1 USD GPU) trên cảnh Waymo chậm hơn, hoặc in `calib.yml` và so với mẫu 100613. Mốc so của plan là ATE 1,19 m trên mẫu Waymo 100613 ở Kaggle. Theo plan, không tự đổi cấu hình. Giả thuyết (chưa kiểm): xe chạy 29 m/s nên khung kế tiếp lệch xa, SLAM mất theo dõi; tỉ lệ căn 31,65 cho thấy bản đồ đơn nhãn không giữ được tỉ lệ thật. Việc cần làm nếu K muốn tiếp: chạy lại trên cảnh Waymo chậm hơn (chỉ số 62 hoặc 2, 17–19 m/s) hoặc tăng số keyframe, để biết lỗi do tốc độ hay do cấu hình.

## 5. #11 trên cả cảnh dài (864 khung) và hai bản sửa

**Kết quả lần chạy thứ 3** (`938c`): PSNR 20,74 / SSIM 0,651 / LPIPS 0,417; vùng người 23,28 dB, vùng xe 15,74 dB; R2 `shifted` có (cc 20,81). 87.433 Gaussian (nền 87.157, xe **0**, vật biến dạng 276); train 1484 s (24,7 phút), eval 280 s, VRAM đỉnh 2,7 GB: **không hết VRAM** (ước tính của plan là 24 GB, đo ở 2,7 GB). 2 đoạn hỏng trên trục quãng đường. So với #15 (320 khung): 4,3 dB thấp hơn và 2,7 lần ít Gaussian hơn (87k so với 239k), vì mỗi ảnh được ít bước tối ưu hơn khoảng 2,7 lần và `RigidNodes` mất hết.

**Hai lỗi chặn #11**, cả hai cùng gốc: FastGS prune gần hết Gaussian của `RigidNodes` trên cảnh dài.

| Lỗi | Nguyên nhân gốc (bằng chứng) | Sửa |
|---|---|---|
| Lần 1 (`95ca`, bước 8400): `RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor` ở `rigid.py:268` `split_gaussians` | `RigidNodes` khởi tạo opacity đúng 0,1 (`rigid.py:99`), `omnire_geer_fastgs.yaml` đặt `prune_min_opacity: 0.1`, nên mọi điểm chưa được tối ưu bị FastGS prune. Số điểm: 150.520 → 23.780 (bước 850) → 1 (bước 8400). Còn 1 điểm thì `.squeeze()` biến mask độ dài 1 thành vô hướng, `_scales[mask]` thêm một chiều | `rigid.py`: các mask dùng `.reshape(-1)` thay `.squeeze()` (9 chỗ, kể cả `export_gaussians_to_ply`). Test `test_rigid_single_point.py::test_refinement_after_survives_single_gaussian` |
| Lần 2 (`7ac1`, bước 9810): `ValueError: NaN detected in loss RigidNodes_sharp_shape_reg` | `RigidNodes` về 0 điểm; `compute_reg_loss` lấy `.mean()` trên tensor rỗng (và `kthvalue` của `scaling_reg` cũng lỗi) | `vanilla.py`, `rigid.py`: `compute_reg_loss` trả sớm khi `num_points == 0`. Test `test_reg_loss_is_finite_when_class_has_no_gaussians` |
| Lần 3 (`938c`): train và eval xong, `export.preview` lỗi `cannot reshape tensor of 0 elements into shape [0, -1]` | `gaussians_from_state` (`export_ply.py`) dùng `reshape(len(means), -1)` trên lớp rỗng | `flatten(start_dim=1)`. Test `test_export_empty_class.py`. Chạy lại stage lỗi bằng `d2s run resume` (không train lại), thành công |

Test mới đều fail trước và pass sau trên máy thuê (CPU, env GPU). `tests/drive2sim/recon/models/test_bezier_nodes.py` có 3 test lỗi CUDA guard, cũng lỗi khi dùng `rigid.py` gốc (có sẵn). Các bản sửa (`drive2sim/recon/models/nodes/rigid.py`, `drive2sim/recon/models/gaussians/vanilla.py`, `drive2sim/recon/tools/export_ply.py` và ba test) đã commit ngày 28/11/2026 trên `feat/bench-week04-wave3`.

**Điều K cần quyết (không tự đổi):** FastGS prune `RigidNodes` vì opacity khởi tạo (0,1) bằng ngưỡng `prune_min_opacity` (0,1). Trên cảnh ngắn, các run A-LB vẫn giữ 4,7k–18,4k Gaussian xe (bảng dưới), nhưng trên cảnh dài 864 khung xe mất hết, nên kết quả #11 là OmniRe không có xe tĩnh. Hướng khả dĩ: (a) giữ nguyên và ghi rõ (đã làm ở trên); (b) hạ `prune_min_opacity` của `RigidNodes` xuống dưới 0,1 hoặc bỏ prune theo opacity cho lớp này, rồi chạy lại #11 (khoảng 0,6 GPU-giờ) và các run A-LB, vì kết quả #1, #10, #15, #16 đều có thể đổi; (c) đưa mặt nạ "đã được nhìn thấy trong view lấy mẫu" vào điều kiện prune. Chưa đo (b) và (c).

| Run OmniRe | Gaussian xe (`RigidNodes`) |
|---|---|
| W1 `182d` | 18.397 |
| A2 `7a64` (đợt 2) | 7.049 |
| A3 `67b7` | 12.079 |
| #15 `85bd` (320 khung) | 8.500 |
| #16 chunk 1, chunk 2 | 10.419; 5.267 |
| #11 `938c` (864 khung) | **0** |

## 6. So chunk với một khối (#16 so với #15)

Kiểm nhóm chunked (mục 6.2 của plan, code `drive2sim/engine/chunked.py` chạy lần đầu trên run thật):

| Kiểm | Kết quả |
|---|---|
| đọc chỉ số khung | đạt: không có cảnh báo `không khớp tập test của split` |
| tập khung test | đạt: `frames_measured` của #16 = 39 = số ảnh test `image_00` của #15 |
| quãng đường | đạt: 431,75 m (#16) so với 431,56 m (#15), lệch 0,04% |
| chỗ nối | đạt: 1 dòng, đoạn chồng 144–175, `chồng: trước / sau` có số với 3 khung |

| | #15 một khối (`85bd`) | #16 gộp hai chunk |
|---|---|---|
| PSNR / SSIM / LPIPS | 25,03 / 0,804 / 0,199 | **25,74** / **0,829** / **0,127** |
| R2 `shifted` sau chỉnh màu (PSNR) | 23,97 | **24,77** |
| Tổng thời gian train | **3265 s (54,4 phút)** | 5968 s (99,5 phút) |
| VRAM đỉnh | 2,6 GB | **2,5 GB** |
| Tổng `.ply` | **54,5 MB** | 147,2 MB |
| Gaussian xe cuối | 8.500 | 10.419 + 5.267 |

Quanh chỗ nối (khung 144–175, giữa đoạn chồng 160, ±24 khung):

| | PSNR |
|---|---|
| #16 quanh nối (6 khung test) | 25,98 (kém trung vị cả lần dựng 0,48 dB) |
| #15 ở 7 khung test cùng vùng (136–184) | 25,01 |
| #16, hai chunk trên 3 khung test chung (152, 160, 168) | chunk trước 27,36 / chunk sau 28,67 |
| #15 trên 3 khung đó | 25,4; 26,8; 27,7 (trung bình 26,6) |

**Kết luận:** chunk cao hơn khối 0,71 dB PSNR (lớn hơn ngưỡng hoà 0,3 dB) và LPIPS tốt hơn rõ (0,127 so với 0,199), nhưng đây chưa đủ để nói chia chunk là cách tốt hơn, vì: (1) mỗi cấu hình chạy một lần, không có phép lặp seed; (2) mỗi chunk dùng 30.000 bước trên 176 khung, mỗi ảnh được nhiều bước hơn khoảng 1,8 lần, một yếu tố gây nhiễu khác với số lần chia; plan đã loại phép thử "một khối với số bước gấp đôi" vì lịch densify (`stop_split_at: 15000`) tính theo bước tuyệt đối; (3) chunk 2 (khung 144–319) chỉ đạt 23,73 dB, kém chunk 1 4,3 dB, tức chất lượng phụ thuộc vào đoạn đường nhiều hơn vào việc chia chunk. Cái giá của chia chunk rõ ràng: tổng thời gian train gấp 1,83 lần, `.ply` gấp 2,7 lần, và hai chỗ trùng cần ghép (chưa có bước ghép ở render). Chỗ nối không gây rơi rõ: PSNR quanh nối chỉ kém trung vị 0,48 dB và hai chunk lệch nhau 1,3 dB trên khung chung. VRAM không phân biệt (2,5–2,6 GB), nên lý do chia chunk không còn là bộ nhớ ở cỡ cảnh này (xem #11: 864 khung vẫn chỉ 2,7 GB).

## 7. Chi phí và vận hành

- Instance đầu (offer `41247253`, Texas) kẹt ở `loading` 10 phút, không có container (`intended_status` là `stopped` ngay từ đầu); huỷ, mất khoảng 0,01 USD. Instance thứ hai (offer `60452650`) chạy từ khoảng 13:30 UTC (SSH được lúc 13:36). Giá thực tế 0,628 USD/giờ (offer ghi 0,484 USD/giờ, chưa tính đĩa 200 GB). Credit từ 8,07 USD còn 1,27 USD khi dừng máy: tiêu khoảng **6,8 USD**, gồm cả thời gian dựng máy và các lần chạy hỏng.
- GPU-giờ đo ở hàng đợi (từ lúc bắt đầu tới lúc xong): #8 0,12; #1 0,64; #2 0,78; #3 1,62; #15 0,99; #10 0,88; #16 chunk 1 0,70 và chunk 2 1,06; #4 0,69; #11 lần 3 0,58. Tổng các run thành công khoảng 8,1 giờ (plan ước tính 8,7). Run hỏng: #11 lần 1 khoảng 1,0 giờ (kể cả mask, SMPL khoảng 0,75 giờ), lần 2 khoảng 0,3 giờ. `EST_GPU_HOURS` đã sửa theo số này.
- Dựng máy (ước tính từ mốc log, chưa đo chính xác): `setup` (biên dịch wheel CUDA) và `extras` (MCGS-SLAM, venv Waymo) khoảng 1 giờ 15 phút từ lúc SSH được; `data-kitti` khoảng 25 phút; `data-waymo` (8 segment, mỗi segment 3–5 phút) khoảng 45 phút. #11 lần 1 chạy trong lúc `extras` và `data-waymo` còn chạy.
- **Chuỗi chạy nền cần `setsid`:** lệnh `nohup ... &` qua `ssh` chết khi phiên SSH đóng (chuỗi `chain_wave3.sh` ban đầu mất như vậy, GPU rảnh khoảng 10 phút). Cách chạy được: `vast.sh exec 'setsid nohup bash <script> > log 2>&1 < /dev/null &'`. Gọi `bench_queue.sh` qua `ssh` thẳng (không qua `vast.sh exec`) báo `d2s: command not found` vì thiếu conda env.
- `week04_vast.sh waymo-auth` (`ssh -t`) không chạy khi không có tty: dùng `gcloud auth login --no-launch-browser` trong `tmux` trên máy thuê, đọc link bằng `tmux capture-pane -p -J`, gửi mã bằng `tmux send-keys`. Mã xác minh hiện ở trang web sau khi cấp quyền (`sdk.cloud.google.com/authcode.html`), không hiện trong terminal; người dùng lúc đầu không thấy link (khi để trong khối mã) và không thấy mã (chưa rõ vì sao), nên đưa link ở dạng văn bản thường và nói rõ mã nằm ở trang web.
- **Kéo kết quả về:** `envs/vast_autopull.sh` (`D2S_PULL_LIGHT=1`) kéo mỗi run khi nó có trạng thái `succeeded` hoặc `failed`, và không kéo lại khi trạng thái đổi: #11 `938c` kéo lúc còn `failed`, phải kéo lại tay (`D2S_PULL_LIGHT=1 bash envs/vast.sh pull <run_id>`). Chưa sửa script.
- Máy thuê `62862145` **đã huỷ** theo yêu cầu của K (28/11/2026), sau khi mọi run đã kéo về `runs/`. Máy dừng vẫn tính phí khoảng 0,13–0,15 USD/giờ (đo, không phải 0,07), và không khởi động lại được vì GPU bị người khác thuê ("Required resources are currently unavailable"). Credit còn 0,87 USD. Chunk 3 (khung 288–463) và hai biến thể MCGS-SLAM (đối tác stereo `FRONT_LEFT`; `mode=prgbd`) chưa chạy; cần thuê máy mới và dựng lại môi trường (khoảng 1,3 giờ).
- Test: 3 test mới cho `recon/` (`test_rigid_single_point.py` 2 test, `test_export_empty_class.py` 1 test); chạy trên máy thuê vì `.venv` nhẹ không có PyTorch. Đã có sẵn và không do đợt này: `tests/drive2sim/engine/test_rig.py::test_generate_rig_reports_script_failure` (`FileNotFoundError` khi chạy script giả) và 3 test `test_bezier_nodes.py` (CUDA guard).

## 8. Việc còn lại

- K quyết cách xử lý FastGS prune `RigidNodes` (mục 5) và có chạy lại #11 và các run A-LB không.
- Các bản sửa (mục 5), file định nghĩa benchmark và tài liệu đã commit, push và merge vào `drive2sim` ngày 28/11/2026. Báo cáo tổng hợp và ảnh render: `Phase 3/results/benchmark_week04/`.
- #8 (ATE 78,6 m, PSNR 13,2 dB trên W1): quyết định thử lại trên cảnh chậm hơn hay bỏ khỏi benchmark.
- Cập nhật Sheet: thêm #15, #16; #11 đổi thành một run; điền cột Trạng thái, Run ID, GPU-giờ thật. Project doc `claude/benchmark-week04-decisions-27-11.md` (mục 8.3 của plan) không có trong repo, chưa cập nhật: thêm W1 = 83, kết quả mục 3–6 và kết luận so chunk.
- Ngoài phạm vi đợt này (K quyết 27/11): R3, #9, #12, #13, #14.
