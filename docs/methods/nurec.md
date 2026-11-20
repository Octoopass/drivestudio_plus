# Instant NuRec trong drive2sim

Instant NuRec (NVIDIA, `nvidia/instant-nurec`, commit `f2ecafc`) dựng cảnh 3DGS bằng một mạng feed-forward: không train lại theo cảnh, chỉ suy luận vài chục giây. drive2sim chạy nó như một **codebase ngoài** (`codebase: external:nurec`, ARCHITECTURE mục 2.9), lấy lại phần đã làm ở Fisheye Fusion Lab (converter KITTI-360 → NCore, script chấm điểm, cài đặt, notebook).

## Cài

- `bash envs/install_nurec.sh` (hoặc nút "Cài trên máy này" ở ô môi trường của trang Chạy mới khi chọn Instant NuRec): mã nguồn đúng commit chốt, venv riêng Python 3.11 bằng `uv sync --frozen` (có `nvidia-ncore`), checkpoint `pa-multiview` (khoảng 0,7 GB). Model khác: `--models pa-front`.
- Nơi cài: `D2S_NUREC_DIR`, mặc định `~/.cache/drive2sim/instant-nurec`. Bản đã cài ở chỗ khác (ví dụ `fisheye-fusion-lab/backends/instant-nurec`) thì trỏ biến này tới đó.
- Bước chấm điểm vẽ PLY bằng gsplat của **môi trường chính**, nên môi trường chính cũng phải có.

## Đường dữ liệu

| Stage | Làm gì |
|---|---|
| `prep.ncore` | Đoạn khung và camera của split → clip NCore V4, bằng `external/nurec/tools/kitti360_to_ncore.py` chạy trong môi trường NuRec. Đọc dữ liệu KITTI-360 gốc (bộ ROSA: `ImageSets-rosa-6class/`, `data_poses.zip`, tar ảnh, LiDAR) ở `raw_source.json` của cảnh (ingest ghi) hoặc `D2S_KITTI360_ROOT`. Camera mắt cá resample về camera F-theta 120° của NVIDIA (`rectify120`, mặc định) hoặc giữ cả vòng (`ftheta`). Cache theo cảnh và split |
| `train.nurec` | `run_inference.py --merge` với camera ngữ cảnh theo thứ tự chuẩn của NuRec: `pa-multiview` dùng `image_00` (trước), `image_02`, `image_03` (chéo trái, phải); `pa-front` dùng `image_00`. Số đoạn thời gian tự tính để phủ hết clip (18 khung mỗi đoạn, tối đa 8) |
| `eval.ncore` | Xuất pose, ảnh thật, mask thân xe của các khung test từ clip; vẽ PLY bằng gsplat (3DGUT cho mắt cá); PSNR, SSIM trên vùng hợp lệ (ngoài mask, alpha > 0,9), LPIPS (alex, torchmetrics) với pixel ngoài vùng hợp lệ lấy theo ảnh thật → `summary.json` |
| `export.nurec_ply` | Đổi PLY từ hệ world của clip NCore (rig FLU ở khung đầu, gốc ở `image_00`) sang hệ world của DriveStudio (Velodyne ở khung đầu của split): `extrinsics/0.txt · R(FLU→camera)`. PLY của NuRec chỉ có SH bậc 0 nên chỉ xoay vị trí và quaternion |

## Giới hạn

- Chỉ có đường KITTI-360 (như Fisheye Fusion Lab); dataset khác bị khoá ở `prep.ncore`.
- `pq-front` không chọn được: NuRec cố định model này vào camera `camera_front_wide_120fov`, clip KITTI-360 không có tên đó.
- Không đẩy được lên Kaggle, Colab như approach khác (máy từ xa chỉ cài môi trường chính); dùng notebook NuRec tải ở trang Chạy mới.
- NuRec dựng từ mọi khung của clip, kể cả khung test: điểm test không phải ảnh chưa thấy như với approach train theo cảnh.
- Bầu trời NuRec ghi riêng (`*.sky.npz`), không nằm trong PLY; ảnh render có trời đen, vùng trời bị loại khỏi PSNR nhờ alpha nhưng vẫn vào LPIPS.

## Chạy thật (máy lab, 20/11/2026)

Cảnh `kitti360_0005_nurec`: KITTI-360 `2013_05_28_drive_0005_sync`, 120 khung từ khung 102 (bộ này chỉ có ảnh mỗi 5 khung gốc, nên 0,5 giây một ảnh), 4 camera, LiDAR, không có box 3D. Split `every8` (424 ảnh train, 56 ảnh test). RTX 6000 Ada.

| Lần | Đoạn | Gaussian | Dựng | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| `max_chunks=1` (như mặc định cũ) | 1 | 458 987 | 10 s | 11,9 | 0,38 | 0,14 |
| tự tính (`7`) | 7 | 454 703 | 20,5 s | 13,1 | 0,42 | 0,30 |

Với một đoạn, 8/14 khung test của camera trước không có Gaussian nào (vùng dựng phủ 0%), nên số đoạn nay tự tính. Ảnh render khớp hình học với ảnh thật (đường, dãy nhà, hàng rào cùng chỗ) nhưng mờ.

Mốc so sánh trên cùng cảnh, split `every8_pinhole` (hai camera pinhole `image_00`, `image_01`; 212 ảnh train, 28 ảnh test), vì Street Gaussians và OmniRe cần box 3D mà cảnh này không có:

| Approach | PSNR | SSIM | LPIPS | Thời gian | Gaussian |
|---|---|---|---|---|---|
| PVG, thử nhanh (3000 iteration) | 13,8 | 0,35 | 0,73 | 167 s train | 2 217 749 |
| Instant NuRec `pa-multiview` (1 camera ngữ cảnh) | 14,7 | 0,41 | 0,41 | 11 s | 494 125 |

PVG train riêng cho cảnh cũng chỉ đạt khoảng 14 dB, nên PSNR thấp đến từ dữ liệu (ảnh cách nhau 0,5 giây, khung test xa khung train), không phải từ bản chuyển NuRec. Hai cách đo không giống hệt: `eval.standard` chấm ở nửa độ phân giải, cả ảnh; `eval.ncore` ở độ phân giải gốc, vùng hợp lệ. Cần thử lại trên chuỗi KITTI-360 có đủ khung 10 Hz.
