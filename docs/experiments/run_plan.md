# Kế hoạch và nhật ký các lượt run

Tổng hợp mọi lượt run từ đầu tới nay (đã chạy, đang chạy, sắp chạy) và lý do của từng lượt. Cập nhật mỗi khi có run mới hoặc kết quả mới. Số liệu lấy từ `runs/<scene>/<approach>/<run_id>/d2s/` (`run.json`, `summary.json` của `eval.standard`, `profile_step*.json`); chỗ nào chưa đo ghi rõ.

Cập nhật lần cuối: 15/11/2026.

## Mục tiêu

1. **Tăng tốc OmniRe**: so FastGS (đổi quy tắc densify, `docs/methods/fastgs.md`) và Faster-GS (tăng tốc tính toán, `docs/methods/fastergs.md`), chọn cấu hình tốt nhất.
2. **Chọn renderer**: so 3DGS, 3DGEER, 3DGUT với cùng cấu hình tăng tốc, trên camera pinhole (3DGS chỉ vẽ được pinhole).
3. **Fisheye và đường dài**: dùng renderer thắng trên KITTI-360 0003 (2 pinhole + 2 mắt cá, 1,29 km), yêu cầu đoạn đường từ 1 km.

Tiêu chí chọn: chất lượng trước (PSNR, SSIM, LPIPS, PSNR vùng người, xe), rồi thời gian train, VRAM, số Gaussian khi chất lượng chênh không đáng kể.

## Quy ước chung

- Cảnh so sánh: nuScenes `scene-0061` (191 khung, ~19 s, 91 m), split `front3` (3 camera trước), ảnh thu 2 lần (800×450), 30 000 bước, seed 0, CamPose tắt, `preload_device=cpu`.
- GPU: Kaggle T4 16 GB (notebook do `d2s kaggle push` sinh). Mỗi cấu hình chạy một lần: chênh PSNR dưới khoảng 0,1 dB chưa đủ để kết luận.
- Tài khoản Kaggle: `quanlam` (mặc định); `quynhctrc` khi `quanlam` hết quota (quota làm mới 21/11/2026). Dữ liệu dùng chung chia sẻ từ `quanlam` (`docs/datasets/overview.md` mục 5).
- Tên notebook theo CONTRACTS mục 10.1 (notebook đẩy trước 15/11 còn tên cũ `d2s-<hash8>-<lần>`).

## 1. Chạy thử trên máy (Quadro T2000 4 GB), 14/11/2026

| Run | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|
| (trước khi có `runs/` hiện tại) | Street GS, `scene-0061` CAM_FRONT, ảnh thu 4 lần, 3000 bước, FastGS + Faster-GS | Xong: PSNR test 27,3 dB (19 ảnh), 389 s, 2,4 GB VRAM | Kiểm tra trainer và bản fork gsplat-geer chạy được |
| (trước đó) | `gsplat_3dgut`, KITTI-360 30 khung, 3000 bước | Xong: PSNR test 21,3 dB | Approach tĩnh của teammate |
| `20261114-114325-omnire-93e2` | OmniRe 3DGS + FastGS + Faster-GS | Lỗi ở `prep.smpl` (không ra `smpl.pkl`) | Chưa tìm nguyên nhân trên máy; các run sau chạy trên Kaggle |
| `20261114-143321-streetgs-54ec`, `…-143803-streetgs-8399` | Street GS 3DGS + FastGS + Faster-GS | Lỗi hết bộ nhớ GPU (4 GB, `logs/smoke_streetgs.log`) | GPU máy quá nhỏ cho 800×450 |

## 2. So sánh tăng tốc OmniRe (Kaggle, `quanlam`), 14–15/11/2026

Các lần đẩy đầu lỗi rồi chạy tiếp trên notebook mới; trong đó có hai lỗi phải sửa code:
- image Kaggle chỉ có `nvcc` 13.0, khác CUDA 12.1 của PyTorch → cài CUDA 12.8 riêng (`envs/install_cuda.sh`, commit `098d980`);
- hết bộ nhớ GPU ở bước ~8800 (2,7 triệu Gaussian nền, config nạp mọi ảnh, mask, LiDAR lên GPU) → `preload_device=cpu` và `expandable_segments` (commit `9299811`).

| Run | Notebook (lần đẩy) | FastGS | Faster-GS | PSNR | SSIM | LPIPS | PSNR người / xe / động | Train | VRAM | Gaussian |
|---|---|---|---|---|---|---|---|---|---|---|
| `2d33` | `quanlam/d2s-bdaf4869-5` (5) | | | 25,15 | 0,786 | 0,251 | 24,43 / 24,98 / 24,91 | 3 giờ 39 phút | 6,6 GB | 1,52 triệu |
| `8d9c` | `quanlam/d2s-6639bd1f-3` (3) | | có | **25,20** | 0,785 | 0,251 | 24,38 / 25,04 / 24,81 | 3 giờ 18 phút | 6,9 GB | 1,52 triệu |
| `453a` | `quanlam/d2s-48619076-2` (2) | có | | 25,02 | 0,777 | 0,268 | 24,09 / 24,73 / 24,61 | 2 giờ 47 phút | **4,4 GB** | 0,55 triệu |
| `f8a6` | `quanlam/d2s-ea15d3fd-2` (2) | có | có | 25,01 | 0,778 | 0,269 | 24,15 / 24,75 / 24,62 | **2 giờ 29 phút** | 4,7 GB | 0,55 triệu |

Kết luận:
- Faster-GS giữ nguyên chất lượng, nhanh hơn ~10 %: với OmniRe, phần nó tăng tốc (rasterizer, Adam, SSIM) chỉ khoảng 1/4 mỗi bước; kernel lớn nhất `indexing_backward` (~98 ms/bước) thuộc phần gom Gaussian theo node của OmniRe (`docs/methods/fastergs.md`).
- FastGS giảm số Gaussian còn ~36 %, nhanh hơn và nhẹ hơn rõ, đổi lại PSNR −0,13…0,15 dB, LPIPS +0,017.
- Hai cách cộng dồn: `f8a6` nhanh nhất (−32 % so với gốc).
- Chưa sạch: `2d33`, `8d9c` bật đo profile ở bước 2000 và 20 000, `453a`, `f8a6` không, nên chênh thời gian có thể bị phóng đại một chút.

**Chọn cho bước so renderer: FastGS + Faster-GS**, để ba run chỉ khác renderer (so với `f8a6`) và tốn ít quota.

## 3. So renderer trên pinhole (đang chạy)

Cùng cảnh, split, số bước như mục 2; FastGS + Faster-GS. 3DGEER và 3DGUT chỉ dùng được phần Python của Faster-GS (fused Adam, fused SSIM, bớt chờ GPU); phần kernel chỉ có cho 3DGS.

| Bước | Run | Tài khoản, notebook | Trạng thái | Ghi chú |
|---|---|---|---|---|
| A1 | 3DGS (`f8a6`) | `quanlam/d2s-ea15d3fd-2` | Xong (mục 2) | Mốc |
| A2 | 3DGEER (`20261115-124649-omnire-6b85`) | `quynhctrc/nuscenes-0061-omnire-3dgeer-fastgs-fastergs-6b85-1` | **Đang chạy** (đẩy 15/11 ~19:46) | Ngưỡng densify 0,01 do Bosch chỉnh |
| A3a | 3DGUT chạy thử ~2000 bước, ghi thống kê densify | `quynhctrc` | Chưa: cần module ghi thống kê (`docs/methods/densify_3dgut.md` mục 5) | Ngưỡng densify 3DGUT đang là của 3DGS, chưa chỉnh |
| A3b | Chọn ngưỡng 3DGUT bằng khớp phân vị với 3DGS; config `omnire_ut_calibrated.yaml` | trên máy | Chưa | Config gốc của Bosch giữ lại để so |
| A3c | 3DGUT 30 000 bước, ngưỡng đã chỉnh | `quynhctrc` | Chưa | |
| (tuỳ chọn) | 3DGUT 30 000 bước, ngưỡng gốc của Bosch | `quynhctrc` | Chưa quyết | Cho biết việc chỉnh ngưỡng có tác dụng không |

Kết thúc: bảng 3DGS / 3DGEER / 3DGUT, chọn renderer thắng cho mục 4.

## 4. Fisheye, đường dài: KITTI-360 0003 (kế hoạch)

Cảnh `kitti360_2013_05_28_drive_0003_sync`: 864 khung (khung gốc 167–1030, đoạn có pose liền), 1289 m, 4 camera (2 pinhole 1408×376, 2 mắt cá 1400×1400), 45 vật có box (41 xe con, 4 xe tải), không có người. Đã ingest trên Kaggle bằng `quanlam` (`quanlam/d2s-ingest-kitti360-0003`, báo cáo 0 lỗi).

| Bước | Việc | Ghi chú |
|---|---|---|
| B0 | Ingest lại bằng `quynhctrc` (chỉ CPU) | Notebook ingest private theo tài khoản; cần `d2s-raw-kitti360-0003` chia sẻ cho `quynhctrc` |
| B1 | Chạy thử ~2000 bước, renderer thắng ở mục 3 (chỉ 3DGEER hoặc 3DGUT vẽ được mắt cá) | Đo VRAM, thời gian mỗi bước trên cả 1,29 km × 4 camera |
| B2 | Chạy đủ, hoặc chia đoạn | Theo B1. Mỗi ảnh chỉ được học ~7 lần ở 30 000 bước (4 camera × 864 khung); chia, ghép đoạn làm sau |

Loader mắt cá (`recon/data/loaders/kitti360.py`), box (`ingest/kitti360_boxes.py`) và bỏ qua `prep.smpl` khi cảnh không có người đã có test, **chưa train thật**.

## 5. Sau đó

- Street GS (`20261114-123228-streetgs-60ad`, đã tạo, chưa đẩy).
- Dataset khác: xem `docs/datasets/overview.md` (PandaSet, Argoverse 2 dễ đưa lên Kaggle nhất).
- Chia và ghép đoạn đường dài.
- Tối ưu phần gom Gaussian theo node của OmniRe (`indexing_backward`), nếu cần nhanh hơn nữa.

## Quota GPU (15/11/2026)

| Tài khoản | Còn lại | Làm mới |
|---|---|---|
| `quanlam` | ~2 giờ | 21/11/2026 |
| `quynhctrc` | ~30 giờ (trước A2) | 21/11/2026 |

Ước lượng: mục 3 khoảng 8–11 giờ, B0–B1 khoảng 1–2 giờ, còn khoảng 15 giờ cho B2. Mỗi tài khoản chạy được 2 notebook GPU cùng lúc.
