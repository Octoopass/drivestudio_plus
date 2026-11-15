# Bảng benchmark các lượt run

Mọi lượt run đã xong, đang chạy và đã lên kế hoạch, chia theo loại camera. Không ghi các lượt run lỗi. Số liệu lấy từ `runs/<scene>/<approach>/<run_id>/d2s/` (`summary.json` của `eval.standard`); ô trống là chưa có.

Ký hiệu:
- **Split**: `front3` = `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`; `front1` = `CAM_FRONT`; cả hai lấy mọi khung 0–190 của cảnh, cứ 10 khung lấy 1 khung làm ảnh test (`every_nth_frame:10`).
- **Ảnh train / test**: số ảnh của split.
- **Độ phân giải**: kích thước ảnh lúc train (ảnh gốc thu nhỏ theo `downscale_when_loading` của config dataset).
- **FGS** = FastGS, **FrGS** = Faster-GS; CamPose tắt ở mọi run.
- GPU: Kaggle T4 16 GB, trừ khi ghi khác. Mỗi cấu hình chạy một lần: chênh PSNR dưới khoảng 0,1 dB chưa đủ để kết luận.
- **Code:** run `2d33`, `8d9c`, `453a`, `f8a6`, `6b85`, `ca4b`, `ac56` chạy code trước commit `5bfcd3d` (gom Gaussian theo node chưa tối ưu, `docs/methods/omnire_node_gather.md`); run đẩy sau commit đó chạy code mới, nhanh hơn mỗi bước mà kết quả không đổi. Khi so thời gian train, chỉ so các run cùng loại code.

Cảnh dùng:
- nuScenes `scene-0061`: 191 khung (nội suy lên 10 Hz), ~19 s, 91 m, 6 camera pinhole 1600×900.
- KITTI-360 `2013_05_28_drive_0003_sync`: 864 khung (khung gốc 167–1030), 1289 m, 2 pinhole 1408×376 + 2 mắt cá 1400×1400, 45 xe có box và 1 người đi bộ (khung 629–662).

## Pinhole

| Run | Trạng thái | Approach | Renderer | FGS | FrGS | Dataset, cảnh | Split (camera) | Ảnh train / test | Độ phân giải | Iter | PSNR | SSIM | LPIPS | PSNR người / xe / động | Thời gian train | VRAM đỉnh | Gaussian |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (chạy thử, 14/11) | Xong (Quadro T2000 4 GB) | Street GS | 3DGS | có | có | nuScenes `scene-0061` | CAM_FRONT | | 400×225 (thu 4 lần) | 3 000 | 27,3 | | | | 389 s | 2,4 GB | |
| `20261114-122510-omnire-2d33` | Xong | OmniRe | 3DGS | | | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 25,15 | 0,786 | 0,251 | 24,43 / 24,98 / 24,91 | 3 giờ 39 phút | 6,6 GB | 1,52 triệu |
| `20261114-123142-omnire-8d9c` | Xong | OmniRe | 3DGS | | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 25,20 | 0,785 | 0,251 | 24,38 / 25,04 / 24,81 | 3 giờ 18 phút | 6,9 GB | 1,52 triệu |
| `20261114-123155-omnire-453a` | Xong | OmniRe | 3DGS | có | | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 25,02 | 0,777 | 0,268 | 24,09 / 24,73 / 24,61 | 2 giờ 47 phút | 4,4 GB | 0,55 triệu |
| `20261114-123210-omnire-f8a6` | Xong | OmniRe | 3DGS | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 25,01 | 0,778 | 0,269 | 24,15 / 24,75 / 24,62 | 2 giờ 29 phút | 4,7 GB | 0,55 triệu |
| `20261115-124649-omnire-6b85` | Đang chạy (`quynhctrc`) | OmniRe | 3DGEER | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | | | | | | | |
| `20261115-132753-omnire-ac56` | Đang chạy (`quanlam`): mốc để chỉnh ngưỡng, ghi thống kê densify | OmniRe | 3DGS | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 2 000 | | | | | | | |
| `20261115-132747-omnire-ca4b` | Đang chạy (`quynhctrc`): chỉnh ngưỡng densify, ghi thống kê densify | OmniRe | 3DGUT | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 2 000 | | | | | | | |
| (kế hoạch) | Sau khi chỉnh ngưỡng | OmniRe | 3DGUT (ngưỡng đã chỉnh) | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | | | | | | | |
| (kế hoạch, tuỳ chọn) | | OmniRe | 3DGUT (ngưỡng gốc của Bosch) | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | | | | | | | |
| `20261114-123228-streetgs-60ad` | Đã tạo, chưa đẩy | Street GS | 3DGS | | | nuScenes `scene-0061` | `front1` | 172 / 19 | 800×450 | 30 000 | | | | | | | |

Ghi chú:
- Run `2d33`, `8d9c` bật đo profile ở bước 2000 và 20 000, các run khác không: chênh thời gian có thể bị phóng đại một chút.
- 3DGEER, 3DGUT chỉ dùng phần Python của Faster-GS (fused Adam, fused SSIM, bớt chờ GPU); phần kernel chỉ có cho 3DGS.
- Ngưỡng densify của 3DGUT: `docs/methods/densify_3dgut.md`.

## Fisheye (pinhole + mắt cá)

| Run | Trạng thái | Approach | Renderer | FGS | FrGS | Dataset, cảnh | Camera | Khung | Độ phân giải | Iter | PSNR | SSIM | LPIPS | Thời gian train | VRAM đỉnh | Gaussian |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (chạy thử, 14/11) | Xong (Quadro T2000 4 GB) | `gsplat_3dgut` (cảnh tĩnh) | 3DGUT | | | KITTI-360 0003 (bản 1/5 khung của teammate) | chưa ghi lại | 30 | chưa ghi lại | 3 000 | 21,3 | | | | | |
| (kế hoạch) | Chạy thử đo VRAM, tốc độ | OmniRe | thắng ở bảng pinhole (3DGEER hoặc 3DGUT) | có | có | KITTI-360 0003 đủ khung | `image_00`–`image_03` | 864 | pinhole 704×188, mắt cá 700×700 | 2 000 | | | | | | |
| (kế hoạch) | Theo kết quả chạy thử | OmniRe | như trên | có | có | KITTI-360 0003 đủ khung | `image_00`–`image_03` | 864 (hoặc chia đoạn) | pinhole 704×188, mắt cá 700×700 | 30 000 | | | | | | |

### Thí nghiệm: hai loại camera có gộp được vào một model không

Câu hỏi: thêm camera mắt cá vào cùng model với camera pinhole có làm kém phần pinhole không (hai loại camera "không fuse được"), hay bổ sung được vùng hai bên. Ba run chỉ khác camera dùng để train; cùng cảnh KITTI-360 0003, cùng đoạn khung, renderer, cấu hình, số bước (theo kết quả chạy thử ở bảng trên).

| Run | Trạng thái | Approach | Renderer | FGS | FrGS | Dataset, cảnh | Camera train | Khung | Độ phân giải | Iter | PSNR `image_00`/`01` (pinhole) | PSNR `image_02`/`03` (mắt cá) | PSNR chung | Thời gian train | VRAM đỉnh | Gaussian |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (kế hoạch) | | OmniRe | thắng ở bảng pinhole | có | có | KITTI-360 0003 | chỉ pinhole: `image_00`, `image_01` | như run đủ ở trên | pinhole 704×188 | như run đủ | | | | | | |
| (kế hoạch) | | OmniRe | như trên | có | có | KITTI-360 0003 | chỉ mắt cá: `image_02`, `image_03` | như trên | mắt cá 700×700 | như trên | | | | | | |
| (kế hoạch) | | OmniRe | như trên | có | có | KITTI-360 0003 | cả 4 camera | như trên | như trên | như trên | | | | | | |

Đọc kết quả:
- PSNR trên ảnh pinhole của run "cả 4 camera" thấp hơn rõ run "chỉ pinhole": thêm mắt cá làm hỏng phần pinhole (hai loại camera xung đột: calib, màu, độ nét).
- Ngang hoặc cao hơn: gộp được; run "cả 4 camera" còn phủ thêm hai bên hông.
- Thêm: render camera pinhole ảo đặt đúng vị trí `image_00` từ model "chỉ mắt cá", so với ảnh `image_00` thật, để biết dựng chỉ từ mắt cá mất bao nhiêu khi xem dạng ảnh thường.

Ghi chú:
- Chỉ 3DGEER và 3DGUT vẽ được camera mắt cá, nên bảng này không có 3DGS.
- Với cả 864 khung × 4 camera, 30 000 bước chỉ cho mỗi ảnh được học khoảng 7 lần; chia và ghép đoạn đường làm sau.
