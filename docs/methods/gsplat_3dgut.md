# gsplat 3DGUT, cảnh tĩnh (approach `gsplat_3dgut`)

Approach dựng **cảnh tĩnh** bằng Gaussian 3DGUT của gsplat, vẽ camera pinhole và camera mắt cá trong cùng một lần dựng. Port trainer KITTI-360 của teammate: repo [thaianh20021/fisheye-fusion-lab](https://github.com/thaianh20021/fisheye-fusion-lab), `tools/colab_cli_kitti360.py`, ghi chú ở `docs/kitti360-3dgut.md` của repo đó. Thiết kế train (khởi tạo LiDAR, vòm trời, MCMC, SelectiveAdam, mask capô, giới hạn `scene_scale`) là của teammate; phần tool viết lại là đọc dữ liệu chuẩn và split, ghi kết quả theo định dạng của tool.

| | |
|---|---|
| File approach | `drive2sim/approaches/gsplat_3dgut.yaml` (`codebase: gsplat`, mức L1) |
| Stage | `train.gsplat_3dgut` (`drive2sim/stages/train/gsplat_3dgut.py`) → `eval.standard` → `export.ply_world` → `export.preview` |
| Trainer | `drive2sim/recon/tools/train_gsplat_3dgut.py`, chạy trong tiến trình con của môi trường chính |
| Cần | LiDAR; camera `opencv_pinhole` không méo (ảnh đã rectify) hoặc `opencv_fisheye`; không cần box |
| Môi trường | Môi trường chính: PyTorch 2.4.1, gsplat-geer 1.5.3 (UT), fused-ssim, OpenCV, SciPy, torchmetrics |

## 1. Cách làm

1. **Dữ liệu.** Đọc thư mục cảnh chuẩn (khung KITTI: `images/`, `lidar/`, `ego_pose/`, `extrinsics/`, `intrinsics/`) và split của run. Pose camera → world = `ego_pose[khung] · extrinsics[camera]`. Ảnh thu nhỏ `downscale` lần (mặc định 2, như teammate), mọi ảnh giữ trên GPU dạng `uint8`.
2. **Khởi tạo.** Điểm LiDAR (khoảng cách 2,5–80 m) của mọi khung trong `frame_range` đưa lên world; màu lấy từ camera đầu tiên cùng khung thấy điểm (chỉ ảnh train); bỏ điểm không camera nào thấy; tối đa `max_init_gaussians` (120 000). Kích thước đầu = khoảng cách trung bình tới 3 điểm gần nhất, kẹp 3 cm–1 m; độ mờ đục 0,3.
3. **Vòm trời** (`sky`, mặc định bật). Khoảng 400 Gaussian mỗi khung trên nửa trên mặt cầu bán kính 200 m quanh xe (cao hơn 12 độ), kích thước 12 m, màu từ ảnh. Theo teammate đây là thay đổi tăng PSNR nhiều nhất.
4. **Render.** `gsplat.rasterization(..., with_ut=True, with_eval3d=True, packed=False)`; camera mắt cá dùng `camera_model="fisheye"` và `radial_coeffs = k1..k4` của intrinsics. Màu là RGB (không có SH).
5. **Loss.** L1 trên vùng được giám sát; `ssim` = `fused` hoặc `torch` thì `0,8·L1 + 0,2·(1 − SSIM)`; cộng `0,01·mean(opacity) + 0,01·mean(scale)`.
6. **Mask.** Camera mắt cá: luôn che vùng ảnh có tia lệch trục hơn `fisheye_max_angle` (1,45 rad): renderer UT không vẽ được tia từ pi/2 trở ra và đa thức OpenCV fisheye chỉ được khớp tới 1,45 rad (`docs/datasets/kitti360.md` mục 5). `hood_mask` (mặc định bật) che thêm nắp capô và vùng đen ngoài vòng tròn ảnh, tìm bằng cạnh ảnh nằm yên qua các khung (`static_masks` của teammate). Đánh giá luôn dùng cả hai mask.
7. **Densify.** `gsplat.strategy.MCMCStrategy`: từ bước 500, mỗi 100 bước, dời Gaussian có độ mờ đục dưới 0,005 và tăng 5 % tới `cap_max` (1 000 000), dừng ở 80 % số bước. `DefaultStrategy` không dùng được vì 3DGUT không có gradient `means2d`.
8. **Tối ưu.** `optimizer = selective_adam` (mặc định, như teammate): `gsplat.optimizers.SelectiveAdam`, chỉ cập nhật Gaussian thấy được trong ảnh đang train. **Kernel CUDA của nó (`gsplat/cuda/csrc/AdamCUDA.cu`) không hiệu chỉnh bias của Adam**: bước cập nhật là `−lr·m/(√v + eps)`, nên vài trăm bước đầu bước lớn hơn Adam chuẩn khoảng `0,1/√0,001 ≈ 3,2` lần. `adam` dùng `torch.optim.Adam`. Teammate chưa so hai lựa chọn. Learning rate: vị trí `1,6e-4 · scene_scale` giảm 100 lần theo bước, quat `1e-3`, scale `5e-3`, opacity `5e-2`, màu `2,5e-2`; `scene_scale = 1,1 ×` bán kính quỹ đạo camera đầu của split, kẹp trong [5, 20] m (teammate: quỹ đạo 370 m cho lr vị trí 0,06 nếu không kẹp).
9. **Chạy tiếp.** Mỗi `checkpoint_every` bước (2500) ghi `checkpoint_resume.pth` (Gaussian, trạng thái optimizer, bước, thời gian train); stage truyền `--resume_from` khi file có sẵn (CONTRACTS mục 5.2). Trạng thái ngẫu nhiên không được lưu.

## 2. Mục cấu hình và preset

| Mục | Lựa chọn (mặc định trước) | Tham số trainer |
|---|---|---|
| `sky` | `true`, `false` | `--sky on|off` |
| `hood_mask` | `true`, `false` | `--hood_mask on|off` |
| `ssim` | `fused`, `torch`, `none` | `--ssim` |
| `optimizer` | `selective_adam`, `adam` | `--optimizer` |

Preset: `base` (không vòm trời, không mask capô, chỉ L1: cấu hình Base trong ablation của teammate), `no_sky`, `adam`. Mặc định là cấu hình tốt nhất của teammate (vòm trời, mask capô, fused-ssim, SelectiveAdam). Tham số khác đặt bằng `--set train.gsplat_3dgut.<tham số>=<giá trị>`: `num_iters` (10 000), `downscale` (2), `max_init_gaussians`, `cap_max`, `fisheye_max_angle` (phải < pi/2), `checkpoint_every`, `seed` (42). `--quick` đặt `num_iters` 3000.

## 3. Kết quả ghi ra và đánh giá

Thư mục `d2s/stages/train.gsplat_3dgut/` có cùng tên file với `train.drivestudio` ở những chỗ stage sau đọc:
- `checkpoint_final.pth`: `{"models": {"Background": {_means, _scales, _quats, _features_dc, _features_rest, _opacities}}, "step"}` theo tên trường của DriveStudio; màu RGB đổi sang SH bậc 0 (`(rgb − 0,5)/0,2821`), `_features_rest` rỗng. `export.ply_world` và `export.preview` đọc file này không cần sửa. Gaussian của vòm trời nằm chung trong `Background` (MCMC dời Gaussian lẫn nhau nên không tách được), nên `scene.ply` có các Gaussian lớn ở 200 m.
- `metrics/images_test_<thời điểm>.json`: `image_metrics/test/{psnr,ssim,lpips,render_fps}` và thêm `image_metrics/test/per_camera/<camera_id>/{psnr,ssim,lpips}`; `eval.standard` ghi `summary.json` với `per_camera` (CONTRACTS mục 6.3).
- `train_stats.json`: `train_time_s` (chỉ vòng train, cộng qua các lần chạy tiếp), `peak_vram_mb`, `num_gaussians`, `num_iters`, `speedups` (`selective_adam`, `fused_ssim`), `resolution` (`downscale=<N>`; `eval.standard` dùng khi không có `config.yaml`).
- `cameras.json` (CONTRACTS mục 7), `compare_<camera>.png` (ảnh gốc | ảnh render), `masks/hood_<camera>.png`, `masks/fov_<camera>.png`, `gsplat_config.json`, `split.json`, `log.txt`.

Đánh giá theo protocol của tool (SPEC mục 9) ở cuối trainer: ảnh render làm tròn 8 bit; PSNR trên điểm ảnh hợp lệ (mask); SSIM `skimage`, LPIPS `torchmetrics` backbone `alex` trên ảnh đã nhân mask; FPS bỏ 3 khung đầu. Khác teammate: teammate báo PSNR "full" và "masked" (chỉ mask capô); tool chỉ báo trên vùng hợp lệ, và vùng hợp lệ của mắt cá nhỏ hơn (thêm mask 1,45 rad), nên số không so thẳng được với bảng của teammate.

**LPIPS cần trọng số AlexNet của torchvision** trong cache torch hub (`~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`). Trainer không tự tải: thiếu thì dừng ngay lúc đầu và in lệnh tải một lần:

```bash
python -c "import torchvision; torchvision.models.alexnet(weights='IMAGENET1K_V1')"
```

## 4. Khác với trainer của teammate

| Teammate | Tool | Lý do |
|---|---|---|
| Tải dữ liệu từ bản sao Hugging Face, đọc calib và khung từ `.pkl` | Đọc thư mục cảnh chuẩn đã ingest (`docs/datasets/kitti360.md`) | Không tải gì, không đọc pickle bên thứ ba |
| Mỗi khung thứ 10 làm test | Theo split của run (`every_nth_frame` hoặc `custom`) | Mọi approach đánh giá trên cùng split |
| Màu khởi tạo, vòm trời và mask capô dùng cả ảnh test | Chỉ ảnh train | Không rò ảnh test vào model |
| Vùng mắt cá ngoài 90 độ vẫn tính loss và PSNR (renderer vẽ đen) | Che vùng ngoài `fisheye_max_angle` khi train và đánh giá | Không phạt model ở vùng không vẽ được |
| PSNR trên ảnh số thực | Ảnh render làm tròn 8 bit; thêm SSIM, LPIPS | Protocol chung của tool |
| `gaussians.ply` riêng | `checkpoint_final.pth` dạng DriveStudio rồi `export.ply_world` | Dùng lại stage export và trang Xem 3D |
| Biến môi trường `KITTI360_*` | Tham số stage, mục cấu hình | Theo CONTRACTS mục 4.1 |

## 5. Đã chạy thật

Máy cá nhân, Quadro T2000 4 GB, `d2s run start gsplat_3dgut ... --executor local` (môi trường `drive2sim` đứng đầu `PATH`). Cảnh: KITTI-360 chuỗi 0003, 30 khung từ khung gốc 167 (bốn camera), split `every_nth_frame:5` (100 ảnh train, 20 ảnh test). Mỗi cấu hình chạy một lần.

| Cấu hình | PSNR test (image_00 / 01 / 02 / 03) | SSIM | LPIPS | Gaussian | Train | VRAM |
|---|---|---|---|---|---|---|
| `--quick` (3000 bước), `downscale` 2, mặc định | 21,27 (21,23 / 21,01 / 20,62 / 22,23) | 0,775 | 0,249 | 305 388 | 82 s | 655 MB |
| 600 bước, `downscale` 4, 30k Gaussian đầu | 18,94 | 0,725 | 0,321 | 36 771 | 4 s | 162 MB |

- Cả bốn stage xong; `summary.json` có `per_camera`; `scene.ply` và gói xem trước ghi được.
- Chạy tiếp: ngắt trainer sau checkpoint ở bước 500 rồi chạy lại với `--resume_from`: tiếp từ bước 501, `train_time_s` cộng dồn, kết quả ghi đủ (thử với `adam`, `ssim torch`).
- Số này **không** so được với kết quả của teammate (100 khung, 10 000 bước, 18,2 dB pinhole / 16,0 dB mắt cá trên T4): ít khung hơn, ít bước hơn, vùng đánh giá khác (mục 3).

## 6. Giới hạn

- Cảnh tĩnh: xe đang chạy bị nhoè; không có actor (mức L1).
- **Mask nắp capô không ổn định.** Trên đoạn 30 khung ở trên, mask che cả phần dưới khoảng 40 % ảnh mắt cá (chỉ giữ 54–58 % điểm ảnh), không chỉ nắp capô: vệt sáng ngang do mặt trời (smear của cảm biến) nằm yên ở cùng hàng ảnh qua các khung nên bị coi là cạnh của capô. Vì đánh giá luôn dùng mask này, PSNR mắt cá của đoạn đó bỏ qua cả phần mặt đường. Xem `masks/hood_<camera>.png` sau mỗi run. Teammate chạy trên đoạn 100 khung khác, chưa kiểm xem có cùng lỗi không.
- Mode Render GPU của Studio không mở được run này (viewer GPU là viewer DriveStudio, cần `config.yaml` của DriveStudio); trang Xem 3D dùng `scene.ply` hoặc gói xem trước.
- Gói xem trước và `scene.ply` có Gaussian vòm trời lớn (tới 40 m) ở 200 m.
- Chưa chạy trên Kaggle (notebook dùng cùng môi trường chính nên về nguyên tắc chạy được, chưa thử).
- Trainer chỉ nhận camera pinhole không méo và OpenCV fisheye; F-theta bị engine khoá.
