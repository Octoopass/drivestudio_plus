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
| `20261115-124649-omnire-6b85` | Xong (`quynhctrc`) | OmniRe | 3DGEER | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 21,66 | 0,744 | 0,317 | 23,39 / 24,42 / 24,09 | 2 giờ 34 phút | 3,6 GB | 0,44 triệu |
| `20261115-132753-omnire-ac56` | Xong (`quanlam`): mốc để chỉnh ngưỡng, ghi thống kê densify (`docs/methods/densify_3dgut.md`) | OmniRe | 3DGS | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 2 000 | | | | | | | |
| `20261115-132747-omnire-ca4b` | Xong (`quynhctrc`): 3DGUT nhánh Bosch densify ít hơn 3DGS 60–380 lần (`docs/methods/densify_3dgut.md`) | OmniRe | 3DGUT | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 2 000 | | | | | | | |
| `20261115-183826-omnire-b8a9` | Xong (`quynhctrc`): 3DGEER densify mỗi 100 bước như 3DGS (`refine_interval=100`, Bosch để 300), so đường học với `f8a6`, `6b85` | OmniRe | 3DGEER | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 3 000 | 20,56 | 0,710 | 0,380 | 22,08 / 22,22 / 22,10 | 17 phút | | 1,32 triệu |
| `20261116-013741-omnire-90dd` | Xong (`quynhctrc`, lần 2; lần 1 lỗi `linalg.solve` ở bước khoảng 600, đã sửa `dc94395`): 3DGUT `ut_densify=screen`, so tỉ lệ densify với `ca4b`, `ac56` (docs/methods/densify_3dgut.md) | OmniRe | 3DGUT | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 2 000 | 21,24 | 0,700 | 0,431 | | 10,5 phút | | 1,07 triệu |
| `20261116-013718-omnire-a8a9` | Xong (`quynhctrc`): 3DGEER densify mỗi 50 bước (`refine_interval=50`), so với 3DGS `f8a6` | OmniRe | 3DGEER | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 21,77 | 0,752 | 0,300 | 23,91 / 24,46 / 24,33 | 2 giờ 43 phút | 4,3 GB | 0,54 triệu |
| `20261116-043455-streetgs-ccec` | Xong (`haicaumuoi`): Street GS + FastGS | Street GS | 3DGS | có | không | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 25,02 | 0,779 | 0,264 | 22,94 / 24,48 / 24,01 | 1 giờ 11 phút | 3,4 GB | 0,43 triệu |
| `20261116-043522-streetgs-ecaa` | Xong (`haicaumuoi`): Street GS + FastGS + Faster-GS | Street GS | 3DGS | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 24,96 | 0,779 | 0,264 | 22,57 / 24,38 / 23,76 | 1 giờ 3 phút | 3,7 GB | 0,43 triệu |
| `20261116-064932-omnire-60e5` | Xong (`quynhctrc`): 3DGUT gốc (`ut_densify=native`) 30k, có profile, so với 3DGS `f8a6`; mask, SMPL dùng lại cache | OmniRe | 3DGUT | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 23,99 | 0,750 | 0,333 | 23,98 / 24,10 / 24,20 | 2 giờ 26 phút | 3,5 GB | 0,33 triệu |
| `20261116-154844-omnire-a022` | Xong (`quynhctrc`, chung notebook với `44d8`): densify MCMC (`densify=mcmc`), SMPL bản phát hành; mốc sạch là `8d9c` | OmniRe | 3DGS | | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 23,56 | 0,716 | 0,403 | 22,39 / 20,82 / 21,76 | 2 giờ 31 phút | 3,8 GB | 1,18 triệu |
| `20261116-154843-omnire-44d8` | Xong (`quynhctrc`, chung notebook với `a022`): densify MCMC; chưa có mốc 3DGUT sạch (`60e5` có FastGS) | OmniRe | 3DGUT | | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | 23,19 | 0,711 | 0,415 | 22,38 / 20,21 / 21,32 | 2 giờ 39 phút | 3,6 GB | 1,18 triệu |
| (kế hoạch) | Sau khi chỉnh ngưỡng | OmniRe | 3DGUT (ngưỡng đã chỉnh) | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | | | | | | | |
| (kế hoạch, tuỳ chọn) | | OmniRe | 3DGUT (ngưỡng gốc của Bosch) | có | có | nuScenes `scene-0061` | `front3` | 516 / 57 | 800×450 | 30 000 | | | | | | | |
| `20261114-123228-streetgs-60ad` | Đã tạo, chưa đẩy | Street GS | 3DGS | | | nuScenes `scene-0061` | `front1` | 172 / 19 | 800×450 | 30 000 | | | | | | | |

Ghi chú:
- Run `2d33`, `8d9c` bật đo profile ở bước 2000 và 20 000, các run khác không: chênh thời gian có thể bị phóng đại một chút.
- 3DGEER, 3DGUT chỉ dùng phần Python của Faster-GS (fused Adam, fused SSIM, bớt chờ GPU); phần kernel chỉ có cho 3DGS.
- Ngưỡng densify của 3DGUT: `docs/methods/densify_3dgut.md`.
- Run `6b85` (3DGEER) thấp hơn run `f8a6` (3DGS, cùng FastGS + Faster-GS) 3,35 dB, và PSNR trên ảnh train cũng thấp (~21–22 dB ở cuối, run `2d33` ~26,5 dB): model học thiếu, không phải học vẹt. Số Gaussian nền gần như không tăng (0,96 triệu lúc đầu, cao nhất ~1,02 triệu ở bước 3000; run `2d33` lên 2,8 triệu) rồi giảm còn 0,29 triệu. Nghi densify của nhánh `geer` (ngưỡng 0,01 trên `geer_gradient`) quá ít; **chưa kiểm chứng**, cần so với log của `f8a6` (cùng FastGS) và thống kê densify của 3DGEER.
- So với `f8a6` (cùng FastGS + Faster-GS): 3DGEER bản Bosch densify mỗi 300 bước (`refine_interval`), 3DGS mỗi 100. Run `b8a9` (3DGEER, densify mỗi 100 bước, 3000 bước) thu hẹp phần lớn khoảng cách ở 3000 bước đầu. PSNR train trung bình cộng dồn tới bước 3000: `f8a6` 20,41, `6b85` 19,65, `b8a9` 20,19 (khoảng cách 0,76 → 0,22 dB); Gaussian nền: 1,30 / 1,02 / 1,16 triệu. Còn ~0,2–0,3 dB có từ bước 500, trước khi densify bắt đầu (cả ba run cùng 0,96 triệu Gaussian): phần này do renderer, không do densify. Chưa biết khoảng cách ở 30 000 bước khi densify cùng nhịp; cần run đủ.
- Run `a8a9` (3DGEER densify mỗi 50 bước, 30 000 bước) so với `6b85` (mỗi 300): PSNR 21,77 so với 21,66, SSIM 0,752 so với 0,744, LPIPS 0,300 so với 0,317; đổi lại train lâu hơn 6% (2 giờ 43 phút so với 2 giờ 34 phút), VRAM thêm 0,7 GB (4,3 so với 3,6 GB), Gaussian thêm 23%. Vẫn thấp hơn 3DGS `f8a6` 3,2 dB, nên mật độ densify không phải nguyên nhân chính. Mỗi cấu hình mới chạy một lần, chưa đo độ dao động giữa các lần chạy. Từ 16/11/2026 `omnire_geer.yaml` dùng `refine_interval: 50` làm mặc định (Street GS + 3DGEER giữ 300, chưa thử).
- Run `60e5` (3DGUT gốc 30k) so với 3DGS `f8a6` (cùng FastGS + Faster-GS): PSNR 23,99 so với 25,01 (−1,0 dB), SSIM 0,750 so với 0,778, LPIPS 0,333 so với 0,269; vùng người, xe, vật động kém 0,2–0,7 dB. Thời gian train gần như bằng (2 giờ 26 phút so với 2 giờ 29 phút), VRAM ít hơn (3,5 so với 4,7 GB), Gaussian ít hơn 40% (0,33 so với 0,55 triệu): 3DGUT densify ít hơn hẳn (docs/methods/densify_3dgut.md), nên phần kém là chất lượng chứ không phải tốc độ. Lần dùng lại cache thật đầu tiên trên Kaggle: mask và SMPL lấy từ output `90dd`, bỏ được khoảng 99 phút GPU.
- Profile `60e5` (30 bước ở bước 2 000 và 20 000, GPU T4): 243 rồi 181 ms mỗi bước; rasterizer 3DGUT chiếm 24% rồi 14% (bước 20 000: backward 17 ms, forward 4 ms), Adam 13%, phần còn lại 62–73% là các phép PyTorch của model (chép bộ nhớ trên GPU 24 ms, phép tính từng phần tử khoảng 45 ms). Kỹ thuật kernel của Faster-GS (sort hai bước, backward theo từng Gaussian) chỉ tác động lên phần rasterizer: giảm một nửa backward cũng chỉ nhanh khoảng 5% mỗi bước. **Kết luận: không port kernel Faster-GS sang 3DGUT**; nếu cần nhanh hơn thì tối ưu phần model dùng chung mọi renderer (chép bộ nhớ, gom Gaussian theo node); việc cần làm với 3DGUT là chất lượng (densify).

- MCMC (`a022` so với mốc sạch `8d9c`, cùng 3DGS + Faster-GS, không FastGS): PSNR 23,56 so với 25,20 (−1,6 dB), SSIM 0,716 so với 0,785, LPIPS 0,403 so với 0,251; train nhanh hơn (2 giờ 31 phút so với 3 giờ 18 phút), VRAM ít hơn (3,8 so với 6,9 GB). Mỗi node chạm đúng mức trần (nền 1 000 000, xe 50 000, vật biến dạng 10 000; SMPL 124 020 không densify), tổng 1,18 triệu so với 1,52 triệu của `8d9c`.
- Mức trần không phải nguyên nhân chính: `f8a6` chỉ có 0,39 triệu Gaussian nền và 29 nghìn Gaussian xe mà đạt 25,01 dB, PSNR xe 24,75. Phần kém nhất của MCMC là xe (20,82 so với 25,04 ở `8d9c`) và vùng động (21,76 so với 24,81); nền kém ít hơn (PSNR vùng không phải trời 23,48 so với 25,21). Giả thuyết chưa kiểm: nhiễu MCMC và phép dời Gaussian áp trong hệ toạ độ riêng của từng xe với cùng hệ số như nền, hoặc dời Gaussian giữa các xe khác nhau (lấy mẫu chung cho mọi xe). Kiểm bằng một run MCMC chỉ cho nền (`Background`), xe và vật biến dạng giữ quy tắc gốc.
- `44d8` (3DGUT + MCMC) so với `60e5` (3DGUT + FastGS) không sạch vì khác cả FastGS; xe cũng tụt mạnh nhất (20,21 so với 24,10).

### Tổ hợp renderer và cách densify trên nuScenes `scene-0061`

| Renderer | Densify gốc (gradient) | + FastGS | + MCMC | FastGS + MCMC |
|---|---|---|---|---|
| 3DGS | `8d9c` 25,20 (có Faster-GS), `2d33` 25,15 (không) | `f8a6` 25,01 (có Faster-GS), `453a` 25,02 (không) | `a022` 23,56 | Không chạy được: MCMC thay đúng quy tắc densify mà FastGS sửa, tool cấm ghép |
| 3DGUT | **Chưa chạy** (mốc 3DGUT sạch còn thiếu) | `60e5` 23,99 | `44d8` 23,19 | Không chạy được, như trên |
| 3DGEER | Chưa chạy | `a8a9` 21,77 (densify mỗi 50 bước), `6b85` 21,66 (mỗi 300 bước) | Chưa chạy | Không chạy được, như trên |

- Faster-GS với 3DGUT, 3DGEER chỉ có phần Python (fused Adam, fused SSIM, ít chờ GPU); ba kỹ thuật kernel chỉ có cho 3DGS (`docs/methods/fastergs.md`). Không port sang 3DGUT vì rasterizer chỉ chiếm khoảng 14% mỗi bước ở bước 20 000 (profile `60e5`).
- Run nên chạy tiếp: 3DGUT densify gốc + Faster-GS (mốc sạch cho `60e5` và `44d8`); MCMC chỉ cho nền (kiểm giả thuyết ở trên).

### Số Gaussian ở bước 15.000 và bước cuối (30.000)

Đếm từ `checkpoint_15000.pth` và `checkpoint_final.pth` của từng run (17/11/2026, đếm số
hàng `_means` của mỗi node trong checkpoint). Cùng nuScenes `scene-0061`, `front3`, 30.000
bước. PSNR là PSNR test trong `summary.json`. "FrGS" là Faster-GS.

| Run | Approach | Renderer | Densify | FastGS | FrGS | Nền 15k → cuối | Xe 15k → cuối | Biến dạng 15k → cuối | Người (SMPL) | Tổng 15k → cuối | Giữ lại | PSNR |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `2d33` | OmniRe | 3DGS | gốc | | | 2.824.372 → 1.281.994 | 117.629 → 103.982 | 17.867 → 13.432 | 124.020 | 3.083.888 → 1.523.428 | 49% | 25,15 |
| `8d9c` | OmniRe | 3DGS | gốc | | có | 2.810.017 → 1.274.449 | 114.497 → 105.459 | 17.783 → 13.580 | 124.020 | 3.066.317 → 1.517.508 | 49% | 25,20 |
| `453a` | OmniRe | 3DGS | gốc | có | | 1.385.935 → 389.932 | 86.782 → 29.148 | 9.738 → 4.839 | 124.020 | 1.606.475 → 547.939 | 34% | 25,02 |
| `f8a6` | OmniRe | 3DGS | gốc | có | có | 1.385.485 → 388.408 | 87.307 → 29.284 | 9.848 → 4.924 | 124.020 | 1.606.660 → 546.636 | 34% | 25,01 |
| `60e5` | OmniRe | 3DGUT | gốc | có | có | 511.129 → 194.450 | 14.399 → 9.398 | 3.871 → 2.695 | 124.020 | 653.419 → 330.563 | 51% | 23,99 |
| `a8a9` | OmniRe | 3DGEER (densify mỗi 50 bước) | gốc | có | có | 1.229.480 → 391.351 | 47.978 → 22.634 | 10.587 → 5.650 | 124.020 | 1.412.065 → 543.655 | 39% | 21,77 |
| `6b85` | OmniRe | 3DGEER (mỗi 300 bước) | gốc | có | có | 806.068 → 292.939 | 25.804 → 15.456 | 7.311 → 4.640 | 124.020 | 963.203 → 437.055 | 45% | 21,66 |
| `a022` | OmniRe | 3DGS | MCMC | | có | 1.000.000 → 1.000.000 | 50.000 → 50.000 | 10.000 → 10.000 | 124.020 | 1.184.020 → 1.184.020 | 100% | 23,56 |
| `44d8` | OmniRe | 3DGUT | MCMC | | có | 1.000.000 → 1.000.000 | 50.000 → 50.000 | 10.000 → 10.000 | 124.020 | 1.184.020 → 1.184.020 | 100% | 23,19 |
| `ccec` | Street GS | 3DGS | gốc | có | | 1.457.051 → 399.921 | 83.245 → 27.697 | | | 1.540.296 → 427.618 | 28% | 25,02 |
| `ecaa` | Street GS | 3DGS | gốc | có | có | 1.461.528 → 399.758 | 83.202 → 28.109 | | | 1.544.730 → 427.867 | 28% | 24,96 |

Đọc bảng:

- Densify gốc không FastGS (`2d33`, `8d9c`) cũng giảm khoảng một nửa số Gaussian sau bước
  15.000, do bước cắt (prune) của DriveStudio; chưa kiểm tra quy tắc cắt nào bỏ nhiều nhất. FastGS bắt đầu
  với ít Gaussian hơn (1,6 triệu so với 3,1 triệu ở bước 15.000) và cắt mạnh hơn (còn 34%),
  nên model cuối nhỏ khoảng 2,8 lần, PSNR thấp hơn khoảng 0,15–0,2 dB.
- Faster-GS gần như không đổi số Gaussian (`453a` so với `f8a6`, `2d33` so với `8d9c`):
  nó tăng tốc tính toán, không đổi quy tắc densify.
- MCMC giữ đúng giới hạn đặt trước (1.000.000 nền, 50.000 xe, 10.000 biến dạng) từ bước
  15.000 tới cuối; cắt và thêm Gaussian chỉ đổi chỗ, không đổi số lượng.
- Người (SMPL) cố định 124.020 Gaussian ở mọi run: densify không áp cho node SMPL.
- Mỗi dòng là một run; chưa lặp lại seed, nên chênh lệch nhỏ giữa hai run gần giống nhau
  (ví dụ `453a` và `f8a6`) chưa đủ để kết luận.

### Chất lượng test ở bước 15.000 so với bước cuối

Đánh giá lại `checkpoint_15000.pth` của từng run trên máy (Quadro T2000, 17/11/2026) bằng
`drive2sim.recon.tools.eval`, cùng tập test 57 ảnh của `front3`; thư mục cảnh ảo dựng lại từ
`data/nuscenes_scene-0061` và cache `prep.masks_segformer@0.1.0-a8f3d932`,
`prep.smpl@0.1.0-d1199b29` như lúc train. Kiểm tra cách đo: đánh giá lại `checkpoint_final.pth`
của `f8a6` ra PSNR 25,0063, khớp `summary.json` gốc. Cột "cuối" lấy từ `summary.json`.

| Run | Renderer | Densify | FastGS | FrGS | PSNR 15k | PSNR cuối | Chênh | SSIM 15k → cuối | LPIPS 15k → cuối |
|---|---|---|---|---|---:|---:|---:|---|---|
| `2d33` | 3DGS | gốc | | | 23,00 | 25,15 | +2,16 | 0,745 → 0,785 | 0,331 → 0,251 |
| `8d9c` | 3DGS | gốc | | có | 22,82 | 25,20 | +2,38 | 0,741 → 0,785 | 0,338 → 0,251 |
| `453a` | 3DGS | gốc | có | | 24,40 | 25,02 | +0,62 | 0,767 → 0,777 | 0,301 → 0,268 |
| `f8a6` | 3DGS | gốc | có | có | 24,33 | 25,01 | +0,68 | 0,767 → 0,778 | 0,302 → 0,269 |
| `60e5` | 3DGUT | gốc | có | có | 23,88 | 23,99 | +0,11 | 0,751 → 0,750 | 0,347 → 0,333 |
| `a8a9` | 3DGEER (mỗi 50 bước) | gốc | có | có | 21,47 | 21,77 | +0,30 | 0,746 → 0,752 | 0,325 → 0,300 |
| `6b85` | 3DGEER (mỗi 300 bước) | gốc | có | có | 21,45 | 21,66 | +0,21 | 0,743 → 0,744 | 0,334 → 0,317 |
| `a022` | 3DGS | MCMC | | có | 20,91 | 23,56 | +2,66 | 0,646 → 0,716 | 0,522 → 0,403 |
| `44d8` | 3DGUT | MCMC | | có | 20,51 | 23,19 | +2,68 | 0,630 → 0,711 | 0,536 → 0,415 |
| `ccec` | Street GS 3DGS | gốc | có | | 24,19 | 25,02 | +0,83 | 0,765 → 0,779 | 0,300 → 0,264 |
| `ecaa` | Street GS 3DGS | gốc | có | có | 24,18 | 24,96 | +0,79 | 0,764 → 0,779 | 0,302 → 0,264 |

Đọc bảng:

- Densify gốc không FastGS và MCMC còn tăng 2,2–2,7 dB sau bước 15.000: dừng ở 15.000 bước
  là thiếu rõ.
- Có FastGS: 3DGS tăng 0,6–0,8 dB; 3DGUT, 3DGEER chỉ tăng 0,1–0,3 dB.
- Giới hạn: checkpoint ở bước 15.000 của một lịch 30.000 bước chưa giảm learning rate tới
  cuối lịch, nên **không bằng** một run train 15.000 bước với lịch rút gọn. Muốn biết rút số
  bước có được không thì phải chạy run 15.000 bước thật; bảng này chỉ cho biết chất lượng còn
  tăng bao nhiêu sau nửa đầu lịch.
- Sau khi đo, các `checkpoint_15000.pth` trên máy đã xoá (17/11/2026) để giải phóng ổ đĩa;
  output notebook Kaggle của các run vẫn còn.

## Fisheye (pinhole + mắt cá)

### Kế hoạch hiện tại: 6 run, mỗi run 30.000 bước

Kiểm tra việc dùng ảnh pinhole và fisheye để học một cảnh 3D chung trên KITTI-360
`kitti360_drive0003`. Chạy đủ ba nhóm camera cho cả OmniRe + 3DGUT và OmniRe + 3DGEER.
Trong mỗi nhóm renderer, ba run chỉ thay camera dùng để train; mỗi run tạo một mô hình
riêng để đối chiếu. Cả sáu run hiện chưa khởi chạy.

Chạy theo ba đợt ưu tiên dưới đây. Trong mỗi đợt, chạy cặp 3DGEER và 3DGUT song
song nếu quota cho phép; nếu chỉ đủ một GPU thì chạy lần lượt theo thứ tự trong bảng.
Hoàn tất và kiểm tra cặp fisheye-only trước khi chuyển sang cặp kết hợp, sau đó mới chạy
cặp pinhole-only. Nếu phát hiện lỗi dữ liệu hoặc phép chiếu, xử lý trước khi chạy đợt sau.

| Đợt ưu tiên | Renderer | Camera train | Split | Ảnh train / test | Bước train | Trạng thái |
|---|---|---|---|---:|---:|---|
| 1 | 3DGEER | Fisheye: `image_02`, `image_03` | `every10_fisheye2` | 362 / 38 | 30.000 | Chưa chạy |
| 1 | 3DGUT | Fisheye: `image_02`, `image_03` | `every10_fisheye2` | 362 / 38 | 30.000 | Chưa chạy |
| 2 | 3DGEER | Kết hợp: cả bốn camera | `every10_mixed4` | 724 / 76 | 30.000 | Chưa chạy |
| 2 | 3DGUT | Kết hợp: cả bốn camera | `every10_mixed4` | 724 / 76 | 30.000 | Chưa chạy |
| 3 | 3DGEER | Pinhole: `image_00`, `image_01` | `every10_pinhole2` | 362 / 38 | 30.000 | Chưa chạy |
| 3 | 3DGUT | Pinhole: `image_00`, `image_01` | `every10_pinhole2` | 362 / 38 | 30.000 | Chưa chạy |

Cấu hình chung:

- Cùng đoạn khung 0–199, khoảng 20 giây; seed 0; khởi tạo từ LiDAR.
- FastGS bật; Faster-GS bật phần Python; MCMC tắt; CamPose tắt.
- Ảnh pinhole đầu vào đã rectify 1408×376, nạp xuống 704×188. Ảnh fisheye
  1400×1400, nạp xuống 700×700 và giữ phép chiếu fisheye, không nắn thành pinhole.
- Giữ lịch độ phân giải của DriveStudio: 250 bước đầu dùng 1/4 kích thước ảnh đã nạp
  theo mỗi chiều, 250 bước tiếp dùng 1/2, từ bước 500 dùng đủ kích thước đã nạp.
- `every_nth_frame:10` giữ các mốc cách nhau 10 khung để đánh giá; không phải chỉ lấy
  một phần mười ảnh để train. Dùng cùng các mốc test giữa những camera tương ứng.
- Kiểm tra người trong đúng đoạn dữ liệu đã ingest trên Kaggle. Nếu xác nhận không có
  người thì bỏ SMPL cho cả ba nhóm camera của mỗi renderer. Nếu có người, dùng chung
  cache SMPL giữa các run để giữ nhất quán dữ liệu người.

Hai renderer dùng config đã có: `omnire_ut_fastgs_fastergs.yaml` và
`omnire_geer_fastgs_fastergs.yaml`. Các thông số densify đặc thù vẫn theo config của
mỗi renderer: 3DGUT có `refine_interval=100`, `densify_grad_thresh=0.0005`;
3DGEER có `refine_interval=50`, `densify_grad_thresh=0.01`. Vì vậy, đối chiếu giữa
hai renderer là đối chiếu hai cấu hình hoàn chỉnh, không chỉ riêng phép render.

Lưu ý phạm vi:

- Cả sáu run chạy đủ 30.000 bước; bước pruning của FastGS bắt đầu từ mốc 15.000
  theo config hiện có.
- Run kết hợp có gấp đôi số ảnh nhưng cùng số bước train. Đây là so sánh ở cùng số bước
  tối ưu, không bảo đảm cùng số lần học trên mỗi ảnh.
- Bước SMPL KITTI-360 hiện lấy tư thế người từ hai camera pinhole. Nếu dùng cache này,
  “fisheye-only” chỉ nói về ảnh tham gia train cảnh, không phải toàn bộ quy trình độc lập
  với pinhole. Người chỉ xuất hiện ở fisheye có thể phải dùng node biến dạng.

### Chạy và đọc kết quả

Kiểm tra calibration, mask và ảnh render ở những bước đầu trước khi để mỗi run chạy
hết 30.000 bước. Dùng lại cache tiền xử lý; mỗi run dùng một GPU T4 và tuân theo
ba đợt ưu tiên ở trên. Ghi thời gian train, VRAM đỉnh và số Gaussian.

Trong từng renderer, so pinhole của run kết hợp với run pinhole-only, và so fisheye của
run kết hợp với run fisheye-only trên cùng ảnh test. Báo PSNR, SSIM và LPIPS theo từng
camera và nhóm camera, đồng thời xem vùng chồng lấn, hai bên hông và vật thể động.
Metric theo camera nằm ở `per_camera` trong `summary.json` (protocol `0.3`); với fisheye, SSIM và
LPIPS chỉ tính trên điểm ảnh hợp lệ như PSNR. Không so SSIM, LPIPS của run mới với run
protocol `0.2` cũ. Không kết luận hai loại camera “không gộp được” chỉ từ một điểm PSNR chung hoặc một
lần chạy ngắn; cần xem cả độ phủ, sai lệch calibration và mức hội tụ.

Có thể render pinhole ảo tại vị trí `image_00` từ mô hình fisheye-only để khảo sát góc
nhìn chưa được train. Kết quả này cần báo riêng với đánh giá trên chính nhóm camera
đã dùng để train.

Tổng ngân sách của sáu run là 180.000 bước tối ưu. Mở rộng đoạn đường làm sau khi
kiểm tra được kết quả trên đoạn 0–199; chưa chốt chạy ngay toàn bộ 864 khung.

### Kết quả chạy thử cũ

Ngày 14/11, trainer 3DGUT tĩnh đã ngừng hỗ trợ từng chạy trên bản KITTI-360 0003 của
teammate (lấy 1/5 số khung), 30 khung, 3.000 bước, Quadro T2000 4 GB, PSNR 21,3.
Camera và độ phân giải chưa được ghi lại. Kết quả này không thay thế sáu run OmniRe ở trên.
