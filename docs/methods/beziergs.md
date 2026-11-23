# BezierGS

Method `beziergs` của stage `train.drivestudio`, approach `beziergs` trên Studio. Port từ BezierGS (ICCV 2025, LogosIntelligenceLab/BezierGS, giấy phép MIT, commit `6df0a36`) vào trainer DriveStudio. Thiết kế đầy đủ: `docs/superpowers/specs/2026-11-20-beziergs-design.md`.

## Mã nằm ở đâu

- `drive2sim/recon/models/nodes/bezier_math.py`: toán Bézier thuần torch (hệ số Bernstein và đạo hàm, fit control point, fit đường đi, hàm map thời gian). Test: `tests/drive2sim/recon/models/test_bezier_math.py`.
- `drive2sim/recon/models/nodes/bezier.py`: node `BezierNodes` thay `RigidNodes`. Test: `tests/drive2sim/recon/models/test_bezier_nodes.py`.
- `drive2sim/recon/trainers/bezier.py`: `BezierTrainer(MultiTrainer)`, ba loss và `collect_bezier_instances`. `BasicTrainer.rasterize_features` (`trainers/base.py`) vẽ một đặc trưng tuỳ ý của Gaussian. Test: `tests/drive2sim/recon/trainers/test_bezier_trainer.py`.
- `drive2sim/configs/methods/beziergs.yaml`, `drive2sim/approaches/beziergs.yaml`.

## Mô hình

- **Nền tĩnh:** 3DGS thường (`VanillaGaussians`), khởi tạo từ LiDAR ngoài box cộng điểm ngẫu nhiên gần và xa.
- **Vật động (xe và người gộp chung):** mỗi vật là một nhóm Gaussian.
  - Tâm nhóm đi theo đường Bézier bậc 3 của vật (`traj_control_points`, học được).
  - Mỗi Gaussian có độ lệch so với tâm cũng là một đường Bézier bậc 3 riêng (`_control_points`, shape `[N, 4, 3]`, đơn vị mét, hệ world).
  - Thời gian chuẩn hoá `[0, 1]` đổi sang tham số `u` của Bézier qua một đa thức Bernstein bậc 3 của từng vật (`time_control_points`, học được).
  - Ngoài khoảng `[khung đầu, khung cuối]` vật có box thì nhóm bị ẩn (opacity bằng 0). Khoảng giữa mà vật bị che và không có box vẫn hiện, vị trí nội suy theo Bézier.
  - Quaternion của Gaussian vật động không xoay theo vật.
- **Khởi tạo vật động:** điểm LiDAR trong box (hệ vật); đường đi là tâm box ở từng khung, fit lặp tối đa 100 vòng theo độ dài dây cung; độ lệch của điểm `p` ở khung `t` là `R_t · p`, fit bằng bình phương tối thiểu theo cùng tham số `u`. Vật chỉ có một khung có box hoặc ít hơn 30 điểm bị bỏ (log cảnh báo).
- **Loss thêm vào** (trọng số trong `beziergs.yaml`):
  - `bezier_dynamic` (0,1): ảnh chỉ vẽ Gaussian vật động so với ảnh thật nhân mask động (L1, SSIM, và alpha so với mask).
  - `bezier_velocity` (1,0): vận tốc ở pixel tĩnh. Vận tốc là đạo hàm theo `u` của Bézier độ lệch (không tính tâm), vẽ như màu rồi chia cho alpha.
  - `bezier_icc` (0,01): độ dài độ lệch hiện tại gần `(‖cp₀‖ + ‖cp₃‖)/2`.
- Cảnh không có vật động nào: `BezierNodes` rỗng bị `MultiTrainer` xoá và các loss trên tự bỏ qua. Dữ liệu không có mask động (`load_dynamic_mask` tắt): `BezierTrainer` báo `ValueError` ngay bước đầu.

## Chỗ khác bản gốc

| Mục | Gốc | drive2sim | Lý do |
|---|---|---|---|
| Hệ toạ độ | Chuẩn hoá PCA về [−1, 1], scale ≤ 0,1 | Mét, world của DriveStudio | Theo khung; LR vị trí nhân `scene_radius` như các method khác |
| Thời gian | [−0,5; 0,5] | [0; 1] (`normalized_timestamps`) | Cùng không gian đa thức bậc 3, mô hình không đổi |
| Mask động | Grounded-SAM khớp với box chiếu | `dynamic_masks` / `fine_dynamic_masks` của drive2sim | Không có Grounded-SAM trong pipeline |
| Color correction | Embedding theo sensor | `AffineTransform` | Module có sẵn |
| Rasterizer | diff-gaussian-rasterization (bản sửa), `neg_fov` | gsplat | Theo khung |
| Densify "contract" | Ngưỡng scale theo khoảng cách | Ngưỡng theo `scene_scale` của DriveStudio | Theo khung |
| 4D SH (Fourier) | Có code nhưng render không dùng | Bỏ | Gốc không dùng |
| LR vị trí | 1,6e-3 → 1,6e-5 (đơn vị chuẩn hoá) | 7,0e-4 → 7,0e-6 × `scene_radius`; đường đi 7,0e-6 → 7,0e-8 (giữ tỉ lệ 0,01 như gốc) | Quy theo mét cho cảnh 017: 0,0268 m (gốc) ≈ 7,0e-4 × 38,4 m. Hệ số phụ thuộc cảnh; bản đầu dùng 1,6e-4 (thấp hơn 4,4 lần) |

Khác nhỏ ngoài bảng của spec:
- Nền khởi tạo: `from_lidar.num_samples` 263.000 (cho ≈ 253 nghìn điểm LiDAR như gốc) cộng 200.000 điểm ngẫu nhiên gần và xa (gốc: 200.000 điểm cầu).
- Scale ban đầu của Gaussian vật động tính riêng từng vật (3 láng giềng gần nhất của control point đầu), như `distCUDA2` từng vật của gốc.
- Split giữ Gaussian cha (scale chia 1,6) và thêm các con, theo quy ước DriveStudio; gốc thay cha bằng các con.
- `ctrl.cull_out_of_bound` bị bỏ qua: Gaussian không còn toạ độ trong hệ box.

## Cách chạy

- Studio: chọn approach `BezierGS` ở trang tạo run. Cảnh cần box 3D và mask vật động tinh (stage `prep.masks_segformer` với `dynamic: true` có trong approach).
- Dòng lệnh: approach `beziergs` như mọi approach khác; hoặc chỉ stage train với `method=beziergs`, `renderer=3dgs`, `campose=false`.
- Yêu cầu: camera pinhole, split `every_nth_frame`.

## Giới hạn

- Chỉ renderer 3DGS, không CamPose. 3DGUT và 3DGEER để sau.
- Approach không có `export.preview`: tool đó chỉ xuất `RigidNodes`. `export.ply_world` chỉ xuất nền.
- Các hàm chỉnh sửa instance và xuất PLY của `RigidNodes` (`remove_instances`, `replace_instances`, `export_gaussians_to_ply`) không có ở `BezierNodes`.
- Phần vẽ (`rasterize_features`, `BezierTrainer.forward`) và việc khởi tạo trong `MultiTrainer` đã chạy thật trên GPU T4 của Kaggle, 1000 iteration (mục "Chạy thật"), kể cả 30.000 iteration. `make test-gpu` chưa chạy; các test CPU dùng `pytorch3d`, `gsplat` giả.

## Chạy thật

### Tóm tắt các run (Waymo 017, khung 61–109, 3 camera, test mỗi khung thứ 4, Kaggle T4)

PSNR / SSIM / LPIPS trên 36 ảnh test của `eval.standard`; thời gian là thời gian stage `train.drivestudio` (các mục dưới ghi `train_time_s` của `summary.json`, thấp hơn vài phút; chạy cùng lúc hai notebook làm run 3 chậm hơn). Chi tiết từng run ở các mục dưới. Thư mục `runs/` của các run này đã xoá khỏi máy sau khi ghi (notebook Kaggle `kascnte/d2s-waymo-017-*` vẫn còn).

| # | Run | Cấu hình | PSNR / SSIM / LPIPS | Gaussian | Train |
|---|---|---|---|---|---|
| 0 | `e1dd` | 1000 iteration | 19,82 / 0,632 / 0,591 | 1,08 triệu nền, 56 nghìn `BezierNodes` | 8 phút |
| 1 | `c2be` | 30k, cấu hình đầu | 27,72 / 0,870 / 0,117 | 3,92 triệu | 286 phút |
| 2 | `9484` | LR vị trí ×4,4, nền khởi tạo như gốc | 27,76 / 0,871 / 0,121 | 4,35 triệu | 295 phút |
| 3 | `de62` | Thêm LR opacity 0,005, trần 5 triệu điểm | 27,27 / 0,847 / 0,154 | 4,71 triệu | 363 phút |
| 4 | `8906` | Run 2 với FastGS | 27,69 / 0,865 / 0,142 | 0,57 triệu | 162 phút |
| 5 | `717a` | FastGS, LR Affine 5e-4 | 27,59 / 0,865 / 0,143 | 0,58 triệu | 165 phút |
| 6 | `28b2` | FastGS, pose camera đúng (`waymo_017_campose`) | 27,16 / 0,859 / 0,148 | 0,57 triệu | 164 phút |
| 7 | `6c42` | StreetGS với FastGS, `waymo_017_campose` | 27,37 / 0,862 / 0,128 | 0,74 triệu | 113 phút |
| | BezierGS gốc | notebook 13 | 29,91 / 0,894 / 0,082 | 1,87 triệu | 169,5 phút |

Rút ra: không thay đổi hyperparameter nào (run 2–3, 5) hay pose (run 6) vượt 0,5 dB; FastGS giữ PSNR với ít hơn 7 lần Gaussian và gần một nửa thời gian (run 4); StreetGS cùng quy trình cũng chỉ 27,4 dB (run 7), nên khoảng cách 2,5 dB với bản gốc nằm ở phần chung với framework, không ở code BezierGS.

### Smoke run 1000 iteration (22/11/2026, Kaggle T4)

Cảnh Waymo 017, khung 61–109 (49 khung), 3 camera (FRONT, FRONT_LEFT, FRONT_RIGHT), split `every_nth_frame:4` (111 ảnh train, 36 ảnh test). Run `20261122-022944-beziergs-e1dd`, `--set train.drivestudio.num_iters=1000`, commit `083fd16`.

Dữ liệu: cảnh không ingest từ tfrecord Waymo mà chuyển từ bản 017 định dạng BezierGS sang thư mục cảnh DriveStudio bằng script dùng một lần (không nằm trong repo). Chỗ khác so với ingest chuẩn: mask động là mask Grounded-SAM của BezierGS (ingest chuẩn dùng mask thô chiếu từ box, rồi `prep.masks_segformer` lọc tiếp); LiDAR không có `origins`, `flows`, `laser_ids` (ghi 0); không có SMPL (`load_smpl=false`); chỉ số khung đánh lại từ 0. Hệ toạ độ box kiểm bằng LiDAR: 8428 điểm trong box xe ở khung 0 với heading gốc, 4084 khi xoay 90°.

Kết quả:

| Mục | Giá trị |
|---|---|
| NaN trong log | không |
| Thời gian | train 497 s (cả `prep.masks_segformer` 428 s riêng) |
| VRAM đỉnh | 3228 MB |
| Gaussian `BezierNodes` | 81.803 lúc đầu, 55.551 ở bước 1000 (cull 53.948 ở lần tinh chỉnh đầu, bước 600), 64.318 sau lần tinh chỉnh cuối |
| Gaussian nền | 777.289 → 1.083.704 |
| `bezier_dynamic_loss` | 0,0076 → 0,0024 (500) → 0,0011 (1000): giảm |
| `bezier_velocity_loss` | 2,5e-6 → 4,0e-5 (500) → 4,5e-5 (1000): **tăng**, rồi phẳng; giá trị rất nhỏ |
| `bezier_icc_loss` | 1,3e-5 → 2,0e-5 → 4,1e-5: **tăng**; cũng rất nhỏ |
| NVS test (36 ảnh) | PSNR 19,82, SSIM 0,632, LPIPS 0,591; PSNR vật động 17,16, xe 16,65 |

Hai loss vận tốc và ICC bắt đầu gần 0 vì độ lệch từng Gaussian khởi tạo đúng bằng đường fit, rồi tăng khi các tham số Bézier được học; chưa kết luận được là đúng hay sai sau 1000 bước. Video test: xe ở đúng vị trí và có hình dạng xe ở các khung đã xem (khung 2 và 40 của video full set), ảnh còn mờ và nhiễu. Chưa kiểm hướng chuyển động bằng số.

Số 1000 bước không so được với notebook 13 (NVS PSNR 29,91, SSIM 0,894, LPIPS 0,082) hay paper (NVS 31,51, trung bình 12 cảnh). Run 30.000 iteration và so sánh chưa làm.

### Run 30.000 iteration (22/11/2026, Kaggle T4)

Cùng cảnh, split và commit code `083fd16` (run `20261122-034526-beziergs-c2be`, notebook chạy commit `7fbbead` chỉ khác docs), mặc định 30.000 iteration, `eval.standard`. Không có NaN trong log.

| Mục | Giá trị |
|---|---|
| Thời gian train | 17.136 s (4 giờ 46 phút) cho stage, `train_time_s` 16.642 s; `prep.masks_segformer` lấy từ cache |
| VRAM đỉnh | 11.777 MB (T4 15 GB) |
| Gaussian cuối | 3.916.240 tổng; `BezierNodes` 101.291, nền 3.814.949 (nền đỉnh 6,4 triệu ở bước 15.000) |
| `render_fps` | 8,3 |
| NVS test (36 ảnh) | PSNR 27,72, SSIM 0,870, LPIPS 0,117 |
| PSNR vật động / xe / người | 25,89 / 25,89 / 25,66 |
| Theo camera (PSNR) | FRONT 27,52; FRONT_LEFT 31,47; FRONT_RIGHT 24,18 |

Loss Bézier theo bước (giá trị tức thời của log):

| Bước | `bezier_dynamic` | `bezier_velocity` | `bezier_icc` | Gaussian `BezierNodes` |
|---|---|---|---|---|
| 0 | 7,6e-3 | 3e-6 | 1,3e-5 | 81.803 |
| 5.000 | 9,4e-4 | 8,5e-5 | 6,5e-5 | 206.743 |
| 15.000 | 3,8e-4 | 1,1e-5 | 3,8e-5 | 258.647 |
| 30.000 | 3,9e-4 | 9e-6 | 2,7e-5 | 101.291 |

`bezier_dynamic` giảm hai bậc độ lớn; `bezier_velocity` tăng ở đầu (đỉnh quanh 1e-4) rồi giảm về cỡ 1e-5, nhỏ hơn đỉnh nhưng lớn hơn giá trị khởi tạo; `bezier_icc` tăng rồi giảm một nửa. Số Gaussian `BezierNodes` tăng tới 259 nghìn rồi giảm còn 101 nghìn khi `stop_split_at` 15.000 và cull tiếp tục.

Video test (khung 1 và 9 của 12): xe ở đúng vị trí so với ảnh thật, hình xe rõ; lớp vật động tách sạch khỏi nền, có vài vệt Gaussian lẻ quanh xe gần camera. Hướng chuyển động chưa kiểm bằng số.

So sánh (cùng cảnh 017, khung 61–109, 3 camera, mỗi khung thứ 4 làm test):

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| drive2sim `beziergs` (run này) | 27,72 | 0,870 | 0,117 |
| Kaggle notebook 13 | 29,91 | 0,894 | 0,082 |
| Chênh | −2,19 | −0,024 | +0,035 |
| Paper (NVS, trung bình 12 cảnh, **không cùng cảnh**) | 31,51 | | |

`beziergs` đang thấp hơn notebook 13 khoảng 2,2 dB. Notebook 13 là run **BezierGS gốc** (`kascnte/beziergs`, code `6df0a36`, `run_summary.json`) trên dữ liệu 017 do tác giả xử lý sẵn, train 169,5 phút, VRAM đỉnh 10,9 GB, 1,87 triệu Gaussian cuối; không phải run của drive2sim. Điểm so chính xác hơn:

| | Gốc (notebook 13) | drive2sim `beziergs` |
|---|---|---|
| NVS test PSNR | 29,91 | 27,72 |
| Train PSNR (đo cùng cách với test) | 33,59 | ước tính ≈ 31,3 (suy từ `full` 30,45 và test 27,72, chưa đo trực tiếp) |
| Chênh train − test | 3,7 | ≈ 3,6 |
| Gaussian cuối | 1,87 triệu | 3,92 triệu (nền đỉnh 6,4 triệu) |
| Thời gian train | 169,5 phút | 286 phút |

Hai bên đo cùng độ phân giải (960×640; BezierGS gốc `load_size = [640, 960]`), nên ý "nửa độ phân giải" ở bản trước của mục này không phải khác biệt. Khác biệt cách đo còn lại: tập test khác khung (gốc `(idx+1) % 4 == 0` tức khung 3, 7, …, 47; drive2sim 4, 8, …, 48), ảnh float so với uint8, `ssim` và LPIPS dùng thư viện khác, PSNR vật động trên `bbox_mask` (gốc) so với mask động (drive2sim).

Chênh train − test của hai bên gần bằng nhau, nên chỗ thua nằm ở độ khớp lúc train (thiếu khoảng 2 dB), không phải ở khả năng suy ra khung chưa thấy. Nghi vấn mạnh nhất tìm được, chưa thử:
- **Learning rate vị trí thấp hơn gốc khoảng 4,4 lần theo đơn vị mét** (đã sửa ở run 2, không cải thiện kết quả). Gốc chuẩn hoá cảnh về [−1, 1] bằng PCA (`scale_factor = min(0,1; 1/max|t|)`, cảnh này 1 đơn vị ≈ 18,4 m) và nhân LR với `cameras_extent` ≈ 0,91. Tính ra LR xyz khởi đầu ≈ 0,0268 m, cuối ≈ 2,7e-4 m, đường đi 2,7e-4 → 2,7e-6 m. Bản port đang dùng 0,0061 m (xyz, cũng áp cho nền) và 6,1e-5 m (đường đi): đổi `1,6e-3 → 1,6e-4` ở bảng "Chỗ khác bản gốc" là quy đổi sai khoảng 4,4 lần.
- Nền: Gaussian khởi tạo 777 nghìn (gốc 253 nghìn điểm nền), và số Gaussian cuối gấp đôi gốc mà vẫn khớp kém hơn.
- Màu: gốc dùng color correction theo sensor (LR 5e-4 → 5e-5), bản port dùng `AffineTransform` LR 1e-5; bản port có thêm mô hình Sky (`EnvLight`), gốc `use_sky: false`.
- FRONT_RIGHT thấp hẳn (24,18, LPIPS 0,195); chưa xem ảnh lỗi nên chưa biết do nền hay vật động.
- Một run, một seed, 36 ảnh test; chưa có khoảng dao động.

Dữ liệu 017 trong run này vốn là bản tác giả BezierGS xử lý sẵn (box, mask Grounded-SAM, LiDAR), nên chênh lệch do chuyển định dạng nhiều khả năng nhỏ; chưa kiểm.

### Run 30.000 iteration lần 2: LR vị trí ×4,4 và nền khởi tạo như gốc (22–23/11/2026)

Run `20261122-110728-beziergs-9484`, commit `75dc6a7`: LR xyz 7e-4 (đường đi 7e-6), `from_lidar.num_samples` 263.000. Không có NaN.

| | Gốc | Run 1 | Run 2 |
|---|---|---|---|
| Test PSNR | 29,91 | 27,72 | 27,76 |
| Test SSIM | 0,894 | 0,870 | 0,871 |
| Test LPIPS | 0,082 | 0,117 | 0,121 |
| Train PSNR (ước tính từ `full` và `test`) | 33,59 (đo trực tiếp) | ≈ 31,3 | ≈ 31,1 |
| Gaussian cuối | 1,87 triệu | 3,92 triệu | 4,35 triệu |
| Gaussian nền đỉnh | | 6,4 triệu | 7,9 triệu |
| Thời gian train | 169,5 phút | 286 phút | 288 phút |
| VRAM đỉnh | 10,9 GB | 11,8 GB | 14,1 GB (T4 có 15 GB) |

Hai thay đổi không cải thiện kết quả (PSNR +0,03, LPIPS xấu đi 0,004, trong khoảng dao động có thể có giữa các lần chạy; chưa đo dao động), và số Gaussian tăng thêm. Giả thuyết "LR vị trí thấp" và "nền khởi tạo nhiều" không giải thích được khoảng cách 2,2 dB.

Số Gaussian nền vượt xa gốc (gốc dừng ở 1,87 triệu tổng, trần `densify_until_num_points` 5 triệu) trong khi khớp ảnh kém hơn, nên nghi vấn tiếp theo nằm ở luật densify và cull. Khác biệt đọc được từ hai file config, chưa thử:
- **LR opacity: 0,05 (bản port, mặc định DriveStudio) so với 0,005 (gốc)**, gấp 10 lần; ảnh hưởng tới cull theo opacity, đặt lại opacity và số Gaussian.
- Gốc có trần 5 triệu điểm (`densify_until_num_points`) và cắt Gaussian quá to theo `big_point_threshold`/`prune_big_point`; bản port không có trần số điểm.
- Color correction theo sensor (gốc) so với `AffineTransform` LR thấp; mô hình Sky chỉ có ở bản port.

### Run 3: LR opacity 0,005 và trần 5 triệu điểm nền (`20261122-164355-beziergs-de62`, commit `2b82ba1`)

Không có NaN. Trần được giữ (nền đỉnh 5,28 triệu điểm ở lần tinh chỉnh cuối trước khi dừng).

| | Gốc | Run 2 | Run 3 (B) |
|---|---|---|---|
| Test PSNR / SSIM / LPIPS | 29,91 / 0,894 / 0,082 | 27,76 / 0,871 / 0,121 | **27,27 / 0,847 / 0,154** |
| Train PSNR (ước tính) | 33,59 | ≈ 31,1 | ≈ 31,0 |
| Gaussian cuối | 1,87 triệu | 4,35 triệu | 4,71 triệu |
| VRAM đỉnh | 10,9 GB | 14,1 GB | 10,0 GB |
| Thời gian train | 169,5 phút | 288 phút | 363 phút (chạy cùng lúc một notebook khác, có thể chậm hơn vì dùng chung CPU) |

LR opacity thấp và trần số điểm không thu hẹp khoảng cách; kết quả kém hơn run 2 (PSNR −0,5, LPIPS +0,03). Giả thuyết "luật densify/cull" qua hai tham số này bị bác bỏ cho cảnh này. Giảm VRAM xuống 10 GB nhờ trần.

### Run 4: FastGS (`beziergs_fastgs.yaml`, `20261122-164307-beziergs-8906`, commit `3421b7d`)

Cùng baseline run 2 cộng `trainer.fastgs.enabled: true`, tham số mặc định (`score_thresh` 1,0, prune từ bước 15.000 với `prune_min_opacity` 0,1). Không có NaN, không có lỗi trong log; `summary.json` ghi `speedups: ["fastgs"]`.

| | Run 2 | Run 4 (FastGS) |
|---|---|---|
| Test PSNR / SSIM / LPIPS | 27,76 / 0,871 / 0,121 | 27,69 / 0,865 / 0,142 |
| PSNR người / xe | 25,94 / 26,08 | **22,89** / 26,05 |
| Train PSNR (ước tính) | ≈ 31,1 | ≈ 30,0 |
| Gaussian cuối | 4,35 triệu (`BezierNodes` 77 nghìn) | 0,57 triệu (`BezierNodes` 18,5 nghìn) |
| VRAM đỉnh | 14,1 GB | 5,5 GB |
| Thời gian train | 288 phút | 162 phút |
| FPS render | 7,7 | 13,3 |

Kết luận: FastGS dùng được với `BezierNodes` (cả điểm số lẫn prune chạy trên nền và vật động, 282 lượt chấm điểm không lỗi). Cùng chất lượng PSNR/SSIM với ít hơn 7 lần Gaussian, một nửa thời gian và 40% VRAM. Giá phải trả: LPIPS xấu đi 0,02, PSNR người đi bộ giảm 3 dB (một run, chưa rõ do dao động hay do prune Gaussian người). Prune chủ yếu là do opacity dưới 0,1 (1,39 triệu Gaussian nền ở bước 17.900); điều kiện sai số cao gần như không kích hoạt (0–2 Gaussian mỗi lượt), nên `prune_score_thresh` chưa có tác dụng. `score_thresh` chưa chỉnh; p90 điểm nền quanh 0,7–1,4, p99 10–29.

### Run 5: FastGS và LR `AffineTransform` 5e-4 → 5e-5 (`20261123-012115-beziergs-717a`)

Như run 4 (FastGS), chỉ đổi LR của `AffineTransform` như BezierGS gốc (log xác nhận LR giảm 5e-4 → 5e-5). Không có NaN.

| | Run 4 | Run 5 |
|---|---|---|
| Test PSNR / SSIM / LPIPS | 27,69 / 0,865 / 0,142 | 27,59 / 0,865 / 0,143 |
| PSNR FRONT / FRONT_LEFT / FRONT_RIGHT | 27,34 / 31,15 / 24,57 | 27,32 / 30,97 / 24,47 |
| PSNR người | 22,89 | 23,48 |
| Train PSNR (ước tính) | ≈ 30,0 | ≈ 29,9 |
| Gaussian cuối | 0,57 triệu | 0,58 triệu |
| Thời gian train | 162 phút | 165 phút |

Giả thuyết "color correction LR thấp làm lệch phơi sáng giữa camera" bị bác bỏ: FRONT_RIGHT không đổi.

### Lỗi trong bản chuyển đổi dữ liệu 017: pose camera

Bản BezierGS gốc dựng pose camera từ pose xe **lúc từng camera chụp** (`ego_pose/<khung>_<camera>.txt`, biến `ego_cam_poses` trong `scene/drivex_waymo_loader.py`). Script chuyển 017 sang định dạng DriveStudio dùng pose chung của khung cho cả ba camera, nên pose camera lệch trung bình 5,5 cm, tối đa 14 cm so với cái bản gốc dùng (đo trên khung 61–109, camera 0–2; xe di chuyển 0,64 m mỗi khung). Cả bốn run trước (và run 5) dùng dữ liệu này. Chưa biết lệch này làm mất bao nhiêu dB.

Cảnh `waymo_017_campose` (`data/`, không commit) ghi pose đúng bằng `camera_pose_start`, `camera_pose` và `camera_timing` (shutter `GLOBAL`), cùng định dạng mà script Waymo của drive2sim ghi và `WaymoCameraData` đọc (đã chạy hàm đọc pose của loader trên cảnh này: 49 pose mỗi camera). Chưa chạy train trên cảnh này.

### Run 6: pose camera đúng, `waymo_017_campose` (`20261123-050839-beziergs-28b2`, commit `6ddd4b3`)

Như run 5, chỉ đổi dữ liệu: pose camera lấy từ `ego_pose/<khung>_<camera>.txt` như BezierGS gốc. Không có NaN; loader đọc `camera_timing` bình thường.

| | Run 5 (pose chung của khung) | Run 6 (pose từng camera) |
|---|---|---|
| Test PSNR / SSIM / LPIPS | 27,59 / 0,865 / 0,143 | 27,16 / 0,859 / 0,148 |
| FRONT / FRONT_LEFT / FRONT_RIGHT | 27,32 / 30,97 / 24,47 | 26,95 / 30,48 / 24,07 |
| Train PSNR (ước tính) | ≈ 29,9 | ≈ 29,3 |
| Gaussian cuối | 0,58 triệu | 0,57 triệu |

Pose đúng không cải thiện mà kém hơn 0,4 dB. Chưa đo dao động giữa các lần chạy cùng cấu hình, nên chưa biết 0,4 dB có vượt nhiễu không; kết luận chắc chắn duy nhất là pose không phải nguyên nhân chính của khoảng cách 2,2 dB.

### Run 7: StreetGS với FastGS làm mốc (`20261123-130229-streetgs-6c42`, commit `6ddd4b3`)

Cùng cảnh `waymo_017_campose`, cùng split `every4_3cams`, cùng `fastgs=true`, chỉ khác method: `streetgs` (RigidNodes, tắt CamPose) thay `beziergs`. Mục đích: xem khoảng cách với BezierGS gốc nằm ở bản port hay ở chung quy trình (dữ liệu, gsplat, framework). Không có NaN.

| | BezierGS (run 6) | StreetGS (run 7) | BezierGS gốc |
|---|---|---|---|
| Test PSNR / SSIM / LPIPS | 27,16 / 0,859 / 0,148 | 27,37 / 0,862 / 0,128 | 29,91 / 0,894 / 0,082 |
| FRONT / FRONT_LEFT / FRONT_RIGHT | 26,95 / 30,48 / 24,07 | 27,09 / 31,07 / 23,95 | |
| PSNR vật động / xe / người | 25,48 / 25,90 / 23,53 | 23,99 / 24,33 / 22,89 | 30,60 (box) / / |
| Train PSNR (ước tính) | ≈ 29,3 | ≈ 30,3 | 33,59 |
| Gaussian cuối | 0,57 triệu | 0,74 triệu | 1,87 triệu |
| Thời gian train | 156 phút | 104 phút | 169,5 phút |
| VRAM đỉnh | 5,4 GB | 3,8 GB | 10,9 GB |

Kết luận (một run mỗi bên, chưa đo dao động):
- PSNR và SSIM tổng của hai method trên cùng quy trình chỉ cách nhau 0,2 dB và 0,003; coi là ngang nhau. Khoảng cách với bản gốc (2,5–2,7 dB) vì vậy **không do code BezierGS** mà do phần chung: dữ liệu chuyển đổi, gsplat hoặc vòng lặp train của DriveStudio. Cả hai method bão hoà quanh 27,2–27,4 dB trên dữ liệu này.
- Bản port BezierGS tốt hơn ở vật động: PSNR vùng vật động +1,5 dB và xe +1,6 dB so với StreetGS, đúng điều BezierGS nhắm tới (quỹ đạo Bézier theo từng Gaussian).
- Bản port kém hơn ở LPIPS (0,148 so với 0,128), ở train PSNR (≈ −1 dB) và tốn thêm khoảng 50% thời gian; chưa rõ nguyên nhân (nghi `bezier_dynamic`/`bezier_velocity`, hoặc BezierNodes học chậm hơn RigidNodes).

### Sai số theo camera và vùng ảnh (run 6, so với bản gốc)

PSNR tính trên MSE gộp, từ video của drive2sim (đã nén H.264, nên thấp hơn `summary.json` khoảng 1 dB) và ảnh của bản gốc (`exp_record`, PNG). **Video xếp panel theo thứ tự FRONT_LEFT, FRONT, FRONT_RIGHT**, không phải chỉ số camera 0, 1, 2; bản đầu của mục này ghép nhầm và đã bị thay. Ảnh thật hai bên khớp nhau (sai khác trung bình 0,013–0,020, do nén).

Khung test (gốc: khung 3, 7, …, 47; drive2sim: 4, 8, …, 48; khác khung nhưng kề nhau). Cột "gốc / drive2sim" theo PSNR (dB):

| Camera | Tổng | Trời | Tĩnh không trời | Vật động |
|---|---|---|---|---|
| FRONT | 30,4 / 26,8 | 36,0 / 28,8 | 29,8 / 26,6 | 30,7 / 28,3 |
| FRONT_LEFT | 32,5 / 30,2 | 34,9 / 27,1 | 33,1 / 31,3 | 25,1 / 24,1 |
| FRONT_RIGHT | 25,9 / 23,8 | 23,4 / 21,3 | 26,2 / 24,0 | 30,8 / 25,9 |

Khung train chung của cả hai bên (24 khung có chỉ số khung chia 4 dư 1 hoặc 2, tức không phải khung test của bên nào):

| Camera | Gốc | drive2sim |
|---|---|---|
| FRONT | 32,9 | 27,9 |
| FRONT_LEFT | 35,9 | 32,4 |
| FRONT_RIGHT | 31,5 | 27,6 |
| Cả ba | 33,1 | 28,8 |

Kết luận: khoảng cách **đồng đều** ở mọi camera và mọi vùng, và có cả ở khung train (hụt khoảng 3–4 dB, trong đó vài phần mười dB đến khoảng 1 dB là do video nén). Đây là thiếu độ khớp lúc train nói chung, không nằm riêng ở trời, ở camera bên hay ở vật động.

Về Sky (kiểm tra theo yêu cầu): màu trời đúng. Màu lệch cả ảnh (colour cast) chỉ chiếm 3–22% MSE vùng trời (FRONT 6%, FRONT_LEFT 22%, FRONT_RIGHT 3%); phần lõi trời (cách biên hơn 10 px) đạt 31,4 dB ở FRONT và 30,8 dB ở FRONT_LEFT, còn dải 10 px sát biên chỉ 23,7–25,0 dB. Lỗi trời vì vậy nằm ở biên trời với tán cây (chi tiết tần số cao), cùng kiểu với lỗi vùng tĩnh, nên mô hình Sky không phải thủ phạm. FRONT_RIGHT lệch lớn cả ở lõi (22,0 dB; bản gốc cũng chỉ 23,4 dB trên vùng trời này), nhiều khả năng do mask trời của camera này kém (IoU giữa mask của tác giả và SegFormer chỉ 0,68). Ảnh lỗi thiên về cạnh sắc (vạch kẻ đường, tán cây): bản port vẽ mờ hơn, khớp với LPIPS cao hơn (0,148 so với 0,082).

### Còn chênh 2,2 dB với bản gốc sau sáu run

LR vị trí, nền khởi tạo, LR opacity, trần số điểm, FastGS và LR của `AffineTransform` đều không thay đổi PSNR quá 0,5 dB, kể cả khi số Gaussian giảm từ 4,7 triệu xuống 0,57 triệu. Số lượng Gaussian và luật densify vì vậy không phải nguyên nhân chính. Pose camera cũng không giúp (run 6), và StreetGS cùng quy trình cũng chỉ đạt 27,4 dB (run 7), nên khoảng cách nằm ở phần chung với framework chứ không ở code BezierGS. Khoảng cách đồng đều ở mọi camera và vùng, và có ở cả khung train: nghi vấn là độ khớp chi tiết nói chung (vẽ mờ hơn): dữ liệu chuyển đổi, bộ rasterizer (gsplat so với `diff-gaussian-rasterization` bản sửa), vòng lặp train và lịch độ phân giải đầu train của DriveStudio; chưa thử riêng cái nào. Mọi run dùng cùng một seed.

Việc còn lại: chạy `make test-gpu`; đo dao động giữa hai seed; tìm nguyên nhân ở phần chung của framework (lịch độ phân giải `res_schedule`, rasterizer, dữ liệu chuyển đổi), vì BezierGS và StreetGS cùng bị; chạy lại với hai seed khác để có khoảng dao động.
