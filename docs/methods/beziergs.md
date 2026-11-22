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

Việc còn lại: chạy `make test-gpu`; thử LR opacity 0,005 và trần số điểm, xem ảnh lỗi FRONT_RIGHT; chạy lại với hai seed khác để có khoảng dao động.
