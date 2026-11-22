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
| LR vị trí | 1,6e-3 → 1,6e-5 (đơn vị chuẩn hoá) | 1,6e-4 → 1,6e-6 × `scene_radius`; đường đi 1,6e-6 → 1,6e-8 (giữ tỉ lệ 0,01 như gốc) | Đổi đơn vị, cần chỉnh sau run đầu |

Khác nhỏ ngoài bảng của spec:
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
- Phần vẽ (`rasterize_features`, `BezierTrainer.forward`) và việc khởi tạo trong `MultiTrainer` đã chạy thật trên GPU T4 của Kaggle, 1000 iteration (mục "Chạy thật"); chưa chạy đủ 30.000 iteration. `make test-gpu` chưa chạy; các test CPU dùng `pytorch3d`, `gsplat` giả.

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
