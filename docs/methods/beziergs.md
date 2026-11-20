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
- Phần vẽ (`rasterize_features`, `BezierTrainer.forward`) và việc khởi tạo trong `MultiTrainer` chưa chạy thật trên GPU; các phần còn lại có test CPU với `pytorch3d`, `gsplat` giả.

## Chạy thật

Chưa chạy. Sẽ ghi ở đây kết quả Waymo cảnh 017, khung 61–109, 3 camera, mỗi khung thứ 4 làm test, so với run Kaggle notebook 13 (NVS PSNR 29,91, SSIM 0,894, LPIPS 0,082) và paper (NVS 31,51 trung bình 12 cảnh).
