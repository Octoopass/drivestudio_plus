# KITTI-360

Converter: `drive2sim/ingest/kitti360.py`. Ghi một chuỗi KITTI-360 (2 camera pinhole, 2 camera mắt cá, LiDAR Velodyne, pose chính thức) thành thư mục cảnh chuẩn của tool. DriveStudio không có script cho KITTI-360, nên converter tự viết, theo khung thư mục KITTI của DriveStudio (`ARCHITECTURE.md` mục 2.4). Thiết kế chuỗi biến đổi và phép khớp mô hình mắt cá dựa trên công việc của teammate (repo [thaianh20021/fisheye-fusion-lab](https://github.com/thaianh20021/fisheye-fusion-lab), `fisheye_fusion/kitti360.py`, `tools/kitti360_projection_check.py`), nhưng mọi số liệu chỉ lấy từ file chính thức.

License dữ liệu: CC BY-NC-SA 3.0.

## 1. Cần có

```
<thư mục KITTI-360>/
├── calibration/                 calib_cam_to_pose.txt, calib_cam_to_velo.txt, perspective.txt, image_02.yaml, image_03.yaml
├── data_poses.zip               hoặc data_poses/<chuỗi>/poses.txt (cam0_to_world.txt để kiểm tra chéo)
├── data_2d_raw/                 <chuỗi>_image_0X.tar, hoặc <chuỗi>/image_0X/data_rect|data_rgb/*.png đã giải nén
└── data_3d_raw/                 <chuỗi>.tar, hoặc <chuỗi>/velodyne_points/data/*.bin
```

- File `.tar` được đọc thẳng bằng `tarfile`, không giải nén gì vào thư mục nguồn.
- **Không đọc file `.pkl`** (ví dụ `ImageSets-rosa-6class/*.pkl` của bản sao trên Hugging Face mà trainer của teammate dùng): đó là pickle từ nguồn bên thứ ba, `pickle.load` có thể chạy mã tuỳ ý. Khung, calib và pose chỉ lấy từ file chính thức. `d2s source inspect` ghi chú là đã bỏ qua chúng.
- `timestamps.txt` không có trong bản tải về chỉ gồm ảnh; khi đó mốc thời gian là `số hiệu khung × 100 ms` (camera 10 Hz) và báo cáo kiểm tra có mục `timestamps.nominal`.

## 2. Ingest

```bash
d2s source inspect <thư mục KITTI-360>
d2s scene ingest <thư mục KITTI-360> --sequence 2013_05_28_drive_0003_sync \
    [--start-frame 167] [--num-frames 100] [--cameras image_00,image_01,image_02,image_03] [--max-gap 5] [--scene <scene_id>]
```

- Khung dùng được là khung có ảnh của mọi camera đã chọn, có LiDAR và có pose trong `poses.txt` (chỉ một phần các khung có pose). Converter chọn một **đoạn liền nhau**: hai khung cạnh nhau cách nhau quá `--max-gap` (theo số hiệu gốc) thì cắt đoạn, giống `--max-gap` của teammate. Lấy đoạn đầu tiên có đủ `--num-frames` khung; không đoạn nào đủ, hoặc không đặt `--num-frames`, thì lấy đoạn dài nhất.
- Bản tải về của chuỗi 0003 có 204 khung, cách nhau 5 (2, 7, 12, ...; vài chỗ cách 1 hoặc 4); 198 khung có pose (thiếu ở 2, 7, 142–152, 162). Với `--max-gap 5` các đoạn là 12–137, 157, 167–1009: mặc định lấy 167–1009 (171 khung); `--num-frames 100` cho 167–654, đúng đoạn mặc định mới của teammate; `--max-gap 1000000 --num-frames 100` cho 12–520, đoạn teammate dùng cho bảng kết quả của họ (đã kiểm bằng `select_frames` trên dữ liệu thật).
- Studio (thêm cảnh) dùng giá trị mặc định: mọi camera, cả đoạn dài nhất.

## 3. Thư mục cảnh

```
data/kitti360_<chuỗi>/
├── images/<khung>_<chỉ số camera>.png   chép nguyên PNG của nguồn (pinhole 1408×376 đã rectify, mắt cá 1400×1400)
├── lidar/<khung>.bin                     chép nguyên file Velodyne: float32 x, y, z, cường độ, hệ Velodyne
├── ego_pose/<khung>.txt                  4×4, Velodyne → world; world = hệ Velodyne ở khung đầu của cảnh
├── extrinsics/<chỉ số camera>.txt        4×4, camera → Velodyne, trục OpenCV
├── intrinsics/<chỉ số camera>.txt        pinhole: fx fy cx cy k1 k2 p1 p2 k3 (méo 0); mắt cá: fx fy cx cy k1 k2 k3 k4
├── frame_ids.txt                         số hiệu khung gốc theo thứ tự khung
├── timestamps.txt                        micro giây mỗi khung
├── manifest.yaml                         model theo camera; track fisheye; lidars [velodyne]; cuboids none
└── reports/validation.json
```

Chỉ số camera theo thứ tự `image_00, image_01, image_02, image_03` (hoặc thứ tự của `--cameras`); `camera_id` giữ tên thư mục của KITTI-360. Khung KITTI của DriveStudio coi xe trùng hệ LiDAR, nên LiDAR không phải biến đổi. Định dạng intrinsics theo mô hình: CONTRACTS mục 2.1.

## 4. Chuỗi biến đổi

Quy ước `kitti360Scripts` (`T_a_b` đưa điểm từ `a` sang `b`, CONTRACTS mục 1):

| Ký hiệu | Công thức |
|---|---|
| Velodyne → pose (IMU/GPS) | `cam_to_pose[image_00] · calib_cam_to_velo⁻¹` |
| Pose → world (khung `f`) | `poses.txt[f]` |
| `ego_pose[t]` | `(pose[f₀] · velo_to_pose)⁻¹ · pose[f] · velo_to_pose` |
| Camera `k` (chưa rectify) → Velodyne | `velo_to_pose⁻¹ · cam_to_pose[k]` |
| Camera pinhole đã rectify → Velodyne | như trên `· R_rect_0k⁻¹` |
| Intrinsics pinhole | `fx, fy, cx, cy` từ `P_rect_0k` (cột 4 của `P_rect_01` là đường cơ sở stereo, đã nằm trong extrinsics) |
| Intrinsics mắt cá | `fx = gamma1/(1+xi)`, `fy = gamma2/(1+xi)`, `cx = u0`, `cy = v0`, `k1..k4` khớp từ Mei (mục 5) |

Pose của camera trong world của cảnh là `ego_pose[t] · extrinsics[camera]`.

## 5. Camera mắt cá: Mei → OpenCV fisheye

`image_02`, `image_03` hiệu chỉnh bằng mô hình Mei (`xi`, `k1`, `k2`, `p1`, `p2`, `gamma1`, `gamma2`, `u0`, `v0`). Renderer 3DGUT của gsplat chỉ vẽ OpenCV fisheye, nên `drive2sim/core/camera.py` khớp `k1..k4` sao cho `θ·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)` bằng `(1+xi)·u·(1 + k1u² + k2u⁴)` với `u = sin θ / (cos θ + xi)`, trên 256 góc từ 0,02 tới 1,45 rad (cách của teammate, cho cùng hệ số với `numpy.linalg.lstsq`).

Giới hạn (đo trên calib thật, ảnh 1400 px):
- Sai số phần xuyên tâm của phép khớp tới 1,45 rad: 0,007 px (`image_02`), 0,022 px (`image_03`); tới pi/2 tăng lên 0,13 px (`image_02`). Teammate ghi "< 0,002 px"; con số đó chưa kiểm lại được (có thể đo ở ảnh nửa độ phân giải hoặc chỉ ở các góc mẫu).
- Méo tiếp tuyến `p1`, `p2` bị bỏ (OpenCV fisheye không có). So với Mei chính xác trên điểm LiDAR thật (khung gốc 167): trung vị 0,10 px, lớn nhất 0,42 px (`image_02`); trung vị 0,17 px, lớn nhất 0,78 px (`image_03`).
- Camera nhìn rộng hơn 180 độ. Renderer UT của gsplat không vẽ được tia từ pi/2 trở ra, và đa thức chỉ khớp tới 1,45 rad (83 độ), nên vùng ảnh ngoài 1,45 rad **phải được che** khi train và đánh giá; trainer `train.gsplat_3dgut` che vùng đó (`docs/methods/gsplat_3dgut.md`). Báo cáo kiểm tra ghi mục `camera.fisheye_fit` và `camera.fisheye_fov`.

## 6. Đã kiểm tra trên dữ liệu thật

Chuỗi `2013_05_28_drive_0003_sync`, 30 khung từ khung gốc 167 (`--start-frame 167 --num-frames 30`), ingest mất 1,2 s, 248 MB.

1. **Báo cáo kiểm tra** (`pose.cam0_to_world`): pose camera 0 dựng từ `poses.txt` và calib so với `cam0_to_world.txt` chính thức ở cả 30 khung: lệch lớn nhất `6,4e-7` (phần tử ma trận xoay) và 0,45 mm (tịnh tiến). `cam0_to_world.txt` chỉ ghi 7 chữ số có nghĩa (toạ độ cỡ 3000 m làm tròn tới 1 mm), nên đây là sai số làm tròn; ngưỡng báo lỗi là 5 mm.
2. **Chiếu LiDAR** (`python -m drive2sim.scripts.check_kitti360_projection --scene ... --raw ... --frame 0`, cần numpy, OpenCV; cách của `tools/kitti360_projection_check.py` của teammate), khung 0 và 15 của cảnh:
   - extrinsics trong thư mục cảnh trùng chuỗi chính thức dựng lại bằng numpy (`R_rect · cam_to_pose[k]⁻¹ · cam_to_pose[00] · calib_cam_to_velo⁻¹`) tới `3e-16`; pixel chiếu lên `image_00`, `image_01` lệch dưới `1e-12` px;
   - mắt cá: như mục 5;
   - nhiều khung: LiDAR khung gốc 167 đưa lên world bằng `ego_pose` rồi chiếu vào `image_00` ở khung gốc 192, so với cùng phép dùng `cam0_to_world.txt`: trung vị 0,0015 px, lớn nhất 0,024 px (khung 242 → 267: trung vị 0,021 px, lớn nhất 0,13 px, do làm tròn của file);
   - điểm cạnh (`edge_ratio`, độ mạnh cạnh ảnh dưới điểm có bước nhảy độ sâu so với dưới mọi điểm): khung 0 `image_00` 1,42, `image_01` 1,50; làm lệch extrinsics 1 độ quanh trục y thì còn 1,34 và 1,44. Với mắt cá tỉ số không phân biệt được (1,31 so với 1,28; 1,09 so với 1,09), như teammate đã ghi: mắt cá chỉ nên xét bằng ảnh chồng điểm. Ảnh chồng điểm cho thấy cột điện, biển báo, đường chân trời khớp ở cả bốn camera.
3. **So với chuỗi của teammate**: teammate đo chuỗi của trainer (lấy `lidar2cam` từ `.pkl`) trùng chuỗi chính thức tới 0,0005 px; chuỗi của tool trùng chuỗi chính thức tới `1e-12` px, nên cũng trùng chuỗi của teammate. Khác ở gốc world: teammate lấy world chính thức dời gốc về vị trí camera 0 khung đầu (không xoay), tool lấy hệ Velodyne ở khung đầu (CONTRACTS mục 1); hai cách chỉ khác một phép biến đổi cứng cố định.

## 7. Chưa làm

- Box 3D (`data_3d_bboxes`): converter chưa đọc; `available.cuboids: none`, nên OmniRe, Street GS bị khoá trên cảnh KITTI-360. Hai approach đó cũng chưa có config dataset KITTI-360.
- `timestamps.txt` thật của ảnh và LiDAR (khi nguồn có) mới được đọc cho ảnh của camera đầu; LiDAR dùng cùng mốc.
- Interpolate pose cho khung không có pose: không làm, như `kitti360Scripts`.
