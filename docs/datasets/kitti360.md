# KITTI-360

Converter: `drive2sim/ingest/kitti360.py`. Ghi một chuỗi KITTI-360 (2 camera pinhole, 2 camera mắt cá, LiDAR Velodyne, pose chính thức) thành thư mục cảnh chuẩn của tool. DriveStudio không có script cho KITTI-360, nên converter tự viết, theo khung thư mục KITTI của DriveStudio (`ARCHITECTURE.md` mục 2.4). Chuỗi biến đổi theo quy ước của kitti360Scripts chính thức; mọi số liệu chỉ lấy từ file chính thức.

License dữ liệu: CC BY-NC-SA 3.0.

## 1. Cần có

Trong repo, đặt ở `data/kitti360/raw/` (như `data/nuscenes/raw/` của nuScenes, ARCHITECTURE mục 3):

```
data/kitti360/raw/
├── calibration/                 calib_cam_to_pose.txt, calib_cam_to_velo.txt, perspective.txt, image_02.yaml, image_03.yaml
├── data_poses.zip               hoặc data_poses/<chuỗi>/poses.txt (cam0_to_world.txt để kiểm tra chéo)
├── data_2d_raw/                 <chuỗi>_image_0X.tar, hoặc <chuỗi>/image_0X/data_rect|data_rgb/*.png đã giải nén
├── data_3d_raw/                 <chuỗi>.tar, hoặc <chuỗi>/velodyne_points/data/*.bin
└── data_3d_bboxes.zip           box 3D (tuỳ chọn), hoặc data_3d_bboxes/train/<chuỗi>.xml; tải ở mục "3D data & labels" (cần đăng nhập)
```

- File `.tar` được đọc thẳng bằng `tarfile`, không giải nén gì vào thư mục nguồn.
- **Không đọc file `.pkl`** (ví dụ `ImageSets-rosa-6class/*.pkl` của bản sao trên Hugging Face): đó là pickle từ nguồn bên thứ ba, `pickle.load` có thể chạy mã tuỳ ý. Khung, calib và pose chỉ lấy từ file chính thức. `d2s source inspect` ghi chú là đã bỏ qua chúng.
- `timestamps.txt` không có trong bản tải về chỉ gồm ảnh; khi đó mốc thời gian là `số hiệu khung × 100 ms` (camera 10 Hz) và báo cáo kiểm tra có mục `timestamps.nominal`.

## 2. Ingest

```bash
d2s source inspect <thư mục KITTI-360>
d2s scene ingest <thư mục KITTI-360> --sequence 2013_05_28_drive_0003_sync \
    [--start-frame 167] [--num-frames 100] [--cameras image_00,image_01,image_02,image_03] [--max-gap 5] [--scene <scene_id>]
```

- Khung dùng được là khung có ảnh của mọi camera đã chọn, có LiDAR và có pose trong `poses.txt` (chỉ một phần các khung có pose). Converter chọn một **đoạn liền nhau**: hai khung cạnh nhau cách nhau quá `--max-gap` (theo số hiệu gốc) thì cắt đoạn. Lấy đoạn đầu tiên có đủ `--num-frames` khung; không đoạn nào đủ, hoặc không đặt `--num-frames`, thì lấy đoạn dài nhất.
- Bản tải về của chuỗi 0003 có 204 khung, cách nhau 5 (2, 7, 12, ...; vài chỗ cách 1 hoặc 4); 198 khung có pose (thiếu ở 2, 7, 142–152, 162). Với `--max-gap 5` các đoạn là 12–137, 157, 167–1009: mặc định lấy 167–1009 (171 khung); `--num-frames 100` cho 167–654; `--max-gap 1000000 --num-frames 100` cho 12–520 (đã kiểm bằng `select_frames` trên dữ liệu thật).
- Studio (thêm cảnh) dùng giá trị mặc định: mọi camera, cả đoạn dài nhất.

## 3. Thư mục cảnh

```
data/kitti360_<chuỗi>/
├── images/<khung>_<chỉ số camera>.png   chép nguyên PNG của nguồn (pinhole 1408×376 đã rectify, mắt cá 1400×1400)
├── lidar/<khung>.bin                     chép nguyên file Velodyne: float32 x, y, z, cường độ, hệ Velodyne
├── ego_pose/<khung>.txt                  4×4, Velodyne → world; world = hệ Velodyne ở khung đầu của cảnh
├── extrinsics/<chỉ số camera>.txt        4×4, camera → Velodyne, trục OpenCV
├── intrinsics/<chỉ số camera>.txt        pinhole: fx fy cx cy k1 k2 p1 p2 k3 (méo 0); mắt cá: fx fy cx cy k1 k2 k3 k4
├── instances/                            instances_info.json, frame_instances.json (khi nguồn có box, mục 3.1)
├── dynamic_masks/{all,human,vehicle}/    <khung>_<chỉ số camera>.png, mặt nạ vùng box (mục 3.1)
├── frame_ids.txt                         số hiệu khung gốc theo thứ tự khung
├── timestamps.txt                        micro giây mỗi khung
├── manifest.yaml                         model theo camera; track fisheye; lidars [velodyne]; cuboids gt (có box) hoặc none
└── reports/validation.json
```

Chỉ số camera theo thứ tự `image_00, image_01, image_02, image_03` (hoặc thứ tự của `--cameras`); `camera_id` giữ tên thư mục của KITTI-360. Khung KITTI của DriveStudio coi xe trùng hệ LiDAR, nên LiDAR không phải biến đổi. Định dạng intrinsics theo mô hình: CONTRACTS mục 2.1.

### 3.1 Box 3D

Converter đọc `data_3d_bboxes/train/<chuỗi>.xml` (`drive2sim/ingest/kitti360_boxes.py`) và ghi theo đúng định dạng script KITTI của DriveStudio (`drive2sim/recon/preprocess/kitti/process.py`, `save_objects`, `save_dynamic_mask`), để loader KITTI đọc được:

- **Lớp**: chỉ giữ xe và người, đổi sang lớp của loader KITTI: `car` → Car; `truck`, `bus`, `trailer`, `caravan` → Truck; `train` → Tram; `person` → Pedestrian; `rider`, `bicycle`, `motorcycle` → Cyclist. Loader không có lớp riêng cho bus, rơ-moóc, xe máy.
- **Vật**: nhận diện bằng cặp (`semanticId`, `instanceId`), đánh lại id từ 0. Vật chuyển động có một box mỗi khung (`timestamp` là số hiệu khung); vật đứng yên (`timestamp = -1`) có cùng box ở mọi khung từ `start_frame` tới `end_frame`. Cả hai đều thành node (xe đỗ cũng là node cứng, như DriveStudio làm với KITTI).
- **Hình học**: `transform` của XML là trục box nhân kích thước; tách thành phép quay và `box_size = [dài, rộng, cao]`, tâm box là `obj_to_world`, đưa về world của cảnh (Velodyne ở khung đầu). Trên chuỗi 0003 mọi box xe có trục x dài nhất và z là chiều cao; 2 trong 1144 box có ma trận lật (định thức âm), được sửa bằng cách đổi chiều trục y.
- **Mặt nạ vật động**: hình chữ nhật bao 8 góc box chiếu lên ảnh, chỉ khi **mọi** góc nằm trong ảnh (như DriveStudio); với camera mắt cá, góc lệch trục quá 1,45 rad coi như không thấy. `all`: mọi lớp; `human`: Pedestrian, Cyclist; `vehicle`: Car, Truck, Tram.
- Nguồn không có box thì không ghi `instances/`, báo cáo có cảnh báo `boxes.missing`, `cuboids: none` (OmniRe bị khoá).

Chuỗi 0003 (file `train/`): 37 xe con, 1 xe tải, 1 rơ-moóc chuyển động (box ở khung 104–865), 8 xe con và 2 xe tải đỗ; **1 người đi bộ** (khung 629–662). Nhãn người đi bộ trong file là `pedestrian`: bản đầu của converter đổi nhãn `person` nên bỏ sót người (sửa 15/11/2026). Mọi chuỗi train đều có người: 0000 có 60 người đi bộ, 0009 có 85, 0010 có 73 (cộng người đi xe, nhãn `rider`). Kiểm tra bằng mắt 15/11/2026 trên 40 khung (bản tải về 1/5 khung, từ khung 400): box chiếu lên `image_00`, `image_01` và mắt cá `image_02` trùng xe trong ảnh.

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

`image_02`, `image_03` hiệu chỉnh bằng mô hình Mei (`xi`, `k1`, `k2`, `p1`, `p2`, `gamma1`, `gamma2`, `u0`, `v0`). Renderer 3DGUT của gsplat chỉ vẽ OpenCV fisheye, nên `drive2sim/core/camera.py` khớp bán kính pixel của hai mô hình: `γ1·m·(1 + k1m² + k2m⁴)` với `m = sin θ / (cos θ + xi)` bằng `fx·θ·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)`. Phương trình tuyến tính theo `fx` và `fx·k_i`, nên giải thẳng bằng bình phương tối thiểu trên 400 góc đều tới 1,45 rad (cho cùng nghiệm với `numpy.linalg.lstsq`); `fy = fx·γ2/γ1`.

Giới hạn (đo trên calib thật, ảnh 1400 px):
- Sai số phần xuyên tâm của phép khớp tới 1,45 rad: 0,004 px (`image_02`), 0,011 px (`image_03`); tới pi/2 tăng lên 0,10 px (`image_02`).
- Méo tiếp tuyến `p1`, `p2` bị bỏ (OpenCV fisheye không có). So với Mei chính xác trên điểm LiDAR thật (30 khung từ khung gốc 167, `check_kitti360_projection`): trung vị 0,10 px, lớn nhất 0,42 px (`image_02`); trung vị 0,17 px, lớn nhất 0,78 px (`image_03`).
- Camera nhìn rộng hơn 180 độ. Renderer UT của gsplat không vẽ được tia từ pi/2 trở ra, và đa thức chỉ khớp tới 1,45 rad (83 độ), nên vùng ảnh ngoài 1,45 rad **phải được che** khi train và đánh giá; loader `KITTI360CameraData` che vùng đó cho cả OmniRe và Street GS. Báo cáo kiểm tra ghi mục `camera.fisheye_fit` và `camera.fisheye_fov`.

## 6. Đã kiểm tra trên dữ liệu thật

Chuỗi `2013_05_28_drive_0003_sync`, 30 khung từ khung gốc 167 (`--start-frame 167 --num-frames 30`), ingest mất 1,2 s, 248 MB.

1. **Báo cáo kiểm tra** (`pose.cam0_to_world`): pose camera 0 dựng từ `poses.txt` và calib so với `cam0_to_world.txt` chính thức ở cả 30 khung: lệch lớn nhất `6,4e-7` (phần tử ma trận xoay) và 0,45 mm (tịnh tiến). `cam0_to_world.txt` chỉ ghi 7 chữ số có nghĩa (toạ độ cỡ 3000 m làm tròn tới 1 mm), nên đây là sai số làm tròn; ngưỡng báo lỗi là 5 mm.
2. **Chiếu LiDAR** (`python -m drive2sim.scripts.check_kitti360_projection --scene ... --raw ... --frame 0 15 --out <thư mục>`, công cụ của dự án, cần numpy và OpenCV của môi trường `drive2sim`). Script chiếu điểm Velodyne vào cả bốn camera theo hai đường: (a) calib và `poses.txt` chính thức theo quy ước `kitti360Scripts` (pinhole bằng `P_rect`, mắt cá bằng Mei đầy đủ, có `p1`, `p2`); (b) chỉ đọc thư mục cảnh (`ego_pose`, `extrinsics`, `intrinsics` pinhole hoặc OpenCV fisheye), rồi in lệch pixel giữa hai đường, tỉ số cạnh và ghi ảnh chồng điểm. Đo trên cả 30 khung của cảnh (chạy lại ngày 16/11/2026):
   - pinhole: `image_00`, `image_01` lệch trung vị `1,3e-11` px, lớn nhất `2e-10` px (sai số dấu phẩy động);
   - mắt cá (trong miền khớp 1,45 rad): `image_02` trung vị 0,10 px, lớn nhất 0,42 px; `image_03` trung vị 0,17 px, lớn nhất 0,78 px (phép khớp đa thức và bỏ méo tiếp tuyến, như mục 5);
   - nhiều khung (`--lidar-frame`): LiDAR khung 0 (khung gốc 167) đưa lên world rồi chiếu vào `image_00` ở khung 5 (khung gốc 192): hai đường lệch trung vị `8e-12` px, lớn nhất `1,2e-10` px, nên `ego_pose` của cảnh giữ đúng chuyển động trong `poses.txt` (so với `cam0_to_world.txt` là việc của báo cáo kiểm tra, mục 1);
   - tỉ số cạnh (không cần nhãn): điểm "mép" là điểm LiDAR gần hơn điểm kề nó trên cùng vòng quét ít nhất `max(1 m, 20 %)`; tỉ số là gradient ngang của ảnh (Sobel) trung bình dưới điểm mép chia cho trung bình dưới mọi điểm chiếu vào ảnh (mắt cá: chỉ điểm trong miền khớp). Gộp 30 khung: `image_00` 1,99, `image_01` 2,03, `image_02` 1,29, `image_03` 1,52; xoay camera lệch 1 độ quanh trục y thì còn 1,39, 1,36, 1,17, 1,33. Theo từng khung, pinhole luôn trên 1,3 và cao hơn khi xoay lệch ở cả 30 khung; mắt cá dao động mạnh (khung 0 `image_03` chỉ 1,14, khung 15 `image_02` 1,08, do cây trong bóng râm và vạch kẻ đường sáng), nên script lấy tỉ số gộp các khung đã chọn để quyết định (ngưỡng mặc định `--min-ratio 1.1`, thoát mã 1 nếu camera nào không vượt) và mắt cá vẫn nên xem thêm ảnh chồng điểm. Ảnh chồng điểm cho thấy cột điện, biển báo, cọc tiêu khớp ở cả bốn camera.
3. **Gốc world**: tool lấy hệ Velodyne ở khung đầu làm world (CONTRACTS mục 1), không dùng gốc world chính thức của KITTI-360; hai cách chỉ khác một phép biến đổi cứng cố định.

## 7. Chưa làm

- Train hồi quy OmniRe, Street GS trên cảnh KITTI-360: loader mắt cá và config `kitti360` đã có; lần chạy GPU sau thay đổi mask chờ người dùng review (ARCHITECTURE mục 7.1).
- `prep.smpl` (dựng người bằng SMPL): nhánh KITTI-360 dùng lại phép chiếu box của KITTI (`recon/preprocess/kitti360/human.py`), chạy 4D-Humans trên hai camera pinhole như DriveStudio làm với KITTI; tư thế SMPL nằm trong hệ thế giới nên camera mắt cá vẫn vẽ được người. Người chỉ thấy ở camera mắt cá thì không có SMPL (thành node biến dạng). **Chưa chạy thật** (cần GPU, 4D-Humans).
- Đoạn đường dài (cả chuỗi 1,4 km): chưa có cách chia và ghép đoạn.
- `timestamps.txt` thật của ảnh và LiDAR (khi nguồn có) mới được đọc cho ảnh của camera đầu; LiDAR dùng cùng mốc.
- Interpolate pose cho khung không có pose: không làm, như `kitti360Scripts`.

## 8. Mask thân xe trong trainer chung

Hai asset `drive2sim/assets/ego_masks/kitti360/image_02.png` và `image_03.png` che phần cứng gắn trên xe. Chúng là ảnh nhị phân 1400x1400: trắng (255) là bỏ, đen (0) là giữ. `outlines.json` lưu biên và các khung tham chiếu để tái tạo; README cạnh asset ghi cách làm. Mask đã được xem trên các khung 2, 257, 507, 759, 1009 của chuỗi 0003; chưa xác nhận trên các chuỗi khác. Bóng xe trên đường và vệt sáng mặt trời vẫn được giữ. Trên calib chuỗi 0003, mask thân xe bỏ thêm 8,8% (`image_02`) và 8,1% (`image_03`) số pixel trong vùng FOV 1,45 rad.

Loader chọn mask theo tên `image_02` hoặc `image_03` trong manifest, kể cả khi ingest chỉ hai camera và chúng trở thành chỉ số 0, 1. Mask được thu nhỏ bằng nearest rồi ghép với vùng tia ngoài 1,45 rad. Loss ảnh và depth, điểm lỗi FastGS, đánh giá và phép chiếu LiDAR dùng mask này; không dò lại trong ingest hoặc trong từng run. Pinhole và loader KITTI gốc giữ đường cũ.

PSNR test chỉ tính trên pixel hợp lệ. SSIM và LPIPS hiện tính trên hai ảnh đã tô đen vùng không hợp lệ, nên vẫn chịu ảnh hưởng của diện tích mask và biên cửa sổ; không phải metric chỉ lấy trung bình trên vùng hợp lệ. PSNR trong đường cong train hiện tính toàn ảnh. Thêm mask không đổi protocol này. Chưa train hồi quy KITTI-360 sau thay đổi; cần người dùng review code và mask trước khi chạy.
