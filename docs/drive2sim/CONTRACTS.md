# drive2sim: hợp đồng dữ liệu và giao diện

Tài liệu này quy định định dạng của dữ liệu đi qua tool, interface mà approach và stage phải theo, cùng quy tắc đặt tên và lệnh CLI (mục 10). Người thêm approach mới chỉ cần đọc tài liệu này cùng `ARCHITECTURE.md` mục 4. Người làm Studio đọc thêm mục 11 (API).

Đây là thiết kế đích (`ARCHITECTURE.md` mục 7). Mọi file do tool ghi đều có trường `schema_version`; thay đổi làm vỡ tương thích thì tăng số đầu của phiên bản. Khi đọc, tool chấp nhận file cùng loại và cùng số đầu; khoá lạ bị báo lỗi để bắt lỗi gõ sai trong file viết tay.

Các schema ở đây được cài trong `drive2sim/core/schema.py` (định danh ở `drive2sim/core/identifiers.py`, đọc ghi ở `drive2sim/core/files.py`). Test `tests/drive2sim/core/test_contracts_examples.py` đọc mọi ví dụ JSON và YAML có `schema_version` trong tài liệu này và kiểm tra chúng theo schema, nên tài liệu và code phải sửa cùng nhau.

Trạng thái: bản nháp, ngày 12/11/2026.

## 1. Quy ước chung

Mọi file và interface **của tool** (manifest, run, preview, interface L2–L4) theo **quy ước của NCore** (`docs/data/conventions.rst` trong repo NCore), vì quy ước này trùng proto `SensorsimService` và NCore là định dạng xuất khi nối NuRec hoặc CARLA. Dữ liệu chuẩn trong `data/<scene_id>/` thì giữ quy ước của script tiền xử lý từng dataset (mục 2.1); loader của dataset đó đổi về quy ước DriveStudio khi đọc. Codebase nào dùng quy ước khác thì tự đổi ở biên của nó.

| Mục | Quy ước |
|---|---|
| Đơn vị | Mét, giây, radian. Độ chỉ dùng trong giao diện. |
| Thời gian | Số nguyên micro giây. NCore lưu `uint64`; file của tool lưu số nguyên, tên trường kết thúc bằng `_us`. |
| Hệ `rig` | Thân xe: x tới trước, y sang trái, z lên trên; gốc ở giữa trục sau, trên mặt đất. |
| Hệ `world` | Hệ cục bộ của cảnh; pose đầu tiên của xe làm gốc. `T_world_world_global` nối với hệ toàn cầu (ECEF) nếu nguồn có. |
| Hệ camera | Theo OpenCV: x sang phải, y xuống dưới, z nhìn tới trước. |
| Toạ độ ảnh | u sang phải, v xuống dưới, gốc ở góc trên trái; pixel đầu tiên phủ khoảng `[0, 1]` trên cả hai trục, nên tâm của nó là `(0.5, 0.5)`. |
| Phép biến đổi | Ma trận 4×4 SE(3). **`T_a_b` đưa điểm từ hệ `a` sang hệ `b`**: `p_b = T_a_b · p_a`; ghép chuỗi: `T_a_c = T_b_c · T_a_b`. Ví dụ: extrinsic camera là `T_camera_rig`; quỹ đạo xe là `T_rig_world`. |
| Lưu phép biến đổi | Trong dữ liệu chuẩn: theo script tiền xử lý của dataset. Trong YAML/JSON của tool: `{translation: [x, y, z], rotation_wxyz: [w, x, y, z]}`. |
| Chỉ số khung | Số nguyên từ 0, theo dòng thời gian tham chiếu của cảnh (mục 2.2). |

Đổi sang quy ước của DriveStudio: DriveStudio lưu pose camera dạng `camtoworlds` (ma trận đưa điểm từ camera sang world), tức `T_camera_world` theo cách viết của NCore. Mọi chuyển đổi đi qua hàm trong `drive2sim/core`.

### 1.1 Lý do của các quy ước

| Quy ước | Lý do | Đánh đổi |
|---|---|---|
| File của tool theo quy ước NCore | Trùng proto `SensorsimService` và định dạng xuất NCore; giữ một quy ước cho mọi file của tool | Ngược chiều với nhiều tài liệu khác về `T_a_b`; phải đọc kỹ tên biến |
| Thời gian là số nguyên micro giây | So sánh bằng nhau chính xác; NCore và proto `SensorsimService` (`frame_start_us`, `frame_end_us`) cùng dùng micro giây | Nguồn ghi nano giây (ví dụ Argoverse 2) bị làm tròn |
| Gốc `world` ở pose đầu của xe | Toạ độ nhỏ, giữ độ chính xác khi train bằng số thực 32 bit | Phải dùng `T_world_world_global` nếu cần vị trí toàn cầu |
| Hệ `rig` x trước, y trái, z lên | Trùng NCore, Waymo, REP 103 của ROS | CARLA dùng hệ tay trái (y sang phải); converter CARLA và server phải đổi dấu trục y |
| Quaternion thứ tự w, x, y, z trong YAML/JSON | Trùng viser và `rot_0..3` trong PLY của 3DGS | scipy mặc định x, y, z, w; dùng hàm trong `core` để tránh nhầm |

## 2. Dữ liệu chuẩn

### 2.1 Cấu trúc thư mục

Dữ liệu chuẩn là **thư mục cảnh của DriveStudio**, đúng như script tiền xử lý có sẵn của từng dataset ghi ra (`drive2sim/recon/preprocess/<dataset>/`, hướng dẫn ở `docs/datasets/<dataset>.md`), cộng thêm các file của tool (`ARCHITECTURE.md` mục 2.2):

```
data/<scene_id>/
├── images/<khung>_<camera>.jpg  # ghi bởi script tiền xử lý của dataset
├── intrinsics/<camera>.txt
├── extrinsics/  ego_pose/  lidar_pose/   # tuỳ dataset, xem bên dưới
├── lidar/<khung>.bin
├── instances/frame_instances.json, instances_info.json
├── dynamic_masks/<khung>_<camera>.png    # nếu script sinh từ box
├── manifest.yaml                # tool ghi khi ingest
├── splits/<split_id>.json
├── reports/validation.json
├── reports/rig.json            # sơ đồ camera, quỹ đạo xe (mục 3.3)
├── reports/lidar_check.json    # LiDAR chiếu lên ảnh: số liệu (mục 3.4)
├── reports/lidar_overlay/      # ảnh LiDAR chồng lên ảnh, <khung>_<camera>.jpg (mục 3.4)
└── kaggle_scene.json            # chỉ có khi cảnh ingest trên Kaggle (mục 9.4)
```

Cảnh ingest bằng `d2s kaggle ingest` (mục 9.4) chỉ có file nhỏ trên máy: `manifest.yaml`, `reports/`, `splits/`, `kaggle_scene.json`, file text ở cấp gốc và pose, calib (`intrinsics/`, `extrinsics/`, `ego_pose/` hoặc `lidar_pose/`) để xem model bằng renderer gốc; ảnh, LiDAR, mask nằm trong output của notebook ingest trên Kaggle.

- `<khung>` là chỉ số 3 chữ số (`000`), `<camera>` là chỉ số camera của DriveStudio cho dataset đó.
- Cách lưu pose khác nhau giữa các dataset và **được giữ nguyên**, vì mỗi dataset đọc bằng loader riêng của nó (`drive2sim/recon/data/loaders/<dataset>.py`): Waymo, Argoverse, KITTI, nuPlan lưu `ego_pose/<khung>.txt` và `extrinsics/<camera>.txt`; PandaSet lưu `ego_pose/` và `extrinsics/<khung>_<camera>.txt`; nuScenes lưu `extrinsics/<khung>_<camera>.txt` và `lidar_pose/`. Bảng đầy đủ ở `ARCHITECTURE.md` mục 2.2.
- Nguồn mới không có loader riêng: converter ghi theo đúng khung thư mục của một dataset có sẵn rồi dùng loader của dataset đó (`ARCHITECTURE.md` mục 2.4).
- **KITTI-360** (`drive2sim/ingest/kitti360.py`, `docs/datasets/kitti360.md`) ghi theo khung KITTI của DriveStudio: xe trùng hệ LiDAR Velodyne (x tới trước, y trái, z lên); `lidar/<khung>.bin` chép nguyên file Velodyne (`float32` x, y, z, cường độ); `ego_pose/<khung>.txt` là ma trận 4×4 Velodyne → world, world là hệ Velodyne ở khung đầu của cảnh; `extrinsics/<camera>.txt` là camera → Velodyne (trục OpenCV), cố định; ảnh `images/<khung>_<camera>.png` chép nguyên PNG của nguồn. Thêm `frame_ids.txt` (số hiệu khung gốc theo thứ tự khung), `timestamps.txt` (micro giây mỗi khung) và `raw_source.json` (`{"raw_dir", "sequence"}`: thư mục KITTI-360 gốc, để `prep.ncore` của Instant NuRec đọc lại dữ liệu gốc; cảnh cũ không có file này thì dùng biến `D2S_KITTI360_ROOT`). Khi nguồn có box 3D, thêm `instances/instances_info.json`, `instances/frame_instances.json` và `dynamic_masks/{all,human,vehicle}/<khung>_<camera>.png` theo đúng định dạng script KITTI của DriveStudio (`docs/datasets/kitti360.md` mục 3.1).
- **`intrinsics/<camera>.txt` theo mô hình camera** ghi trong manifest (`cameras[].model`): `opencv_pinhole` là 9 số `fx fy cx cy k1 k2 p1 p2 k3` (như Argoverse, KITTI); `opencv_fisheye` là 8 số `fx fy cx cy k1 k2 k3 k4` (thứ tự của `cv2.fisheye`, `theta_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)`). Bốn số đầu luôn là `fx fy cx cy` (pixel, ảnh gốc). Hai dạng phân biệt bằng `model`, không bằng số phần tử. Camera mắt cá KITTI-360 (Mei) được khớp sang `opencv_fisheye` trên miền 0–1,45 rad (`drive2sim/core/camera.py`); ngoài miền đó sai số tăng dần. Converter chuyển gần đúng mô hình ống kính như vậy thì ghi vào manifest góc lệch trục lớn nhất mà mô hình mới còn lệch mô hình gốc không quá `FIT_TOLERANCE_PX` (0,5 px), tính từ calib của từng camera: `cameras[].max_angle` (rad, không bắt buộc, trong (0, π]; thêm sau `d2s-manifest/1.1` mà không tăng phiên bản; KITTI-360 ghi từ đầu ra phiên bản 3). Chuỗi 0003: `image_02` 1,678 (96,1 độ), `image_03` 1,615 (92,5 độ). Loader che theo góc nhỏ nhất giữa trường này, vùng đen dò từ ảnh và giới hạn của renderer (3DGUT, 3DGEER: 88 độ, `drive2sim/stages/train/drivestudio.py` `RENDERER_MAX_ANGLE`); không có trường này là không giới hạn ngoài miền đa thức ống kính còn tăng. So sánh manifest giữa hai tài khoản Kaggle (mục 9.4) bỏ qua trường này.
- Thư mục cảnh chỉ chứa những gì có từ nguồn. Mask bầu trời và SMPL do stage `prep.*` sinh ra là artifact trong `cache/` (mục 5), không ghi vào đây.

Mask thân xe KITTI-360 là asset của loader trong package, không phải dữ liệu nguồn hay artifact `prep.masks_segformer`. `image_02.png` và `image_03.png` có kích thước 1400x1400: 255 là vùng bỏ qua, 0 là vùng giữ lại. Tên asset theo `camera_id` trong manifest, không theo chỉ số camera đã được ingest đánh lại. `available.egocar_mask` vẫn mô tả mask có từ nguồn.

### 2.2 `manifest.yaml`

```yaml
schema_version: d2s-manifest/1.1
scene_id: nuscenes_scene-0061
source:
  dataset: nuscenes               # waymo | nuscenes | nuplan | argoverse2 | pandaset | kitti | kitti360 | physicalai | carla | colmap | custom
  sequence_id: scene-0061
  converter: drivestudio.nuscenes # converter đã dùng: script tiền xử lý DriveStudio cho nuScenes
  converter_version: "0.1.0+96876ad"   # <phiên bản drive2sim>+<commit>[.dirty-<hash8>] lúc ingest
  output_version: 1               # phiên bản đầu ra của converter; tăng tay khi đầu ra đổi
  data_license: "CC BY-NC-SA 4.0"
track: pinhole                    # pinhole | fisheye; fisheye khi có ít nhất một camera mắt cá
cameras:
  - camera_id: CAM_FRONT
    model: opencv_pinhole         # opencv_pinhole | opencv_fisheye | ftheta | ideal_pinhole
    resolution: [1600, 900]
    shutter: GLOBAL
lidars: [LIDAR_TOP]
timeline:                         # dòng thời gian tham chiếu cho chỉ số khung
  reference_sensor: CAM_FRONT
  num_frames: 39
available:
  lidar: true
  cuboids: gt                     # gt | autolabel | none
  camera_labels: []               # nhãn theo ảnh có sẵn trong nguồn: depth, segmentation, ...
  egocar_mask: false
```

- `timeline`: khung thứ `i` là khung thứ `i` của `reference_sensor`. Ảnh của camera khác được ghép với khung gần nhất về thời gian.
- `available` là cơ sở để khoá approach (mục 4.2).
- Manifest `1.2` thêm `available.human_frames`: các đoạn `[đầu, cuối]` có box người, tính cả hai đầu. `[]` nghĩa là không có người; trường vắng hoặc `null` nghĩa là chưa biết. Với cảnh cũ có dữ liệu local, `Workdir` đọc lại `instances/` trong bộ nhớ. Nhãn hỏng giữ trạng thái chưa biết. Split không giao đoạn có người thì `smpl: auto` chọn `none`; chưa biết thì vẫn chọn stage SMPL.
- `cameras` xếp đúng thứ tự chỉ số camera của DriveStudio cho dataset đó, vì `images/<khung>_<camera>.jpg` và config dataset dùng chỉ số này; stage train đổi `camera_id` của split sang chỉ số theo thứ tự đó.
- `source.dataset` chọn loader và config dataset có sẵn của DriveStudio khi train.
- `source.converter_version`: `<phiên bản drive2sim>+<commit>`; khi mã converter (`drive2sim/ingest/`, `drive2sim/recon/preprocess/`, `drive2sim/core/`) có thay đổi chưa commit thì thêm `.dirty-<hash8>` (8 ký tự hex đầu của sha256 trên diff so với `HEAD` và các file mới chưa theo dõi), ví dụ `0.1.0+2867e48.dirty-5b0e1796`. Trường này nằm trong khoá cache của mọi stage nhận cảnh, nên hai lần ingest bằng hai bản code chưa commit khác nhau không dùng chung cache.
- `source.output_version` (thêm từ `1.1`, không bắt buộc): phiên bản đầu ra của converter lúc ingest, số nguyên ≥ 1 tăng tay **chỉ khi dữ liệu converter ghi ra đổi** (`drive2sim/ingest/manifest.py`: `KITTI360_OUTPUT_VERSION` hiện là `2` từ lần viết lại khớp Mei; `DRIVESTUDIO_OUTPUT_VERSION` dùng chung cho mọi converter `drivestudio.<dataset>`, hiện là `1`). Manifest `1.0` không có trường này vẫn đọc được và được coi là `1` (riêng `converter_version` ghi commit `188fb9b`, đã có khớp Mei mới, được coi là `2` với KITTI-360). Cảnh có phiên bản khác phiên bản hiện tại của converter là **cảnh cũ**: `scene validate`, Studio, `approach check`, `run start` cảnh báo (không chặn), `kaggle ingest` dùng notebook mới (mục 9.4). Converter không do tool quản lý (cảnh tự làm) không có phiên bản, không bị xét. Trường này không nằm trong khoá cache (mục 5.2) hay tên dataset cảnh trên Kaggle (mục 10.1), để cache và dataset đã có vẫn dùng được.
- Thay đổi từ `0.1`: bỏ phần `ncore` (dữ liệu chuẩn không còn là NCore); file `0.x` không đọc được nữa.
- `cameras[].model` ghi đúng mô hình trong nguồn, nên có thể là `ideal_pinhole` (với nguồn NCore); approach chỉ khai báo nhận ba mô hình chuẩn (mục 10.3), và `ideal_pinhole` được coi là `opencv_pinhole` với hệ số méo bằng 0. Mỗi camera ghi mô hình riêng; một cảnh có thể có cả camera pinhole lẫn mắt cá (KITTI-360). `track` của cảnh là `fisheye` khi có ít nhất một camera thuộc track fisheye. Engine vẫn kiểm từng camera của split với mô hình approach nhận (mục 4.2), nên approach chỉ vẽ pinhole bị khoá trên split có camera mắt cá.
- Các giá trị trong ví dụ chỉ minh hoạ định dạng.

## 3. Split và báo cáo kiểm tra

### 3.1 `splits/<split_id>.json`

```json
{
  "schema_version": "d2s-split/0.1",
  "split_id": "every8_allcams",
  "scene_id": "nuscenes_scene-0061",
  "strategy": "every_nth_frame:8",
  "frame_range": [0, 38],
  "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
  "train": [[0, "CAM_FRONT"], [0, "CAM_FRONT_LEFT"], [1, "CAM_FRONT"]],
  "test":  [[8, "CAM_FRONT"], [8, "CAM_FRONT_LEFT"], [8, "CAM_FRONT_RIGHT"]]
}
```

Mỗi phần tử là cặp `[frame_index, camera_id]`. Một cặp không được nằm ở cả `train` và `test`.

`strategy` lấy một trong:
- `every_nth_frame:N`: các khung N, 2N, 3N… (tính từ khung đầu của `frame_range`) làm test cho mọi camera trong `cameras`; các khung còn lại, kể cả khung đầu, làm train. Trùng với `test_image_stride` của DriveStudio (`split_train_test` trong `drive2sim/recon/data/driving_dataset.py`).
- `custom`: danh sách `train` và `test` bất kỳ.

Approach khai báo các `strategy` mà codebase của nó hỗ trợ (`requires.split_strategies`, mục 4.1). Approach dòng DriveStudio hiện chỉ hỗ trợ `every_nth_frame`.

### 3.2 `reports/validation.json`

```json
{
  "schema_version": "d2s-report/0.1",
  "scene_id": "nuscenes_scene-0061",
  "created_utc": "2026-11-12T08:00:00Z",
  "summary": {"errors": 0, "warnings": 1},
  "checks": [
    {"check_id": "timestamps.camera_sync", "severity": "warning",
     "message": "Lệch thời gian giữa camera lớn hơn 5 ms", "camera_id": "CAM_BACK", "frame_index": 12,
     "details": {"max_offset_us": 6200}}
  ]
}
```

`severity` lấy một trong: `error` (không chạy được), `warning`, `info`.

`converter.outdated` (`warning`): cảnh được ingest bằng converter cũ (mục 2.2), `details` là `{"output_version", "current_output_version"}`; cảnh vẫn chạy được.

### 3.3 `reports/rig.json`

Sơ đồ camera trên xe và quỹ đạo xe, nhìn từ trên xuống, cho Studio (SPEC mục 8A). Ghi bởi `drive2sim/recon/tools/scene_rig.py` (cần PyTorch): nó dựng `DrivingDataset` của DriveStudio ở chế độ viewer (đọc pose, calib, không đọc ảnh, LiDAR) và lấy pose camera, intrinsics mà loader của dataset đưa cho trainer, **không** tự đọc file pose của từng dataset (mỗi dataset lưu khác nhau), nên sơ đồ đúng bằng dữ liệu train và chạy được với cảnh ingest trên Kaggle. `d2s scene ingest` (và ingest trong Studio) ghi nó sau khi ingest; lỗi ở bước này chỉ là cảnh báo. Cảnh có trước: `d2s scene rig <scene_id>` hoặc nút tạo ở tab Camera của trang cảnh. Notebook ingest trên Kaggle chạy `d2s scene ingest`, và file nằm trong `reports/` nên được tải về cùng file nhỏ (mục 9.4).

Hệ toạ độ: `x` tới, `y` trái, `z` lên, mét; gốc là tâm các camera ở khung đầu. Trục lên là trung bình hướng lên (`-Y` của camera OpenCV) của mọi camera ở mọi khung. Trục tới là hướng xe chạy: trung bình (trên vòng tròn) góc giữa mỗi đoạn đường dài hơn 0,3 m và hướng nhìn của camera tham chiếu (`timeline.reference_sensor` nếu là camera, không thì camera đầu), nên đúng cả khi cảnh không có camera nhìn thẳng phía trước; sai số cỡ 2 độ (cảnh thật: camera trước nuScenes −1,2 độ, KITTI-360 `image_00` 1,8 độ). Xe đứng yên thì lấy hướng camera tham chiếu (`forward_source: "camera"`).

```json
{
  "version": 1, "scene_id": "nuscenes_scene-0061",
  "frame": "x forward, y left, z up (m); origin: camera centre at frame 0",
  "reference_camera": "CAM_FRONT", "forward_source": "motion",
  "cameras": [{"camera_id": "CAM_FRONT", "model": "opencv_pinhole", "position_m": [0.75, -0.01, -0.05],
               "yaw_deg": -1.2, "pitch_deg": -2.2, "hfov_deg": 64.6}],
  "trajectory": [{"frame": 0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}]
}
```

`yaw_deg` dương là sang trái; vị trí camera tính so với tâm các camera ở khung đầu. `hfov_deg`: pinhole từ tiêu cự; mắt cá, F-theta là góc giữa hai pixel hợp lệ xa nhất trên hàng giữa ảnh theo tia của loader (camera mắt cá KITTI-360: trong góc che của loader, ví dụ 191,8 độ với `image_02` của cảnh ghi `max_angle`). `trajectory`: tâm các camera và hướng xe theo từng khung.

### 3.4 `reports/lidar_check.json`, `reports/lidar_overlay/`

LiDAR chiếu lên ảnh để kiểm tra calib, pose (`drive2sim/recon/tools/lidar_overlay.py`, cần PyTorch và ảnh trên máy). Phép chiếu là của loader DriveStudio (`lidar_depth_maps`, đúng hình học khi train). Ghi sau ingest cùng `rig.json` (`drive2sim/ingest/reports.py`; notebook ingest trên Kaggle chạy nó, máy tải về `lidar_check.json` cùng file nhỏ, không có ảnh); cảnh có trước: `d2s scene lidar-check <scene_id>` hoặc nút ở tab Kiểm tra. Trên 3 khung (đầu, giữa, cuối):

- `lidar_overlay/<khung>_<camera>.jpg`: điểm LiDAR tô màu độ sâu (thang của `depth_visualizer`) trên ảnh loader nạp; LiDAR thưa (dưới 40 000 điểm mỗi megapixel) vẽ điểm to ra 1 pixel. Để người xem, **không** phải bước duyệt.
- `lidar_check.json`: `{"version", "scene_id", "frames", "cameras": [{"camera_id", "frames", "points_per_frame", "points_per_megapixel", "median_depth_m", "p95_depth_m"}], "overlays": [{"frame", "camera_id", "file"}]}`.

`validate_scene` đọc hai file này và `rig.json`: `projection.camera_sparse` (cảnh báo, mật độ dưới 2 % camera nhiều điểm nhất), `projection.depth_range` (cảnh báo, độ sâu trung vị ngoài 1–150 m), `projection.camera_tilt` (cảnh báo, `tilt_deg` của camera quá 60 độ), `projection.done` (thông tin); chưa có `lidar_check.json` thì `projection.not_run`. Không bắt được tự động: lệch nhỏ vài pixel, và calib hai camera bị đổi chỗ cho nhau; hai lỗi này chỉ thấy trên ảnh chồng (ARCHITECTURE mục 7.1).

## 4. Approach

### 4.1 File khai báo

`drive2sim/approaches/<approach_id>.yaml`. Approach là danh sách stage chạy lần lượt; các khâu thay thế được cho nhau là *mục cấu hình*, quyết định trước khi chạy (`ARCHITECTURE.md` mục 2.5).

```yaml
schema_version: d2s-approach/0.3
approach_id: omnire
name: "OmniRe"
version: 0.1.0
codebase: drivestudio              # drivestudio | external:<tên>
level: L1                          # L1 | L2 | L3 | L4
requires:
  camera_models: [opencv_pinhole, opencv_fisheye, ftheta]   # nhận được với ít nhất một tổ hợp lựa chọn
  cuboids: required                # required | optional | none
  split_strategies: [every_nth_frame]

groups:                            # nhóm mục cấu hình trên Studio, theo thứ tự hiện (tuỳ chọn)
  core: {label: "Cách dựng"}
  speedup: {label: "Thuật toán phụ trợ", help: "Thuật toán gắn thêm vào quá trình train"}

options:                           # mục cấu hình
  init:
    choices: [init.lidar, init.depth]
    default: auto                  # auto: có LiDAR → init.lidar, không có → init.depth
  renderer:
    choices: [3dgs, 3dgut, 3dgeer]
    default: 3dgs
    camera_models:                 # mô hình camera từng lựa chọn nhận, cùng thứ tự với choices (tuỳ chọn)
      - [opencv_pinhole]
      - [opencv_pinhole, opencv_fisheye, ftheta]
      - [opencv_pinhole, opencv_fisheye, ftheta]
  campose:
    choices: [false, true]
    default: false
    label: "CamPose"               # tên hiển thị trên Studio (tuỳ chọn)
    help: "Cho model tự chỉnh lại vị trí và hướng camera trong lúc train."   # icon (i) cạnh tên mục cấu hình
    labels: ["Tắt", "Bật"]         # tên của từng lựa chọn, cùng thứ tự với choices
    descriptions: ["Dùng nguyên pose của dataset", "Học thêm độ lệch pose cho từng ảnh"]
    group: core
  fastgs:                          # thuật toán phụ trợ: mỗi thuật toán là một mục bật/tắt riêng
    choices: [false, true]
    default: false
    group: speedup

constraints:                       # tổ hợp không hợp lệ
  - when: {renderer: 3dgeer}
    forbid: {campose: true}
    reason: "3DGEER không tối ưu được pose camera"

presets:                           # tổ hợp đặt sẵn; chỉ ghi các mục cấu hình khác giá trị mặc định
  paper: {init: init.lidar}
  no_lidar: {init: init.depth}
  fisheye: {renderer: 3dgeer}

preset_help:                       # mô tả ngắn của preset, hiện trên Studio (tuỳ chọn)
  paper: "Cấu hình gốc của DriveStudio"

stages:                            # chạy lần lượt; ${...} được thay trước khi chạy
  - {stage: prep.select, params: {downscale: 2}}
  - {stage: prep.masks_segformer, params: {dynamic: true}}
  - {stage: prep.smpl}
  - {stage: "${options.init}"}
  - stage: train.drivestudio
    params:
      method: omnire
      renderer: "${options.renderer}"
      campose: "${options.campose}"
      fastgs: "${options.fastgs}"
  - {stage: eval.standard}
  - {stage: export.ply_world}
```

Ví dụ trên là dạng đích: stage `init.*` chưa có (ARCHITECTURE mục 4.3), nên `drive2sim/approaches/omnire.yaml` hiện chưa có mục `init`; video do `train.py` ghi, chưa có stage `render.videos`.

- Một stage dùng được artifact của bất kỳ stage nào đứng trước. Engine nối theo kiểu artifact: stage cần `points_init` nhận artifact `points_init` gần nhất phía trước.
- `${options.<tên mục>}` được thay bằng lựa chọn cụ thể trước khi chạy. Sau khi thay, danh sách stage là cố định; không có điều kiện, nhánh song song hay vòng lặp lúc đang chạy.
- `groups`, và `label`, `help`, `labels`, `descriptions`, `group` của mục cấu hình, và `preset_help` chỉ dùng để hiển thị (thêm từ `0.2`; file `0.1` vẫn đọc được). Studio hiện các mục cấu hình theo nhóm; giải thích của mục cấu hình, của từng lựa chọn, của preset và lý do một lựa chọn bị khoá nằm trong icon (i), không hiện thành dòng chữ trên trang. Thuật toán phụ trợ (FastGS, và sau này các thuật toán tăng tốc khác) đặt mỗi thuật toán thành một mục cấu hình bật/tắt riêng trong nhóm `speedup`; tổ hợp chưa có file config thì khai báo trong `constraints`. Viết tiếng Việt như `reason` của ràng buộc. Thêm approach hay lựa chọn mới không cần sửa Studio: thiếu các trường này thì Studio hiện tên mục cấu hình và giá trị thô.
- `retired_options` (tuỳ chọn, thêm sau `0.3` mà không tăng phiên bản): mục cấu hình đã gỡ khỏi approach, cùng dạng với `options`. Engine không đọc; Studio dùng nhãn và `default` cũ để hiện cấu hình của run cũ còn ghi mục đó (ẩn khi run dùng giá trị mặc định). Tên không được trùng mục trong `options`. Ví dụ `omnire.yaml` giữ `densify` (MCMC, gỡ 19/11/2026). Ô cấu hình trên Studio luôn hiện `renderer` đầu tiên, kể cả khi là mặc định, trừ khi preset của run đã đặt renderer.
- **Loại camera (track).** Approach không ghi track. Track suy ra từ mô hình camera: `opencv_pinhole`, `ideal_pinhole` là track `pinhole`; `opencv_fisheye`, `ftheta` là track `fisheye`. `requires.camera_models` là mọi mô hình approach nhận với **ít nhất một** tổ hợp lựa chọn. Lựa chọn chỉ nhận một phần thì ghi trong `camera_models` của mục cấu hình đó, mỗi lựa chọn một danh sách, chỉ dùng mô hình có trong `requires.camera_models` (thêm từ `0.3`). Ví dụ OmniRe: 3DGS gốc chỉ vẽ pinhole, 3DGUT và 3DGEER vẽ thêm fisheye, F-theta (`drive2sim/recon/trainers/base.py`, `camera_rasterization_kwargs`). Vậy track của từng preset suy ra từ lựa chọn của preset đó: `paper` chỉ pinhole, `geer` cả hai. API Studio (`GET /api/approaches`) và `d2s approach list`, `show` trả thêm `camera_support`: `{"tracks", "camera_models", "default": {...}, "presets": {tên: {...}}}` (`resolve.camera_support`). File `0.1`, `0.2` có trường `track` vẫn đọc được; trường đó bị bỏ qua.
- Preset chỉ là cách điền nhanh các mục cấu hình. `run.json` ghi tên preset (`approach.preset`) chỉ khi cấu hình cuối đúng bằng preset đó; người dùng chọn preset rồi chỉnh một mục cấu hình khác đi (Studio, hoặc `--preset` kèm `--set` ở CLI) thì `preset` là `null`. Studio làm mờ lựa chọn vi phạm ràng buộc với các mục cấu hình đang chọn, kèm mục cấu hình cần đổi và `reason`.
- Tham số cuối cùng của một stage được gộp theo thứ tự: mặc định của stage → `params` trong file → preset → lựa chọn của người dùng (Studio hoặc `--set` của CLI).
- `codebase`: `drivestudio` là trainer DriveStudio trong `drive2sim/recon/`, chạy trong môi trường chính. `external:<tên>` là codebase ngoài, môi trường riêng (mục 9). Thư viện `gsplat-geer` cung cấp renderer cho DriveStudio, không phải một codebase của approach.
- `train.drivestudio` chọn **một file config có sẵn** trong `drive2sim/configs/methods/` theo năm tham số `method`, `renderer`, `campose`, `fastgs`, `fastergs` (bảng `CONFIG_FILES` trong `drive2sim/stages/train/drivestudio.py`), không tự sửa tham số train. Ví dụ `streetgs` + `3dgeer` + `fastgs: true` dùng `streetgs_geer_fastgs.yaml`. Tổ hợp chưa có file config bị từ chối khi kiểm tra tham số, và file approach khai báo `constraints` để Studio, CLI khoá sẵn các tổ hợp đó.

### 4.2 Điều kiện chạy

Engine kiểm tra trước khi chạy. Approach hoặc một lựa chọn bị khoá, kèm lý do, khi:
- lựa chọn cần LiDAR (ví dụ `init.lidar`) mà `available.lidar` là `false`;
- `cuboids: required` mà `available.cuboids` là `none`;
- mô hình camera trong split không nằm trong `requires.camera_models`, và danh sách stage không có `prep.undistort`;
- lựa chọn đang chọn có `camera_models` không chứa mô hình của một camera trong split, và danh sách stage không có `prep.undistort` (ví dụ renderer `3dgs` trên cảnh camera `ftheta`); lựa chọn đó bị khoá và `auto` bỏ qua nó;
- `strategy` của split không nằm trong `split_strategies`;
- tổ hợp lựa chọn vi phạm một mục trong `constraints`.

Studio và CLI dùng chung bước kiểm tra này (`drive2sim/engine/resolve.py`). Quy tắc chi tiết:
- Giá trị của mục cấu hình gộp theo thứ tự: `default` → preset → lựa chọn của người dùng. Mục còn `auto` lấy lựa chọn **đầu tiên** trong `choices` không bị khoá vì dữ liệu, nên thứ tự `choices` là thứ tự ưu tiên.
- Stage có `uses_lidar: true` nghĩa là **cần** LiDAR; lựa chọn là stage đó bị khoá khi `available.lidar` là `false`, và stage cố định (không do mục cấu hình chọn) cần LiDAR thì cả approach bị khoá trên cảnh không có LiDAR. `train.drivestudio` có `uses_lidar: true` vì mọi config OmniRe, Street GS hiện có khởi tạo Gaussian từ LiDAR (nền từ `model.Background.init.from_lidar`, xe và người từ điểm LiDAR trong box). Hỗ trợ cảnh không có LiDAR sẽ là approach hoặc preset riêng (SPEC mục 5).
- Ngoài LiDAR và camera, stage có thể tự khai báo điều kiện riêng bằng `Stage.unavailable_reason(manifest, split=None)` (trả lý do hoặc `None`): lựa chọn là stage đó bị khoá kèm lý do và `auto` bỏ qua nó; stage cố định không thoả thì cả approach bị khoá. Ví dụ `prep.smpl_release` chỉ chạy với cảnh có trong bản SMPL DriveStudio phát hành; mục `smpl` của OmniRe (`choices: [prep.smpl_release, prep.smpl, none]`, `default: auto`) nhờ đó lấy bản phát hành khi có, không thì tự tính.
- Một ràng buộc bị vi phạm khi mọi cặp trong `when` và mọi cặp trong `forbid` đều khớp với lựa chọn. So sánh cả kiểu: `1` không khớp `true`.
- Mô hình `ideal_pinhole` trong manifest được coi là `opencv_pinhole` khi so với `camera_models`.
- Engine gom **mọi** lý do bị khoá rồi báo một lần (mã lỗi `blocked`, mã thoát `3`). Preset, mục cấu hình, giá trị hay tham số stage không tồn tại, stage chưa đăng ký, hoặc một đầu vào artifact không có stage nào phía trước tạo ra, là lỗi yêu cầu (`invalid_request`, mã thoát `2`).
- Studio lấy danh sách lựa chọn bị khoá theo từng mục cấu hình để làm mờ, kèm lý do, trước khi người dùng bấm chạy.

### 4.3 Khai báo dùng LiDAR

Mỗi stage khai báo `uses_lidar`. Engine ghi danh sách stage dùng LiDAR vào `run.json`; Studio đọc để hiện "LiDAR dùng ở khâu nào".

## 5. Stage, artifact và cache

### 5.1 Interface của stage

```python
class Stage(ABC):                      # lớp gốc; lớp con khai báo thuộc tính và cài run
    stage_id: str                      # "<nhóm>.<tên>", ví dụ "init.depth" (mục 10.2)
    version: str                       # tăng khi đổi hành vi
    cacheable: bool                    # True: kết quả vào cache/ dùng chung; False: vào thư mục run
    uses_lidar: bool
    inputs: dict[str, str]             # tên đầu vào → loại artifact, hoặc "scene" / "split"
    optional_inputs: frozenset[str]    # tên trong inputs có thể thiếu (mặc định rỗng)
    outputs: dict[str, str]            # tên đầu ra → loại artifact
    Params: type[pydantic.BaseModel]

    def run(self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel) -> dict[str, Path]:
        """Ghi đầu ra vào ctx.output_dir và trả về đường dẫn của từng đầu ra."""

class StageContext(Protocol):
    output_dir: Path
    scene: Scene                       # thư mục dữ liệu chuẩn + manifest
    split: Split
    def progress(self, fraction: float, message: str = "") -> None: ...
    def log_metric(self, name: str, value: float, step: int) -> None: ...
    def report_peak_vram(self, megabytes: int) -> None: ...  # stage chạy GPU ở tiến trình con báo VRAM đỉnh; engine ghi giá trị lớn hơn vào run.json
```

Stage đăng ký bằng `@register_stage("<stage_id>")`. Engine lo cache, file `done`, thư mục `.partial` và `artifact.json`; stage chỉ ghi file vào `output_dir`.

- `inputs` của `run` chỉ chứa artifact: tên đầu vào → thư mục của artifact. Cảnh và split đọc từ `ctx`. Khai báo `"scene"` hoặc `"split"` trong `inputs` của lớp nghĩa là kết quả phụ thuộc cảnh hoặc split, nên `scene_id` hoặc `split_id` vào khoá cache.
- Decorator kiểm tra khai báo ngay lúc import và từ chối: tên lớp sai mẫu `<Nhóm><Tên>Stage`; `version` sai dạng `MAJOR.MINOR.PATCH`; `cacheable` khác quy tắc của nhóm (`true` với `prep`, `init`; `false` với nhóm khác, mục 5.2); loại artifact không có trong bảng mục 5.4; không có **đúng một** đầu ra (thư mục kết quả của stage là một artifact, `artifact.json` có một `artifact_type`); `Params` không phải lớp Pydantic; `stage_id` đã được lớp khác đăng ký.
- Cài trong `drive2sim/engine/stage.py` (interface) và `drive2sim/engine/registry.py` (đăng ký, nạp mọi stage trong `drive2sim/stages/`).
- `optional_inputs`: đầu vào artifact có thể thiếu. Nếu không stage nào phía trước tạo ra loại artifact đó, engine bỏ qua thay vì báo lỗi, và `inputs` của `run` không có tên đó. Ví dụ `train.drivestudio` nhận `masks` và `smpl` khi approach có `prep.masks_segformer`, `prep.smpl`; thiếu thì tắt `load_sky_mask`, `load_smpl` của DriveStudio.

### 5.2 Nơi ghi kết quả

| Loại stage | Nơi ghi | Dùng lại khi |
|---|---|---|
| `cacheable: true` (`prep.*`, `init.*`) | `cache/<scene_id>/<stage_id>@<version>-<hash8>/` | Đã có thư mục cùng tên |
| `cacheable: false` (`train.*`, `eval.*`, `render.*`, `export.*`) | `runs/.../<run_id>/d2s/stages/<stage_id>/` | Có file `done` trong thư mục đó |

`<hash8>` là 8 ký tự hex đầu của sha256 trên JSON chuẩn hoá của {tham số đã gộp, khoá cache của các artifact đầu vào}. Không băm nội dung dữ liệu: dữ liệu chuẩn sau ingest không đổi, `scene_id` cùng `source.converter_version` trong manifest là đủ để nhận diện. Thư mục đang ghi mang hậu tố `.partial`, chỉ đổi tên khi ghi xong.

Riêng `train.*`: chạy tiếp bằng `checkpoint_resume.pth` (model, optimizer, grad scaler) trong thư mục run, truyền qua `--resume_from`; file `done` chỉ ghi khi train xong.

### 5.3 `artifact.json`

```json
{
  "schema_version": "d2s-artifact/0.2",
  "artifact_type": "masks",
  "cache_key": "nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e",
  "producer": {"stage_id": "prep.masks_segformer", "version": "0.1.0", "code_commit": "286ad9b"},
  "params": {"dynamic": true},
  "inputs": ["scene:nuscenes_scene-0061"],
  "created_utc": "2026-11-12T08:10:00Z",
  "files": ["sky_masks"],
  "meta": {"model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"},
  "duration_s": 810.4
}
```

- `duration_s`: thời gian stage chạy để tạo ra kết quả này. Run sau dùng lại kết quả từ cache chép số này vào `stages[].reused_duration_s` của `run.json` (mục 6.2), để tính tổng thời gian dựng cảnh. Thêm từ `0.2`; cache tạo trước đó không có trường này.

### 5.4 Loại artifact

| `artifact_type` | Nội dung |
|---|---|
| `scene_subset` | Kết quả của `prep.select`: danh sách camera, đoạn khung, độ phân giải đã chọn |
| `masks` | Thư mục có `sky_masks/<khung>_<camera>.png` và (khi `dynamic` bật) `fine_dynamic_masks/{all,human,vehicle}/<khung>_<camera>.png`, đúng tên DriveStudio. Do `prep.masks_segformer` ghi |
| `smpl` | Thư mục có `humanpose/smpl.pkl` (định dạng của `drive2sim/recon/preprocess/smpl/run.py`) cùng file trung gian trong `humanpose/temp/`, `humanpose/postprocess/`. Do `prep.smpl` ghi |
| `ncore_clip` | Clip NCore V4 của đoạn khung trong split (Instant NuRec đọc): file `<sequence_id>.json` và các store `.itar` do converter ghi, cùng `clip.json` (`clip`, `frame_ids` của clip, `scene_frame_ids`, `scene_frames`, `sensors` (`camera_id` → tên sensor NCore), `fisheye_model`, `lidar`). Do `prep.ncore` ghi |
| `depth_prior` | Depth metric theo camera và khung, mỗi ảnh một file `.npy` `float32`, mét; kèm tên model trong `artifact.json` |
| `points_init` | `points_init.npz`: `points` `float32[M,3]` trong hệ `world`, `colors` `uint8[M,3]`, `track_ids` (tuỳ chọn) |
| `model` | Thư mục log của codebase; với DriveStudio có `checkpoint_final.pth`, `config.yaml`, `metrics/`, `videos/`, `train_stats.json`, `cameras.json` (mục 6.1). Codebase khác cung cấp kết quả qua định dạng chung của các stage sau |
| `eval_report` | Thư mục chứa `summary.json` (mục 6.3) và metric gốc của codebase |
| `videos` | Video MP4 H.264 |
| `export_ply` | PLY 3DGS trong hệ `world` |
| `preview` | Gói xem trước cho Spark (mục 7) |

## 6. Run

### 6.1 Cấu trúc

Mỗi stage chỉ ghi vào thư mục của nó (mục 5.1). Thư mục log của DriveStudio vì vậy là thư mục của stage `train.drivestudio`: stage đặt `output_root`, `project`, `run_name` của `train.py` sao cho `os.path.join(output_root, project, run_name)` trùng thư mục đó.

```
runs/<scene_id>/<approach_id>/<run_id>/
└── d2s/
    ├── run.json
    ├── env.json
    ├── log.txt                          # log của engine, traceback của stage lỗi
    ├── pid                              # pid của tiến trình đang chạy run; xoá khi xong
    ├── cancel                           # có file này là có yêu cầu huỷ; engine xoá khi đã huỷ
    ├── kaggle.json                      # run chạy trên Kaggle: notebook, lần đẩy, dataset, commit (mục 9.2)
    ├── colab.json                       # run chạy trên Colab: phiên, job, lần đẩy, commit (mục 9.3)
    └── stages/
        ├── train.drivestudio/           # thư mục log của DriveStudio
        │   ├── scene_view/000/          # cảnh ảo bằng symlink: thư mục cảnh + mask, SMPL từ cache
        │   ├── config.yaml              # config đầy đủ sau khi gộp
        │   ├── metrics.json             # metric trong lúc train
        │   ├── metrics/images_test_*.json
        │   ├── videos/ images/ configs_bk/ backup/
        │   ├── checkpoint_<bước>.pth     # chỉ model, sau mỗi lần lưu; train.py xoá khi train xong (từ 21/11/2026), trừ khi tham số keep_step_checkpoints
        │   ├── metrics_steps/test_<bước>.json  # đo test giữa chừng (từ 21/11/2026, tham số eval_at; mặc định nửa số bước): psnr, ssim, lpips, per_camera (chỉ số camera như metrics/), num_gaussians, train_time_s
        │   ├── checkpoint_resume.pth    # model + optimizer + grad scaler; xoá khi train xong
        │   ├── checkpoint_final.pth
        │   ├── train_stats.json         # thời gian train thực, VRAM, số Gaussian, iteration; ghi cùng mỗi checkpoint
        │   ├── cameras.json             # pose và nội tham số camera gốc theo khung (mục 7)
        │   ├── log.txt                  # stdout và stderr của train.py
        │   ├── artifact.json
        │   └── done
        ├── eval.standard/               # summary.json, artifact.json, done
        └── export.ply_world/            # scene.ply, scene.json, cameras.json, artifact.json, done
```

Gốc thư mục run chỉ có `d2s/`, nên file của tool không trùng tên với file của codebase.

**Video trong `train.drivestudio/videos/`.** `train.py` gọi `do_evaluation` (`drive2sim/recon/tools/eval.py`) sau khi train xong; hầu hết video do model render ra, chỉ file đuôi `_gt_rgbs` là ảnh camera thật. Tên file có số bước, ví dụ `full_set_30000_rgbs.mp4`.

| Video | Camera đặt ở đâu | Là gì |
|---|---|---|
| `full_set_*` | Đúng vị trí camera thật, **mọi khung** (cả khung train lẫn khung test) | Model render lại toàn bộ đoạn lái xe |
| `test_set_*` | Đúng vị trí camera thật, **chỉ các khung test**: cứ 10 khung giữ lại 1 (`test_image_stride: 10`), model không được học các khung này | Đây là phần đo chất lượng thật: PSNR, SSIM, LPIPS tính trên các khung này |
| `novel/front_center_interp.mp4` | **Đường đi mới**: lấy 5 vị trí của camera trước (cách đều theo thời gian) rồi nối thành đường cong mượt, 24 khung/giây | Góc nhìn camera chưa từng chụp; **không có ảnh thật** để so |

| Đuôi | Nội dung |
|---|---|
| `_gt_rgbs` | **Ảnh camera thật** (ground truth), đã hạ cỡ như lúc train (nuScenes: 800×450) |
| `_rgbs` | Ảnh model render đầy đủ (nền, xe, người, trời ghép lại) |
| `_Background_rgbs` | Chỉ phần nền |
| `_RigidNodes_rgbs` | Chỉ xe |
| `_SMPLNodes_rgbs` | Chỉ người |
| `_DeformableNodes_rgbs` | Chỉ vật biến dạng |
| `_Dynamic_rgbs` | Mọi vật chuyển động (xe, người, vật biến dạng) |
| `_rgb_error_maps` | Sai khác giữa ảnh render (8 bit) và ảnh thật: trung bình trị tuyệt đối trên ba kênh màu, thang màu `inferno` **cố định** từ 0 tới `ERROR_MAP_MAX` = 0,25 (64/255) cho mọi khung, camera và run; điểm ảnh không tính metric (`valid_masks`: rìa mắt cá, thân xe) tô xám 0,35. Từ 20/11/2026; DriveStudio chuẩn hoá mỗi khung theo min, max riêng nên không so được giữa các khung |
| `_depths` | Độ sâu model vẽ ra, thang `turbo` theo log từ 4 tới 120 m (`depth_visualizer`), cố định cho mọi khung. Từ 20/11/2026; renderer 3DGUT, 3DGEER chỉ vẽ độ sâu khi train có depth loss |

**`<bộ>_<bước>_layout.json`** (từ 20/11/2026, `write_video_layout` trong `drive2sim/recon/rendering/videos.py`) nằm cạnh video của mỗi bộ: `width`, `height` (khung ghép, pixel), `fps`, `cameras` (`[{"name", "x", "y", "width", "height"}]`: khung của từng camera trong khung ghép, tìm bằng cách cho hàm ghép của dataset ghép ảnh dò, nên đúng với mọi bố cục, kể cả bố cục đổi cỡ ảnh như KITTI-360), `error_map` (`max`, `cmap`, `invalid_grey`) và `depth` (`min_m`, `max_m`, `curve`, `cmap`) để vẽ chú thích. Video MP4 bị imageio đổi cỡ cho chia hết 16 (ví dụ khung ghép 2399×449 thành video 2400×464), nên người đọc cắt camera theo **tỉ lệ** của khung ghép, không theo pixel của video. Run train trước đó tạo bù bằng `python -m drive2sim.scripts.video_layouts [<run_id> ...]` (đọc `cameras.json`, `config.yaml`); fps của bộ test là `TEST_VIDEO_FPS` = 2 (`recon/tools/eval.py`), bộ full là `render.fps`.

- **`full_set` trông đẹp hơn thực tế.** Khoảng 90 % khung trong đó model đã thấy lúc train, nên gần như "học thuộc".
- **Muốn biết model tốt đến đâu, xem `test_set`.** Đặt `test_set_..._rgbs` cạnh `test_set_..._gt_rgbs` để so cùng khung.
- **`novel` cho thấy chất lượng khi đi lệch khỏi đường xe đã chạy.** Đó là thứ cần cho mô phỏng, nhưng không đo được bằng số vì không có ảnh thật.
- Danh sách video theo `render_keys` trong `train.py`; đánh giá cuối luôn thêm `rgb_error_maps` và `depths` (từ 20/11/2026); bật `render.vis_lidar`, `render.vis_sky`, `render.vis_error` trong config thì có thêm `lidar_on_images`, `rgb_sky_blend`/`rgb_sky`, `rgb_error_maps`. Đường đi `novel` theo `render.render_novel.traj_types` (`front_center_interp`, `s_curve`, `three_key_poses`, `drive2sim/recon/utils/camera.py`).

Huỷ run (`d2s run cancel`, `POST /api/runs/{run_id}/cancel`) chỉ ghi file `cancel`, nên huỷ được từ tiến trình khác. Engine kiểm tra file này trước mỗi stage; stage đang chạy một tiến trình con thì kiểm tra mỗi giây, gửi SIGTERM cho cả nhóm tiến trình, chờ 10 giây rồi SIGKILL. Stage bị huỷ có trạng thái `cancelled` và chạy tiếp được bằng `run resume`. Run có trạng thái `running` mà không có tiến trình nào giữ `pid` (máy tắt giữa chừng) thì lệnh huỷ đổi ngay sang `cancelled`.

### 6.2 `d2s/run.json`

```json
{
  "schema_version": "d2s-run/0.3",
  "run_id": "20261112-081000-omnire-a1b2",
  "scene_id": "nuscenes_scene-0061",
  "split_id": "every8_allcams",
  "approach": {"approach_id": "omnire", "version": "0.1.0", "preset": "no_lidar", "quick": false,
               "options": {"init": "init.depth", "renderer": "3dgs", "campose": false, "fastgs": false}},
  "level": "L1",
  "executor": "kaggle",
  "status": "failed",
  "created_utc": "2026-11-12T08:10:00Z",
  "lidar_usage": [],
  "stages": [
    {"stage_id": "prep.masks_segformer", "status": "succeeded", "cache_hit": true,
     "cache_key": "nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e", "duration_s": 0, "peak_vram_mb": 0,
     "params": {"dynamic": true}},
    {"stage_id": "train.drivestudio", "status": "failed", "cache_hit": false, "cache_key": null,
     "duration_s": 4380, "peak_vram_mb": 7400,
     "params": {"method": "omnire", "renderer": "3dgs", "campose": false, "fastgs": false},
     "error": {"code": "stage_failed", "message": "stage train.drivestudio lỗi: RuntimeError: CUDA out of memory",
               "details": {"log": "d2s/log.txt"}}}
  ]
}
```

- `status` lấy một trong: `pending`, `running`, `succeeded`, `failed`, `cancelled`.
- `approach.options` ghi lựa chọn đã quyết định cho mọi mục cấu hình, kể cả mục để `auto`. Studio lọc và so sánh run theo trường này.
- `stages[].params` ghi tham số đã gộp và đã điền mặc định của từng stage. `d2s run resume` chạy tiếp theo đúng các tham số này, không đọc lại file approach (file có thể đã bị sửa).
- `stages[].error` chỉ có khi stage `failed`, cùng hình dạng lỗi với API (mục 11.3). Traceback đầy đủ nằm trong `d2s/log.txt`.
- `duration_s` là thời gian máy thực sự chạy stage trong run này. Stage bị ngắt (huỷ, lỗi, hết phiên Kaggle) rồi chạy tiếp thì **cộng dồn** thời gian mọi lần chạy; `peak_vram_mb` lấy lớn nhất qua các lần. Stage đã `succeeded` từ lần trước giữ nguyên số đo cũ.
- Stage dùng lại kết quả cũ (cache hoặc file `done`) có `duration_s` và `peak_vram_mb` bằng 0. Khi dùng lại từ cache, `reused_duration_s` ghi thời gian của lần chạy gốc (đọc từ `artifact.json`, mục 5.3), để tổng thời gian dựng cảnh không phụ thuộc run nào chạy trước.
- `lidar_usage` liệt kê stage **dùng LiDAR để train** (`Stage.lidar_used(params)`), không phải stage chỉ nạp LiDAR. `train.drivestudio` với tham số `lidar=false` (ô A-0 của benchmark, mục 6.5; chỉ `pvg`, `deformablegs`) không có trong danh sách: nền khởi tạo từ điểm ngẫu nhiên, không có loss depth so với LiDAR, vùng cảnh (aabb) tính từ vị trí camera. LiDAR vẫn được nạp để đo depth lúc đánh giá nên cảnh phải có LiDAR (cảnh không LiDAR vẫn bị khoá). Đặt bằng `--set train.drivestudio.lidar=false`.
- `peak_vram_mb` do engine đo chỉ thấy bộ nhớ GPU của tiến trình engine, nên bằng 0 với stage chạy một tiến trình con (`prep.*`, `train.drivestudio`, `export.ply_world`). VRAM của train lấy từ `train_stats.json` qua `summary.json` (mục 6.3).
- `approach.quick` là `true` với run chạy thử nhanh (`--quick`): mọi stage có tham số `num_iters` train 3000 iteration, trừ khi người dùng tự đặt `num_iters` bằng `--set`. Run này để kiểm tra approach chạy hết từ đầu tới cuối; metric không so được với run đủ iteration, nên Studio gắn nhãn "thử nhanh" và không đưa run này vào bảng so sánh.
- Thay đổi từ `0.1`: thêm `created_utc`, `stages[].params`, `stages[].error`. Thay đổi từ `0.2`: thêm `approach.quick`, `stages[].reused_duration_s`; `duration_s` cộng dồn khi chạy tiếp. File `0.1`, `0.2` vẫn đọc được.
- `executor` là `local`, `kaggle` hoặc `colab` (cả trong `env.json`). Giá trị `colab` thêm sau mà không tăng phiên bản schema, vì file cũ vẫn đọc được như trước; chỉ bản tool trước đó không đọc được run Colab.

### 6.3 `summary.json`

Tóm tắt metric theo protocol của SPEC mục 9, do `eval.standard` ghi vào thư mục của nó: `d2s/stages/eval.standard/summary.json`.

```json
{
  "schema_version": "d2s-summary/0.6",
  "protocol": {"version": "0.4", "image_dtype": "uint8", "resolution": "downscale_when_loading=[2, 2, 2], downscale=1",
               "lpips": {"net": "alex", "package": "torchmetrics", "package_version": "0.10.3"}},
  "split_id": "every8_allcams",
  "overall": {"psnr": 27.1, "ssim": 0.83, "lpips": 0.21,
              "human_psnr": 23.4, "vehicle_psnr": 24.0, "dynamic_psnr": 23.6, "occupied_psnr": 26.2},
  "per_camera": {"CAM_FRONT": {"psnr": 28.0, "ssim": 0.85, "lpips": 0.19}},
  "train_time_s": 4105,
  "peak_vram_mb": 7400,
  "num_gaussians": 1280000,
  "num_iters": 30000,
  "render_fps": 38.6,
  "render_resolution": "800x450",
  "speedups": ["fastgs", "fastergs.fused_adam", "fastergs.fused_ssim"],
  "depth": {"abs_rel": 0.12, "rmse": 3.4, "delta_1_25": 0.93}
}
```

Các con số trong ví dụ chỉ minh hoạ định dạng. Metric vùng người, vùng xe chỉ có khi cảnh có box.

- `depth` (từ `d2s-summary/0.6`, protocol `0.4`): độ sâu render so với độ sâu LiDAR trên ảnh test, yêu cầu R4 của benchmark. Chỉ đo ở điểm ảnh có tia LiDAR trúng (`lidar_depth_map` của loader, cùng đại lượng với loss depth), hợp lệ và có độ sâu render dương; ảnh có dưới 50 điểm như vậy bị bỏ; mỗi ảnh có trọng số bằng nhau. `abs_rel` = trung bình `|pred − lidar| / lidar`, `rmse` tính bằng mét, `delta_1_25` = tỉ lệ điểm có `max(pred/lidar, lidar/pred) < 1,25`. Để trống khi cảnh không có LiDAR hoặc run đo bằng protocol cũ. Chưa chạy thật trên GPU (mục 7.1 của ARCHITECTURE).
- `d2s/stages/eval.standard/frames.jsonl` (từ protocol `0.4`): mỗi dòng một JSON là metric của một ảnh test: `frame_index` (khung trong timeline của camera), `camera_index` (vị trí camera trong manifest), `time` (thời gian chuẩn hoá, có thể `null`), `position` (toạ độ camera trong hệ thế giới của cảnh, mét), `psnr`, `ssim`, `lpips`, `depth` (`abs_rel`, `rmse`, `delta_1_25`, `num_points`, hoặc `null`). `d2s benchmark` dùng file này để tìm đoạn đường hỏng. Run đo bằng protocol 0.3 trở xuống không có file này; `eval.standard` đo lại từ checkpoint nếu còn (mục 6.3 phía trên).
- Protocol `0.4` không đổi cách tính PSNR, SSIM, LPIPS so với `0.3`, chỉ thêm `depth` và `frames.jsonl`, nên số của hai protocol so được với nhau. `eval.ncore` (Instant NuRec) và `eval.mcgs` (MCGS-SLAM) ghi `protocol.version` là `ncore-0.3` vì SSIM và vùng hợp lệ của chúng khác (cùng script `ncore_eval_render.py`); `eval.mcgs` còn ghi `frames.jsonl` (không có `depth`); `d2s benchmark` cảnh báo khi một nhóm trộn các protocol khác nhau.

- Cách đo (`protocol.version` `0.3`): cuối `train.py`, sau bước train cuối, model vẽ lại mọi ảnh của tập test; ảnh render được làm tròn về 8 bit như ảnh gốc rồi so với ảnh camera từng ảnh một, sau đó lấy trung bình. PSNR là `−10·log10(MSE)`; SSIM dùng `skimage.metrics.structural_similarity`; LPIPS dùng `torchmetrics` với backbone `alex`. Khi loader đánh dấu điểm ảnh không hợp lệ (`valid_masks`: mắt cá ngoài vùng nhìn dùng được, thân xe), cả ba metric chỉ xét điểm ảnh hợp lệ: PSNR trên các điểm đó, SSIM lấy trung bình bản đồ SSIM trên các điểm đó, LPIPS đo trên khung bao vùng hợp lệ với ảnh gốc dán vào điểm không hợp lệ (điểm không hợp lệ còn lại trong khung, như góc ảnh mắt cá, vẫn làm LPIPS loãng đi một chút). Ảnh không có điểm hợp lệ nào bị bỏ qua. `eval.standard` chỉ đọc kết quả này, không đo lại. Thay đổi từ `0.1`: `0.1` so ảnh render ở dạng số thực. Thay đổi từ `0.2`: `0.2` tính SSIM, LPIPS trên toàn ảnh sau khi tô đen vùng không hợp lệ ở cả hai ảnh, nên vùng đó được tính là khớp hoàn toàn và SSIM, LPIPS của camera có `valid_masks` tốt hơn thực tế; thêm metric theo camera. Hai run khác protocol không so SSIM, LPIPS của các camera này với nhau.
- Metric theo vùng (lấy từ metric test của DriveStudio): `human_psnr` vùng người (gồm cả xe đạp, xe máy vì người ngồi trên đó đổi dáng), `vehicle_psnr` vùng xe, `dynamic_psnr` vùng mọi vật động (`masked_psnr` của DriveStudio), `occupied_psnr` vùng **không phải bầu trời** (`Non-Sky PSNR`, không phải vùng vật động). Vùng người, xe, vật động là hình chữ nhật bao quanh box 3D chiếu lên ảnh.
- `train_time_s`, `peak_vram_mb`, `num_gaussians`, `num_iters` đọc từ `train_stats.json` mà `train.py` ghi cùng mỗi checkpoint. `train_time_s` là **thời gian train thực**: chỉ tính các bước train (lấy batch, forward, backward, cập nhật), không tính lúc khởi động, nạp checkpoint, vẽ ảnh kiểm tra, lưu checkpoint hay đánh giá cuối; chạy tiếp từ checkpoint thì cộng tiếp từ số đã lưu cùng checkpoint đó, nên các bước bị làm lại sau khi ngắt không bị tính hai lần. `num_gaussians` là tổng mọi node của model (nền, xe, người, vật biến dạng).
- `render_fps`: số khung chia cho thời gian model vẽ ảnh tập test (chỉ tính lời gọi model, có đồng bộ GPU, bỏ 3 khung đầu), ở độ phân giải `render_resolution`. FPS phụ thuộc GPU; tên GPU nằm trong `env.json`.
- Các trường trên để trống (`null`) khi không có số liệu, ví dụ model train bằng `train.py` cũ chưa ghi `train_stats.json`.
- `per_camera`: `train.py` ghi khoá `image_metrics/test/per_camera/<chỉ số camera>/<psnr|ssim|lpips>` trong file metric test, với chỉ số là vị trí của camera trong `manifest.yaml` (cách `train.drivestudio` đánh số camera cho DriveStudio); `eval.standard` đổi chỉ số sang `camera_id`. Trainer khác có thể ghi thẳng `camera_id`. Model train bằng `train.py` cũ (protocol `0.2` trở về trước) không có khoá này nên `per_camera` để trống.
- `protocol.resolution` đọc từ `config.yaml` của DriveStudio; không có file này thì lấy trường `resolution` trong `train_stats.json` (ví dụ `downscale=2`).
- `speedups`: các thuật toán, kỹ thuật tăng tốc **thật sự** đã bật khi train, đọc từ `train_stats.json`: `fastgs` (`docs/methods/fastgs.md`), `fastergs.<kỹ thuật>` (`docs/methods/fastergs.md`). Rỗng khi không bật gì hoặc model train bằng `train.py` cũ. Mục cấu hình của run chỉ ghi công tắc; trường này ghi cái công tắc thật sự làm, để hai run cùng `fastergs: true` mà khác kỹ thuật (ví dụ ở hai phiên bản code) không bị coi là cùng cấu hình.
- Thay đổi từ `0.1`: `train_time_s`, `peak_vram_mb`, `num_gaussians` thành tuỳ chọn. Thay đổi từ `0.2`: thêm `dynamic_psnr`, `num_iters`, `render_fps`, `render_resolution`; `occupied_psnr` ghi đúng nghĩa là vùng không phải bầu trời. Thay đổi từ `0.3`: thêm `speedups`. Thay đổi từ `0.4`: thêm `eval_time_s`: thời gian đánh giá cuối trong stage train (vẽ ảnh test, tính metric, render video), `train.py` đo rồi ghi vào `train_stats.json`; trang run tách thời gian stage train thành train, khởi động và lưu checkpoint, đánh giá và render video. Stage `eval.standard` chỉ đọc kết quả nên gần như không tốn thời gian.

### 6.4 `d2s/env.json`

Môi trường và **máy** đã chạy run. Thời gian train, VRAM và FPS phụ thuộc phần cứng, nên chỉ so được giữa các run chạy trên cùng loại GPU; Studio hiện GPU ở trang run và trong bảng so sánh, và đánh dấu cảnh báo ở tiêu đề các cột thời gian, VRAM, FPS khi bảng có run trên nhiều GPU.

```json
{
  "schema_version": "d2s-env/0.2",
  "code_commit": "7851701",
  "python": "3.9.19",
  "pytorch": "2.0.0+cu117",
  "cuda": "11.7",
  "gsplat": "1.3.0",
  "gpu": "Tesla T4",
  "seed": 0,
  "executor": "kaggle",
  "machines": [
    {"cpu": "Intel(R) Xeon(R) CPU @ 2.00GHz", "cpu_cores": 4, "ram_gb": 31.4,
     "gpu": "Tesla T4", "gpu_count": 1, "gpu_memory_mb": 15360, "gpu_driver": "535.104.05",
     "os": "Linux 6.6.56+", "first_seen_utc": "2026-11-12T08:10:05Z"}
  ]
}
```

Các giá trị trong ví dụ chỉ minh hoạ định dạng. Trường nào không xác định được thì bỏ trống.

- Engine ghi file khi tạo run (`machines` rỗng) rồi cập nhật **mỗi lần bắt đầu chạy hoặc chạy tiếp**, trên đúng máy chạy. Với run Kaggle đó là máy của notebook, với run Colab là VM, không phải máy đã tạo run.
- `machines`: các máy đã chạy run theo thứ tự; chạy tiếp trên máy có cấu hình khác (ví dụ phiên Kaggle mới được cấp GPU khác) thì thêm một máy. `gpu` ở ngoài là GPU của máy chạy gần nhất.
- Cách đọc, không cần nạp PyTorch: CPU từ `/proc/cpuinfo`, RAM từ `/proc/meminfo`, GPU (tên, VRAM, driver, số GPU) từ `nvidia-smi`; phiên bản `pytorch`, `gsplat` từ metadata gói đã cài, `cuda` từ đuôi phiên bản PyTorch (`+cu117` → `11.7`).
- Thay đổi từ `0.1`: thêm `machines`; các trường phiên bản và `gpu` ghi lúc chạy thay vì lúc tạo run. Thay đổi từ `0.2`: mỗi máy có thêm `stages`, các stage đã thật sự chạy trên máy đó (stage dùng lại kết quả cũ không tính); `same_machine` bỏ qua trường này. Báo cáo lấy GPU của máy đã chạy stage `train.*` (run train trên Kaggle rồi xuất lại gói xem trước trên máy cá nhân không thành "nhiều GPU"); `env.json` cũ chưa có `stages` thì gộp mọi máy. Các trường cấp trên (`gpu`, `executor`, phiên bản) là của lần chạy gần nhất.

### 6.5 Benchmark: `benchmarks/<tên>.yaml` và kết quả

Benchmark so các run đã có kết quả (thiết kế ở báo cáo tuần 03 mục 4.2, yêu cầu R1–R8). File định nghĩa nằm trong thư mục làm việc, do người viết tay; mỗi mục là một nhóm run được xếp hạng với nhau. `bench_track` là track của benchmark (`A` dựng từng cảnh, `B` quãng đường dài, `C` feed-forward), **không phải** loại camera pinhole hay fisheye. Track `A` bắt buộc có `cell`: `A-LB` (có LiDAR và box), `A-L` (chỉ LiDAR), `A-0` (không cả hai); track khác không có `cell`. Mọi run trong một mục phải cùng cảnh và split.

```yaml
schema_version: d2s-benchmark/0.1
entries:
  - bench_track: A
    cell: A-L
    scene_id: kitti360_drive-0003
    split_id: every10_front2
    runs: [20261124-101500-pvg-a1b2, 20261124-103000-deformablegs-c3d4]
```

`d2s benchmark <file> [--out <kết quả.json>]` đọc các file kết quả đã có của run (không chạy GPU) và ghi `d2s-benchresult`:

```json
{
  "schema_version": "d2s-benchresult/0.1",
  "tie_threshold_db": 0.3,
  "groups": [
    {
      "bench_track": "A",
      "cell": "A-L",
      "scene_id": "kitti360_drive-0003",
      "split_id": "every10_front2",
      "winner_run_id": "20261124-103000-deformablegs-c3d4",
      "rows": [
        {"run_id": "20261124-103000-deformablegs-c3d4", "approach_id": "deformablegs", "rank": 1,
         "tied_with_best": true, "psnr": 24.8, "ssim": 0.78, "lpips": 0.26, "avg_error": 0.0405,
         "psnr_pinhole": 24.8, "depth": {"abs_rel": 0.12, "rmse": 3.4, "delta_1_25": 0.93},
         "ply_gaussians": 640000, "ply_size_mb": 151.2, "train_time_s": 5400, "peak_vram_mb": 6100,
         "render_fps": 41.0, "gpu": "NVIDIA GeForce RTX 4090",
         "bad_segments": [{"start_m": 60.0, "end_m": 100.0, "num_frames": 5, "min_psnr": 19.5}],
         "frames_measured": 20}
      ],
      "warnings": []
    }
  ]
}
```

Các con số trong ví dụ chỉ minh hoạ định dạng. Quy tắc:
- **Xếp hạng trong nhóm** theo Avg của Mip-NeRF (mục 6.3; thấp là tốt). Run kém run tốt nhất dưới `tie_threshold_db` (0,3 dB PSNR) được coi là **ngang** (`tied_with_best`), và các run ngang được xếp theo export gap (nhỏ trước, chưa đo xếp sau) rồi thời gian train. Lệch làn và độ sạch cảnh nền chưa đo nên chưa xét. `winner_run_id` là run xếp hạng 1; `null` khi không run nào có kết quả.
- **Đoạn đường hỏng** (R7): PSNR một khung test là trung bình các camera của khung; đoạn hỏng là từ 5 khung test liền nhau trở lên, mỗi khung thấp hơn trung vị quá 2 dB. `start_m`, `end_m` là quãng đường dọc đường gấp khúc qua vị trí camera của các khung test (giữa hai khung test liền nhau là khung train nên đường thực dài hơn một chút). Cần `frames.jsonl` (mục 6.3); `frames_measured` bằng 0 nghĩa là chưa đo, không phải không có đoạn hỏng.
- `psnr_pinhole`, `psnr_fisheye` (R5): trung bình PSNR theo camera, nhóm camera theo `model` trong manifest (`opencv_fisheye`, `ftheta` là fisheye).
- `export_gap_db` (R3) đọc từ `d2s/stages/eval.ply/ply_gap.json` nếu có; chưa có stage đó nên hiện luôn để trống.
- Trường để trống nghĩa là **chưa đo**. Chưa đo: KID khi lệch làn (R2, hoãn), export gap (R3), PSNR quanh chỗ nối chunk (R6, làm khi có run chunk đầu tiên). Chủ ý **không đo**: `ghost_ratio` (chưa có định nghĩa, chỉ có nghĩa với method tách lớp nền và vật động), `static_psnr` (hoãn tới khi cần so method chỉ dựng cảnh tĩnh như Instant NuRec), ATE (chỉ method tự ước lượng pose có; ghi số do method báo, không tính lại). Lý do ở ARCHITECTURE mục 2.17.
- `warnings` nêu khi nhóm không so được trực tiếp: run chưa có kết quả, GPU khác nhau (thời gian, VRAM, FPS), protocol khác nhau.

## 7. Gói xem trước

Gói xem trước cho phép xem cảnh trong trình duyệt mà không cần GPU CUDA (`ARCHITECTURE.md` mục 2.13). Không bắt buộc.

```
runs/.../<run_id>/d2s/stages/export.preview/
├── artifact.json
├── preview.json
├── background.spz               # Gaussian của nền tĩnh, hệ world (gói 0.1: background.ply)
├── actors/<track_id>.spz        # Gaussian của từng xe (actor rigid), hệ của box
├── actor_poses.json
├── humans/<track_id>.spz        # Gaussian của từng người (SMPL) ở dáng gốc
├── humans/<track_id>.skin       # 4 xương và trọng số của mỗi Gaussian người
├── human_bones.bin              # ma trận 24 xương của mọi người theo khung
├── human_poses.json             # vị trí của từng người theo khung
├── sky_cube.jpg                 # cubemap trời của model, nếu có
├── color_affine.bin             # ma trận chỉnh màu 3×4 của từng ảnh (khung, camera), nếu có
└── cameras.json                 # chép từ stage train, nếu có
```

- File Gaussian từ bản 0.2 là SPZ bản 3 (`drive2sim/recon/tools/spz.py`, chỉ cần numpy): header 16 byte (`magic` 0x5053474e, `version` 3, số Gaussian, bậc SH, số bit phần lẻ, `flags`, 1 byte dự trữ), rồi vị trí (3 số 24 bit, 12 bit phần lẻ, ít hơn khi cảnh rộng quá 2 km), độ đục, màu, kích thước, hướng ("smallest-three" 4 byte), SH (theo hệ số rồi theo kênh màu), nén gzip. Toạ độ giữ nguyên như PLY. Nền của run `f8a6` (388.408 Gaussian): PLY 96,3 MB, SPZ 8,7 MB; sai số vị trí tối đa 0,24 mm, hướng trung bình 0,08°; bố cục byte khớp file do `writeSpz` của Spark ghi từ cùng dữ liệu; ảnh vẽ bằng Spark từ SPZ so với từ PLY ở cùng camera: PSNR 35,7 dB.
- Người: `humans/<track_id>.skin` là uint16 little-endian, 4 số mỗi Gaussian theo đúng thứ tự trong `.spz`, mỗi số `trọng số 8 bit | (chỉ số xương << 8)` như `SplatSkinning.setSplatBones` của Spark. Trọng số lấy như `SMPLTemplate.forward` (lưới `voxel_deformer` tại vị trí gốc), giữ 4 xương lớn nhất rồi nhân để tổng bằng tổng gốc. `human_bones.bin` là float32 (F khung, H người, 24 xương, 12 số): 3 cột phần xoay rồi phần dời của ma trận xương (`batch_rigid_transform` của smplx trên góc khớp trong checkpoint, nhân `A0_inv`), không cần file mô hình SMPL. `human_poses.json`: `{"track_ids": [H tên], "T_object_world": [F][H]}`, mỗi phần tử là ma trận dời 4x4 (16 số theo hàng) cộng sau khi uốn, hoặc `null` khi người không có mặt. Trang Xem 3D uốn người bằng `SplatSkinning` chế độ dual quaternion: lệch so với linear blend của OmniRe trung bình 4 mm, 99% điểm dưới 3,3 cm (đo trên run `f8a6`); chế độ linear blend của Spark 2.4.0 cần `covSplats`, mà đường đó thiếu bộ sắp xếp độ sâu nên không vẽ được. SH của người chỉ có bậc 1; viewer giới hạn bậc SH của người ở 1.
- `color_affine.bin`: ma trận chỉnh màu của `AffineTransform` cho từng ảnh train, float32 (F khung, C camera, 12): ma trận 3×4 theo hàng, `rgb' = A·rgb + b` (đã cộng ma trận đơn vị); ảnh (khung f, camera c) có chỉ số `f × C + c` như `unique_img_idx` của loader, `c` theo thứ tự `cameras.json`. Trang Xem 3D áp ma trận của camera đang chọn (đường novel: camera đầu) ở khung đang xem lên ảnh đã ghép, bằng cách chép ảnh màn hình rồi vẽ lại qua một shader; frame server nhận cùng chỉ số camera (`camera` trong `render-frame`) nên ảnh chính xác dùng cùng ma trận. Đo ở run `f8a6`: ma trận gần đơn vị (đường chéo khoảng 0,995), Gói xem trước so với ảnh renderer gốc tăng 28,61 → 28,83 dB.
- `sky_cube.jpg`: `EnvLight.base` (6 mặt, thứ tự OpenGL +x, -x, +y, -y, +z, -z) xếp 3 cột × 2 hàng, mặt `i` ở cột `i % 3`, hàng `i // 3`; trong mỗi mặt hàng ảnh là toạ độ `t`, cột là `s` của bảng chọn mặt OpenGL, khớp `nvdiffrast.texture(boundary_mode="cube")` (đã so 100% điểm). Hướng tra là `EnvLight.to_opengl @ hướng nhìn world`. Viewer vẽ trời sau mọi splat, như OmniRe ghép `rgb + sky × (1 − opacity)`.

- File `.ply` theo định dạng PLY của 3DGS gốc: `x, y, z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`. Mọi PLY do tool xuất (`scene.ply`, `background.ply`, `actors/*.ply`) có dòng header `comment drive2sim up=<trục> frame=drivestudio_world units=m`, ví dụ `up=-y`. Trục lên tuỳ dataset vì hệ world của DriveStudio lấy theo camera trước ở khung đầu (nuScenes, KITTI, PandaSet: `-y`; Waymo: `+z`), nên được suy từ pose camera (`cameras.json` cạnh checkpoint): cộng hướng lên (`-y` của camera OpenCV) của mọi camera, mọi khung rồi lấy trục gần nhất; không có `cameras.json` thì bỏ phần `up=`. Trang Xem 3D đọc dòng này khi kéo thả file; PLY xuất trước 16/11/2026 ghi cứng `up=+z`, sai với nuScenes. PLY từ nơi khác không có dòng này thì trang đoán trục lên từ mặt đường (mục 2.13 của ARCHITECTURE).
- `cameras.json`: `{"frame": "drivestudio_world", "convention": "opencv", "units": "metres", "cameras": [{"name", "width", "height", "c2w": [F khung][16 số theo hàng], "K": [F khung][9 số theo hàng]}]}`. `c2w` theo quy ước OpenCV (x phải, y xuống, z nhìn tới). Do `train.py` ghi ngay sau khi nạp dữ liệu; `export.ply_world` và `export.preview` chép sang thư mục của mình. Run train trước khi có file này thì không có; trang Xem 3D khi đó mở ở góc nhìn bao quát.
- `actor_poses.json`: `{"track_ids": [K tên], "T_object_world": [F khung][K xe]}`; mỗi phần tử là ma trận 4x4 viết thành 16 số theo hàng (đưa điểm từ hệ của xe sang `world`), hoặc `null` khi xe không có mặt ở khung đó. Khung đánh số theo thứ tự khung của DriveStudio (checkpoint không lưu mốc thời gian). Dùng JSON thay `.npz` để trình duyệt đọc thẳng, không cần numpy ở Studio; 200 khung × 30 xe khoảng 1 MB.
- `track_id` của xe là `rigid_<chỉ số>`, của người và vật biến dạng là `smpl_<chỉ số>`, `deformable_<chỉ số>`, theo thứ tự instance trong checkpoint.
- Do stage `export.preview` ghi (`drive2sim/recon/tools/export_preview.py`); nằm trong danh sách stage mặc định của OmniRe và Street GS, sau `export.ply_world`; run cũ hoặc approach khác thì thêm bằng `d2s export preview <run_id>` hay nút trên trang run. `GET /api/runs/{run_id}/splats` trả đường dẫn các file này và `scene.ply`.

```json
{
  "schema_version": "d2s-preview/0.2",
  "source_run": "runs/nuscenes_scene-0061/omnire/20261112-081000-omnire-a1b2",
  "sh_degree": 3,
  "splat_format": "spz",
  "up_axis": "-y",
  "antialiased": false,
  "rigid_actors": ["rigid_0000", "rigid_0001"],
  "humans": ["smpl_0000"],
  "sky": "sky_cube.jpg",
  "color_affine": "color_affine.bin",
  "box_only_actors": ["deformable_0000"],
  "omitted": ["fisheye", "DeformableNodes"]
}
```

- `splat_format`: đuôi file Gaussian (`spz`; gói 0.1 cũ không có trường này, coi là `ply`). `antialiased`: `render.antialiased` trong `config.yaml` của stage train; trang Xem 3D đặt `preBlurAmount` 0,3 và `blurAmount` 0 của Spark khi model train không chống răng cưa (như gsplat cộng 0,3 vào kích thước đốm mà không bù độ đục), ngược lại thì `blurAmount` 0,3. `humans`, `sky`: người và trời có trong gói.
- `box_only_actors`: actor chưa có Gaussian trong gói (vật biến dạng: chuyển động do mạng nơ-ron tính, trình duyệt không chạy lại); chưa hiện trong viewer.
- `omitted`: những phần của model không có trong gói.

## 8. Interface theo mức năng lực

```python
@dataclass
class CameraSpec:                  # cùng trường với một camera trong manifest + intrinsics
    model: str                     # opencv_pinhole | opencv_fisheye | ftheta
    width: int
    height: int
    intrinsics: dict
    shutter: str

@dataclass
class ActorState:
    track_id: str
    T_object_world_start: np.ndarray   # 4x4
    T_object_world_end: np.ndarray     # 4x4
    visible: bool = True

class Renderer(Protocol):              # L2
    def render(self, camera: CameraSpec, T_camera_world_start: np.ndarray,
               T_camera_world_end: np.ndarray, timestamp_us: int,
               actors: list[ActorState] | None = None,
               outputs: tuple[str, ...] = ("rgb",)) -> dict[str, np.ndarray]:
        """rgb: uint8[H,W,3]; depth (nếu yêu cầu): float32[H,W], mét."""

class ActorEditor(Protocol):           # L3
    def list_actors(self) -> list[dict]: ...      # track_id, class, size_lwh, rigid, editable
    def get_trajectory(self, track_id: str) -> list[tuple[int, np.ndarray]]: ...
    def set_trajectory(self, track_id: str, poses: list[tuple[int, np.ndarray]]) -> None: ...
    def set_visible(self, track_id: str, visible: bool) -> None: ...
    def remove(self, track_id: str) -> None: ...
```

- `T_camera_world` đưa điểm từ camera sang `world` (quy ước mục 1), tương ứng `camtoworlds` của DriveStudio.
- **L2:** approach cung cấp `Renderer`. Khi `actors` là `None`, render theo quỹ đạo đã dựng.
- **L3:** approach cung cấp thêm `ActorEditor`. Với dòng DriveStudio, actor rigid sửa được qua `instances_trans`, `instances_quats`, `instances_fv` của RigidNodes.
- **L4:** `Renderer` xử lý đúng `actors` trong từng yêu cầu mà không giữ trạng thái giữa các yêu cầu, như yêu cầu của `DynamicObject` trong proto.

Ánh xạ sang proto `SensorsimService` ở mức L4:

| Proto | drive2sim |
|---|---|
| `CameraRenderRequest.camera_intrinsics` (`CameraSpec`) | `CameraSpec` (cùng ba mô hình) |
| `sensor_pose.start_pose` / `end_pose` | `T_camera_world_start` / `T_camera_world_end`, đổi theo quy ước `Pose` của proto bằng hàm trong `core` |
| `frame_start_us`, `frame_end_us` | `timestamp_us` lấy ở giữa; pose đầu và cuối dùng cho rolling shutter |
| `dynamic_objects` | `actors` |
| `get_dynamic_objects` | `list_actors` + `get_trajectory` |

## 9. Worker, Kaggle và Colab

### 9.1 Worker

Dùng cho codebase ngoài (môi trường riêng) và cho tiến trình render của viewer và server:

```
d2s-worker --job job.json --result result.json
```

`job.json` do engine ghi, worker đọc:

```json
{
  "schema_version": "d2s-job/0.1",
  "job_id": "20261112-081000-omnire-a1b2/train.external",
  "kind": "stage",
  "stage_id": "train.external",
  "params": {"codebase": "nurec", "num_iters": 30000},
  "inputs": {"scene": "data/nuscenes_scene-0061", "points_init": "cache/nuscenes_scene-0061/init.depth@0.1.0-9b03c2d1"},
  "output_dir": "runs/nuscenes_scene-0061/omnire/20261112-081000-omnire-a1b2/d2s/stages/train.external",
  "env": "nurec"
}
```

- `kind` lấy một trong: `stage` (chạy một stage), `render` (giữ model trong bộ nhớ và render cho viewer hoặc server).
- `stage_id` và `params` chỉ có khi `kind` là `stage`. `inputs` ánh xạ tên đầu vào của stage sang đường dẫn.
- `env` là tên file môi trường trong `envs/` (không có đuôi `.yml`).
- Mọi đường dẫn tương đối với `D2S_WORKDIR` (mục 10.3).

`result.json` do worker ghi khi xong, kể cả khi thất bại:

```json
{
  "schema_version": "d2s-result/0.1",
  "job_id": "20261112-081000-omnire-a1b2/train.external",
  "status": "failed",
  "outputs": {},
  "duration_s": 12.5,
  "peak_vram_mb": 0,
  "error": {"code": "env_missing", "message": "Không tìm thấy môi trường envs/nurec.yml"}
}
```

- `status` lấy một trong: `succeeded`, `failed`. Thành công thì `outputs` ánh xạ tên đầu ra sang đường dẫn và `error` là `null`.
- `error.code` là `snake_case`, dùng chung bảng mã với API (mục 11.3); `error.message` bằng tiếng Việt, hiện thẳng cho người dùng.
- Trong lúc chạy, worker in ra **stdout** mỗi dòng một JSON, trường `event` lấy một trong:

| `event` | Các trường | Nghĩa |
|---|---|---|
| `progress` | `fraction` (0–1), `message` | Tiến độ |
| `metric` | `name`, `value`, `step` | Một giá trị metric trong lúc train |
| `log` | `level` (`info` / `warning` / `error`), `message` | Dòng log cho người đọc |

  Dòng nào không phải JSON thì engine coi là `log` mức `info`. Log của thư viện ngoài đi ra **stderr**.
- Mã thoát: `0` thành công; khác `0` thất bại. Engine đọc `result.json` trước, mã thoát sau; không có `result.json` thì coi là lỗi `worker_crashed`.
- Worker nhận **SIGTERM** là yêu cầu dừng: ghi `result.json` với `status: failed`, `error.code: cancelled`, rồi thoát.

### 9.2 Gói job cho Kaggle

Run được tạo trên máy (`executor: kaggle`) rồi đẩy lên Kaggle bằng `d2s kaggle push` (`d2s run start --executor kaggle` tự gọi). Gói job là vài file JSON nhỏ, nhúng thẳng vào notebook sinh từ `drive2sim/kaggle/templates/run.ipynb`:

```
bundle/
├── run.json          # run.json đã tạo trên máy: approach, lựa chọn, tham số từng stage
├── split.json        # split của run
├── scene_ref.json    # {"scene_id", "converter_version", "kaggle_dataset"}
└── resume.json       # {"kernels": các notebook của tối đa 5 lần đẩy trước, mới nhất trước; [] ở lần đầu}
```

- **Cảnh** được upload một lần thành dataset private `<người dùng>/d2s-scene-<cảnh>-<commit converter>` (mục 10.1, ví dụ `d2s-scene-nuscenes-0061-96876ad`); ingest lại bằng converter khác thì upload dataset mới. Dataset đặt theo tên cũ `d2s-<hash8 của scene_id@converter_version>` vẫn được dùng lại, không upload lại. Run có stage `prep.smpl` (OmniRe dựng người) cần file mô hình SMPL `SMPL_NEUTRAL.pkl`, có bản quyền nên không đi kèm repo: người dùng upload file thành một dataset private và đặt biến `D2S_KAGGLE_SMPL_DATASET=<người dùng>/<tên>`; `kaggle push` gắn thêm dataset đó, thiếu biến thì từ chối. Trước khi đẩy notebook, `kaggle push` chờ dataset ở trạng thái `ready` (tối đa 15 phút), vì notebook chạy khi dataset còn đang xử lý sẽ không thấy cảnh.
- **Code**: notebook `git clone --branch <branch>` đúng repo (remote `origin`, hoặc biến `D2S_KAGGLE_REPO`) rồi checkout đúng commit, nên commit phải có trên **chính remote đó** và remote phải có nhánh `<branch>`; `kaggle push` từ chối nếu không (commit chỉ có trên remote khác, ví dụ `fork`, cũng bị từ chối). URL không thuộc remote nào của repo thì không kiểm tra được.
- **Notebook** `<người dùng>/d2s-<cảnh>-<approach>-<biến thể>-<4 hex của run>-<lần đẩy>` (mục 10.1, ví dụ `d2s-nuscenes-0061-omnire-3dgs-fastgs-453a-1`), private, có GPU (`machine_shape` mặc định `NvidiaTeslaT4`, đổi bằng `--accelerator`) và internet. Notebook in phiên bản GPU và PyTorch, clone đúng commit, cài môi trường chính bằng `envs/install_main.sh` (cài lại PyTorch 2.4.1 + CUDA 12.1, build gsplat-geer, nvdiffrast, pytorch3d cho đúng compute capability của GPU phiên đó), rồi chạy `d2s kaggle run bundle/ --workdir /kaggle/working --input-root /kaggle/input`. Các ô lệnh shell chạy bằng `%%bash` với `set -e`: clone hay cài lỗi thì notebook dừng và kết thúc với trạng thái `error`.
- `d2s kaggle run` tìm cảnh dưới `--input-root` theo `manifest.yaml` (không giả định đường dẫn gắn của Kaggle), dựng `data/<scene_id>/` bằng symlink tới từng mục của cảnh, ghi `splits/` mới, rồi chạy run như `run resume`. Kết quả ở `/kaggle/working/runs/` và `/kaggle/working/cache/` theo đúng cấu trúc mục 5 và 6; ô chạy run luôn xoá `data/` và mọi symlink sau khi run kết thúc, kể cả khi run lỗi, để output chỉ còn kết quả và lần đẩy sau chạy tiếp được; mã lỗi của run được trả lại sau khi dọn.
- **Chạy tiếp**: phiên bị ngắt thì `d2s kaggle pull` rồi `d2s kaggle push` lần nữa. Lần đẩy sau gắn output của tối đa 5 lần đẩy trước (`kernel_sources`) và ghi tên chúng vào `resume.json`, mới nhất trước; `d2s kaggle run` chọn lần mới nhất có `run.json` của run (lần gần nhất có thể lỗi trước khi ghi được gì, ví dụ lúc cài môi trường) rồi chép run và cache trong output đó vào thư mục làm việc trước khi chạy, nên stage đã xong không chạy lại và `train.drivestudio` chạy tiếp từ `checkpoint_resume.pth`. `d2s kaggle push --rerun <stage>` (lặp được) chạy lại stage đó dù lần trước đã xong, ví dụ `--rerun eval.standard` để đo lại bằng code mới mà không train lại: `resume.json` có thêm `rerun`, và `materialize` gọi `reset_stage` ngay sau khi chép run của lần trước (lần gọi sau, ví dụ `--prepare-only` rồi chạy, không đặt lại). Run chưa đẩy lần nào hoặc không có stage đó thì `push` từ chối.
- **Dùng lại cache của run khác**: mỗi lần đẩy, `push` tìm trong `run.json` trên máy (run đã tải về) các run Kaggle khác cùng cảnh, đẩy bằng cùng tài khoản, có stage cache (`prep.masks_segformer`, `prep.smpl`...) cùng `stage_id`, cùng tham số, đã xong; notebook mới nhất của các run đó (bỏ notebook đã xoá hoặc đang chạy, mỗi stage một run, mới nhất trước) được gắn thêm vào `kernel_sources`. `d2s kaggle run` chép mọi mục `cache/<cảnh>/<khoá>/` có `artifact.json` trong các output đã gắn mà thư mục làm việc chưa có (chép, không symlink, để output notebook này cũng có cache); engine so khoá cache đầy đủ như trên máy, khớp thì dùng lại, không thì chạy stage như thường. Với nuScenes `scene-0061` việc này bỏ được khoảng 1,6 giờ GPU (mask 27 phút, SMPL 72 phút ở run `b8a9`).
- **Hai run một notebook**: `d2s kaggle push RUN1 RUN2` (chỉ với máy `NvidiaTeslaT4`, có 2 GPU T4 15 GB) đẩy hai run vào một notebook `<người dùng>/d2s-<cảnh>-pair-<4 hex run 1>-<4 hex run 2>-<lần đẩy lớn nhất>`. Gói job nhúng dạng `{run_id: gói}`, mỗi run một thư mục `/kaggle/tmp/bundle/<run_id>/`. Ô chạy run gọi `d2s kaggle run <gói> --prepare-only` lần lượt cho từng run (dựng cảnh, chép cache và run chạy tiếp, không chạy; hai tiến trình dựng cùng lúc sẽ tranh nhau tạo symlink), rồi chạy hai run song song, run thứ i với `CUDA_VISIBLE_DEVICES=i`, log mỗi dòng có tiền tố 4 hex của run; mã lỗi là 1 nếu có run lỗi. Notebook đặt `D2S_CACHE_WAIT_S=43200`: run cần khoá cache mà run kia đang ghi thì chờ (kiểm tra 30 giây một lần) rồi dùng lại kết quả, thay vì lỗi `CacheBusyError` (không đặt biến thì engine báo lỗi ngay như trước). Quota GPU của Kaggle tính theo giờ phiên, nên hai run tốn giờ như một. Mỗi run giữ `kaggle.json` riêng trỏ cùng notebook; đẩy lại thì mỗi run chạy tiếp từ output của chính nó, riêng hay đôi đều được. `pull` một run tải output một lần và ghép luôn run kia (đánh dấu cả hai đã tải); `pull --light` vốn đọc kết quả nhỏ của mọi run trong log.
- **Cache cho tài khoản mới**: output notebook chỉ tài khoản đã chạy mới đọc được. Stage cache nào không notebook nào cùng tài khoản có (lần đẩy đầu, không phải chạy tiếp) mà máy có cache cùng tham số (`cache/<cảnh>/<khoá>/` có `artifact.json`, của run đã tải về) thì `push` upload các thư mục đó thành dataset private `<người dùng>/d2s-cache-<cảnh>-<8 hex băm các khoá>` (chép bằng hard link khi được) và gắn qua `dataset_sources`; `d2s kaggle run` chép sang như với output notebook. Dataset cùng tên đã có thì dùng lại, không upload lại.
- **Dùng lại môi trường đã build**: notebook đặt `D2S_WHEEL_SOURCES=/kaggle/input` và `D2S_WHEEL_CACHE=/kaggle/working/wheels` cho `envs/install_main.sh`. Mỗi gói build từ mã nguồn (gsplat-geer, nvdiffrast, fused-ssim, pytorch3d, detectron2, neural_renderer) có khoá = băm(gói kèm commit, bản PyTorch, `TORCH_CUDA_ARCH_LIST`, bản Python); có `wheels/<khoá>/*.whl` trong đầu vào thì cài thẳng, không thì build bằng `pip wheel` rồi cài; cả hai trường hợp đều chép wheel vào `wheels/<khoá>/` của output. Chỉ cài CUDA 12.8 riêng khi thật sự phải build. `push` gắn thêm notebook mới nhất đã xong của cùng tài khoản (thử tối đa 5 notebook, bỏ notebook đã xoá hoặc đang chạy) vào `kernel_sources` làm nguồn wheel. `pull` không tải `wheels/`. Ở run `90dd` lần 2, clone và cài môi trường mất 31 trong 61 phút của notebook.
- **Lệnh hỏi trạng thái** (`kaggle kernels status`, `kaggle datasets status`) dừng sau 5 phút và báo lỗi (đã gặp lệnh treo hơn 30 phút); lệnh khác dừng sau 6 giờ.
- **Kết quả nhỏ trong log**: ô cuối của notebook in `run.json`, `env.json`, các file `done`, `summary.json`, `*metrics*.json`, `train_stats.json`, `profile_step*.json`, `densify_stats.jsonl` của run (mỗi file tối đa 4 MB) vào stderr của tiến trình sau chữ `D2S_RUN_LIGHT` (base64 của JSON `{đường dẫn: nội dung}`), rồi mới trả mã lỗi của run. `d2s kaggle pull <run_id> --light` đọc chúng bằng một lệnh `kaggle kernels logs` và ghép như `pull`, không đánh dấu đã tải (`pulled_utc`), nên `pull` vẫn tải đủ sau; từ chối khi run đã tải đủ. Lý do: output đầy đủ có hàng nghìn file, tải mất cỡ một giờ và dễ bị 429.
- `d2s kaggle pull` tải output rồi ghép vào `d2s/` của run: mỗi stage có trong bản Kaggle thay stage cùng tên trên máy; stage chỉ có trên máy (ví dụ `export.preview` chạy trên máy sau lần tải trước) được giữ lại và nối vào cuối `run.json`; `env.json` lấy theo Kaggle cộng các máy chỉ có trên máy; `kaggle.json` giữ nguyên. Chép thêm các kết quả cache máy chưa có, không ghi đè cache đã có; cache máy đã có thì không tải (`--file-pattern` loại `cache/<cảnh>/<stage>@<khoá>/` đã có, vì lệnh `kaggle` tải từng file một và cache mask có hàng nghìn file). Từ chối khi run đang chạy trên máy.
- **Gói output** (`drive2sim/kaggle/packing.py`, từ 21/11/2026): ô cuối của notebook gói `runs/` thành `d2s-pack/runs.tar` và mỗi mục `cache/<cảnh>/<stage>@<khoá>/` thành `d2s-pack/cache/<cảnh>/<stage>@<khoá>.tar` (không nén, bỏ mục `.partial`); file rời vẫn giữ để notebook sau gắn output làm input đọc thẳng. `pull` tải `d2s-pack/` trước (bỏ gói của mục cache máy đã có), bung ra đúng cây `runs/`, `cache/` rồi ghép như trên; Nếu bước gói lỗi thì xoá toàn bộ gói đã tạo; output không có gói (notebook cũ, hoặc bước gói lỗi) thì tải file rời như trước. Gói từ chối đường dẫn hay link trỏ ra ngoài thư mục đích. Lý do: lệnh `kaggle kernels output` tải từng file một, khoảng 0,7 giây mỗi file; run `82bc` (cảnh mới, chưa có cache trên máy) tải 11.570 file mask (51 MB) mất 2 giờ 15 phút, còn gói của chúng là một file 47 MB, gói trong 1,9 giây.
- **Trạng thái lần hỏi gần nhất**: mỗi lần Studio (vòng theo dõi, tab Kaggle) hay `d2s kaggle status` hỏi được trạng thái notebook, `kaggle.json` ghi `remote_status` (`running`, `complete`...) và `checked_utc`. `run.json` trên máy vẫn `pending` tới khi tải về; Studio dựa vào `remote_status` để hiện "đang chạy", "xong trên Kaggle, chưa tải về", "chưa đẩy lên" (`frontend/src/runStatus.ts`). Run đẩy bằng tài khoản Kaggle khác không hỏi được nên hiện "trên Kaggle, chưa rõ trạng thái".
- **Tài khoản**: run dùng tài khoản mà lệnh `kaggle` đang đăng nhập (`kaggle config view`), ghi vào `kaggle.json` (`username`). Notebook, dataset đều private, nên đổi khoá API sang tài khoản khác thì `kaggle status`, `kaggle pull` báo rõ run đã đẩy bằng tài khoản nào để đổi lại khoá, và `kaggle push` không chạy tiếp run đó (tài khoản mới không đọc được notebook cũ); xoá `kaggle.json` của run để chạy lại từ đầu bằng tài khoản mới. `kaggle.json` ghi cả `kernels`: notebook của mọi lần đẩy, cũ trước mới sau, để lần đẩy sau gắn đúng output các lần trước; file ghi trước khi có trường này thì tên các lần trước dựng lại theo quy tắc cũ `d2s-<hash8 của run_id>-<lần đẩy>`.

Chưa chạy thật: cờ của lệnh `kaggle` đã kiểm với Kaggle CLI 2.2.5, còn ô cài phụ thuộc trong notebook phụ thuộc câu hỏi mở về phiên bản PyTorch (ARCHITECTURE mục 8).

### 9.3 Chạy trên Colab

Run được tạo trên máy (`executor: colab`) rồi đẩy lên một VM Google Colab bằng `d2s colab push` (`d2s run start --executor colab` tự gọi). Tool lái Colab qua lệnh `colab` (gói `google-colab-cli`, đọc cách dùng từ `colab --help` và mã nguồn bản 0.7.5); code ở `drive2sim/colab/`. Colab dùng **cùng gói job** như Kaggle (mục 9.2, `write_bundle`) và **cùng lệnh** `d2s kaggle run` trên VM, nên cùng một run cho ra cùng thư mục run và cache; chỉ khác cách đưa code, dữ liệu lên và lấy kết quả về:

```
/content/d2s/                    trên VM, mất khi phiên mất
├── repo/                        git clone --recurse-submodules --branch <branch> <url>, checkout <commit>
├── venv/                        Python 3.10 (uv venv), cài bằng envs/install_main.sh; env.done ghi khoá của lần cài
├── input/<tên dataset cảnh>/<scene_id>/   cảnh, cùng tên với dataset cảnh trên Kaggle (d2s-scene-<cảnh>-<commit>)
├── work/                        thư mục làm việc của job: data/, runs/, cache/, smpl_models/SMPL_NEUTRAL.pkl
├── bundles/<job>/               gói job: run.json, split.json, scene_ref.json ("kaggle_dataset": null), resume.json ({"kernels": []})
├── jobs/<job>.{sh,log,pid,exit} job.sh (từ drive2sim/colab/templates/job.sh), log, pid, mã thoát
├── upload/, out/<job>/          phần file đang upload; output đã gói chờ tải về
```

- **Phiên**: mặc định tên `d2s`, GPU `T4` (`--session`, `--gpu`: `T4`, `L4`, `G4`, `A100`, `H100`); phiên cùng tên đang có thì dùng lại (môi trường đã cài, cảnh đã upload), phiên đó phải có GPU. Một phiên chạy một job một lúc: `push` từ chối khi VM còn job đang chạy.
- **Code**: như Kaggle (mục 9.2), commit phải có trên remote VM clone (`origin`, hoặc biến `D2S_COLAB_REPO`). Môi trường cài vào venv Python 3.10 chứ không vào Python của Colab, vì `requirements.txt` chốt `numpy==1.23.5` chỉ có tới Python 3.11. Cài lại khi `envs/install_main.sh` hoặc `requirements.txt` đổi.
- **Cảnh**: gói thành tar (không nén, theo symlink), upload một lần cho mỗi phiên. **SMPL** (run có `prep.smpl`): `push` tìm `SMPL_NEUTRAL.pkl` trên máy như run trên máy (`D2S_SMPL_MODEL`, `<thư mục làm việc>/smpl_models/`, `<repo>/smpl_models/`) rồi upload thẳng vào `work/smpl_models/` của VM; file không đi qua git hay Drive; thiếu thì từ chối. Upload và download đi qua API nội dung của Jupyter (cả file trong một request), nên file được cắt thành phần 32 MiB, ghép lại trên VM (hoặc trên máy) và kiểm tra sha256; thư mục đích được tạo trước (API báo 500 khi thư mục chưa có). Khi `colab download` lỗi (đã gặp phiên mà API nội dung trả 500 cho mọi file, kể cả vài byte, trong khi `colab exec` vẫn chạy), phần đó được đọc trên VM qua `colab exec` và in ra base64 kèm MD5; các lần tải sau của cùng client đi thẳng đường này (`ColabClient.download`). Run không dùng Google Drive: `colab drivemount` cần người bấm đồng ý trên trang Google, không tự động được. Drive chỉ dùng để lấy dữ liệu gốc vào máy qua Studio (`drive2sim/colab/drive.py`, route `/api/colab/drive/...` ở mục 11.1): một phiên CPU mount Drive, người dùng đồng ý một lần, rồi file được đọc trên VM theo phần 32 MiB (kèm MD5) và tải về như trên.
- **Tài khoản Google**: lệnh `colab` giữ một token Google trong keyring; phiên Colab và Drive theo tài khoản đó. `drive2sim/colab/accounts.py` xem, chuyển và thêm tài khoản; phần đọc ghi token chạy trong `drive2sim/colab/account_helper.py` bằng Python của google-colab-cli (đường dẫn lấy từ dòng `#!` của lệnh `colab`), vì cần mã đăng nhập của gói đó (hàm nội bộ của bản 0.8.0; bản khác chưa kiểm tra). Token của các tài khoản đã đăng nhập lưu ở `~/.config/drive2sim/colab_accounts.json` (quyền `600`, không nằm trong thư mục làm việc). Thêm tài khoản: tool in đường dẫn đăng nhập Google, người dùng mở trong trình duyệt bất kỳ, đăng nhập và dán mã xác thực Google hiện ra; tool lưu token vào keyring của lệnh `colab` và chuyển sang tài khoản đó. Đổi tài khoản không ảnh hưởng run đang chạy trên phiên cũ, nhưng phiên đó không còn thấy được khi đã chuyển sang tài khoản khác.
- **Phiên bản google-colab-cli**: `colab version` và `colab update` (chỉ kiểm tra) cho bản đang cài và bản mới nhất; cập nhật bằng `uv tool upgrade google-colab-cli` (không có `uv` thì `colab update --install`).
- **Job** chạy **tách khỏi kết nối**: một đoạn Python gửi qua `colab exec` chép `job.sh` và gói job lên rồi chạy `bash job.sh ...` trong một session riêng, log ra `jobs/<job>.log`, mã thoát ghi vào `jobs/<job>.exit`. `job.sh` in GPU, clone, cài môi trường, chạy `d2s kaggle run bundles/<job> --workdir work --input-root input/<tên dataset cảnh>`, rồi xoá symlink trong `runs/` và `cache/` như notebook Kaggle. Tham số của job ghi thẳng vào lệnh, không dùng `colab exec --env` (biến giữ lại trong kernel cho mọi lần gọi sau). Tên job `<run_id>-<lần đẩy>`.
- **Theo dõi**: `d2s colab status` và Studio đọc trạng thái và 40 dòng log cuối bằng một `colab exec` ngắn; `d2s colab logs --follow` đọc log theo từng đoạn tới khi job xong, lần đọc lỗi thì thử lại (tối đa 20 lần liên tiếp), job vẫn chạy khi việc đọc dừng. Trạng thái job: `running`, `succeeded`, `failed` (mã thoát khác 0), `died` (tiến trình mất mà không ghi mã thoát), `lost` (phiên đã mất, hoặc phiên cùng tên là VM mới không có job này), `stopped` (tool đã tắt phiên sau khi tải kết quả về; không hỏi Colab nữa).
- `d2s colab pull` gói `runs/<...>/<run_id>` và các cache mà `run.json` trên VM dùng thành tar, cắt phần, tải về, kiểm tra sha256 rồi ghép **đúng như** `d2s kaggle pull` (mục 9.2, `drive2sim/engine/remote.py`); `colab.json` trên máy giữ nguyên. Từ chối khi job còn chạy hoặc run đang chạy trên máy. Run `succeeded` thì tắt phiên sau khi tải (trừ khi `--keep-session`, hoặc còn run khác của cùng phiên chưa tải về); run lỗi thì giữ phiên để chạy tiếp nhanh. `d2s colab stop` tắt phiên.
- **Mất phiên** (Colab thu hồi VM, hết giờ, bị tắt): mọi thứ trên VM mất, gồm kết quả chưa tải về. Lệnh nào gặp cũng báo rõ, ghi `lost_utc` vào `colab.json` và Studio thôi theo dõi run đó. Chạy tiếp: `d2s colab push` lần nữa tạo phiên mới, upload lại cảnh và **bản run trên máy** (thư mục run và cache nó dùng, nếu run đã có kết quả tải về), nên stage đã tải về không chạy lại. Phiên còn sống thì lần đẩy sau chạy tiếp từ bản trên VM (mới hơn), không upload run.
- `d2s/colab.json` trong thư mục run:

```json
{"session": "d2s", "gpu": "T4", "attempt": 1, "job": "20261112-081000-omnire-a1b2-1", "commit": "c0ffee...", "branch": "drive2sim", "pushed_utc": "2026-11-14T10:00:00+00:00", "pulled_utc": null, "lost_utc": null, "stopped_utc": null}
```

Chưa chạy thật: mọi đoạn mã chạy trên VM đã chạy trong test với một VM giả (thư mục tạm), nhưng chưa chạy trên Colab; giới hạn kích thước một request upload của Colab, thời gian cài môi trường trên VM và việc `uv venv --python 3.10` trên Colab chưa kiểm tra.

### 9.4 Ingest trên Kaggle

Dữ liệu gốc lớn (ví dụ KITTI-360 đủ khung, khoảng 8,8 GB một chuỗi) không cần nằm trên máy, và cảnh ingest từ nó không cần upload từ máy. Dữ liệu gốc là một dataset private hoặc output của một notebook trên Kaggle; `d2s kaggle ingest <nguồn> [--path <thư mục con>] [tuỳ chọn của scene ingest]` ingest ngay trên Kaggle (`drive2sim/kaggle/ingest.py`):

- **Notebook** `<người dùng>/d2s-ingest-<nguồn>[-<số của --sequence>]`, hoặc `d2s-ingest-<--scene>` khi đặt `--scene` (mục 10.1, ví dụ `d2s-ingest-nuscenes-mini-0061`), private, chỉ CPU, có internet, sinh từ `drive2sim/kaggle/templates/ingest.ipynb`. Chạy lại cùng lệnh thì dùng lại notebook: đang chạy thì chờ tiếp, đã xong thì chỉ tải về, lỗi hoặc chưa có thì đẩy (lại). Notebook ghi nguồn, `--path` và tuỳ chọn vào `ingest_result.json`; cùng tên mà khác tham số thì lệnh từ chối (đẩy lại sẽ ghi đè output mà run cũ đang dùng) và bảo đặt `--scene` khác.
- **Converter đổi đầu ra**: notebook đã xong mà `manifest.yaml` của nó là cảnh cũ (mục 2.2) thì lệnh **không** đẩy lại notebook đó (run cũ còn gắn output của nó) mà dùng notebook `<tên trên>-c<phiên bản đầu ra hiện tại>` (ví dụ `d2s-ingest-kitti360-0003-c2`), theo đúng luật dùng lại ở trên; chạy lại cùng lệnh thì đi đúng con đường đó. Notebook mới vẫn ra cảnh cũ (commit đẩy lên chưa có converter mới) thì báo lỗi. Hậu tố chỉ có khi phiên bản > 1, nên notebook đã có giữ tên. Cảnh cùng `scene_id` đã có trên máy mà là cảnh cũ thì lệnh không tự thay (split, cache, run cũ làm từ dữ liệu cũ): báo xoá cảnh cũ bằng `d2s scene delete` rồi chạy lại lệnh (notebook mới đã xong nên chỉ tải về), hoặc đặt `--scene` khác. Nguồn là dataset (`kaggle datasets` thấy nó) thì gắn qua `dataset_sources`, không thì coi là notebook và gắn qua `kernel_sources`.
- Notebook clone đúng commit như mục 9.2 (không cần submodule), cài môi trường nhẹ trong venv Python 3.10: tool, numpy 1.23.5, PyTorch 2.4.1 bản CPU (mã tiền xử lý import torch), `nuscenes-devkit` 1.2.0, OpenCV. Nó tìm thư mục nguồn theo tên dưới `/kaggle/input` (sâu tối đa ba mức), chạy `d2s scene ingest <nguồn>/<--path> --workdir /kaggle/working <tuỳ chọn>`, rồi chỉ giữ `data/<scene_id>/` và `ingest_result.json` (`scene_id`, `commit`, `exit_code`, `source`, `path`, `args`) trong output. Mã thoát 1 (báo cáo kiểm tra có lỗi) vẫn giữ cảnh để xem báo cáo.
- Lệnh chờ notebook xong (hỏi mỗi 60 giây, tối đa 6 giờ), đọc **chỉ** kết quả và **file nhỏ của cảnh** (`drive2sim/kaggle/scene_files.py`: file text ở cấp gốc như `manifest.yaml`, `frame_ids.txt`, cộng `reports/*.json`, `intrinsics/`, `extrinsics/`, `ego_pose/`, `lidar_pose/`; tối đa 32 MB) từ **log** của notebook (`kaggle kernels logs`): notebook ghi chúng vào stderr của tiến trình sau chữ `D2S_INGEST_LIGHT`, dạng base64 của JSON `{"result", "files"}` chia thành các dòng `D2S_INGEST_LIGHT <i>/<n> <đoạn>` dài tối đa 4000 byte, mỗi dòng một lần ghi (`write_log_lines`): ghi nguyên tử nên stdout không chen vào giữa; máy ghép theo số thứ tự, thiếu đoạn hoặc hỏng thì đẩy lại notebook. Log của notebook cũ (một dòng) vẫn đọc được. Không dùng `kaggle kernels output --file-pattern`, vì lệnh đó liệt kê mọi file của output trước khi lọc, và với cảnh lớn (KITTI-360 0003: khoảng 17 000 file) bị Kaggle trả 429. Notebook đã xong mà log không có dòng này (template cũ) thì đẩy lại. Ghi vào `data/<scene_id>/` cùng `kaggle_scene.json`; file đã có trên máy không bị ghi đè. Viewer ở chế độ renderer gốc (Render GPU, ảnh chính xác của Gói xem trước) không đọc ảnh, LiDAR, chỉ cần pose, calib, nên xem được model của cảnh này trên máy. Notebook private, nên mỗi tài khoản Kaggle cần notebook ingest của riêng nó: chạy lệnh bằng tài khoản thứ hai (dữ liệu gốc là dataset đã chia sẻ cho tài khoản đó) thì thêm một mục vào `kaggle_scene.json`, với điều kiện manifest giống cảnh đã có (bỏ qua phiên bản converter); cảnh cùng tên mà ingest trên máy, hoặc manifest khác, thì từ chối. Mã thoát 1 nếu báo cáo có lỗi.
- `kaggle push` của run trên cảnh có `kaggle_scene.json` không upload cảnh: notebook train gắn output của notebook ingest qua `kernel_sources` (trước các notebook chạy tiếp), `kaggle.json` ghi notebook đó ở trường `dataset`; `d2s kaggle run` tìm cảnh theo `manifest.yaml` như mục 9.2. `push` lấy notebook ingest của đúng tài khoản lệnh `kaggle` đang đăng nhập; tài khoản đó chưa ingest thì từ chối và chỉ lệnh cần chạy.
- Cảnh như vậy không chạy được trên máy hay Colab (máy không có ảnh): `run start` với `--executor local` hoặc `colab` báo lỗi, `scene validate` cũng vậy.
- **Lấy bù pose, calib** cho cảnh ingest trước khi notebook in chúng (máy chỉ có `manifest.yaml`, `reports/`): `d2s kaggle scene-files <scene_id>` đẩy notebook CPU `<người dùng>/d2s-files-<phần sau d2s-ingest->` (sinh từ `templates/scene_files.ipynb`) gắn output notebook ingest của tài khoản đang đăng nhập qua `kernel_sources`. Notebook tìm cảnh theo `manifest.yaml` dưới `/kaggle/input`, in file nhỏ sau chữ `D2S_SCENE_FILES` (base64 của JSON `{"scene_id", "files"}`), không ingest lại, không ghi output; lệnh đọc log rồi ghi file còn thiếu. Dùng lại notebook đã xong; cảnh đã đủ thì không đẩy gì. Render GPU thiếu pose, calib thì báo đúng lệnh này.

```json
{"kernels": [{"username": "alice", "kernel": "alice/d2s-ingest-nuscenes-mini-0061", "source": "alice/d2s-nuscenes-mini-raw", "commit": "c0ffee...", "pulled_utc": "2026-11-15T10:00:00+00:00"}]}
```

Đã chạy thật 15/11/2026 với nuScenes `scene-0061` (ô cài môi trường chạy được trên CPU Kaggle). Chưa train run nào từ cảnh ingest trên Kaggle.

## 10. Quy tắc đặt tên và CLI

### 10.1 Định danh

| Định danh | Mẫu | Ví dụ |
|---|---|---|
| `scene_id` | `<dataset>_<sequence_id>`; chỉ gồm `a-z 0-9 _ -`, chữ thường | `nuscenes_scene-0061`, `waymo_10017090168044687777` |
| `split_id` | `every<N>_<nhóm camera>` với chiến lược `every_nth_frame:N`; `custom_<tên>` với chiến lược `custom`. Nhóm camera: `allcams` (mọi camera trong manifest), `front1` / `front3` (camera trước và hai bên), hoặc `<k>cams` (k camera theo thứ tự trong manifest). Cảnh có cả camera pinhole lẫn mắt cá (theo `CAMERA_TRACK`, ví dụ KITTI-360) thì tên tự đặt ghi loại ống kính của các camera đã chọn và số camera: `pinhole<k>`, `fisheye<k>` hoặc `mixed<k>` (từ 16/11/2026; split đã tạo giữ tên cũ) | `every8_allcams`, `every10_front3`, `every8_pinhole2`, `every8_mixed4`, `custom_night_a` |
| `approach_id` | `snake_case`, tên method, không chứa lựa chọn của mục cấu hình | `omnire`, `streetgs` |
| Tên mục cấu hình | `snake_case`, danh từ ngắn | `init`, `renderer`, `campose`, `fastgs` |
| Tên preset | `snake_case`, mô tả mục đích | `paper`, `no_lidar`, `fisheye` |
| `stage_id` | `<nhóm>.<tên>`; nhóm là một trong `prep`, `init`, `train`, `eval`, `render`, `export`; tên là `snake_case` | `prep.masks_segformer`, `init.depth`, `export.ply_world` |
| `run_id` | `<YYYYMMDD-HHMMSS>-<approach_id>-<4 ký tự hex>` | `20261112-081000-omnire-a1b2` |
| `cache_key` | `<scene_id>/<stage_id>@<version>-<hash8>` | `nuscenes_scene-0061/init.depth@0.1.0-9b03c2d1` |
| Id camera, LiDAR | Giữ nguyên id của nguồn, không đổi tên | `CAM_FRONT`, `camera_front_wide_120fov` |
| `schema_version` | `d2s-<loại>/<major>.<minor>` | `d2s-run/0.1` |
| Notebook train trên Kaggle | `d2s-<cảnh>-<approach>-<biến thể>-<4 hex của run>-<lần đẩy>`. `<cảnh>` là `<dataset>-<số cuối của chuỗi gốc>`; `<biến thể>` là lựa chọn của mọi mục chọn (renderer luôn có) cộng các mục bật/tắt khác mặc định (bật: tên mục, tắt: `no-<tên>`), `base` nếu không có gì. Tối đa 50 ký tự (giới hạn của Kaggle): quá thì bỏ tiền tố `d2s-`, rồi bỏ nguyên từ ở cuối biến thể (`drive2sim/kaggle/names.py`) | `d2s-nuscenes-0061-omnire-3dgs-fastgs-453a-1`, `d2s-nuscenes-0061-omnire-3dgeer-fastergs-8d9c-2` |
| Dataset cảnh trên Kaggle, thư mục cảnh trên VM Colab | `d2s-scene-<cảnh>-<commit converter>` | `d2s-scene-nuscenes-0061-96876ad` |
| Notebook ingest trên Kaggle | `d2s-ingest-<nguồn bỏ d2s-, raw>[-<số của --sequence>]`, hoặc `d2s-ingest-<--scene>`; thêm `-c<phiên bản đầu ra>` khi notebook tên đó ingest bằng converter cũ (chỉ khi phiên bản > 1, mục 9.4) | `d2s-ingest-nuscenes-mini-0061`, `d2s-ingest-kitti360-0003`, `d2s-ingest-kitti360-0003-c2` |
| Dataset dữ liệu gốc trên Kaggle (người dùng tạo) | `d2s-raw-<dataset>-<phần>` | `d2s-raw-kitti360-0003`, `d2s-raw-nuscenes-mini` |
| Dataset mô hình trên Kaggle (người dùng tạo, có bản quyền riêng) | `d2s-model-<tên>` | `d2s-model-smpl` (đặt vào `D2S_KAGGLE_SMPL_DATASET`) |

### 10.2 File và thư mục

- File người viết tay hoặc sửa tay dùng YAML: `manifest.yaml`, `<approach_id>.yaml`. File tool ghi ra dùng JSON: `run.json`, `summary.json`, `env.json`, `artifact.json`, `validation.json`, file split.
- Mảng số dùng NPZ; mask PNG; video MP4 H.264; Gaussian PLY.
- Tên file và thư mục: `snake_case`, chữ thường. Thư mục của một stage đặt đúng theo `stage_id`.
- Mã Python: module `snake_case`; lớp `PascalCase`; lớp stage đặt theo `<Nhóm><Tên>Stage`, ví dụ `PrepMasksSegformerStage` cho `prep.masks_segformer`. Quy tắc đặt tên đầy đủ cho module, lớp, hàm, biến, tham số ở `AGENTS.md` mục 4.2.
- Package Python có `__init__.py`; thư mục chưa có file giữ trong git bằng `.gitkeep`, xoá `.gitkeep` khi thư mục đã có file.
- File môi trường trong `envs/`: `main.yml` cho môi trường chính, `<codebase>.yml` cho codebase ngoài.
- Config method (`drive2sim/configs/methods/`): `<method>.yaml` là config gốc; biến thể là `<method>_<biến thể>[_<biến thể>].yaml`, các biến thể xếp theo thứ tự cố định **renderer → densify → tăng tốc → campose** (`geer`, `ut`; `fastgs`; `fastergs`; `nocampose`), ví dụ `streetgs_geer_fastgs.yaml`, `omnire_fastergs_nocampose.yaml`. Biến thể kế thừa file cha bằng `extends` và chỉ ghi phần khác (mục 10.5); `tests/drive2sim/core/test_config.py` (`VARIANTS`) ghi khoá nào mỗi biến thể được phép khác.
- Config dataset (`drive2sim/configs/datasets/<dataset>/`): `<k>cams.yaml`; biến thể `<k>cams_<biến thể>.yaml`, ví dụ `7cams_native.yaml` (giữ mô hình camera gốc, không khử méo).
- Mask thân xe: `drive2sim/assets/ego_masks/<dataset>/<chỉ số camera>.png`, chỉ số là vị trí camera trong danh sách `cameras` của manifest (bắt đầu từ 0). Mỗi dataset có một bảng chỉ số ↔ `camera_id` trong `drive2sim/recon/data/dataset_meta.py`, trừ KITTI-360: loader của nó lấy `camera_id`, mô hình và cỡ camera từ `manifest.yaml` (chỉ số là vị trí trong `cameras`); code mới dùng `camera_id`, chỉ đổi sang chỉ số ở biên với mã DriveStudio.
- File bên trong một artifact (mục 5.4): `videos/<camera_id>_<loại>.mp4` với loại `gt`, `render`, `diff`, `side_by_side`; `eval_report` gồm `summary.json` và metric gốc của codebase (`drivestudio_metrics.json` với DriveStudio); `masks` là PNG như DriveStudio, `depth_prior` là `.npy`; `smpl/<track_id>.npz`.
- Log: `d2s/log.txt` trong thư mục run là log của engine cho cả run, gồm traceback của stage lỗi. Stage chạy qua worker (mục 9.1) có thêm `log.txt` trong thư mục kết quả của stage, ghi stdout và stderr của worker.

### 10.3 Tên trường và tham số

- `snake_case` cho mọi khoá trong YAML, JSON và tham số stage.
- Hậu tố đơn vị bắt buộc khi trường có đơn vị: `_us` (micro giây), `_s` (giây), `_m` (mét), `_mb` (megabyte), `_px` (pixel). Góc lưu bằng radian, không hậu tố; độ chỉ dùng trong giao diện, hậu tố `_deg`.
- Phép biến đổi: `T_<a>_<b>` theo quy ước NCore (mục 1), ví dụ `T_camera_world`, `T_object_world`.
- Boolean: tính từ hoặc dạng `uses_`, `has_`: `cacheable`, `uses_lidar`, `has_cuboids`. Mục cấu hình bật/tắt dùng `true` / `false`.
- Giá trị liệt kê: chữ thường `snake_case`; mô hình camera: `opencv_pinhole`, `opencv_fisheye`, `ftheta`; renderer: `3dgs`, `3dgut`, `3dgeer` (engine đổi sang `render_mode` của DriveStudio: `default`, `ut`, `geer`).
- Định danh trong tham chiếu: `scene_id`, `split_id`, `approach_id`, `stage_id`, `run_id`, `cache_key`; không dùng `id` trơn trong file có nhiều loại định danh.
- Thời điểm tuyệt đối: chuỗi ISO 8601 theo UTC, kết thúc bằng `Z`, tên trường kết thúc bằng `_utc` (`created_utc`). Thời điểm trong dữ liệu cảm biến vẫn là số nguyên micro giây `_us` (mục 1).
- Đường dẫn trong file JSON/YAML do tool ghi: **tương đối với `D2S_WORKDIR`**, dùng dấu `/`, không bắt đầu bằng `/` hay `./` (`runs/nuscenes_scene-0061/...`). Nhờ vậy chép cả thư mục làm việc sang máy khác là dùng được (`ARCHITECTURE.md` mục 2.6). Đường dẫn tuyệt đối chỉ xuất hiện trong log.
- Giá trị `--set <khoá>=<giá trị>` và giá trị mục cấu hình từ CLI được phân tích như YAML: `true`/`false` thành boolean, `10` thành số nguyên, `0.5` thành số thực, `a,b,c` thành danh sách chuỗi, phần còn lại là chuỗi; muốn giữ chuỗi `"10"` thì đặt trong ngoặc kép. Sau khi phân tích, giá trị được kiểm tra theo `Params` của stage hoặc `choices` của mục cấu hình; sai kiểu là lỗi mã thoát `2`.
- Số thực trong metric ghi đủ độ chính xác của `float32` (tối đa 7 chữ số có nghĩa), không làm tròn thêm; giá trị không tính được ghi `null`, không ghi `NaN`.

### 10.4 CLI

Lệnh có dạng `d2s <nhóm> <hành động> [tuỳ chọn]`. Tuỳ chọn dài viết `kebab-case`; giá trị khớp với khoá `snake_case` tương ứng.

| Lệnh | Việc |
|---|---|
| `d2s source inspect <thư mục>` | Xem trước thư mục dữ liệu gốc: nhận ra dataset, báo có gì, gợi ý lệnh ingest; không ghi gì |
| `d2s scene ingest <nguồn> [--source-type <dataset>] [--version <phiên bản>] [--sequence <chuỗi gốc>] [--scene <scene_id>] [--interpolate-n <N>] [--workers <N>] [--start-frame <số hiệu>] [--num-frames <N>] [--cameras <id,...>] [--max-gap <N>]` | Chuyển một chuỗi của nguồn sang `data/<scene_id>/`, ghi manifest và báo cáo kiểm tra; bỏ `--source-type` thì tự nhận dạng như `source inspect`. `scene_id` mặc định `<dataset>_<chuỗi gốc>`. Mã thoát `1` nếu báo cáo có lỗi. Hỗ trợ nuScenes, Waymo, Argoverse 2, PandaSet, KITTI, nuPlan, PhysicalAI, KITTI-360; phiên bản và chuỗi được tự chọn khi chỉ có một. Bốn tuỳ chọn cuối chỉ dùng cho KITTI-360: số hiệu khung gốc nhỏ nhất, số khung tối đa (mặc định cả đoạn liền nhau dài nhất), camera cần ghi, khoảng cách lớn nhất giữa hai khung liền nhau (mặc định 5) |
| `d2s scene validate <scene_id>` | Kiểm tra lại, ghi `validation.json`; cảnh ingest bằng converter cũ có mục cảnh báo `converter.outdated` (mục 3.2) |
| `d2s scene list` / `d2s scene show <scene_id>` | Liệt kê cảnh; xem manifest và báo cáo |
| `d2s scene delete <scene_id>` | Xoá dữ liệu chuẩn, split, cache và run của cảnh; hỏi xác nhận |
| `d2s scene rig <scene_id>` | Ghi `reports/rig.json` (mục 3.3) cho cảnh đã ingest, rồi kiểm tra lại; cần PyTorch |
| `d2s scene lidar-check <scene_id>` | Chiếu LiDAR lên ảnh, ghi `reports/lidar_check.json`, `reports/lidar_overlay/` (mục 3.4) rồi kiểm tra lại; mã thoát 1 khi có cảnh báo `projection.*`; cần PyTorch và ảnh trên máy |
| `d2s split create <scene_id> --strategy every_nth_frame:8 [--cameras <id,...>] [--frame-range <đầu>,<cuối>] [--split <split_id>]` | Tạo split; mặc định mọi camera và cả cảnh, tên theo mục 10.1 |
| `d2s split list <scene_id>` / `d2s split show <scene_id> <split_id>` / `d2s split delete <scene_id> <split_id>` | Liệt kê, xem, xoá split |
| `d2s approach list` / `d2s approach show <approach_id>` | Liệt kê approach; xem mục cấu hình, preset, ràng buộc |
| `d2s approach check <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <mục>=<giá trị>]... [--quick]` | Kiểm tra điều kiện và in danh sách stage sẽ chạy, stage nào dùng lại cache; cảnh ingest bằng converter cũ (mục 2.2) thì in cảnh báo ra stderr, không chặn (`run start` cũng vậy) |
| `d2s run start <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <mục>=<giá trị>]... [--quick] [--executor local\|kaggle\|colab] [--no-push]` | Chạy; `--quick` là chạy thử nhanh 3000 iteration (mục 6.2); `--no-push` (executor kaggle, colab) chỉ tạo run, đẩy sau, ví dụ hai run chung một notebook bằng `d2s kaggle push RUN1 RUN2` |
| `d2s run resume <run_id>` | Chạy tiếp từ stage dở |
| `d2s run status <run_id>` / `d2s run list [--scene <scene_id>]` / `d2s run cancel <run_id>` / `d2s run delete <run_id>` | Theo dõi, liệt kê, huỷ, xoá thư mục run |
| `d2s kaggle push <run_id> [<run_id>] [--branch <branch>] [--accelerator <GPU>]` / `d2s kaggle status <run_id>` / `d2s kaggle pull <run_id> [--light]` | Đẩy run lên Kaggle (lần sau chạy tiếp; hai run thì chung một notebook, mỗi run một GPU), xem trạng thái notebook, tải kết quả về (mục 9.2); `push` do `run start --executor kaggle` tự gọi |
| `d2s kaggle ingest <nguồn> [--path <thư mục con>] [--branch <branch>] [tuỳ chọn của scene ingest]` | Ingest một cảnh trên Kaggle từ dữ liệu gốc là dataset hoặc output notebook trên Kaggle; chờ xong rồi ghi file nhỏ của cảnh về `data/<scene_id>/` (mục 9.4) |
| `d2s kaggle scene-files <scene_id> [--branch <branch>]` | Tải pose, calib của cảnh ingest trên Kaggle về máy bằng notebook CPU, không ingest lại (mục 9.4), để xem model bằng Render GPU |
| `d2s kaggle run <gói> [--input-root <thư mục>] [--prepare-only]` | Chỉ dùng trên máy chạy (notebook Kaggle, job trên VM Colab): dựng thư mục làm việc từ gói job rồi chạy run; `--prepare-only` chỉ dựng (notebook hai run) |
| `d2s colab push <run_id> [--branch <branch>] [--gpu <GPU>] [--session <tên>]` / `d2s colab status <run_id>` / `d2s colab logs <run_id> [--follow]` / `d2s colab pull <run_id> [--keep-session]` / `d2s colab stop <run_id>` | Chạy run trên VM Colab (mục 9.3): đẩy và chạy job tách khỏi kết nối, xem trạng thái và log cuối, đọc log tới khi xong, tải kết quả về (run xong thì tắt phiên), tắt phiên; `push` do `run start --executor colab` tự gọi |
| `d2s colab account list` / `d2s colab account use <email>` / `d2s colab account forget <email>` / `d2s colab account login` / `d2s colab update` | Tài khoản Google của lệnh `colab` (mục 9.3): liệt kê (dấu `*` là tài khoản đang dùng), chuyển, bỏ khỏi danh sách, thêm (in đường dẫn đăng nhập Google rồi hỏi mã xác thực trong terminal); cập nhật google-colab-cli |
| `d2s report <run_id>...` | Bảng so sánh nhiều run |
| `d2s benchmark <file.yaml> [--out <file.json>]` | Bảng benchmark theo file định nghĩa (mục 6.5): xếp hạng từng nhóm run, chọn run thắng, liệt kê đoạn đường hỏng; chỉ đọc file kết quả của run, không chạy GPU |
| `d2s pack <run_id>... --output <file.tar> [--dry-run]` | Gói đầy đủ các run đã chọn, gồm mọi checkpoint, cảnh chuẩn và cache được tham chiếu. `--dry-run` chỉ kiểm kê đường dẫn và dung lượng. Không ghi đè file có sẵn, không gói run đang chạy. Kèm mô hình SMPL cần thiết tại smpl_models/SMPL_NEUTRAL.pkl; bỏ view/viewer tạm có thể dựng lại và thông tin đăng nhập; không lấy dữ liệu gốc ngoài cảnh chuẩn. |
| `d2s export ply <run_id>` / `d2s export preview <run_id>` | Chạy lại stage export cho một run |
| `d2s cache list [--scene <scene_id>]` / `d2s cache gc [--scene <scene_id>]` | Xem, dọn cache |
| `d2s studio [--port <số>] [--kaggle-watch-minutes <phút>] [--colab-watch-minutes <phút>]` | Bật Studio. Studio cứ `--kaggle-watch-minutes` phút (mặc định 5; `0` để tắt) hỏi trạng thái notebook của các run đã đẩy lên Kaggle mà chưa tải kết quả lần đẩy mới nhất; notebook xong (thành công, lỗi, huỷ) thì tự xếp việc `kaggle_pull` như nút "Tải kết quả về". Không có run nào cần theo dõi thì không gọi Kaggle; hỏi lỗi thì khoảng chờ nhân đôi, tối đa 1 giờ. `--colab-watch-minutes` (mặc định 5) làm như vậy với job trên Colab (việc `colab_pull` khi job `succeeded`, `failed` hoặc `died`; phiên mất thì thôi theo dõi run đó) |
| `d2s serve <run_id> [--port <số>]` | Bật server gRPC cho một run (L4) |

Quy tắc chung:
- Viewer SMPL dùng hướng SH đơn vị dự phòng cho điểm trùng camera (norm ≤ 1e-8); người vắng mặt vẫn có opacity 0, không sửa checkpoint hay giảm số Gaussian.
- Gói bàn giao là tar không nén, giữ đường dẫn tương đối với workdir, có `README.md` và `pack.json` (`format_version: 1`, `run_ids`, `scene_ids`, `cache_keys`, danh sách `files` gồm `path`, `size_bytes`, `sha256`). Symlink được đổi sang tương đối và chỉ được trỏ vào dữ liệu đã chọn trong gói. Ghi qua file tạm, kiểm dung lượng trước, chỉ xuất file cuối khi hoàn tất. Giải nén vào thư mục trống rồi dùng `--workdir` hoặc `make studio WORKDIR=...`.
- Lựa chọn stage `none` bỏ bước đó khỏi plan. `Stage.unavailable_reason(manifest, split=None)` có thể khoá stage theo split; khi chưa có split, điều kiện theo split chưa được xét. Train không nhận artifact SMPL thì tắt `load_smpl`.
- Run local thiếu môi trường trả CLI `3`, API `422`, mã `env_missing`; thông báo có hướng dẫn cài và executor từ xa. Source inspect trả thêm `errors`, `missing_sweeps`, `candidates`; ingest nuScenes nội suy bị từ chối nếu thiếu sweeps, nhưng `--interpolate-n 0` vẫn được phép.
- Mẫu lệnh là `d2s <nhóm> <hành động>`. Nhóm chỉ có một hành động thì là lệnh đơn (`report`, `studio`, `serve`); khi nhóm đó có hành động thứ hai thì tách thành `<nhóm> <hành động>` và giữ lệnh đơn cũ làm bí danh một phiên bản. `d2s-worker` là entrypoint riêng vì chạy trong môi trường khác (mục 9.1).
- Đối tượng chính đi theo vị trí (`<run_id>`, `<scene_id>`); các định danh khác đi theo tuỳ chọn (`--scene`, `--split`). Lệnh về split nhận `<scene_id> <split_id>` theo thứ tự cha trước con.
- Tuỳ chọn chung cho mọi lệnh: `--workdir <thư mục>` (ưu tiên hơn `D2S_WORKDIR`), `--json`, `--verbose` / `--quiet`, `--yes` (bỏ hỏi xác nhận), `--help`; viết **sau** tên lệnh (`d2s run list --json`). `d2s --version` in phiên bản tool.
- **Không dùng cờ ngắn** (`-s`, `-v`), trừ `-h` cho `--help`. Tuỳ chọn dài đọc là hiểu, không phải tra bảng; thêm cờ ngắn sau này không phá gì vì chỉ là bí danh.
- Tham số danh sách: một chuỗi cách nhau bằng dấu phẩy, không có khoảng trắng (`--cameras CAM_FRONT,CAM_BACK`). Chỉ `--set` lặp lại tuỳ chọn, vì mỗi lần là một cặp khoá và giá trị.
- `--set <mục>=<giá trị>` lặp lại được, dùng cho cả mục cấu hình và tham số stage (`--set train.drivestudio.num_iters=10000`). Kiểu giá trị theo mục 10.3.
- Lệnh xoá (`scene delete`, `split delete`, `run delete`, `cache gc`) hỏi xác nhận, in ra những gì sẽ xoá trước; `--yes` bỏ hỏi. Lệnh xoá không xoá gì ngoài `D2S_WORKDIR`.
- Kết quả có cấu trúc (bảng khi không có `--json`, JSON khi có) đi ra **stdout**; log, tiến độ và cảnh báo đi ra **stderr**. Nhờ vậy `d2s run list --json | jq` luôn nhận JSON sạch. `--json` in một đối tượng hoặc một mảng JSON, trường `snake_case`, cùng tên với file trong CONTRACTS.
- Thông báo lỗi bằng tiếng Việt (mục 10.6), một dòng đầu nói việc gì hỏng, các dòng sau nói vì sao và làm gì tiếp; kèm `error.code` như API (mục 11.3) khi có `--json`.
- Mã thoát: `0` thành công; `1` lỗi khi chạy; `2` sai cú pháp hoặc sai kiểu giá trị; `3` bị khoá vì không thoả điều kiện hoặc ràng buộc; `130` bị ngắt bằng Ctrl-C.
- Biến môi trường: `D2S_WORKDIR` (thư mục chứa `data/`, `cache/`, `runs/`, và `uploads/` khi có dữ liệu gốc tải lên qua Studio, `logs/` khi cài môi trường từ Studio, `compare/` khi ghép video ở trang So sánh; mặc định là thư mục hiện tại) và `D2S_DEVICE` (thiết bị chạy model trong stage, ví dụ `cuda:0`, `cpu`; mặc định `cuda:0`). `D2S_DEVICE` không phải tham số stage vì không được nằm trong khoá cache: mask sinh trên GPU hay CPU là như nhau. Không có biến môi trường nào khác; mọi cấu hình khác đi qua tuỳ chọn hoặc file.

### 10.5 Config và kế thừa

Áp dụng cho config method và config dataset trong `drive2sim/configs/` (file YAML người viết tay).

- Một config có thể khai báo `extends: <tên file>` ở cấp ngoài cùng; tên file tính tương đối với file đang đọc và chỉ được trỏ vào file **trong cùng thư mục** (`extends: streetgs.yaml`). Kế thừa được nhiều cấp; vòng lặp là lỗi.
- Gộp theo quy tắc: mapping gộp sâu theo khoá; mọi giá trị khác (số, chuỗi, boolean, **danh sách**) ở file con thay hẳn giá trị ở file cha; khoá có giá trị `null` ở file con là xoá khoá đó. Danh sách không gộp vì thứ tự và số phần tử của danh sách thường có nghĩa (ví dụ các mốc lịch learning rate).
- Engine gộp `extends` xong rồi mới đưa config cho DriveStudio (OmegaConf). Khoá `extends` bị bỏ khỏi config đã gộp; `config.yaml` ghi trong thư mục run là bản đã gộp, không còn `extends`.
- File con chỉ ghi phần khác với file cha. Lý do: sửa một tham số chung thì sửa ở một chỗ; đọc file con là thấy ngay biến thể khác gì.
- Lỗi khi đọc config (file cha không có, vòng lặp, YAML sai) báo kèm tên **cả hai** file và dừng ngay trước khi tạo thư mục run, để không để lại run rỗng.
- Config dataset vẫn chọn qua khoá `dataset: <dataset>/<k>cams` như DriveStudio; không dùng `extends` giữa config method và config dataset.
- Cài ở `drive2sim/core/config.py` (`load_config`). YAML được đọc giống hệt OmegaConf 2.3 (`1e-15` là số thực, không có kiểu timestamp, trùng khoá là lỗi), nên config gộp cho cùng giá trị như `OmegaConf.load`. `train.drivestudio` gộp sẵn, ghi bản đầy đủ vào `method_config.yaml` trong thư mục stage và đưa bản đó cho DriveStudio; config sai bị báo khi engine kiểm tra tham số, trước khi tạo run. `drive2sim/recon/tools/train.py` cũng đọc qua `load_config`, nên chạy tay với một file có `extends` vẫn được.

### 10.6 Ngôn ngữ

- **Tiếng Việt** cho mọi thứ người đọc: docstring, comment trong code và trong YAML mới, thông báo của CLI và của worker (`error.message`, `log`), tài liệu trong `docs/`, giao diện Studio mặc định.
- Ngoại lệ: thông báo của `ValueError` trong `drive2sim/core/` (identifiers, schema, files) là thông báo kỹ thuật cho lập trình viên, giữ tiếng Anh. CLI và API bắt các lỗi này và bọc thành `error.message` tiếng Việt kèm tên file và trường sai; không in thẳng `ValueError` cho người dùng.
- **Tiếng Anh** cho mọi thứ máy đọc hoặc chia sẻ với mã nguồn mở: tên biến, hàm, lớp, module, khoá YAML/JSON, `error.code`, tên sự kiện, tên lệnh và tuỳ chọn CLI, commit message, README gốc của DriveStudio và hướng dẫn dataset kế thừa từ upstream.
- File kế thừa từ DriveStudio (`drive2sim/recon/`, config method cũ, `docs/datasets/`) giữ comment tiếng Anh cho tới khi được sửa tới; phần sửa mới viết tiếng Việt.
- Studio có hai ngôn ngữ, mặc định tiếng Việt, chuyển sang tiếng Anh bằng nút trên giao diện; lựa chọn lưu trong trình duyệt. Chuỗi giao diện đặt trong `drive2sim/studio/frontend/src/locales/vi.json` và `en.json`, khoá dạng `<màn hình>.<thành phần>.<tên>` (`runs.table.status`); không viết chuỗi cứng trong component. Thông báo từ API (`error.message`) đi thẳng ra giao diện và chỉ có tiếng Việt; `error.code` là khoá để giao diện tra bản dịch khi đang ở tiếng Anh.

## 11. API của Studio

FastAPI phục vụ REST và Server-Sent Events, chỉ nghe trên `127.0.0.1` (`ARCHITECTURE.md` mục 2.10). API chỉ là một lớp mỏng trên engine: mọi lệnh CLI ở mục 10.4 có một route tương ứng, cùng bước kiểm tra, cùng mã lỗi.

### 11.1 Route

- Tiền tố `/api/`. Nhóm route trùng nhóm lệnh CLI, viết **số nhiều**: `/api/scenes`, `/api/splits`, `/api/approaches`, `/api/runs`, `/api/cache`, `/api/kaggle`, `/api/colab`, `/api/report`. Định danh đi trong đường dẫn: `/api/runs/{run_id}`, `/api/scenes/{scene_id}/splits/{split_id}`.
- HTTP verb theo loại hành động:

| Hành động | Verb và đường dẫn | Ví dụ |
|---|---|---|
| Liệt kê, xem | `GET` | `GET /api/runs?scene_id=...`, `GET /api/runs/{run_id}` |
| Tạo | `POST` vào collection | `POST /api/runs` (thân là approach, scene, split, preset, options, executor) |
| Hành động trên một đối tượng | `POST /…/{id}/<hành động>` | `POST /api/runs/{run_id}/cancel`, `POST /api/runs/{run_id}/resume` |
| Kiểm tra trước khi chạy | `POST` vào `/check` | `POST /api/approaches/{approach_id}/check` (tương ứng `d2s approach check`) |
| Xoá | `DELETE` | `DELETE /api/runs/{run_id}` |
| File tĩnh (video, ảnh, PLY, gói xem trước) | `GET /files/<đường dẫn tương đối với D2S_WORKDIR>` | `GET /files/runs/.../videos/CAM_FRONT_render.mp4` |

- Không dùng `PUT` hay `PATCH`: tool không sửa tại chỗ file nào ngoài qua một hành động có tên.
- Tham số lọc của `GET` đi theo query string, tên `snake_case` trùng tên trường (`?scene_id=`, `?status=`).
- Frontend tĩnh phục vụ ở `/`. Viewer GPU (viser) không đi qua Studio: nó nghe `127.0.0.1` ở cổng riêng, `GET /api/gpu-viewer` trả `url`, trang Xem 3D nhúng bằng iframe.

Route đã có (`drive2sim/studio/api/routes/`):

| Route | Việc | Tương ứng CLI |
|---|---|---|
| `GET /api/scenes`, `GET /api/scenes/{scene_id}` | Liệt kê cảnh; manifest, `validation` (báo cáo kiểm tra hoặc `null`), `splits` (danh sách `split_id`), `split_cameras` (camera của từng split, `{split_id: [camera_id]}`; giao diện hiện loại ống kính và xét approach theo camera của split, như engine), `kaggle` (danh sách notebook nạp cảnh trong `kaggle_scene.json` khi cảnh được nạp trên Kaggle, máy không có ảnh; ngược lại `null`), `converter_outdated` (`{"output_version", "current_output_version"}` khi cảnh được ingest bằng converter cũ, mục 2.2; ngược lại `null`) | `scene list`, `scene show` |
| `GET /api/scenes/{scene_id}/lidar`, `POST` cùng đường dẫn | `{"check": nội dung lidar_check.json | null, "overlays": [{"frame", "camera_id", "file", "url"}]}`; `POST` chiếu lại rồi kiểm tra lại (`409` với cảnh chỉ có trên Kaggle hoặc không có LiDAR) | `scene lidar-check` |
| `GET /api/scenes/{scene_id}/rig`, `POST` cùng đường dẫn | `{"rig": nội dung reports/rig.json | null}`; `POST` tạo lại file (vài chục giây, `500` kèm log khi lỗi) | `scene rig` |
| `GET /api/scenes/{scene_id}/frames/{frame_index}` | `{"frame_index", "images": [{"camera_id", "url"}]}` cho lưới ảnh đồng bộ; `url` là `null` khi thiếu ảnh | |
| `GET`, `POST /api/scenes/{scene_id}/splits`, `GET .../splits/{split_id}` | Liệt kê, tạo, xem split; thân của `POST`: `strategy`, `cameras`, `frame_range`, `split_id` | `split list`, `split create`, `split show` |
| `GET /api/approaches`, `GET /api/approaches/{approach_id}` | Đọc file approach | `approach list`, `approach show` |
| `POST /api/approaches/{approach_id}/check` | Thân: `scene_id`, `split_id`, `preset`, `options`, `stage_params`, `quick`. Trả `quick`, `options` đã quyết định, `lidar_usage`, `stages` (kèm `cache_hit`) và `blocked_choices` (mục → lựa chọn bị khoá kèm lý do) để giao diện làm mờ | `approach check` |
| `POST /api/runs` | Thân như `check` cộng `approach_id`, `executor`; tạo run, xếp vào hàng đợi chạy nền (`local`), đẩy lên Kaggle ở nền (`kaggle`) hoặc đẩy lên Colab ở nền với phiên `d2s`, GPU T4 (`colab`) | `run start` |
| `GET /api/runs`, `GET /api/runs/{run_id}` | Liệt kê (lọc `scene_id`, `status`); `run.json` cộng `remote`, `train_params`, `summary` và `run_dir`. `train_params`: tham số stage train khác mặc định mà cấu hình approach không nói lên (bỏ `method`, `num_iters`, tên trùng lựa chọn của approach, cờ đo đạc `densify_stats`, `profile_at`, `profile_steps`; `overrides` trải ra từng khoá), ví dụ `{"refine_interval": 50}`; giao diện hiện cạnh cấu hình để hai run cùng cấu hình approach phân biệt được. `remote` chỉ có ở run Kaggle chưa tải về (`status` `pending`, ngược lại `null`): `{"pushed", "status", "checked_utc"}`, trong đó `status` là trạng thái notebook lần hỏi gần nhất (`kaggle.json`, `remote_status`), để Studio hiện run chưa đẩy, đang chạy hay đã xong trên Kaggle | `run list`, `run status` |
| `POST /api/runs/{run_id}/resume` | Chạy tiếp; `409` nếu run đang chạy | `run resume` |
| `GET /api/runs/{run_id}/events`, `GET /api/events` | Sự kiện SSE (mục 11.4) | |
| `GET /api/runs/{run_id}/log` | `{"engine": <d2s/log.txt>, "stages": {stage_id: <200 dòng cuối log.txt>}}` | |
| `GET /api/runs/{run_id}/videos` | Video MP4 trong thư mục run: `{"name", "stage_id", "url", "preview_url", "ready"}`; `url` là video gốc qua `/files/`, `preview_url` là bản xem trước (dòng dưới), `ready` là `false` khi video còn đang được ghi (chưa có hộp `moov`, ví dụ `train.py` đang render cả đoạn sau bước cuối) | |
| `GET /api/runs/{run_id}/videos/preview` | Query: `stage_id`, `name` (như ở danh sách video). Bản trình duyệt phát được: H.264 `yuv420p`, rộng tối đa 1920 điểm ảnh, chỉ mục ở đầu file; tạo lần đầu bằng ffmpeg (`ffmpeg` trên `PATH` hoặc của `imageio-ffmpeg`) và lưu ở `d2s/previews/<stage_id>/<name>`, dùng lại khi video gốc không đổi. Không có ffmpeg thì trả video gốc. Video đang ghi hoặc ffmpeg lỗi: `409`. Lý do: video ghép mọi camera (KITTI-360 bốn camera: 4224x2304) vượt mức H.264 trình duyệt giải mã được, thẻ `<video>` hiện khung đen | |
| `GET /api/runs/{run_id}/video_sets` | Video đánh giá gom theo bộ: `{"id", "stage_id", "set" (test_set, full_set), "step", "postfix", "channels" (kênh → url, theo thứ tự `CHANNELS`), "layout" (nội dung `layout.json` hoặc null)}` (`drive2sim/engine/videos.py`) | |
| `GET /api/runs/{run_id}/artifacts` | Nơi lưu kết quả của từng stage: `{"run_dir", "run_dir_abs", "files" (file của tool trong `d2s/`), "stages": [{"stage_id", "status", "location": "run" \| "cache", "path", "path_abs", "size_bytes", "files": [{"name", "is_dir", "size_bytes", "path"}]}]}`; `path` tương đối với thư mục làm việc. Run chạy trên Kaggle sau `kaggle pull` cũng nằm ở đây (mục 9.2) | |
| `GET /api/runs/{run_id}/curve` | Đường cong train đọc từ `metrics.json` của stage `train.*` (mục 6.1): `{"stage_id", "series": {"psnr", "loss", "gaussians": [[bước, giá trị], ...]}}`; PSNR trên ảnh train, tổng các loss, tổng số Gaussian; tối đa 400 điểm mỗi đường | |
| `POST /api/reveal` | Thân: `path` (tương đối với thư mục làm việc). Mở thư mục đó (với file: thư mục chứa file) trong trình quản lý file của máy chạy Studio; đường dẫn ra ngoài thư mục làm việc, không tồn tại, hoặc máy không có giao diện đồ hoạ thì `409` kèm lý do | |
| `POST /api/pick-folder` | Thân: `start` (tuỳ chọn, thư mục mở sẵn). Mở hộp thoại chọn thư mục của hệ điều hành trên máy chạy Studio (Linux: `zenity` hoặc `kdialog`; macOS: `osascript`; Windows: PowerShell) và chờ người dùng chọn; trả `{"path"}` là đường dẫn tuyệt đối, hoặc `null` nếu người dùng đóng hộp thoại. Máy không có giao diện đồ hoạ hoặc thiếu công cụ thì `409` kèm lý do. Dùng ở form nạp cảnh, vì trình duyệt không cho trang web biết đường dẫn thật của thư mục | |
| `GET /api/sources/uploads`, `POST /api/sources/uploads`, `POST /api/sources/uploads/{name}/chunk`, `DELETE /api/sources/uploads/{name}` | Tải dữ liệu gốc từ máy người dùng lên `uploads/<name>/` của thư mục làm việc, cho Studio chạy trên máy khác (GPU thuê, server lab qua port forward), nơi `/api/pick-folder` mở hộp thoại trên server. `POST /api/sources/uploads` (thân: `name` theo mẫu `[A-Za-z0-9][A-Za-z0-9_.-]{0,79}`, `files: [{"path", "size"}]` với `path` tương đối trong thư mục người dùng chọn) trả kế hoạch `{"name", "path", "path_abs", "files": [{"path", "size", "offset", "done"}], "skipped", "needed_bytes", "free_bytes"}`: file đã đủ thì `done`, file dở (`<file>.part`) tải tiếp từ `offset`; bỏ file rác của hệ điều hành; đường dẫn tuyệt đối, có `..`, sâu quá 12 cấp hoặc không đủ chỗ trống (chừa 5 GiB) thì `400`. `.../chunk?path=&offset=&total=` nhận thân là byte thô (tối đa 64 MiB, Studio gửi 8 MiB), nối vào `<file>.part`, đủ `total` byte thì đổi tên; khối không nối tiếp phần đã có thì `409` (lập kế hoạch lại rồi gửi tiếp). Danh sách: `{"name", "path", "path_abs", "files", "size_bytes", "partial"}`. Tải xong, ô nạp cảnh gọi `POST /api/sources/inspect` với `path_abs` | Không có lệnh riêng: trên máy chạy tool, chép thư mục rồi dùng `source inspect` |
| `GET /api/runs/{run_id}/splats` | `{"preview", "ply", "cameras"}`: gói xem trước (`preview.json` cộng `background_url`, `actor_urls`, `poses_url`) nếu đã chạy `export.preview`, url của `scene.ply`, và url của `cameras.json` (lấy ở `export.preview`, rồi `export.ply_world`, rồi `train.*`; `null` nếu không có) | |
| `GET /api/runs/{run_id}/render-check` | `{"ready", "checks": [{"id", "ok", "value"}]}`: máy chạy Studio có render được run này bằng GPU không. `id` theo thứ tự: `gpu` (nvidia-smi; kèm `memory_mb`), `pytorch` (gói đã cài), `cuda` (PyTorch thấy GPU, hỏi trong tiến trình con), `gsplat`, `checkpoint` (`checkpoint_final.pth` của stage `train.*`), thêm `smpl_model` khi config có người SMPL và bật `load_smpl`; `value` là thứ đã thấy hoặc lý do lỗi. Chỉ kiểm tra, không nạp model | |
| `GET /api/approaches/{approach_id}/environment?stages=` | Môi trường Python của máy chạy Studio có đúng môi trường approach cần không (theo `codebase`; dòng DriveStudio cần `main`): `{"env", "file", "install", "level": "ok" \| "warn" \| "missing", "checks": [{"id", "level", "found", "expected"}]}` với `id` là `gpu`, `pytorch`, `cuda`, `gsplat`, và `smpl_model` khi run có stage làm ra SMPL: `stages` (stage đã giải theo cảnh, split, cấu hình, từ `POST .../check`, cách nhau bởi dấu phẩy) có thì xét đúng các stage đó như lúc tạo run; không có thì xét mọi stage approach có thể chạy và thiếu file SMPL chỉ là `warn` (SMPL `auto` bỏ bước này khi split không có người); approach `nurec` (`codebase: external:nurec`, môi trường `nurec`) có `gpu`, `nurec_source` (mã nguồn ở `D2S_NUREC_DIR`, mặc định `~/.cache/drive2sim/instant-nurec`, đúng commit chốt), `nurec_env` (venv Python 3.11 có torch CUDA, `nvidia-ncore`), `nurec_checkpoint` (file `SMPL_NEUTRAL.pkl` tìm theo biến `D2S_SMPL_MODEL`, `<thư mục làm việc>/smpl_models/`, `<repo>/smpl_models/`, rồi dataset gắn dưới `/kaggle/input`). `warn`: khác phiên bản, vẫn chạy được nhưng kết quả có thể lệch; `missing`: run sẽ lỗi. Chỉ cho run chạy trên máy này; notebook Kaggle tự cài. `env` là `null` khi codebase không cần môi trường GPU | `approach check` (khoá `environment`) |
| `GET /api/environment/install`, `POST /api/environment/install`, `POST /api/environment/install/cancel` | Cài môi trường ngay từ Studio: chạy `envs/install_<env>.sh --python <Python đang chạy Studio> --from <bước>` trong nhóm tiến trình riêng, log ở `<thư mục làm việc>/logs/install-<env>.log`. Thân của `POST`: `env` (`main`, hoặc `nurec`: `envs/install_nurec.sh`, venv riêng Python 3.11 do uv dựng nên không kiểm Python của Studio), `from_step` (1 là từ đầu; cài tiếp từ bước lỗi). Trả `{"state", "env", "python", "step", "steps", "label", "from_step", "started_at", "step_started_at", "finished_at", "log", "log_path"}`; `state`: `idle`, `running`, `succeeded`, `failed`, `cancelled`; `step`/`steps`/`label` đọc từ dòng `=== hh:mm:ss N/M ...` của script; `log` là phần cuối. Không có GPU NVIDIA, Python khác bản `envs/<env>.yml` chốt (main: 3.10) hoặc không thấy script (cài drive2sim không từ mã nguồn) thì `400`. Mỗi lúc một lần cài; không đi qua hàng việc nền (`/api/jobs`); tắt Studio thì lần cài dở bị dừng. Cài xong mở lại Studio để nạp gói mới | `bash envs/install_main.sh` |
| `GET /api/environment/nurec/notebook/{platform}` | Notebook Instant NuRec cho `colab` hoặc `kaggle` (`drive2sim/external/nurec/tools/nurec_notebook.py`, như Fisheye Fusion Lab): tự cài NuRec rồi dựng cảnh từ một clip NCore và checkpoint có trên máy đó; trả file `.ipynb`. Run của approach codebase ngoài không đẩy được lên Kaggle, Colab (`POST /api/runs` với `executor` khác `local` trả `400`) | Không có lệnh riêng |
| `GET /api/libraries`, `POST /api/libraries/{item_id}/install` | Bảng thư viện (`drive2sim/engine/libraries.py`): `{"groups", "items": [{"id", "group", "name", "desc", "links", "size", "minutes", "requires", "dataset", "installable", "status": "ok" \| "partial" \| "missing", "found"}], "install"}` (`install` như `GET /api/environment/install`). Cài: mục của môi trường chính, NuRec chạy script môi trường từ một bước (`renderers` từ bước 4, `smpl-tools` từ bước 8); mục khác chạy một lệnh (pip vào Python đang chạy Studio, tải weight SegFormer, `uv tool install`) qua cùng `InstallFlow`: mỗi lúc một việc, log `<thư mục làm việc>/logs/install-<id>.log`. Mục cài tay (Node.js) hoặc mục lạ `400` | Không có lệnh riêng |
| `POST /api/compare/videos`, `GET /api/compare/videos/{name}` | Ghép video của hai run thành một MP4 (`studio/api/video_compare.py`): thân `{"a": {"run_id", "stage_id", "name"}, "b": {...}, "layout": "vstack" | "hstack"}` (video như `GET /api/runs/{id}/videos`); trả `{"url", "name", "layout"}`. ffmpeg co về cùng rộng 1280 px (trên dưới) hoặc cùng cao 480 px (cạnh nhau), nhãn là `run_id` (không vẽ được chữ thì ghép không nhãn), H.264, dừng ở video ngắn hơn; lưu ở `<thư mục làm việc>/compare/<khoá>_<layout>.mp4`, khoá theo hai file nguồn và thời điểm sửa. Không có video `404`; video đang ghi, thiếu ffmpeg, ffmpeg lỗi `409`; hai bên là một video `400` | Không có lệnh riêng |
| `GET /api/scenes/{scene_id}/frames/{frame_index}/panorama?width=`, `GET /api/scenes/{scene_id}/frames/{frame_index}/lidar?camera=` | Công cụ kiểm tra dữ liệu (`studio/api/frame_tools.py`, như Fisheye Fusion Lab). `panorama`: PNG equirectangular rộng `width` (512 tới 4096, mặc định 2048), cao một nửa, ghép mọi camera của khung bằng phần xoay của `extrinsics/<camera>.txt` và mô hình ống kính của `intrinsics/<camera>.txt`, giữa ảnh là trục tới của xe; header `X-D2S-Coverage`. `lidar`: PNG ảnh của `camera` có điểm `lidar/<khung>.bin` (dạng KITTI, hệ xe) chiếu lên, tô theo độ sâu cùng thang với video độ sâu và ảnh chồng của mục 3.4 (turbo theo log, 4–120 m), điểm rơi lên thân xe (mask ego của dataset) tô xám; header `X-D2S-Points`, `X-D2S-Source: fast`. Phép chiếu này được test trùng phép chiếu của loader (`tests/drive2sim/studio/test_frame_tools.py`; trên KITTI-360 thật mọi pixel loader chiếu đều trùng). Cảnh mà cách nhanh không đọc được (calib theo khung như nuScenes, LiDAR dạng khác): trả JPEG ảnh chồng do loader tạo sẵn (`reports/lidar_overlay/`, mục 3.4) nếu khung đó có, header `X-D2S-Source: loader`; không có thì `400` kèm danh sách khung có sẵn. Camera lạ, thiếu file: `400` | Không có lệnh riêng |
| `GET /api/assistant/config`, `POST /api/assistant/config`, `GET /api/assistant/models`, `POST /api/assistant/chat` | Trợ lý hỏi đáp (`studio/api/assistant.py`). Cài đặt: `{"provider", "base_url", "model", "has_key", "key_hint", "providers"}`; `provider` là `anthropic` (SDK `anthropic`, gói tuỳ chọn; thiếu thì `400` kèm lệnh cài), `openai`, `gemini`, `openrouter` hoặc `custom` (endpoint chuẩn OpenAI, `base_url` dạng `https://host/v1`). `POST .../config` (thân: `provider`, `base_url`, `model`, `api_key`, `clear_key`, bỏ trống thì giữ) lưu ở `~/.config/drive2sim/assistant.json` (quyền 600, `D2S_ASSISTANT_STORE`), mỗi nhà cung cấp một key; key không bao giờ trả về, chỉ `key_hint` 4 ký tự cuối; `D2S_ASSISTANT_KEY` thay key. `models` trả `{"models": [id]}`. `chat` (thân: `messages: [{"role": "user"|"assistant", "content"}]`, `page`) gửi 12 câu gần nhất kèm dữ kiện Studio lúc hỏi (cảnh và split, approach với preset, tuỳ chọn và `camera_support`, 15 run gần nhất, kiểm tra môi trường, GPU; không có key, đã rút gọn) và trả `{"reply", "apply", "model"}`; `apply` là khối `d2s-apply` cuối câu trả lời (`scene_id`, `split_id`, `approach_id`, `preset`, `executor`, `quick`) hoặc `null`. Claude dùng `claude-opus-5-5` mặc định, effort `low`, và `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) với model nhận nó; bị từ chối (`refusal`), thiếu key, model, URL hoặc nhà cung cấp lỗi thì `400` kèm câu báo | Không có lệnh riêng |
| `POST /api/runs/{run_id}/gpu-viewer`, `POST /api/runs/{run_id}/frame-server`, `POST /api/runs/{run_id}/render-frame`, `GET /api/gpu-viewer`, `DELETE /api/gpu-viewer[?kind=viewer\|frames]` | Bật, xem, tắt tiến trình GPU của một run: viewer GPU (`drive2sim/recon/tools/viewer.py --host 127.0.0.1`, mode Render GPU) hoặc frame server (`drive2sim/recon/tools/frame_server.py`, mode Kết hợp: nạp model một lần, `render-frame` nhận JSON `{"c2w": 16 số theo hàng OpenCV hệ world, "K": 9 số, "width", "height", "frame"}` và trả JPEG vẽ bằng đúng hàm của viewer GPU, header `X-Render-Ms`). Mỗi lúc một tiến trình cho cả Studio (GPU nhỏ không giữ nổi hai bản model); bật cái khác thì cái cũ tắt; bật lại frame server cùng run thì giữ nguyên. Trang Xem 3D gọi `DELETE ...?kind=` khi rời mode, rời trang hay đóng tab; `kind` chỉ tắt nếu tiến trình đang chạy đúng loại đó. Trạng thái có thêm `kind`. Trước khi bật, Studio dựng lại cảnh ảo trong `d2s/viewer/scene_view/` từ thư mục cảnh và cache trên máy (mask, SMPL), và ghi `d2s/viewer/config.yaml` là `config.yaml` của stage train với `data.data_root` trỏ vào đó (run tải từ Kaggle có đường dẫn Kaggle). Trạng thái `{"state": "stopped" \| "starting" \| "running" \| "exited", "run_id", "url", "exit_code", "log"}`; máy chưa đủ điều kiện hoặc run chưa có checkpoint thì `409` | |
| `GET /api/kaggle/setup/status` | Kiểm tra trước khi chạy trên Kaggle: `cli` (lệnh `kaggle` đã cài, phiên bản), `credentials` (`source`: nguồn khoá đứng đầu theo thứ tự Kaggle CLI đọc: `env`, `access_token`, `kaggle.json`, `oauth`; `active`: nguồn CLI thật sự dùng, theo `kaggle config view`; `warning`: khác `null` khi hai nguồn khác nhau, tức token sai hoặc đã thu hồi bị CLI bỏ qua và run sẽ chạy bằng tài khoản khác; thư mục cấu hình), `account` (đăng nhập được không, tên), `gpu` (giờ GPU đã dùng, còn lại, tổng, lúc làm mới; từ `kaggle quota`), `code` (commit đã push chưa, có thay đổi chưa commit không), và `ready` | |
| `POST /api/kaggle/setup/token` | Thân: `token`, `username` (chỉ với khoá kiểu cũ), `replace`. Ghi khoá vào đúng chỗ Kaggle CLI đọc, quyền `600`: token `KGAT_...` vào `~/.kaggle/access_token` (CLI không theo `KAGGLE_CONFIG_DIR` với token mới), khoá cũ vào `kaggle.json` của thư mục cấu hình (`KAGGLE_CONFIG_DIR`, `~/.kaggle`, hoặc `~/.config/kaggle` trên Linux khi chưa có `~/.kaggle`). Trả `kind`, `path`, `note` (khác `null` khi biến `KAGGLE_API_TOKEN` hoặc token mới được CLI đọc trước khoá vừa lưu). Không trả lại khoá; khoá sai dạng `400`, đã có khoá mà `replace` là `false` thì `409` | |
| `GET /api/studio` | `{"workdir", "demo"}`: thư mục làm việc, và `demo` là `true` khi đó là thư mục dữ liệu mẫu (có file `DEMO`, do `drive2sim/scripts/make_demo.py` tạo); Studio hiện nhãn "Dữ liệu mẫu" | |
| `GET /api/report` | Bảng so sánh (lọc `scene_id`, `split_id`, `run_id` lặp lại được); mỗi dòng có `train_params` (như `GET /api/runs`), `preset` (tên preset của run, hoặc `null`), metric của `summary.json`, `avg_error` (chỉ số gộp của Mip-NeRF: trung bình nhân của `10^(−PSNR/10)`, `√(1 − SSIM)`, LPIPS; thấp là tốt), cờ `quick`, `num_iters`, `train_resolution` (cỡ ảnh lúc train và đánh giá, `render_resolution` của `summary.json`, ví dụ `800x450`), `source_resolution` (cỡ ảnh gốc của camera trong split theo manifest, ví dụ `1600x900`; nhiều cỡ nối bằng ` / `), `train_images`, `test_images` (số ảnh, tức cặp khung–camera, của phần train và test trong split), `test_every` (N của split `every_nth_frame:N`, `null` với split khác), `train_time_s` (train thực), `recon_time_s` (tổng thời gian các stage từ đầu tới khi xuất xong `scene.ply`, stage dùng lại cache tính thời gian gốc), `render_fps`, `num_gaussians` (cả model), `ply_gaussians` và `ply_size_mb` (của `scene.ply`), `preview_gaussians` và `preview_size_mb` (tổng mọi file của gói xem trước), `occupied_psnr`, `gpu`, `software` (phiên bản PyTorch và gsplat từ `env.json`, ví dụ `PyTorch 2.4.1+cu121, gsplat 1.5.3`), và `stage_times` (`stage_id` → `{"seconds", "reused"}`, stage dùng lại cache tính thời gian gốc; không có trong CSV); `?format=csv` để xuất | `report` |
| `POST /api/sources/inspect` | Thân `{"path"}`; kết quả nhận dạng như `source inspect --json`; `404` nếu không có thư mục | `source inspect` |
| `POST /api/sources/ingest` | Thân `path`, `source_type`, `version`, `sequence`, `scene_id`, `interpolate_n`; chọn chuỗi ngay (`400` kèm các lựa chọn nếu thiếu), rồi ingest nền; trả `202` kèm việc | `scene ingest` |
| `GET /api/jobs`, `GET /api/jobs/{job_id}` | Việc nền (ingest, đẩy và tải Kaggle, Colab): `job_id`, `kind` (`ingest`, `kaggle_push`, `kaggle_pull`, `colab_push`, `colab_pull`, `colab_update`, `colab_session`, `drive_pull`), `label`, `status` (`queued`, `running`, `succeeded`, `failed`), `result` (`scene_id` hoặc `run_id`; `drive_pull` là thư mục đã tải), `error`, `progress` (`null` hoặc `{"done", "total", "detail", "at", "first"}` với việc biết tiến độ, hiện là `drive_pull` tính theo byte; `first` là phần đã có lúc bắt đầu, giao diện tính tốc độ và thời gian còn lại từ `started_at`, `at` và phần làm thêm), `created_at`, `started_at`, `finished_at`; giữ trong bộ nhớ tới khi tắt Studio. Tắt Studio (Ctrl+C) thoát ngay: lệnh `kaggle`, `colab` đang chạy bị dừng (việc thành `failed`, lỗi "dừng vì tắt Studio"; tải dở không ghép gì vào run), run đang chạy trên máy bị huỷ như nút huỷ (chạy tiếp được), ingest đang chạy vẫn chờ xong | |
| `GET /api/kaggle/{run_id}` | `kaggle.json` của run cộng `status` của notebook | `kaggle status` |
| `POST /api/kaggle/{run_id}/push`, `POST /api/kaggle/{run_id}/pull` | Đẩy, tải ở nền; trả `202` kèm việc | `kaggle push`, `kaggle pull` |
| `GET /api/colab/setup/status` | Kiểm tra trước khi chạy trên Colab: `cli` (lệnh `colab` đã cài, phiên bản), `account` (đăng nhập được không, `error`, `hint` khi thiếu keyring), `gpu` (`sessions`: các phiên GPU đang có, `balance` và `rate_per_h` compute unit từ `colab usage`, `choices`: GPU chọn được), `smpl` (`found`, `path` của `SMPL_NEUTRAL.pkl` trên máy), `code` (như Kaggle), và `ready` (cài, đăng nhập, code đã push). Colab không cho biết trước còn GPU không, nên `gpu` không tính vào `ready`. Đăng nhập và đổi tài khoản làm ngay trên Studio (các route tài khoản bên dưới) | |
| `GET /api/colab/accounts`, `POST /api/colab/accounts/{email}/use`, `DELETE /api/colab/accounts/{email}` | Tài khoản Google của lệnh `colab` (mục 9.3): `{"accounts": [email], "active": email hoặc null}`; `use` chuyển sang tài khoản đã đăng nhập, `DELETE` bỏ khỏi danh sách (token đang dùng giữ nguyên). Email sai dạng `400`; chưa lưu tài khoản đó hoặc lệnh lỗi `502 colab_failed` | `colab account list`, `use`, `forget` |
| `GET /api/colab/login`, `POST /api/colab/login`, `POST /api/colab/login/code`, `POST /api/colab/login/cancel` | Thêm tài khoản: `POST /api/colab/login` bắt đầu và trả `{"state", "url", "email", "error"}` với `url` là trang đăng nhập Google (`https://accounts.google.com/...`); `POST .../code` (thân: `code`, mã Google hiện ra) chờ lệnh `colab` lưu token. `state`: `idle`, `waiting_code`, `checking`, `succeeded`, `failed`. Mỗi lúc một lần đăng nhập cho cả Studio; mã sai dạng hoặc không có lần nào chờ mã `400` | `colab account login` |
| `GET /api/colab/cli`, `POST /api/colab/cli/update` | Phiên bản google-colab-cli: `{"version", "latest", "up_to_date"}`; `update` xếp việc nền `colab_update` (`202`) | `colab update` |
| `GET /api/colab/drive/sessions`, `POST /api/colab/drive/session`, `POST /api/colab/drive/session/stop`, `GET /api/colab/drive/mount`, `POST /api/colab/drive/mount`,`POST /api/colab/drive/mount/confirm`, `POST /api/colab/drive/mount/cancel`, `GET /api/colab/drive/browse`, `POST /api/colab/drive/pull` (`include`: chỉ tải file có đường dẫn bắt đầu bằng một tiền tố) | Lấy dữ liệu gốc từ Google Drive. Thân chung: `session` (mặc định `d2s-drive`). `session` xếp việc nền `colab_session` tạo phiên CPU nếu chưa có (`202`); `session/stop` tắt phiên. `mount` chạy `colab drivemount` và trả `{"state", "session", "url", "log"}`; `state`: `idle`, `needs_consent` (mở `url` của Google, đồng ý, rồi `mount/confirm`), `running`, `succeeded`, `failed`; mỗi lúc một lần mount. `browse?session=&path=` (đường dẫn trong `MyDrive`, không có `..`) trả `{"path", "mounted", "exists", "dirs", "files": [[tên, byte]], "truncated"}` (tối đa 500 mục). `pull` (thân: `path`, `name` như tên lần tải ở `/api/sources/uploads`) xếp việc `drive_pull` (`202`, có `progress` theo byte) tải thư mục hoặc file về `uploads/<name>/`: file đã đủ bỏ qua, file dở (`<file>.part`) tải tiếp; không đủ chỗ (chừa 5 GiB), Drive chưa mount hoặc hơn 50000 file thì việc lỗi; `result` là đường dẫn tuyệt đối để `POST /api/sources/inspect` | Không có lệnh riêng |
| `GET /api/colab/drive/inspect?session=&path=` | Nhận dạng dataset ở một thư mục trong `MyDrive` (`colab/drive.py: inspect`): quét tên và kích thước (sâu 7 cấp, tối đa 600 thư mục), dựng lại cây với file rỗng trong thư mục tạm rồi chạy `inspect_source` như `d2s source inspect`. Trả `{path, mounted, exists, truncated, report, sequence_bytes, pull_sets}`; `report` có đường dẫn tương đối trong `MyDrive`, `pull_sets` là tiền tố file của từng chuỗi KITTI-360 cho `include` của `/drive/pull` | Không có lệnh riêng |
| `GET /api/colab/{run_id}` | `colab.json` của run (mục 9.3) cộng `status` (`running`, `succeeded`, `failed`, `died`, `lost`, `stopped`, hoặc `null` khi không hỏi được), `exit_code`, `log_tail` (40 dòng log cuối), `status_error` | `colab status` |
| `POST /api/colab/{run_id}/push`, `POST /api/colab/{run_id}/pull` | Đẩy (thân tuỳ chọn `gpu`, `session`), tải ở nền; trả `202` kèm việc | `colab push`, `colab pull` |
| `POST /api/colab/{run_id}/stop` | Tắt phiên Colab của run; trả `colab.json` | `colab stop` |
| `POST /api/scenes/{scene_id}/validate` | Kiểm tra lại, ghi và trả báo cáo | `scene validate` |
| `DELETE /api/scenes/{scene_id}`, `DELETE /api/scenes/{scene_id}/splits/{split_id}` | `{"removed": [...]}`; `409` khi run đang chạy hoặc split còn run dùng | `scene delete`, `split delete` |
| `POST /api/runs/{run_id}/cancel` | Ghi yêu cầu huỷ (mục 6.1); trả `run.json` hiện tại | `run cancel` |
| `DELETE /api/runs/{run_id}` | `{"removed": [...]}`; `409` khi run đang chạy | `run delete` |
| `POST /api/runs/{run_id}/export` | Thân `{"kind": "ply" \| "preview"}`; đặt lại stage export rồi xếp run vào hàng đợi | `export ply`, `export preview` |
| `GET /api/cache`, `POST /api/cache/gc` | Liệt kê (kèm `size_bytes`, `used_by`); dọn, thân `scene_id`, `dry_run` | `cache list`, `cache gc` |
| `GET /files/<đường dẫn>` | File trong `D2S_WORKDIR`; đường dẫn ra ngoài thư mục làm việc trả `404` | |

Run chạy nền **lần lượt từng run** trong một luồng (`drive2sim/studio/api/jobs.py`): một người, một máy, hai run train cùng lúc sẽ tranh GPU. Việc nền dài (ingest, Kaggle, Colab) chạy ở luồng riêng, nên không phải chờ run train xong.

### 11.2 Thân và kết quả

- JSON, trường `snake_case`, **cùng tên và cùng kiểu với file trong CONTRACTS**: `GET /api/runs/{run_id}` trả đúng nội dung `run.json` cộng thêm `summary` (nội dung `summary.json`, nếu có) và `run_dir` (tương đối với `D2S_WORKDIR`). Frontend không đổi sang `camelCase`; kiểu TypeScript sinh từ JSON Schema của `drive2sim/core/schema.py` (`d2s schema export`), nên không có bảng ánh xạ tên nào phải giữ tay.
- Danh sách trả về `{"items": [...], "total": <n>}`; không phân trang trong bản đầu (một người dùng, vài trăm run).
- Tạo run và hành động trên run trả `202` kèm `run.json`; việc thật chạy nền, theo dõi qua SSE. Tạo split trả `201` vì xong ngay.
- Frontend bỏ qua trường lạ trong kết quả (thêm trường ở server không làm hỏng giao diện cũ); server từ chối trường lạ trong thân yêu cầu (bắt lỗi gõ sai), cùng quy tắc với file (phần đầu tài liệu).

### 11.3 Lỗi

Mọi lỗi trả về cùng một hình dạng, HTTP status theo loại:

```json
{"error": {"code": "blocked", "message": "Approach omnire bị khoá: cảnh không có LiDAR mà mục cấu hình init là init.lidar", "details": {"option": "init", "choice": "init.lidar"}}}
```

| HTTP | `error.code` | Khi nào | Mã thoát CLI tương ứng |
|---|---|---|---|
| 400 | `invalid_request` | Thân hoặc query sai cú pháp, sai kiểu, trường lạ | 2 |
| 403 | `forbidden` | Request không đến từ chính máy này: `Host` không phải `127.0.0.1`/`localhost`, hoặc lệnh ghi có `Origin` lạ (chống DNS rebinding và request chéo trang) | – |
| 404 | `not_found` | `scene_id`, `run_id`... không tồn tại | 1 |
| 409 | `conflict` | Đối tượng đã tồn tại, hoặc run đang chạy mà bị yêu cầu xoá | 1 |
| 422 | `blocked` | Không thoả điều kiện dữ liệu hoặc `constraints` (mục 4.2) | 3 |
| 500 | `internal` | Lỗi không lường trước; `message` không lộ traceback, traceback vào log | 1 |
| 502 | `kaggle_failed` | Lệnh `kaggle` lỗi: chưa cài, chưa có khoá API, commit chưa push, notebook trước còn chạy, vẫn bị giới hạn 429 sau 3 lần chờ (60, 180, 600 s) | 1 |
| 502 | `colab_failed` | Lệnh `colab` lỗi: chưa cài, chưa đăng nhập, commit chưa push, thiếu file SMPL, VM đang chạy job khác, phiên đã mất | 1 |
| 503 | `worker_unavailable` | Worker render chưa sẵn sàng hoặc đã chết | 1 |

Mã lỗi của worker (mục 9.1) và của run (mục 6.2) dùng cùng bảng, thêm `env_missing`, `worker_crashed`, `cancelled`, và `stage_failed` (một stage ném lỗi khi chạy; ghi trong `run.json`, mã thoát CLI `1`). `code` là `snake_case`, thêm mã mới thì ghi vào bảng này. `message` tiếng Việt, viết cho người dùng, không phải cho lập trình viên.

### 11.4 Server-Sent Events

- Endpoint: `GET /api/runs/{run_id}/events` (một run) và `GET /api/events` (mọi run đang chạy, cho màn hình tổng quan).
- Mỗi sự kiện SSE có `event:` là tên và `data:` là một JSON. Tên sự kiện và trường trùng với sự kiện của worker (mục 9.1) cộng thêm `run_id` và `stage_id`:

| `event` | `data` | Khi nào |
|---|---|---|
| `status` | `run_id`, `stage_id` (hoặc `null`), `status` | Run hoặc stage đổi trạng thái (giá trị như `run.json`) |
| `progress` | `run_id`, `stage_id`, `fraction`, `message` | Tiến độ trong stage |
| `metric` | `run_id`, `stage_id`, `name`, `value`, `step` | Metric trong lúc train |
| `log` | `run_id`, `stage_id`, `level`, `message` | Dòng log |
| `done` | `run_id`, `status` | Run kết thúc; sau sự kiện này server đóng kết nối |

- Khi client kết nối, server gửi ngay một `status` cho trạng thái hiện tại rồi mới phát tiếp, để giao diện mở muộn vẫn đúng.
- Mỗi sự kiện có `id:` tăng dần trong một run; client gửi `Last-Event-ID` để nhận lại những gì bỏ lỡ (server giữ 1000 sự kiện gần nhất trong bộ nhớ, ngoài ra có `d2s/log.txt`).
- Nhịp `: keepalive` mỗi 15 giây để kết nối không bị proxy hoặc port forward cắt.
- Studio ước lượng thời gian còn lại của stage đang chạy từ chuỗi `progress` của stage đó (lần báo đầu và lần mới nhất của `fraction`, `frontend/src/eta.ts`), không cần trường riêng; vì vậy `fraction` nên tăng đều theo khối lượng việc (ví dụ bước train đã xong / tổng bước).
