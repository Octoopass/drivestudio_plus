# drive2sim: hợp đồng dữ liệu và giao diện

Tài liệu này quy định định dạng của dữ liệu đi qua tool và interface mà approach, bước chuẩn bị, runner phải theo. Người thêm approach mới chỉ cần đọc tài liệu này cùng `ARCHITECTURE.md` mục 4.

Đây là thiết kế đích (`ARCHITECTURE.md` mục 7). Mọi file do tool ghi đều có trường `schema_version`; thay đổi làm vỡ tương thích thì tăng số đầu của phiên bản.

Trạng thái: bản nháp, ngày 12/11/2026.

## 1. Quy ước chung

Dữ liệu chuẩn là NCore V4, nên tool **theo quy ước của NCore** (`docs/data/conventions.rst` trong repo NCore). Codebase nào dùng quy ước khác thì tự đổi ở biên của nó.

| Mục | Quy ước |
|---|---|
| Đơn vị | Mét, giây, radian. Độ chỉ dùng trong giao diện. |
| Thời gian | Số nguyên micro giây. NCore lưu `uint64`; file của tool lưu số nguyên, tên trường kết thúc bằng `_us`. |
| Hệ `rig` | Thân xe: x tới trước, y sang trái, z lên trên; gốc ở giữa trục sau, trên mặt đất. |
| Hệ `world` | Hệ cục bộ của cảnh; pose đầu tiên của xe làm gốc. `T_world_world_global` nối với hệ toàn cầu (ECEF) nếu nguồn có. |
| Hệ camera | Theo OpenCV: x sang phải, y xuống dưới, z nhìn tới trước. |
| Toạ độ ảnh | u sang phải, v xuống dưới, gốc ở góc trên trái; pixel đầu tiên phủ khoảng `[0, 1]` trên cả hai trục, nên tâm của nó là `(0.5, 0.5)`. |
| Phép biến đổi | Ma trận 4×4 SE(3). **`T_a_b` đưa điểm từ hệ `a` sang hệ `b`**: `p_b = T_a_b · p_a`; ghép chuỗi: `T_a_c = T_b_c · T_a_b`. Ví dụ: extrinsic camera là `T_camera_rig`; quỹ đạo xe là `T_rig_world`. |
| Lưu phép biến đổi | Trong NCore: theo NCore. Trong YAML/JSON của tool: `{translation: [x, y, z], rotation_wxyz: [w, x, y, z]}`. |
| Chỉ số khung | Số nguyên từ 0, theo dòng thời gian tham chiếu của cảnh (mục 2.2). |

Đổi sang quy ước của DriveStudio: DriveStudio lưu pose camera dạng `camtoworlds` (ma trận đưa điểm từ camera sang world), tức `T_camera_world` theo cách viết của NCore. Mọi chuyển đổi đi qua hàm trong `drive2sim/core`.

### 1.1 Lý do của các quy ước

| Quy ước | Lý do | Đánh đổi |
|---|---|---|
| Theo hẳn quy ước NCore | Dữ liệu chuẩn là NCore; giữ một quy ước thì không phải đổi qua lại trong tool | Ngược chiều với nhiều tài liệu khác về `T_a_b`; phải đọc kỹ tên biến |
| Thời gian là số nguyên micro giây | So sánh bằng nhau chính xác; NCore và proto `SensorsimService` (`frame_start_us`, `frame_end_us`) cùng dùng micro giây | Nguồn ghi nano giây (ví dụ Argoverse 2) bị làm tròn |
| Gốc `world` ở pose đầu của xe | Toạ độ nhỏ, giữ độ chính xác khi train bằng số thực 32 bit | Phải dùng `T_world_world_global` nếu cần vị trí toàn cầu |
| Hệ `rig` x trước, y trái, z lên | Trùng NCore, Waymo, REP 103 của ROS | CARLA dùng hệ tay trái (y sang phải); converter CARLA và server phải đổi dấu trục y |
| Quaternion thứ tự w, x, y, z trong YAML/JSON | Trùng viser và `rot_0..3` trong PLY của 3DGS | scipy mặc định x, y, z, w; dùng hàm trong `core` để tránh nhầm |

## 2. Dữ liệu chuẩn

### 2.1 Cấu trúc thư mục

```
data/<scene_id>/
├── ncore/                       # kho NCore V4 do converter ghi ra (các component group, .zarr hoặc .zarr.itar)
├── manifest.yaml
├── splits/<split_id>.json
└── reports/validation.json
```

Thư mục cảnh chỉ chứa những gì có sẵn từ nguồn. Mask, depth, SMPL do tool sinh ra là artifact (mục 5), không ghi vào đây.

### 2.2 `manifest.yaml`

```yaml
schema_version: d2s-manifest/0.1
scene_id: nuscenes_mini_scene-0061
source:
  dataset: nuscenes               # waymo | nuscenes | nuplan | argoverse2 | pandaset | kitti | physicalai | carla | colmap | custom
  sequence_id: scene-0061
  converter: ncore.nuscenes       # converter đã dùng
  converter_version: "19.9.0"
  data_license: "CC BY-NC-SA 4.0"
ncore:
  version: v4
  sequence_id: scene-0061
  timestamp_interval_us: {start: 1532402927647951, stop: 1532402947147951}
track: pinhole                    # pinhole | fisheye
cameras:
  - id: CAM_FRONT
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
- Các giá trị trong ví dụ chỉ minh hoạ định dạng.

## 3. Split và báo cáo kiểm tra

### 3.1 `splits/<split_id>.json`

```json
{
  "schema_version": "d2s-split/0.1",
  "split_id": "every8_allcams",
  "scene_id": "nuscenes_mini_scene-0061",
  "strategy": "every_nth_frame:8",
  "frame_range": [0, 38],
  "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
  "train": [[0, "CAM_FRONT"], [0, "CAM_FRONT_LEFT"], [1, "CAM_FRONT"]],
  "test":  [[8, "CAM_FRONT"], [8, "CAM_FRONT_LEFT"], [8, "CAM_FRONT_RIGHT"]]
}
```

Mỗi phần tử là cặp `[frame_index, camera_id]`. Một cặp không được nằm ở cả `train` và `test`.

`strategy` lấy một trong:
- `every_nth_frame:N`: các khung N, 2N, 3N… (tính từ khung đầu của `frame_range`) làm test cho mọi camera trong `cameras`; các khung còn lại, kể cả khung đầu, làm train. Trùng với `test_image_stride` của DriveStudio (`split_train_test` trong `datasets/driving_dataset.py`).
- `custom`: danh sách `train` và `test` bất kỳ.

Approach khai báo các `strategy` mà codebase của nó hỗ trợ (mục 4). Approach dòng DriveStudio hiện chỉ hỗ trợ `every_nth_frame`.

### 3.2 `reports/validation.json`

```json
{
  "schema_version": "d2s-report/0.1",
  "scene_id": "nuscenes_mini_scene-0061",
  "created_utc": "2026-11-12T08:00:00Z",
  "summary": {"errors": 0, "warnings": 1},
  "checks": [
    {"id": "timestamps.camera_sync", "severity": "warning",
     "message": "Lệch thời gian giữa camera lớn hơn 5 ms", "camera_id": "CAM_BACK", "frame_index": 12,
     "details": {"max_offset_us": 6200}}
  ]
}
```

`severity` lấy một trong: `error` (không chạy được), `warning`, `info`.

## 4. Approach

### 4.1 File khai báo

`drive2sim/approaches/<approach_id>.yaml`:

```yaml
schema_version: d2s-approach/0.1
id: omnire
name: "OmniRe (3DGS, khởi tạo từ LiDAR, không tinh chỉnh pose camera)"
version: 0.1.0
codebase: drivestudio             # drivestudio | external:<tên>
track: pinhole                    # pinhole | fisheye
level: L1                         # L1 | L2 | L3 | L4
requires:
  camera_models: [opencv_pinhole]
  lidar: required                 # required | optional | none
  cuboids: required
  split_strategies: [every_nth_frame]
prep:                             # các bước chuẩn bị riêng, chạy theo thứ tự
  - {step: select, params: {downscale: 2}}
  - {step: masks.segformer_hf, params: {dynamic: true}}
  - {step: smpl}
  - {step: to_drivestudio}
train:
  config: configs/omnire_nocampose.yaml
  overrides: {trainer.render.render_mode: default}
presets:
  default: {}
  fast: {train: {overrides: {trainer.optim.num_iters: 10000}}}
```

- `train.overrides` là tham số ghi đè kiểu OmegaConf, giống cách truyền trên dòng lệnh của `tools/train.py`.
- Tham số cuối cùng được gộp theo thứ tự: config gốc → `overrides` → preset đã chọn → tham số người dùng chỉnh trên Studio.
- Tên khoá trong `overrides` và `presets` ở ví dụ chỉ minh hoạ; tên thật lấy theo config của DriveStudio.

### 4.2 Điều kiện chạy

Runner so `requires` với manifest và split. Approach bị khoá, kèm lý do, khi:
- `lidar: required` mà `available.lidar` là `false`;
- `cuboids: required` mà `available.cuboids` là `none`;
- mô hình camera của các camera trong split không nằm trong `camera_models`, và `prep` không có bước `undistort`;
- `strategy` của split không nằm trong `split_strategies`;
- `track` của approach khác `track` của cảnh.

### 4.3 Khai báo dùng LiDAR

Mỗi bước chuẩn bị và mỗi codebase khai báo có dùng LiDAR hay không. Runner ghi danh sách bước dùng LiDAR vào `run.json`; Studio đọc để hiện "LiDAR dùng ở khâu nào".

## 5. Artifact và cache

### 5.1 Cấu trúc

```
cache/<artifact_id>/
├── artifact.json
└── ...                          # file của artifact
```

```json
{
  "schema_version": "d2s-artifact/0.1",
  "artifact_id": "sha256:4f1c...",
  "type": "masks",
  "producer": {"step": "masks.segformer_hf", "step_version": "0.1.0", "params_hash": "sha256:9a0e...",
               "code_commit": "be54147"},
  "inputs": ["scene:nuscenes_mini_scene-0061@sha256:77ab..."],
  "created_utc": "2026-11-12T08:10:00Z",
  "files": ["ncore4-masks_segformer.zarr.itar"],
  "meta": {"model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"}
}
```

### 5.2 Loại artifact

| `type` | Nội dung |
|---|---|
| `masks` | Mask bầu trời, mask vật động (người, xe), lưu thành một component group NCore loại nhãn theo ảnh (`CameraLabelsComponent`) |
| `smpl` | Tư thế người theo khung, định dạng của `datasets/tools/humanpose_process.py` |
| `depth_prior` | Depth metric theo camera và khung, lưu như `masks`; kèm tên model |
| `points_init` | Point cloud khởi tạo: `points` `float32[M,3]` trong hệ `world`, `colors` `uint8[M,3]`, `track_id` (tuỳ chọn) |
| `drivestudio_scene` | Thư mục cảnh theo định dạng tiền xử lý của DriveStudio, do bước `to_drivestudio` ghi |
| `model` | Checkpoint ở định dạng gốc của codebase |
| `export` | PLY 3DGS trong hệ `world` |
| `preview` | Gói xem trước cho Spark (mục 7) |

### 5.3 Khoá cache

`artifact_id = sha256(JSON chuẩn hoá của {step, step_version, params, danh sách id của đầu vào})`. Mã băm của cảnh tính từ nội dung thư mục `ncore/` và `manifest.yaml`. Artifact đang ghi nằm trong thư mục có hậu tố `.partial`, chỉ đổi sang tên chính thức khi ghi xong.

## 6. Run

### 6.1 Cấu trúc

Thư mục run chính là thư mục log của DriveStudio: `tools/train.py` ghi vào `os.path.join(output_root, project, run_name)`. Runner đặt `output_root = runs/<scene_id>`, `project = <approach_id>`, `run_name = <run_id>`.

```
runs/<scene_id>/<approach_id>/<run_id>/
├── config.yaml                  # DriveStudio ghi: config đầy đủ sau khi gộp
├── metrics.json                 # DriveStudio ghi: log metric trong lúc train
├── metrics/images_test_*.json   # DriveStudio ghi: kết quả của tools/eval.py
├── videos/ images/ configs_bk/ backup/
├── checkpoint_*.pth
└── d2s/                         # tool ghi
    ├── run.json
    ├── summary.json
    └── env.json
```

File của tool nằm trong `d2s/` để không trùng tên với file của DriveStudio (DriveStudio đã dùng `metrics.json`).

### 6.2 `d2s/run.json`

```json
{
  "schema_version": "d2s-run/0.1",
  "run_id": "20261112-081000-omnire-a1b2",
  "scene_id": "nuscenes_mini_scene-0061",
  "split_id": "every8_allcams",
  "approach": {"id": "omnire", "version": "0.1.0", "preset": "default"},
  "level": "L1",
  "executor": "kaggle",
  "status": "succeeded",
  "lidar_usage": ["train"],
  "steps": [
    {"id": "masks.segformer_hf", "status": "succeeded", "cache_hit": true, "artifact": "sha256:4f1c...",
     "duration_s": 0, "peak_vram_mb": 0},
    {"id": "train", "status": "succeeded", "cache_hit": false, "artifact": null,
     "duration_s": 4380, "peak_vram_mb": 7400}
  ]
}
```

`status` lấy một trong: `pending`, `running`, `succeeded`, `failed`, `cancelled`.

### 6.3 `d2s/summary.json`

Tóm tắt metric theo protocol của SPEC mục 9, lấy từ `metrics/images_test_*.json` của DriveStudio.

```json
{
  "schema_version": "d2s-summary/0.1",
  "protocol": {"version": "0.1", "image_dtype": "uint8", "resolution": "native",
               "lpips": {"net": "alex", "package": "lpips", "package_version": "0.1.4"}},
  "split_id": "every8_allcams",
  "overall": {"psnr": 27.1, "ssim": 0.83, "lpips": 0.21,
              "human_psnr": 23.4, "vehicle_psnr": 24.0, "occupied_psnr": 26.2},
  "per_camera": {"CAM_FRONT": {"psnr": 28.0, "ssim": 0.85, "lpips": 0.19}},
  "train_time_s": 4380,
  "peak_vram_mb": 7400,
  "num_gaussians": 360000
}
```

Các con số trong ví dụ chỉ minh hoạ định dạng. Metric vùng người, vùng xe chỉ có khi cảnh có box.

### 6.4 `d2s/env.json`

Commit của repo, phiên bản Python, PyTorch, CUDA, gsplat, tên GPU, seed, tên executor.

## 7. Gói xem trước

Gói xem trước cho phép xem cảnh trong trình duyệt mà không cần GPU CUDA (`ARCHITECTURE.md` mục 2.13). Không bắt buộc.

```
cache/<artifact_id>/
├── artifact.json
├── preview.json
├── background.ply               # Gaussian của nền tĩnh, hệ world
├── actors/<track_id>.ply        # Gaussian của từng xe (actor rigid), hệ của box
└── actor_poses.npz
```

- File `.ply` theo định dạng PLY của 3DGS gốc: `x, y, z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`.
- `actor_poses.npz`: `timestamps_us` `int64[F]`; `track_ids` `str[K]`; `T_object_world` `float64[F,K,4,4]` (đưa điểm từ hệ của box sang `world`); `visible` `bool[F,K]`.

```json
{
  "schema_version": "d2s-preview/0.1",
  "source_model": "runs/nuscenes_mini_scene-0061/omnire/20261112-081000-omnire-a1b2",
  "sh_degree": 3,
  "rigid_actors": ["veh_0007", "veh_0012"],
  "box_only_actors": ["ped_0003"],
  "omitted": ["sky", "per_camera_color", "fisheye"]
}
```

- `box_only_actors`: actor không rigid, chỉ hiện bằng box lấy từ cuboid của dữ liệu chuẩn.
- `omitted`: những phần của model không có trong gói.

## 8. Interface theo mức năng lực

```python
@dataclass
class CameraSpec:                  # cùng trường với một camera trong manifest + intrinsics từ NCore
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

## 9. Worker và Kaggle

### 9.1 Worker

Dùng cho codebase ngoài (môi trường riêng) và cho tiến trình render của viewer và server:

```
d2s-worker --job job.json --result result.json
```

- `job.json`: `schema_version`, loại việc, tham số đã gộp, đường dẫn đầu vào, `output_dir`.
- `result.json`: trạng thái, đường dẫn đầu ra, `peak_vram_mb`, thông báo lỗi nếu có.
- Trong lúc chạy, worker in ra stdout mỗi dòng một JSON: `{"event": "progress", "fraction": 0.42, "message": "..."}` hoặc `{"event": "metric", "name": "loss", "value": 0.031, "step": 4200}`.
- Mã thoát khác 0 nghĩa là thất bại.

### 9.2 Gói job cho Kaggle

```
bundle/
├── job.json                     # run_id, approach, preset, tham số đã gộp, commit của repo
├── approach.yaml
├── cache_refs.json              # artifact đã có, nhận diện bằng mã băm
├── resume.json                  # checkpoint để chạy tiếp, nếu có
└── scene_ref.json               # slug của Kaggle dataset chứa cảnh, cùng mã băm của cảnh
```

Notebook chạy `d2s run --bundle bundle/`, ghi kết quả ra `/kaggle/working/runs/` và `/kaggle/working/cache/` theo đúng cấu trúc mục 5 và 6. Checkpoint giữa chừng phải đủ để chạy tiếp (model và trạng thái optimizer). Lệnh `d2s pull` tải hai thư mục này về và ghép vào thư mục làm việc trên máy.
