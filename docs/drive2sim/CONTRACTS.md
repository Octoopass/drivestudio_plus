# drive2sim: hợp đồng dữ liệu và giao diện

Tài liệu này quy định định dạng của dữ liệu đi qua tool, interface mà approach và stage phải theo, cùng quy tắc đặt tên và lệnh CLI (mục 10). Người thêm approach mới chỉ cần đọc tài liệu này cùng `ARCHITECTURE.md` mục 4.

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
scene_id: nuscenes_scene-0061
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
    {"id": "timestamps.camera_sync", "severity": "warning",
     "message": "Lệch thời gian giữa camera lớn hơn 5 ms", "camera_id": "CAM_BACK", "frame_index": 12,
     "details": {"max_offset_us": 6200}}
  ]
}
```

`severity` lấy một trong: `error` (không chạy được), `warning`, `info`.

## 4. Approach

### 4.1 File khai báo

`drive2sim/approaches/<approach_id>.yaml`. Approach là danh sách stage chạy lần lượt; các khâu thay thế được cho nhau là *ô chọn*, quyết định trước khi chạy (`ARCHITECTURE.md` mục 2.5).

```yaml
schema_version: d2s-approach/0.1
approach_id: omnire
name: "OmniRe"
version: 0.1.0
codebase: drivestudio              # drivestudio | external:<tên>
track: pinhole                     # pinhole | fisheye
level: L1                          # L1 | L2 | L3 | L4
requires:
  camera_models: [opencv_pinhole]
  cuboids: required                # required | optional | none
  split_strategies: [every_nth_frame]

options:                           # ô chọn
  init:
    choices: [init.lidar, init.depth]
    default: auto                  # auto: có LiDAR → init.lidar, không có → init.depth
  renderer:
    choices: [3dgs, 3dgut, 3dgeer]
    default: 3dgs
  campose:
    choices: [false, true]
    default: false
  fastgs:
    choices: [false, true]
    default: false

constraints:                       # tổ hợp không hợp lệ
  - when: {renderer: 3dgeer}
    forbid: {campose: true}
    reason: "3DGEER không tối ưu được pose camera"

presets:                           # tổ hợp đặt sẵn; chỉ ghi các ô khác giá trị mặc định
  paper: {init: init.lidar}
  no_lidar: {init: init.depth}
  fisheye: {renderer: 3dgeer}

stages:                            # chạy lần lượt; ${...} được thay trước khi chạy
  - {stage: prep.select, params: {downscale: 2}}
  - {stage: prep.masks_segformer, params: {dynamic: true}}
  - {stage: prep.smpl}
  - {stage: prep.to_drivestudio}
  - {stage: "${options.init}"}
  - stage: train.drivestudio
    params:
      method: omnire
      renderer: "${options.renderer}"
      campose: "${options.campose}"
      fastgs: "${options.fastgs}"
  - {stage: eval.standard}
  - {stage: render.videos}
  - {stage: export.ply_world}
```

- Một stage dùng được artifact của bất kỳ stage nào đứng trước. Engine nối theo kiểu artifact: stage cần `points_init` nhận artifact `points_init` gần nhất phía trước.
- `${options.<tên ô>}` được thay bằng lựa chọn cụ thể trước khi chạy. Sau khi thay, danh sách stage là cố định; không có điều kiện, nhánh song song hay vòng lặp lúc đang chạy.
- Tham số cuối cùng của một stage được gộp theo thứ tự: mặc định của stage → `params` trong file → preset → lựa chọn của người dùng (Studio hoặc `--set` của CLI).
- `train.drivestudio` đổi `method`, `renderer`, `campose`, `fastgs` sang config và tham số của DriveStudio (ví dụ `renderer: 3dgeer` → `trainer.render.render_mode: geer`, kèm `densify_grad_thresh: 0.01` và `refine_interval: 300` như các config 3DGEER của teammate).

### 4.2 Điều kiện chạy

Engine kiểm tra trước khi chạy. Approach hoặc một lựa chọn bị khoá, kèm lý do, khi:
- lựa chọn cần LiDAR (ví dụ `init.lidar`) mà `available.lidar` là `false`;
- `cuboids: required` mà `available.cuboids` là `none`;
- mô hình camera trong split không nằm trong `camera_models`, và danh sách stage không có `prep.undistort`;
- `strategy` của split không nằm trong `split_strategies`;
- `track` của approach khác `track` của cảnh;
- tổ hợp lựa chọn vi phạm một mục trong `constraints`.

Studio và CLI dùng chung bước kiểm tra này.

### 4.3 Khai báo dùng LiDAR

Mỗi stage khai báo `uses_lidar`. Engine ghi danh sách stage dùng LiDAR vào `run.json`; Studio đọc để hiện "LiDAR dùng ở khâu nào".

## 5. Stage, artifact và cache

### 5.1 Interface của stage

```python
class Stage(Protocol):
    stage_id: str                      # "<nhóm>.<tên>", ví dụ "init.depth" (mục 10.2)
    version: str                       # tăng khi đổi hành vi
    cacheable: bool                    # True: kết quả vào cache/ dùng chung; False: vào thư mục run
    uses_lidar: bool
    inputs: dict[str, str]             # tên đầu vào → loại artifact, hoặc "scene" / "split"
    outputs: dict[str, str]            # tên đầu ra → loại artifact
    Params: type[pydantic.BaseModel]

    def run(self, ctx: StageContext, inputs: dict[str, Any], params: BaseModel) -> dict[str, Path]:
        """Ghi đầu ra vào ctx.output_dir và trả về đường dẫn của từng đầu ra."""

class StageContext(Protocol):
    output_dir: Path
    scene: Scene                       # đọc dữ liệu chuẩn (NCore + manifest)
    split: Split
    def progress(self, fraction: float, message: str = "") -> None: ...
    def log_metric(self, name: str, value: float, step: int) -> None: ...
```

Stage đăng ký bằng `@register_stage("<stage_id>")`. Engine lo cache, file `done`, thư mục `.partial` và `artifact.json`; stage chỉ ghi file vào `output_dir`.

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
  "schema_version": "d2s-artifact/0.1",
  "artifact_type": "masks",
  "cache_key": "nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e",
  "producer": {"stage_id": "prep.masks_segformer", "version": "0.1.0", "code_commit": "286ad9b"},
  "params": {"dynamic": true},
  "inputs": ["scene:nuscenes_scene-0061"],
  "created_utc": "2026-11-12T08:10:00Z",
  "files": ["ncore4-masks_segformer.zarr.itar"],
  "meta": {"model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"}
}
```

### 5.4 Loại artifact

| `artifact_type` | Nội dung |
|---|---|
| `scene_subset` | Kết quả của `prep.select`: danh sách camera, đoạn khung, độ phân giải đã chọn |
| `masks` | Mask bầu trời, mask vật động (người, xe), lưu thành một component group NCore loại nhãn theo ảnh (`CameraLabelsComponent`) |
| `smpl` | Tư thế người theo khung, định dạng của `drive2sim/recon/preprocess/smpl/run.py` |
| `depth_prior` | Depth metric theo camera và khung, lưu như `masks`; kèm tên model |
| `points_init` | `points_init.npz`: `points` `float32[M,3]` trong hệ `world`, `colors` `uint8[M,3]`, `track_ids` (tuỳ chọn) |
| `drivestudio_scene` | Thư mục cảnh theo định dạng tiền xử lý của DriveStudio, do `prep.to_drivestudio` ghi |
| `model` | Checkpoint ở định dạng gốc của codebase |
| `eval_report` | Kết quả đánh giá theo protocol, nguồn của `d2s/summary.json` |
| `videos` | Video MP4 H.264 |
| `export_ply` | PLY 3DGS trong hệ `world` |
| `preview` | Gói xem trước cho Spark (mục 7) |

## 6. Run

### 6.1 Cấu trúc

Thư mục run chính là thư mục log của DriveStudio: `drive2sim/recon/tools/train.py` ghi vào `os.path.join(output_root, project, run_name)`. Engine đặt `output_root = runs/<scene_id>`, `project = <approach_id>`, `run_name = <run_id>`.

```
runs/<scene_id>/<approach_id>/<run_id>/
├── config.yaml                  # DriveStudio ghi: config đầy đủ sau khi gộp
├── metrics.json                 # DriveStudio ghi: log metric trong lúc train
├── metrics/images_test_*.json   # DriveStudio ghi: kết quả đánh giá
├── videos/ images/ configs_bk/ backup/
├── checkpoint_<bước>.pth         # chỉ model, sau mỗi lần lưu
├── checkpoint_resume.pth        # model + optimizer + grad scaler, ghi đè mỗi lần lưu; xoá khi train xong
├── checkpoint_final.pth         # chỉ model
└── d2s/                         # tool ghi
    ├── run.json
    ├── summary.json
    ├── env.json
    └── stages/<stage_id>/       # kết quả stage không cache, có file done
```

File của tool nằm trong `d2s/` để không trùng tên với file của DriveStudio (DriveStudio đã dùng `metrics.json`).

### 6.2 `d2s/run.json`

```json
{
  "schema_version": "d2s-run/0.1",
  "run_id": "20261112-081000-omnire-a1b2",
  "scene_id": "nuscenes_scene-0061",
  "split_id": "every8_allcams",
  "approach": {"approach_id": "omnire", "version": "0.1.0", "preset": "no_lidar",
               "options": {"init": "init.depth", "renderer": "3dgs", "campose": false, "fastgs": false}},
  "level": "L1",
  "executor": "kaggle",
  "status": "succeeded",
  "lidar_usage": [],
  "stages": [
    {"stage_id": "prep.masks_segformer", "status": "succeeded", "cache_hit": true,
     "cache_key": "nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e", "duration_s": 0, "peak_vram_mb": 0},
    {"stage_id": "train.drivestudio", "status": "succeeded", "cache_hit": false, "cache_key": null,
     "duration_s": 4380, "peak_vram_mb": 7400}
  ]
}
```

- `status` lấy một trong: `pending`, `running`, `succeeded`, `failed`, `cancelled`.
- `approach.options` ghi lựa chọn đã quyết định cho mọi ô, kể cả ô để `auto`. Studio lọc và so sánh run theo trường này.

### 6.3 `d2s/summary.json`

Tóm tắt metric theo protocol của SPEC mục 9, do `eval.standard` ghi.

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
runs/.../<run_id>/d2s/stages/export.preview/
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
  "source_run": "runs/nuscenes_scene-0061/omnire/20261112-081000-omnire-a1b2",
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

- `job.json`: `schema_version`, `stage_id` hoặc loại việc, tham số đã gộp, đường dẫn đầu vào, `output_dir`.
- `result.json`: trạng thái, đường dẫn đầu ra, `peak_vram_mb`, thông báo lỗi nếu có.
- Trong lúc chạy, worker in ra stdout mỗi dòng một JSON: `{"event": "progress", "fraction": 0.42, "message": "..."}` hoặc `{"event": "metric", "name": "loss", "value": 0.031, "step": 4200}`.
- Mã thoát khác 0 nghĩa là thất bại.

### 9.2 Gói job cho Kaggle

```
bundle/
├── job.json                     # run_id, approach_id, preset, options, commit của repo
├── approach.yaml
├── cache_refs.json              # cache_key của các artifact đã có
├── resume.json                  # checkpoint để chạy tiếp, nếu có
└── scene_ref.json               # slug của Kaggle dataset chứa cảnh, scene_id, converter_version
```

Notebook chạy `d2s run start --bundle bundle/`, ghi kết quả ra `/kaggle/working/runs/` và `/kaggle/working/cache/` theo đúng cấu trúc mục 5 và 6. Checkpoint giữa chừng phải đủ để chạy tiếp (model và trạng thái optimizer). Lệnh `d2s kaggle pull` tải hai thư mục này về và ghép vào thư mục làm việc trên máy.

## 10. Quy tắc đặt tên và CLI

### 10.1 Định danh

| Định danh | Mẫu | Ví dụ |
|---|---|---|
| `scene_id` | `<dataset>_<sequence_id>`; chỉ gồm `a-z 0-9 _ -`, chữ thường | `nuscenes_scene-0061`, `waymo_10017090168044687777` |
| `split_id` | `<chiến lược><N>_<nhóm camera>` | `every8_allcams`, `every10_front3` |
| `approach_id` | `snake_case`, tên method, không chứa lựa chọn của ô | `omnire`, `streetgs` |
| Tên ô chọn | `snake_case`, danh từ ngắn | `init`, `renderer`, `campose`, `fastgs` |
| Tên preset | `snake_case`, mô tả mục đích | `paper`, `no_lidar`, `fisheye` |
| `stage_id` | `<nhóm>.<tên>`; nhóm là một trong `prep`, `init`, `train`, `eval`, `render`, `export`; tên là `snake_case` | `prep.masks_segformer`, `init.depth`, `export.ply_world` |
| `run_id` | `<YYYYMMDD-HHMMSS>-<approach_id>-<4 ký tự hex>` | `20261112-081000-omnire-a1b2` |
| `cache_key` | `<scene_id>/<stage_id>@<version>-<hash8>` | `nuscenes_scene-0061/init.depth@0.1.0-9b03c2d1` |
| Id camera, LiDAR | Giữ nguyên id của nguồn trong NCore, không đổi tên | `CAM_FRONT`, `camera_front_wide_120fov` |
| `schema_version` | `d2s-<loại>/<major>.<minor>` | `d2s-run/0.1` |

### 10.2 File và thư mục

- File người viết tay hoặc sửa tay dùng YAML: `manifest.yaml`, `<approach_id>.yaml`. File tool ghi ra dùng JSON: `run.json`, `summary.json`, `env.json`, `artifact.json`, `validation.json`, file split.
- Mảng số dùng NPZ; mask PNG; video MP4 H.264; Gaussian PLY.
- Tên file và thư mục: `snake_case`, chữ thường. Thư mục của một stage đặt đúng theo `stage_id`.
- Mã Python: module `snake_case`; lớp `PascalCase`; lớp stage đặt theo `<Nhóm><Tên>Stage`, ví dụ `PrepMasksSegformerStage` cho `prep.masks_segformer`.

### 10.3 Tên trường và tham số

- `snake_case` cho mọi khoá trong YAML, JSON và tham số stage.
- Hậu tố đơn vị bắt buộc khi trường có đơn vị: `_us` (micro giây), `_s` (giây), `_m` (mét), `_mb` (megabyte), `_px` (pixel). Góc lưu bằng radian, không hậu tố; độ chỉ dùng trong giao diện, hậu tố `_deg`.
- Phép biến đổi: `T_<a>_<b>` theo quy ước NCore (mục 1), ví dụ `T_camera_world`, `T_object_world`.
- Boolean: tính từ hoặc dạng `uses_`, `has_`: `cacheable`, `uses_lidar`, `has_cuboids`. Ô chọn bật/tắt dùng `true` / `false`.
- Giá trị liệt kê: chữ thường `snake_case`; mô hình camera: `opencv_pinhole`, `opencv_fisheye`, `ftheta`; renderer: `3dgs`, `3dgut`, `3dgeer` (engine đổi sang `render_mode` của DriveStudio: `default`, `ut`, `geer`).
- Định danh trong tham chiếu: `scene_id`, `split_id`, `approach_id`, `stage_id`, `run_id`, `cache_key`; không dùng `id` trơn trong file có nhiều loại định danh.

### 10.4 CLI

Lệnh có dạng `d2s <nhóm> <hành động> [tuỳ chọn]`. Tuỳ chọn dài viết `kebab-case`; giá trị khớp với khoá `snake_case` tương ứng.

| Lệnh | Việc |
|---|---|
| `d2s scene ingest <nguồn> --source-type <dataset> [--scene <scene_id>]` | Chuyển nguồn sang dữ liệu chuẩn, ghi manifest, kiểm tra |
| `d2s scene validate <scene_id>` | Kiểm tra lại, ghi `validation.json` |
| `d2s scene list` / `d2s scene show <scene_id>` | Liệt kê cảnh; xem manifest và báo cáo |
| `d2s split create <scene_id> --strategy every_nth_frame:8 --cameras <id,...> [--split <split_id>]` | Tạo split |
| `d2s approach list` / `d2s approach show <approach_id>` | Liệt kê approach; xem ô chọn, preset, ràng buộc |
| `d2s approach check <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <ô>=<giá trị>]...` | Kiểm tra điều kiện và in danh sách stage sẽ chạy, stage nào dùng lại cache |
| `d2s run start <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <ô>=<giá trị>]... [--executor local\|kaggle]` | Chạy |
| `d2s run resume <run_id>` | Chạy tiếp từ stage dở |
| `d2s run status <run_id>` / `d2s run list [--scene <scene_id>]` / `d2s run cancel <run_id>` | Theo dõi, liệt kê, huỷ |
| `d2s kaggle push <run_id>` / `d2s kaggle status <run_id>` / `d2s kaggle pull <run_id>` | Làm việc với Kaggle (thường do `run start --executor kaggle` tự gọi) |
| `d2s report <run_id>...` | Bảng so sánh nhiều run |
| `d2s export ply <run_id>` / `d2s export preview <run_id>` | Chạy lại stage export cho một run |
| `d2s cache list [--scene <scene_id>]` / `d2s cache gc [--scene <scene_id>]` | Xem, dọn cache |
| `d2s studio [--port <số>]` | Bật Studio |
| `d2s serve <run_id> [--port <số>]` | Bật server gRPC cho một run (L4) |

Quy tắc chung:
- Đối tượng chính đi theo vị trí (`<run_id>`, `<scene_id>`); các định danh khác đi theo tuỳ chọn (`--scene`, `--split`).
- `--set <ô>=<giá trị>` lặp lại được, dùng cho cả ô chọn và tham số stage (`--set train.drivestudio.num_iters=10000`).
- `--json` cho mọi lệnh liệt kê và xem, để script đọc được.
- Mã thoát: `0` thành công; `1` lỗi khi chạy; `2` sai cú pháp; `3` bị khoá vì không thoả điều kiện hoặc ràng buộc.
- Biến môi trường: `D2S_WORKDIR` (thư mục chứa `data/`, `cache/`, `runs/`; mặc định là thư mục hiện tại).
