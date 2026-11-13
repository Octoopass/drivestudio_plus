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
└── reports/validation.json
```

- `<khung>` là chỉ số 3 chữ số (`000`), `<camera>` là chỉ số camera của DriveStudio cho dataset đó.
- Cách lưu pose khác nhau giữa các dataset và **được giữ nguyên**, vì mỗi dataset đọc bằng loader riêng của nó (`drive2sim/recon/data/loaders/<dataset>.py`): Waymo, Argoverse, KITTI, nuPlan lưu `ego_pose/<khung>.txt` và `extrinsics/<camera>.txt`; PandaSet lưu `ego_pose/` và `extrinsics/<khung>_<camera>.txt`; nuScenes lưu `extrinsics/<khung>_<camera>.txt` và `lidar_pose/`. Bảng đầy đủ ở `ARCHITECTURE.md` mục 2.2.
- Nguồn mới không có loader riêng: converter ghi theo đúng khung thư mục của một dataset có sẵn rồi dùng loader của dataset đó (`ARCHITECTURE.md` mục 2.4).
- Thư mục cảnh chỉ chứa những gì có từ nguồn. Mask bầu trời và SMPL do stage `prep.*` sinh ra là artifact trong `cache/` (mục 5), không ghi vào đây.

### 2.2 `manifest.yaml`

```yaml
schema_version: d2s-manifest/1.0
scene_id: nuscenes_scene-0061
source:
  dataset: nuscenes               # waymo | nuscenes | nuplan | argoverse2 | pandaset | kitti | physicalai | carla | colmap | custom
  sequence_id: scene-0061
  converter: drivestudio.nuscenes # converter đã dùng: script tiền xử lý DriveStudio cho nuScenes
  converter_version: "0.1.0+96876ad"   # <phiên bản drive2sim>+<commit> lúc ingest
  data_license: "CC BY-NC-SA 4.0"
track: pinhole                    # pinhole | fisheye
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
- `cameras` xếp đúng thứ tự chỉ số camera của DriveStudio cho dataset đó, vì `images/<khung>_<camera>.jpg` và config dataset dùng chỉ số này; stage train đổi `camera_id` của split sang chỉ số theo thứ tự đó.
- `source.dataset` chọn loader và config dataset có sẵn của DriveStudio khi train.
- Thay đổi từ `0.1`: bỏ phần `ncore` (dữ liệu chuẩn không còn là NCore); file `0.x` không đọc được nữa.
- `cameras[].model` ghi đúng mô hình trong nguồn, nên có thể là `ideal_pinhole` (với nguồn NCore); approach chỉ khai báo nhận ba mô hình chuẩn (mục 10.3), và `ideal_pinhole` được coi là `opencv_pinhole` với hệ số méo bằng 0.
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

## 4. Approach

### 4.1 File khai báo

`drive2sim/approaches/<approach_id>.yaml`. Approach là danh sách stage chạy lần lượt; các khâu thay thế được cho nhau là *mục cấu hình*, quyết định trước khi chạy (`ARCHITECTURE.md` mục 2.5).

```yaml
schema_version: d2s-approach/0.2
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
  - {stage: render.videos}
  - {stage: export.ply_world}
```

- Một stage dùng được artifact của bất kỳ stage nào đứng trước. Engine nối theo kiểu artifact: stage cần `points_init` nhận artifact `points_init` gần nhất phía trước.
- `${options.<tên mục>}` được thay bằng lựa chọn cụ thể trước khi chạy. Sau khi thay, danh sách stage là cố định; không có điều kiện, nhánh song song hay vòng lặp lúc đang chạy.
- `groups`, và `label`, `help`, `labels`, `descriptions`, `group` của mục cấu hình, và `preset_help` chỉ dùng để hiển thị (thêm từ `0.2`; file `0.1` vẫn đọc được). Studio hiện các mục cấu hình theo nhóm; giải thích của mục cấu hình, của từng lựa chọn, của preset và lý do một lựa chọn bị khoá nằm trong icon (i), không hiện thành dòng chữ trên trang. Thuật toán phụ trợ (FastGS, và sau này các thuật toán tăng tốc khác) đặt mỗi thuật toán thành một mục cấu hình bật/tắt riêng trong nhóm `speedup`; tổ hợp chưa có file config thì khai báo trong `constraints`. Viết tiếng Việt như `reason` của ràng buộc. Thêm approach hay lựa chọn mới không cần sửa Studio: thiếu các trường này thì Studio hiện tên mục cấu hình và giá trị thô.
- Preset chỉ là cách điền nhanh các mục cấu hình. `run.json` ghi tên preset (`approach.preset`) chỉ khi cấu hình cuối đúng bằng preset đó; người dùng chọn preset rồi chỉnh một mục cấu hình khác đi (Studio, hoặc `--preset` kèm `--set` ở CLI) thì `preset` là `null`. Studio làm mờ lựa chọn vi phạm ràng buộc với các mục cấu hình đang chọn, kèm mục cấu hình cần đổi và `reason`.
- Tham số cuối cùng của một stage được gộp theo thứ tự: mặc định của stage → `params` trong file → preset → lựa chọn của người dùng (Studio hoặc `--set` của CLI).
- `train.drivestudio` chọn **một file config có sẵn** trong `drive2sim/configs/methods/` theo bốn tham số `method`, `renderer`, `campose`, `fastgs` (bảng `CONFIG_FILES` trong `drive2sim/stages/train/drivestudio.py`), không tự sửa tham số train. Ví dụ `streetgs` + `3dgeer` + `fastgs: true` dùng `streetgs_geer_fastgs.yaml`. Tổ hợp chưa có file config bị từ chối khi kiểm tra tham số, và file approach khai báo `constraints` để Studio, CLI khoá sẵn các tổ hợp đó.

### 4.2 Điều kiện chạy

Engine kiểm tra trước khi chạy. Approach hoặc một lựa chọn bị khoá, kèm lý do, khi:
- lựa chọn cần LiDAR (ví dụ `init.lidar`) mà `available.lidar` là `false`;
- `cuboids: required` mà `available.cuboids` là `none`;
- mô hình camera trong split không nằm trong `camera_models`, và danh sách stage không có `prep.undistort`;
- `strategy` của split không nằm trong `split_strategies`;
- `track` của approach khác `track` của cảnh;
- tổ hợp lựa chọn vi phạm một mục trong `constraints`.

Studio và CLI dùng chung bước kiểm tra này (`drive2sim/engine/resolve.py`). Quy tắc chi tiết:
- Giá trị của mục cấu hình gộp theo thứ tự: `default` → preset → lựa chọn của người dùng. Mục còn `auto` lấy lựa chọn **đầu tiên** trong `choices` không bị khoá vì dữ liệu, nên thứ tự `choices` là thứ tự ưu tiên.
- Stage có `uses_lidar: true` nghĩa là **cần** LiDAR; lựa chọn là stage đó bị khoá khi `available.lidar` là `false`.
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
| `depth_prior` | Depth metric theo camera và khung, mỗi ảnh một file `.npy` `float32`, mét; kèm tên model trong `artifact.json` |
| `points_init` | `points_init.npz`: `points` `float32[M,3]` trong hệ `world`, `colors` `uint8[M,3]`, `track_ids` (tuỳ chọn) |
| `model` | Thư mục log của codebase; với DriveStudio có `checkpoint_final.pth`, `config.yaml`, `metrics/`, `videos/` (mục 6.1) |
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
    └── stages/
        ├── train.drivestudio/           # thư mục log của DriveStudio
        │   ├── scene_view/000/          # cảnh ảo bằng symlink: thư mục cảnh + mask, SMPL từ cache
        │   ├── config.yaml              # config đầy đủ sau khi gộp
        │   ├── metrics.json             # metric trong lúc train
        │   ├── metrics/images_test_*.json
        │   ├── videos/ images/ configs_bk/ backup/
        │   ├── checkpoint_<bước>.pth     # chỉ model, sau mỗi lần lưu
        │   ├── checkpoint_resume.pth    # model + optimizer + grad scaler; xoá khi train xong
        │   ├── checkpoint_final.pth
        │   ├── train_stats.json         # thời gian train thực, VRAM, số Gaussian, iteration; ghi cùng mỗi checkpoint
        │   ├── log.txt                  # stdout và stderr của train.py
        │   ├── artifact.json
        │   └── done
        ├── eval.standard/               # summary.json, artifact.json, done
        └── export.ply_world/            # scene.ply, scene.json, artifact.json, done
```

Gốc thư mục run chỉ có `d2s/`, nên file của tool không trùng tên với file của codebase.

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
- `peak_vram_mb` do engine đo chỉ thấy bộ nhớ GPU của tiến trình engine, nên bằng 0 với stage chạy một tiến trình con (`prep.*`, `train.drivestudio`, `export.ply_world`). VRAM của train lấy từ `train_stats.json` qua `summary.json` (mục 6.3).
- `approach.quick` là `true` với run chạy thử nhanh (`--quick`): mọi stage có tham số `num_iters` train 3000 iteration, trừ khi người dùng tự đặt `num_iters` bằng `--set`. Run này để kiểm tra approach chạy hết từ đầu tới cuối; metric không so được với run đủ iteration, nên Studio gắn nhãn "thử nhanh" và không đưa run này vào bảng so sánh.
- Thay đổi từ `0.1`: thêm `created_utc`, `stages[].params`, `stages[].error`. Thay đổi từ `0.2`: thêm `approach.quick`, `stages[].reused_duration_s`; `duration_s` cộng dồn khi chạy tiếp. File `0.1`, `0.2` vẫn đọc được.

### 6.3 `summary.json`

Tóm tắt metric theo protocol của SPEC mục 9, do `eval.standard` ghi vào thư mục của nó: `d2s/stages/eval.standard/summary.json`.

```json
{
  "schema_version": "d2s-summary/0.3",
  "protocol": {"version": "0.2", "image_dtype": "uint8", "resolution": "downscale_when_loading=[2, 2, 2], downscale=1",
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
  "render_resolution": "800x450"
}
```

Các con số trong ví dụ chỉ minh hoạ định dạng. Metric vùng người, vùng xe chỉ có khi cảnh có box.

- Cách đo (`protocol.version` `0.2`): cuối `train.py`, sau bước train cuối, model vẽ lại mọi ảnh của tập test; ảnh render được làm tròn về 8 bit như ảnh gốc rồi so với ảnh camera từng ảnh một, sau đó lấy trung bình. PSNR là `−10·log10(MSE)` trên điểm ảnh hợp lệ; SSIM dùng `skimage.metrics.structural_similarity`; LPIPS dùng `torchmetrics` với backbone `alex`. `eval.standard` chỉ đọc kết quả này, không đo lại. Thay đổi từ `0.1`: `0.1` so ảnh render ở dạng số thực.
- Metric theo vùng (lấy từ metric test của DriveStudio): `human_psnr` vùng người (gồm cả xe đạp, xe máy vì người ngồi trên đó đổi dáng), `vehicle_psnr` vùng xe, `dynamic_psnr` vùng mọi vật động (`masked_psnr` của DriveStudio), `occupied_psnr` vùng **không phải bầu trời** (`Non-Sky PSNR`, không phải vùng vật động). Vùng người, xe, vật động là hình chữ nhật bao quanh box 3D chiếu lên ảnh.
- `train_time_s`, `peak_vram_mb`, `num_gaussians`, `num_iters` đọc từ `train_stats.json` mà `train.py` ghi cùng mỗi checkpoint. `train_time_s` là **thời gian train thực**: chỉ tính các bước train (lấy batch, forward, backward, cập nhật), không tính lúc khởi động, nạp checkpoint, vẽ ảnh kiểm tra, lưu checkpoint hay đánh giá cuối; chạy tiếp từ checkpoint thì cộng tiếp từ số đã lưu cùng checkpoint đó, nên các bước bị làm lại sau khi ngắt không bị tính hai lần. `num_gaussians` là tổng mọi node của model (nền, xe, người, vật biến dạng).
- `render_fps`: số khung chia cho thời gian model vẽ ảnh tập test (chỉ tính lời gọi model, có đồng bộ GPU, bỏ 3 khung đầu), ở độ phân giải `render_resolution`. FPS phụ thuộc GPU; tên GPU nằm trong `env.json`.
- Các trường trên để trống (`null`) khi không có số liệu, ví dụ model train bằng `train.py` cũ chưa ghi `train_stats.json`.
- `per_camera` để trống với DriveStudio, vì `train.py` không tách metric theo camera.
- Thay đổi từ `0.1`: `train_time_s`, `peak_vram_mb`, `num_gaussians` thành tuỳ chọn. Thay đổi từ `0.2`: thêm `dynamic_psnr`, `num_iters`, `render_fps`, `render_resolution`; `occupied_psnr` ghi đúng nghĩa là vùng không phải bầu trời.

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

- Engine ghi file khi tạo run (`machines` rỗng) rồi cập nhật **mỗi lần bắt đầu chạy hoặc chạy tiếp**, trên đúng máy chạy. Với run Kaggle đó là máy của notebook, không phải máy đã tạo run.
- `machines`: các máy đã chạy run theo thứ tự; chạy tiếp trên máy có cấu hình khác (ví dụ phiên Kaggle mới được cấp GPU khác) thì thêm một máy. `gpu` ở ngoài là GPU của máy chạy gần nhất.
- Cách đọc, không cần nạp PyTorch: CPU từ `/proc/cpuinfo`, RAM từ `/proc/meminfo`, GPU (tên, VRAM, driver, số GPU) từ `nvidia-smi`; phiên bản `pytorch`, `gsplat` từ metadata gói đã cài, `cuda` từ đuôi phiên bản PyTorch (`+cu117` → `11.7`).
- Thay đổi từ `0.1`: thêm `machines`; các trường phiên bản và `gpu` ghi lúc chạy thay vì lúc tạo run.

## 7. Gói xem trước

Gói xem trước cho phép xem cảnh trong trình duyệt mà không cần GPU CUDA (`ARCHITECTURE.md` mục 2.13). Không bắt buộc.

```
runs/.../<run_id>/d2s/stages/export.preview/
├── artifact.json
├── preview.json
├── background.ply               # Gaussian của nền tĩnh, hệ world
├── actors/<track_id>.ply        # Gaussian của từng xe (actor rigid), hệ của box
└── actor_poses.json
```

- File `.ply` theo định dạng PLY của 3DGS gốc: `x, y, z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`.
- `actor_poses.json`: `{"track_ids": [K tên], "T_object_world": [F khung][K xe]}`; mỗi phần tử là ma trận 4x4 viết thành 16 số theo hàng (đưa điểm từ hệ của xe sang `world`), hoặc `null` khi xe không có mặt ở khung đó. Khung đánh số theo thứ tự khung của DriveStudio (checkpoint không lưu mốc thời gian). Dùng JSON thay `.npz` để trình duyệt đọc thẳng, không cần numpy ở Studio; 200 khung × 30 xe khoảng 1 MB.
- `track_id` của xe là `rigid_<chỉ số>`, của người và vật biến dạng là `smpl_<chỉ số>`, `deformable_<chỉ số>`, theo thứ tự instance trong checkpoint.
- Do stage `export.preview` ghi (`drive2sim/recon/tools/export_preview.py`); nằm trong danh sách stage mặc định của OmniRe và Street GS, sau `export.ply_world`; run cũ hoặc approach khác thì thêm bằng `d2s export preview <run_id>` hay nút trên trang run. `GET /api/runs/{run_id}/splats` trả đường dẫn các file này và `scene.ply`.

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

- `box_only_actors`: actor không rigid; chưa hiện trong viewer (sau này hiện bằng box lấy từ cuboid của dữ liệu chuẩn).
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

## 9. Worker và Kaggle

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
└── resume.json       # {"kernel": notebook của lần đẩy trước, hoặc null}
```

- **Cảnh** được upload một lần thành dataset private `<người dùng>/d2s-<hash8>`, với `hash8` băm từ `<scene_id>@<converter_version>`; ingest lại bằng converter khác thì upload dataset mới.
- **Code**: notebook `git clone` đúng commit của repo (remote `origin`, hoặc biến `D2S_KAGGLE_REPO`), nên commit phải có trên remote; `kaggle push` từ chối nếu chưa push.
- **Notebook** `<người dùng>/d2s-<hash8 của run_id>-<lần đẩy>`, private, có GPU (`machine_shape` mặc định `NvidiaTeslaT4`, đổi bằng `--accelerator`) và internet. Notebook in phiên bản GPU và PyTorch, cài phụ thuộc (giữ PyTorch có sẵn), rồi chạy `d2s kaggle run bundle/ --workdir /kaggle/working --input-root /kaggle/input`.
- `d2s kaggle run` tìm cảnh dưới `--input-root` theo `manifest.yaml` (không giả định đường dẫn gắn của Kaggle), dựng `data/<scene_id>/` bằng symlink tới từng mục của cảnh, ghi `splits/` mới, rồi chạy run như `run resume`. Kết quả ở `/kaggle/working/runs/` và `/kaggle/working/cache/` theo đúng cấu trúc mục 5 và 6; ô cuối notebook xoá `data/` và mọi symlink để output chỉ còn kết quả.
- **Chạy tiếp**: phiên bị ngắt thì `d2s kaggle pull` rồi `d2s kaggle push` lần nữa. Lần đẩy sau gắn output của notebook trước (`kernel_sources`) và ghi tên nó vào `resume.json`; `d2s kaggle run` chép run và cache trong output đó vào thư mục làm việc trước khi chạy, nên stage đã xong không chạy lại và `train.drivestudio` chạy tiếp từ `checkpoint_resume.pth`.
- `d2s kaggle pull` tải output rồi ghép vào `d2s/` của run: mỗi stage có trong bản Kaggle thay stage cùng tên trên máy; stage chỉ có trên máy (ví dụ `export.preview` chạy trên máy sau lần tải trước) được giữ lại và nối vào cuối `run.json`; `env.json` lấy theo Kaggle cộng các máy chỉ có trên máy; `kaggle.json` giữ nguyên. Chép thêm các kết quả cache máy chưa có, không ghi đè cache đã có. Từ chối khi run đang chạy trên máy.

Chưa chạy thật: cờ của lệnh `kaggle` đã kiểm với Kaggle CLI 2.2.5, còn ô cài phụ thuộc trong notebook phụ thuộc câu hỏi mở về phiên bản PyTorch (ARCHITECTURE mục 8).

## 10. Quy tắc đặt tên và CLI

### 10.1 Định danh

| Định danh | Mẫu | Ví dụ |
|---|---|---|
| `scene_id` | `<dataset>_<sequence_id>`; chỉ gồm `a-z 0-9 _ -`, chữ thường | `nuscenes_scene-0061`, `waymo_10017090168044687777` |
| `split_id` | `every<N>_<nhóm camera>` với chiến lược `every_nth_frame:N`; `custom_<tên>` với chiến lược `custom`. Nhóm camera: `allcams` (mọi camera trong manifest), `front1` / `front3` (camera trước và hai bên), hoặc `<k>cams` (k camera theo thứ tự trong manifest) | `every8_allcams`, `every10_front3`, `custom_night_a` |
| `approach_id` | `snake_case`, tên method, không chứa lựa chọn của mục cấu hình | `omnire`, `streetgs` |
| Tên mục cấu hình | `snake_case`, danh từ ngắn | `init`, `renderer`, `campose`, `fastgs` |
| Tên preset | `snake_case`, mô tả mục đích | `paper`, `no_lidar`, `fisheye` |
| `stage_id` | `<nhóm>.<tên>`; nhóm là một trong `prep`, `init`, `train`, `eval`, `render`, `export`; tên là `snake_case` | `prep.masks_segformer`, `init.depth`, `export.ply_world` |
| `run_id` | `<YYYYMMDD-HHMMSS>-<approach_id>-<4 ký tự hex>` | `20261112-081000-omnire-a1b2` |
| `cache_key` | `<scene_id>/<stage_id>@<version>-<hash8>` | `nuscenes_scene-0061/init.depth@0.1.0-9b03c2d1` |
| Id camera, LiDAR | Giữ nguyên id của nguồn, không đổi tên | `CAM_FRONT`, `camera_front_wide_120fov` |
| `schema_version` | `d2s-<loại>/<major>.<minor>` | `d2s-run/0.1` |

### 10.2 File và thư mục

- File người viết tay hoặc sửa tay dùng YAML: `manifest.yaml`, `<approach_id>.yaml`. File tool ghi ra dùng JSON: `run.json`, `summary.json`, `env.json`, `artifact.json`, `validation.json`, file split.
- Mảng số dùng NPZ; mask PNG; video MP4 H.264; Gaussian PLY.
- Tên file và thư mục: `snake_case`, chữ thường. Thư mục của một stage đặt đúng theo `stage_id`.
- Mã Python: module `snake_case`; lớp `PascalCase`; lớp stage đặt theo `<Nhóm><Tên>Stage`, ví dụ `PrepMasksSegformerStage` cho `prep.masks_segformer`. Quy tắc đặt tên đầy đủ cho module, lớp, hàm, biến, tham số ở `AGENTS.md` mục 4.2.
- Package Python có `__init__.py`; thư mục chưa có file giữ trong git bằng `.gitkeep`, xoá `.gitkeep` khi thư mục đã có file.
- File môi trường trong `envs/`: `main.yml` cho môi trường chính, `<codebase>.yml` cho codebase ngoài.
- Config method (`drive2sim/configs/methods/`): `<method>.yaml` là config gốc; biến thể là `<method>_<biến thể>[_<biến thể>].yaml`, các biến thể xếp theo thứ tự cố định **renderer → densify → campose** (`geer`, `ut`; `fastgs`; `nocampose`), ví dụ `streetgs_geer_fastgs.yaml`. Biến thể kế thừa từ config gốc (mục 10.5). Config cũ chép nguyên file được giữ tới khi chuyển sang kế thừa (`ARCHITECTURE.md` mục 7.4 bước 5).
- Config dataset (`drive2sim/configs/datasets/<dataset>/`): `<k>cams.yaml`; biến thể `<k>cams_<biến thể>.yaml`, ví dụ `7cams_native.yaml` (giữ mô hình camera gốc, không khử méo).
- Mask thân xe: `drive2sim/assets/ego_masks/<dataset>/<chỉ số camera>.png`, chỉ số là vị trí camera trong danh sách `cameras` của manifest (bắt đầu từ 0). Mỗi dataset có một bảng chỉ số ↔ `camera_id` trong `drive2sim/recon/data/dataset_meta.py`; code mới dùng `camera_id`, chỉ đổi sang chỉ số ở biên với mã DriveStudio.
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
| `d2s scene ingest <nguồn> [--source-type <dataset>] [--version <phiên bản>] [--sequence <chuỗi gốc>] [--scene <scene_id>] [--interpolate-n <N>] [--workers <N>]` | Chuyển một chuỗi của nguồn sang `data/<scene_id>/`, ghi manifest và báo cáo kiểm tra; bỏ `--source-type` thì tự nhận dạng như `source inspect`. `scene_id` mặc định `<dataset>_<chuỗi gốc>`. Mã thoát `1` nếu báo cáo có lỗi. Hỗ trợ nuScenes, Waymo, Argoverse 2, PandaSet, KITTI, nuPlan, PhysicalAI; phiên bản và chuỗi được tự chọn khi chỉ có một |
| `d2s scene validate <scene_id>` | Kiểm tra lại, ghi `validation.json` |
| `d2s scene list` / `d2s scene show <scene_id>` | Liệt kê cảnh; xem manifest và báo cáo |
| `d2s scene delete <scene_id>` | Xoá dữ liệu chuẩn, split, cache và run của cảnh; hỏi xác nhận |
| `d2s split create <scene_id> --strategy every_nth_frame:8 [--cameras <id,...>] [--frame-range <đầu>,<cuối>] [--split <split_id>]` | Tạo split; mặc định mọi camera và cả cảnh, tên theo mục 10.1 |
| `d2s split list <scene_id>` / `d2s split show <scene_id> <split_id>` / `d2s split delete <scene_id> <split_id>` | Liệt kê, xem, xoá split |
| `d2s approach list` / `d2s approach show <approach_id>` | Liệt kê approach; xem mục cấu hình, preset, ràng buộc |
| `d2s approach check <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <mục>=<giá trị>]... [--quick]` | Kiểm tra điều kiện và in danh sách stage sẽ chạy, stage nào dùng lại cache |
| `d2s run start <approach_id> --scene <scene_id> --split <split_id> [--preset <tên>] [--set <mục>=<giá trị>]... [--quick] [--executor local\|kaggle]` | Chạy; `--quick` là chạy thử nhanh 3000 iteration (mục 6.2) |
| `d2s run resume <run_id>` | Chạy tiếp từ stage dở |
| `d2s run status <run_id>` / `d2s run list [--scene <scene_id>]` / `d2s run cancel <run_id>` / `d2s run delete <run_id>` | Theo dõi, liệt kê, huỷ, xoá thư mục run |
| `d2s kaggle push <run_id> [--branch <branch>] [--accelerator <GPU>]` / `d2s kaggle status <run_id>` / `d2s kaggle pull <run_id>` | Đẩy run lên Kaggle (lần sau chạy tiếp), xem trạng thái notebook, tải kết quả về (mục 9.2); `push` do `run start --executor kaggle` tự gọi |
| `d2s kaggle run <gói> [--input-root <thư mục>]` | Chỉ dùng trong notebook Kaggle: dựng thư mục làm việc từ gói job rồi chạy run |
| `d2s report <run_id>...` | Bảng so sánh nhiều run |
| `d2s export ply <run_id>` / `d2s export preview <run_id>` | Chạy lại stage export cho một run |
| `d2s cache list [--scene <scene_id>]` / `d2s cache gc [--scene <scene_id>]` | Xem, dọn cache |
| `d2s studio [--port <số>] [--kaggle-watch-minutes <phút>]` | Bật Studio. Studio cứ `--kaggle-watch-minutes` phút (mặc định 5; `0` để tắt) hỏi trạng thái notebook của các run đã đẩy lên Kaggle mà chưa tải kết quả lần đẩy mới nhất; notebook xong (thành công, lỗi, huỷ) thì tự xếp việc `kaggle_pull` như nút "Tải kết quả về". Không có run nào cần theo dõi thì không gọi Kaggle; hỏi lỗi thì khoảng chờ nhân đôi, tối đa 1 giờ |
| `d2s serve <run_id> [--port <số>]` | Bật server gRPC cho một run (L4) |

Quy tắc chung:
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
- Biến môi trường: `D2S_WORKDIR` (thư mục chứa `data/`, `cache/`, `runs/`; mặc định là thư mục hiện tại) và `D2S_DEVICE` (thiết bị chạy model trong stage, ví dụ `cuda:0`, `cpu`; mặc định `cuda:0`). `D2S_DEVICE` không phải tham số stage vì không được nằm trong khoá cache: mask sinh trên GPU hay CPU là như nhau. Không có biến môi trường nào khác; mọi cấu hình khác đi qua tuỳ chọn hoặc file.

### 10.5 Config và kế thừa

Áp dụng cho config method và config dataset trong `drive2sim/configs/` (file YAML người viết tay).

- Một config có thể khai báo `extends: <tên file>` ở cấp ngoài cùng; tên file tính tương đối với file đang đọc và chỉ được trỏ vào file **trong cùng thư mục** (`extends: streetgs.yaml`). Kế thừa được nhiều cấp; vòng lặp là lỗi.
- Gộp theo quy tắc: mapping gộp sâu theo khoá; mọi giá trị khác (số, chuỗi, boolean, **danh sách**) ở file con thay hẳn giá trị ở file cha; khoá có giá trị `null` ở file con là xoá khoá đó. Danh sách không gộp vì thứ tự và số phần tử của danh sách thường có nghĩa (ví dụ các mốc lịch learning rate).
- Engine gộp `extends` xong rồi mới đưa config cho DriveStudio (OmegaConf). Khoá `extends` bị bỏ khỏi config đã gộp; `config.yaml` ghi trong thư mục run là bản đã gộp, không còn `extends`.
- File con chỉ ghi phần khác với file cha. Lý do: sửa một tham số chung thì sửa ở một chỗ; đọc file con là thấy ngay biến thể khác gì.
- Lỗi khi đọc config (file cha không có, vòng lặp, YAML sai) báo kèm tên **cả hai** file và dừng ngay trước khi tạo thư mục run, để không để lại run rỗng.
- Config dataset vẫn chọn qua khoá `dataset: <dataset>/<k>cams` như DriveStudio; không dùng `extends` giữa config method và config dataset.

### 10.6 Ngôn ngữ

- **Tiếng Việt** cho mọi thứ người đọc: docstring, comment trong code và trong YAML mới, thông báo của CLI và của worker (`error.message`, `log`), tài liệu trong `docs/`, giao diện Studio mặc định.
- Ngoại lệ: thông báo của `ValueError` trong `drive2sim/core/` (identifiers, schema, files) là thông báo kỹ thuật cho lập trình viên, giữ tiếng Anh. CLI và API bắt các lỗi này và bọc thành `error.message` tiếng Việt kèm tên file và trường sai; không in thẳng `ValueError` cho người dùng.
- **Tiếng Anh** cho mọi thứ máy đọc hoặc chia sẻ với mã nguồn mở: tên biến, hàm, lớp, module, khoá YAML/JSON, `error.code`, tên sự kiện, tên lệnh và tuỳ chọn CLI, commit message, README gốc của DriveStudio và hướng dẫn dataset kế thừa từ upstream.
- File kế thừa từ DriveStudio (`drive2sim/recon/`, config method cũ, `docs/datasets/`) giữ comment tiếng Anh cho tới khi được sửa tới; phần sửa mới viết tiếng Việt.
- Studio có hai ngôn ngữ, mặc định tiếng Việt, chuyển sang tiếng Anh bằng nút trên giao diện; lựa chọn lưu trong trình duyệt. Chuỗi giao diện đặt trong `drive2sim/studio/frontend/src/locales/vi.json` và `en.json`, khoá dạng `<màn hình>.<thành phần>.<tên>` (`runs.table.status`); không viết chuỗi cứng trong component. Thông báo từ API (`error.message`) đi thẳng ra giao diện và chỉ có tiếng Việt; `error.code` là khoá để giao diện tra bản dịch khi đang ở tiếng Anh.

## 11. API của Studio

FastAPI phục vụ REST và Server-Sent Events, chỉ nghe trên `127.0.0.1` (`ARCHITECTURE.md` mục 2.10). API chỉ là một lớp mỏng trên engine: mọi lệnh CLI ở mục 10.4 có một route tương ứng, cùng bước kiểm tra, cùng mã lỗi.

### 11.1 Route

- Tiền tố `/api/`. Nhóm route trùng nhóm lệnh CLI, viết **số nhiều**: `/api/scenes`, `/api/splits`, `/api/approaches`, `/api/runs`, `/api/cache`, `/api/kaggle`, `/api/report`. Định danh đi trong đường dẫn: `/api/runs/{run_id}`, `/api/scenes/{scene_id}/splits/{split_id}`.
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
- Frontend tĩnh phục vụ ở `/`; viser nhúng ở `/viz/`.

Route đã có (`drive2sim/studio/api/routes/`):

| Route | Việc | Tương ứng CLI |
|---|---|---|
| `GET /api/scenes`, `GET /api/scenes/{scene_id}` | Liệt kê cảnh; manifest, `validation` (báo cáo kiểm tra hoặc `null`), `splits` (danh sách `split_id`) | `scene list`, `scene show` |
| `GET /api/scenes/{scene_id}/frames/{frame_index}` | `{"frame_index", "images": [{"camera_id", "url"}]}` cho lưới ảnh đồng bộ; `url` là `null` khi thiếu ảnh | |
| `GET`, `POST /api/scenes/{scene_id}/splits`, `GET .../splits/{split_id}` | Liệt kê, tạo, xem split; thân của `POST`: `strategy`, `cameras`, `frame_range`, `split_id` | `split list`, `split create`, `split show` |
| `GET /api/approaches`, `GET /api/approaches/{approach_id}` | Đọc file approach | `approach list`, `approach show` |
| `POST /api/approaches/{approach_id}/check` | Thân: `scene_id`, `split_id`, `preset`, `options`, `stage_params`, `quick`. Trả `quick`, `options` đã quyết định, `lidar_usage`, `stages` (kèm `cache_hit`) và `blocked_choices` (mục → lựa chọn bị khoá kèm lý do) để giao diện làm mờ | `approach check` |
| `POST /api/runs` | Thân như `check` cộng `approach_id`, `executor`; tạo run, xếp vào hàng đợi chạy nền (`local`) hoặc đẩy lên Kaggle ở nền (`kaggle`) | `run start` |
| `GET /api/runs`, `GET /api/runs/{run_id}` | Liệt kê (lọc `scene_id`, `status`); `run.json` cộng `summary` và `run_dir` | `run list`, `run status` |
| `POST /api/runs/{run_id}/resume` | Chạy tiếp; `409` nếu run đang chạy | `run resume` |
| `GET /api/runs/{run_id}/events`, `GET /api/events` | Sự kiện SSE (mục 11.4) | |
| `GET /api/runs/{run_id}/log` | `{"engine": <d2s/log.txt>, "stages": {stage_id: <200 dòng cuối log.txt>}}` | |
| `GET /api/runs/{run_id}/videos` | Video MP4 trong thư mục run: `{"name", "stage_id", "url"}` | |
| `GET /api/runs/{run_id}/artifacts` | Nơi lưu kết quả của từng stage: `{"run_dir", "run_dir_abs", "files" (file của tool trong `d2s/`), "stages": [{"stage_id", "status", "location": "run" \| "cache", "path", "path_abs", "size_bytes", "files": [{"name", "is_dir", "size_bytes", "path"}]}]}`; `path` tương đối với thư mục làm việc. Run chạy trên Kaggle sau `kaggle pull` cũng nằm ở đây (mục 9.2) | |
| `GET /api/runs/{run_id}/curve` | Đường cong train đọc từ `metrics.json` của stage `train.*` (mục 6.1): `{"stage_id", "series": {"psnr", "loss", "gaussians": [[bước, giá trị], ...]}}`; PSNR trên ảnh train, tổng các loss, tổng số Gaussian; tối đa 400 điểm mỗi đường | |
| `POST /api/reveal` | Thân: `path` (tương đối với thư mục làm việc). Mở thư mục đó (với file: thư mục chứa file) trong trình quản lý file của máy chạy Studio; đường dẫn ra ngoài thư mục làm việc, không tồn tại, hoặc máy không có giao diện đồ hoạ thì `409` kèm lý do | |
| `POST /api/pick-folder` | Thân: `start` (tuỳ chọn, thư mục mở sẵn). Mở hộp thoại chọn thư mục của hệ điều hành trên máy chạy Studio (Linux: `zenity` hoặc `kdialog`; macOS: `osascript`; Windows: PowerShell) và chờ người dùng chọn; trả `{"path"}` là đường dẫn tuyệt đối, hoặc `null` nếu người dùng đóng hộp thoại. Máy không có giao diện đồ hoạ hoặc thiếu công cụ thì `409` kèm lý do. Dùng ở form nạp cảnh, vì trình duyệt không cho trang web biết đường dẫn thật của thư mục | |
| `GET /api/runs/{run_id}/splats` | `{"preview", "ply"}`: gói xem trước (`preview.json` cộng `background_url`, `actor_urls`, `poses_url`) nếu đã chạy `export.preview`, và url của `scene.ply` | |
| `GET /api/runs/{run_id}/render-check` | `{"ready", "checks": [{"id", "ok", "value"}]}`: máy chạy Studio có render được run này bằng GPU không. `id` theo thứ tự: `gpu` (nvidia-smi; kèm `memory_mb`), `pytorch` (gói đã cài), `cuda` (PyTorch thấy GPU, hỏi trong tiến trình con), `gsplat`, `checkpoint` (`checkpoint_final.pth` của stage `train.*`); `value` là thứ đã thấy hoặc lý do lỗi. Chỉ kiểm tra, không nạp model | |
| `POST /api/runs/{run_id}/gpu-viewer`, `GET /api/gpu-viewer`, `DELETE /api/gpu-viewer` | Bật, xem, tắt viewer GPU (`drive2sim/recon/tools/viewer.py --host 127.0.0.1`) cho một run; mỗi lúc một viewer, bật run khác thì viewer cũ tắt. Trước khi bật, Studio dựng lại cảnh ảo trong `d2s/viewer/scene_view/` từ thư mục cảnh và cache trên máy (mask, SMPL), và ghi `d2s/viewer/config.yaml` là `config.yaml` của stage train với `data.data_root` trỏ vào đó (run tải từ Kaggle có đường dẫn Kaggle). Trạng thái `{"state": "stopped" \| "starting" \| "running" \| "exited", "run_id", "url", "exit_code", "log"}`; máy chưa đủ điều kiện hoặc run chưa có checkpoint thì `409` | |
| `GET /api/kaggle/setup/status` | Kiểm tra trước khi chạy trên Kaggle: `cli` (lệnh `kaggle` đã cài, phiên bản), `credentials` (khoá lấy từ `env`, `access_token`, `kaggle.json` hay `oauth`; thư mục cấu hình), `account` (đăng nhập được không, tên), `gpu` (giờ GPU đã dùng, còn lại, tổng, lúc làm mới; từ `kaggle quota`), `code` (commit đã push chưa, có thay đổi chưa commit không), và `ready` | |
| `POST /api/kaggle/setup/token` | Thân: `token`, `username` (chỉ với khoá kiểu cũ), `replace`. Ghi khoá vào thư mục cấu hình của Kaggle CLI (`KAGGLE_CONFIG_DIR` hoặc `~/.kaggle`), quyền `600`: token `KGAT_...` vào `access_token`, khoá cũ vào `kaggle.json`. Không trả lại khoá; khoá sai dạng `400`, đã có khoá mà `replace` là `false` thì `409` | |
| `GET /api/studio` | `{"workdir", "demo"}`: thư mục làm việc, và `demo` là `true` khi đó là thư mục dữ liệu mẫu (có file `DEMO`, do `drive2sim/scripts/make_demo.py` tạo); Studio hiện nhãn "Dữ liệu mẫu" | |
| `GET /api/report` | Bảng so sánh (lọc `scene_id`, `split_id`, `run_id` lặp lại được); mỗi dòng có metric của `summary.json`, `avg_error` (chỉ số gộp của Mip-NeRF: trung bình nhân của `10^(−PSNR/10)`, `√(1 − SSIM)`, LPIPS; thấp là tốt), cờ `quick`, `num_iters`, `train_time_s` (train thực), `recon_time_s` (tổng thời gian các stage từ đầu tới khi xuất xong `scene.ply`, stage dùng lại cache tính thời gian gốc), `render_fps`, `num_gaussians` (cả model), `ply_gaussians` và `ply_size_mb` (của `scene.ply`), `preview_gaussians` và `preview_size_mb` (tổng mọi file của gói xem trước), `occupied_psnr`, `gpu`, và `stage_times` (`stage_id` → `{"seconds", "reused"}`, stage dùng lại cache tính thời gian gốc; không có trong CSV); `?format=csv` để xuất | `report` |
| `POST /api/sources/inspect` | Thân `{"path"}`; kết quả nhận dạng như `source inspect --json`; `404` nếu không có thư mục | `source inspect` |
| `POST /api/sources/ingest` | Thân `path`, `source_type`, `version`, `sequence`, `scene_id`, `interpolate_n`; chọn chuỗi ngay (`400` kèm các lựa chọn nếu thiếu), rồi ingest nền; trả `202` kèm việc | `scene ingest` |
| `GET /api/jobs`, `GET /api/jobs/{job_id}` | Việc nền (ingest, đẩy và tải Kaggle): `job_id`, `kind` (`ingest`, `kaggle_push`, `kaggle_pull`), `label`, `status` (`queued`, `running`, `succeeded`, `failed`), `result` (`scene_id` hoặc `run_id`), `error`; giữ trong bộ nhớ tới khi tắt Studio | |
| `GET /api/kaggle/{run_id}` | `kaggle.json` của run cộng `status` của notebook | `kaggle status` |
| `POST /api/kaggle/{run_id}/push`, `POST /api/kaggle/{run_id}/pull` | Đẩy, tải ở nền; trả `202` kèm việc | `kaggle push`, `kaggle pull` |
| `POST /api/scenes/{scene_id}/validate` | Kiểm tra lại, ghi và trả báo cáo | `scene validate` |
| `DELETE /api/scenes/{scene_id}`, `DELETE /api/scenes/{scene_id}/splits/{split_id}` | `{"removed": [...]}`; `409` khi run đang chạy hoặc split còn run dùng | `scene delete`, `split delete` |
| `POST /api/runs/{run_id}/cancel` | Ghi yêu cầu huỷ (mục 6.1); trả `run.json` hiện tại | `run cancel` |
| `DELETE /api/runs/{run_id}` | `{"removed": [...]}`; `409` khi run đang chạy | `run delete` |
| `POST /api/runs/{run_id}/export` | Thân `{"kind": "ply" \| "preview"}`; đặt lại stage export rồi xếp run vào hàng đợi | `export ply`, `export preview` |
| `GET /api/cache`, `POST /api/cache/gc` | Liệt kê (kèm `size_bytes`, `used_by`); dọn, thân `scene_id`, `dry_run` | `cache list`, `cache gc` |
| `GET /files/<đường dẫn>` | File trong `D2S_WORKDIR`; đường dẫn ra ngoài thư mục làm việc trả `404` | |

Run chạy nền **lần lượt từng run** trong một luồng (`drive2sim/studio/api/jobs.py`): một người, một máy, hai run train cùng lúc sẽ tranh GPU. Việc nền dài (ingest, Kaggle) chạy ở luồng riêng, nên không phải chờ run train xong.

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
| 404 | `not_found` | `scene_id`, `run_id`... không tồn tại | 1 |
| 409 | `conflict` | Đối tượng đã tồn tại, hoặc run đang chạy mà bị yêu cầu xoá | 1 |
| 422 | `blocked` | Không thoả điều kiện dữ liệu hoặc `constraints` (mục 4.2) | 3 |
| 500 | `internal` | Lỗi không lường trước; `message` không lộ traceback, traceback vào log | 1 |
| 502 | `kaggle_failed` | Lệnh `kaggle` lỗi: chưa cài, chưa có khoá API, commit chưa push, notebook trước còn chạy | 1 |
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
