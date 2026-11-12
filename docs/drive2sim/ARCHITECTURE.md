# drive2sim: kiến trúc

Tài liệu này mô tả cách xây tool theo `SPEC.md`, ngay trên repo này. Định dạng dữ liệu và interface nằm trong `CONTRACTS.md`.

Đây là **thiết kế đích**. Mục 7 so sánh với hiện trạng của repo và đưa ra lộ trình refactor. Những gì đã kiểm tra trong mã nguồn hoặc tài liệu gốc thì ghi rõ nguồn; những gì chưa kiểm tra thì ghi là chưa kiểm tra.

Trạng thái: bản nháp, ngày 12/11/2026.

## 1. Tổng quan

Repo gồm bốn tầng:

```
┌─ 1. DỮ LIỆU ──────────────────────────────────────────────────────────────────────────┐
│ ingest/<nguồn> ──► data/<scene>/ : NCore V4 + manifest.yaml + reports/validation.json │
│ (7 dataset, CARLA, mentor, COLMAP; mỗi cảnh chạy một lần, không phụ thuộc approach)   │
└───────────────────────────────────────┬───────────────────────────────────────────────┘
┌─ 2. PIPELINE (approach = danh sách stage, có ô chọn và preset) ───────────────────────┐
│  prep.*  ──►  init.*  ──►  train.*  ──►  eval.standard  ──►  render.*  ──►  export.*  │
│  select       lidar        drivestudio   (một protocol      videos        ply_world   │
│  undistort    depth        external:X    cho mọi approach)                preview     │
│  masks        slam                                                                    │
│  smpl                                                                                 │
│  ── cache dùng chung giữa các approach ──┤── trong thư mục run, đánh dấu done ────────│
└───────────────────────────────────────┬───────────────────────────────────────────────┘
┌─ 3. TÁI DỰNG (DriveStudio, sau refactor) ─────────────────────────────────────────────┐
│ drive2sim/recon/: data/ncore (một loader) · models/nodes · trainers · renderer 3DGS/3DGUT/3DGEER │
│ train.drivestudio nhận points_init từ ngoài; không tự khởi tạo, không tự đánh giá     │
└───────────────────────────────────────┬───────────────────────────────────────────────┘
┌─ 4. DÙNG KẾT QUẢ ─────────────────────────────────────────────────────────────────────┐
│ engine + cli (d2s) · kaggle · studio (api, frontend, viewer)                           │
│ sim: Renderer (L2) · ActorEditor (L3) · server gRPC (L4)                               │
└───────────────────────────────────────────────────────────────────────────────────────┘
   thư mục làm việc: data/ (dữ liệu chuẩn) · cache/ (kết quả stage dùng chung) · runs/ (kết quả run)
```

Nguyên tắc chung:
1. **Một dữ liệu chuẩn duy nhất.** Không approach nào đọc dữ liệu gốc.
2. **Approach là một tổ hợp stage.** Mỗi stage làm một việc, đầu vào và đầu ra có kiểu rõ ràng.
3. **Tách đúng những khâu mà các approach khác nhau:** khởi tạo, train, renderer, đánh giá, export.
4. **Chỉ thêm độ phức tạp khi có lý do:** thứ tự chạy tuyến tính, cache chỉ cho thứ dùng chung, không cơ sở dữ liệu.
5. **Các thành phần dùng kết quả** (Studio, Kaggle, server) chỉ đọc qua `run.json`, `Renderer`, `ActorEditor`; không biết bên trong approach.

Toàn bộ mã nằm trong một package `drive2sim/`. Tầng 3 là mã DriveStudio có sẵn, đã chuyển vào `drive2sim/recon/` (mục 3), và được refactor tiếp ở những chỗ ghi trong mục 7.

## 2. Quyết định thiết kế

Mỗi quyết định gồm: giải thích công cụ hoặc khái niệm, lý do, ưu điểm, nhược điểm.

| # | Quyết định |
|---|---|
| 2.1 | Xây tool ngay trong repo `drivestudio_plus`, trên branch `drive2sim`, tái cấu trúc toàn bộ thành một package |
| 2.2 | Dùng NCore V4 làm định dạng dữ liệu chuẩn |
| 2.3 | Tách ingest (chung cho mọi approach) khỏi các stage chuẩn bị riêng cho approach |
| 2.4 | Một loader NCore chung thay cho loader riêng của từng dataset |
| 2.5 | Approach là danh sách stage chạy tuyến tính, có ô chọn, ràng buộc và preset |
| 2.6 | Lưu trạng thái bằng thư mục, không dùng cơ sở dữ liệu |
| 2.7 | Cache chỉ cho stage dùng chung (`prep.*`, `init.*`), khoá đọc được |
| 2.8 | Train trên Kaggle qua một executor riêng |
| 2.9 | Môi trường: một môi trường cho repo, môi trường riêng cho codebase ngoài |
| 2.10 | API của Studio: FastAPI và Server-Sent Events, chỉ nghe trên `127.0.0.1` |
| 2.11 | Giao diện Studio: Vite + React |
| 2.12 | Viewer 3D chính: viser + nerfview, render ở server |
| 2.13 | Chế độ xem trước không cần GPU CUDA: Spark trong trình duyệt, file PLY |
| 2.14 | Server mô phỏng theo proto `SensorsimService` của NuRec |
| 2.15 | Ba mô hình camera chuẩn: `opencv_pinhole`, `opencv_fisheye`, `ftheta` |

### 2.1 Xây tool trong repo `drivestudio_plus`, tái cấu trúc toàn bộ thành một package

**Khái niệm.** `drivestudio_plus` ([Octoopass/drivestudio_plus](https://github.com/Octoopass/drivestudio_plus)) là fork DriveStudio của một thành viên trong nhóm. Repo gồm:
- DriveStudio ở commit `e59bda4` (MIT): OmniRe, Street Gaussians, PVG, Deformable-GS, loader của 6 dataset, script tiền xử lý, `tools/eval.py`.
- Patch 3DGEER/3DGUT của Bosch (AGPL-3.0), áp nguyên vẹn từ nhánh `gsplat-geer`, commit `f15cee2`. Patch này thêm: renderer chọn bằng `trainer.render.render_mode: default | ut | geer`, camera fisheye/F-theta gốc, rolling shutter, loader PhysicalAI (đọc NCore V4) và `tools/viewer.py` (viser 1.0.30 + nerfview 0.1.2).
- Phần teammate thêm: densify và prune kiểu FastGS (`configs/fastgs.md`), `tools/export_ply.py`, `datasets/tools/extract_masks_hf.py`, các config `*_nocampose`.

Tool được làm trên branch `drive2sim`. Trên branch này, mã DriveStudio được **chuyển hẳn vào package `drive2sim/recon/`** (mục 3), cùng một `pyproject.toml`, thay vì để các thư mục `datasets/`, `models/`, `utils/`, `tools/` ở gốc repo. Branch `drive2sim` được dùng làm dòng chính của nhóm từ đây.

**Lý do.**
- Repo đã làm được phần lớn mức L1–L2. Làm một repo riêng bọc bên ngoài thì phải viết lại lớp dữ liệu và lớp điều phối mà DriveStudio đã có.
- L3 (chỉnh actor) và L4 (gRPC) phải chạm vào bên trong trainer và node; làm trong cùng package thì gọi trực tiếp được.
- Một package, một quy tắc đặt tên (CONTRACTS mục 10) cho toàn bộ mã thì dễ bảo trì nhất.
- Thư mục gốc tên `datasets/` trùng tên với thư viện `datasets` của Hugging Face; chuyển vào `drive2sim.recon.data` thì hết trùng.

**Ưu điểm.**
- Cài bằng `pip install -e .`; import theo đường dẫn đầy đủ `drive2sim.recon...`, không cần đặt `PYTHONPATH` khi đã cài.
- Mã mới và mã DriveStudio cùng một cấu trúc.

**Nhược điểm.**
- Không merge trực tiếp được với `main` cũ: mọi file đã đổi chỗ. Thay đổi teammate làm trên `main` sau thời điểm tách phải chuyển tay sang branch `drive2sim`.
- Các run cũ lưu `config.yaml` với chuỗi import cũ (`models.nodes.RigidNodes`, `datasets.<dataset>...`) không nạp lại được bằng mã mới nếu không đổi chuỗi.
- Phải chứng minh hành vi không đổi sau khi di chuyển: train lại cùng cảnh trên Kaggle và so PSNR (mục 10).
- Nhánh `main` có code AGPL-3.0, nên toàn bộ package cũng phân phối theo AGPL.

### 2.2 NCore V4 làm định dạng dữ liệu chuẩn

**Khái niệm.** NCore là định dạng dữ liệu cảm biến cho xe tự lái của NVIDIA (Apache-2.0); NuRec là một trong những chương trình đọc nó. Theo `docs/data/` của repo NCore:
- Một bản ghi gồm nhiều *component group*, mỗi group là một kho zarr, lưu dạng thư mục hoặc một file `.itar`.
- Component có sẵn: pose, intrinsics, mask tĩnh, ảnh camera, LiDAR, radar, box 3D (cuboid), point cloud, và nhãn theo ảnh (depth, segmentation, mask, normal, feature).
- Mô hình camera: F-theta, pinhole lý tưởng, pinhole OpenCV, fisheye OpenCV; có thêm mô hình méo do kính chắn gió.
- Thời gian tính bằng micro giây (`uint64`); mỗi ảnh có thời điểm bắt đầu và kết thúc, dùng cho rolling shutter.
- Hệ toạ độ: `rig` x tới trước, y sang trái, z lên trên, gốc ở giữa trục sau trên mặt đất; `world` đặt pose đầu tiên của xe làm gốc; camera theo OpenCV.

Đọc và ghi bằng gói `nvidia-ncore` (PyPI, bản 19.9.0, Python ≥ 3.8). Repo NCore có converter cho Waymo, nuScenes, Argoverse 2, KITTI, PhysicalAI và COLMAP (`drive2sim/recon/tools/data_converter/`).

**Lý do.**
- Cần một định dạng chung cho mọi nguồn, đúng tinh thần NCore của NuRec. NCore có sẵn converter cho phần lớn nguồn cần dùng, nên ít phải tự viết nhất.
- Có đủ ba mô hình camera và rolling shutter, khớp `CameraSpec` của proto `SensorsimService` (mục 2.14).
- NuRec đọc thẳng NCore, nên muốn so với NuRec thì không phải chuyển dữ liệu thêm lần nữa.
- Patch Bosch đã có một đường đọc NCore (`drive2sim/recon/data/physicalai/`), làm nền cho loader chung (mục 2.4).

**Ưu điểm.**
- Không phải tự thiết kế và giữ một định dạng riêng.
- Nhãn theo ảnh (mask, depth) lưu được ngay trong NCore.

**Nhược điểm.**
- File zarr/itar không xem trực tiếp bằng mắt được; cần công cụ đọc.
- Hướng dẫn chính thức chạy converter bằng Bazel. Code converter là Python thường, import `ncore` từ PyPI cùng vài thư viện (`click`, `pycolmap` cho COLMAP), nên có vẻ chạy được không cần Bazel; chưa thử.
- Không có converter cho nuPlan, PandaSet, CARLA và dữ liệu mentor; phải tự viết theo khung `ncore.data_converter`.
- Quy ước `T_a_b` của NCore đưa điểm từ hệ `a` sang hệ `b`, ngược với nhiều codebase khác. Mọi chỗ chuyển đổi phải theo một hàm chung (CONTRACTS mục 1).

### 2.3 Tách ingest khỏi các stage chuẩn bị riêng

**Khái niệm.**
- *Ingest*: chuyển mọi thứ có trong nguồn sang dữ liệu chuẩn, không bỏ bớt; ghi manifest và báo cáo kiểm tra. Không phụ thuộc approach; mỗi cảnh chạy một lần.
- *Stage chuẩn bị* (`prep.*`, `init.*`): chạy trong pipeline của approach; lấy bớt (camera, đoạn khung, độ phân giải, khử méo), sinh thêm (mask, SMPL, depth prior, điểm khởi tạo) và đổi dạng sang thứ codebase đọc được.

**Lý do.**
- Mọi approach bắt đầu từ cùng một dữ liệu, nên so sánh công bằng.
- Thử approach thứ hai trên cùng cảnh không phải ingest lại.
- Có manifest sau ingest thì mới khoá đúng approach, ô chọn không chạy được.

**Ưu điểm.** Kết quả của stage chuẩn bị được cache và dùng chung (mục 2.7).

**Nhược điểm.** Dữ liệu chuẩn giữ đủ mọi thứ nên tốn ổ đĩa hơn bản đã lọc cho một approach.

### 2.4 Một loader NCore chung thay cho loader riêng của từng dataset

**Khái niệm.** DriveStudio hiện có 7 loader riêng (`drive2sim/recon/data/*/*_sourceloader.py`, 460–720 dòng mỗi file, tổng 3678 dòng). Thư mục sau tiền xử lý có cùng khung, nhưng mỗi loader tự xử lý quy ước calib, trục camera và box của dataset mình. Phần thật sự chung là interface Python (`ScenePixelSource`, `SceneLidarSource`). Loader được chọn trong config dataset bằng `data.pixel_source.type`, ví dụ `datasets.physicalai.physicalai_sourceloader.PhysicalAIPixelSource`.

Thiết kế đích: một đường duy nhất `NCore → thư mục DriveStudio → loader NCore`, mở rộng từ đường PhysicalAI có sẵn.

**Lý do.** Nguồn mới chỉ cần converter sang NCore, không phải viết thêm loader 500 dòng.

**Ưu điểm.** Mọi approach dòng DriveStudio đọc dữ liệu qua một chỗ; sửa lỗi ở một chỗ.

**Nhược điểm.**
- Phải sửa phần dữ liệu của teammate; cần báo trước.
- Phải chứng minh không tụt chất lượng: train cùng cảnh bằng loader cũ và loader mới, so PSNR (SPEC mục 13). Loader cũ chỉ được bỏ sau khi qua bước này.
- Đường PhysicalAI hiện gắn cứng nhiều chỗ (mục 7.2).

### 2.5 Approach là danh sách stage, có ô chọn, ràng buộc và preset

**Khái niệm.**
- *Stage* là một lớp Python đăng ký theo tên (ví dụ `init.depth`), khai báo phiên bản, tham số, kiểu artifact vào và ra, có dùng LiDAR không, có được cache không (CONTRACTS mục 5.1).
- *Approach* là file `drive2sim/approaches/<id>.yaml` gồm: danh sách stage chạy lần lượt; *ô chọn* (`options`) cho những khâu thay thế được cho nhau; *ràng buộc* (`constraints`) giữa các ô; *preset* là tổ hợp ô đặt sẵn; cùng track, mức năng lực, điều kiện dữ liệu (CONTRACTS mục 4).
- *Tuyến tính*: các stage chạy đúng thứ tự trong file, mỗi lần một stage. Một stage dùng được kết quả của bất kỳ stage nào đứng trước. Không có điều kiện, nhánh song song hay vòng lặp lúc đang chạy.
- *Ô chọn* được quyết định **trước khi chạy** (người dùng chọn trên Studio hoặc CLI, hoặc engine tự chọn theo manifest với giá trị `auto`). Sau khi thay ô bằng lựa chọn cụ thể, danh sách stage là cố định.

**Lý do.**
- Các approach của dự án khác nhau đúng ở từng khâu: khởi tạo (LiDAR, depth, SLAM), renderer (3DGS, 3DGUT, 3DGEER), luật densify (mặc định, FastGS), CamPose, có bước feed-forward hay không. Stage biến mỗi khâu thành một mảnh thay được.
- Ô chọn tránh nhân file: 2 cách khởi tạo × 3 renderer × FastGS bật/tắt là 12 tổ hợp nhưng vẫn là một approach. Thí nghiệm "chỉ đổi một khâu rồi so" là đổi một ô.
- Mọi nhu cầu hiện có đều làm được mà không cần rẽ nhánh: điều kiện chuyển thành ô chọn `auto` hoặc khoá theo manifest; so sánh song song chuyển thành nhiều run dùng chung cache; vòng lặp đặt trong một stage.

**Quy tắc dùng ô chọn.**
1. Ô chọn chỉ dành cho khâu thay thế được cho nhau: khởi tạo, renderer, luật densify, CamPose, độ phân giải. Method khác nhau (OmniRe so với Street GS) hoặc codebase khác nhau là approach riêng.
2. Tổ hợp không hợp lệ phải khai báo trong `constraints`, kèm lý do; Studio và CLI khoá tổ hợp đó.

**Ưu điểm.**
- Thay một khâu không đụng khâu khác; mỗi stage test riêng được.
- Thêm approach mới chủ yếu là ghép stage có sẵn; thêm một lựa chọn mới (ví dụ `init.slam`) là thêm một giá trị vào ô.
- Studio so sánh được theo từng ô, vì `run.json` ghi lại lựa chọn.

**Nhược điểm.**
- Phải viết engine nhỏ (registry, thay ô, kiểm tra ràng buộc, chạy tuần tự, đánh dấu done).
- Phải tách khởi tạo và đánh giá ra khỏi `drive2sim/recon/tools/train.py` và trainer của DriveStudio (mục 7.2).
- File approach có ô chọn khó đọc hơn một file cố định: phải nhìn cả ô, giá trị mặc định và preset.
- Nếu sau này cần rẽ nhánh lúc đang chạy thì phải nâng engine lên kiểu đồ thị; các stage đã viết vẫn dùng lại được vì chỉ khai báo đầu vào và đầu ra.

### 2.6 Lưu trạng thái bằng thư mục, không dùng cơ sở dữ liệu

**Khái niệm.** Mọi thứ tool cần nhớ là file trong `data/`, `cache/`, `runs/`. Không có SQLite, PostgreSQL hay dịch vụ chạy nền.

**Lý do.** Tool chạy cho một người trên một máy (SPEC mục 2). Kết quả từ Kaggle về dạng file, ghép thẳng vào thư mục là xong.

**Ưu điểm.** Mang cả thư mục sang máy khác là dùng được; dễ gỡ lỗi; không có dịch vụ phải giữ cho chạy.

**Nhược điểm.**
- Không có khoá ghi giữa nhiều tiến trình; tool phải tự tránh bằng file khoá và thư mục `.partial`.
- Liệt kê hàng nghìn run phải quét thư mục.

### 2.7 Cache chỉ cho stage dùng chung, khoá đọc được

**Khái niệm.**
- Stage được đánh dấu `cacheable` (các stage `prep.*` và `init.*`) ghi kết quả vào `cache/<scene_id>/<stage_id>@<version>-<hash8>/` (CONTRACTS mục 5.2). Phần băm là 8 ký tự đầu của sha256 trên tham số đã chuẩn hoá cùng khoá của các artifact đầu vào; không băm nội dung dữ liệu.
- Các stage còn lại (`train.*`, `eval.*`, `render.*`, `export.*`) ghi vào `d2s/stages/<stage_id>/` trong thư mục run, có file `done` khi xong. Train chạy tiếp bằng checkpoint đầy đủ (model và trạng thái optimizer).

**Lý do.**
- Mask, SMPL, depth, điểm khởi tạo của một cảnh dùng chung được giữa mọi approach, và thường tốn thời gian sinh.
- Kết quả train không bao giờ dùng chung giữa các approach, nên cache cho train là thừa.
- Dữ liệu chuẩn sau ingest không đổi nữa; `scene_id` cùng phiên bản converter trong manifest là đủ để nhận diện cảnh. Băm vài GB dữ liệu mỗi lần chạy là tốn mà không thêm an toàn.

**Ưu điểm.**
- Dùng chung giữa các approach; chạy tiếp được khi phiên Kaggle bị ngắt.
- Tên thư mục đọc được, dễ gỡ lỗi và dễ xoá tay.

**Nhược điểm.**
- Sửa code của một stage cacheable mà quên tăng `version` thì cache cũ vẫn được dùng. Số stage phải để ý ít (khoảng 5–6), nhưng vẫn cần kỷ luật.
- Nếu dữ liệu chuẩn của một cảnh bị ingest lại với cùng `scene_id` và cùng phiên bản converter, cache cũ không tự mất hiệu lực; khi ingest lại phải đổi `scene_id` hoặc dọn cache của cảnh đó.
- `cache/` lớn dần; dùng `d2s cache gc`.

### 2.8 Train trên Kaggle qua một executor riêng

**Khái niệm.** Kaggle cho chạy notebook trên GPU miễn phí, giới hạn thời gian mỗi phiên và quota theo tuần. Kaggle API cho phép upload dữ liệu, đẩy notebook, hỏi trạng thái và tải kết quả.

**Lý do.** Máy hiện tại chỉ có Quadro T2000 4 GB, không đủ train. Teammate cũng đang train trên Kaggle T4.

**Ưu điểm.** Không tốn tiền GPU; cùng lệnh `d2s run start`, chỉ đổi `--executor`.

**Nhược điểm.**
- Phiên bị ngắt sau một thời gian, nên phải lưu checkpoint đầy đủ và chạy tiếp được. Hiện chạy tiếp đang hỏng (mục 7.2).
- Upload cảnh lớn lâu.
- Không xem viewer được trong lúc train.

### 2.9 Môi trường

**Khái niệm.** Một conda env hoặc venv chứa đúng phiên bản Python, PyTorch, CUDA và thư viện.

**Quyết định.**
- Mọi approach dòng DriveStudio dùng chung một môi trường của repo, với gsplat-geer. gsplat-geer có đủ ba renderer.
- Codebase ngoài (NuRec, feed-forward...) chạy trong môi trường riêng, gọi qua tiến trình con: tool chạy `d2s-worker` trong môi trường đó, đưa vào file job JSON, đọc lại file kết quả JSON, nhận tiến độ qua stdout (CONTRACTS mục 9).
- Tiến trình Studio không nạp model. Render cho viewer và server chạy trong một tiến trình worker riêng, giữ sống suốt phiên.

**Lý do.** Các codebase thường không cài chung được. Ví dụ đã kiểm: DriveStudio gốc cài `gsplat` 1.3.0 với `torch==2.0.0+cu117`; gsplat-geer cũng tên `gsplat` nhưng bản 1.5.3, và README của nó (`app/README.md`) ghi đã chạy với CUDA 12.1 và PyTorch 2.4.1. Hai bản cùng tên không nằm chung một môi trường được.

**Ưu điểm.** Codebase ngoài không làm hỏng môi trường chính; worker lỗi thì Studio vẫn chạy.

**Nhược điểm.**
- Mỗi môi trường tốn vài GB; máy hiện tại chỉ còn khoảng 11 GB trống.
- Phiên bản PyTorch thật sự của môi trường chính chưa chốt (mục 8).

### 2.10 API của Studio: FastAPI và Server-Sent Events, chỉ nghe trên `127.0.0.1`

**Khái niệm.** FastAPI là framework Python để viết HTTP API. Server-Sent Events (SSE) là cách server đẩy liên tục dữ liệu xuống trình duyệt qua một kết nối HTTP, một chiều; dùng cho tiến độ và log. `127.0.0.1` là địa chỉ chỉ máy đang chạy truy cập được.

**Lý do.** Một người dùng, một máy; không cần tài khoản. Trên máy GPU thuê, mở qua port forward của SSH.

**Ưu điểm.** SSE đơn giản hơn WebSocket cho việc đẩy tiến độ; không mở cổng ra ngoài nên không cần xác thực.

**Nhược điểm.** Lệnh từ người dùng phải đi qua REST; muốn chia sẻ từ xa phải tự port forward.

### 2.11 Giao diện Studio: Vite + React

**Khái niệm.** React dựng giao diện web từ các thành phần; Vite build mã React thành file HTML, JS, CSS tĩnh.

**Lý do.** Build sẵn thành file tĩnh và phục vụ bằng FastAPI, nên người dùng không cần Node.js. Client của viser cũng viết bằng React + Vite (mã nguồn trong wheel viser, thư mục `viser/client/`).

**Ưu điểm.** Có sẵn thư viện biểu đồ, bảng, trình phát video.

**Nhược điểm.** Phát triển cần Node.js; hai ngôn ngữ nên kiểu dữ liệu phải giữ khớp, ví dụ sinh kiểu TypeScript từ JSON Schema.

### 2.12 Viewer 3D chính: viser + nerfview, render ở server

**Khái niệm.**
- *viser* (MIT; paper arXiv 2507.22885) là thư viện Python dựng cảnh 3D xem trên trình duyệt. Mỗi lệnh như `scene.add_box()` hiện ngay; thao tác của người dùng gửi ngược về Python qua WebSocket.
- *nerfview* (MIT, bản 0.1.2) gọi hàm `render_fn(camera, kích thước)` mỗi khi camera đổi, rồi đưa ảnh (kèm depth nếu có) lên làm nền bằng `set_background_image`. Khi đang xoay, render độ phân giải thấp, JPEG chất lượng 40; khi dừng, render đầy đủ, chất lượng 70 (`_renderer.py`).
- `drive2sim/recon/tools/viewer.py` của repo đã dùng viser + nerfview: có thanh thời gian, phát lại, bật tắt từng loại Gaussian, đổi mô hình camera.

**Lý do.**
- Chỉ renderer của codebase mới vẽ đúng mọi thứ trong model: fisheye, GEER, spherical harmonics, bầu trời và chỉnh màu theo camera của OmniRe (`SkyModel`, `AffineTransform`), người SMPL, vật biến dạng.
- Không dùng `add_gaussian_splats` của viser: trong viser 1.0.30, hàm này còn ghi "Experimental", chỉ nhận màu RGB, 32 byte mỗi Gaussian, vẽ theo camera pinhole.
- Phụ thuộc bắt buộc của viser 1.0.30 không có thư viện GPL; `plyfile` chỉ nằm trong nhóm `dev`.

**Ưu điểm.**
- Ảnh hiện đúng như kết quả đánh giá.
- Trình duyệt chỉ nhận JPEG nên máy xem nhẹ.
- Có sẵn gizmo (`add_transform_controls`), sự kiện click; depth gửi kèm giúp overlay bị che khuất đúng chỗ.

**Nhược điểm.**
- Cần GPU CUDA và worker chạy suốt phiên xem.
- Có độ trễ mỗi lần xoay; lúc xoay ảnh mờ hơn.
- viser không có khái niệm thời gian (paper viser, mục 5); thanh thời gian phải tự làm hoặc lấy từ `drive2sim/recon/tools/viewer.py`.
- nerfview gọi `os._exit(1)` khi `render_fn` ném lỗi (`_renderer.py`), tắt cả tiến trình. `render_fn` của tool phải tự bắt mọi lỗi và trả về ảnh báo lỗi.
- Không dùng Rerun (đã thử trong project P-087, chạy lag), SuperSplat, PlayCanvas (vẽ splat trong trình duyệt, cùng giới hạn với chế độ xem trước).

### 2.13 Chế độ xem trước không cần GPU CUDA: Spark trong trình duyệt, file PLY

**Khái niệm.**
- *Spark* (MIT, World Labs) là thư viện JavaScript vẽ Gaussian splat trong trình duyệt, chạy trên three.js, dùng card đồ hoạ của máy xem qua WebGL2; card tích hợp thường cũng đủ. Theo README và mã nguồn (bản ngày 19/10/2026): spherical harmonics tới bậc 3 (`maxSh = 3`, `src/ExtSplats.ts`); đọc PLY, SPZ, SPLAT, KSPLAT, SOG; có level-of-detail; nhiều đối tượng splat di chuyển riêng được; không có fisheye.
- *PLY* là định dạng file của 3DGS gốc: 59–62 số thực 32 bit mỗi Gaussian, khoảng 240 byte, nên 3 triệu Gaussian là khoảng 720 MB. `drive2sim/recon/tools/export_ply.py` của repo xuất đúng định dạng này trong hệ world; export gốc của DriveStudio (`drive2sim/recon/utils/misc.py:50`) co toạ độ vào hộp đơn vị.
- *SPZ* (Niantic) lưu cùng thông tin nhưng làm tròn rồi nén ZSTD; README của SPZ ghi file thường nhỏ hơn PLY khoảng 10 lần. Vị trí là số nguyên 24 bit, mặc định 12 bit phần lẻ (`load-spz.cc`): độ phân giải khoảng 0,24 mm, phạm vi ±2048 m. SH bậc 1 còn 5 bit, bậc cao hơn còn 4 bit.

**Lý do.** Mở được kết quả trên máy không có GPU CUDA, ví dụ laptop của mentor.

**Ưu điểm.** Không cần worker; xoay mượt; giữ màu theo góc nhìn; xe đổi pose theo khung nên phát lại được quỹ đạo.

**Nhược điểm.**
- Chỉ gần đúng: chỉ pinhole; không có bầu trời và chỉnh màu theo camera; người và vật biến dạng chỉ hiện bằng box.
- PLY lớn. SPZ là bước tuỳ chọn, chưa bật; còn phải đo ảnh hưởng của việc làm tròn SH và tỉ lệ nén thật trên checkpoint DriveStudio.
- Cần một bước xuất gói xem trước (CONTRACTS mục 7).

### 2.14 Server mô phỏng theo proto `SensorsimService` của NuRec

**Khái niệm.** gRPC là cách hai chương trình gọi hàm của nhau qua mạng; proto (Protocol Buffers) mô tả hàm và kiểu dữ liệu. NuRec định nghĩa dịch vụ `SensorsimService` (Apache-2.0) với `render_camera`, `batch_render_camera`, `get_dynamic_objects`...

**Lý do.** CARLA và AlpaSim đã có client cho proto này; chỉ cần làm phía server. Proto có sẵn ba mô hình camera và pose đầu, cuối khung cho rolling shutter.

**Ưu điểm.** Dùng lại client có sẵn; cùng license Apache-2.0.

**Nhược điểm.** Giao thức do NVIDIA quyết định; hàm nào approach chưa làm được thì trả `UNIMPLEMENTED`.

### 2.15 Ba mô hình camera chuẩn

**Khái niệm.** `opencv_pinhole` (camera thường, méo OpenCV), `opencv_fisheye` (mô hình fisheye OpenCV, 4 hệ số k1–k4), `ftheta` (mô hình của NVIDIA, đa thức biến góc tia thành bán kính trên ảnh).

**Lý do.** Trùng với `CameraSpec` của proto và với ba mô hình tương ứng của NCore. Hàm chiếu của DriveStudio (`drive2sim/recon/utils/geometry.py`, `project_camera_model`) đã hỗ trợ `pinhole`, `fisheye`, `ftheta`.

**Ưu điểm.** Bao phủ cả hai track; dữ liệu chuẩn, renderer và server dùng cùng một bộ tham số.

**Nhược điểm.**
- Mô hình khác (omnidirectional, double sphere) phải chuyển gần đúng; kiểm tra khi biết dữ liệu của mentor.
- NCore còn có pinhole lý tưởng và mô hình méo kính chắn gió; đường PhysicalAI hiện từ chối mô hình kính chắn gió.

## 3. Cấu trúc repo đích

```
drivestudio_plus/                          branch drive2sim
├── pyproject.toml                         package drive2sim; pip install -e .
├── requirements.txt                       thư viện nặng (PyTorch, gsplat-geer, ...)
├── configs/                               config method (<method>*.yaml) và config dataset (datasets/<dataset>/)
│   └── datasets/ncore/        (mới)       <số camera>cams.yaml cho loader NCore
├── drive2sim/
│   ├── paths.py                           đường dẫn tới assets/ trong package
│   ├── assets/                            ego_masks/<dataset>/<camera>.png, scene_lists/<dataset>_*.txt
│   ├── recon/                             mã DriveStudio (đã chuyển vào)
│   │   ├── data/                          ← datasets/: base/, <dataset>/ (tiền xử lý + loader), tools/, driving_dataset.py
│   │   │   └── ncore/         (mới)       ncore_sourceloader.py: loader NCore chung
│   │   ├── models/                        ← models/: gaussians/, nodes/, modules.py, losses.py, human_body.py, video_utils.py
│   │   ├── trainers/                      ← models/trainers/
│   │   ├── utils/                         ← utils/
│   │   └── tools/                         ← tools/: train.py, eval.py, viewer.py, export_ply.py (tạm, thay dần bằng stage và CLI)
│   ├── core/                  (mới)       schema.py, manifest.py, split.py, transforms.py, camera.py, ncore_io.py
│   ├── ingest/                (mới)       <dataset>.py: một converter mỗi nguồn; validate.py
│   ├── stages/                (mới)
│   │   ├── prep/                          select.py, undistort.py, masks_segformer.py, smpl.py, to_drivestudio.py
│   │   ├── init/                          lidar.py, depth.py, slam.py
│   │   ├── train/                         drivestudio.py, external.py
│   │   ├── eval/                          standard.py
│   │   ├── render/                        videos.py
│   │   └── export/                        ply_world.py, preview.py
│   ├── engine/                (mới)       registry.py, resolve.py (ô chọn, preset, ràng buộc), run.py, cache.py
│   ├── approaches/            (mới)       <approach_id>.yaml
│   ├── kaggle/                (mới)       client.py, bundle.py, templates/<tên>.ipynb
│   ├── sim/                   (mới)       renderer.py (L2), actors.py (L3), server/ (L4; protos/ chép từ NuRec)
│   ├── studio/                (mới)
│   │   ├── api/                           app.py, routes/<nhóm>.py
│   │   ├── viz/                           viser_app.py, render_fn.py
│   │   └── frontend/                      Vite + React (src/pages/, src/components/)
│   ├── external/              (mới)       <codebase>/: môi trường và worker cho codebase ngoài
│   ├── scripts/               (mới)       công cụ phát triển chạy một lần, không phải tính năng
│   └── cli/                   (mới)       main.py, <nhóm>.py: một file cho mỗi nhóm lệnh (scene, split, run, ...)
├── third_party/                           smplx (cài bằng pip -e, import là smplx), Humans4D (submodule, import là hmr2); ncore (mới)
├── envs/                      (mới)       main.yml, <codebase>.yml
├── docs/                                  hướng dẫn dataset của DriveStudio; docs/drive2sim/ (SPEC, ARCHITECTURE, CONTRACTS)
└── tests/drive2sim/           (mới)       cùng cây thư mục với drive2sim/
```

Chạy các script cũ của DriveStudio bằng `python -m`, ví dụ `python -m drive2sim.recon.tools.train`, `python -m drive2sim.recon.data.preprocess`. Chuỗi import trong config cũng theo đường dẫn mới, ví dụ `drive2sim.recon.models.nodes.RigidNodes`.

### 3.1 Code mới đặt ở đâu

| Cần thêm | Đặt ở | Tên file, tên lớp | Đăng ký hoặc khai báo ở |
|---|---|---|---|
| Nguồn dữ liệu mới (dataset, CARLA, dữ liệu mentor) | `drive2sim/ingest/` | `<dataset>.py`, lớp `<Dataset>Converter` (theo `ncore.data_converter`) | Giá trị mới cho `--source-type` trong `drive2sim/cli/scene.py` |
| Stage mới | `drive2sim/stages/<nhóm>/` | `<tên>.py`, lớp `<Nhóm><Tên>Stage`, `stage_id` là `<nhóm>.<tên>` | `@register_stage("<nhóm>.<tên>")` trong chính file đó |
| Loại artifact mới | `drive2sim/core/schema.py` | giá trị mới của `artifact_type` | Bảng CONTRACTS mục 5.4 |
| Approach mới | `drive2sim/approaches/` | `<approach_id>.yaml` | Tự nạp khi engine khởi động |
| Lựa chọn mới cho một ô (ví dụ `init.slam`) | Stage mới như trên | | Thêm vào `options.<ô>.choices` của approach |
| Method, node, loại Gaussian mới | `drive2sim/recon/models/nodes/` hoặc `drive2sim/recon/models/gaussians/` | `<tên>.py`; lớp theo quy ước sẵn có (`<Tên>Nodes`, `<Tên>Gaussians`) | Config method trong `configs/`; giá trị mới cho tham số `method` của `train.drivestudio` |
| Loader dữ liệu | `drive2sim/recon/data/ncore/` | Chỉ một loader NCore; không thêm loader riêng cho dataset mới, dataset mới đi qua `ingest/` | |
| Tiện ích dùng chung của phần tái dựng | `drive2sim/recon/utils/` | `<chủ_đề>.py` | |
| Config method | `configs/` | `<method>.yaml`; biến thể kế thừa từ config gốc, chỉ ghi phần khác | |
| Config dataset cho loader NCore | `configs/datasets/ncore/` | `<số camera>cams.yaml` | |
| File đi kèm package (mask thân xe, danh sách cảnh) | `drive2sim/assets/<loại>/` | | Truy cập qua `drive2sim/paths.py`, không dùng đường dẫn tương đối theo thư mục đang chạy |
| Codebase ngoài | `drive2sim/external/<codebase>/` | `worker.py`; môi trường ở `envs/<codebase>.yml` | Stage `train.external` với tham số `codebase: <tên>` |
| Lệnh CLI mới | `drive2sim/cli/<nhóm>.py` | hàm `<hành_động>` | Theo CONTRACTS mục 10.4 |
| API cho Studio | `drive2sim/studio/api/routes/<nhóm>.py` | đường dẫn `/api/<nhóm>/...` | `app.py` |
| Màn hình Studio | `drive2sim/studio/frontend/src/pages/` | `PascalCase.tsx` | Router của frontend |
| Mẫu notebook Kaggle | `drive2sim/kaggle/templates/` | `<tên>.ipynb` | `bundle.py` |
| Test | `tests/drive2sim/<cùng đường dẫn>/` | `test_<module>.py` | |
| Fixture dữ liệu nhỏ cho test | `tests/drive2sim/fixtures/<scene_id>/` | | |
| Công cụ phát triển chạy một lần | `drive2sim/scripts/` | `<động_từ>_<đối_tượng>.py` | Không dùng trong luồng chính; việc dùng thường xuyên phải thành lệnh CLI |
| Tài liệu | `docs/drive2sim/` | `UPPER_CASE.md` cho tài liệu chính | Cập nhật cả ba tài liệu chính khi đổi thiết kế |

Quy tắc chung:
- Mọi mã nằm trong package `drive2sim/`; không thêm thư mục mã mới ở gốc repo.
- Import luôn theo đường dẫn đầy đủ (`from drive2sim.recon.models.nodes import RigidNodes`), không import tương đối giữa các nhóm lớn.
- Một file một việc. Stage, converter, nhóm lệnh CLI, nhóm route API: mỗi cái một file.
- Tên file, tên định danh và tham số theo CONTRACTS mục 10.

## 4. Các khối

### 4.1 core

Schema Pydantic v2 cho manifest, split, artifact, approach (stage, ô chọn, ràng buộc, preset), run, metric (CONTRACTS); phép biến đổi toạ độ theo quy ước NCore; đọc dữ liệu chuẩn qua `nvidia-ncore`. Không nạp model.

### 4.2 ingest

Mỗi nguồn một converter, ghi ra một kho NCore V4 trong `data/<scene_id>/`, kèm `manifest.yaml` và báo cáo kiểm tra.

| Nguồn | Converter |
|---|---|
| Waymo, nuScenes, Argoverse 2, KITTI, PhysicalAI, COLMAP | Converter có sẵn của NCore (`third_party/ncore/tools/data_converter/`) |
| nuPlan, PandaSet | Tự viết. Một phương án đáng thử: chạy script tiền xử lý có sẵn của DriveStudio, rồi viết một converter từ thư mục DriveStudio sang NCore; một converter này dùng được cho mọi dataset DriveStudio đọc được |
| CARLA | Tự viết, từ script sinh rig fisheye |
| Dữ liệu mentor | Tự viết khi biết định dạng; nếu chỉ là ảnh kèm pose thì thử đường COLMAP |

Bộ kiểm tra: đồng bộ thời gian giữa camera, calib, mô hình ống kính, FOV, ảnh hỏng hoặc thiếu (CONTRACTS mục 3.2).

### 4.3 stages

Mỗi stage là một lớp đăng ký theo tên, theo interface ở CONTRACTS mục 5.1; nơi ghi kết quả ở CONTRACTS mục 5.2.

| Stage | Việc | Cache | Mã có sẵn để dựa vào |
|---|---|---|---|
| `prep.select` | Chọn camera, đoạn khung, độ phân giải | có | |
| `prep.undistort` | Khử méo về pinhole cho approach chỉ nhận pinhole | có | `data.pixel_source.undistort` của loader PhysicalAI |
| `prep.masks_segformer` | Mask bầu trời và mask vật động | có | `drive2sim/recon/data/tools/extract_masks_hf.py` |
| `prep.smpl` | Tư thế người | có | `drive2sim/recon/data/tools/humanpose_process.py`, `extract_smpl.py` |
| `prep.to_drivestudio` | NCore → thư mục DriveStudio cho loader NCore | có | `drive2sim/recon/data/physicalai/physicalai_preprocess.py` |
| `init.lidar` | Điểm khởi tạo từ LiDAR | có | `get_lidar_samples` trong `drive2sim/recon/data/driving_dataset.py` |
| `init.depth` | Điểm khởi tạo từ pose dataset + depth metric | có | Chưa có |
| `init.slam` | Pose và điểm từ SLAM, khi rig không có pose | có | Chưa có |
| `train.drivestudio` | Train bằng trainer DriveStudio; method, renderer, FastGS, CamPose chọn qua tham số | không | `drive2sim/recon/tools/train.py` |
| `train.external` | Train bằng codebase ngoài, trong môi trường riêng | không | Chưa có |
| `eval.standard` | Đánh giá theo protocol chung (SPEC mục 9) | không | `drive2sim/recon/tools/eval.py` (`do_evaluation`) |
| `render.videos` | Video ảnh gốc, ảnh render, sai khác theo camera | không | `drive2sim/recon/models/video_utils.py` |
| `export.ply_world` | PLY 3DGS trong hệ world | không | `drive2sim/recon/tools/export_ply.py` |
| `export.preview` | Gói xem trước cho Spark | không | Mở rộng `drive2sim/recon/tools/export_ply.py`; dữ liệu cần (`instances_quats`, `instances_trans`, `instances_fv`, `points_ids`) có sẵn trong `state_dict()` của RigidNodes (`drive2sim/recon/models/nodes/rigid.py:483`) |

Trong `train.drivestudio`, tổ hợp scene graph, renderer và luật densify vẫn chọn bằng config của DriveStudio. Chỉ tách thêm thành stage riêng khi config không làm được.

### 4.4 Tầng tái dựng (DriveStudio)

Stage `train.drivestudio` train bằng trainer trong `drive2sim/recon/trainers/` với config method và config dataset `configs/datasets/ncore/`. Sau refactor: trainer nhận `points_init` từ stage `init.*`; đánh giá do `eval.standard` gọi, không còn chạy ở cuối `drive2sim/recon/tools/train.py`.

### 4.5 engine và approaches

Engine chạy một approach theo các bước:
1. Đọc file approach; thay preset và lựa chọn của người dùng vào các ô; giá trị `auto` được chọn theo manifest.
2. Kiểm tra điều kiện dữ liệu (manifest, split) và `constraints`; sai thì dừng, kèm lý do.
3. Tạo thư mục run, ghi `d2s/run.json` với lựa chọn đã quyết định.
4. Chạy lần lượt từng stage: stage cacheable có sẵn trong cache thì dùng lại; stage trong run đã có `done` thì bỏ qua.
5. Ghi trạng thái từng stage vào `run.json` sau mỗi bước.

Studio và CLI dùng chung bước 1–2 (`engine/resolve.py`), nên chạy từ đâu cũng cho cùng một danh sách stage. `d2s approach check` in ra danh sách đó mà không chạy.

Approach dự kiến:

| Approach | Ô chọn | Config DriveStudio dùng làm gốc |
|---|---|---|
| `omnire` | `init`: lidar / depth / auto; `renderer`: default / ut / geer; `campose`; `fastgs` | `omnire*.yaml` |
| `streetgs` | như trên | `streetgs*.yaml` |

Các config hiện có của teammate map sang tổ hợp ô:

| Config hiện có | Tổ hợp ô tương ứng |
|---|---|
| `streetgs_nocampose.yaml` / `omnire_nocampose.yaml` | renderer default, campose tắt, fastgs tắt |
| `streetgs_ut.yaml` | renderer ut, campose tắt |
| `streetgs_geer.yaml` / `omnire_geer.yaml` | renderer geer, campose tắt |
| `streetgs_fastgs.yaml` / `omnire_fastgs.yaml` | renderer default, campose tắt, fastgs bật |
| `streetgs_geer_fastgs.yaml` | renderer geer, campose tắt, fastgs bật |

Ràng buộc đã biết: renderer `geer` không đi với campose bật, vì 3DGEER không tối ưu được pose camera. Các config 3DGEER của teammate còn đổi `densify_grad_thresh: 0.01` và `refine_interval: 300` (README của fork); hai giá trị này đi theo lựa chọn renderer `geer`.

### 4.6 kaggle

Đóng gói job, upload cảnh thành dataset private (một lần, nhận diện theo `scene_id` và phiên bản converter), sinh notebook từ `kaggle/templates/`, chỉ cài repo và chạy `d2s run start --bundle`, theo dõi qua Kaggle API, tải output về và ghép vào `runs/` và `cache/`.

### 4.7 export

Xem các stage `export.*` ở mục 4.3.

### 4.8 sim

- `renderer.py` (L2): nạp checkpoint, render ở `CameraSpec` và pose bất kỳ, kèm trạng thái actor. Tách từ `drive2sim/recon/tools/viewer.py` và trainer.
- `actors.py` (L3): liệt kê, ẩn, xoá xe và sửa quỹ đạo bằng cách thay `instances_trans`, `instances_quats`, `instances_fv` của RigidNodes.
- `server/` (L4): cài `SensorsimService` theo proto của NuRec (giữ nguyên ghi chú license), ánh xạ `render_camera` và `batch_render_camera` sang `renderer.py`, `dynamic_objects` sang `actors.py`; hàm chưa hỗ trợ trả `UNIMPLEMENTED`.

### 4.9 studio

- `api/`: FastAPI phục vụ REST cho cảnh, approach, run, metric; SSE cho tiến độ và log; file tĩnh của frontend, video, ảnh, gói xem trước. Chỉ nghe trên `127.0.0.1`.
- `frontend/`: Vite + React theo luồng sử dụng ở SPEC mục 5; màn hình chọn approach có preset, ô chọn, lựa chọn bị khoá kèm lý do, và danh sách stage sẽ chạy (stage nào dùng lại cache); video là thẻ `<video>` MP4 H.264, phát đồng bộ nhiều camera; nút của mức approach không đạt thì ẩn.
- `viz/`: viser chạy trong tiến trình Studio, nhúng bằng iframe; ảnh và depth lấy từ worker render (mục 2.9). Hai chế độ:

| | Chế độ chính | Chế độ xem trước |
|---|---|---|
| Cần GPU CUDA | Có | Không |
| Cách render | Worker render bằng renderer của codebase; Studio đẩy ảnh lên viser bằng `set_background_image(ảnh, depth=...)` | Spark vẽ trong trình duyệt từ gói xem trước |
| Đúng với model | Hoàn toàn | Gần đúng (mục 2.13) |

### 4.10 cli

Lệnh dạng `d2s <nhóm> <hành động>`. Danh sách đầy đủ và quy tắc tuỳ chọn ở CONTRACTS mục 10.4.

| Nhóm | Hành động |
|---|---|
| `scene` | `ingest`, `validate`, `list`, `show` |
| `split` | `create` |
| `approach` | `list`, `show`, `check` |
| `run` | `start`, `resume`, `status`, `list`, `cancel` |
| `kaggle` | `push`, `status`, `pull` |
| `export` | `ply`, `preview` |
| `cache` | `list`, `gc` |
| `report`, `studio`, `serve` | (lệnh đơn) |

## 5. Luồng dữ liệu

```
nguồn ──d2s scene ingest──► data/<scene_id>/ (NCore V4 + manifest.yaml + reports/validation.json)
                              │
                         d2s split create ──► data/<scene_id>/splits/<split_id>.json
                              │
<approach_id>.yaml + preset + ô chọn ──engine──► danh sách stage cố định
                              │
        prep.* · init.* ──► cache/<scene_id>/<stage_id>@<version>-<hash8>/  (dùng chung)
                              │
        train.* · eval.standard · render.* · export.* ──► runs/<scene_id>/<approach_id>/<run_id>/
                                                         d2s/run.json · log DriveStudio · checkpoint · video
                              │
                 Studio (đọc) · server gRPC (L4)
```

## 6. Mức năng lực và interface

| Mức | Interface (CONTRACTS mục 8) | Ai dùng |
|---|---|---|
| L1 | Run ghi đúng `run.json`, metric và ảnh render của tập test | Studio nhóm A–D, `d2s report` |
| L2 | `Renderer.render(camera, pose_start, pose_end, timestamp_us, actors)` | viewer, render quỹ đạo mới |
| L3 | `ActorEditor`: liệt kê, lấy và đặt quỹ đạo, ẩn, xoá | viewer (chỉnh actor), server |
| L4 | Render đúng actor theo từng yêu cầu, không giữ trạng thái giữa các yêu cầu | server, CARLA |

## 7. Hiện trạng và lộ trình refactor

### 7.1 Đã có sẵn trong repo

Loader 7 dataset; tiền xử lý (mask, SMPL); OmniRe, Street GS, PVG, Deformable-GS; renderer 3DGS / 3DGUT / 3DGEER; rolling shutter ở `ut`/`geer`; FastGS; `drive2sim/recon/tools/eval.py`; video; `drive2sim/recon/tools/viewer.py`; `drive2sim/recon/tools/export_ply.py`; đọc NCore V4 cho PhysicalAI.

### 7.2 Chênh lệch với thiết kế đích (đã kiểm trong code)

| Chênh lệch | Vị trí | Việc cần làm |
|---|---|---|
| Mỗi dataset một loader riêng | `drive2sim/recon/data/*/*_sourceloader.py` | Loader NCore chung (mục 2.4) |
| Đường NCore chỉ dành cho PhysicalAI: gắn cứng 7 camera, tìm file `pai_*.json`, chỉ nhận F-theta, bắt buộc có LiDAR | `drive2sim/recon/data/physicalai/physicalai_preprocess.py` dòng 16, 36, 82–83, 411 | Đọc camera từ dữ liệu; thêm pinhole và fisheye OpenCV; cho phép không có LiDAR |
| Khởi tạo nằm trong dataset và trainer, bắt buộc có LiDAR | `drive2sim/recon/data/driving_dataset.py:189` (`get_lidar_samples`) | Tách thành stage `init.*`; trainer nhận `points_init` |
| Đánh giá chạy ở cuối `drive2sim/recon/tools/train.py` | `do_evaluation` trong `drive2sim/recon/tools/train.py`, `drive2sim/recon/tools/eval.py` | Tách thành stage `eval.standard` |
| Chỉ một kiểu split: test là các khung N, 2N, 3N... với mọi camera; 19 trên 21 config dataset đặt `test_image_stride: 0`, tức không có khung test | `split_train_test` trong `drive2sim/recon/data/driving_dataset.py` | Trước mắt chỉ dùng kiểu `every_nth_frame` (CONTRACTS mục 3.1); sau mới nhận split tuỳ ý |
| Chạy tiếp hỏng: khi có `--resume_from`, `do_save` luôn sai nên không lưu checkpoint; checkpoint chỉ lưu model, không có trạng thái optimizer | `drive2sim/recon/tools/train.py:295-301` | Lưu checkpoint đầy đủ, cho phép lưu khi chạy tiếp |
| Config chép nguyên file: `streetgs_nocampose.yaml` khác `streetgs.yaml` 9 dòng trên 148; các bản `_ut`, `_geer`, `_fastgs` khác 10–19 dòng | `configs/*.yaml` | Một config gốc cho mỗi method; biến thể chỉ ghi phần khác, ứng với các ô chọn |
| Render nằm lẫn trong viewer và trainer | `drive2sim/recon/tools/viewer.py` (2523 dòng), `drive2sim/recon/trainers/base.py` (1125 dòng) | Tách hàm render dùng riêng cho `sim/` |
| Export không có sky, SMPL, vật biến dạng; SH của xe không xoay theo xe | `drive2sim/recon/tools/export_ply.py` | Ghi rõ giới hạn; mở rộng cho `export.preview` |
| Không có test | | `tests/drive2sim/`, bắt đầu bằng kiểm tra hồi quy của loader |
| Phiên bản PyTorch chưa khớp: `requirements.txt` ghi `torch==2.0.0+cu117`, gsplat-geer ghi đã chạy với 2.4.1 | `requirements.txt` | Hỏi teammate; chốt một môi trường |

### 7.3 Nguyên tắc khi refactor

- Làm trên branch `drive2sim`; `main` cũ của teammate không bị ảnh hưởng. Thay đổi teammate làm trên `main` sau thời điểm tách được chuyển tay sang branch này.
- Mỗi bước refactor một commit riêng: bước di chuyển file chỉ đổi chỗ và sửa import, không đổi hành vi; các bước sau mới đổi hành vi.
- Sau bước di chuyển và sau mỗi bước đổi hành vi ở phần tái dựng: train lại cùng cảnh trên Kaggle, so PSNR (mục 10).
- Loader cũ và config cũ chỉ bỏ sau khi qua kiểm tra hồi quy.

### 7.4 Thứ tự làm

0. **Di chuyển mã DriveStudio vào `drive2sim/recon/`** (đã làm): đổi chỗ file, sửa import trong mã và chuỗi import trong config, thêm `pyproject.toml`, `drive2sim/paths.py` cho `assets/`. Đã kiểm tra tĩnh (mọi import `drive2sim...` và mọi chuỗi import trong config trỏ tới lớp có thật); **chưa chạy thật**.
1. **Kaggle chạy tiếp được:** sửa `drive2sim/recon/tools/train.py` (mục 7.2) và làm `drive2sim/kaggle/`. Chạy được OmniRe trên cảnh nuScenes mini mà teammate đã dùng; đây cũng là lần kiểm tra hành vi của bước 0.
2. **Engine và stage bọc mã hiện có:** registry, file approach có ô chọn, `train.drivestudio`, `eval.standard`, `render.videos`, `export.ply_world`, `run.json`, CLI `d2s`. Đạt L1 theo thiết kế; Studio nhóm A–D làm song song.
3. **Tầng dữ liệu:** nuScenes mini → NCore bằng converter có sẵn (thử chạy không cần Bazel, trên Kaggle) → `prep.to_drivestudio` → loader NCore → kiểm tra hồi quy với loader cũ.
4. **Tách khởi tạo:** `init.lidar` tách khỏi dataset, rồi `init.depth`. Thí nghiệm có LiDAR so với không LiDAR là đổi ô `init`.
5. **Config kế thừa** thay cho config chép file.
6. **sim/renderer.py** → L2 và viewer chế độ chính.
7. **sim/actors.py** → L3.
8. **sim/server/** + CARLA → L4.
9. Converter còn thiếu theo thứ tự cần dùng: CARLA, dữ liệu mentor, nuPlan, PandaSet.

## 8. Câu hỏi mở

1. Phiên bản PyTorch của môi trường chính; teammate chạy trên Kaggle bằng bản nào.
2. Chế độ `default` của gsplat-geer có cho kết quả giống DriveStudio gốc chạy gsplat 1.3.0 không. Nếu cần baseline đúng DriveStudio gốc thì thêm một môi trường riêng.
3. Converter của NCore chạy được không cần Bazel không.
4. Cách đóng gói môi trường: conda, venv hay container.
5. AD stack và tiêu chí chấm điểm ở mức L4 (SPEC mục 14).
6. Định dạng dữ liệu của mentor.

## 9. License

- Repo: MIT (DriveStudio) cộng AGPL-3.0 (patch Bosch, FastGS dựa trên patch), nên phần thêm vào cũng phân phối theo AGPL.
- gsplat-geer đi kèm `3dgs-license.txt` của Inria (phi thương mại).
- `drive2sim/recon/tools/export_ply.py` dùng `plyfile` (GPLv3+).
- Weights SegFormer của NVIDIA dùng trong `extract_masks_hf.py` chỉ cho dùng phi thương mại.
- NCore, proto `SensorsimService`: Apache-2.0; giữ nguyên ghi chú bản quyền khi đưa vào repo.
- viser, nerfview, Spark: MIT.
- Dữ liệu (Waymo, nuScenes...) không nằm trong repo; dataset upload lên Kaggle luôn để private.

## 10. Kiểm thử

- Kiểm tra hành vi sau khi di chuyển mã (mục 7.4 bước 0): cùng cảnh, cùng config, cùng seed, so PSNR test với kết quả trước khi di chuyển.
- Kiểm tra hồi quy của tầng dữ liệu: cùng cảnh, loader cũ và loader NCore, so PSNR test (SPEC mục 13).
- Kiểm tra hồi quy khi tách `init.lidar` và `eval.standard`: cùng cảnh, cùng seed, so với `drive2sim/recon/tools/train.py` gốc.
- Cảnh mẫu nhỏ (vài khung, 2 camera) làm fixture cho `core`, `ingest`, từng stage, `engine`.
- Kiểm tra schema cho mọi file trong `drive2sim/approaches/`, kể cả `constraints` và `presets`.
- Kiểm tra chạy tiếp: ngắt một run giữa chừng rồi chạy lại, so checkpoint và các file `done`.

