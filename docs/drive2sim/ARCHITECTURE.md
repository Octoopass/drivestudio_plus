# drive2sim: kiến trúc

Tài liệu này mô tả cách xây tool theo `SPEC.md`, ngay trên repo này. Định dạng dữ liệu và interface nằm trong `CONTRACTS.md`.

Đây là **thiết kế đích**. Mục 7 so sánh với hiện trạng của repo và đưa ra lộ trình refactor. Những gì đã kiểm tra trong mã nguồn hoặc tài liệu gốc thì ghi rõ nguồn; những gì chưa kiểm tra thì ghi là chưa kiểm tra.

Trạng thái: bản nháp, ngày 12/11/2026.

## 1. Tổng quan

Repo gồm ba tầng:

```
┌──────────────────────────── TẦNG DỮ LIỆU ─────────────────────────────────────────────┐
│ nguồn bất kỳ ──ingest──► DỮ LIỆU CHUẨN (NCore V4) + manifest + báo cáo kiểm tra       │
│ (7 dataset, CARLA,        (một lần mỗi cảnh, không phụ thuộc approach)                │
│  mentor, COLMAP)                      │                                               │
│                                       ▼                                               │
│                 CHUẨN BỊ RIÊNG cho approach: lấy bớt · sinh thêm · đổi dạng (cache)   │
└───────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
┌──────────────────────────── TẦNG TÁI DỰNG ────────────────────────────────────────────┐
│ DriveStudio (datasets/ · models/ · tools/): một loader NCore chung → trainer           │
│   OmniRe, Street GS, PVG... · renderer 3DGS / 3DGUT / 3DGEER · FastGS                 │
│ Codebase khác (sau này): bọc trong drive2sim/external/, môi trường riêng              │
└───────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
┌──────────────────────────── TẦNG TOOL (drive2sim/) ───────────────────────────────────┐
│ approaches/ · runner + cli (d2s) · kaggle/ · sim/ (L2 renderer, L3 actor, L4 gRPC)     │
│ studio/ (FastAPI + React + viser) · export/                                           │
└───────────────────────────────────────────────────────────────────────────────────────┘
      thư mục làm việc: data/ (dữ liệu chuẩn) · cache/ (dữ liệu sinh thêm) · runs/ (kết quả)
```

- Tầng dữ liệu và tầng tool là phần thêm mới, nằm chủ yếu trong thư mục `drive2sim/`.
- Tầng tái dựng là mã DriveStudio có sẵn của repo. Thay đổi ở tầng này giữ ở mức tối thiểu (mục 7.3).

## 2. Quyết định thiết kế

Mỗi quyết định gồm: giải thích công cụ hoặc khái niệm, lý do, ưu điểm, nhược điểm.

| # | Quyết định |
|---|---|
| 2.1 | Xây tool ngay trong repo `drivestudio_plus`, trên một branch riêng |
| 2.2 | Dùng NCore V4 làm định dạng dữ liệu chuẩn |
| 2.3 | Tách hai tầng chuẩn bị dữ liệu: ingest chung và chuẩn bị riêng cho approach |
| 2.4 | Một loader NCore chung thay cho loader riêng của từng dataset |
| 2.5 | Approach là một file khai báo nhỏ, trỏ tới config có sẵn |
| 2.6 | Lưu trạng thái bằng thư mục, không dùng cơ sở dữ liệu |
| 2.7 | Cache dữ liệu sinh thêm theo mã băm nội dung |
| 2.8 | Train trên Kaggle qua một executor riêng |
| 2.9 | Môi trường: một môi trường cho repo, môi trường riêng cho codebase ngoài |
| 2.10 | API của Studio: FastAPI và Server-Sent Events, chỉ nghe trên `127.0.0.1` |
| 2.11 | Giao diện Studio: Vite + React |
| 2.12 | Viewer 3D chính: viser + nerfview, render ở server |
| 2.13 | Chế độ xem trước không cần GPU CUDA: Spark trong trình duyệt, file PLY |
| 2.14 | Server mô phỏng theo proto `SensorsimService` của NuRec |
| 2.15 | Ba mô hình camera chuẩn: `opencv_pinhole`, `opencv_fisheye`, `ftheta` |

### 2.1 Xây tool ngay trong repo `drivestudio_plus`

**Khái niệm.** `drivestudio_plus` ([Octoopass/drivestudio_plus](https://github.com/Octoopass/drivestudio_plus)) là fork DriveStudio của một thành viên trong nhóm. Repo gồm:
- DriveStudio ở commit `e59bda4` (MIT): OmniRe, Street Gaussians, PVG, Deformable-GS, loader của 6 dataset, script tiền xử lý, `tools/eval.py`.
- Patch 3DGEER/3DGUT của Bosch (AGPL-3.0), áp nguyên vẹn từ nhánh `gsplat-geer`, commit `f15cee2`. Patch này thêm: renderer chọn bằng `trainer.render.render_mode: default | ut | geer`, camera fisheye/F-theta gốc, rolling shutter, loader PhysicalAI (đọc NCore V4) và `tools/viewer.py` (viser 1.0.30 + nerfview 0.1.2).
- Phần teammate thêm: densify và prune kiểu FastGS (`configs/fastgs.md`), `tools/export_ply.py`, `datasets/tools/extract_masks_hf.py`, các config `*_nocampose`.

Tool được làm trên branch `drive2sim` của chính repo này.

**Lý do.**
- Repo đã làm được phần lớn mức L1–L2: nhiều method, nhiều dataset, ba renderer, đánh giá, video, viewer. Làm một repo riêng bọc bên ngoài thì phải viết lại lớp dữ liệu và lớp điều phối mà DriveStudio đã có.
- Phần cốt lõi của L3 (chỉnh actor) và L4 (gRPC) phải chạm vào bên trong trainer và node của DriveStudio. Làm ngay trong repo thì gọi trực tiếp được.

**Ưu điểm.**
- Ít việc nhất; dùng ngay approach của teammate.
- Một codebase chung, merge qua lại bằng git.

**Nhược điểm.**
- Nhánh `main` có code AGPL-3.0, nên phần thêm vào cũng phân phối theo AGPL. Nếu cần license dễ dãi hơn thì phải dựa trên nhánh `phase3` (chỉ có phần MIT).
- Dễ xung đột với teammate khi cùng sửa `datasets/` hay `models/`; giảm bằng cách đặt phần mới trong `drive2sim/` và làm thay đổi nhỏ ở lõi.

### 2.2 NCore V4 làm định dạng dữ liệu chuẩn

**Khái niệm.** NCore là định dạng dữ liệu cảm biến cho xe tự lái của NVIDIA (Apache-2.0); NuRec là một trong những chương trình đọc nó. Theo `docs/data/` của repo NCore:
- Một bản ghi gồm nhiều *component group*, mỗi group là một kho zarr, lưu dạng thư mục hoặc một file `.itar`.
- Component có sẵn: pose, intrinsics, mask tĩnh, ảnh camera, LiDAR, radar, box 3D (cuboid), point cloud, và nhãn theo ảnh (depth, segmentation, mask, normal, feature).
- Mô hình camera: F-theta, pinhole lý tưởng, pinhole OpenCV, fisheye OpenCV; có thêm mô hình méo do kính chắn gió.
- Thời gian tính bằng micro giây (`uint64`); mỗi ảnh có thời điểm bắt đầu và kết thúc, dùng cho rolling shutter.
- Hệ toạ độ: `rig` x tới trước, y sang trái, z lên trên, gốc ở giữa trục sau trên mặt đất; `world` đặt pose đầu tiên của xe làm gốc; camera theo OpenCV.

Đọc và ghi bằng gói `nvidia-ncore` (PyPI, bản 19.9.0, Python ≥ 3.8). Repo NCore có converter cho Waymo, nuScenes, Argoverse 2, KITTI, PhysicalAI và COLMAP (`tools/data_converter/`).

**Lý do.**
- Cần một định dạng chung cho mọi nguồn, đúng tinh thần NCore của NuRec. NCore có sẵn converter cho phần lớn nguồn cần dùng, nên ít phải tự viết nhất.
- Có đủ ba mô hình camera và rolling shutter, khớp `CameraSpec` của proto `SensorsimService` (mục 2.14).
- NuRec đọc thẳng NCore, nên muốn so với NuRec thì không phải chuyển dữ liệu thêm lần nữa.
- Patch Bosch đã có một đường đọc NCore (`datasets/physicalai/`), làm nền cho loader chung (mục 2.4).

**Ưu điểm.**
- Không phải tự thiết kế và giữ một định dạng riêng.
- Nhãn theo ảnh (mask, depth) lưu được ngay trong NCore.

**Nhược điểm.**
- File zarr/itar không xem trực tiếp bằng mắt được; cần công cụ đọc.
- Hướng dẫn chính thức chạy converter bằng Bazel. Code converter là Python thường, import `ncore` từ PyPI cùng vài thư viện (`click`, `pycolmap` cho COLMAP), nên có vẻ chạy được không cần Bazel; chưa thử.
- Không có converter cho nuPlan, PandaSet, CARLA và dữ liệu mentor; phải tự viết theo khung `ncore.data_converter`.
- Quy ước `T_a_b` của NCore đưa điểm từ hệ `a` sang hệ `b`, ngược với nhiều codebase khác. Mọi chỗ chuyển đổi phải theo một hàm chung (CONTRACTS mục 1).

### 2.3 Hai tầng chuẩn bị dữ liệu

**Khái niệm.**
- *Ingest*: chuyển mọi thứ có trong nguồn sang dữ liệu chuẩn, không bỏ bớt; ghi manifest và báo cáo kiểm tra. Không phụ thuộc approach; mỗi cảnh chạy một lần.
- *Chuẩn bị riêng*: sau khi chọn approach, lấy bớt (camera, đoạn khung, độ phân giải, khử méo), sinh thêm (mask, SMPL, depth prior, box) và đổi dạng sang thứ codebase đọc được.

**Lý do.**
- Mọi approach bắt đầu từ cùng một dữ liệu, nên so sánh công bằng.
- Thử approach thứ hai trên cùng cảnh không phải ingest lại.
- Manifest có sẵn sau ingest thì mới khoá đúng approach không chạy được.

**Ưu điểm.** Dữ liệu sinh thêm được cache và dùng chung (mục 2.7).

**Nhược điểm.** Dữ liệu chuẩn giữ đủ mọi thứ nên tốn ổ đĩa hơn bản đã lọc cho một approach.

### 2.4 Một loader NCore chung thay cho loader riêng của từng dataset

**Khái niệm.** DriveStudio hiện có 7 loader riêng (`datasets/*/*_sourceloader.py`, 460–720 dòng mỗi file, tổng 3678 dòng). Thư mục sau tiền xử lý có cùng khung, nhưng mỗi loader tự xử lý quy ước calib, trục camera và box của dataset mình. Phần thật sự chung là interface Python (`ScenePixelSource`, `SceneLidarSource`). Loader được chọn trong config dataset bằng `data.pixel_source.type`, ví dụ `datasets.physicalai.physicalai_sourceloader.PhysicalAIPixelSource`.

Thiết kế đích: một đường duy nhất `NCore → thư mục DriveStudio → loader NCore`, mở rộng từ đường PhysicalAI có sẵn.

**Lý do.** Nguồn mới chỉ cần converter sang NCore, không phải viết thêm loader 500 dòng.

**Ưu điểm.** Mọi approach dòng DriveStudio đọc dữ liệu qua một chỗ; sửa lỗi ở một chỗ.

**Nhược điểm.**
- Phải sửa phần dữ liệu của teammate; cần báo trước.
- Phải chứng minh không tụt chất lượng: train cùng cảnh bằng loader cũ và loader mới, so PSNR (SPEC mục 13). Loader cũ chỉ được bỏ sau khi qua bước này.
- Đường PhysicalAI hiện gắn cứng nhiều chỗ (mục 7.2).

### 2.5 Approach là một file khai báo nhỏ

**Khái niệm.** Mỗi approach là `drive2sim/approaches/<id>.yaml`. File ghi: codebase, config method có sẵn trong `configs/` và tham số ghi đè, track, mức năng lực, điều kiện dữ liệu, các bước chuẩn bị riêng, preset (CONTRACTS mục 4).

**Lý do.** DriveStudio đã có config method và config dataset ghép bằng OmegaConf, cùng thứ tự chạy cố định (tiền xử lý → `tools/train.py`, tự đánh giá ở cuối → export). Không cần một engine pipeline tổng quát; chỉ cần ghi thêm những gì DriveStudio không có: mức năng lực, điều kiện dữ liệu, bước chuẩn bị.

**Ưu điểm.**
- Không sửa config của teammate.
- Thêm approach = thêm một file nhỏ.

**Nhược điểm.**
- Config của DriveStudio đang chép nguyên file thay vì kế thừa (`streetgs_nocampose.yaml` khác `streetgs.yaml` 9 dòng trên 148). File approach không sửa được chuyện này.
- Approach từ codebase ngoài cần thêm một runner riêng trong `drive2sim/external/`.

### 2.6 Lưu trạng thái bằng thư mục, không dùng cơ sở dữ liệu

**Khái niệm.** Mọi thứ tool cần nhớ là file trong `data/`, `cache/`, `runs/`. Không có SQLite, PostgreSQL hay dịch vụ chạy nền.

**Lý do.** Tool chạy cho một người trên một máy (SPEC mục 2). Kết quả từ Kaggle về dạng file, ghép thẳng vào thư mục là xong.

**Ưu điểm.** Mang cả thư mục sang máy khác là dùng được; dễ gỡ lỗi; không có dịch vụ phải giữ cho chạy.

**Nhược điểm.**
- Không có khoá ghi giữa nhiều tiến trình; tool phải tự tránh bằng file khoá và thư mục `.partial`.
- Liệt kê hàng nghìn run phải quét thư mục.

### 2.7 Cache dữ liệu sinh thêm theo mã băm nội dung

**Khái niệm.** Dữ liệu do bước chuẩn bị sinh ra (mask, SMPL, depth, thư mục DriveStudio) là *artifact*, đặt tên theo sha256 của {bước, phiên bản, tham số, mã băm đầu vào} (CONTRACTS mục 5). Artifact đã ghi xong thì không sửa.

**Lý do.** Hai approach cùng cần mask bầu trời trên một cảnh thì chỉ sinh một lần.

**Ưu điểm.** Tiết kiệm GPU, nhất là với quota Kaggle.

**Nhược điểm.**
- `cache/` lớn dần; cần lệnh dọn cache.
- Sửa mã một bước mà quên tăng phiên bản thì cache cũ vẫn được dùng.

### 2.8 Train trên Kaggle qua một executor riêng

**Khái niệm.** Kaggle cho chạy notebook trên GPU miễn phí, giới hạn thời gian mỗi phiên và quota theo tuần. Kaggle API cho phép upload dữ liệu, đẩy notebook, hỏi trạng thái và tải kết quả.

**Lý do.** Máy hiện tại chỉ có Quadro T2000 4 GB, không đủ train. Teammate cũng đang train trên Kaggle T4.

**Ưu điểm.** Không tốn tiền GPU; cùng lệnh `d2s run`, chỉ đổi executor.

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
- `tools/viewer.py` của repo đã dùng viser + nerfview: có thanh thời gian, phát lại, bật tắt từng loại Gaussian, đổi mô hình camera.

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
- viser không có khái niệm thời gian (paper viser, mục 5); thanh thời gian phải tự làm hoặc lấy từ `tools/viewer.py`.
- nerfview gọi `os._exit(1)` khi `render_fn` ném lỗi (`_renderer.py`), tắt cả tiến trình. `render_fn` của tool phải tự bắt mọi lỗi và trả về ảnh báo lỗi.
- Không dùng Rerun (đã thử trong project P-087, chạy lag), SuperSplat, PlayCanvas (vẽ splat trong trình duyệt, cùng giới hạn với chế độ xem trước).

### 2.13 Chế độ xem trước không cần GPU CUDA: Spark trong trình duyệt, file PLY

**Khái niệm.**
- *Spark* (MIT, World Labs) là thư viện JavaScript vẽ Gaussian splat trong trình duyệt, chạy trên three.js, dùng card đồ hoạ của máy xem qua WebGL2; card tích hợp thường cũng đủ. Theo README và mã nguồn (bản ngày 19/10/2026): spherical harmonics tới bậc 3 (`maxSh = 3`, `src/ExtSplats.ts`); đọc PLY, SPZ, SPLAT, KSPLAT, SOG; có level-of-detail; nhiều đối tượng splat di chuyển riêng được; không có fisheye.
- *PLY* là định dạng file của 3DGS gốc: 59–62 số thực 32 bit mỗi Gaussian, khoảng 240 byte, nên 3 triệu Gaussian là khoảng 720 MB. `tools/export_ply.py` của repo xuất đúng định dạng này trong hệ world; export gốc của DriveStudio (`utils/misc.py:50`) co toạ độ vào hộp đơn vị.
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

**Lý do.** Trùng với `CameraSpec` của proto và với ba mô hình tương ứng của NCore. Hàm chiếu của DriveStudio (`utils/geometry.py`, `project_camera_model`) đã hỗ trợ `pinhole`, `fisheye`, `ftheta`.

**Ưu điểm.** Bao phủ cả hai track; dữ liệu chuẩn, renderer và server dùng cùng một bộ tham số.

**Nhược điểm.**
- Mô hình khác (omnidirectional, double sphere) phải chuyển gần đúng; kiểm tra khi biết dữ liệu của mentor.
- NCore còn có pinhole lý tưởng và mô hình méo kính chắn gió; đường PhysicalAI hiện từ chối mô hình kính chắn gió.

## 3. Cấu trúc repo đích

```
drivestudio_plus/                         branch drive2sim
├── datasets/                             (DriveStudio)
│   ├── ncore/                (mới)       loader NCore chung: ncore_sourceloader.py
│   └── waymo/ nuscenes/ ...  (giữ tạm)   loader cũ, bỏ sau khi qua kiểm tra hồi quy
├── models/                               (DriveStudio, hạn chế sửa)
├── configs/                              (DriveStudio + teammate)
│   └── datasets/ncore/       (mới)       config dataset cho loader NCore
├── tools/                                (DriveStudio + teammate)
├── drive2sim/                (mới)       toàn bộ tầng dữ liệu và tầng tool
│   ├── core/                             schema, manifest, split, artifact, run; phép biến đổi toạ độ
│   ├── ingest/                           converter từng nguồn → NCore V4; kiểm tra; manifest
│   ├── prep/                             chuẩn bị riêng: select, undistort, masks, smpl, depth, to_drivestudio
│   ├── approaches/                       <id>.yaml
│   ├── runner/                           chạy một approach: prep → train → eval → export; ghi run.json
│   ├── kaggle/                           sinh notebook, đẩy job, theo dõi, kéo kết quả
│   ├── export/                           PLY world, gói xem trước
│   ├── sim/                              renderer.py (L2), actors.py (L3), server/ (L4, proto NuRec)
│   ├── studio/                           api/ (FastAPI), frontend/ (Vite + React), viz/ (viser)
│   ├── external/                         runner cho codebase ngoài (sau này)
│   └── cli/                              lệnh d2s
├── third_party/                          smplx, Humans4D; ncore (converter, git submodule, mới)
├── docs/drive2sim/           (mới)       SPEC, ARCHITECTURE, CONTRACTS
└── tests/drive2sim/          (mới)
```

## 4. Các khối

### 4.1 core

Schema Pydantic v2 cho manifest, split, artifact, approach, run, metric (CONTRACTS); phép biến đổi toạ độ theo quy ước NCore; đọc dữ liệu chuẩn qua `nvidia-ncore`. Không nạp model.

### 4.2 ingest

Mỗi nguồn một converter, ghi ra một kho NCore V4 trong `data/<scene_id>/`, kèm `manifest.yaml` và báo cáo kiểm tra.

| Nguồn | Converter |
|---|---|
| Waymo, nuScenes, Argoverse 2, KITTI, PhysicalAI, COLMAP | Converter có sẵn của NCore (`third_party/ncore/tools/data_converter/`) |
| nuPlan, PandaSet | Tự viết. Một phương án đáng thử: chạy script tiền xử lý có sẵn của DriveStudio, rồi viết một converter từ thư mục DriveStudio sang NCore; một converter này dùng được cho mọi dataset DriveStudio đọc được |
| CARLA | Tự viết, từ script sinh rig fisheye |
| Dữ liệu mentor | Tự viết khi biết định dạng; nếu chỉ là ảnh kèm pose thì thử đường COLMAP |

Bộ kiểm tra: đồng bộ thời gian giữa camera, calib, mô hình ống kính, FOV, ảnh hỏng hoặc thiếu (CONTRACTS mục 3.2).

### 4.3 prep

Mỗi bước chuẩn bị là một hàm có tên, phiên bản, tham số, đầu vào và đầu ra; kết quả là artifact (mục 2.7).

| Bước | Việc | Mã có sẵn để dựa vào |
|---|---|---|
| `select` | Chọn camera, đoạn khung, độ phân giải | |
| `undistort` | Khử méo về pinhole cho approach chỉ nhận pinhole | `data.pixel_source.undistort` của loader PhysicalAI |
| `masks.segformer_hf` | Mask bầu trời và mask vật động | `datasets/tools/extract_masks_hf.py` |
| `smpl` | Tư thế người | `datasets/tools/humanpose_process.py` |
| `depth` | Depth prior cho khởi tạo không dùng LiDAR | Chưa có |
| `to_drivestudio` | NCore → thư mục DriveStudio cho loader NCore | `datasets/physicalai/physicalai_preprocess.py` |

### 4.4 Tầng tái dựng (DriveStudio)

Train bằng `tools/train.py` với config method và config dataset `configs/datasets/ncore/`. Đánh giá bằng `tools/eval.py` (PSNR, SSIM, LPIPS; thêm PSNR vùng người, vùng xe, vùng có vật). Không đổi cấu trúc `models/`.

### 4.5 approaches và runner

`runner` đọc file approach, kiểm tra điều kiện với manifest và split, chạy các bước `prep`, gọi train, đánh giá, export, rồi ghi `run.json` cạnh thư mục log của DriveStudio (CONTRACTS mục 6).

Approach dự kiến, map sang config có sẵn:

| Approach | Config | Method | Renderer | CamPose | FastGS |
|---|---|---|---|---|---|
| `streetgs` | `streetgs_nocampose.yaml` | Street Gaussians | 3DGS | tắt | tắt |
| `omnire` | `omnire_nocampose.yaml` | OmniRe | 3DGS | tắt | tắt |
| `streetgs_ut` | `streetgs_ut.yaml` | Street Gaussians | 3DGUT | tắt | tắt |
| `streetgs_geer` | `streetgs_geer.yaml` | Street Gaussians | 3DGEER | tắt | tắt |
| `omnire_geer` | `omnire_geer.yaml` | OmniRe | 3DGEER | tắt | tắt |
| `streetgs_fastgs` | `streetgs_fastgs.yaml` | Street Gaussians | 3DGS | tắt | bật |
| `streetgs_geer_fastgs` | `streetgs_geer_fastgs.yaml` | Street Gaussians | 3DGEER | tắt | bật |
| `omnire_fastgs` | `omnire_fastgs.yaml` | OmniRe | 3DGS | tắt | bật |

Các config này đều tắt CamPose (3DGEER không tối ưu được pose camera), nên các approach chỉ khác nhau ở renderer hoặc luật densify.

### 4.6 kaggle

Đóng gói job, upload cảnh thành dataset private (một lần, nhận diện theo mã băm), sinh notebook chỉ cài repo và chạy `d2s run`, theo dõi qua Kaggle API, tải output về và ghép vào `runs/` và `cache/`.

### 4.7 export

- `ply_world`: bọc `tools/export_ply.py` (nền và xe nướng ở một khung).
- `preview`: mở rộng từ đó, xuất từng xe trong hệ object kèm pose theo khung (CONTRACTS mục 7). Dữ liệu cần (`instances_quats`, `instances_trans`, `instances_fv`, `points_ids`) có sẵn trong `state_dict()` của RigidNodes (`models/nodes/rigid.py:483`).

### 4.8 sim

- `renderer.py` (L2): nạp checkpoint, render ở `CameraSpec` và pose bất kỳ, kèm trạng thái actor. Tách từ `tools/viewer.py` và trainer.
- `actors.py` (L3): liệt kê, ẩn, xoá xe và sửa quỹ đạo bằng cách thay `instances_trans`, `instances_quats`, `instances_fv` của RigidNodes.
- `server/` (L4): cài `SensorsimService` theo proto của NuRec (giữ nguyên ghi chú license), ánh xạ `render_camera` và `batch_render_camera` sang `renderer.py`, `dynamic_objects` sang `actors.py`; hàm chưa hỗ trợ trả `UNIMPLEMENTED`.

### 4.9 studio

- `api/`: FastAPI phục vụ REST cho cảnh, approach, run, metric; SSE cho tiến độ và log; file tĩnh của frontend, video, ảnh, gói xem trước. Chỉ nghe trên `127.0.0.1`.
- `frontend/`: Vite + React theo luồng 8 bước của SPEC mục 5; video là thẻ `<video>` MP4 H.264, phát đồng bộ nhiều camera; nút của mức approach không đạt thì ẩn.
- `viz/`: viser chạy trong tiến trình Studio, nhúng bằng iframe; ảnh và depth lấy từ worker render (mục 2.9). Hai chế độ:

| | Chế độ chính | Chế độ xem trước |
|---|---|---|
| Cần GPU CUDA | Có | Không |
| Cách render | Worker render bằng renderer của codebase; Studio đẩy ảnh lên viser bằng `set_background_image(ảnh, depth=...)` | Spark vẽ trong trình duyệt từ gói xem trước |
| Đúng với model | Hoàn toàn | Gần đúng (mục 2.13) |

### 4.10 cli

| Lệnh | Việc |
|---|---|
| `d2s ingest` | Chuyển nguồn sang dữ liệu chuẩn, ghi manifest, kiểm tra |
| `d2s split` | Tạo split |
| `d2s run` | Chạy một approach trên một cảnh, chọn executor |
| `d2s status` / `d2s pull` | Xem trạng thái, tải kết quả từ Kaggle |
| `d2s eval` / `d2s report` | Đánh giá lại một run; bảng so sánh nhiều run |
| `d2s export` | Xuất PLY, gói xem trước |
| `d2s studio` / `d2s serve` | Bật Studio; bật server gRPC cho một run |
| `d2s cache gc` | Dọn cache |

## 5. Luồng dữ liệu

```
nguồn ──d2s ingest──► data/<scene_id>/ (NCore V4 + manifest.yaml + reports/validation.json)
                              │
                         d2s split ──► data/<scene_id>/splits/<split_id>.json
                              │
approach.yaml ──runner──► prep (select · masks · smpl · to_drivestudio ...) ──► cache/<artifact_id>/
                              │
                              ▼
                   tools/train.py + tools/eval.py ──► runs/<scene_id>/<approach_id>/<run_id>/
                                                        run.json · log của DriveStudio · checkpoint · video
                              │
                   Studio (đọc) · d2s export · server gRPC (L4)
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

Loader 7 dataset; tiền xử lý (mask, SMPL); OmniRe, Street GS, PVG, Deformable-GS; renderer 3DGS / 3DGUT / 3DGEER; rolling shutter ở `ut`/`geer`; FastGS; `tools/eval.py`; video; `tools/viewer.py`; `tools/export_ply.py`; đọc NCore V4 cho PhysicalAI.

### 7.2 Chênh lệch với thiết kế đích (đã kiểm trong code)

| Chênh lệch | Vị trí | Việc cần làm |
|---|---|---|
| Mỗi dataset một loader riêng | `datasets/*/*_sourceloader.py` | Loader NCore chung (mục 2.4) |
| Đường NCore chỉ dành cho PhysicalAI: gắn cứng 7 camera, tìm file `pai_*.json`, chỉ nhận F-theta, bắt buộc có LiDAR | `datasets/physicalai/physicalai_preprocess.py` dòng 16, 36, 82–83, 411 | Đọc camera từ dữ liệu; thêm pinhole và fisheye OpenCV; cho phép không có LiDAR |
| Khởi tạo bắt buộc có LiDAR | `datasets/driving_dataset.py:189` (`get_lidar_samples`) | Nhận point cloud khởi tạo từ bước `prep` |
| Chỉ một kiểu split: test là các khung N, 2N, 3N... với mọi camera; 19 trên 21 config dataset đặt `test_image_stride: 0`, tức không có khung test | `split_train_test` trong `datasets/driving_dataset.py` | Trước mắt chỉ dùng kiểu `every_nth_frame` (CONTRACTS mục 3.1); sau mới nhận split tuỳ ý |
| Chạy tiếp hỏng: khi có `--resume_from`, `do_save` luôn sai nên không lưu checkpoint; checkpoint chỉ lưu model, không có trạng thái optimizer | `tools/train.py:295-301` | Lưu checkpoint đầy đủ, cho phép lưu khi chạy tiếp |
| Render nằm lẫn trong viewer và trainer | `tools/viewer.py` (2523 dòng), `models/trainers/base.py` (1125 dòng) | Tách hàm render dùng riêng cho `sim/` |
| Export không có sky, SMPL, vật biến dạng; SH của xe không xoay theo xe | `tools/export_ply.py` | Ghi rõ giới hạn; mở rộng cho gói xem trước |
| Không có test | | `tests/drive2sim/`, bắt đầu bằng kiểm tra hồi quy của loader |
| Phiên bản PyTorch chưa khớp: `requirements.txt` ghi `torch==2.0.0+cu117`, gsplat-geer ghi đã chạy với 2.4.1 | `requirements.txt` | Hỏi teammate; chốt một môi trường |

### 7.3 Nguyên tắc khi sửa lõi

- Code mới đặt trong `drive2sim/`, `datasets/ncore/`, `configs/datasets/ncore/`.
- Thay đổi bắt buộc ở `datasets/`, `tools/`, `models/` làm nhỏ, mỗi thay đổi một commit, báo teammate trước.
- Loader cũ chỉ bỏ sau khi qua kiểm tra hồi quy.

### 7.4 Thứ tự làm

1. **Kaggle chạy tiếp được:** sửa `tools/train.py` (mục 7.2) và làm `drive2sim/kaggle/`. Chạy được `omnire` trên cảnh nuScenes mini mà teammate đã dùng.
2. **Tầng dữ liệu:** nuScenes mini → NCore bằng converter có sẵn (thử chạy không cần Bazel, trên Kaggle) → loader NCore → kiểm tra hồi quy với loader cũ.
3. **runner, approaches, run.json** → đạt L1 theo thiết kế; Studio nhóm A–D làm song song.
4. **sim/renderer.py** → L2 và viewer chế độ chính.
5. **sim/actors.py** → L3.
6. **sim/server/** + CARLA → L4.
7. Converter còn thiếu theo thứ tự cần dùng: CARLA, dữ liệu mentor, nuPlan, PandaSet. Bước `depth` cho khởi tạo không dùng LiDAR.

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
- `tools/export_ply.py` dùng `plyfile` (GPLv3+).
- Weights SegFormer của NVIDIA dùng trong `extract_masks_hf.py` chỉ cho dùng phi thương mại.
- NCore, proto `SensorsimService`: Apache-2.0; giữ nguyên ghi chú bản quyền khi đưa vào repo.
- viser, nerfview, Spark: MIT.
- Dữ liệu (Waymo, nuScenes...) không nằm trong repo; dataset upload lên Kaggle luôn để private.

## 10. Kiểm thử

- Kiểm tra hồi quy của tầng dữ liệu: cùng cảnh, loader cũ và loader NCore, so PSNR test (SPEC mục 13).
- Cảnh mẫu nhỏ (vài khung, 2 camera) làm fixture cho `core`, `ingest`, `prep`, `runner`.
- Kiểm tra schema cho mọi file trong `drive2sim/approaches/`.
- Kiểm tra chạy tiếp: ngắt một run giữa chừng rồi chạy lại, so checkpoint.
