# drive2sim: kiến trúc

Tài liệu này mô tả cách xây tool theo `SPEC.md`, ngay trên repo này. Định dạng dữ liệu và interface nằm trong `CONTRACTS.md`.

Đây là **thiết kế đích**. Mục 7 so sánh với hiện trạng của repo và đưa ra lộ trình refactor. Những gì đã kiểm tra trong mã nguồn hoặc tài liệu gốc thì ghi rõ nguồn; những gì chưa kiểm tra thì ghi là chưa kiểm tra.

Trạng thái: bản nháp, ngày 12/11/2026.

**Đọc theo thứ tự nào.** Người mới đọc mục 1 (tổng quan), rồi mục 7 (đã làm gì, còn gì, làm theo thứ tự nào), rồi mục 3.1 (code mới đặt ở đâu). Mười lăm quyết định ở mục 2 đọc khi cần biết lý do của một lựa chọn cụ thể. Mục 4–6 là chi tiết từng khối, luồng dữ liệu và interface theo mức năng lực.

## 1. Tổng quan

Repo gồm bốn tầng:

```
┌─ 1. DỮ LIỆU ──────────────────────────────────────────────────────────────────────────┐
│ ingest/<nguồn> ──► data/<scene>/ : thư mục DriveStudio + manifest.yaml + validation.json │
│ (7 dataset, CARLA, mentor, COLMAP; mỗi cảnh chạy một lần, không phụ thuộc approach)   │
└───────────────────────────────────────┬───────────────────────────────────────────────┘
┌─ 2. PIPELINE (approach = danh sách stage, có mục cấu hình và preset) ───────────────────────┐
│  prep.*  ──►  init.*  ──►  train.*  ──►  eval.standard  ──►  render.*  ──►  export.*  │
│  select       lidar        drivestudio   (một protocol      videos        ply_world   │
│  undistort    depth        external:X    cho mọi approach)                preview     │
│  masks        slam                                                                    │
│  smpl                                                                                 │
│  ── cache dùng chung giữa các approach ──┤── trong thư mục run, đánh dấu done ────────│
└───────────────────────────────────────┬───────────────────────────────────────────────┘
┌─ 3. TÁI DỰNG (DriveStudio, sau refactor) ─────────────────────────────────────────────┐
│ drive2sim/recon/: data/loaders (một loader mỗi dataset) · models/nodes · trainers · renderer 3DGS/3DGUT/3DGEER │
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
| 2.2 | Dùng thư mục cảnh của DriveStudio (như script có sẵn ghi ra) cộng manifest làm định dạng dữ liệu chuẩn; NCore là nguồn vào và định dạng xuất |
| 2.3 | Tách ingest (chung cho mọi approach) khỏi các stage chuẩn bị riêng cho approach |
| 2.4 | Giữ loader riêng của từng dataset; gộp khi có điều kiện |
| 2.5 | Approach là danh sách stage chạy tuyến tính, có mục cấu hình, ràng buộc và preset |
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
- Phần teammate thêm: densify và prune kiểu FastGS (`docs/methods/fastgs.md`), `tools/export_ply.py`, `datasets/tools/extract_masks_hf.py`, các config `*_nocampose`.

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

### 2.2 Thư mục cảnh của DriveStudio làm định dạng dữ liệu chuẩn

**Khái niệm.** Script tiền xử lý của DriveStudio (`drive2sim/recon/preprocess/<dataset>/`) chuyển dữ liệu gốc của một dataset thành một thư mục cảnh: ảnh theo khung và camera, intrinsics, pose, LiDAR, box (`instances/`). Loader của dataset đó (`drive2sim/recon/data/loaders/<dataset>.py`) đọc thư mục này khi train. *Định dạng chuẩn* của tool là đúng thư mục đó, **giữ nguyên như script ghi ra**, cộng `manifest.yaml`, split và báo cáo kiểm tra (CONTRACTS mục 2.1).

Cách lưu pose khác nhau giữa các dataset (đã kiểm trong các loader) và được giữ nguyên, vì mỗi dataset đọc bằng loader riêng:

| Dataset | Pose |
|---|---|
| Waymo, Argoverse, KITTI, nuPlan | `ego_pose/<khung>.txt` cùng extrinsics cố định `extrinsics/<camera>.txt` |
| PandaSet | `ego_pose/` cùng extrinsics theo từng khung `extrinsics/<khung>_<camera>.txt` |
| nuScenes | không có `ego_pose/`; extrinsics theo từng khung và `lidar_pose/` |

Các khác biệt khác: gốc toạ độ (nuScenes lấy camera trước ở khung đầu, các dataset khác lấy xe ở khung đầu), trục camera (Waymo lưu theo trục riêng). Loader của từng dataset đã xử lý các khác biệt này.

**Lý do.**
- Script và loader cho 7 dataset đã có và đang được teammate dùng để train; tool chỉ thêm `manifest.yaml` là dùng được, không sửa mã tái dựng.
- Dữ liệu chỉ chuyển một lần, chạy bằng Python thường; không cần Bazel, token GitHub hay GPU để chuyển dữ liệu (xem phương án NCore bên dưới).
- Không đổi hành vi đọc dữ liệu nên không phải kiểm tra hồi quy cho tầng dữ liệu.

**Ưu điểm.**
- Ít việc và ít rủi ro nhất để tool chạy trọn đường.
- Dễ xem bằng mắt: ảnh, file `.txt`, `.json`.
- Không lưu hai bản dữ liệu.

**Nhược điểm.**
- Không có một quy ước duy nhất trong dữ liệu chuẩn; muốn biết pose lưu thế nào phải xem dataset gốc (bảng trên).
- Định dạng do DriveStudio quyết định, không phải chuẩn bên ngoài; muốn so với NuRec phải thêm bước xuất sang NCore.
- Track fisheye cần F-theta và rolling shutter: chỉ đường PhysicalAI có sẵn, phải mở rộng khi làm CARLA.

**Phương án đã cân nhắc: NCore V4 làm định dạng chuẩn** (thiết kế trước ngày 12/11/2026). NCore là định dạng dữ liệu cảm biến của NVIDIA (Apache-2.0); NuRec đọc thẳng nó; có đủ ba mô hình camera và rolling shutter. Không chọn vì, theo tài liệu chính thức trong repo NCore (`skills/ncore-data-conversion/SKILL.md`, tag `v19.9.0`):
- Converter và công cụ kiểm tra là Bazel target, *"Bazel... Not optional"*; gói `nvidia-ncore` trên PyPI chỉ có thư viện. Lần build đầu cần GitHub token quyền `read:packages`. Converter PAI cần GPU NVIDIA để khởi động.
- 6 converter có sẵn dài 600–1100 dòng mỗi cái; dataset khác phải chép converter gần nhất rồi sửa (*"There is no scaffold or template"*).
- Trainer DriveStudio không đọc NCore, nên vẫn phải chuyển tiếp NCore sang thư mục DriveStudio: chuyển hai lần, tốn ổ đĩa gấp đôi.

NCore vẫn dùng ở hai chỗ: là **một nguồn vào** (PhysicalAI, CARLA ghi NCore), đi qua đường PhysicalAI có sẵn (`drive2sim/recon/preprocess/physicalai/process.py`); và là **định dạng xuất** khi làm approach NuRec hoặc mức L4 (`export.ncore`, làm khi cần). Submodule `third_party/ncore` giữ ở tag `v19.9.0` cho hai việc này.

### 2.3 Tách ingest khỏi các stage chuẩn bị riêng

**Khái niệm.**
- *Ingest*: chuyển nguồn sang dữ liệu chuẩn, giữ mọi camera, LiDAR, box và pose mà script tiền xử lý của dataset giữ lại (không lọc theo approach); ghi manifest và báo cáo kiểm tra. Không phụ thuộc approach; mỗi cảnh chạy một lần.
- *Stage chuẩn bị* (`prep.*`, `init.*`): chạy trong pipeline của approach; lấy bớt (camera, đoạn khung, độ phân giải, khử méo), sinh thêm (mask, SMPL, depth prior, điểm khởi tạo) và đổi dạng sang thứ codebase đọc được.

**Lý do.**
- Mọi approach bắt đầu từ cùng một dữ liệu, nên so sánh công bằng.
- Thử approach thứ hai trên cùng cảnh không phải ingest lại.
- Có manifest sau ingest thì mới khoá đúng approach, mục cấu hình không chạy được.

**Ưu điểm.** Kết quả của stage chuẩn bị được cache và dùng chung (mục 2.7).

**Nhược điểm.** Dữ liệu chuẩn giữ đủ mọi thứ nên tốn ổ đĩa hơn bản đã lọc cho một approach.

### 2.4 Giữ loader riêng của từng dataset; gộp khi có điều kiện

**Khái niệm.** DriveStudio có 7 loader (`drive2sim/recon/data/loaders/<dataset>.py`, 460–720 dòng mỗi file, tổng 3678 dòng) với cùng bộ hàm (`load_calibrations`, `load_cameras`, `load_objects`, `load_lidar`). Đo bằng `diff` so với loader Argoverse: nuPlan khác 92 dòng, KITTI 119, PandaSet 168, nuScenes 248, Waymo 379 (có thêm rolling shutter của patch Bosch), PhysicalAI 671; hàm `load_objects` của 5 trên 6 loader chỉ khác 4 dòng. Phần lớn là mã chép lại.

**Quyết định.** Chưa gộp. Nguồn mới không viết loader mới: converter của nguồn đó ghi theo đúng khung thư mục của một dataset có sẵn (ví dụ khung Argoverse: `ego_pose/` theo khung, extrinsics cố định, trục OpenCV) rồi dùng loader của dataset đó.

**Lý do.**
- Tool chạy trọn đường với loader có sẵn; gộp không thêm tính năng nào bây giờ.
- Teammate vẫn sửa loader trên `main`, và thay đổi đó được chuyển tay sang `drive2sim` (mục 7.3). Chuyển một sửa đổi vào file cùng tên dễ hơn nhiều so với vào một loader đã gộp.
- Mask và SMPL do stage `prep.*` sinh ra trong `cache/` không buộc phải sửa loader: stage dựng một thư mục cảnh ảo bằng symlink, gồm thư mục cảnh và mask, SMPL từ cache, rồi DriveStudio đọc như một thư mục cảnh bình thường (mục 4.3).

**Khi nào gộp.** Khi có một trong các điều kiện: thêm một nguồn không ghi được theo khung của dataset nào có sẵn; một sửa đổi phải lặp lại ở từ ba loader trở lên; teammate không còn sửa loader trên `main`.

**Cách gộp khi tới lúc.** Một loader chung từ mã có sẵn, cộng một bảng quy ước nhỏ cho từng dataset (cách lưu pose, gốc toạ độ, trục camera, hệ số méo, danh sách camera); dữ liệu trên đĩa và script tiền xử lý giữ nguyên; PhysicalAI giữ loader riêng vì khác quá nhiều. Kiểm tra bằng cách cho loader cũ và loader chung đọc cùng một cảnh giả theo khung của từng dataset rồi so từng giá trị (pose, intrinsics, đường dẫn ảnh, box); cần môi trường test có PyTorch bản CPU. Loader cũ chỉ xoá khi phép so khớp hoàn toàn.

### 2.5 Approach là danh sách stage, có mục cấu hình, ràng buộc và preset

**Khái niệm.**
- *Stage* là một lớp Python đăng ký theo tên (ví dụ `init.depth`), khai báo phiên bản, tham số, kiểu artifact vào và ra, có dùng LiDAR không, có được cache không (CONTRACTS mục 5.1).
- *Approach* là file `drive2sim/approaches/<id>.yaml` gồm: danh sách stage chạy lần lượt; *mục cấu hình* (`options`) cho những khâu thay thế được cho nhau; *ràng buộc* (`constraints`) giữa các mục cấu hình; *preset* là tổ hợp cấu hình đặt sẵn; cùng track, mức năng lực, điều kiện dữ liệu (CONTRACTS mục 4).
- *Tuyến tính*: các stage chạy đúng thứ tự trong file, mỗi lần một stage. Một stage dùng được kết quả của bất kỳ stage nào đứng trước. Không có điều kiện, nhánh song song hay vòng lặp lúc đang chạy.
- *Mục cấu hình* được quyết định **trước khi chạy** (người dùng chọn trên Studio hoặc CLI, hoặc engine tự chọn theo manifest với giá trị `auto`). Sau khi thay mục cấu hình bằng lựa chọn cụ thể, danh sách stage là cố định.

**Lý do.**
- Các approach của dự án khác nhau đúng ở từng khâu: khởi tạo (LiDAR, depth, SLAM), renderer (3DGS, 3DGUT, 3DGEER), luật densify (mặc định, FastGS), CamPose, có bước feed-forward hay không. Stage biến mỗi khâu thành một mảnh thay được.
- Mục cấu hình tránh nhân file: 2 cách khởi tạo × 3 renderer × FastGS bật/tắt là 12 tổ hợp nhưng vẫn là một approach. Thí nghiệm "chỉ đổi một khâu rồi so" là đổi một mục cấu hình.
- Mọi nhu cầu hiện có đều làm được mà không cần rẽ nhánh: điều kiện chuyển thành mục cấu hình `auto` hoặc khoá theo manifest; so sánh song song chuyển thành nhiều run dùng chung cache; vòng lặp đặt trong một stage.

**Quy tắc dùng mục cấu hình.**
1. Mục cấu hình chỉ dành cho khâu thay thế được cho nhau: khởi tạo, renderer, luật densify, CamPose, độ phân giải. Method khác nhau (OmniRe so với Street GS) hoặc codebase khác nhau là approach riêng.
2. Tổ hợp không hợp lệ phải khai báo trong `constraints`, kèm lý do; Studio và CLI khoá tổ hợp đó.

**Ưu điểm.**
- Thay một khâu không đụng khâu khác; mỗi stage test riêng được.
- Thêm approach mới chủ yếu là ghép stage có sẵn; thêm một lựa chọn mới (ví dụ `init.slam`) là thêm một giá trị vào mục cấu hình.
- Studio so sánh được theo từng mục cấu hình, vì `run.json` ghi lại lựa chọn.

**Nhược điểm.**
- Phải viết engine nhỏ (registry, thay mục cấu hình, kiểm tra ràng buộc, chạy tuần tự, đánh dấu done).
- Phải tách khởi tạo và đánh giá ra khỏi `drive2sim/recon/tools/train.py` và trainer của DriveStudio (mục 7.2).
- File approach có mục cấu hình khó đọc hơn một file cố định: phải nhìn cả mục cấu hình, giá trị mặc định và preset.
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

**Lý do.** Máy phát triển thường không có GPU đủ VRAM để train (các run trong README mục "Results so far" dùng 7–15 GB VRAM). Kaggle T4 16 GB miễn phí là nơi các run hiện có đã chạy.

**Ưu điểm.** Không tốn tiền GPU; cùng lệnh `d2s run start`, chỉ đổi `--executor`.

**Nhược điểm.**
- Phiên bị ngắt sau một thời gian, nên phải lưu checkpoint đầy đủ và chạy tiếp được. Chạy tiếp đã sửa trong code, chưa chạy thử (mục 7.2).
- Upload cảnh lớn lâu.
- Không xem viewer được trong lúc train.

### 2.9 Môi trường

**Khái niệm.** Một conda env hoặc venv chứa đúng phiên bản Python, PyTorch, CUDA và thư viện.

**Quyết định.**
- Mọi approach dòng DriveStudio dùng chung một môi trường của repo, với gsplat-geer. gsplat-geer có đủ ba renderer.
- Codebase ngoài (NuRec, feed-forward...) chạy trong môi trường riêng, gọi qua tiến trình con: tool chạy `d2s-worker` trong môi trường đó, đưa vào file job JSON, đọc lại file kết quả JSON, nhận tiến độ qua stdout (CONTRACTS mục 9).
- Tiến trình Studio không nạp model. Render cho viewer và server chạy trong một tiến trình worker riêng, giữ sống suốt phiên.
- Mỗi môi trường mô tả bằng một file trong `envs/`: `main.yml` cho môi trường chính, `<codebase>.yml` cho codebase ngoài. Môi trường chính đã có: `envs/main.yml` chỉ dựng Python 3.10 bằng conda, còn mọi gói Python cài bằng `envs/install_main.sh` (PyTorch 2.4.1 + CUDA 12.1, `requirements.txt`, gsplat-geer commit `f15cee2`, nvdiffrast, pytorch3d, smplx, tool). Notebook Kaggle chạy đúng script đó với Python của Kaggle, nên máy cá nhân và Kaggle cùng phiên bản. Phiên bản chốt trong script khớp `EXPECTED` của `drive2sim/engine/environment.py` (có test kiểm); Studio và `d2s approach check` dùng nó để báo môi trường máy này đúng, khác phiên bản (vẫn chạy, kết quả có thể lệch) hay thiếu.

**Lý do.** Các codebase thường không cài chung được. Ví dụ đã kiểm: DriveStudio gốc cài `gsplat` 1.3.0 với `torch==2.0.0+cu117`; gsplat-geer cũng tên `gsplat` nhưng bản 1.5.3, và README của nó (`app/README.md`) ghi đã chạy với CUDA 12.1 và PyTorch 2.4.1. Hai bản cùng tên không nằm chung một môi trường được.

**Ưu điểm.** Codebase ngoài không làm hỏng môi trường chính; worker lỗi thì Studio vẫn chạy.

**Nhược điểm.**
- Mỗi môi trường tốn vài GB ổ đĩa.
- Đã chọn PyTorch 2.4.1 + CUDA 12.1 theo gsplat-geer; chưa xác nhận bằng một run thật trên Kaggle (mục 8 câu 1). Notebook Kaggle build gsplat-geer, nvdiffrast, pytorch3d mỗi phiên, tốn vài chục phút GPU; lưu sẵn bản đã build là việc sau.

### 2.10 API của Studio: FastAPI và Server-Sent Events, chỉ nghe trên `127.0.0.1`

**Khái niệm.** FastAPI là framework Python để viết HTTP API. Server-Sent Events (SSE) là cách server đẩy liên tục dữ liệu xuống trình duyệt qua một kết nối HTTP, một chiều; dùng cho tiến độ và log. `127.0.0.1` là địa chỉ chỉ máy đang chạy truy cập được.

**Lý do.** Một người dùng, một máy; không cần tài khoản. Trên máy GPU thuê, mở qua port forward của SSH.

**Ưu điểm.** SSE đơn giản hơn WebSocket cho việc đẩy tiến độ; không mở cổng ra ngoài nên không cần xác thực.

**Nhược điểm.** Lệnh từ người dùng phải đi qua REST; muốn chia sẻ từ xa phải tự port forward.

Route, thân, lỗi và sự kiện SSE quy định ở CONTRACTS mục 11; API chỉ là lớp mỏng trên engine, mỗi lệnh CLI có một route tương ứng.

### 2.11 Giao diện Studio: Vite + React

**Khái niệm.** React dựng giao diện web từ các thành phần; Vite build mã React thành file HTML, JS, CSS tĩnh.

**Lý do.** Build sẵn thành file tĩnh và phục vụ bằng FastAPI, nên người dùng không cần Node.js. Client của viser cũng viết bằng React + Vite (mã nguồn trong wheel viser, thư mục `viser/client/`).

**Ưu điểm.** Có sẵn thư viện biểu đồ, bảng, trình phát video.

**Nhược điểm.** Phát triển cần Node.js; hai ngôn ngữ nên kiểu dữ liệu phải giữ khớp, ví dụ sinh kiểu TypeScript từ JSON Schema.

**Cách làm đã chọn.** Bản build (`drive2sim/studio/frontend/dist/`, có font Be Vietnam Pro kèm bộ chữ tiếng Việt) được commit và đóng gói theo package, nên người dùng không cần Node.js. Kiểu TypeScript của các file trong CONTRACTS sinh tự động bằng `npm run types` (gọi `d2s schema export`) vào `src/api/generated/`. Không dùng thư viện router hay biểu đồ: điều hướng bằng hash URL, biểu đồ vẽ bằng SVG. Phụ thuộc: `react`, `react-dom`, font, `lucide-react` (icon), và cho trang xem 3D `three` cùng `@sparkjsdev/spark` (mục 2.13). Trang xem 3D nạp lười: gói của nó khoảng 3,3 MB (Spark kèm WASM) chỉ tải khi mở trang đó, các trang khác khoảng 350 KB JS. Server trả `Cache-Control: no-cache` cho `index.html`, nên bản build mới có hiệu lực ngay; file trong `assets/` mang hash nên vẫn được cache.

**Bố cục.** Thanh điều hướng trái; thanh trên có đường dẫn trang, nút Hoạt động (run đang chạy kèm tiến độ trực tiếp, việc nền ingest và Kaggle), tìm nhanh `Ctrl+K` (trang, cảnh, run), đổi sáng/tối và ngôn ngữ. Trạng thái trực tiếp dùng chung một kết nối SSE `/api/events` và đọc định kỳ `/api/jobs` (`src/live.tsx`); run xong hoặc lỗi, việc nền xong hoặc lỗi đều có thông báo nổi. Thao tác xoá luôn qua hộp xác nhận liệt kê những gì bị xoá.

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
- Không dùng Rerun (đã thử, chạy lag với cảnh lớn), SuperSplat, PlayCanvas (vẽ splat trong trình duyệt, cùng giới hạn với chế độ xem trước).

### 2.13 Chế độ xem trước không cần GPU CUDA: Spark trong trình duyệt, file PLY

**Khái niệm.**
- *Spark* (MIT, World Labs) là thư viện JavaScript vẽ Gaussian splat trong trình duyệt, chạy trên three.js, dùng card đồ hoạ của máy xem qua WebGL2; card tích hợp thường cũng đủ. Theo README và mã nguồn (bản ngày 19/10/2026): spherical harmonics tới bậc 3 (`maxSh = 3`, `src/ExtSplats.ts`); đọc PLY, SPZ, SPLAT, KSPLAT, SOG; có level-of-detail; nhiều đối tượng splat di chuyển riêng được; không có fisheye.
- *PLY* là định dạng file của 3DGS gốc: 59–62 số thực 32 bit mỗi Gaussian, khoảng 240 byte, nên 3 triệu Gaussian là khoảng 720 MB. `drive2sim/recon/tools/export_ply.py` của repo xuất đúng định dạng này trong hệ world; export gốc của DriveStudio (`drive2sim/recon/utils/misc.py:50`) co toạ độ vào hộp đơn vị.
- *SPZ* (Niantic) lưu cùng thông tin nhưng làm tròn rồi nén ZSTD; README của SPZ ghi file thường nhỏ hơn PLY khoảng 10 lần. Vị trí là số nguyên 24 bit, mặc định 12 bit phần lẻ (`load-spz.cc`): độ phân giải khoảng 0,24 mm, phạm vi ±2048 m. SH bậc 1 còn 5 bit, bậc cao hơn còn 4 bit.

**Lý do.** Mở được kết quả trên máy không có GPU CUDA, ví dụ laptop khi bàn giao.

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
├── docs/
│   ├── datasets/                          hướng dẫn tải và tiền xử lý: <dataset>.md, human_pose.md
│   ├── methods/                           ghi chú phương pháp: 3dgeer.md, fastgs.md
│   ├── media/                             hình cho README
│   └── drive2sim/                         SPEC.md, ARCHITECTURE.md, CONTRACTS.md
├── drive2sim/
│   ├── paths.py                           đường dẫn tới assets/ và configs/ trong package
│   ├── assets/                            ego_masks/<dataset>/<camera>.png, scene_lists/<dataset>_*.txt
│   ├── configs/
│   │   ├── methods/                       config method: <method>[_<biến thể>].yaml; paper_legacy/
│   │   └── datasets/                      <dataset>/<số camera>cams.yaml
│   ├── recon/                             mã tái dựng (từ DriveStudio)
│   │   ├── data/                          đọc dữ liệu đã tiền xử lý để train
│   │   │   ├── base/                      pixel_source, lidar_source, scene_dataset, split_wrapper
│   │   │   ├── loaders/                   <dataset>.py: một loader mỗi dataset (mục 2.4)
│   │   │   ├── driving_dataset.py
│   │   │   └── dataset_meta.py
│   │   ├── preprocess/                    dữ liệu gốc → thư mục DriveStudio
│   │   │   ├── run.py                     điểm vào: python -m drive2sim.recon.preprocess.run --dataset <dataset>
│   │   │   ├── <dataset>/                 process.py, human.py, preview.py; waymo/download.py; utils.py khi cần
│   │   │   ├── masks/                     segformer_mmseg.py, segformer_hf.py
│   │   │   ├── smpl/                      run.py, extract.py, postprocess.py
│   │   │   └── multiprocess.py
│   │   ├── models/                        gaussians/, nodes/, modules.py, losses.py, human_body.py
│   │   ├── trainers/                      base.py, single.py, scene_graph.py
│   │   ├── rendering/                     videos.py
│   │   ├── utils/                         camera, geometry, logging, misc, visualization, backup
│   │   └── tools/                         train.py, eval.py, viewer.py, export_ply.py, gather_results.py (thay dần bằng stage và CLI)
│   ├── core/                              identifiers.py, schema.py, files.py, split.py (đã có); transforms.py, camera.py (chưa có)
│   ├── ingest/                            detect.py (source inspect), dispatch.py (chọn chuỗi), drivestudio.py (converter chung cho 7 dataset), images.py, manifest.py, validate.py
│   ├── stages/                            commands.py, scene_view.py; prep/masks_segformer.py, prep/smpl.py, train/drivestudio.py, eval/standard.py, export/ply_world.py, export/preview.py (đã có); init/, render/ (khung)
│   ├── engine/                            stage.py, registry.py, resolve.py, approaches.py, workdir.py, scenes.py, cache.py, run.py, report.py, housekeeping.py (xoá, dọn cache)
│   ├── approaches/                        omnire.yaml, streetgs.yaml
│   ├── kaggle/                            client.py, bundle.py, executor.py, templates/run.ipynb
│   ├── sim/                   (khung)     renderer.py (L2), actors.py (L3), server/ (L4; protos/ từ NuRec)
│   ├── studio/                            api/ (app.py, state.py, events.py, jobs.py, routes/<nhóm>.py); frontend/ (Vite + React: src/pages/, src/components/, src/viewer/ (Spark), src/live.tsx, src/locales/, src/api/generated/, dist/ đã build); viz/ (khung)
│   ├── external/              (khung)     <codebase>/: worker cho codebase ngoài
│   ├── scripts/               (khung)     công cụ phát triển chạy một lần
│   └── cli/                               main.py, output.py; source.py, scene.py, split.py, approach.py, run.py, report.py, export.py, cache.py, kaggle.py, studio.py, schema.py
├── third_party/                           smplx (pip -e, import smplx); Humans4D (submodule, import hmr2); ncore (submodule, tag v19.9.0: converter NCore)
├── envs/                                  main.yml (conda: Python 3.10), install_main.sh (cài môi trường chính, dùng chung cho Kaggle) (mục 2.9)
└── tests/drive2sim/           (khung)     cùng cây thư mục với drive2sim/; fixtures/
```

Chú thích: **(khung)** là thư mục đã tạo nhưng chưa có code: package Python chỉ có `__init__.py` rỗng, thư mục khác giữ bằng `.gitkeep`. **(chưa có)** là file chưa tạo. Tên file liệt kê trong các thư mục (khung) là tên dự kiến theo CONTRACTS mục 10, tạo khi viết code.

Ở gốc repo chỉ còn những thứ không phải code của package: `docs/`, `tests/`, `third_party/`, `envs/`, `pyproject.toml`, `requirements.txt`. Config method, config dataset, file approach và assets đều nằm trong package, nên cài bằng `pip install` là có đủ; code tìm chúng qua `drive2sim/paths.py`, không phụ thuộc thư mục đang đứng.

Phân chia trong `drive2sim/recon/`:
- `preprocess/` chỉ chạy một lần trên dữ liệu gốc, ghi ra thư mục DriveStudio; `data/` chỉ đọc thư mục đó khi train. Loader không import gì từ `preprocess/`.
- Trong một nhóm, file đặt theo vai trò (`process.py`, `human.py`, `preview.py`), không lặp tên dataset trong tên file vì thư mục đã mang tên dataset.

Chạy script bằng `python -m`, ví dụ `python -m drive2sim.recon.tools.train --config_file drive2sim/configs/methods/omnire.yaml`, `python -m drive2sim.recon.preprocess.run`. Chuỗi import trong config theo đường dẫn mới, ví dụ `drive2sim.recon.models.nodes.RigidNodes`, `drive2sim.recon.data.loaders.nuscenes.NuScenesPixelSource`.

### 3.1 Code mới đặt ở đâu

| Cần thêm | Đặt ở | Tên file, tên lớp | Đăng ký hoặc khai báo ở |
|---|---|---|---|
| Dataset đã có script tiền xử lý của DriveStudio | `drive2sim/ingest/drivestudio.py` | Một dòng `DriveStudioSource` trong `SOURCES` (camera, LiDAR, cách đổi tên chuỗi sang `--scene_ids`) | Hàm `_detect_<dataset>` trong `drive2sim/ingest/detect.py`, thêm vào `_DETECTORS` |
| Nguồn dữ liệu mới chưa có script (CARLA, COLMAP, dữ liệu mentor) | `drive2sim/ingest/` | `<dataset>.py`, lớp `<Dataset>Converter`; ghi ra định dạng chuẩn (CONTRACTS mục 2.1) | Nhánh trong `drive2sim/ingest/dispatch.py` và hàm nhận dạng trong `detect.py` |
| Stage mới | `drive2sim/stages/<nhóm>/` | `<tên>.py`, lớp `<Nhóm><Tên>Stage`, `stage_id` là `<nhóm>.<tên>` | `@register_stage("<nhóm>.<tên>")` trong chính file đó |
| Loại artifact mới | `drive2sim/core/schema.py` | giá trị mới của `artifact_type` | Bảng CONTRACTS mục 5.4 |
| Approach mới | `drive2sim/approaches/` | `<approach_id>.yaml` | Tự nạp khi engine khởi động |
| Lựa chọn mới cho một mục cấu hình (ví dụ `init.slam`) | Stage mới như trên | | Thêm vào `options.<mục>.choices` của approach |
| Method, node, loại Gaussian mới | `drive2sim/recon/models/nodes/` hoặc `drive2sim/recon/models/gaussians/` | `<tên>.py`; lớp theo quy ước sẵn có (`<Tên>Nodes`, `<Tên>Gaussians`) | Config method trong `drive2sim/configs/methods/`; giá trị mới cho tham số `method` của `train.drivestudio` |
| Loader dữ liệu | `drive2sim/recon/data/loaders/` | Không thêm loader cho nguồn mới: converter ghi theo khung thư mục của một dataset có sẵn (mục 2.4) | |
| Tiện ích dùng chung của phần tái dựng | `drive2sim/recon/utils/` | `<chủ_đề>.py` | |
| Config method | `drive2sim/configs/methods/` | `<method>.yaml`; biến thể `<method>_<biến thể>.yaml` kế thừa từ config gốc, chỉ ghi phần khác | |
| Bước tiền xử lý dữ liệu gốc cho DriveStudio (tạm, cho tới khi có `ingest/`) | `drive2sim/recon/preprocess/<dataset>/` hoặc `drive2sim/recon/preprocess/<loại>/` | `process.py`, `human.py`, `preview.py`; loại chung: `masks/`, `smpl/` | Nhánh `--dataset` trong `drive2sim/recon/preprocess/run.py` |
| Hàm render, video | `drive2sim/recon/rendering/` | `<chủ_đề>.py` | |
| Hướng dẫn dataset, ghi chú phương pháp | `docs/datasets/`, `docs/methods/` | `<tên>.md` chữ thường | |
| File đi kèm package (mask thân xe, danh sách cảnh) | `drive2sim/assets/<loại>/` | | Truy cập qua `drive2sim/paths.py`, không dùng đường dẫn tương đối theo thư mục đang chạy |
| Codebase ngoài | `drive2sim/external/<codebase>/` | `worker.py`; môi trường ở `envs/<codebase>.yml` | Stage `train.external` với tham số `codebase: <tên>` |
| Lệnh CLI mới | `drive2sim/cli/<nhóm>.py` | hàm `<hành_động>` | Theo CONTRACTS mục 10.4 |
| API cho Studio | `drive2sim/studio/api/routes/<nhóm>.py` | đường dẫn `/api/<nhóm số nhiều>/...`, verb, lỗi và sự kiện theo CONTRACTS mục 11 | `app.py` |
| Màn hình Studio | `drive2sim/studio/frontend/src/pages/` | `PascalCase.tsx`; chuỗi giao diện trong `src/locales/vi.json`, `en.json` (CONTRACTS mục 10.6) | Router của frontend |
| Mẫu notebook Kaggle | `drive2sim/kaggle/templates/` | `<tên>.ipynb` | `bundle.py` |
| Test | `tests/drive2sim/<cùng đường dẫn>/` | `test_<module>.py` | |
| Fixture dữ liệu nhỏ cho test | `tests/drive2sim/fixtures/<scene_id>/` | | |
| Công cụ phát triển chạy một lần | `drive2sim/scripts/` | `<động_từ>_<đối_tượng>.py` | Không dùng trong luồng chính; việc dùng thường xuyên phải thành lệnh CLI |
| Tài liệu | `docs/drive2sim/` | `UPPER_CASE.md` cho tài liệu chính | Cập nhật cả ba tài liệu chính khi đổi thiết kế |

Quy tắc viết code (PEP 8, đặt tên, chú thích kiểu, docstring tiếng Việt, SOLID, KISS, kiểm thử, commit) nằm trong `AGENTS.md` ở gốc repo.

Quy tắc chung:
- Mọi mã nằm trong package `drive2sim/`; không thêm thư mục mã mới ở gốc repo.
- Import luôn theo đường dẫn đầy đủ (`from drive2sim.recon.models.nodes import RigidNodes`), không import tương đối giữa các nhóm lớn.
- Một file một việc. Stage, converter, nhóm lệnh CLI, nhóm route API: mỗi cái một file.
- Tên file, tên định danh và tham số theo CONTRACTS mục 10.

## 4. Các khối

### 4.1 core

Schema Pydantic v2 cho manifest, split, artifact, approach (stage, mục cấu hình, ràng buộc, preset), run, metric (CONTRACTS); phép biến đổi toạ độ theo quy ước NCore; đọc dữ liệu chuẩn qua `nvidia-ncore`. Không nạp model.

### 4.2 ingest

`d2s scene ingest` chuyển một nguồn thành thư mục cảnh `data/<scene_id>/` (CONTRACTS mục 2.1), ghi `manifest.yaml` và báo cáo kiểm tra.

| Nguồn | Converter |
|---|---|
| nuScenes, Waymo, Argoverse 2, KITTI, nuPlan, PandaSet | Gọi script tiền xử lý có sẵn (`python -m drive2sim.recon.preprocess.run --dataset <dataset>`), rồi sinh `manifest.yaml` từ thư mục kết quả |
| NCore (PhysicalAI; CARLA khi có) | Đường PhysicalAI có sẵn (`drive2sim/recon/preprocess/physicalai/`); với CARLA phải bỏ các chỗ gắn cứng 7 camera và F-theta (mục 7.2) |
| Ảnh kèm pose (COLMAP) | Tự viết, ghi theo khung thư mục của một dataset có sẵn |
| Dữ liệu mentor | Tự viết khi biết định dạng, ghi theo khung thư mục của một dataset có sẵn |

Bảy dataset ở hai dòng đầu dùng chung một converter, `drive2sim/ingest/drivestudio.py`: mỗi dataset chỉ khác nhau ở một dòng mô tả (tên camera theo thứ tự chỉ số, tên LiDAR, cách đổi tên chuỗi sang `--scene_ids` của script). Converter chạy script trong thư mục tạm, tìm thư mục cảnh duy nhất có `images/` mà script vừa ghi (mỗi script đặt ở một chỗ khác nhau), chuyển tới `data/<scene_id>/`, rồi ghi manifest và báo cáo. `drive2sim/ingest/dispatch.py` chọn phiên bản và chuỗi từ kết quả nhận dạng; CLI và Studio cùng gọi nó. Với Argoverse 2, PandaSet, nuPlan, PhysicalAI, script không ghi tên cảm biến LiDAR nên manifest ghi tên chung `LIDAR`. Chỉ số chuỗi của nuPlan giả định danh sách log sắp theo tên file `.db`; chưa kiểm với nuplan-devkit.

Converter mới viết theo cách NVIDIA làm với NCore (`skills/ncore-data-conversion/references/adapting-a-converter.md`): chép converter gần nhất *về hình dạng dữ liệu*, chỉ thay phần đọc file nguồn, phần dựng mô hình camera và LiDAR, phần tính mốc thời gian; phần ghi ra giữ nguyên.

`d2s source inspect <thư mục>` xem trước thư mục dữ liệu gốc, không ghi gì. Lệnh nhận ra dataset đã biết theo dấu hiệu trong thư mục (`v1.0-*/scene.json` là nuScenes, `.tfrecord` là Waymo, `<log>/sensors/cameras/` là Argoverse 2, `<3 chữ số>/camera/` là PandaSet, `<ngày>/<ngày>_drive_*_sync/image_02/` là KITTI, `nuplan-v1.1/splits/<split>/*.db` là nuPlan, `pai_*.json` là PhysicalAI, `sparse/0/cameras.bin` là COLMAP, `.zarr.itar` là NCore khác), báo có bao nhiêu cảnh, camera, có LiDAR và box không, rồi gợi ý lệnh ingest. Với dữ liệu lạ, lệnh chỉ báo thư mục có gì (thư mục ảnh, số ảnh, độ phân giải, file pose, LiDAR); **không đoán quy ước** (trục camera, chiều của pose, đơn vị), vì những thứ đó không suy ra được từ file. `scene ingest` không có `--source-type` thì dùng cùng bước nhận dạng.

Bộ kiểm tra (CONTRACTS mục 3.2) theo cách NVIDIA kiểm tra dữ liệu (`skills/ncore-data-conversion/SKILL.md`): không có phép kiểm tra nào đủ một mình, nên đi từ rẻ tới kỹ. Kiểm tra cấu trúc (đủ file, ảnh đọc được, số khung khớp); rồi chiếu LiDAR lên ảnh. Nếu lệch, chiếu lại ở các khung xe đứng yên: còn lệch là lỗi calib, hết lệch là lỗi mốc thời gian.

### 4.3 stages

Mỗi stage là một lớp đăng ký theo tên, theo interface ở CONTRACTS mục 5.1; nơi ghi kết quả ở CONTRACTS mục 5.2.

| Stage | Việc | Cache | Mã có sẵn để dựa vào |
|---|---|---|---|
| `prep.select` | Chọn camera, đoạn khung, độ phân giải | có | |
| `prep.undistort` | Khử méo về pinhole cho approach chỉ nhận pinhole | có | `data.pixel_source.undistort` của loader PhysicalAI |
| `prep.masks_segformer` | Mask bầu trời và mask vật động tinh; cần GPU (`D2S_DEVICE`) | có | `drive2sim/recon/preprocess/masks/segformer_hf.py` |
| `prep.smpl` | Tư thế người cho OmniRe; cần GPU, `hmr2` và box 3D | có | `drive2sim/recon/preprocess/smpl/run.py`, `extract.py` |
| `init.lidar` | Điểm khởi tạo từ LiDAR | có | `get_lidar_samples` trong `drive2sim/recon/data/driving_dataset.py` |
| `init.depth` | Điểm khởi tạo từ pose dataset + depth metric | có | Chưa có |
| `init.slam` | Pose và điểm từ SLAM, khi rig không có pose | có | Chưa có |
| `train.drivestudio` | Train bằng trainer DriveStudio; method, renderer, FastGS, CamPose chọn qua tham số. Cần LiDAR: config hiện có khởi tạo Gaussian từ LiDAR | không | `drive2sim/recon/tools/train.py` |
| `train.external` | Train bằng codebase ngoài, trong môi trường riêng | không | Chưa có |
| `eval.standard` | Đánh giá theo protocol chung (SPEC mục 9) | không | `drive2sim/recon/tools/eval.py` (`do_evaluation`) |
| `render.videos` | Video ảnh gốc, ảnh render, sai khác theo camera | không | `drive2sim/recon/rendering/videos.py` |
| `export.ply_world` | PLY 3DGS trong hệ world | không | `drive2sim/recon/tools/export_ply.py` |
| `export.preview` | Gói xem trước cho Spark | không | Mở rộng `drive2sim/recon/tools/export_ply.py`; dữ liệu cần (`instances_quats`, `instances_trans`, `instances_fv`, `points_ids`) có sẵn trong `state_dict()` của RigidNodes (`drive2sim/recon/models/nodes/rigid.py:483`) |

**Thư mục cảnh ảo.** Script DriveStudio đều đọc và ghi trong một thư mục cảnh `<data_root>/<số>/`, trong khi thư mục cảnh `data/<scene_id>/` không được sửa và stage chỉ được ghi vào thư mục của nó (CONTRACTS mục 5.1). Các stage bọc DriveStudio vì vậy dựng trong thư mục của mình một `view/000/` gồm symlink tới từng mục của thư mục cảnh, cộng các artifact cần thêm (ví dụ `sky_masks/` từ cache), rồi chạy script trên đó (`drive2sim/stages/scene_view.py`). Mục script tạo mới là thư mục thật trong thư mục của stage. View không chứa symlink tới các mục stage sắp ghi, nên script không thể ghi qua symlink vào dữ liệu thật. Cách này không sửa dòng nào trong mã DriveStudio; đã kiểm `segformer_hf.py` chỉ ghi `sky_masks/`, `fine_dynamic_masks/` và quy trình SMPL chỉ ghi dưới `humanpose/`.

Trong `train.drivestudio`, tổ hợp scene graph, renderer và luật densify vẫn chọn bằng config của DriveStudio. Chỉ tách thêm thành stage riêng khi config không làm được.

### 4.4 Tầng tái dựng (DriveStudio)

Stage `train.drivestudio` train bằng trainer trong `drive2sim/recon/trainers/` với config method và config dataset có sẵn của dataset gốc (`drive2sim/configs/datasets/<dataset>/<k>cams.yaml`), đọc thẳng `data/<scene_id>/` (DriveStudio nhận `data.scene_idx` dạng chuỗi: `data_root/<scene_idx>`). Sau refactor: trainer nhận `points_init` từ stage `init.*`; đánh giá do `eval.standard` gọi, không còn chạy ở cuối `drive2sim/recon/tools/train.py`.

### 4.5 engine và approaches

Engine chạy một approach theo các bước:
1. Đọc file approach; thay preset và lựa chọn của người dùng vào các mục cấu hình; giá trị `auto` được chọn theo manifest.
2. Kiểm tra điều kiện dữ liệu (manifest, split) và `constraints`; sai thì dừng, kèm lý do.
3. Tạo thư mục run, ghi `d2s/run.json` với lựa chọn đã quyết định.
4. Chạy lần lượt từng stage: stage cacheable có sẵn trong cache thì dùng lại; stage trong run đã có `done` thì bỏ qua.
5. Ghi trạng thái từng stage vào `run.json` sau mỗi bước.

Studio và CLI dùng chung bước 1–2 (`engine/resolve.py`), nên chạy từ đâu cũng cho cùng một danh sách stage. `d2s approach check` in ra danh sách đó mà không chạy.

Approach dự kiến:

| Approach | Mục cấu hình | Config DriveStudio dùng làm gốc |
|---|---|---|
| `omnire` | `init`: lidar / depth / auto; `renderer`: 3dgs / 3dgut / 3dgeer; `campose`; `fastgs` | `omnire*.yaml` |
| `streetgs` | như trên | `streetgs*.yaml` |

Các config hiện có của teammate map sang tổ hợp cấu hình:

| Config hiện có | Tổ hợp cấu hình tương ứng |
|---|---|
| `streetgs_nocampose.yaml` / `omnire_nocampose.yaml` | renderer 3dgs, campose tắt, fastgs tắt |
| `streetgs_ut.yaml` | renderer 3dgut, campose tắt |
| `streetgs_geer.yaml` / `omnire_geer.yaml` | renderer 3dgeer, campose tắt |
| `streetgs_fastgs.yaml` / `omnire_fastgs.yaml` | renderer 3dgs, campose tắt, fastgs bật |
| `streetgs_geer_fastgs.yaml` | renderer 3dgeer, campose tắt, fastgs bật |

Tên renderer theo CONTRACTS mục 10.3 (`3dgs`, `3dgut`, `3dgeer`); engine đổi sang `render_mode` của DriveStudio (`default`, `ut`, `geer`). Ràng buộc đã biết: renderer `3dgeer` không đi với campose bật, vì 3DGEER không tối ưu được pose camera. Các config 3DGEER của teammate còn đổi `densify_grad_thresh: 0.01` và `refine_interval: 300` (README của fork); hai giá trị này đi theo lựa chọn renderer `3dgeer`.

### 4.6 kaggle

Đóng gói job, upload cảnh thành dataset private (một lần, nhận diện theo `scene_id` và phiên bản converter), sinh notebook từ `kaggle/templates/run.ipynb`, cài repo đúng commit và chạy `d2s kaggle run <gói>`, theo dõi bằng lệnh `kaggle`, tải output về và ghép vào `runs/` và `cache/` (CONTRACTS mục 9.2). `client.py` gọi lệnh `kaggle` trong tiến trình con, nên máy không dùng Kaggle không cần cài gì và test thay được Kaggle bằng hàm giả; `bundle.py` dựng và đọc gói; `executor.py` đẩy, hỏi trạng thái, tải về.

### 4.7 export

Xem các stage `export.*` ở mục 4.3.

### 4.8 sim

- `renderer.py` (L2): nạp checkpoint, render ở `CameraSpec` và pose bất kỳ, kèm trạng thái actor. Tách từ `drive2sim/recon/tools/viewer.py` và trainer.
- `actors.py` (L3): liệt kê, ẩn, xoá xe và sửa quỹ đạo bằng cách thay `instances_trans`, `instances_quats`, `instances_fv` của RigidNodes.
- `server/` (L4): cài `SensorsimService` theo proto của NuRec (giữ nguyên ghi chú license), ánh xạ `render_camera` và `batch_render_camera` sang `renderer.py`, `dynamic_objects` sang `actors.py`; hàm chưa hỗ trợ trả `UNIMPLEMENTED`.

### 4.9 studio

- `api/`: FastAPI phục vụ REST cho cảnh, approach, run, metric; SSE cho tiến độ và log; file tĩnh của frontend, video, ảnh, gói xem trước. Chỉ nghe trên `127.0.0.1`. Hợp đồng ở CONTRACTS mục 11.
- `frontend/`: Vite + React theo luồng sử dụng ở SPEC mục 5; màn hình chọn approach có preset, mục cấu hình, lựa chọn bị khoá kèm lý do, và danh sách stage sẽ chạy (stage nào dùng lại cache); video là thẻ `<video>` MP4 H.264, phát đồng bộ nhiều camera; nút của mức approach không đạt thì ẩn.
- `viz/`: viser chạy trong tiến trình Studio, nhúng bằng iframe; ảnh và depth lấy từ worker render (mục 2.9). Hai chế độ:

| | Chế độ chính | Chế độ xem trước |
|---|---|---|
| Cần GPU CUDA | Có | Không |
| Cách render | Worker render bằng renderer của codebase; Studio đẩy ảnh lên viser bằng `set_background_image(ảnh, depth=...)` | Spark vẽ trong trình duyệt từ gói xem trước |
| Đúng với model | Hoàn toàn | Gần đúng (mục 2.13) |

### 4.10 cli

Lệnh dạng `d2s <nhóm> <hành động>`. Danh sách đầy đủ, tuỳ chọn chung, stdout/stderr và mã thoát ở CONTRACTS mục 10.4; kiểu giá trị `--set` ở 10.3; ngôn ngữ thông báo ở 10.6.

| Nhóm | Hành động |
|---|---|
| `scene` | `ingest`, `validate`, `list`, `show`, `delete` |
| `split` | `create`, `list`, `show`, `delete` |
| `approach` | `list`, `show`, `check` |
| `run` | `start`, `resume`, `status`, `list`, `cancel`, `delete` |
| `kaggle` | `push`, `status`, `pull` |
| `export` | `ply`, `preview` |
| `cache` | `list`, `gc` |
| `report`, `studio`, `serve` | (lệnh đơn) |

## 5. Luồng dữ liệu

```
nguồn ──d2s scene ingest──► data/<scene_id>/ (thư mục DriveStudio + manifest.yaml + reports/validation.json)
                              │
                         d2s split create ──► data/<scene_id>/splits/<split_id>.json
                              │
<approach_id>.yaml + preset + mục cấu hình ──engine──► danh sách stage cố định
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

### 7.1 Tiến độ

| Việc | Trạng thái | Commit |
|---|---|---|
| Docs thiết kế (SPEC, ARCHITECTURE, CONTRACTS) | Xong, cập nhật theo từng quyết định | |
| Di chuyển mã DriveStudio vào `drive2sim/recon/` (ba lượt) | Xong; kiểm tra tĩnh đạt; chưa chạy thật | `7a7c121`, `726b3ba`, `9f35aff` |
| Sửa chạy tiếp khi train | Xong trong code; chưa chạy thật | `7851701` |
| Khung thư mục của tool, submodule NCore (pin `v19.9.0`, mục 2.2) | Xong; các thư mục còn trống | `ceacb2c` |
| `core`: định danh, schema mọi file trong CONTRACTS (kể cả `job.json`, `result.json`), đọc ghi, split | Xong; 66 test đạt trên Python 3.9 và 3.12; test đọc mọi ví dụ trong CONTRACTS; qua `ruff check` | `9bb7bbc`, `96876ad` |
| `core`: phép biến đổi toạ độ, mô hình camera | Chưa làm | |
| `engine`: interface stage và registry | Xong; kiểm tra khai báo stage lúc đăng ký | `35bc590` |
| `engine`: resolve (mục cấu hình, preset, `auto`, điều kiện dữ liệu, ràng buộc, nối đầu vào) | Xong | `37d5159` |
| `engine`: thư mục làm việc, cache, tạo và chạy run (chạy tiếp, huỷ, ghi lỗi) | Xong; chạy với stage giả; 108 test đạt trên Python 3.9 và 3.12 | |
| `engine`: huỷ run đang chạy (file `cancel`, dừng nhóm tiến trình con), chạy lại một stage, xoá run, split, cảnh, liệt kê và dọn cache | Xong; test huỷ dùng tiến trình con thật | |
| Stage bọc DriveStudio: `train.drivestudio`, `eval.standard`, `export.ply_world`; approach `omnire`, `streetgs` | Xong trong code; dựng lệnh có test; **chưa chạy thật** (cần GPU); approach chưa chạy được tới khi có `prep.to_drivestudio` | |
| CLI `d2s`: `scene list/show`, `split create/list/show`, `approach list/show/check`, `run start/resume/status/list` | Xong; 138 test đạt trên Python 3.9 và 3.12 | |
| CLI: `scene ingest/validate/delete`, `split delete`, `run cancel/delete`, `report`, `export`, `cache list/gc`, `studio` | Xong | |
| CLI còn lại: `serve` | Chưa làm; làm cùng `sim/server/` | |
| Studio API (`studio/api`): cảnh, split, approach (`check` kèm lựa chọn bị khoá), run chạy nền, SSE, log, video, bảng so sánh, `/files/`; lệnh `d2s studio` | Xong; 163 test đạt trên Python 3.9 và 3.12; đã chạy thử server thật | |
| Studio API: nhận dạng và ingest nền (`/api/sources`, `/api/jobs`), huỷ, xoá, export run, xoá cảnh và split, kiểm tra lại cảnh, cache | Xong; 182 test đạt trên Python 3.9 và 3.12 | |
| Studio giao diện (`studio/frontend`): nhóm A (tổng quan cảnh, camera, ảnh đồng bộ theo khung, split trên thanh thời gian, báo cáo kiểm tra, tạo split), chạy mới (mục cấu hình bị khoá kèm lý do, xem trước stage và cache), nhóm B và C (làn đường stage theo SSE, metric theo năm nhóm, đường cong train, artifact từng stage kèm mở thư mục, video, log, chạy tiếp), nhóm D (một bảng xếp hạng có tab nhóm metric và bộ sắp xếp theo cột, chiều tăng hoặc giảm, biểu đồ PSNR theo thời gian, VRAM, tốc độ render, số Gaussian có đường Pareto, PSNR theo vùng, phân rã thời gian, đường cong train, CSV, video cạnh nhau); tiếng Việt và tiếng Anh; chế độ sáng, tối; màn hình hẹp | Xong; đã xem bằng trình duyệt trên dữ liệu mẫu (không phải dữ liệu thật) | |
| Studio giao diện làm lại: tổng quan (việc đang chạy, dẫn đầu theo PSNR, run gần đây), thêm cảnh từ thư mục gốc (ingest nền), xoá cảnh, split, run, kiểm tra lại cảnh, huỷ và chạy tiếp từ danh sách, xuất PLY và gói xem trước, Kaggle (trạng thái, đẩy, tải), lưu trữ (cache, dọn cache có xem trước), xem 3D bằng Spark (gói xem trước có phát lại xe theo khung, `scene.ply`, kéo thả file splat trên máy), tìm nhanh, thông báo nổi | Xong; đã xem bằng trình duyệt trên dữ liệu mẫu và splat tổng hợp (không phải dữ liệu thật) | |
| Studio còn thiếu: sơ đồ rig nhìn từ trên xuống và quỹ đạo xe trên bản đồ 2D (cần đọc extrinsics, pose theo cách lưu của từng dataset); depth và tách lớp; xuất hình cho báo cáo (mới có CSV); viewer chế độ chính viser (`studio/viz`, mục 2.12); hiện người và vật biến dạng bằng box trong viewer Spark; nhóm E | Chưa làm | |
| Kaggle (`kaggle/`): upload cảnh, notebook có gói job, đẩy, trạng thái, tải về, chạy tiếp qua output notebook trước; `run start --executor kaggle`; Studio `/api/kaggle` | Xong trong code; test với Kaggle giả; cờ lệnh `kaggle` đã kiểm với Kaggle CLI 2.2.5; **chưa chạy thật** | |
| `envs/main.yml`, `envs/install_main.sh`, ô cài phụ thuộc trong notebook Kaggle dùng chung script; kiểm tra môi trường (`engine/environment.py`, `GET /api/approaches/{id}/environment`, `d2s approach check`) | Đã cài thử trên máy cá nhân (Quadro T2000, CUDA toolkit 12.8); notebook Kaggle **chưa chạy thử** | |
| Stage `prep.masks_segformer`, `prep.smpl` (cache, thư mục cảnh ảo); đầu vào tuỳ chọn của stage; approach `omnire`, `streetgs` có stage prep | Xong trong code; có test với script giả; **chưa chạy thật** (cần GPU, `transformers`, `hmr2`) | |
| Tầng dữ liệu: `source inspect`; `scene ingest` cho nuScenes (gọi script có sẵn, sinh manifest); kiểm tra cấu trúc | Xong trong code; có test với dữ liệu giả; **chưa chạy thật** (cần nuScenes và `nuscenes-devkit`) | |
| Stage `export.preview` (`recon/tools/export_preview.py`: nền, từng xe, pose theo khung) | Xong trong code; test với lệnh giả; **chưa chạy thật** (cần checkpoint và PyTorch) | |
| Viewer chính (mục 2.12) trong Studio: trang Xem 3D, mode Render GPU kiểm tra máy rồi bật `tools/viewer.py --host 127.0.0.1` cho một run và nhúng vào trang (`studio/api/gpu_viewer.py`) | Xong trong code; test với viewer giả; **chưa chạy thật** (cần checkpoint thật) | |
| Bộ kiểm tra: chiếu LiDAR lên ảnh | Chưa làm; báo cáo ghi mục `projection.not_run` | |
| `scene ingest` cho Waymo, Argoverse 2, KITTI, nuPlan, PandaSet, PhysicalAI (converter chung, nhận dạng từng dataset) | Xong trong code; có test với dữ liệu giả; **chưa chạy thật** (cần dữ liệu và devkit của từng dataset) | |
| `sim/` (L2–L4) | Chưa làm | |

Mã DriveStudio đã có sẵn trong `drive2sim/recon/`: loader 7 dataset; tiền xử lý (mask, SMPL); OmniRe, Street GS, PVG, Deformable-GS; renderer 3DGS / 3DGUT / 3DGEER; rolling shutter ở `ut`/`geer`; FastGS; `tools/eval.py`; video; `tools/viewer.py`; `tools/export_ply.py`; đọc NCore V4 cho PhysicalAI.

### 7.2 Chênh lệch với thiết kế đích (đã kiểm trong code)

| Chênh lệch | Vị trí | Việc cần làm |
|---|---|---|
| Mỗi dataset một loader riêng, phần lớn là mã chép lại | `drive2sim/recon/data/loaders/<dataset>.py` | Giữ nguyên; gộp khi có điều kiện (mục 2.4) |
| Đường đọc NCore (nguồn vào) chỉ dành cho PhysicalAI: gắn cứng 7 camera, tìm file `pai_*.json`, chỉ nhận F-theta, bắt buộc có LiDAR | `drive2sim/recon/preprocess/physicalai/process.py` dòng 16, 36, 82–83, 411 | Đọc camera từ dữ liệu; thêm pinhole và fisheye OpenCV; cho phép không có LiDAR |
| Khởi tạo nằm trong dataset và trainer, bắt buộc có LiDAR | `drive2sim/recon/data/driving_dataset.py:189` (`get_lidar_samples`) | Tách thành stage `init.*`; trainer nhận `points_init` |
| Đánh giá chạy ở cuối `drive2sim/recon/tools/train.py` | `do_evaluation` trong `drive2sim/recon/tools/train.py`, `drive2sim/recon/tools/eval.py` | Tách thành stage `eval.standard` |
| Chỉ một kiểu split: test là các khung N, 2N, 3N... với mọi camera; 19 trên 21 config dataset đặt `test_image_stride: 0`, tức không có khung test | `split_train_test` trong `drive2sim/recon/data/driving_dataset.py` | Trước mắt chỉ dùng kiểu `every_nth_frame` (CONTRACTS mục 3.1); sau mới nhận split tuỳ ý |
| ~~Chạy tiếp hỏng~~ **Đã sửa, chưa chạy thử:** trước đây khi có `--resume_from` thì không lưu checkpoint nữa; checkpoint chỉ có model; nạp optimizer báo `NotImplementedError`; chạy tiếp lặp lại bước đã lưu. Nay mỗi lần lưu ghi thêm `checkpoint_resume.pth` (model, optimizer, grad scaler; ghi đè, ghi qua file tạm rồi đổi tên), chạy tiếp nạp lại optimizer và bắt đầu từ bước sau bước đã lưu; khi train xong thì xoá file này | `drive2sim/recon/tools/train.py`, `drive2sim/recon/trainers/base.py` (`state_dict`, `load_state_dict`, `initialize_optimizer`, `save_checkpoint`) | Chạy thử trên Kaggle: ngắt giữa chừng, chạy tiếp, so với run không ngắt. Số liệu tích luỹ cho densify giữa hai lần refine và trạng thái ngẫu nhiên của bộ lấy mẫu ảnh chưa được lưu |
| Config chép nguyên file: `streetgs_nocampose.yaml` khác `streetgs.yaml` 9 dòng trên 148; các bản `_ut`, `_geer`, `_fastgs` khác 10–19 dòng | `drive2sim/configs/methods/*.yaml` | Một config gốc cho mỗi method; biến thể chỉ ghi phần khác, ứng với các mục cấu hình |
| Render nằm lẫn trong viewer và trainer | `drive2sim/recon/tools/viewer.py` (2523 dòng), `drive2sim/recon/trainers/base.py` (1125 dòng) | Tách hàm render dùng riêng cho `sim/` |
| Export không có sky, SMPL, vật biến dạng; SH của xe không xoay theo xe | `drive2sim/recon/tools/export_ply.py` | Ghi rõ giới hạn; mở rộng cho `export.preview` |
| Chưa có test cho `drive2sim/recon/` (`core/` đã có 60 test) | `drive2sim/recon/` | `tests/drive2sim/recon/`, bắt đầu bằng kiểm tra hồi quy của loader |
| Phiên bản PyTorch chưa khớp: `requirements.txt` ghi `torch==2.0.0+cu117`, gsplat-geer ghi đã chạy với 2.4.1 | `requirements.txt` | Đã sửa: `requirements.txt` ghi `torch==2.4.1+cu121`, bỏ `xformers` (không dùng), `numpy 1.23.5` và `open3d 0.17.0` (bản 1.23.1 và 0.16.0 không có cho Python 3.11 của Kaggle) |

### 7.3 Nguyên tắc khi refactor

- Làm trên branch `drive2sim`; `main` cũ của teammate không bị ảnh hưởng. Thay đổi teammate làm trên `main` sau thời điểm tách được chuyển tay sang branch này.
- Mỗi bước refactor một commit riêng: bước di chuyển file chỉ đổi chỗ và sửa import, không đổi hành vi; các bước sau mới đổi hành vi.
- Sau bước di chuyển và sau mỗi bước đổi hành vi ở phần tái dựng: train lại cùng cảnh trên Kaggle, so PSNR (mục 10).
- Loader cũ và config cũ chỉ bỏ sau khi qua kiểm tra hồi quy.

### 7.4 Thứ tự làm

0. **Di chuyển mã DriveStudio vào `drive2sim/recon/`** (đã làm, ba lượt): đổi chỗ file, sửa import trong mã và chuỗi import trong config, thêm `pyproject.toml`, `drive2sim/paths.py` cho `assets/`; lượt hai tách `preprocess/` khỏi `data/loaders/`, chuyển import tương đối sang tuyệt đối, gom config method vào `configs/methods/`, docs vào `docs/datasets/` và `docs/methods/`; lượt ba chuyển `configs/` vào `drive2sim/configs/`. Sau đó tạo khung thư mục của tool và thêm submodule NCore. Đã kiểm tra tĩnh (mọi import `drive2sim...` và mọi chuỗi import trong config trỏ tới lớp có thật); **chưa chạy thật**.
1. **Kaggle chạy tiếp được:** sửa chạy tiếp (mục 7.2; đã sửa trong code, chưa chạy thử) và làm `drive2sim/kaggle/`. Chạy được OmniRe trên cảnh nuScenes mini mà teammate đã dùng; đây cũng là lần kiểm tra hành vi của bước 0.
2. **Engine và stage bọc mã hiện có:** registry, file approach có mục cấu hình, `train.drivestudio`, `eval.standard`, `render.videos`, `export.ply_world`, `run.json`, CLI `d2s`. Đạt L1 theo thiết kế; Studio nhóm A–D làm song song.
3. **Tầng dữ liệu:** `source inspect`; `scene ingest` cho nuScenes (gọi script có sẵn, sinh `manifest.yaml`); bộ kiểm tra. Sau bước này `d2s run start omnire` chạy trọn đường. Chạy thật trên Kaggle cùng cảnh nuScenes mini mà teammate đã dùng, so PSNR với đường cũ. Rồi stage `prep.masks_segformer`, `prep.smpl` ghi vào cache.
4. **Tách khởi tạo:** `init.lidar` tách khỏi dataset, rồi `init.depth`. Thí nghiệm có LiDAR so với không LiDAR là đổi mục cấu hình `init`.
5. **Config kế thừa** thay cho config chép file.
6. **sim/renderer.py** → L2 và viewer chế độ chính.
7. **sim/actors.py** → L3.
8. **sim/server/** + CARLA → L4.
9. `scene ingest` cho các dataset còn lại; converter NCore cho CARLA, COLMAP, dữ liệu mentor theo thứ tự cần dùng; `export.ncore` khi làm NuRec hoặc L4; gộp loader khi có điều kiện (mục 2.4).

## 8. Câu hỏi mở

1. Phiên bản PyTorch và CUDA của môi trường chính. Đã chọn PyTorch 2.4.1 + CUDA 12.1 (bản gsplat-geer ghi đã chạy) và ghi vào `requirements.txt`, `envs/install_main.sh`. Còn phải xác nhận ở lần chạy thật đầu tiên trên Kaggle, và hỏi teammate bản PyTorch đã dùng khi train trên Kaggle T4 trước đây.
2. Chế độ `default` của gsplat-geer có cho kết quả giống DriveStudio gốc chạy gsplat 1.3.0 không. Nếu cần baseline đúng DriveStudio gốc thì thêm một môi trường riêng.
3. Cách đóng gói môi trường (`envs/`): hiện dùng conda cho Python và một script pip dùng chung với Kaggle. Còn mở: có lưu sẵn bản đã build (wheel hoặc container) để Kaggle khỏi build lại mỗi phiên không.
4. AD stack và tiêu chí chấm điểm ở mức L4 (SPEC mục 14).
5. Định dạng dữ liệu của mentor.

## 9. License

- Repo: MIT (DriveStudio) cộng AGPL-3.0 (patch Bosch, FastGS dựa trên patch), nên phần thêm vào cũng phân phối theo AGPL.
- gsplat-geer đi kèm `3dgs-license.txt` của Inria (phi thương mại).
- `drive2sim/recon/tools/export_ply.py` dùng `plyfile` (GPLv3+).
- Weights SegFormer của NVIDIA dùng trong `drive2sim/recon/preprocess/masks/segformer_hf.py` chỉ cho dùng phi thương mại.
- NCore, proto `SensorsimService`: Apache-2.0; giữ nguyên ghi chú bản quyền khi đưa vào repo.
- viser, nerfview, Spark: MIT.
- Dữ liệu (Waymo, nuScenes...) không nằm trong repo; dataset upload lên Kaggle luôn để private.

## 10. Kiểm thử

- Kiểm tra hành vi sau khi di chuyển mã (mục 7.4 bước 0): cùng cảnh, cùng config, cùng seed, so PSNR test với kết quả trước khi di chuyển.
- Kiểm tra tầng dữ liệu: train cùng cảnh bằng đường của teammate (gọi `train.py` tay) và bằng `d2s run start`, so PSNR test (SPEC mục 13). Hai đường dùng cùng script và loader nên kết quả phải trùng, trừ sai khác do seed.
- Khi gộp loader (mục 2.4): cho loader cũ và loader chung đọc cùng cảnh giả, so từng giá trị.
- Kiểm tra hồi quy khi tách `init.lidar` và `eval.standard`: cùng cảnh, cùng seed, so với `drive2sim/recon/tools/train.py` gốc.
- Cảnh mẫu nhỏ (vài khung, 2 camera) làm fixture cho `core`, `ingest`, từng stage, `engine`.
- Kiểm tra schema cho mọi file trong `drive2sim/approaches/`, kể cả `constraints` và `presets`.
- Kiểm tra chạy tiếp: ngắt một run giữa chừng rồi chạy lại, so checkpoint và các file `done`.

