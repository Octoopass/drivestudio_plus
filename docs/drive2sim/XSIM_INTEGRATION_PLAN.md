# Kế hoạch tích hợp XSIM vào drive2sim

Ngày lập: 21/11/2026. Branch: `feat/xsim-integration`, tách từ `drive2sim` tại `cd993b7`.
Trạng thái (28/11/2026): **tạm dừng, approach `xsim` đã tắt trong tool** (ẩn, không tạo hay chạy run; code và run cũ được giữ). XSIM train được trên Kaggle nhưng T4 16 GB hết VRAM ở bước densify khoảng 7.000/40.000; xem mục 12.
Trạng thái (25/11/2026): **đã patch đủ đường local, Kaggle, Colab với đầu vào chung (`prep.observations`) và chấm điểm chung. Chưa cài môi trường, chưa chạy train/eval trên GPU, chưa chạy `prep.observations` trên cảnh thật**. Mục 11 ghi thay đổi của đợt này.

Mục 1–7 ghi phạm vi và thứ tự triển khai. Mục 8 ghi bằng chứng theo revision đã kiểm, mục 9 ghi kết quả thử và mục 10 liệt kê các điều kiện nghiệm thu.

## 1. Mục tiêu và phạm vi

Đưa XSIM vào danh sách approach của tool, đọc scene và split của drive2sim, chạy trong môi
trường riêng, ghi model/metric/video theo contract để CLI và Studio quản lý như các approach khác.
Không biến `xsimgs` thành một lựa chọn renderer của OmniRe: XSIM có scene graph, feature decoder,
trajectory và loss riêng; chỉ thay rasterizer sẽ không tái hiện phương pháp.

Mốc đầu tiên là **L1 với cả local executor, Kaggle và Colab**: tạo run, chạy tại nơi đã chọn,
theo dõi trạng thái, lưu checkpoint, render đúng tập test,
đánh giá RGB theo protocol chung và xem video/metric trong Studio. Render LiDAR là một đầu ra
riêng cần kiểm chứng; có output LiDAR không tự nâng approach lên L4. L2, actor editor L3 và
server/CARLA L4 là các mốc tiếp theo, không khai báo hỗ trợ trước khi nghiệm thu.
**Tool phải hỗ trợ chạy local trên máy đáp ứng yêu cầu GPU/môi trường. Riêng máy đang làm việc
này không dùng để cài/build/train/render XSIM thật**; chỉ điều khiển qua CLI/Studio, chạy test
nhẹ và tải/xem artifact trong đợt này. Máy chỉ điều khiển job remote không cần cài CUDA XSIM;
máy được chọn chạy local phải có môi trường XSIM và GPU phù hợp. Khả năng huỷ job remote phải
theo API thực tế của từng dịch vụ, không hứa dừng notebook nếu chỉ worker hỗ trợ SIGTERM.

Không chạy job Kaggle/Colab, cài môi trường, sửa upstream, commit hoặc push trong lượt lập plan.
Bộ artifact mentor trong `method_compare/` tiếp tục chỉ lưu local và bị Git bỏ qua.

XSIM được giữ như một codebase độc lập (pin upstream và patch có provenance), còn adapter
nằm trong package drive2sim. Người dùng chọn một approach XSIM trên cùng UI, không phải mở tool
thứ hai. Việc port renderer vào OmniRe vẫn khả thi về kỹ thuật nhưng là một hướng nghiên cứu khác,
không phải điều kiện để tích hợp full XSIM.

## 2. Căn cứ đã kiểm tra

| Nguồn | Trạng thái / kết quả |
|---|---|
| SPEC §3–7; ARCHITECTURE §2.9, §3.1, §4.6, §6–7; CONTRACTS §2, §4–6, §9.1 | Approach ngoài chạy riêng; dùng scene/split chung; mức năng lực phải đúng thực tế |
| `drive2sim/approaches/nurec.yaml`, `external/nurec/`, `stages/train/nurec.py` | Có tiền lệ external approach và môi trường riêng; đường thực tế gọi subprocess, không phải toàn bộ worker contract đã được triển khai |
| `drive2sim/stages/eval/standard.py` | Hiện phụ thuộc metric/checkpoint của trainer DriveStudio; không dùng trực tiếp cho XSIM |
| `drive2sim/engine/remote.py::require_main_codebase` | Kaggle và Colab hiện từ chối codebase ngoài DriveStudio |
| `engine/environment.py::CODEBASE_ENV` | Hiện có `drivestudio` và `external:nurec`; cần đăng ký XSIM khi implement |
| `repository/XSIM`, commit `5c86a33a036f3163f78eff5dcd8d3fc75c05d2a3` | Framework local tại `/home/lqaq/PROJECT/AI20K/PHASE03/repository/XSIM` |
| XSIM gitlinks | Ghim xsimgs `f14c15020b86fb1d88eac46809c70c67a915e654`, pytoast `6dadc763629503477e3783595258c4f737b0f290`; submodule local chưa khởi tạo |
| `repository/xsimgs`, commit `2ac35a81e915b65a9192704e8d71b773de466330` | Bản clone riêng mới hơn gitlink; chưa chứng minh tương thích với XSIM |
| XSIM `configs/waymo_default.yaml` | Không bật hai opacity; split mặc định không giữ camera test theo thời gian |
| XSIM `configs/paper/waymo_lidar_density_nvs.yaml` | Có hai opacity và `test_ratio: 0.5`, nhưng vẫn import renderer `use_phm: false` |
| `XSIMGSRenderNode.create` | Nhận `use_phm` nhưng bỏ khi gọi constructor; đã xác nhận bằng đọc code và AST, chưa test CUDA |
| `trainers/utils/checkpointer.py`; `nodes/common/scene_node.py` | Lưu `nodes.state_dict()`, **có step dạng buffer**; thiếu optimizer/scheduler/RNG, sampler ngoài nodes và buffer densify không persistent; vòng train vẫn khởi đầu biến step cục bộ từ 0 |
| `GaussianModel.load_state_dict_pre_hook` | Thay đối tượng Parameter để khớp số Gaussian; optimizer tạo trước load có thể giữ Parameter cũ; đã tái hiện cơ chế này bằng CPU |
| `Trainer.build_pipeline`; `init_loader_indices` | Init nhận toàn dataset trước khi được gắn train subset; cần giới hạn RGB ở cả init, không chỉ loss |
| `Subset.eval_iterator` / `train_iterator` | Nhánh lọc component dùng `.keys()` sai; num_workers=0 không mồi queue; đã kiểm tra cô lập bằng source AST, chưa chạy DataLoader XSIM thật |
| `engine/run.py::summary_path`; `engine/videos.py` | Summary cố định ở `eval.standard`; video có quy ước tên/layout. Chỉ thêm `eval.xsim` chưa đủ để UI/report đọc đúng |
| API runs, executor, notebook Kaggle, script Colab | Có nhiều guard và setup Python 3.10/main env; hỗ trợ remote cần sửa đồng bộ các điểm này |
| API PyTorch trong XSIM | Dùng `torch.nn.Buffer` và `DataLoader(in_order=...)`; PyTorch 2.4.1 của main env hiện không có cả hai API, không được dùng chung runtime |
| Paper Eq. 10 và `configs/nodes/losses/base.yaml` | Paper ghi L1 0.2 / SSIM 0.8; YAML là L1 0.8 / SSIM 0.2; không âm thầm gọi hai thiết lập là giống nhau |
| Phần cứng local ngày lập plan | Quadro T2000 4 GB; ổ repo còn khoảng 7.5 GB. README XSIM yêu cầu train GPU >=24 GB, inference >=12 GB; chưa chạy để xác nhận mức tối thiểu |

Paper local: `PHASE03/papers/[37]XSIM-arXiv2026.pdf` và `[37]XSIM-IJCAI2026.pdf`.
Repo upstream: https://github.com/avision-team/XSIM và https://github.com/avision-team/xsimgs .
Các ghi chú cũ `PHASE03/research/notes_xsim.md` chỉ dùng làm chỉ mục; revision trong đó đã cũ.

## 3. Phân tích lựa chọn trước khi implement

| Lựa chọn | Lợi ích | Giá phải trả / kết luận |
|---|---|---|
| Thêm XSIM làm external approach, subprocess/worker riêng | Giữ trainer, CNN và renderer gốc; không xung đột PyTorch/CUDA với môi trường chính | Thêm adapter dữ liệu và môi trường vài GB trở lên; **hướng đề xuất**, theo kiến trúc sẵn có |
| Port xsimgs vào trainer OmniRe ngay | Có thể khảo sát projection fisheye riêng | Đổi gradient, camera, densify và semantics model; không còn là full XSIM. Không chọn cho tích hợp này |
| Chạy XSIM độc lập, nhập metric/video thủ công | Nhanh để kiểm tra upstream | Không bảo đảm split/protocol/run lifecycle của tool; chỉ dùng làm phép đối chứng upstream |
| Không tích hợp | Không tăng chi phí bảo trì | Không có năng lực camera/LiDAR chung của XSIM; không đáp ứng yêu cầu mới |

Không hứa T4 16 GB chạy được chỉ vì còn quota. Máy local hiện chỉ phù hợp lint/test nhẹ;
train nghiệm thu thực hiện trên Kaggle/Colab, cần kiểm loại GPU được cấp và VRAM thực tế.
T4 16 GB không đạt khuyến nghị upstream >=24 GB; Colab GPU đủ VRAM là baseline full config
khi tài khoản được cấp. Kaggle cần cấu hình giảm bộ nhớ có đo đạc trước hoặc báo cấu hình chưa
vừa GPU, không tự đổi model/resolution rồi gọi là cùng baseline. Giới hạn máy hiện tại không
phải lý do bỏ local executor hoặc khoá khả năng chạy local trên máy đủ tài nguyên.

## 4. Kiến trúc đề xuất và vị trí file

```text
Studio / CLI (dependency nhẹ, không nạp XSIM hoặc model)
    ↓ chọn local executor hoặc Kaggle / Colab
Máy thực thi local đủ yêu cầu / runtime Kaggle / Colab
    drive2sim engine, main env
        prep.masks_segformer / prep.smpl (khi cần)
        prep.xsim → manifest/mapping/mask tương thích, cache xác định được
        train.xsim → subprocess Python XSIM → native checkpoint + trạng thái resume
        eval.xsim  → native render → scorer RGB chung → eval_report + video/layout
    ↓ artifact tại workdir; pack/pull nếu chạy remote
Studio / CLI → đọc summary/log/video; chưa bật viewer GPU/editor cho XSIM
```

Pipeline L1 dự kiến (mask và SMPL chỉ có khi preset thực sự cần):

`scene + split → prep.masks_segformer → prep.xsim → train.xsim → eval.xsim`

Nếu bật người SMPL, thêm stage SMPL phù hợp trước `prep.xsim`; cấu hình không người phải không
đòi weights SMPL. `prep.xsim` là bước đổi dạng riêng, không thêm một ingest nguồn dữ liệu mới.
Chuyển `smpl.pkl` sang dữ liệu XSIM phải kiểm joint order, shape, local/world pose và track ID;
không coi hai bên tương thích chỉ vì cùng dùng tên SMPL.
Các định danh dưới đây **là đề xuất**, phải bổ sung CONTRACTS trước hoặc cùng code triển khai.

| File / nơi sửa | Trách nhiệm |
|---|---|
| `drive2sim/approaches/xsim.yaml` | Approach `external:xsim`, L1; options/preset chỉ gồm tổ hợp đã hỗ trợ |
| `drive2sim/external/xsim/__init__.py` | Đường dẫn, revision, Python riêng, probe môi trường; không import torch khi Studio khởi động |
| `drive2sim/external/xsim/provider.py` | Provider `DatasetScene` cho dữ liệu chuẩn; mapping camera/frame/actor và đơn vị thời gian |
| `drive2sim/external/xsim/worker.py` | Nhận job/result theo CONTRACTS §9.1; dispatch train/eval, progress và huỷ |
| `drive2sim/external/xsim/training.py` | Gọi pipeline XSIM, checkpoint đầy đủ và resume |
| `drive2sim/external/xsim/evaluation.py` | Render đúng test pairs, ghi dự đoán để chấm theo protocol chung |
| `drive2sim/external/xsim/patches/` | Chỉ khi upstream cần vá; patch có revision, lý do, hash và test |
| `drive2sim/stages/prep/xsim.py` | Ghi manifest xác định được, mapping và mask đã đổi dạng; artifact đề xuất `xsim_scene`, không sửa scene nguồn và không cache khởi tạo ngẫu nhiên |
| `drive2sim/stages/train/xsim.py`, `stages/eval/xsim.py` | Adapter Stage; mỗi stage một artifact, engine quản lý artifact/done |
| `envs/xsim.yml`, `envs/install_xsim.sh` | Môi trường riêng, dependency pin, cài pytoast trước xsimgs, probe CUDA |
| `engine/environment.py`, `engine/remote.py`, API runs và hai executor | Khai báo env, kiểm codebase/backend, validation trước upload/tạo job, readiness đúng nơi chạy |
| `kaggle/templates/run.ipynb`, `colab/templates/job.sh`, bundle | Setup hai env, vận chuyển dữ liệu, cache env, pack/pull/resume đúng codebase |
| `engine/run.py` phần tìm summary; metric scorer dùng chung | Tìm báo cáo theo artifact/record thay vì tên stage DriveStudio; giữ khả năng đọc run cũ |
| `tests/drive2sim/...` cùng đường dẫn code | Test nhẹ với worker giả, test dữ liệu và test GPU riêng |

Không thêm CLI hay API riêng khi lệnh run/approach và trang Studio hiện có đã đủ. Import asset/config
qua `paths.py`. Adapter điều phối giữ Python 3.9; worker dùng Python phù hợp upstream (README
nêu 3.11–3.12), nằm trong env riêng; quy định ruff/docstring cho code viết mới vẫn áp dụng.

### 4.1 Ranh giới dữ liệu và artifact

- `prep.xsim` chạy phần chuyển đổi xác định được; khai báo phụ thuộc scene, split, masks/SMPL
  thực sự dùng. Manifest đề xuất lưu schema version, camera/frame/actor mapping, timestamp
  origin, đơn vị, camera model, timing capability, train/test pairs, digest split/config và
  chính sách LiDAR/box cho init. Tên trường cụ thể phải chốt trong CONTRACTS trước code.
- `xsim_scene` là **loại artifact mới được đề xuất**, tương tự vai trò `ncore_clip`; chưa có
  trong enum/CONTRACTS. Không ghi nhầm thành `scene_subset` vốn chỉ mô tả lựa chọn cảnh.
  `train.xsim` trả một artifact `model`, `eval.xsim` trả một `eval_report`; video và layout là
  file con của báo cáo, không trả thêm output thứ hai từ cùng stage.
- Manifest lưu tham chiếu tương đối trong workdir hoặc ID tài nguyên để worker resolve lại.
  Cache không dựa vào symlink/view hay đường dẫn tuyệt đối của máy chuẩn bị. Symlink tạm phải
  dựng lại được sau pull/restart vì template remote hiện xoá mọi symlink trong runs/cache.
- Cache split hiện chỉ định danh bằng `split_id`. Prep cần thêm digest nội dung split chuẩn hoá
  vào tham số đã resolve **trước tính cache key**; kiểm scene/converter, adapter version và mask
  provenance. Test đổi pairs nhưng giữ split_id phải miss cache. Không sửa toàn bộ cache engine
  chỉ để tích hợp một provider.
- Init ngẫu nhiên và sử dụng ảnh để gán màu diễn ra trong `train.xsim` với seed của run.
  Train split phải đi vào cả sampler lẫn init loader; cache chuyển đổi không chứa init phụ thuộc
  seed hoặc RGB test. Cho phép dùng LiDAR/box nào phải ghi trong protocol, không ngầm suy ra từ
  danh sách ảnh test.

### 4.2 Ranh giới môi trường và nơi thực thi

Mốc đầu ưu tiên giữ main env hiện có để chạy engine, `prep.masks_segformer` và `prep.smpl`,
rồi gọi XSIM bằng interpreter riêng Python >=3.11. Cách này áp dụng ở **máy chạy local hoặc
runtime Kaggle/Colab**, dùng cùng worker và contract. Máy chỉ điều khiển remote không cần cài
cả hai env; riêng máy hiện tại không cài/chạy XSIM thật trong đợt này.
Giá phải trả là thời gian cài, dung lượng và wheel cache lớn
hơn; đo trong smoke. Phương án controller nhẹ + chuyển preprocessing sang env khác chỉ làm
sau khi có số đo và thiết kế interpreter cho từng stage, vì stage hiện dùng `sys.executable`.

Không import `xsim`, `xsimgs`, `toast` hoặc torch từ module Stage/provider metadata khi registry
được nạp ở Studio. Provider native được import trong worker; metadata/preflight phía controller
chỉ dùng dependency nhẹ. Worker dùng lại engine/run lifecycle, không tự tạo run ID/thư mục
experiment timestamp khác với `ctx.output_dir`, không tự ghi artifact.json/done của engine.

Mốc đầu giới hạn **một run XSIM mỗi job**; giữ đường pair DriveStudio hiện có. Pair XSIM hoặc
mixed codebase bị từ chối rõ trước upload cho tới khi có test chia GPU/cache/env tương ứng.
GPU số 0 bên trong worker là thiết bị nhìn thấy sau `CUDA_VISIBLE_DEVICES`, không phải GPU số 0
của cả máy chủ. Chuỗi progress/result theo CONTRACTS §9.1 cần adapter thực sự đọc, vì
`run_logged` hiện chỉ lưu stdout/stderr và quản lý nhóm tiến trình.

Local executor gọi cùng Stage/worker bằng interpreter XSIM trên máy được chọn; train/eval không
được phụ thuộc đường dẫn Kaggle/Colab, bước upload hay metadata cloud. Kiểm môi trường/GPU tại
máy đó trước khi chạy local. Thiếu môi trường thì báo rõ, không tự chuyển executor hoặc bắt
người dùng chạy cloud. Các đường remote chỉ bổ sung setup, vận chuyển và quản lý phiên.

### 4.3 Ranh giới đánh giá và Studio

Tách render native XSIM khỏi scorer RGB dùng chung. Không dùng trực tiếp `eval.standard` vì
nó đọc/nạp checkpoint DriveStudio. Đề xuất sửa phần chọn summary theo stage record/artifact
`eval_report`, với quy tắc báo cáo chính xác định rõ và fallback run cũ; không thêm nhánh tên
`xsim` trong UI. Xử lý nhiều báo cáo/rerun, file thiếu hoặc stage chưa done một cách xác định,
không glob lấy file đầu tiên rồi báo thành công.

Video tuân tên `test_set_<step>_rgbs.mp4`, `..._gt_rgbs.mp4`,
`full_set_<step>_...` và `<set>_<step>_layout.json` mà `engine/videos.py` đang nhận. Ghi đúng
thứ tự camera, pairs/timestamp và vùng valid; custom split không đủ camera phải có cách ghép
được định nghĩa, không chèn ảnh thiếu rồi dùng nó để chấm điểm. Viewer GPU, export PLY/preview,
render lại và pack phải kiểm năng lực/model format tương ứng; L1 chỉ cam kết metric/video/native
checkpoint, không tự bật các thao tác dành cho DriveStudio.

## 5. Các bước thực hiện và điều kiện hoàn thành

### P0 — Chốt baseline, dependency và contract

- Kiểm kê revision, license Apache-2.0 và thành phần kế thừa; SMPL có license riêng, không đưa
  weight lên Git. Giữ copyright gốc, cập nhật ARCHITECTURE §9.
- Baseline tái lập ưu tiên XSIM + đúng các gitlink, **chưa phải bộ đã nghiệm thu tương thích**.
  pytoast chưa có source local để audit, phải kiểm sau khi lấy đúng pin. Bản xsimgs `2ac35a8` là ứng viên nâng cấp, phải kiểm
  API/build/render regression trước khi thay; không gắn hai repo khác revision rồi gọi là pin.
- Khoanh patch `use_phm` và test truyền tham số; không sửa bản nguồn tham khảo trong PHASE03.
  Lỗi nghi ngờ trong nhánh PHM CUDA cần synthetic test trước khi kết luận/vá.
- Pin thêm Python/PyTorch/torchvision, CUDA toolkit/compiler, NumPy, jetcon, plytorch,
  nvdiffrast, chamferdist, GLM và build dependencies theo bộ thực sự build được; ba commit repo
  chưa đủ làm environment lock. Wheel key chứa ABI/Python/torch/CUDA/arch và hash source/patch.
- Khoanh lỗi lọc component và queue zero-worker; không đổi mặc định workers=0 như một cách
  tiết kiệm RAM trước khi có test. Checkpoint cần test thứ tự load/thay Parameter/khởi tạo optimizer.
- Ghi khác biệt loss paper/code; preset baseline dùng YAML có provenance rõ. Chỉ đặt tên `paper`
  khi đã đối chiếu đủ; không tuyên bố tái hiện bảng paper từ quickstart.
- Cập nhật SPEC, ARCHITECTURE (quyết định, cây, bảng §3.1, tiến độ) và CONTRACTS trong commit code
  đầu tiên. Mọi field/option/error/artifact mới phải khai báo trước khi dùng.

**Đạt khi:** có danh sách pin/patch, contract đầu vào/đầu ra và ma trận năng lực được review.

### P1 — Kiểm toán dữ liệu và adapter

- Chạy phép đối chứng upstream trên một clip Waymo gốc để tách lỗi adapter khỏi lỗi kernel.
  Đây là kiểm tra upstream, không phải run chuẩn của tool hay kết quả so OmniRe.
- Đường tích hợp thực tế phải đọc scene/split chuẩn. Bắt đầu adapter KITTI-360 vì có run/camera
  kiểm chứng sẵn; kiểm calibration và projection trước khi train. Sau đó bổ sung nuScenes/Waymo
  theo cùng provider; không tuyên bố hỗ trợ mọi dataset chỉ vì schema có tên dataset đó.
- Giữ mapping frame/camera gốc, pose camera → world, quaternion wxyz, mét, us ↔ giây; test
  camera pinhole/fisheye riêng rồi mixed. F-theta và các loại chưa map đúng bị từ chối rõ ràng.
- Chuẩn hoá timestamp bằng cách trừ origin ở int64/float64 rồi mới đổi sang giây float32;
  mapping ID string → ID native phải hai chiều và ổn định, giữ riêng frame nguồn và sample index.
  Override synchronization của DatasetScene: mặc định upstream giả định mọi sensor đồng bộ và
  cùng số mẫu; Subset còn chọn LiDAR gần nhất bằng sample index, không phải thời gian.
- KITTI-360 Mei đã được ingest khớp sang OpenCV fisheye; dùng đúng intrinsics/valid angle đã
  khai báo, không áp Mei lần hai hoặc tự mở rộng FOV. Resize/crop phải đổi K, mask và pixel center
  đồng bộ; test chiếu khứ hồi với điểm sau camera và ảnh có kích thước lẻ.
- Chuyển mask theo semantics: loader XSIM argmax ba kênh human/vehicle/sky, rồi sky == 2;
  mask sky boolean/0–255 của tool không thể nối trực tiếp. Dùng nearest cho nhãn, kiểm vùng
  invalid/ego/sky riêng, không cho provider tự chạy segmentation hoặc ghi vào scene read-only.
- Đọc nguyên danh sách `(frame, camera)` train/test từ split, không dựng lại bằng test_ratio của
  XSIM. Frame test không được góp RGB vào khởi tạo màu hoặc sampler train; định nghĩa rõ phạm vi
  LiDAR/box dùng cho init và đánh giá, nhất quán với baseline, không chỉ chặn loss ảnh test.
- Kiểm tra raw camera shutter, timestamp/pose và per-ray time/origin/direction của LiDAR còn
  được giữ sau ingest không. KITTI-360 `.bin` chỉ có xyzi và timestamp mỗi khung không tự đủ
  để tái hiện lượt quét rolling shutter. Không tự bịa per-ray time bằng cách đổi tên field.
- Chốt cách dựng Sweep/range grid từ point cloud: beam/channel, azimuth/elevation, nhiều return,
  collision và phân biệt không có đo với đo không trúng vật. Không reshape xyzi thành range
  image rồi gọi đó là sensor gốc. Mốc đầu yêu cầu LiDAR hợp lệ cho đường init/supervision đang
  dùng; ảnh-only là cấu hình khác, chưa hỗ trợ nếu chưa có init và test riêng.
- Thiếu timing: cấu hình tương thích global shutter chỉ được chạy khi khai báo rõ hạn chế;
  cấu hình đòi rolling shutter phải báo thiếu metadata trước train. **PHM xử lý seam góc và có
  thể cần ngay ở scene tĩnh; không đồng nhất bật PHM với đòi rolling shutter.** Nếu cần thêm sidecar
  timing, sửa schema/converter/docs và backward compatibility trước, giữ dữ liệu cũ đọc được.

**Đạt khi:** kiểm tra chiếu LiDAR lên ảnh đúng, test chuyển toạ độ/thời gian và split không rò rỉ;
có báo cáo từng scene giữ được năng lực nào và dữ liệu nào thiếu.

### P2a — Kaggle/Colab và môi trường remote (bắt buộc cho L1)

- Cho bundle/remote setup chọn env theo codebase, với danh sách codebase hỗ trợ rõ; không chỉ
  xoá guard `require_main_codebase`. Giữ nguyên đường DriveStudio đang hoạt động.
- Validation đồng bộ API tạo run, CLI và executor; kiểm codebase, tổ hợp pair, resource/config
  trước `_prepare` có thể upload scene/cache. Không chỉ mở guard ở một đường gọi.
- Notebook Kaggle và script Colab tải XSIM/xsimgs/pytoast đúng revision, áp patch đã kiểm và
  tạo env XSIM riêng. Máy điều khiển chỉ kiểm CLI/auth và metadata, không probe CUDA local để
  quyết định khả năng chạy remote. Không dùng đường dẫn PHASE03 của máy này trong notebook.
- Bundle đúng scene/split, raw timing sidecar nếu cần, mask/SMPL và config; credential và weights
  có license riêng không đi qua Git. Ghi env/revision/patch/hardware vào provenance của run.
- Chốt bundle metadata/setup selector có kiểm allowlist, không nhận command tuỳ ý. Commit
  drive2sim phải có trên remote/branch được chọn như contract hiện tại. File patch/config native
  cần đi vào package data nếu hỗ trợ cài wheel; không chỉ chạy được ở checkout này.
- Colab env cache hiện chỉ băm install_main.sh + requirements.txt: bổ sung key XSIM/pin/patch/
  interpreter/ABI và GPU arch, không dùng marker env.done cũ để báo đã cài. Khoá setup/checkout
  dùng chung nếu phiên có job khác; không thay env đang có worker dùng.
- Thử GPU smoke trên từng backend; đánh dấu rõ cấu hình T4 tiết kiệm bộ nhớ nếu khác baseline.
  T4x2 là hai GPU riêng; không coi là một GPU 32 GB và không thêm multi-GPU khi chưa cần.
- Giữ theo dõi status/log, retry/restart, pack/pull metric/video/checkpoint; kiểm checkpoint còn
  nguyên sau ngắt phiên. Huỷ remote theo đúng khả năng dịch vụ và trạng thái quan sát được.

**Đạt khi:** worker giả kiểm đường đẩy/pull, env selection và hồi quy; job smoke thật trên
Kaggle và Colab tạo kết quả đọc được trong Studio. Nếu không được cấp GPU đủ yêu cầu thì ghi
rõ backend/config chưa nghiệm thu, không dùng test giả để báo đã chạy cloud.

### P2b — Worker và train L1

- Nối Stage/worker với local executor hiện có, dùng cùng cấu hình và artifact như remote.
  Test bằng worker giả trên máy hiện tại; nghiệm thu XSIM thật qua local executor trên máy/VM
  GPU khác đủ yêu cầu, không dùng máy này. Kiểm đường dẫn, readiness và huỷ process group.
- Tạo job/result bằng schema sẵn có; kiểm phạm vi đường dẫn và trả lỗi theo bảng hiện hành.
  Stage gọi worker bằng Python env riêng, dùng cơ chế dừng nhóm tiến trình sẵn có.
- Worker phát progress/metric/log; adapter chuyển vào `StageContext` nếu runner hiện tại chỉ
  lưu log. Không xây lại scheduler/executor của engine để chạy riêng XSIM.
- Ghi config resolved, split chính xác, revision/patch/dependency, seed, runtime và thống kê
  train. Model artifact giữ checkpoint native + CNN + sky + trajectory + config cần nạp lại.
- Giữ step đã có trong state_dict, đồng bộ step vòng train và scheduler; bổ sung optimizer,
  scheduler/RNG và trạng thái sampler/densify không được state_dict hiện tại lưu.
  Nạp topology/Parameter trước khi dựng optimizer, sau đó nạp optimizer/scheduler; kiểm các key
  bắt buộc thay vì bỏ qua `strict=False`. Ghi checkpoint atomically, phân biệt final với resume.
  Test train liên tục so với train-dừng-resume ở cùng seed; nếu không khôi phục được thì báo
  chưa hỗ trợ, không gọi nạp weights là resume đầy đủ.
- Exact resume với prefetch/multiprocessing cần giữ pending sample order hoặc checkpoint ở
  ranh giới drain queue đã định nghĩa; chỉ lưu RNG chưa đảm bảo cùng chuỗi batch. Tách warm start
  thành chế độ riêng nếu chưa bảo toàn được, không ghi đè run gốc như resume thành công.
- Huỷ/worker crash/hết VRAM không được tạo `done` thành công; giữ checkpoint hợp lệ gần nhất.
- Exit 0 nhưng thiếu/hỏng result, khác job ID, summary không finite hoặc thiếu test pairs vẫn
  là lỗi. Mất heartbeat phải phân biệt train chậm với deadlock và timeout huỷ; không đánh dấu
  failed/cancelled/succeeded chỉ từ một dòng log. Kết quả nhỏ/light pull không chứa checkpoint
  không được coi là đủ cho resume.

**Đạt khi:** test vòng đời bằng worker giả; trên GPU remote chạy smoke, huỷ theo năng lực
backend, resume, nạp lại và render.

### P3 — Đánh giá và artifact chung

- `eval.xsim` render đúng test pairs bằng native XSIM; dùng chung định nghĩa uint8, valid/ego/sky
  mask và SSIM/LPIPS của protocol tool. Tách phần metric tái sử dụng ở đúng nơi nếu cần;
  không ép metric upstream hoặc `metrics.txt` vào Summary rồi coi là protocol 0.3.
- Ghi `summary.json`/protocol/per-camera, thống kê và video final render/GT đồng bộ. Video
  fullset phải phân biệt với testset; không chấm lại điểm từ MP4 đã nén.
- Dùng đường tìm summary và quy ước video tại §4.3; test cả API run, report/compare và pull
  từ mỗi backend. Tập test rỗng, mask không còn pixel valid hoặc thiếu camera phải báo rõ,
  không ghi NaN/Inf hay metric thành công trên một phần tập mà không khai báo.
- LiDAR: lưu range/depth/point cloud và metric riêng có đơn vị, mask, ray/sweep IDs, phép đổi hệ,
  cách tính Chamfer, sampling và khoảng cách cutoff. Chỉ thêm trường Summary nếu đã cập nhật
  CONTRACTS/schema/generated types; không trộn Chamfer vào bảng RGB hay so khác protocol.
- Không xuất PLY thường rồi gọi là model XSIM đầy đủ: feature/CNN/sky/trajectory/two-opacity
  không được bảo toàn trong viewer PLY hiện tại. Preview gần đúng chỉ thêm sau và ghi rõ.

**Đạt khi:** kiểm test pixels/frame IDs, metric độc lập trên fixture và model native nạp lại cho
cùng render trong sai số đã đo; có một run thật mở được trong Studio.

### P4 — Studio, CLI và nghiệm thu L1

- Approach xuất hiện qua registry/YAML; schema/kiểu sinh tự động nếu thay đổi contract.
- Environment checks theo executor: chọn local thì kiểm env/GPU của máy thực thi local;
  chọn Kaggle/Colab thì máy điều khiển không có CUDA XSIM vẫn được đẩy job hợp lệ, readiness
  CUDA thực hiện trong runtime remote. UI giữ lựa chọn local khi approach hỗ trợ, báo rõ nếu
  máy hiện tại chưa đáp ứng. Không nạp model
  vào server Studio. Thông báo rõ dataset/camera/
  timing chưa hỗ trợ; không để env XSIM bị nhận nhầm là ready vì codebase chưa có trong map.
- Metric, log và video dùng trang sẵn có. GPU viewer hiện gắn với DriveStudio cần adapter riêng
  trước khi cho mở XSIM; không trỏ checkpoint XSIM vào viewer OmniRe. Chưa nghiệm thu L2 thì
  không hiện năng lực render tự do như đã hỗ trợ.
- Thêm demo cho loại artifact mới nếu có; chuỗi giao diện ở cả vi/en. Có sửa frontend thì
  build và commit cả dist theo AGENTS.md.

**Đạt khi:** browser smoke cho tạo run, theo dõi, huỷ, mở kết quả; cùng run thao tác được từ CLI.

### P5 — Nghiệm thu GPU và hồi quy

1. Synthetic camera và LiDAR: scene tĩnh, chuyển động đã biết, rolling shutter, seam ±π,
   Gaussian xuất hiện ở hai mép; bật/tắt PHM phải tạo khác biệt đúng. Test fisheye gần/qua 90°
   theo cả đường projection–tiling–rasterization, không suy ra chỉ từ atan2 trong một hàm.
   Kiểm nhánh PHM chỉ left hoặc right hợp lệ: biểu thức chọn nhánh có phản ví dụ tĩnh ở cả
   bản clone mới và gitlink, cần tái hiện trên GPU trước khi quyết định patch. Kiểm gradient camera-only
   so camera+LiDAR để xác nhận densify dùng đúng tín hiệu đã công bố, không tự thay thuật toán.
2. Run ngắn trên scene thật: finite loss/gradient, đúng split và actor; đo VRAM, tốc độ,
   dung lượng. Chốt số bước smoke riêng, không gọi run smoke là chất lượng cuối.
3. Run đầy đủ trên cùng scene/split/protocol với OmniRe. Giữ riêng kết quả pipeline tương thích
   và full sensor khi timing thiếu; không gán cải thiện cho một thành phần nếu đổi nhiều thứ.
4. Kiểm model nạp lại, resume, cancel, metric, video; báo cả trường hợp XSIM kém baseline.
   Kiểm riêng đường local executor trên máy/VM GPU đủ yêu cầu khác máy hiện tại; job qua
   executor Kaggle/Colab không tự chứng minh đường local đã chạy. Nếu chưa có máy phù hợp thì
   ghi local runtime chưa nghiệm thu, vẫn giữ yêu cầu hỗ trợ và test adapter bằng worker giả.
5. `make check`, Python 3.9 cho adapter và test GPU tương ứng; contract examples, env probes,
   tests NuRec/OmniRe hồi quy. Unit/integration tests tự động dùng runner giả; các lần nghiệm thu
   cloud thật chạy riêng ngoài pytest, ghi run ID, tài khoản, GPU, log và artifact.

**Đạt khi:** có báo cáo commit/config/artifact/checksums, số đo thật và các giới hạn chưa nghiệm thu.
Hiện T2000 4 GB và 7.5 GB ổ trống chưa đáp ứng baseline README; không cài/build/train nặng trên máy
này theo mặc định và không tự dọn dữ liệu để lấy chỗ.

### P6 — Mở rộng sau L1 (chưa thuộc mốc đầu)

Kaggle/Colab đã thuộc P2a và điều kiện nghiệm thu L1, không hoãn tới P6.
- L2: worker giữ native model và adapter Renderer cho pose/timestamp/camera; nghiệm thu view mới.
- L3: ActorEditor với trajectory/visibility đúng thời gian.
- L4: SensorsimService/CARLA sau L2/L3; thiết kế contract LiDAR request/response riêng nếu proto
  hiện hành không đủ. Chứng minh closed-loop bằng run thật, không chỉ video replay.

## 6. Thứ tự commit và các quyết định còn cần chốt

Mỗi bước code đi kèm docs/test, không trộn di chuyển upstream với sửa thuật toán:

1. Docs/contract + pin/license/env/probe.
2. Provider/prep + test calibration/split/timing.
3. Worker/train/checkpoint/resume cho local executor và setup remote Kaggle/Colab.
4. Eval/artifact/protocol.
5. Studio/demo và nghiệm thu GPU trên Kaggle/Colab.
6. L2–L4 thành các việc riêng sau khi mốc L1 đạt.

Các đánh đổi phải được chốt trước phần implement phụ thuộc chúng:
- Đã chốt tool hỗ trợ local, Kaggle và Colab; đợt này không chạy XSIM thật trên máy hiện tại.
  Nghiệm thu cloud ưu tiên Kaggle/Colab; đường local thật cần máy/VM GPU khác phù hợp. Cần xác
  định GPU, VRAM và dung lượng thực tế tại từng nơi chạy; không lấy phần cứng máy hiện tại làm
  giới hạn capability của approach.
- Baseline gitlink hay nâng xsimgs sau kiểm tra tương thích; không tự nâng thư viện chính.
- Mở rộng ingest để giữ timing thật hay chấp nhận bước đầu global shutter có ghi rõ hạn chế.
- Camera/dataset đủ dữ liệu nào là scene nghiệm thu đầu tiên sau kiểm toán P1.
- API/định dạng LiDAR và mức preview nếu cần mở rộng contract hiện tại.

## 7. Tiến độ triển khai

- [x] Tạo branch `feat/xsim-integration` từ `drive2sim`.
- [x] Inspect luồng hiện tại và paper/code local, ghi các khoảng trống có bằng chứng.
- [x] Lập plan, đường dẫn file dự kiến, thứ tự và điều kiện nghiệm thu.
- [x] Làm rõ phạm vi theo người dùng: L1 hỗ trợ local executor, Kaggle và Colab; riêng máy
      hiện tại chỉ điều khiển/xem/test nhẹ, không chạy XSIM thật.
- [x] Audit lại source, sửa thông tin checkpoint/PHM và bổ sung rủi ro cache, mask, summary,
      môi trường, worker, remote/resume; có probe AST/CPU cô lập, chưa có nghiệm thu XSIM runtime.
- [x] Script cài XSIM theo gitlink và patch factory/sampler, commit `f73a04c`.
- [x] Stage chỉ mục scene/split và khoá cache theo nội dung, commit `80391fd`.
- [x] Provider camera, box và sweep dạng lưới cầu theo khung, commit `edb7dce`. Kiểm tra CPU trên file Velodyne thật, còn phải nghiệm thu native runtime.
- [x] Worker train/eval/resume, checkpoint giữa hai batch, video cho Studio (commit `91213d0`–`4cda0a5`).
- [x] Đầu vào chung: `prep.observations` xuất gói quan sát từ `DrivingDataset` như `train.drivestudio`; provider XSIM chỉ đọc gói (mục 11).
- [x] Chấm điểm chung: worker render PNG 8 bit, `eval.xsim` chấm ở môi trường chính như `eval.standard`.
- [x] Môi trường: `envs/install_xsim.sh` dựng venv uv, drive2sim editable, pin gói git; `CODEBASE_ENV` có `external:xsim`; nút cài Studio.
- [x] Kaggle và Colab cài thêm venv XSIM khi run dùng approach XSIM; kiểm codebase trước khi upload.
- [ ] Cài môi trường / chạy GPU / cloud: chưa thực hiện, chờ người dùng đồng ý.
- [ ] Push branch để chạy notebook từ đúng revision.

## 8. Phạm vi audit và phát hiện từ mã nguồn

Đã lần theo các luồng liên quan sau, không tuyên bố đọc mọi dòng của cả hai dự án:

- drive2sim: docs thiết kế; registry/Stage/cache/run; external NuRec; preprocessing mask/SMPL;
  KITTI-360 ingest/calibration; train/eval; API tạo run, môi trường, metric/video; bundle,
  executor, setup template và pull/resume Kaggle/Colab; worker contract.
- XSIM: provider/sensor loader, dataset/split/sampler/init, trainer/checkpointer, scene node,
  Gaussian model/strategy, renderer/CNN và dữ liệu cấu hình liên quan.
- xsimgs: camera/projection/shutter/ray, PHM, tiling/rasterization interfaces, build dependency
  và khác biệt revision. Phép kiểm CUDA/gradient số học toàn kernel chưa thực hiện.
- Paper arXiv local: phần phương pháp, loss, phase modeling và phụ lục densification đối chiếu
  source/config. Không dùng ghi chú nghiên cứu cũ làm bằng chứng phiên bản hiện hành.

Ký hiệu nguồn trong các mục dưới:

| Ký hiệu | Nguồn đã kiểm |
|---|---|
| `D` | Checkout drive2sim hiện tại, đường dẫn tương đối từ gốc repo |
| `X` | `/home/lqaq/PROJECT/AI20K/PHASE03/repository/XSIM`, HEAD `5c86a33a036f3163f78eff5dcd8d3fc75c05d2a3`; đường mã dưới `src/xsim/` |
| `G` | `/home/lqaq/PROJECT/AI20K/PHASE03/repository/xsimgs`, HEAD `2ac35a81e915b65a9192704e8d71b773de466330` |
| `G-pin` | Git object xsimgs `f14c15020b86fb1d88eac46809c70c67a915e654`, đọc bằng `git show`, không checkout/cài |
| Paper | `/home/lqaq/PROJECT/AI20K/PHASE03/papers/[37]XSIM-arXiv2026.pdf`, Eq. 10, §3.3 và phụ lục Gaussian strategy |

XSIM pin pytoast `6dadc763629503477e3783595258c4f737b0f290`; submodule chưa khởi tạo và
chưa audit source pytoast. Bản clone G khác G-pin ở 51 file, 621 dòng thêm/511 dòng bỏ; có
thay đổi CUDA, camera timing và build interface. Không suy từ ngày commit rằng bản nào chắc
chắn tương thích hơn. Pin gitlink là điểm xuất phát tái lập, chưa phải bộ được chứng nhận chạy.

P0: chặn tính đúng đắn hoặc thực thi cơ bản. P1: cần xử lý/khai báo trước nghiệm thu. Một số
mục là lỗi source đã tái hiện, các mục còn lại là rủi ro khi nối hai hệ thống; phân biệt ngay
trong từng phát hiện.

### A01 — P0: chặn ảnh test phải áp dụng cả lúc khởi tạo

**Bằng chứng:** X `trainers/trainer.py:120–128` tạo split nhưng gọi
`prepare_initialization(self.sim_ds, ...)` trước khi thêm `init['train_split']`.
`data/reconstruction/init/init_loader.py::init_loader_indices` duyệt LiDAR toàn dataset và
camera đồng bộ từ provider, không nhận train pairs.

**Hệ quả:** chỉ thay `split_dataset` hoặc sampler chưa ngăn ảnh test được dùng để gán màu
điểm ban đầu. Đây là đường rò dữ liệu có thể xảy ra với adapter mới, không phải kết luận mọi
protocol upstream đều cấm dữ liệu mà upstream đang dùng.

**Yêu cầu:** đưa allowed train pairs vào init loader/color lookup; khai báo riêng LiDAR/box
được dùng theo protocol. Test sentinel RGB trên ảnh test và giám sát mọi lần mở ảnh trong init
và train. Không cache khởi tạo ngẫu nhiên trong prep chuyển đổi dùng chung.

### A02 — P0: môi trường chính không chạy được source XSIM hiện tại

**Bằng chứng:** X `setup.py` yêu cầu Python >=3.11; SceneNode dùng `torch.nn.Buffer`, loader
dùng `DataLoader(in_order=True)`. Python notebook/Colab của D cố định 3.10, main PyTorch 2.4.1.
Probe trên env main hiện tại xác nhận thiếu cả hai API PyTorch này.

**Yêu cầu:** interpreter/env riêng, lock dependency thực tế sau build smoke. `requirements.txt`
XSIM còn nhiều gói/git URL không pin. G `setup.py` import torch/toast/pybind11 lúc build và cần
GLM; không chỉ `pip install -r requirements.txt` rồi kết luận cài xong. Probe API trước GPU
smoke và thử import extension/render/backward thật trên remote sau build.

### A03 — P0: mở guard executor chưa đủ để chạy remote

**Bằng chứng:** D `studio/api/routes/runs.py:175`, `engine/remote.py:40`,
`kaggle/executor.py:608–610`, `colab/executor.py:541`; template và script vẫn chỉ cài main env.
Kaggle gọi `_prepare` trước guard executor; `_prepare` có thể upload tài nguyên.

**Yêu cầu:** allowlist codebase/backend và setup selector được kiểm cùng tại API/CLI/executor,
validation trước upload/job side effect. Bundle và worker phải khớp approach/run, không nhận
lệnh shell tuỳ ý. Thử rollback/thông báo khi setup lỗi, không để run trông như đã train.

### A04 — P0: summary có thể tồn tại nhưng UI/report không thấy

**Bằng chứng:** D `engine/run.py:932–953` cố định `eval.standard/summary.json`.
`load_summary` được cả API và report dùng. `eval.standard` lại có đường nạp/render checkpoint
DriveStudio; không dùng nguyên stage này để chạy XSIM.

**Yêu cầu:** native eval riêng, scorer chung và cách chọn báo cáo chính theo stage/artifact
chung, có fallback run cũ. Chốt hành vi nhiều eval/rerun và stage chưa hoàn thành; không thêm
nhánh `if xsim` trong UI và không tạo file summary giả ở stage không chạy.

### A05 — P0: resume cần nhiều hơn weights, và có bẫy Parameter/optimizer

**Sửa thông tin plan cũ:** X `modeling/nodes/common/scene_node.py:60` lưu `step` là persistent
buffer, nên weights **có step**. Tuy nhiên `trainers/trainer.py:152` bắt đầu step cục bộ từ 0;
`Checkpointer.save` chỉ lưu nodes.state_dict; optimizer/scheduler không phải state module,
sampler ngoài nodes, ba buffer gradient/radii của strategy có `persistent=False`.

**Bằng chứng thêm:** `modeling/gaussian/model/model.py:26–48` thay Parameter để khớp topology.
CPU probe gọi chính hook cô lập xác nhận optimizer tạo trước load vẫn giữ Parameter cũ, và
`optimizer.step()` không cập nhật Parameter mới có gradient. Đây là lỗi cơ chế nếu nối resume
theo thứ tự đó, chưa phải phép thử full resume XSIM.

**Yêu cầu:** nạp topology/weights trước dựng optimizer, rồi state optimizer/scheduler/step,
RNG và strategy/sampler. Với prefetch, lưu order chờ hoặc định nghĩa điểm drain queue. Kiểm key
bắt buộc, ghi atomic và không chọn checkpoint ghi dở. Phân biệt inference load, warm start và
resume; run thay split/config/patch không được tiếp tục như cùng thí nghiệm mà không kiểm tra.

### A06 — P0 với preset PHM: cờ bật đang bị bỏ và có phản ví dụ chọn nhánh

**Bằng chứng đã tái hiện:** X `nodes/render/xsimgs_renderer.py:20–39` nhận `use_phm` nhưng không
truyền vào constructor; AST probe truyền True không thu được keyword đó ở constructor.

G và G-pin `include/cameras/ut_phm_kernels.h:105–113` dùng
`use_left = result_left != 1`. Khi central không hợp lệ, chỉ left hợp lệ (`1,0` hoặc `1,2`),
code chọn right; khi chỉ right hợp lệ thì chọn left. Đã đối chiếu cả git object G-pin; chưa
chạy kernel để đo ảnh hưởng hay xác nhận tổ hợp input nào đi đến nhánh đó trên sensor thật.

**Yêu cầu:** test factory tới runtime flag, rồi synthetic seam ±π trên GPU cho central/left/
right/both/none, chuyển động, forward/backward và tiling. Không chỉ vá factory rồi tuyên bố
đã có full PHM; patch kernel chỉ sau khi có ca tái hiện CUDA và review semantics.

**Sửa plan cũ:** PHM xử lý seam góc cả với scene tĩnh; thiếu rolling shutter timing không có
nghĩa mọi cấu hình PHM đều phải bị khoá. Capability timing và PHM phải xét riêng.

### A07 — P0 khi dùng nhánh đó: eval components sai và zero-worker có thể chờ mãi

**Bằng chứng:** X `data/reconstruction/sampling/subset.py:102` duyệt `cur_index.keys()` rồi
unpack thành k,v. AST probe với `components=['image']` ném
`ValueError: too many values to unpack (expected 2)`.

`Subset.train_iterator:130–137` mồi queue bằng `workers * 3`, rồi chỉ enqueue tiếp sau khi nhận
batch; `ErrorBasedIndicesProvider.__getitem__` dùng blocking `mp_queue.get()`. Probe loader
giả xác nhận workers=0 đọc queue rỗng trước mọi lần put. Chưa chạy DataLoader XSIM thật;
không để probe tự treo hoặc coi đây là đo thời gian deadlock.

**Yêu cầu:** regression riêng cho components unset/empty/subset và workers 0/1/nhiều, preloaded
data, clip ngắn; watchdog/cancel khi không có batch đầu. Không chỉnh mặc định sang workers=0
để giảm RAM trước khi giải quyết đường này.

### A08 — P1: ID và đồng bộ sensor không thể suy ra từ số khung

**Bằng chứng:** X `DatasetScene.camera_lidar_synchronization` mặc định dùng cùng sample index
cho mọi sensor và giả định bằng số mẫu. `Subset:48–60` tìm LiDAR trong subset bằng khoảng cách
sample index, không timestamp. Upstream cũng có utility synchronization theo thời gian; cần
chọn/override đúng cho provider mới, không chép mặc định BasicScene.

**Yêu cầu:** mapping hai chiều frame nguồn/sample index/camera/actor; kiểm sensor lệch tần số,
drop frame, timestamp lặp/không tăng, khoảng ghép tối đa và biên clip. Trừ timestamp origin
ở int64/float64 trước khi float32 để không mất độ phân giải microsecond. Track vắng/không
actor/không LiDAR/trajectory một keyframe phải được hỗ trợ hoặc từ chối trước train.

### A09 — P1: mask và calibration có semantics khác

**Bằng chứng:** X `data/loaders/image_loaders.py:97–100` argmax ba kênh; dataset:337 đọc
`sky == 2`. D prep xuất sky mask riêng, không phải segmentation RGB ba kênh XSIM. Nối thẳng có
thể làm mất sky mà shape vẫn hợp lệ. Native provider Waymo có thể tự sinh segmentation;
adapter tool không được tự ghi vào scene read-only hoặc tải model ngoài bước được khai báo.

D `ingest/kitti360.py::camera_intrinsics` đã fit Mei sang OpenCV fisheye và ghi valid angle.
Map sai mô hình, distort hai lần, đổi resize nhưng giữ K hoặc nội suy nhãn bilinear đều gây
sai phép chiếu/mask mà không nhất thiết crash.

**Yêu cầu:** fixture mask có các lớp và invalid/ego độc lập; xác nhận thứ tự kênh/polarity,
resize nearest cho nhãn; projection round trip pinhole/fisheye/mixed và valid angle ingest.
SMPL cần map định dạng/khớp/hệ toạ độ riêng; không rename `smpl.pkl` rồi dùng thẳng.

### A10 — P1: dữ liệu sensor bị mất không thể phục hồi bằng adapter

Scene chuẩn không tự bảo đảm có shutter start/end hoặc từng ray LiDAR time/origin/direction.
KITTI-360 xyzi và timestamp khung không chứng minh có rolling scan metadata.
X `BaseRenderNode.sweep_rays` có nhánh dùng measured ray và fallback ray lý tưởng; phải biết
đang dùng nhánh nào thay vì báo full sensor chỉ vì tensor đã đủ shape.

**Yêu cầu:** manifest capability, preflight theo preset, protocol global-shutter tương thích
được ghi rõ; nếu cần raw timing sidecar thì mở schema/converter/bundle trước. Range dọc ray
và depth camera-plane phải có đơn vị/semantics khác nhau, cùng mask/alpha threshold hợp lệ.
Chốt range grid/beam/azimuth/elevation, collision/multi-return và mask đo: point cloud xyzi
không tự là Sweep có cấu trúc. Không coi ô chưa có phép đo là một ray đo không trúng vật.
Đường init/supervision hiện tại cần LiDAR; scene ảnh-only phải bị chặn cho tới khi có cấu hình
init và kiểm chứng riêng, thay vì sinh tensor LiDAR rỗng rồi hy vọng trainer tự thích nghi.

### A11 — P1: cache và artifact không được mang phụ thuộc máy trước

**Bằng chứng:** D `engine/cache.py:61–70` dùng `split:<split_id>`, không băm pairs. Đây là rủi ro
nếu nội dung split cùng ID bị thay/chuẩn bị lại, không phải kết luận đường CLI hiện cho phép
ghi đè tùy ý. Template Kaggle/Colab xoá mọi symlink trong runs/cache sau job.

**Yêu cầu:** split digest trong params trước cache lookup; adapter/stage version và input
artifact keys đầy đủ. Manifest/cache dùng đường tương đối hoặc ID để resolve lại; view/symlink
tạm dựng lại được. Thử giải nén/pull sang workdir khác, chưa có checkout/PHASE03 của máy cũ.
Native config cần vừa có provenance resolved vừa có mapping path để relocate, không chỉ
config tuyệt đối hoặc `${env:...}` phụ thuộc môi trường đã mất.

### A12 — P1: cache env, chia GPU và remote lifecycle

**Bằng chứng:** D `colab/templates/job.sh:40` chỉ băm install_main.sh/requirements.txt.
Template Kaggle pair gán `CUDA_VISIBLE_DEVICES=i`. Bundle thực tế gồm run/split/scene_ref/
resume, chưa có cơ chế tự chọn XSIM env. Main env preprocessing gọi `sys.executable`.

**Yêu cầu:** key env/wheel gồm pins/patch/Python/torch/CUDA/arch; setup lock không thay checkout
hoặc env đang được worker khác dùng. Không cộng VRAM T4x2 thành một GPU 32 GB. Phân biệt cancel
worker với dừng notebook/VM; mất phiên, retry, output cũ và stage rerun có state hợp lệ.
Light pull không có model không được xác nhận resume-ready. Chỉ commit đã có trên remote/branch
được chọn mới chạy được notebook theo contract hiện tại.

### A13 — P1: worker và Studio có thể báo thành công sai

**Bằng chứng:** D `stages/commands.py::run_logged` lưu log và dừng process group; chưa parse
JSON event thành StageContext. `engine/environment.py` coi codebase không có trong map là
không cần GPU env; XSIM chưa đăng ký sẽ có thể trông như ready. Video_sets có regex và layout
riêng; viewer/export còn phụ thuộc model DriveStudio.

**Yêu cầu:** đối chiếu job/result ID, exit code, required outputs, test pair coverage, metric
finite và trạng thái done. Exit 0 thiếu result vẫn lỗi; tránh đọc result cũ khi retry. Stream
event không được trộn traceback/log thường rồi parse như JSON tuỳ tiện. Readiness theo
executor: chạy local phải có env/GPU ở máy đó, còn máy chỉ đẩy job remote không cần CUDA.
Giữ lựa chọn local trên UI, báo thiếu tài nguyên đúng máy; capability gate viewer/render/export.
Chạy video theo tên/layout đã hỗ trợ; không sửa UI theo tên approach cho từng thao tác.

### A14 — P1: tái lập paper và công bằng metric chưa được chứng minh

Paper Eq. 10 đặt L1 0.2, SSIM 0.8; X `configs/nodes/losses/base.yaml` đặt 0.8 và 0.2.
Paper phụ lục ghi densify tích luỹ gradient từ RGB. Trainer tổng loss rồi backward chung;
`Eval3DStrategy:128` đọc `model.positions.grad`, trong khi `GaussianSceneNode.color_render`
chỉ chọn camera cho render info. Đây là **điểm nghi ngờ cần thí nghiệm tách gradient**, chưa
đủ để kết luận tái hiện GPU sai hoặc tự đổi loss/densification của upstream.

**Yêu cầu:** baseline đặt tên theo config/revision đã dùng. Test camera-only so camera+LiDAR
cho tín hiệu densify. Metric chính dùng lossless prediction, cùng uint8/mask/SSIM/LPIPS,
resolution và aggregation per-camera như protocol tool. Test rỗng, mask rỗng, ảnh thiếu/NaN
phải lỗi hoặc được ghi rõ theo contract, không silently chấm phần còn lại. LiDAR metric/range
và cutoff/sampling giữ protocol riêng. PLY preview không chứa đủ feature/CNN/trajectory để
gọi là model XSIM đầy đủ.

## 9. Các kiểm chứng đã chạy và giới hạn

Không cài package, khởi tạo submodule, build extension, gọi Kaggle/Colab hoặc dùng CUDA trong
lượt audit. Các probe dùng AST lấy đúng hàm hiện tại, thay dependency bằng fixture, hoặc dùng
torch CPU có sẵn. Vì vậy kết quả là bằng chứng cơ chế, không phải full integration test.

| Nhóm probe | Kết quả thực tế |
|---|---|
| Factory PHM | Truyền True nhưng keyword không tới constructor |
| Eval components | Tái hiện ValueError bằng body hàm gốc |
| Queue workers=0 | Phát hiện lần đọc đầu khi chưa enqueue; fixture chủ động ném lỗi để không treo |
| PHM nhánh một phía | 4 phản ví dụ logic; xác nhận source ở G và G-pin; chưa chạy kernel |
| Load hook/optimizer | Parameter mới có gradient nhưng optimizer cũ không cập nhật nó |
| Main env API | Torch `2.4.1+cu121`: không có `nn.Buffer` và DataLoader `in_order` |

Script tái hiện trong phiên audit: `/tmp/xsim_integration_audit_probes.py`, chạy bằng
`/home/lqaq/miniconda3/envs/drive2sim/bin/python`; file tạm, không phải test đã thêm vào repo.
Khi implement, chuyển từng phép thử thành regression test ở vị trí tương ứng, không phụ thuộc
đường PHASE03 tuyệt đối của máy này. Paper được trích văn bản bằng `pdftotext -layout` để
đối chiếu Eq. 10 và phụ lục; không suy kết quả chất lượng từ số liệu paper.

## 10. Ma trận nghiệm thu L1

| Cổng | Ca kiểm chứng cần có | Loại bằng chứng |
|---|---|---|
| Source/env | Đúng pins/patch, import khi cài wheel, build pytoast→xsimgs, API torch | Test nhẹ + GPU remote |
| Registry | Studio discover approach khi local không có torch/XSim/CUDA | CPU/import test |
| Local executor | Cùng Stage/worker, không phụ thuộc cloud; readiness, path, cancel/resume và artifact | Worker giả tại máy này + XSIM thật trên máy/VM GPU khác |
| Prep/cache | Same ID khác pairs, đổi calibration/mask/adapter, relocate cache, read-only scene | Fixture CPU |
| Split/init | Sentinel ảnh test không được mở cho train/init, custom sparse pairs, subset camera | Fixture + smoke |
| Sensor | IDs rời rạc, timestamps lớn, tần số lệch, gap/drop, K/pose/shutter/mask, empty actors | Fixture + projection GPU |
| Sampler | Workers 0/1/nhiều, preload, component filter, empty split, all-zero error weights | Regression + smoke |
| PHM | Cờ tới kernel, seam left/right/both/none, tĩnh/chuyển động, forward/backward | GPU synthetic |
| Train/resume | Liên tục so dừng/nạp lại; topology đổi; optimizer bind; pending batch/strategy/step | CPU cơ chế + GPU |
| Worker | Exit 0 thiếu result, result cũ/sai ID, crash/OOM/SIGTERM/timeout, checkpoint ghi dở | Worker giả + GPU |
| Remote | API/CLI cùng validation; pair reject trước upload; setup key; restart/pull/rerun | Client giả + cloud riêng |
| Eval | Scorer độc lập, đúng test coverage/pixels, invalid/ego/sky, depth semantics | Fixture + GPU render |
| Studio/report | Đọc eval_report mới và run cũ; nhiều report/rerun; video layout; capability gates | API + browser smoke |
| Artifact | Pull full/light, pack và nạp model tại workdir khác; hash/config/dependency đủ | Fixture + remote reload |
| Chất lượng | Cùng scene/split/protocol với OmniRe; VRAM/thời gian/dung lượng và chất lượng | Run đầy đủ remote |

Chỉ nghiệm thu cloud khi có run ID/GPU/log/artifact thực; test giả không đáp ứng cổng đó.
Đường local executor cũng cần chạy thật trên máy/VM GPU phù hợp khác máy hiện tại; chạy qua
executor Kaggle/Colab không thay cho bằng chứng local. Chưa có bằng chứng thì ghi chưa nghiệm
thu local runtime, không bỏ yêu cầu hỗ trợ local của tool.
Theo README hiện tại, train khuyến nghị >=24 GB, inference >=12 GB. T4 16 GB không tự đạt
baseline; cấu hình giảm bộ nhớ cần tên/provenance và số đo riêng. Đây là thông tin tài liệu,
không phải mức VRAM tối thiểu đã đo trong audit.

## 11. Thay đổi ngày 25/11/2026: đầu vào và chấm điểm chung

Review ngày 25/11 thấy adapter tự đọc lại thư mục cảnh KITTI-360 nên lệch loader DriveStudio:
mask mắt cá thiếu viền tối và thân xe, ảnh thu nhỏ bằng nearest thay vì bilinear, độ phân giải
mặc định khác OmniRe, lưới LiDAR ±90° bỏ khoảng 2/3 số điểm, fingerprint băm cả mã chấm điểm,
drive2sim cài không editable trong venv XSIM, bảng kiểm môi trường báo XSIM sẵn sàng khi chưa
cài. Đợt sửa:

| Phần | Trước | Sau |
|---|---|---|
| Đầu vào | `prep.xsim` (chỉ mục) + provider đọc file ingest KITTI-360 | `prep.observations` dựng `DrivingDataset` bằng đúng câu lệnh `train.drivestudio` (method OmniRe, renderer để che pixel mắt cá), xuất gói `observations`; `BundleScene` chỉ đọc gói, mọi dataset loader DriveStudio đọc được |
| Mask | Chỉ góc ống kính | `masks/train_*` = `egocar_masks` của loader (góc, viền tối, thân xe); `masks/eval_*` = `valid_masks` |
| Độ phân giải | XSIM tự thu ảnh, nearest | Ảnh 8 bit đúng như loader giữ (`downscale_when_loading` của dataset, như OmniRe); XSIM không resize |
| LiDAR | Lưới 128×2048 phủ ±90°; ô trống bị coi là có đo | Dải góc nâng đo từ dữ liệu, bước 0,2°; ô trống có gốc tia 0; chính sách `all`/`train`; in tỉ lệ điểm giữ |
| Actor | Lớp KITTI tự map, không SMPL | Loại node của DriveStudio → loại box XSIM; SMPL từ `prep.smpl`/`prep.smpl_release` đổi bằng toast như XSIM gốc; mô hình SMPL chuyển sang `.pth` trong thư mục cài XSIM, đặt `SMPL_PATH` cho worker |
| Chấm điểm | Trong venv XSIM, ảnh thật đọc riêng | Môi trường chính, ảnh thật và mask của gói, `compute_image_metrics`, LPIPS torchmetrics như `eval.standard` |
| Fingerprint | Mọi `*.py` của adapter, cả tham số runtime | Chỉ module quyết định model và tham số model |
| Môi trường | `pip install -e . "$root"` (drive2sim không editable), gói git không chốt | venv uv trong `D2S_XSIM_DIR`, `-e` cho cả hai, `envs/xsim-constraints.txt`; kiểm môi trường thật; nút cài Studio |
| Remote | Kaggle, Colab từ chối XSIM | Cài thêm venv XSIM (Kaggle: ô riêng; Colab: tham số `xsim`, cache theo khoá riêng) |

Giữ nguyên mã gốc XSIM ở revision chốt và bản vá `runtime.patch`; submodule pytoast (gitlink `6dadc76`) có thêm `pytoast.patch` (27/11/2026): `sample_trajectory` của backend torch gán `out_quats[invalid_mask, 0] = 1` với mask bool nhiều chiều, PyTorch báo `IndexError` ở cả SMPL lẫn `init_loader` của XSIM; bản vá thay bằng `torch.where` cho cùng kết quả (đã thử trên CPU với hai dạng đầu vào đó). Khác cấu hình gốc có chủ ý:
`undistort_images` tắt (xsimgs vẽ thẳng pinhole có méo và mắt cá), đọc mẫu đồng bộ từng bước thay
cho hàng đợi prefetch (để checkpoint giữa hai batch). Chưa có bằng chứng chạy thật: gói quan sát
trên cảnh thật, build xsimgs, train, render và chấm trên GPU đều chưa chạy.

### Rà soát trước khi commit (25/11/2026)

Rà soát tĩnh và kiểm CPU phát hiện, đã sửa: worker không đặt `SMPL_PATH`; `dynamic: true` làm cảnh
không box lỗi ở stage mask; Studio từ chối nút cài `xsim`; bảng kiểm môi trường bỏ qua môi trường
chính; `D2S_XSIM_PYTHON` bị khoá vì thiếu mã nguồn ở thư mục mặc định; chấm điểm nạp PyTorch trong
tiến trình engine; cài lại sau khi đổi bản vá bị kẹt; `uv` không nằm trên PATH; người không có
SMPL ở keyframe nào vẫn thành node SMPL; ô notebook thiếu `id`; hard link trỏ vào symlink; module
stage import numpy làm `d2s` hỏng trên máy nhẹ (test Python 3.9 bắt được); mô hình SMPL bị chép
vào cache.

Đo trên 40 sweep thật của `kitti360_0003_local`: góc nâng −24,6° tới 7,9°; lưới mới 164×2048 giữ
91,4% điểm, lưới cũ 128×2048 phủ ±90° giữ 30,4%. `convert_smpl.py` chạy trên file SMPL thật cho
kết quả mà `load_smpl_model_data` của XSIM đọc được (6890 đỉnh, 24 khớp).

Còn mở, chưa sửa:

- Khởi tạo màu điểm: sweep ở khung test không có ảnh train cùng khung nên điểm của chúng không
  được tô màu từ ảnh, XSIM nội suy từ điểm lân cận. XSIM gốc tô bằng mọi ảnh.
- Loss L1, SSIM của XSIM nhân mask rồi lấy trung bình trên cả ảnh: vùng che lớn (mắt cá) làm
  trọng số loss ảnh giảm so với loss LiDAR.
- Khoá cache của `prep.observations` không gồm nội dung config method, dataset và mã loader: đổi
  chúng thì phải tăng `version` của stage hoặc xoá cache.
- `summary.json` của XSIM chưa có thời gian train, số Gaussian, VRAM train và PSNR theo vùng
  người, xe, vật động.


## 12. Chạy thử trên Kaggle và quyết định tạm dừng (26–28/11/2026)

Hai run trên PHENIKAA `residential-area` (200 khung, 6 camera, có người và box): `every10_fisheye2`
(run `20261126-153154-xsim-9cac`) và `every10_pinhole4` (`20261126-153155-xsim-64ad`), mặc định
40.000 bước, Kaggle T4.

### 12.1 Lỗi đã sửa, theo thứ tự gặp

| Lỗi | Nguyên nhân (đã đo) | Sửa |
|---|---|---|
| Cài venv XSIM hỏng ở bước kiểm import | Kernel Jupyter đặt `MPLBACKEND` inline, venv XSIM không có backend đó | `75c6b8f`: `MPLBACKEND=Agg` cho installer, worker, probe |
| `prep.smpl`: `Unknown dataset phenikaa` | PHENIKAA có trong danh sách SMPL nhưng `smpl/run.py` không có nhánh | `879e2aa`: dùng phép chiếu box của KITTI trên camera pinhole của cảnh (`--cameras`) |
| `prep.observations` bị giết (mã `-9`) | Hai run chung một máy hết RAM | Đẩy mỗi run một notebook |
| `train.xsim`: `relative_to` sai gốc | Cảnh đọc qua thư mục ảo `.runtime/<run>/scene_view/000`, worker suy workdir từ cảnh | `a06ea2a`: `StageContext.workdir` |
| Chạy lại không dùng output lần trước | Kaggle bỏ `kernel_sources` lỗi; notebook kết thúc `error` khi run lỗi | `4f9c242`: notebook kết thúc `complete`, chỉ gắn nguồn `complete` |
| `IndexError` của pytoast khi nạp SMPL | `out_quats[invalid_mask, 0] = 1` với mask bool hai chiều | `f6c3eb8`: gọi từng khớp (adapter) |
| Config gốc đòi `WAYMO_ROOT_PATH` | `provider.yaml` dùng biến môi trường bắt buộc, adapter không dựng provider Waymo | `0cf5106`: giá trị giữ chỗ |
| Cùng `IndexError` trong `init_loader` của XSIM | Như trên, ở chỗ gọi của XSIM | `19e3d5b`: `patches/pytoast.patch` áp vào submodule trước khi build |
| Chạy lại từ bước 1 dù có checkpoint | Kaggle gắn output ở `input/notebooks/<user>/<tên>/`, code chỉ xét thành phần đầu nên chọn output cũ nhất | `ab087e3` |

### 12.2 Kết quả: train được, hết VRAM

- Sau các sửa trên, `train.xsim` chạy: loss 2,36 ở bước 1 xuống khoảng 0,36–0,43 ở bước 7.000–7.800.
- Cả hai split hết VRAM của T4 (`total capacity of 14.56 GiB`) **trong lúc densify**
  (`eval3d_strategy.densify` → `split_by_mask`): `pinhole4` ở bước 7.000, `fisheye2` ở 7.800; lặp lại
  ở lần chạy sau (cùng bước). Tốc độ khoảng 0,9 giây mỗi bước, 40.000 bước khoảng 10 giờ.
- Checkpoint ở khoảng bước 7.000 nặng 2,5 GB (trọng số, trạng thái Adam, bộ đệm sampler).
- Đẩy với `--accelerator NvidiaL4` vẫn chạy trên `Tesla T4` (log `nvidia-smi`, `TORCH_CUDA_ARCH_LIST=7.5`):
  tài khoản không được cấp L4 (chưa kiểm vì sao).

### 12.3 Vì sao nặng

- Chiến lược densify của cấu hình gốc (`nodes/gaussian/strategy/eval3d_static.yaml`): densify tới
  bước 20.000, reset độ mờ mỗi 4.000 bước rồi nghỉ 1.000 bước, **không có trần số Gaussian**. Từ bước
  5.000 densify chạy lại và số Gaussian tăng tới khi hết bộ nhớ. XSIM có chiến lược MCMC có trần
  (`mcmc_static` 5 triệu, `mcmc_rigid` 2 triệu, `mcmc_deformable` 300 nghìn) nhưng cấu hình mặc
  định không dùng.
- Mỗi Gaussian có thêm 13 kênh đặc trưng (`features/raw.yaml`) cho decoder CNN; Adam giữ hai mô-men
  cho mọi tham số.
- Ảnh pinhole 960×768, mắt cá 960×540 (thu nhỏ 2 lần); khởi tạo từ LiDAR của cả 200 sweep.
- README XSIM: train cần GPU từ 24 GB, suy luận từ 12 GB.

### 12.4 Quyết định (28/11/2026)

Không dùng XSIM trong tool ở giai đoạn này. Approach `xsim` bị tắt bằng trường `disabled`
(CONTRACTS mục 4.1): ẩn khỏi Studio, CLI, Trợ lý; tạo, chạy tiếp, xuất lại, đẩy run và cài môi trường
riêng đều từ chối. Code, bản vá, test adapter và run cũ được giữ.

XSIM là **method** (thuật toán của bài báo, codebase ngoài) được tool bọc thành **approach** (chuỗi
`prep.masks_segformer` → `prep.smpl` → `prep.observations` → `train.xsim` → `eval.xsim`); khác OmniRe,
approach không chỉnh được bên trong method (renderer, densify). Method này hợp với **mô phỏng cảm
biến**: log nhiều camera và nhiều LiDAR, cần render lại cả camera lẫn LiDAR và sửa được cảnh (vật động
có quỹ đạo, dùng cho mô phỏng vòng kín), rig trộn pinhole và mắt cá, có GPU từ 24 GB. Không hợp khi
chỉ cần ảnh camera từ góc mới (OmniRe, PVG nhẹ hơn, đã có số), GPU dưới 24 GB, hoặc không có LiDAR.
Tool hiện chỉ đánh giá RGB; xuất LiDAR của XSIM chưa nối.

### 12.5 Để bật lại

1. Có GPU từ 24 GB (vast.ai RTX 4090, Colab có trả phí) hoặc chấp nhận cấu hình khác bản gốc (MCMC
   có trần, hoặc `downscale=4`) và ghi rõ trong tên preset, kết quả.
2. Bỏ dòng `disabled` trong `drive2sim/approaches/xsim.yaml`.
3. Đẩy lại hai run trên; chúng chạy tiếp từ checkpoint khoảng bước 7.000 (output `64ad-7`, `9cac-6`
   trên Kaggle) nếu output còn.
