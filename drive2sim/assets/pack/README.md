# Kết quả dựng cảnh bằng drive2sim

[English below](#english)

Gói này dùng để mở lại các mô hình cảnh đã huấn luyện, xem ảnh và video, rồi so sánh kết quả giữa các cấu hình. Bạn không phải huấn luyện lại để xem chúng. Phần dữ liệu đã có trong gói; phần mềm drive2sim và các thư viện chạy nó cần được cài trên máy nhận.

Bộ kết quả tham chiếu ở đây gồm **một đoạn đường nuScenes `scene-0061`, được dựng lại bằng 11 cấu hình**. Một lần huấn luyện với một cấu hình được gọi là **run**; mỗi run có mô hình đã học riêng. Các run dùng chung dữ liệu của đoạn đường để có thể so sánh với nhau. Danh sách run thực sự có trong file tar nằm trong `pack.json`.

## Mở kết quả lần đầu

Nếu nhận kèm file `drive2sim-0.1.0-py3-none-any.whl`, bạn có thể cài bản tool đó bằng các lệnh sau trên Linux. File `.whl` là gói cài phần mềm Python. Cách cài nhẹ này dùng [uv](https://docs.astral.sh/uv/) và cho phép xem kết quả mà không cần GPU NVIDIA hoặc CUDA:

```bash
uv venv ~/.venvs/drive2sim
uv pip install --python ~/.venvs/drive2sim/bin/python './drive2sim-0.1.0-py3-none-any.whl[studio]'
```

Chạy hai lệnh trên từ thư mục chứa file wheel. Nếu tên wheel bạn nhận khác, thay đúng tên đó. Bạn cũng có thể cài từ mã nguồn theo [README của dự án](https://github.com/thaianh20021/drive2sim).

Tiếp theo, giải nén dữ liệu vào một **thư mục làm việc** trống. Tool sẽ đọc cảnh và kết quả bên trong thư mục này. Trong ví dụ dưới đây, `results.tar` là tên minh hoạ: thay bằng tên tar bạn thực sự nhận, chẳng hạn `results-nuscenes-11runs.tar`.

```bash
mkdir drive2sim-workdir
tar -xf results.tar -C drive2sim-workdir
~/.venvs/drive2sim/bin/d2s studio --workdir "$PWD/drive2sim-workdir"
```

Mở địa chỉ mà lệnh cuối in ra. Trong Studio, trang **Các run** liệt kê các lần thực nghiệm; mở một run để xem kết quả của nó. Trang **So sánh** đặt các chỉ số của nhiều run cạnh nhau. Trang **Xem 3D** cho chọn mô hình để xem cảnh và phát chuyển động theo thời gian.

Nếu đã cài tool trong một môi trường khác, dùng lệnh `d2s` của môi trường đó thay cho đường dẫn `~/.venvs/drive2sim/bin/d2s`. Luôn chỉ đúng thư mục dữ liệu bằng `--workdir`; việc đứng trong thư mục mã nguồn không tự làm tool tìm được gói bạn đã giải nén ở nơi khác.

## Chọn cách xem cảnh

**Gói xem trước trong trình duyệt** dùng dữ liệu 3D đã xuất sẵn, cho phép đổi góc nhìn và phát chuyển động mà không cần môi trường CUDA. **Video** là ảnh đã render từ camera gốc; bạn có thể phát nhưng không xoay góc nhìn của video. **Render GPU** nạp mô hình đã học và tạo ảnh mới bằng renderer của mô hình, nên cần môi trường GPU với PyTorch, CUDA và các thư viện dựng ảnh.

Để dùng Render GPU, xuất lại hoặc huấn luyện, cài môi trường GPU theo README của dự án rồi cài wheel vào chính môi trường đó. Nếu dùng môi trường conda tên `drive2sim`, sau khi cài xong có thể chạy từ thư mục dữ liệu đã giải nén:

```bash
conda activate drive2sim
d2s studio --workdir /duong/dan/drive2sim-workdir
```

Thay đường dẫn bằng thư mục thật của bạn. Gói đã kèm mô hình cơ thể người **`smpl_models/SMPL_NEUTRAL.pkl`** để nạp người SMPL trong OmniRe; tool tự tìm file này trong thư mục làm việc. File mô hình cơ thể này khác với checkpoint lưu kết quả học của từng run.

Các run nuScenes trong bộ kết quả này đều học từ **ba camera pinhole phía trước**. Việc một run dùng renderer 3DGUT hoặc 3DGEER không có nghĩa run đó đã dùng ảnh mắt cá. Bản xem trước trong trình duyệt hiện nhìn cảnh bằng camera ảo pinhole và hiển thị một khung nhìn 3D.

## Các thư mục trong gói dùng để làm gì?

| Đường dẫn | Nội dung và mục đích |
|---|---|
| `data/nuscenes_scene-0061/` | Dữ liệu của đoạn đường đã chuyển sang dạng tool đọc được: ảnh, các điểm đo 3D từ LiDAR, vị trí/hướng camera, nhãn vật thể và cách chia ảnh để học/kiểm tra. |
| `runs/nuscenes_scene-0061/` | Kết quả từng thực nghiệm, nằm trong các thư mục `omnire/` hoặc `streetgs/`: cấu hình, log, chỉ số đánh giá, ảnh, video, bản xuất 3D và checkpoint. |
| `cache/nuscenes_scene-0061/` | Kết quả chuẩn bị, như ảnh đánh dấu vùng trời/vật chuyển động và tư thế người. Tool dùng lại khi cảnh và cấu hình tương ứng khớp, để không tính lại những bước đó. |
| `smpl_models/SMPL_NEUTRAL.pkl` | Mô hình cơ thể có khớp mà OmniRe dùng khi dựng người. Cần cho train, render GPU và đánh giá bằng trainer có người SMPL. |
| `pack.json` | Danh sách run, dữ liệu đi kèm và mã băm SHA-256 của các file, để đối chiếu dữ liệu sau khi chuyển. |

File `.ply` là bản xuất **tĩnh** gồm nền và, nếu được chọn khi xuất, xe ở một khung. File này không chứa đầy đủ chuyển động và người SMPL như checkpoint hoặc bản xem trước trong trình duyệt. Các thư mục cảnh ảo tạm do tool dựng khi chạy không được chép vào tar; tool dựng lại chúng trên máy nhận từ dữ liệu đã có.

## Đọc bảng thực nghiệm

**OmniRe** mô hình hoá nền, xe, người và vật thể biến dạng trong một cảnh. **Street Gaussians** mô hình hoá nền và xe. **Renderer** là bộ phận tạo ảnh từ mô hình 3D theo thông số camera; 3DGS, 3DGUT và 3DGEER là các lựa chọn renderer.

Trong lúc học, phần mềm có thể thêm hoặc bỏ Gaussian, là các phần tử 3D tạo nên cảnh. Cột tương ứng cho biết run dùng quy tắc gốc của DriveStudio, quy tắc FastGS hay cách MCMC. Chu kỳ 50 hoặc 300 bước cho biết khoảng cách giữa các lần điều chỉnh số Gaussian. Faster-GS là nhóm tuỳ chọn tăng tốc một số phép tính; phần áp dụng phụ thuộc renderer.

Bộ 11 run này cùng dùng ba camera `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`. Split tên **`front3`** quy định dùng 516 ảnh để học và giữ lại 57 ảnh để kiểm tra, bằng cách giữ một khung trong mỗi mười khung. Mỗi run học 30.000 bước, ở độ phân giải 800×450, trên một GPU Tesla T4 của Kaggle.

PSNR và SSIM đo mức khớp giữa ảnh tạo ra và ảnh kiểm tra, giá trị cao hơn là tốt hơn trên cùng tập ảnh. LPIPS đo sai khác về đặc trưng hình ảnh, giá trị thấp hơn là tốt hơn. Thời gian trong bảng là thời gian train; chưa gồm toàn bộ thời gian chuẩn bị dữ liệu.

Mã như `f8a6` là **bốn ký tự cuối** của mã run đầy đủ để bảng dễ đọc. Khi dùng lệnh CLI, lấy mã đầy đủ từ Studio hoặc `d2s run list --workdir /duong/dan/drive2sim-workdir`, ví dụ `20261114-123210-omnire-f8a6`.

| Run | Phương pháp | Renderer | Cách thêm/bớt Gaussian | Faster-GS | PSNR | SSIM | LPIPS | Thời gian train |
|---|---|---|---|---|---|---|---|---|
| `2d33` | OmniRe | 3DGS | DriveStudio gốc | tắt | 25,15 | 0,786 | 0,251 | 3 giờ 39 phút |
| `8d9c` | OmniRe | 3DGS | DriveStudio gốc | bật | 25,20 | 0,785 | 0,251 | 3 giờ 18 phút |
| `453a` | OmniRe | 3DGS | FastGS | tắt | 25,02 | 0,777 | 0,268 | 2 giờ 47 phút |
| `f8a6` | OmniRe | 3DGS | FastGS | bật | 25,01 | 0,778 | 0,269 | 2 giờ 29 phút |
| `a022` | OmniRe | 3DGS | MCMC | bật | 23,56 | 0,716 | 0,403 | 2 giờ 31 phút |
| `60e5` | OmniRe | 3DGUT | FastGS | bật | 23,99 | 0,750 | 0,333 | 2 giờ 26 phút |
| `44d8` | OmniRe | 3DGUT | MCMC | bật | 23,19 | 0,711 | 0,415 | 2 giờ 39 phút |
| `6b85` | OmniRe | 3DGEER | FastGS, chu kỳ 300 bước | bật | 21,66 | 0,744 | 0,317 | 2 giờ 34 phút |
| `a8a9` | OmniRe | 3DGEER | FastGS, chu kỳ 50 bước | bật | 21,77 | 0,752 | 0,300 | 2 giờ 43 phút |
| `ccec` | Street Gaussians | 3DGS | FastGS | tắt | 25,02 | 0,779 | 0,264 | 1 giờ 11 phút |
| `ecaa` | Street Gaussians | 3DGS | FastGS | bật | 24,96 | 0,779 | 0,264 | 1 giờ 3 phút |

## Checkpoint giữ những gì?

Checkpoint lưu các tham số cảnh mà quá trình học đã tìm ra. Mỗi run trong bộ tham chiếu có các file sau trong `d2s/stages/train.drivestudio/` của nó:

| File | Dùng để |
|---|---|
| `checkpoint_final.pth` | Nạp mô hình cuối để render GPU, đánh giá hoặc xuất lại dữ liệu 3D. |
| `checkpoint_15000.pth` | Nạp mô hình ở bước 15.000 để so với mô hình cuối hoặc dùng trong thử nghiệm tiếp theo. |

Hai file này lưu mô hình, không lưu đầy đủ trạng thái của bộ tối ưu dùng để tiếp tục y nguyên một lần học đang dở. Với run bị ngắt, tool dùng `checkpoint_resume.pth` nếu có để khôi phục cả trạng thái đó. Gọi `d2s run resume` cho một run đã hoàn tất không tự kéo dài quá trình học.

Ảnh, video, bảng chỉ số và bản xem trước 3D đã xuất có thể được mở mà không nạp checkpoint. Muốn tạo lại các kết quả từ mô hình thì mới cần checkpoint và các thư viện tương ứng.

## Nếu muốn chạy thử một cấu hình mới

Sau khi cài môi trường GPU, bạn có thể tạo một run mới trên cảnh đã có. Ví dụ, lệnh sau dùng OmniRe với renderer 3DGUT và split `front3`:

```bash
d2s run start omnire --scene nuscenes_scene-0061 --split front3 \
  --set renderer=3dgut --workdir /duong/dan/drive2sim-workdir
```

Lệnh tạo một thực nghiệm mới, không sửa mô hình của 11 run đã nhận. Tool kiểm tra các kết quả chuẩn bị trong `cache/` và dùng lại những kết quả phù hợp. Nếu muốn đổi nhóm camera hoặc ảnh kiểm tra, cần tạo split mới trước khi tạo run.

Gói không chứa thông tin đăng nhập và dữ liệu nuScenes gốc. Dữ liệu cảnh đã nạp đủ để mở các run và thử cấu hình trên cảnh đó; chỉ cần tải dataset gốc ở [nuScenes](https://www.nuscenes.org/download) nếu muốn nạp lại hoặc chọn đoạn ghi hình khác.

## Giấy phép

Ảnh, LiDAR và box trong gói lấy từ nuScenes v1.0-mini (Motional), giấy phép CC BY-NC-SA 4.0: chỉ dùng phi thương mại, ghi nguồn nuScenes, và không phát tán lại ngoài nhóm được chia sẻ. Package drive2sim được phân phối theo AGPL-3.0 như quy định của repo; xem các thông báo bản quyền đi kèm mã nguồn.

---

<a id="english"></a>

# Viewing drive2sim reconstruction results

This archive lets you reopen trained scene models, inspect images and videos, and compare reconstruction configurations without retraining them. It contains the data; drive2sim and its runtime libraries still need to be installed on the receiving machine.

The reference collection contains **one recorded nuScenes road segment, `scene-0061`, reconstructed with 11 configurations**. A **run** is one training experiment, with its own learned scene model. The runs share the recorded input segment so their results can be compared. `pack.json` lists the runs actually present in this archive.

## Open the results

If the handoff includes `drive2sim-0.1.0-py3-none-any.whl`, install that Python software package with [uv](https://docs.astral.sh/uv/) on Linux. Run these commands from the folder containing the wheel, substituting its actual name if different:

```bash
uv venv ~/.venvs/drive2sim
uv pip install --python ~/.venvs/drive2sim/bin/python './drive2sim-0.1.0-py3-none-any.whl[studio]'
```

Extract the data into an empty **working directory**, the folder from which the tool will read scenes and results. `results.tar` below is an example name; replace it with the archive you received, such as `results-nuscenes-11runs.tar`.

```bash
mkdir drive2sim-workdir
tar -xf results.tar -C drive2sim-workdir
~/.venvs/drive2sim/bin/d2s studio --workdir "$PWD/drive2sim-workdir"
```

Open the address printed by Studio. **Runs** lists the experiments, **Compare** places their measurements side by side, and **3D viewer** opens a scene and plays its motion. If you installed drive2sim in another environment, use that environment's `d2s` command instead. Always select the extracted data folder with `--workdir`.

## Choose a viewing mode

The **browser preview** uses exported 3D data for camera movement and motion playback without an NVIDIA GPU or CUDA environment. **Video** plays images already rendered from the recorded cameras; its viewpoint cannot be rotated. **GPU rendering** loads the learned scene and generates new images with its renderer, requiring the GPU environment with PyTorch, CUDA and the rendering libraries.

For GPU rendering, re-export or training, install the GPU environment described in the [repository README](https://github.com/thaianh20021/drive2sim), then install the supplied wheel into that environment. Once it is installed, for a conda environment named `drive2sim`, run:

```bash
conda activate drive2sim
d2s studio --workdir /path/to/drive2sim-workdir
```

The required SMPL body model is already included at **`smpl_models/SMPL_NEUTRAL.pkl`**. The tool finds it inside the workdir. This articulated body model is a separate dependency from the checkpoint containing each run's learned scene parameters.

All reference runs in this archive trained on **three front pinhole cameras**. A run using the 3DGUT or 3DGEER renderer does not therefore imply that it used fisheye training images. The browser preview currently displays the scene through one virtual pinhole camera.

## What the folders contain

| Path | Purpose |
|---|---|
| `data/nuscenes_scene-0061/` | Images, LiDAR 3D measurements, camera poses/calibration, object annotations and training/test selections in the tool's scene format. |
| `runs/nuscenes_scene-0061/` | Experiment results under `omnire/` or `streetgs/`: configuration, logs, measurements, images, videos, 3D exports and checkpoints. |
| `cache/nuscenes_scene-0061/` | Reusable preparation results, including masks marking sky/moving regions and estimated human poses. Reuse requires matching inputs and settings. |
| `smpl_models/SMPL_NEUTRAL.pkl` | The body model needed to train, GPU-render and evaluate SMPL people through the trainer. |
| `pack.json` | The selected runs, included files and SHA-256 hashes for checking transferred data. |

The `.ply` export is **static**: background and optionally vehicles at one frame. It does not include the full motion and SMPL people available through the checkpoint or browser preview. Temporary linked scene views are rebuilt on the receiving machine from the included data.

## Read the experiment table

**OmniRe** models background, vehicles, people and deformable objects. **Street Gaussians** models background and vehicles. A **renderer** turns the 3D model into an image using camera parameters; 3DGS, 3DGUT and 3DGEER are renderer choices.

Training can add or remove the 3D elements called Gaussians. The table states whether a run uses the original DriveStudio rule, FastGS or MCMC. The 50/300-step interval is the frequency of Gaussian-count refinement. Faster-GS optimizes some computations; the optimizations used depend on the renderer.

The 11 reference runs use `CAM_FRONT`, `CAM_FRONT_LEFT` and `CAM_FRONT_RIGHT`. The **`front3` split** selects 516 training and 57 held-out test images, holding out every tenth frame. Each run trains for 30,000 steps at 800×450 on one Kaggle Tesla T4.

On the same test images, higher PSNR and SSIM indicate closer image agreement, while lower LPIPS indicates a smaller perceptual feature difference. Training time excludes the full data-preparation cost. The table above lists the measurements. Codes such as `f8a6` are the **last four characters** of full run IDs; use the full ID shown in Studio or `d2s run list` for CLI commands.

## What checkpoints store

The reference runs store `checkpoint_final.pth` and `checkpoint_15000.pth` under `d2s/stages/train.drivestudio/`. The former loads the final scene for GPU rendering, evaluation or re-export; the latter loads the model at step 15,000 for comparison or further experiments.

These files contain model parameters, not the complete optimizer state needed to continue an interrupted optimization unchanged. Interrupted runs use `checkpoint_resume.pth`, when present, for that state too. `d2s run resume` does not add training steps to a completed run. Existing images, videos, measurements and browser previews can be opened without loading checkpoints.

## Try a new configuration

After installing the GPU environment, create a new OmniRe experiment with the 3DGUT renderer on the existing `front3` split:

```bash
d2s run start omnire --scene nuscenes_scene-0061 --split front3 \
  --set renderer=3dgut --workdir /path/to/drive2sim-workdir
```

This creates a new run without changing the received models. Compatible preparation results are reused from `cache/`. To change the cameras or held-out images, create a new split first.

Credentials and raw nuScenes downloads are not included. The prepared scene is enough to open these runs and try configurations on this scene. Raw data from [nuScenes](https://www.nuscenes.org/download) is needed to re-ingest or choose another recorded segment.

## Licence

The images, LiDAR and boxes come from nuScenes v1.0-mini (Motional) under CC BY-NC-SA 4.0: non-commercial use only, credit nuScenes, and do not redistribute outside the group it was shared with. The drive2sim package is distributed under AGPL-3.0 as specified by the repository. See the copyright notices supplied with the source.
