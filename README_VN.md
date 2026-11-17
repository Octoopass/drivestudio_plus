<a id="top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/media/logo.png">
    <img alt="Logo" src="docs/media/logo.png" width="560">
  </picture>
</p>

# drive2sim

[English: README.md](README.md)

drive2sim biến dữ liệu lái xe đã ghi (ảnh camera, LiDAR, vị trí xe) thành một cảnh 3D Gaussian Splatting để xem, đo chất lượng và xuất ra file. Tool gom các phương pháp dựng cảnh của [DriveStudio](https://github.com/ziyc/drivestudio) (OmniRe, Street Gaussians) vào một chỗ: lệnh `d2s` và một trang web chạy trên máy tên là Studio. Bạn chọn dataset, cảnh và phương pháp, tool tự chạy hết các bước, trên GPU của máy bạn hoặc trên Kaggle, Google Colab. Tool dành cho người muốn so sánh nhiều phương pháp trên cùng một cảnh mà không phải tự viết script train.

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc kho mã nguồn](#cấu-trúc-kho-mã-nguồn)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn bắt đầu](#hướng-dẫn-bắt-đầu)
- [Tài liệu hướng dẫn](#tài-liệu-hướng-dẫn)
- [Đóng góp](#đóng-góp)
- [Lời cảm ơn](#lời-cảm-ơn)
- [Bản quyền](#bản-quyền)

## Tính năng chính

- **Một đường đi từ dữ liệu gốc tới kết quả.** Nạp cảnh, chọn khung nào để dành kiểm tra, train, đánh giá (PSNR, SSIM, LPIPS) rồi xuất file `.ply` chuẩn. Mỗi bước là một lệnh, hoặc vài cú bấm trong Studio.
- **Nhiều phương pháp trên cùng một cảnh.** OmniRe (nền, xe và người), Street Gaussians (nền và xe) và một trainer 3DGUT cho cảnh tĩnh. Có ba renderer: 3DGS chuẩn, 3DGUT và 3DGEER.
- **Camera pinhole và mắt cá.** Camera mắt cá của KITTI-360 được đổi sang mô hình OpenCV fisheye và train chung với camera pinhole.
- **Các cách tăng tốc tuỳ chọn**, mỗi cách là một công tắc: densify kiểu FastGS (ít Gaussian hơn), kernel Faster-GS, densify MCMC, và dùng lại mask, tư thế người giữa các run.
- **Chạy ở đâu cũng được.** Cùng một run chạy trên máy, trên GPU của Kaggle hoặc trên VM Colab; tool tự gửi đi, theo dõi và tải kết quả về.
- **Studio** hiện từng run với dòng thời gian của các bước, bảng so sánh giữa các run và trang xem 3D phát cảnh theo thời gian.

Kết quả trên cảnh nuScenes `scene-0061`, ba camera phía trước, 30k bước, một GPU Tesla T4:

| Phương pháp | Renderer | PSNR tập test | Thời gian train | VRAM cao nhất |
|---|---|---|---|---|
| OmniRe + FastGS + Faster-GS | 3DGS | 25.01 | 2 giờ 29 phút | 4.7 GB |
| Street Gaussians + FastGS | 3DGS | 25.02 | 1 giờ 11 phút | 3.4 GB |
| OmniRe | 3DGUT | 23.99 | 2 giờ 26 phút | 3.5 GB |
| OmniRe | 3DGEER | 21.77 | 2 giờ 43 phút | 4.3 GB |

Mỗi con số là một lần chạy trên một cảnh. Danh sách đầy đủ, kèm cấu hình từng run, nằm ở [docs/experiments/run_plan.md](docs/experiments/run_plan.md).

## Cấu trúc kho mã nguồn

| Đường dẫn | Nội dung |
|---|---|
| `drive2sim/` | Tool: CLI (`cli/`), bộ chạy run (`engine/`), converter dữ liệu (`ingest/`), các bước của pipeline (`stages/`), phương pháp (`approaches/`), Studio (`studio/`) |
| `drive2sim/recon/` | Code dựng cảnh của DriveStudio mà các bước gọi tới |
| `envs/` | Script cài môi trường GPU |
| `third_party/` | gsplat-geer (renderer), code SMPL và 4D-Humans |
| `docs/` | Thiết kế, định dạng file, hướng dẫn từng dataset và ghi chú phương pháp |
| `tests/` | Test (`make test`) |

## Yêu cầu hệ thống

- Muốn train cần Linux, GPU NVIDIA và bộ CUDA toolkit (`nvcc`), vì renderer được biên dịch lúc cài. Các run ở bảng trên dùng tối đa 5 GB bộ nhớ GPU.
- Máy không có GPU vẫn dùng được Studio và CLI, rồi gửi run lên Kaggle hoặc Colab. Cách cài nhẹ này cần [uv](https://docs.astral.sh/uv/) và Node.js.
- OmniRe dựng người bằng mô hình cơ thể SMPL. File `SMPL_NEUTRAL.pkl` có giấy phép riêng nên không nằm trong repo. Tải từ [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) rồi đặt vào `<workdir>/smpl_models/SMPL_NEUTRAL.pkl`.

## Hướng dẫn bắt đầu

Các bước dưới dùng nuScenes v1.0-mini, bộ mẫu miễn phí khoảng 4 GB tải ở [nuscenes.org](https://www.nuscenes.org/download) (cần đăng ký tài khoản).

1. **Tải mã nguồn.**

   ```bash
   git clone --recursive --branch drive2sim https://github.com/Octoopass/drivestudio_plus.git
   cd drivestudio_plus
   ```

2. **Cài đặt.** Trên máy có GPU NVIDIA:

   ```bash
   conda env create -f envs/main.yml
   conda activate drive2sim
   bash envs/install_main.sh        # bước nào lỗi thì sửa rồi chạy lại từ bước đó: bash envs/install_main.sh --from <bước>
   ```

   Trên máy không có GPU, `make setup` tạo `.venv/` chỉ gồm CLI và Studio (gọi lệnh bằng `.venv/bin/d2s`).

3. **Nạp một cảnh.** Tool làm việc trong một thư mục chứa `data/`, `cache/` và `runs/`. Đó là thư mục ghi trong biến `D2S_WORKDIR`, hoặc thư mục hiện tại.

   ```bash
   d2s source inspect /đường/dẫn/nuscenes      # xem tool nhận ra gì trong thư mục
   d2s scene ingest /đường/dẫn/nuscenes --version v1.0-mini --sequence scene-0061
   d2s split create <scene_id> --strategy every_nth_frame:8
   ```

   `scene ingest` in ra `scene_id`, còn `split create` in ra `split_id` để dùng ở bước sau.

4. **Train.**

   ```bash
   d2s run start omnire --scene <scene_id> --split <split_id>                     # chạy trên máy này
   d2s run start omnire --scene <scene_id> --split <split_id> --executor kaggle   # chạy trên Kaggle
   ```

   `d2s approach list` liệt kê các phương pháp, `d2s approach show <phương pháp>` cho xem các tuỳ chọn. Chạy trên Kaggle cần token API của Kaggle và file SMPL đã upload thành dataset private ([CONTRACTS mục 9.2](docs/drive2sim/CONTRACTS.md)).

5. **Xem kết quả.** `d2s report` in metric của mọi run. Muốn xem kỹ hơn thì bật Studio rồi mở địa chỉ nó in ra:

   ```bash
   make studio            # hoặc: d2s studio
   ```

   Bước 3 và 4 cũng làm được ngay trong Studio, ở trang Cảnh và trang Chạy mới.

## Tài liệu hướng dẫn

| Tài liệu | Nội dung |
|---|---|
| [docs/drive2sim/SPEC.md](docs/drive2sim/SPEC.md) | Tool làm gì, từng bước, theo góc nhìn người dùng |
| [docs/drive2sim/ARCHITECTURE.md](docs/drive2sim/ARCHITECTURE.md) | Tool được xây thế nào và vì sao |
| [docs/drive2sim/CONTRACTS.md](docs/drive2sim/CONTRACTS.md) | Định dạng file, CLI, API, chi tiết Kaggle và Colab |
| [docs/datasets/overview.md](docs/datasets/overview.md) | Bố trí camera của từng dataset, kèm hướng dẫn tải và nạp cho từng dataset |
| [docs/methods/](docs/methods/) | Mỗi phương pháp hoặc cách tăng tốc một trang: làm gì, khác bài báo ở đâu |
| [docs/drivestudio.md](docs/drivestudio.md) | Gọi thẳng script của DriveStudio, không qua tool |

Dataset hỗ trợ: nuScenes, Waymo, Argoverse 2, PandaSet, nuPlan, KITTI, KITTI-360 và PhysicalAI. nuScenes đã train trọn đường. Phần nạp KITTI-360 đã kiểm trên dữ liệu thật, train trên KITTI-360 là việc tiếp theo. Các dataset còn lại dùng converter và loader của DriveStudio, nhưng dự án chưa train thật lần nào với chúng.

## Đóng góp

Đọc [AGENTS.md](AGENTS.md) trước. File này ghi quy tắc viết code, lệnh chạy test và quy định sửa code thì sửa tài liệu cùng lúc. Chạy `make check` (format, lint, test) trước mỗi commit. Gặp lỗi thì báo ở mục Issues của repo.

## Lời cảm ơn

Code dựng cảnh lấy từ [DriveStudio](https://github.com/ziyc/drivestudio) và bài báo OmniRe. Renderer 3DGUT và 3DGEER lấy từ [bản vá 3DGEER](https://github.com/boschresearch/3dgeer) của Bosch Research cho [gsplat](https://github.com/nerfstudio-project/gsplat). Các cách tăng tốc dựa theo [FastGS](https://github.com/fastgs/FastGS), [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) và [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc); chỗ nào khác bản gốc đều ghi trong [docs/methods/](docs/methods/). Nếu dùng code này, xin trích dẫn OmniRe:

```bibtex
@inproceedings{chen2025omnire,
  title={OmniRe: Omni Urban Scene Reconstruction},
  author={Ziyu Chen and Jiawei Yang and Jiahui Huang and Riccardo de Lutio and Janick Martinez Esturo and Boris Ivanovic and Or Litany and Zan Gojcic and Sanja Fidler and Marco Pavone and Li Song and Yue Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Bản quyền

DriveStudio dùng giấy phép MIT ([LICENSE](LICENSE)). Bản vá 3DGEER của Bosch đi kèm repo dùng giấy phép AGPL-3.0, nên cả repo phải tuân theo điều khoản AGPL-3.0. Hãy đọc kỹ trước khi phát tán lại hoặc dựng thành dịch vụ. Dataset và mô hình SMPL giữ giấy phép riêng; ví dụ KITTI-360 chỉ được dùng phi thương mại.

[Về đầu trang](#top)
