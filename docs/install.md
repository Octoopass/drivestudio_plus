# Cài drive2sim bằng một lệnh

Ba cách cài cho cùng một kết quả: lệnh Node.js (tiện cho máy chủ không màn hình), script cho
Linux, macOS, và script cho Windows. Cả ba tạo venv riêng trong `~/.drive2sim/` (Windows:
`%USERPROFILE%\.drive2sim\`), cài drive2sim, Studio, Kaggle CLI, Colab CLI, và một lệnh
`drive2sim-studio` để bật Studio.

| Máy | Cài được gì | Chế độ |
|---|---|---|
| Linux có GPU NVIDIA | Đủ: Studio, CLI, môi trường train (PyTorch 2.4.1 + CUDA 12.1, gsplat-geer, nvdiffrast, pytorch3d), Instant NuRec, SDK dataset | `full` (mặc định khi có `nvidia-smi`) |
| Linux không GPU, macOS | Studio + CLI: quản lý cảnh, nạp cảnh PHENIKAA ngay trên máy (extra `ingest`: numpy, OpenCV, SciPy, laspy), chạy run trên GPU của Kaggle, Colab, xem kết quả, so sánh | `studio` |
| Windows | Như macOS. Muốn train trên máy có GPU NVIDIA: cài WSL2 (Ubuntu) rồi dùng cách Linux bên trong WSL | `studio` |

Chế độ `full` mất cỡ 1,5 giờ (build gsplat-geer, nvdiffrast, pytorch3d bằng `nvcc`; khi `nvcc`
của máy khác major với CUDA 12.1, script tự cài CUDA 12.8 riêng, khoảng 3 GB, không cần root).

## Lệnh Node.js (máy chủ không màn hình)

Cần Node.js 18 trở lên, và git (không có git thì lệnh tải gói nén của bản phát hành).

```bash
npx github:thaianh20021/drive2sim                  # cài: Linux GPU → full, máy khác → studio
npx github:thaianh20021/drive2sim start            # bật Studio ở nền, log ~/.drive2sim/studio.log
npx github:thaianh20021/drive2sim status
npx github:thaianh20021/drive2sim stop
```

Tuỳ chọn hay dùng: `--studio-only`, `--full`, `--no-nurec`, `--workdir <thư mục>`,
`--port <cổng>`, `--start` (bật Studio ngay sau khi cài), `--ref main` (bản mới nhất thay vì bản
phát hành), `--src <thư mục>` (dùng mã nguồn có sẵn, không tải). Xem đủ: `--help`.

Mặc định Studio chỉ nghe ở `127.0.0.1`. Cả nhóm cùng vào qua mạng riêng (Tailscale): `start --host
<IP Tailscale>`, hoặc `start --allow-host <IP Tailscale>` khi tailscaled chạy
`--tun=userspace-networking` (máy không có sudo). Studio không có đăng nhập: không dùng IP công khai.
Lệnh `drive2sim-studio` nhận cùng các cờ này. Không dùng `--host` thì `start` in sẵn lệnh SSH để mở
từ máy của bạn:

```bash
ssh -N -L 8765:127.0.0.1:8765 <người dùng>@<máy chủ>
```

rồi mở <http://127.0.0.1:8765>.

## Linux, macOS

Từ bản clone (`git clone --recurse-submodules https://github.com/thaianh20021/drive2sim`) hoặc
gói `drive2sim-v<phiên bản>-linux.tar.gz`, `-macos.tar.gz` của bản phát hành:

```bash
bash envs/install.sh                # tự chọn full hoặc studio
bash envs/install.sh --studio-only
bash envs/install.sh --full --start
```

Tuỳ chọn khác: `--workdir`, `--port`, `--no-nurec`, `--skip-tools` (không cài, nâng cấp Kaggle
CLI, Colab CLI), `--no-link` (không tạo lệnh trong `~/.local/bin`). Biến `D2S_HOME` đổi chỗ đặt venv.

Chế độ `full` gọi `d2s libraries install --all`, giống nút **Cài tất cả** ở trang Thư viện của
Studio: môi trường chính bằng `envs/install_main.sh`, Instant NuRec bằng `envs/install_nurec.sh`,
rồi các SDK dataset và công cụ. Mục lỗi không dừng cả lượt; sửa nguyên nhân rồi chạy lại lệnh, các
mục đã có được bỏ qua và môi trường chính cài tiếp từ bước lỗi.

## Windows

Từ bản clone hoặc gói `drive2sim-v<phiên bản>-windows.zip`, trong PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File envs\install.ps1
powershell -ExecutionPolicy Bypass -File envs\install.ps1 -Workdir D:\d2s-work -Port 8765 -Start
```

Script thêm `%USERPROFILE%\.drive2sim\bin` vào PATH của tài khoản (`-NoPath` để bỏ), nên mở cửa
sổ dòng lệnh mới rồi gõ `drive2sim-studio`. `-SkipTools` bỏ cài Kaggle CLI, Colab CLI. Cài lại khi
Studio đang chạy thì script dừng và nhắc tắt Studio trước, vì Windows khoá `d2s.exe`.

## Cài tiếp, thêm thư viện

- Trang **Thư viện** của Studio: nút cài từng mục và nút **Cài tất cả**.
- Dòng lệnh: `d2s libraries list`, `d2s libraries install <id>...`, `d2s libraries install --all`.

Cài tất cả bỏ qua Waymo SDK (kéo TensorFlow 2.11, có thể đổi numpy của môi trường chính) và nuPlan
devkit (ghim nhiều gói); hai mục này có nút cài riêng. Mọi lệnh pip của bảng chạy với file ràng buộc
ghim các gói đã có, nên chỉ thêm gói mới, không nâng, hạ gói của môi trường chính.

## Đã kiểm tra

| Đường cài | Máy | Kết quả |
|---|---|---|
| `envs/install.ps1` chế độ studio, lệnh `drive2sim-studio`, lệnh Node `install --start`, `status`, `stop` | Windows 10, Python 3.11 qua uv | Chạy hết; Studio trả lời, tắt được |
| `envs/install.sh --studio-only`, lệnh Node `start`, `status`, `stop` | Ubuntu (máy lab), Node 20 | Chạy hết; Studio trả lời, tắt được |
| v0.3.0: `envs/install.sh --studio-only`, lệnh Node `start --allow-host`, `status`, `stop` | Ubuntu (máy lab), Node 20 | Chạy hết; Studio nhận tên máy thêm, Host lạ bị chặn (403), tắt được |
| Cài tất cả (nút Studio) | Máy lab, môi trường chính đã có | 4 mục cài thêm, 9 mục đã có, Argoverse 2 bỏ qua (thiếu Rust); numpy, torch, matplotlib, OpenCV giữ nguyên |
| `envs/install.sh --full` từ đầu | | **Chưa chạy trọn** (build CUDA cỡ 1,5 giờ); các phần nó gọi đã chạy thật riêng: nút cài môi trường chính (8 bước) và Cài tất cả |
| macOS | | **Chưa kiểm tra** (không có máy Mac) |
