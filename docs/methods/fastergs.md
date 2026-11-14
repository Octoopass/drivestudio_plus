# Faster-GS: tăng tốc train 3DGS

**Trạng thái:** phần Python đã cài (`drive2sim/recon/speedups/fastergs.py`): fused Adam, fused SSIM, bớt chờ GPU, Z-order (thử nghiệm). Kernel (bản fork `third_party/gsplat-geer`, `awun0105/3dgeer` nhánh `drive2sim`): chọn tile chính xác, sort tách hai bước, backward theo từng Gaussian, chỉ cho renderer default. SelectiveAdam dừng.

## Faster-GS là gì

"Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm và cộng sự, CVPR 2026), code tại [nerficg-project/faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting), license Apache-2.0.

Faster-GS không đổi thuật toán 3DGS: densify, số Gaussian, chất lượng giữ nguyên. Nó gom kỹ thuật tăng tốc đã có (Taming-3DGS, StopThePop, Speedy-Splat, Splatshop) và thêm tối ưu riêng. Paper báo nhanh hơn 3DGS gốc 4,1 lần (Mip-NeRF 360), 5,2 lần (Deep Blending) trên RTX 4090, cảnh tĩnh.

Bản gốc chạy trong NeRFICG, rasterizer riêng `FasterGSCudaBackend` chỉ nhận pinhole và SH, chỉ trả RGB; OmniRe cần độ sâu, alpha, fisheye, nên dự án không dùng bản này.

## Áp dụng vào tầng nào

| Phần | Tầng | Nơi sửa |
|---|---|---|
| Fused Adam, fused SSIM, bớt đồng bộ host | Trainer | `drive2sim/recon/trainers/base.py` |
| Z-order (`morton_order`, mặc định tắt) | Trainer | `speedups/fastergs.py` |
| Chọn tile chính xác, sort tách hai bước, backward theo từng Gaussian | Rasterizer | bản fork gsplat-geer của dự án |

## Cách tích hợp

- **Công tắc:** mục cấu hình `fastergs` (bật/tắt) trong nhóm "Thuật toán phụ trợ" của OmniRe và Street GS, mặc định tắt. Bật thì `train.drivestudio` chọn config biến thể `*_fastergs.yaml`, viết bằng `extends` (CONTRACTS mục 10.5), nên chỉ khác config gốc ở khối `trainer.fastergs`.
- **Phần Python:** mỗi kỹ thuật một cờ trong `trainer.fastergs` (`fused_adam`, `fused_ssim`, `fewer_syncs`), bật theo công tắc. Kiểm trên GPU: fused SSIM (`fused-ssim` `a7c48d6`) lệch 6e-8, nhanh ~5 lần; fused Adam lệch 1,2e-7.
- **Phần kernel:** mỗi kỹ thuật là một tham số mới của `rasterization()` trong bản fork, mặc định giữ hành vi cũ; cờ `exact_tiles`, `two_stage_sort`, `per_gaussian_backward`. Bản fork tự khai renderer nào có kỹ thuật nào (`gsplat.fastergs_support()`); trainer chỉ bật kỹ thuật được hỗ trợ.
- **Ghi lại:** kỹ thuật thật sự đã bật và commit của bản fork được ghi vào `train_stats.json` rồi `summary.json`, để trang So sánh phân biệt hai run cùng `fastergs: true` nhưng khác kỹ thuật.

Hỗ trợ theo renderer (`trainer.render.render_mode`):

| Kỹ thuật | default | ut | geer |
|---|---|---|---|
| Phần Python | có | có | có |
| Chọn tile chính xác | có | không: UT vẽ bằng `eval3d` dọc tia 3D, phép thử ellipse 2D có thể bỏ sót tile | không cần: `isect_tiles_geer` đã ghép chính xác |
| Sort tách hai bước | có | chưa (cần test riêng) | chưa (`IntersectGEERTile.cu`) |
| Backward theo từng Gaussian | có | chưa (cần bản cho `eval3d`) | chưa |

Bật cùng FastGS được.

## Dùng an toàn và hiệu quả

- **So sánh công bằng:** chỉ đổi công tắc, tham số train giữ nguyên.
- **Kernel đã qua test so khớp** (`third_party/gsplat-geer/tests/test_drive2sim_*.py`, GPU): tắt cờ thì trùng từng bit bản gốc `f15cee2`; bật thì ảnh trùng từng bit, gradient lệch trong mức dao động của bản cũ. Cảnh giả lập 300k Gaussian, 1600×900, forward+backward: 68 ms → 48 ms với tile chính xác và backward theo Gaussian. Backward theo Gaussian tốn thêm ~300 MB GPU. Chế độ `packed` của bản gốc lỗi truy cập bộ nhớ; DriveStudio không dùng.
- **SelectiveAdam dừng:** `SelectiveAdam` của gsplat-geer bỏ cả hiệu chỉnh bias của Adam (đổi cách tối ưu); code công khai của Faster-GS không có bản bỏ qua Gaussian không nhìn thấy. Cần kernel giữ hiệu chỉnh bias.
- **Z-order chỉ áp cho nền:** xe, người có dữ liệu gắn theo từng Gaussian. Chỉ tương đương thống kê.
- **Đo trước khi sửa kernel:** run riêng với `--set train.drivestudio.profile_at=2000,20000` ghi `profile_step<bước>.json` (tỉ lệ thời gian GPU: rasterizer, optimizer, SSIM). Không bật profile trong run so `train_time_s`.
