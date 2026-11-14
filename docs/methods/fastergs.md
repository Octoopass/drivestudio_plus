# Faster-GS: tăng tốc train 3DGS

**Trạng thái:** phần Python đã cài (`drive2sim/recon/speedups/fastergs.py`): fused Adam, fused SSIM, bớt chờ GPU, Z-order (thử nghiệm). SelectiveAdam dừng; phần kernel còn là kế hoạch.

## Faster-GS là gì

"Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm và cộng sự, CVPR 2026), code tại [nerficg-project/faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting), license Apache-2.0.

Faster-GS không đổi thuật toán 3DGS: vẫn densify theo cách gốc, số Gaussian và chất lượng giữ nguyên. Nó gom kỹ thuật tăng tốc đã có (Taming-3DGS, StopThePop, Speedy-Splat, Splatshop) và thêm tối ưu riêng: backward theo từng Gaussian, sort tách hai bước, Adam fused, Z-order. Trên RTX 4090, paper báo nhanh hơn 3DGS gốc 4,1 lần (Mip-NeRF 360) và 5,2 lần (Deep Blending), VRAM giảm tới 30%, đo trên cảnh tĩnh, camera pinhole.

Bản gốc chạy trong NeRFICG, rasterizer riêng `FasterGSCudaBackend` chỉ nhận pinhole và SH, chỉ trả RGB; OmniRe cần độ sâu, alpha, fisheye, nên dự án không dùng bản này.

## Áp dụng vào tầng nào

| Phần | Tầng | Nơi sửa |
|---|---|---|
| Fused Adam, fused SSIM, bớt đồng bộ host | Trainer | `drive2sim/recon/trainers/base.py` |
| Z-order (`morton_order`, mặc định tắt) | Trainer | `speedups/fastergs.py` |
| Chọn tile chính xác, sort tách hai bước, backward theo từng Gaussian | Rasterizer | bản fork gsplat-geer của dự án |

## Cách tích hợp

- **Công tắc:** mục cấu hình `fastergs` (bật/tắt) trong nhóm "Thuật toán phụ trợ" của OmniRe và Street GS, mặc định tắt. Bật thì `train.drivestudio` chọn config biến thể `*_fastergs.yaml`, viết bằng `extends` (CONTRACTS mục 10.5), nên chỉ khác config gốc ở khối `trainer.fastergs`.
- **Phần Python:** mỗi kỹ thuật một cờ trong `trainer.fastergs` (`fused_adam`, `fused_ssim`, `fewer_syncs`), bật theo công tắc. Đã kiểm trên GPU: fused SSIM (`fused-ssim` commit `a7c48d6`, `padding="valid"`) lệch `pytorch_msssim` 6e-8, gradient 2e-6, nhanh hơn khoảng 5 lần ở 900×1600; fused Adam qua densify và prune lệch 1,2e-7. Giữ nguyên cách đo `train_time_s`.
- **Phần kernel:** cài lại kỹ thuật của `FasterGSCudaBackend` vào bản fork gsplat-geer, mỗi kỹ thuật là một tham số mới của `rasterization()`; giá trị mặc định giữ đúng hành vi cũ. Bản fork tự khai renderer nào có kỹ thuật nào; trainer đọc bảng đó (như `_init_renderer()` đang dò `with_geer`) và chỉ bật kỹ thuật được hỗ trợ.
- **Ghi lại:** kỹ thuật thật sự đã bật và commit của bản fork được ghi vào `train_stats.json` rồi `summary.json`, để trang So sánh phân biệt hai run cùng `fastergs: true` nhưng khác kỹ thuật.

Hỗ trợ theo renderer (`trainer.render.render_mode`):

| Kỹ thuật | default | ut | geer |
|---|---|---|---|
| Phần Python | có | có | có |
| Chọn tile chính xác | có | có | không cần: `isect_tiles_geer` đã ghép chính xác |
| Sort tách hai bước | có | có | có, sửa thêm `IntersectGEERTile.cu` |
| Backward theo từng Gaussian | có | cần bản riêng cho kernel `eval3d` | như ut |

Bật cùng FastGS được.

## Dùng an toàn và hiệu quả

- **So sánh công bằng:** chỉ đổi công tắc; nhờ `extends`, run có và không có Faster-GS dùng cùng tham số train.
- **Kernel mới phải qua test so khớp:** khi tắt cờ, bản fork phải cho ảnh và gradient trùng từng bit với gsplat-geer gốc (commit `f15cee2`) ở cả ba renderer; khi bật, chỉ lệch mức làm tròn.
- **SelectiveAdam dừng:** `SelectiveAdam` của gsplat-geer bỏ cả hiệu chỉnh bias của Adam (đổi cách tối ưu); code công khai của Faster-GS không có bản bỏ qua Gaussian không nhìn thấy. Cần kernel giữ hiệu chỉnh bias.
- **Z-order chỉ áp cho nền:** xe, người có dữ liệu gắn theo từng Gaussian. Chỉ tương đương thống kê: ảnh trùng, Adam lệch 2e-5 sau 20 bước, densify ngẫu nhiên khác.
- **Đo trước khi sửa kernel:** rasterizer chiếm 30% mỗi bước thì nhanh gấp đôi chỉ bớt 15% tổng thời gian.
