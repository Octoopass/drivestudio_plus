# Faster-GS: tăng tốc train 3DGS

**Trạng thái:** kế hoạch, chưa cài. Trang này mô tả phương pháp và cách dự án sẽ tích hợp.

## Faster-GS là gì

"Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm và cộng sự, CVPR 2026), code tại [nerficg-project/faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting), license Apache-2.0.

Faster-GS không đổi thuật toán 3DGS: vẫn densify theo cách gốc, số Gaussian và chất lượng giữ nguyên. Nó gom các kỹ thuật tăng tốc đã có (Taming-3DGS, StopThePop, Speedy-Splat, Splatshop) và thêm tối ưu riêng: backward theo từng Gaussian, sort tách theo độ sâu rồi theo tile, Adam fused, sắp Gaussian theo Z-order. Trên RTX 4090, paper báo nhanh hơn 3DGS gốc 4,1 lần (Mip-NeRF 360) và 5,2 lần (Deep Blending), VRAM giảm tới 30%, đo trên cảnh tĩnh, camera pinhole.

Bản gốc chạy trong framework NeRFICG, với rasterizer riêng `FasterGSCudaBackend`: chỉ camera pinhole, nhận hệ số SH, chỉ trả RGB. Dự án **không** dùng bản này, vì OmniRe cần độ sâu, alpha, màu theo từng node, fisheye và F-theta.

## Áp dụng vào tầng nào

| Phần | Tầng | Nơi sửa |
|---|---|---|
| Fused Adam, fused SSIM, bớt đồng bộ host | Trainer | `drive2sim/recon/trainers/base.py` |
| Z-order, SelectiveAdam | Trainer và model | trainer, `models/gaussians/vanilla.py` |
| Chọn tile chính xác, sort tách hai bước, backward theo từng Gaussian | Rasterizer | bản fork gsplat-geer của dự án |

## Cách tích hợp

- **Công tắc:** mục cấu hình `fastergs` (bật/tắt) trong nhóm "Thuật toán phụ trợ" của OmniRe và Street GS, mặc định tắt. Bật thì `train.drivestudio` chọn config biến thể `*_fastergs.yaml`, viết bằng `extends` (CONTRACTS mục 10.5), nên chỉ khác config gốc ở khối `trainer.fastergs`.
- **Phần Python:** mỗi kỹ thuật một cờ trong `trainer.fastergs`. Fused Adam và fused SSIM bật theo công tắc; Z-order và SelectiveAdam mặc định tắt, dùng khi thử nghiệm.
- **Phần kernel:** cài lại kỹ thuật của `FasterGSCudaBackend` vào bản fork gsplat-geer, mỗi kỹ thuật là một tham số mới của `rasterization()`; giá trị mặc định giữ đúng hành vi cũ. Bản fork tự khai renderer nào có kỹ thuật nào; trainer đọc bảng đó (như `_init_renderer()` đang dò `with_geer`) và chỉ bật kỹ thuật được hỗ trợ.
- **Ghi lại:** kỹ thuật thật sự đã bật và commit của bản fork được ghi vào `train_stats.json` rồi `summary.json`, để trang So sánh phân biệt hai run cùng `fastergs: true` nhưng khác kỹ thuật.

Hỗ trợ theo renderer (`trainer.render.render_mode`):

| Kỹ thuật | default | ut | geer |
|---|---|---|---|
| Phần Python | có | có | có |
| Chọn tile chính xác | có | có | không cần: `isect_tiles_geer` đã ghép chính xác |
| Sort tách hai bước | có | có | có, sửa thêm `IntersectGEERTile.cu` |
| Backward theo từng Gaussian | có | cần bản riêng cho kernel `eval3d` | như ut |

Faster-GS không đổi model nên bật cùng FastGS được.

## Dùng an toàn và hiệu quả

- **So sánh công bằng:** chỉ đổi công tắc, giữ nguyên cảnh, split, số iteration và config gốc. Nhờ `extends`, run có và không có Faster-GS dùng cùng tham số train.
- **Kernel mới phải qua test so khớp:** khi tắt cờ, bản fork phải cho ảnh và gradient trùng từng bit với gsplat-geer gốc (commit `f15cee2`) ở cả ba renderer; khi bật, sai lệch chỉ ở mức làm tròn số thực.
- **SelectiveAdam giảm chất lượng nhẹ:** paper báo PSNR ngoài trời từ 24,72 xuống 24,59.
- **Z-order chỉ áp cho nền (`Background`):** xe và người có dữ liệu gắn theo từng Gaussian (liên kết với xe, trọng số skinning), sắp lại sai sẽ làm vỡ chúng.
- **Đo trước khi sửa kernel:** profile để biết rasterizer chiếm bao nhiêu thời gian mỗi bước; chiếm 30% thì nhanh gấp đôi chỉ bớt khoảng 15% tổng thời gian.
