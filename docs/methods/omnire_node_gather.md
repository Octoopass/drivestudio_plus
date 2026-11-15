# Gom Gaussian theo node của OmniRe: bỏ phép lấy thừa

**Trạng thái:** đã sửa trong code DriveStudio (`drive2sim/recon/models/`, không thuộc patch của Bosch) cho nền (`VanillaGaussians`), xe (`RigidNodes`), node biến dạng (`DeformableNodes`); kết quả không đổi (kiểm chứng ở mục 4). Node người (`SMPLNodes`) chưa sửa. Mức nhanh hơn của cả bước train trên run thật **chưa đo**.

Liên quan: `fastergs.md` (tăng tốc rasterizer, Adam, SSIM; không chạm phần này).

## 1. Vấn đề

Profile OmniRe trên Kaggle T4 (nuScenes `scene-0061`, `front3`; `profile_step*.json` của run `2d33`, `8d9c`): rasterizer chiếm ~9 % thời gian GPU mỗi bước, Adam ~14 %, còn ~77 % là phần riêng của OmniRe. Kernel lớn nhất là `indexing_backward_kernel_small_stride`: ~31,7 ms/bước ở bước 2000 và ~98 ms/bước ở bước 20 000 (tăng theo số Gaussian), ở cả run có và không có Faster-GS.

Kernel này là backward của phép lấy phần tử theo mặt nạ hoặc chỉ số trong PyTorch (`x[mask]`, `x[idx]`): gradient phải được "rải" ngược về đúng vị trí, và khi nhiều phần tử trỏ vào cùng một chỗ thì phải cộng dồn.

## 2. Hai nguyên nhân trong code

**(a) Lấy 5 tensor qua một mặt nạ luôn toàn True.** `get_gaussians` của nền, xe, node biến dạng tạo `filter_mask = torch.ones_like(...)` rồi trả `means[filter_mask]`, `opacities[filter_mask]`, màu, scale, quaternion. Kết quả trùng tensor gốc, nhưng forward vẫn sao chép và backward vẫn rải gradient cho mọi Gaussian, mỗi bước.

**(b) Lấy pose của xe theo chỉ số từng Gaussian.** `RigidNodes.transform_means`, `transform_quats` lấy `pose_khung_này[point_ids]`: hàng trăm nghìn Gaussian cùng đọc từ vài chục pose. Backward của `x[idx]` dùng `index_put_` có cộng dồn: trên GPU nó sắp xếp chỉ số rồi cộng từng nhóm trùng, rất chậm với kiểu "nhiều chỉ số trỏ vào ít ô". `DeformableNodes` cũng lấy embedding theo cách này.

## 3. Vì sao code gốc viết như vậy

- **(a) là chỗ để dành chưa làm xong.** `filter_mask` là giao diện chung của mọi lớp Gaussian: `after_train` dùng `self.filter_mask` để đưa gradient của phần đã vẽ về đúng vị trí. Lớp PVG dùng mặt nạ thật (`pvg.py`: chỉ giữ Gaussian có độ đục theo thời gian > 0,05 ở thời điểm vẽ). Các lớp còn lại để toàn True, kèm ghi chú `# NOTE: hack here, need to consider a gaussian filter for efficient rendering` (`rigid.py`, `smpl.py`): tác giả định lọc Gaussian (ngoài tầm nhìn, xe không có mặt) nhưng chưa làm.
- **(b) là cách viết quen thuộc.** `x[idx]` và `torch.index_select(x, 0, idx)` tương đương về toán học; chênh lệch tốc độ nằm ở cách PyTorch cài backward (sắp xếp rồi cộng, so với cộng nguyên tử `index_add_`). Lý do duy nhất có thể khiến chọn cách cũ là gradient lặp lại từng bit, nhưng DriveStudio không bật chế độ tất định (`use_deterministic_algorithms`) và rasterizer gsplat vốn dùng cộng nguyên tử, nên kết quả train vốn đã không lặp lại từng bit.

Đây là suy luận từ code và comment, không phải lời giải thích của tác giả DriveStudio.

## 4. Cách sửa và kiểm chứng

- (a): dùng thẳng tensor; vẫn giữ `self.filter_mask` (toàn True) nên phần cộng dồn gradient để densify không đổi. Không dùng chung bộ nhớ với tham số, vì `collect_gaussians` gộp các node bằng `torch.cat` (tạo tensor mới). **Nếu sau này cài bộ lọc thật cho các lớp này, phải thêm lại phép lấy qua mặt nạ.**
- (b): `torch.index_select` cho pose xoay, dịch, quaternion của xe và embedding của node biến dạng.

**Đo từng phép** (Quadro T2000, forward + backward):

| Phép | Trước | Sau |
|---|---|---|
| 5 tensor của 1 triệu Gaussian qua mặt nạ toàn True | 36,3 ms | 5,8 ms |
| Pose của 40 xe cho 400 000 Gaussian | 8,6 ms | 2,3 ms |

**Kết quả không đổi:** dựng nền, xe, node biến dạng thật bằng config OmniRe (dữ liệu ngẫu nhiên nhỏ), chạy `get_gaussians` và backward qua một loss có trọng số ngẫu nhiên, so output và gradient của mọi tham số trước và sau khi sửa: 52/57 tensor trùng từng bit; 5 tensor (gradient pose xe, embedding) lệch tối đa 2,5 × 10⁻⁷ tương đối. Test CPU giữ lâu dài: `tests/drive2sim/recon/models/test_node_gather.py` (so với phép lấy theo chỉ số kiểu cũ).

## 5. Sai khác 2,5 × 10⁻⁷ ảnh hưởng thế nào

- **Bằng giới hạn của float32.** Float32 giữ ~7 chữ số có nghĩa (sai số làm tròn ~1,2 × 10⁻⁷ tương đối); phép cộng số thực không có tính kết hợp, nên cộng cùng các số theo thứ tự khác lệch ở chữ số cuối.
- **Code cũ vốn đã có loại sai khác này.** Rasterizer gsplat cộng gradient bằng phép cộng nguyên tử, thứ tự đổi giữa các lần chạy; hai lần chạy code cũ cùng seed đã lệch nhau ở mức này ngay từ bước đầu. Thay đổi này chỉ thêm vài nguồn cùng loại.
- **Qua 30 000 bước, sai khác có thể lớn dần nhưng không có hướng.** Train phi tuyến và có quyết định rẽ nhánh (một Gaussian sát ngưỡng densify có thể bị tách hay không), nên hai lần chạy ra hai tập Gaussian khác nhau, như đổi seed; PSNR, SSIM, LPIPS dao động quanh cùng giá trị, trong biên độ dao động giữa các lần chạy.
- **Tham số bị ảnh hưởng cập nhật rất chậm**: pose xe có learning rate xoay 1 × 10⁻⁵, dịch 5 × 10⁻⁴; Adam chia gradient cho độ lớn trung bình của nó, nên lệch tương đối của bước cập nhật cùng cỡ 10⁻⁷.

**Chưa đo:** biên độ dao động giữa các lần chạy cùng cấu hình (mỗi cấu hình mới chạy một lần). Cách đo: chạy lại một cấu hình (ví dụ OmniRe 3DGS + FastGS + Faster-GS như run `f8a6`) và so PSNR.

## 6. Việc còn lại

- **Đo trên run thật:** OmniRe 3DGS + Faster-GS 2000 bước có profile, so với `profile_step2000.json` của run `8d9c` (cùng cấu hình, 260,7 ms GPU/bước, `indexing_backward` 31,7 ms). Ở bước 2000 kernel này còn nhỏ; mức lợi ở cuối quá trình train (~98 ms ở bước 20 000) lớn hơn.
- **`SMPLNodes`**: cùng hai kiểu code (mặt nạ toàn True, `trans_cur_frame[point_ids]`); sửa sau khi có phép so sánh output và gradient cho node người.
- Run so sánh trong `docs/experiments/run_plan.md` ghi rõ dùng code trước hay sau thay đổi này (commit `5bfcd3d`) khi so thời gian.
