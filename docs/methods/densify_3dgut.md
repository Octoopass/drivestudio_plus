# Densify và ngưỡng densify của 3DGUT

**Trạng thái:** giải thích và kế hoạch. Ngưỡng densify của 3DGUT **chưa được chỉnh**: config `omnire_ut*.yaml`, `streetgs_ut*.yaml` dùng lại ngưỡng của 3DGS. Cách chỉnh (mục 4) và cách gắn vào code mà không sửa patch của Bosch (mục 5) chưa làm.

Liên quan: `fastgs.md` (FastGS đổi quy tắc densify), `fastergs.md` (tăng tốc không đổi densify), `3dgeer.md` (renderer 3DGEER và ngưỡng riêng của nó).

## 1. Densify từ nguyên lý gốc

**Mục tiêu.** 3DGS tìm một tập Gaussian (đốm màu mờ trong 3D: vị trí, kích thước, hướng, độ đục, màu) sao cho vẽ lên ảnh thì giống ảnh chụp. Từ đó có hai điều phải đúng:

1. Chỗ có chi tiết (cạnh biển báo, vạch kẻ đường) phải có **đủ đốm nhỏ**: một đốm chỉ vẽ được một vệt màu trơn.
2. **Không thừa đốm**: mỗi đốm tốn bộ nhớ, và mỗi bước train tính và cập nhật mọi đốm; thừa đốm thì chậm, tốn VRAM mà ảnh không đẹp hơn.

Lúc bắt đầu không biết chỗ nào cần bao nhiêu đốm: đốm khởi tạo từ điểm LiDAR, thưa ở xa và gần như không có trên trời, mái nhà. Nên số đốm phải tự điều chỉnh trong lúc train: thêm đốm là **densify**, bớt đốm là **prune**.

**Dấu hiệu thiếu đốm.** Ví dụ: dán giấy màu tròn lên tường để chép một bức tranh. Một tờ cứ bị kéo sang phải để khớp màu bên phải, rồi bị kéo sang trái để khớp màu bên trái: một tờ không vừa đỏ bên này vừa xanh bên kia được. Bị kéo mạnh và liên tục là dấu hiệu chỗ đó cần thêm giấy. Trong 3DGS, "lực kéo" là **gradient theo vị trí** của đốm: model muốn dời đốm đi bao nhiêu để ảnh khớp hơn.

**Trong code DriveStudio** (`drive2sim/recon/models/gaussians/vanilla.py`, `after_train`, `refinement_after`), đã đọc code:

- mỗi `refine_interval` bước (OmniRe: 100), lấy trung bình độ lớn lực kéo của từng đốm qua các ảnh đã thấy nó;
- đốm vượt `densify_grad_thresh` (3DGS: 0,0005) thì densify: đốm nhỏ (dưới `densify_size_thresh × scene_radius`) **nhân bản** (clone), đốm lớn **tách đôi** (split);
- prune: đốm gần trong suốt (`cull_alpha_thresh`) hoặc quá to (`cull_scale_thresh × scene_radius`) bị xoá;
- việc tách dừng ở `stop_split_at` (OmniRe: 15 000).

**Điểm yếu.** Lực kéo lớn chỉ là dấu hiệu gián tiếp, không phải bằng chứng ảnh đang sai: một đốm có thể bị kéo mạnh vì nằm ở rìa ảnh hay vì góc nhìn lạ, trong khi ảnh chỗ đó đã đúng; và lực kéo đo trên từng ảnh một. Nên densify gốc thêm thừa khá nhiều đốm (run OmniRe gốc `2d33` trên nuScenes `scene-0061`: 1,52 triệu Gaussian). FastGS sửa đúng điểm này (`fastgs.md`).

## 2. Mỗi renderer đo "lực kéo" bằng đại lượng khác nhau

`drive2sim/recon/trainers/base.py`, `postprocess_per_train_step` (từ patch Bosch 3DGEER/3DGUT, commit `389f806`), đã đọc code:

| Renderer (`render_mode`) | Đại lượng densify | Đơn vị | Ngưỡng trong config |
|---|---|---|---|
| 3DGS (`default`) | gradient theo vị trí trên ảnh (`means2d`), nhân `W/2`, `H/2` | pixel | 0,0005 (giá trị của 3DGS) |
| 3DGEER (`geer`) | `geer_gradient` (tự cộng giá trị tuyệt đối và nhân theo khoảng cách) | riêng của 3DGEER | 0,01, Bosch chỉnh (`omnire_geer.yaml`, `refine_interval: 300`) |
| 3DGUT (`ut`) | gradient theo vị trí 3D (`means3d`) | mét | 0,0005, **dùng lại của 3DGS** |

`3dgeer.md` (README của Bosch) ghi: thang gradient của GEER khác 3DGS và 3DGUT, nên ngưỡng giữa chúng không thay cho nhau được. Bosch chỉnh ngưỡng cho GEER nhưng không có config 3DGUT với ngưỡng riêng; `streetgs_ut.yaml` của teammate ghi "Densification values are the 3DGS defaults. Untested."

## 3. Vì sao ngưỡng của 3DGS không hợp với 3DGUT

**Suy luận (chưa đo).** Một đốm ở độ sâu `z` trước camera tiêu cự `f` (pixel): dời đốm 1 mét theo phương ngang thì điểm chiếu dời khoảng `f / z` pixel. Theo quy tắc chuỗi, gradient theo vị trí 3D ≈ gradient theo pixel × `f / z`. Với ảnh nuScenes 800 px ngang (`f` khoảng 630 px), đốm cách 20 m có gradient 3D lớn hơn gradient pixel khoảng 30 lần; đốm gần lớn hơn nữa, đốm rất xa nhỏ hơn. Giống như đọc nhiệt kế độ F bằng mốc "sốt 38" của độ C.

**Hệ quả dự đoán:** với ngưỡng 0,0005, phần lớn đốm 3DGUT vượt ngưỡng, số đốm tăng nhanh (dễ hết VRAM), và đốm gần camera được ưu tiên quá mức so với đốm xa. FastGS che bớt (đốm phải qua cả cửa gradient lẫn cửa điểm sai), nhưng khi cửa gradient quá dễ qua thì mọi thứ dựa vào điểm sai.

## 4. Cách chỉnh (kế hoạch)

Chọn ngưỡng theo số đo, như Bosch làm với 3DGEER, sao cho hai renderer densify với **cùng cường độ**:

1. **Chạy thử ngắn** OmniRe + 3DGUT trên cùng cảnh, khoảng 2000 bước, có ghi thống kê của đại lượng densify ở mỗi lần refine (các phân vị, phần trăm đốm vượt ngưỡng).
2. **Mốc:** cùng thống kê với 3DGS (run `f8a6`, hoặc chạy ngắn cùng số bước): phần trăm đốm vượt 0,0005 ở mỗi lần refine.
3. **Chọn ngưỡng 3DGUT** là phân vị cho cùng phần trăm đó (khớp phân vị).
4. **Kiểm chứng:** run 3DGUT đủ 30 000 bước với ngưỡng mới; số Gaussian cuối và VRAM phải cùng cỡ run 3DGS tương ứng; ghi ngưỡng và số đo làm căn cứ.

Hướng khác (chưa chọn): đổi gradient 3D về "tương đương pixel" bằng cách nhân `z / f` cho từng đốm trước khi so ngưỡng, để dùng lại được 0,0005. Đúng về nguyên lý hơn, nhưng phải chọn `f`, `z` của camera nào khi một đốm thấy từ nhiều camera; khớp phân vị đơn giản hơn và đủ cho việc so renderer.

## 5. Gắn vào code mà không sửa patch của Bosch (kế hoạch)

Patch của Bosch (`base.py`, `postprocess_per_train_step`) giữ nguyên, để luôn chạy lại được đúng bản của Bosch và so sánh:

- **Ghi thống kê densify**: một module riêng (cùng kiểu `drive2sim/recon/speedups/`), bật bằng một khối config (ví dụ `trainer.densify_stats.enabled`), mặc định tắt; chỉ đọc gradient đã có, không đổi việc densify.
- **Ngưỡng đã chỉnh**: một config biến thể kế thừa bằng `extends` (ví dụ `omnire_ut_calibrated.yaml` chỉ khác `omnire_ut.yaml` ở `densify_grad_thresh`), cộng một mục cấu hình của approach để chọn; config gốc của Bosch vẫn còn để so.
- Nếu sau này cần đổi về gradient tương đương pixel (mục 4, hướng khác): cũng là module riêng bật bằng config, không sửa nhánh `ut` trong `base.py`.

Run so sánh luôn ghi config nào đã dùng (`run.json`, `summary.json`), nên phân biệt được ngưỡng Bosch và ngưỡng đã chỉnh.
