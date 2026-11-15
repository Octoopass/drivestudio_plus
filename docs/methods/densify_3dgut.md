# Densify và ngưỡng densify của 3DGUT

**Trạng thái:** nhánh 3DGUT của Bosch dùng lại ngưỡng của 3DGS cho một đại lượng khác thang đo. Đã có: module ghi thống kê densify, và trainer đổi gradient 3DGUT về gradient tương đương pixel (mục 5) để dùng lại ngưỡng 3DGS mà không phụ thuộc dataset; cả hai bật bằng tham số, mặc định tắt. Đang chạy thử trên nuScenes; chưa có kết quả.

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

### Ngưỡng chỉnh được phụ thuộc gì

Theo công thức trên (gradient 3D ≈ gradient pixel × `f / z`), tỉ lệ giữa hai đại lượng phụ thuộc cả renderer lẫn dataset: tiêu cự tính bằng pixel `f` (camera và độ phân giải lúc train; nuScenes 800×450: f ≈ 630 px, KITTI-360 pinhole 704×188: f ≈ 276 px), phân bố khoảng cách `z` của Gaussian (cảnh), và loại camera (mắt cá không có một `f` cố định: số pixel dời khi dời 1 m đổi theo góc lệch trục). Ngưỡng 3DGS ít phụ thuộc hơn vì đã đo bằng pixel. Nên ngưỡng 3DGUT khớp phân vị trên một cảnh, một độ phân giải chỉ chắc đúng ở đó; đổi dataset hoặc độ phân giải thì đo lại (bật `densify_stats` ở run chạy thử). Nếu ngưỡng chỉnh trên nuScenes cho tỉ lệ vượt ngưỡng lệch nhiều trên KITTI-360, đó là lý do làm hướng "gradient tương đương pixel" ở mục 4.

## 4. Cách chỉnh (kế hoạch)

Chọn ngưỡng theo số đo, như Bosch làm với 3DGEER, sao cho hai renderer densify với **cùng cường độ**:

1. **Chạy thử ngắn** OmniRe + 3DGUT trên cùng cảnh, khoảng 2000 bước, có ghi thống kê của đại lượng densify ở mỗi lần refine (các phân vị, phần trăm đốm vượt ngưỡng).
2. **Mốc:** cùng thống kê với 3DGS (run `f8a6`, hoặc chạy ngắn cùng số bước): phần trăm đốm vượt 0,0005 ở mỗi lần refine.
3. **Chọn ngưỡng 3DGUT** là phân vị cho cùng phần trăm đó (khớp phân vị).
4. **Kiểm chứng:** run 3DGUT đủ 30 000 bước với ngưỡng mới; số Gaussian cuối và VRAM phải cùng cỡ run 3DGS tương ứng; ghi ngưỡng và số đo làm căn cứ.

Hướng tổng quát hơn (đã chọn, mục 5): đổi gradient 3D về gradient tương đương pixel qua Jacobian của phép chiếu, dùng lại ngưỡng của 3DGS.

## 5. Đổi về gradient tương đương pixel: không phụ thuộc dataset

Khớp phân vị (mục 4) cho một ngưỡng chỉ đúng với một cảnh, một độ phân giải. Cách tổng quát là đổi chính đại lượng densify của 3DGUT về đại lượng mà 3DGS dùng, rồi dùng lại ngưỡng của 3DGS.

**Lý thuyết.** Gọi `u` là vị trí tâm Gaussian trên ảnh (pixel), `X` là vị trí tâm trong world. Phép chiếu `u = π(X)` có Jacobian `J = ∂u/∂X` (ma trận 2×3); với pinhole, trong hệ camera, `J = [[fx/z, 0, −fx·x/z²], [0, fy/z, −fy·y/z²]]`; với mắt cá và F-Theta là đạo hàm của mô hình tương ứng tại góc của điểm. Nếu loss chỉ phụ thuộc `X` qua `u` thì theo quy tắc chuỗi `g3d = Jᵀ · g2d` (`g3d = ∂L/∂X`, `g2d = ∂L/∂u`). Biết `g3d` (3DGUT có), tìm `g2d` bằng nghiệm bình phương tối thiểu:

```
g2d = (J Jᵀ)⁻¹ J · g3d
```

`J` chứa tiêu cự, khoảng cách và mô hình ống kính của đúng camera đã render bước đó, nên ba yếu tố làm ngưỡng phụ thuộc dataset (mục 3) bị khử trong phép đổi. Rồi nhân `W/2`, `H/2` và `batch_size` như nhánh 3DGS của `base.py`, so với cùng `densify_grad_thresh` của 3DGS.

**Xấp xỉ.** 3DGUT chiếu Gaussian bằng các điểm sigma (unscented transform), không qua đúng một tâm 2D; `g3d` còn phần do hình dạng ảnh chiếu đổi theo vị trí. Phép đổi giữ phần đi qua tâm, nên kết quả gần chứ không trùng 3DGS. Mỗi bước train chỉ dùng một ảnh, nên `J` của bước đó xác định rõ (giống 3DGS cộng dồn gradient trên ảnh theo từng bước).

**Cài đặt** (`drive2sim/recon/trainers/screen_grad.py`): `ScreenGradMultiTrainer` là lớp con của `MultiTrainer` (OmniRe, Street GS); với renderer `ut`, trước khi gọi `postprocess_per_train_step` của lớp cha, nó thay `info["means3d"].grad` bằng `[g2d_x, g2d_y, 0]` đã nhân hệ số (code densify chỉ dùng độ lớn của vectơ). `J` tính bằng sai phân trung tâm qua `project_camera_model` (bước bằng 1/1000 khoảng cách tới camera). Code của Bosch không đổi. Bật: `--set train.drivestudio.ut_densify=screen` (đặt `trainer.type`); mặc định `native` là nhánh của Bosch.

**Đã kiểm tra** (test CPU, PyTorch thật): dựng `g3d = Jᵀ g2d` với `J` tính bằng autograd, module tính ngược đúng `g2d` (sai số tương đối dưới 1e-4) với camera pinhole và mắt cá OpenCV, camera xoay trong world; điểm sau camera pinhole cho 0; cùng `g3d`, điểm xa gấp đôi cho `g2d` gấp đôi (`g2d ≈ g3d · z/f`).

**Chưa kiểm tra trên run thật.** Kế hoạch:
1. Run chạy thử 2000 bước OmniRe 3DGUT với `ut_densify=native` và 3DGS (đều bật `densify_stats`) cho phân bố thô; tỉ lệ giữa hai phân bố phải cỡ `f/z` như mục 3 dự đoán.
2. Chạy thử 2000 bước 3DGUT với `ut_densify=screen`: tỉ lệ Gaussian vượt ngưỡng phải gần 3DGS mà không chỉnh gì.
3. Lặp lại bước 2 trên KITTI-360 (độ phân giải khác, có mắt cá): nếu vẫn gần, phép đổi không phụ thuộc dataset.
4. Run 3DGUT 30 000 bước với `ut_densify=screen` để so renderer.

## 6. Gắn vào code mà không sửa patch của Bosch

Patch của Bosch (`base.py`, `postprocess_per_train_step`) giữ nguyên, để luôn chạy lại được đúng bản của Bosch và so sánh:

- **Ghi thống kê densify** (đã có): `drive2sim/recon/utils/densify_stats.py`, bật bằng `trainer.densify_stats.enabled` (stage: `--set train.drivestudio.densify_stats=true`), mặc định tắt. Ngay trước mỗi lần refine, mỗi lớp Gaussian ghi một dòng vào `densify_stats.jsonl` trong thư mục stage: các phân vị 50/75/90/95/99 % của đại lượng densify (trung bình trên các ảnh đã thấy Gaussian), ngưỡng, số và tỉ lệ Gaussian vượt ngưỡng. Chỉ đọc; móc vào `VanillaGaussians.postprocess_per_train_step` (code DriveStudio), không đụng nhánh `ut` trong `base.py`.
- **Ngưỡng đã chỉnh**: một config biến thể kế thừa bằng `extends` (ví dụ `omnire_ut_calibrated.yaml` chỉ khác `omnire_ut.yaml` ở `densify_grad_thresh`), cộng một mục cấu hình của approach để chọn; config gốc của Bosch vẫn còn để so.
- **Gradient tương đương pixel** (đã có, mục 5): lớp trainer con chọn bằng `trainer.type`, không sửa nhánh `ut` trong `base.py`.

Run so sánh luôn ghi config nào đã dùng (`run.json`, `summary.json`), nên phân biệt được ngưỡng Bosch và ngưỡng đã chỉnh.
