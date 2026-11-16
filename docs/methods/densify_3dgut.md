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

**Suy luận (chưa đo).** Đại lượng của 3DGS (`base.py`, nhánh `default`) là gradient theo tâm trên ảnh **tính bằng pixel rồi nhân `W/2`, `H/2`** (đổi sang toạ độ chuẩn hoá [−1, 1] như 3DGS gốc); đại lượng của 3DGUT là gradient theo tâm 3D (mét), không nhân gì. Một đốm ở độ sâu `z` trước camera tiêu cự `f` (pixel): dời đốm 1 m theo phương ngang thì điểm chiếu dời khoảng `f / z` pixel, nên theo quy tắc chuỗi gradient 3D ≈ gradient pixel × `f / z`. Tỉ lệ giữa hai đại lượng:

```
đại lượng 3DGUT / đại lượng 3DGS ≈ (f / z) / (W / 2) = 2f / (W · z)
```

Ảnh nuScenes 800 px ngang (`f` ≈ 633 px): ≈ 1,58 / z (z tính bằng mét), tức bằng nhau ở 1,6 m, nhỏ hơn ~3 lần ở 5 m, ~13 lần ở 20 m, ~30 lần ở 50 m.

**Yếu tố thứ hai, cùng hướng: `absgrad`.** OmniRe đặt `trainer.render.absgrad: true`. Với 3DGS, gsplat cộng **độ lớn** gradient từ từng pixel (không triệt tiêu); với 3DGUT, nhánh `ut` lấy giá trị tuyệt đối **sau khi** cộng gradient có dấu, nên các phần ngược chiều triệt tiêu trước. Tổng có dấu không lớn hơn tổng độ lớn, nên đại lượng 3DGUT còn nhỏ hơn nữa, theo mức tuỳ cảnh.

**Hệ quả dự đoán:** với cùng ngưỡng 0,0005, phần lớn đốm 3DGUT **không** vượt ngưỡng, nên 3DGUT **densify quá ít** (ít Gaussian, có thể mờ), không phải quá nhiều. (Bản trước của mục này dự đoán ngược lại vì bỏ sót hệ số `W/2`; sửa 15/11/2026.)

**Kiểm tra:** run chạy thử `ca4b` (3DGUT nhánh Bosch) và `ac56` (3DGS) ghi phân bố đại lượng densify (`densify_stats.jsonl`); lý thuyết đúng nếu các phân vị của 3DGUT nhỏ hơn của 3DGS cỡ 1,58 / z với z điển hình của cảnh (vài đến vài chục mét), có thể nhỏ hơn nữa do `absgrad`, và tỉ lệ đốm vượt ngưỡng của 3DGUT thấp hơn hẳn.

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

**Xấp xỉ.** 3DGUT chiếu Gaussian bằng các điểm sigma (unscented transform), không qua đúng một tâm 2D; `g3d` còn phần do hình dạng ảnh chiếu đổi theo vị trí. Phép đổi giữ phần đi qua tâm, nên kết quả gần chứ không trùng 3DGS. Phép đổi cũng **không khôi phục được `absgrad`** (mục 3): thông tin từng pixel đã mất khi cộng, nên sau khi đổi, đại lượng 3DGUT có lẽ vẫn nhỏ hơn 3DGS một phần tuỳ cảnh. Nếu phần lệch này lớn, cách so công bằng là đặt 3DGS cũng không `absgrad`, hoặc nhân một hệ số đo được. Mỗi bước train chỉ dùng một ảnh, nên `J` của bước đó xác định rõ (giống 3DGS cộng dồn gradient trên ảnh theo từng bước).

**Cài đặt** (`drive2sim/recon/trainers/screen_grad.py`): `ScreenGradMultiTrainer` là lớp con của `MultiTrainer` (OmniRe, Street GS); với renderer `ut`, trước khi gọi `postprocess_per_train_step` của lớp cha, nó thay `info["means3d"].grad` bằng `[g2d_x, g2d_y, 0]` đã nhân hệ số (code densify chỉ dùng độ lớn của vectơ). `J` tính bằng sai phân trung tâm qua `project_camera_model` (bước bằng 1/1000 khoảng cách tới camera). Code của Bosch không đổi. Bật: `--set train.drivestudio.ut_densify=screen` (đặt `trainer.type`); mặc định `native` là nhánh của Bosch.

**Đã kiểm tra** (test CPU, PyTorch thật): dựng `g3d = Jᵀ g2d` với `J` tính bằng autograd, module tính ngược đúng `g2d` (sai số tương đối dưới 1e-4) với camera pinhole và mắt cá OpenCV, camera xoay trong world; điểm sau camera pinhole cho 0; cùng `g3d`, điểm xa gấp đôi cho `g2d` gấp đôi (`g2d ≈ g3d · z/f`).

**Kết quả bước 1** (16/11/2026; run `ca4b` 3DGUT nhánh Bosch, `ac56` 3DGS; cùng OmniRe + FastGS + Faster-GS, nuScenes `scene-0061` `front3`, 2000 bước; Gaussian nền):

| Bước | Tỉ lệ vượt ngưỡng 3DGUT | Tỉ lệ vượt ngưỡng 3DGS | Phân vị 50 / 90 / 99 %, 3DGUT ÷ 3DGS |
|---|---|---|---|
| 100 | 0,008 % | 0,48 % | 0,13 / 0,16 / 0,15 |
| 300 | 0,007 % | 1,30 % | 0,10 / 0,11 / 0,10 |
| 600 | 0,006 % | 2,28 % | 0,07 / 0,07 / 0,07 |
| 1000 | 0,017 % | 4,41 % | 0,04 / 0,04 / 0,05 |
| 1900 | 0,040 % | 4,60 % | 0,03 / 0,04 / 0,07 |

- **Đúng chiều dự đoán (mục 3 đã sửa):** tỉ lệ Gaussian vượt ngưỡng của 3DGUT thấp hơn 3DGS 60–380 lần; số Gaussian nền của 3DGUT giảm (0,96 → 0,88 triệu) trong khi 3DGS tăng (→ 1,14 triệu).
- **Đúng cỡ lúc đầu:** ở bước 100 tỉ lệ ~0,13–0,16, ứng với 1,58 / z ở z ≈ 10–12 m, hợp với khoảng cách điển hình của cảnh.
- **Lệch so với lý thuyết:** tỉ lệ giảm dần theo bước train (0,15 → 0,04), trong khi `1,58 / z` chỉ phụ thuộc hình học. Đại lượng của 3DGS tăng dần (phân vị 99 %: 3,1e-4 → 1,1e-3) còn của 3DGUT gần như đứng yên (~5–8e-5). Một nguyên nhân có thể: 3DGS dùng `absgrad` (cộng độ lớn gradient từng pixel), 3DGUT lấy độ lớn sau khi cộng, nên khi ảnh render đã gần đúng, các gradient pixel triệt tiêu nhau ở 3DGUT mà không triệt tiêu ở 3DGS. **Chưa kiểm chứng.**
- Hệ quả: một ngưỡng cố định chia 10 không đủ (tỉ lệ đổi theo thời gian), nên gradient tương đương pixel (bước 2) đáng thử trước; nó vẫn không lấy lại được phần `absgrad` (mục 5).

**Kết quả bước 2** (16/11/2026; run `90dd` 3DGUT `ut_densify=screen`, so với `ca4b` 3DGUT gốc và `ac56` 3DGS; cùng OmniRe + FastGS + Faster-GS, nuScenes `scene-0061` `front3`, 2000 bước; Gaussian nền). Lần chạy đầu của `90dd` lỗi ở bước khoảng 600: `torch.linalg.solve` gặp ma trận suy biến (Jacobian chiếu suy biến, giảm chấn quá nhỏ bị float32 làm tròn); đã đổi sang nghịch đảo 2×2 trực tiếp, điểm suy biến cho gradient 0 (commit `dc94395`).

| Bước | Vượt ngưỡng: 3DGUT gốc | 3DGUT `screen` | 3DGS |
|---|---|---|---|
| 100 | 0,008 % | 0,042 % | 0,484 % |
| 300 | 0,007 % | 0,106 % | 1,301 % |
| 600 | 0,006 % | 0,187 % | 2,282 % |
| 1000 | 0,017 % | 0,507 % | 4,408 % |
| 1900 | 0,040 % | 0,806 % | 4,599 % |

| Run | PSNR test | SSIM | LPIPS | PSNR train trung bình | Gaussian nền ở bước 1900 |
|---|---|---|---|---|---|
| `ca4b` 3DGUT gốc | 22,34 | 0,708 | 0,436 | 21,07 | 0,88 triệu |
| `90dd` 3DGUT `screen` | 21,24 | 0,700 | 0,431 | 21,18 | 0,92 triệu |
| `ac56` 3DGS | 21,60 | 0,701 | 0,418 | 21,20 | 1,14 triệu |

- `screen` tăng tỉ lệ vượt ngưỡng 5–30 lần so với nhánh gốc, nhưng vẫn thấp hơn 3DGS 6–12 lần: phép đổi sang gradient tương đương pixel mới bù được một phần. Phần còn thiếu hợp với giả thuyết `absgrad` ở mục 5 (3DGS cộng độ lớn gradient từng pixel; 3DGUT lấy độ lớn sau khi cộng, các gradient pixel triệt tiêu nhau) — **chưa kiểm chứng**.
- PSNR ở 2000 bước gần nhau và thứ tự không khớp lượng Gaussian (3DGUT gốc cao nhất ở test); một seed, 2000 bước, chưa đủ để nói `screen` tốt hay kém hơn về chất lượng.

Kế hoạch còn lại:
3. Lặp lại bước 2 trên KITTI-360 (độ phân giải khác, có mắt cá): nếu vẫn gần, phép đổi không phụ thuộc dataset.
4. Run 3DGUT 30 000 bước với `ut_densify=screen` để so renderer.

## 6. Gắn vào code mà không sửa patch của Bosch

Patch của Bosch (`base.py`, `postprocess_per_train_step`) giữ nguyên, để luôn chạy lại được đúng bản của Bosch và so sánh:

- **Ghi thống kê densify** (đã có): `drive2sim/recon/utils/densify_stats.py`, bật bằng `trainer.densify_stats.enabled` (stage: `--set train.drivestudio.densify_stats=true`), mặc định tắt. Ngay trước mỗi lần refine, mỗi lớp Gaussian ghi một dòng vào `densify_stats.jsonl` trong thư mục stage: các phân vị 50/75/90/95/99 % của đại lượng densify (trung bình trên các ảnh đã thấy Gaussian), ngưỡng, số và tỉ lệ Gaussian vượt ngưỡng. Chỉ đọc; móc vào `VanillaGaussians.postprocess_per_train_step` (code DriveStudio), không đụng nhánh `ut` trong `base.py`.
- **Ngưỡng đã chỉnh**: một config biến thể kế thừa bằng `extends` (ví dụ `omnire_ut_calibrated.yaml` chỉ khác `omnire_ut.yaml` ở `densify_grad_thresh`), cộng một mục cấu hình của approach để chọn; config gốc của Bosch vẫn còn để so.
- **Gradient tương đương pixel** (đã có, mục 5): lớp trainer con chọn bằng `trainer.type`, không sửa nhánh `ut` trong `base.py`.

Run so sánh luôn ghi config nào đã dùng (`run.json`, `summary.json`), nên phân biệt được ngưỡng Bosch và ngưỡng đã chỉnh.
