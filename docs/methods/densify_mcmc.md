# Densify bằng MCMC cho OmniRe (3DGUT và các renderer khác)

**Trạng thái:** đề xuất, chưa có code trong OmniRe, chưa đo. Approach `gsplat_3dgut` (cảnh tĩnh, port từ teammate) đã dùng MCMC (`docs/methods/gsplat_3dgut.md`); tài liệu này bàn việc đưa cùng cách đó vào OmniRe.

Liên quan: `densify_3dgut.md` (densify theo gradient và vì sao ngưỡng 3DGS không hợp 3DGUT), `fastgs.md` (FastGS đổi quy tắc densify theo gradient), `docs/experiments/run_plan.md` (số liệu các run).

## 1. Vấn đề cần giải

**Mục tiêu của densify** (chi tiết ở `densify_3dgut.md` mục 1): chỗ ảnh có chi tiết phải có đủ Gaussian nhỏ, và không thừa Gaussian ở chỗ khác. Lúc bắt đầu không biết chỗ nào cần bao nhiêu, nên số Gaussian phải tự điều chỉnh trong lúc train.

**Cách hiện tại của OmniRe** (`drive2sim/recon/models/gaussians/vanilla.py`, `refinement_after`, đã đọc code): mỗi `refine_interval` bước, Gaussian có gradient vị trí trung bình vượt `densify_grad_thresh` thì được nhân bản hoặc tách đôi; Gaussian gần trong suốt hoặc quá to bị xoá; định kỳ đặt lại độ đục (`reset_alpha_interval`).

**Chỗ yếu khi đổi renderer.** Quy tắc trên dựa vào một **ngưỡng có đơn vị**: 3DGS đo gradient theo vị trí trên ảnh (pixel), 3DGUT đo gradient theo vị trí 3D (mét), 3DGEER đo một đại lượng riêng. Cùng ngưỡng 0,0005 cho ra số Gaussian rất khác nhau. Số đo thật trên nuScenes `scene-0061`, `front3`, 30 000 bước, FastGS + Faster-GS:

| Run | Renderer | PSNR | Số Gaussian |
|---|---|---|---|
| `f8a6` | 3DGS | 25,01 | 0,55 triệu |
| `60e5` | 3DGUT, ngưỡng của 3DGS | 23,99 | 0,33 triệu |

Thử đổi gradient 3DGUT về đơn vị pixel (`ut_densify=screen`, run `90dd`, 2 000 bước) thì 3DGUT densify nhiều hơn 5–30 lần nhưng vẫn ít hơn 3DGS 6–12 lần (`densify_3dgut.md`). Nghĩa là vẫn phải dò ngưỡng cho từng renderer, có thể cho từng dataset.

## 2. MCMC từ nguyên lý gốc

Bài báo: Kheradmand và cộng sự, *3D Gaussian Splatting as Markov Chain Monte Carlo* (NeurIPS 2024, [arXiv 2404.09591](https://arxiv.org/abs/2404.09591)); cài đặt dùng ở đây là `gsplat/strategy/mcmc.py` của gsplat-geer 1.5.3 trong repo (đã đọc code).

**Đổi câu hỏi.** Densify theo gradient hỏi: "Gaussian nào đang bị kéo mạnh, nên nhân thêm?". Câu hỏi này cần một ngưỡng. MCMC hỏi câu khác: "với **một ngân sách cố định** N Gaussian, đặt chúng ở đâu thì ảnh khớp nhất?". Không còn ngưỡng gradient; chỉ còn ngân sách.

**Ví dụ đời thường.** Có 100 tờ giấy màu để chép một bức tranh. Tờ nào gần như trong suốt (không góp gì vào ảnh) thì gỡ ra, dán chồng lên chỗ đang có những tờ đậm màu: chỗ đó đang cần nhiều giấy. Thỉnh thoảng xê dịch nhẹ các tờ mờ để chúng thử chỗ mới. Có thêm giấy thì thêm từ từ, cũng dán vào chỗ đang đậm.

**Bốn việc MCMC làm** (giá trị mặc định đọc trong `mcmc.py`):

| Việc | Cách làm | Mặc định gsplat |
|---|---|---|
| Dời Gaussian chết | Gaussian có độ đục ≤ `min_opacity` được chuyển tới vị trí một Gaussian sống, chọn ngẫu nhiên theo độ đục; độ đục và kích thước chia lại để ảnh không đổi đột ngột (kernel `compute_relocation`); trạng thái Adam của chúng đặt về 0 | `min_opacity` 0,005, mỗi `refine_every` 100 bước, từ bước 500 tới 25 000 |
| Thêm Gaussian | Mỗi lần thêm 5 % số hiện có, cũng chọn theo độ đục, tới khi đủ `cap_max` | `cap_max` 1 000 000 |
| Nhiễu vị trí | Sau mỗi bước, cộng nhiễu vào vị trí, mạnh với Gaussian mờ, gần như 0 với Gaussian đậm; giúp Gaussian mờ "đi thử" chỗ khác thay vì kẹt | `noise_lr` 5·10⁵ × learning rate vị trí |
| Điều chuẩn | Loss nhỏ kéo độ đục và kích thước về 0, để Gaussian không cần thiết mờ dần rồi được dời đi | preset `mcmc` của gsplat: `opacity_reg` 0,01, `scale_reg` 0,01 |

**Vì sao hợp với 3DGUT.** Không bước nào dùng gradient trên ảnh (`means2d`), thứ mà 3DGUT không có theo cùng đơn vị. Dấu hiệu duy nhất là **độ đục**, có cùng ý nghĩa ở mọi renderer. Thêm nữa, ngân sách `cap_max` ghi thẳng ra, nên VRAM đoán trước được.

**Bằng chứng đã có (chưa phải của mình):** bảng thử của SimULi trong `examples/AV_TRAINER.md` của gsplat 1.6, teammate chép vào notebook `08_simuli`, đo trên GPU A40, cảnh tĩnh 4 giây, ảnh 480×270: 3DGUT thường 26,46 dB, thêm MCMC 27,97 dB (+1,51). Chưa chạy lại; không phải OmniRe, không có xe hay người chuyển động.

## 3. Áp vào từng loại node của OmniRe

Mỗi node của OmniRe giữ thêm dữ liệu riêng cho từng Gaussian, nên khi dời hay thêm Gaussian phải chép theo (đã đọc `drive2sim/recon/models/nodes/`):

| Node | Dữ liệu riêng mỗi Gaussian | Hiện densify? | MCMC áp được không |
|---|---|---|---|
| Nền (`Background`, khoảng 70 % số Gaussian ở `f8a6`) | không | có | **Được**, đơn giản nhất |
| Xe (`RigidNodes`) | `point_ids`: Gaussian thuộc xe nào; vị trí lưu trong hệ toạ độ của xe | có | **Được**, khi dời hay thêm phải chép `point_ids` của Gaussian nguồn; nhiễu cộng trong hệ toạ độ của xe |
| Vật biến dạng (`DeformableNodes`, kế thừa `RigidNodes`) | `point_ids`, chọn embedding của vật cho mạng biến dạng | có | **Được**, như xe |
| Người (`SMPLNodes`) | đúng 6 890 Gaussian mỗi người, gắn với đỉnh của mô hình SMPL, có láng giềng KNN cho loss điều chuẩn | **không**: code ghi "SMPL gaussians are never densified or culled" | **Không áp**: không có densify để thay; dời Gaussian sẽ làm vỡ liên kết với đỉnh SMPL |

Vậy "áp MCMC cho toàn bộ" nghĩa là nền, xe và vật biến dạng; người giữ nguyên như bản gốc.

**Những chỗ phải xử lý khi cài:**

1. **Optimizer.** Các hàm của gsplat cần mỗi tham số một optimizer; OmniRe dùng một optimizer chung chia nhóm tham số. Dùng hàm có sẵn của OmniRe (`dup_in_optim`, `remove_from_optim`), thêm bước đặt trạng thái Adam về 0 cho Gaussian bị dời.
2. **Ngân sách.** Mỗi node một `cap_max` riêng. Gợi ý ban đầu: lấy số Gaussian của 3DGS ở `f8a6` (nền 0,39 triệu, xe 29 nghìn, vật biến dạng 5 nghìn) để so công bằng về bộ nhớ; chưa thử.
3. **Tắt những thứ xung đột:** đặt lại độ đục định kỳ (`reset_alpha_interval`; MCMC dựa vào độ đục để chọn, đặt lại sẽ làm mọi Gaussian thành "chết"), và FastGS (FastGS sửa quy tắc densify theo gradient, MCMC thay hẳn quy tắc đó).
4. **Điều chuẩn** cộng thêm vào loss của OmniRe, chỉ cho các node dùng MCMC.
5. **Không phụ thuộc renderer:** lựa chọn `densify: default | mcmc` dùng được với 3DGS, 3DGUT, 3DGEER.

## 4. Kiểm chứng

- **Test:** dời và thêm Gaussian giữ đúng `point_ids` và số Gaussian không vượt `cap_max`; Gaussian dời đi có trạng thái Adam bằng 0; node người không đổi.
- **Run so sánh** (nuScenes `scene-0061`, `front3`, 30 000 bước, FastGS tắt, Faster-GS bật):
  - 3DGUT + MCMC so với `60e5` (3DGUT, 23,99 dB) và `f8a6` (3DGS, 25,01 dB);
  - nếu tốt, chạy thêm 3DGS + MCMC để biết MCMC tự nó có lợi không, hay chỉ bù cho 3DGUT.
- **Ngưỡng chấp nhận** (đề xuất): 3DGUT + MCMC thu hẹp ít nhất một nửa khoảng cách 1 dB với 3DGS, thời gian train không tăng quá 20 %.

## 5. Chưa biết

- Mức tăng trên OmniRe: con số +1,5 dB của SimULi đo trên cảnh tĩnh ngắn, renderer và dữ liệu khác.
- `noise_lr` và hệ số điều chuẩn có cần chỉnh cho cảnh lái xe dài, nhiều node hay không.
- MCMC cho xe: Gaussian của một xe có thể được dời sang xe khác (chọn theo độ đục trên cả node); có thể cần giới hạn chọn trong cùng xe nếu xe nhỏ bị mất hết Gaussian.
