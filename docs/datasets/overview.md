# So sánh các dataset lái xe

Tổng hợp camera, độ dài cảnh và quãng đường của các dataset mà tool hỗ trợ (và vài dataset có camera mắt cá chưa hỗ trợ), để chọn dataset thử nghiệm. Hướng dẫn tải và ingest từng dataset nằm ở `<dataset>.md` cùng thư mục.

Nguồn số liệu:
- Tên và thứ tự camera: code ingest của repo (`drive2sim/ingest/drivestudio.py`, `drive2sim/ingest/kitti360.py`).
- Số cảnh, độ dài, tần số: trang chính thức hoặc paper của dataset (danh sách cuối file), tra ngày 15/11/2026.
- Quãng đường, tốc độ: chỉ có ở cảnh đã đo bằng pose sau khi ingest; ô "chưa đo" là chưa có dữ liệu trong tay.

## 1. Vị trí camera

| Dataset | Số camera | Vị trí | Phủ 360° | Mô hình ống kính |
|---|---|---|---|---|
| nuScenes | 6 | trước, trước-trái, trước-phải, sau-trái, sau-phải, sau | Có | pinhole |
| Waymo Open | 5 | trước, trước-trái, trước-phải, hông trái, hông phải | Không (thiếu phía sau) | pinhole |
| Argoverse 2 | 7 vòng quanh (+2 stereo phía trước, tool không dùng) | trước giữa, trước-trái, trước-phải, hông trái, hông phải, sau-trái, sau-phải | Có | pinhole |
| PandaSet | 6 | trước, trước-trái, trước-phải, trái, phải, sau | Có | pinhole |
| nuPlan | 8 | F0 (trước), L0/R0, L1/R1, L2/R2 (hai bên), B0 (sau) | Có | pinhole |
| KITTI | 2 | cặp stereo nhìn thẳng phía trước | Không | pinhole |
| KITTI-360 | 4 | 2 nhìn thẳng phía trước (stereo) + 2 mắt cá 180° hai bên hông | Gần như có (thiếu phía sau) | pinhole + mắt cá (Mei, đổi sang OpenCV fisheye) |
| PhysicalAI (NVIDIA) | 7 | trước góc rộng 120°, trước tele 30°, chéo trái/phải 120°, sau-trái/sau-phải 70°, sau tele 30° | Có | F-Theta (góc rộng; trang NCore ghi Kannala–Brandt equidistant) |

## 2. Nhóm theo loại camera

| Nhóm | Dataset | Tool hỗ trợ |
|---|---|---|
| Chỉ pinhole | nuScenes, Waymo, Argoverse 2, PandaSet, nuPlan, KITTI | Có |
| Pinhole + mắt cá | KITTI-360 | Có: converter có box 3D, loader vẽ mắt cá trực tiếp (renderer 3DGEER hoặc 3DGUT); chưa train thật |
| Góc rộng không phải pinhole | PhysicalAI (F-Theta) | Có loader; chưa chạy thật trong dự án |
| Chỉ mắt cá | WoodScape (Valeo): 4 mắt cá quanh xe, có LiDAR, có box 3D | Chưa: cần converter và loader mới |
| Chưa xác nhận | ZOD (Zenseact): camera trước 120°; nguồn đã đọc không nói là mắt cá | Chưa |

## 3. Độ dài cảnh, số ảnh, quãng đường, tốc độ

| Dataset | Độ dài một cảnh | Tần số camera | Số khung mỗi camera mỗi cảnh | Quãng đường, tốc độ |
|---|---|---|---|---|
| nuScenes | ~20 s (1000 cảnh) | 12 Hz; box ở keyframe 2 Hz, DriveStudio nội suy lên 10 Hz | ~190–200 sau nội suy | Đo `scene-0061`: 191 khung, 91 m, ~17 km/h |
| Waymo Open | 20 s (2030 đoạn) | 10 Hz | ~200 | Chưa đo |
| Argoverse 2 | Nguồn không thống nhất: paper ghi 30 s, nơi khác ~20 s (1000 cảnh) | 20 Hz | ~400–600 | Chưa đo |
| PandaSet | 8 s (103 cảnh) | ~10 Hz (48 000 ảnh / 103 cảnh / 6 camera ≈ 78) | ~80 | Chưa đo |
| PhysicalAI | 20 s (hơn 300 000 đoạn) | Chưa kiểm | Chưa kiểm | Chưa đo |
| ZOD | ~20 s (1473 đoạn); 29 lần chạy dài vài phút | 10 Hz | ~200 | Chưa đo |
| KITTI-360 | 9 sequence dài, tổng 73,7 km (trung bình ~8 km) | Giả định 10 Hz (chưa thấy nguồn chính thức xác nhận; bản tải về không có `timestamps.txt`) | 0003: 1031 | Đo 0003: 1378 m (từ pose); nếu 10 Hz thì 103 s, ~48 km/h. 0003 là một trong các sequence ngắn nhất |
| nuPlan, KITTI | Chưa kiểm | Chưa kiểm | Chưa kiểm | Chưa đo |

## 4. Với yêu cầu đoạn đường từ 1 km

- Phần lớn dataset chỉ có đoạn 8–30 s, tức vài chục đến vài trăm mét mỗi cảnh; không cảnh nào đạt 1 km.
- Có đường liền dài hơn 1 km: KITTI-360 (mọi sequence), ZOD Drives (vài phút mỗi lần chạy, tool chưa hỗ trợ), nuPlan (log dài; độ dài chưa kiểm).
- Ứng viên chính là KITTI-360, cũng là dataset duy nhất tool hỗ trợ có camera mắt cá thật.

## 5. Dữ liệu đã có trên Kaggle (tài khoản `quanlam`, private)

| Dataset Kaggle | Nội dung |
|---|---|
| `d2s-raw-nuscenes-mini` | nuScenes v1.0-mini nguyên bản (10 cảnh) |
| `d2s-raw-kitti360-0003` | KITTI-360 0003 đủ 1031 khung, 4 camera, LiDAR |
| `d2s-raw-kitti360-meta` | KITTI-360: calib, pose, box 3D của 0003 |
| `d2s-model-smpl` | `SMPL_NEUTRAL.pkl` cho `prep.smpl` |

Quy tắc tên: CONTRACTS mục 10.1. Cách đưa dữ liệu gốc lên và ingest trên Kaggle: CONTRACTS mục 9.4.

## Nguồn

- Waymo Open Dataset: https://waymo.com/open/about
- Argoverse 2: https://argoverse.github.io/user-guide/datasets/sensor.html, https://arxiv.org/pdf/2301.00493
- PandaSet: https://arxiv.org/abs/2112.12610
- PhysicalAI (NVIDIA): https://nvidia.github.io/ncore/conversions/pai/pai.html, https://perspectives.nvidia.com/nvidia-alpamayo/task/faq/av-training-datasets-global-driving-footage-1/
- WoodScape: https://arxiv.org/pdf/1905.01489
- ZOD: https://openaccess.thecvf.com/content/ICCV2023/html/Alibeigi_Zenseact_Open_Dataset_A_Large-Scale_and_Diverse_Multimodal_Dataset_for_ICCV_2023_paper.html
- KITTI-360: https://www.cvlibs.net/datasets/kitti-360
- nuScenes: https://arxiv.org/abs/1903.11027
