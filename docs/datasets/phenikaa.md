# PHENIKAA (Phenikaa-X Perception Data)

Bộ dữ liệu perception 3D của Phenikaa-X, thu trong giao thông đông ở Hà Nội
([trang giới thiệu](https://phenikaa-x.com/perception-data/)). Bản mini gồm các chuỗi 20 giây,
10 Hz. Tool đọc bộ này bằng converter `drive2sim/ingest/phenikaa.py` và loader
`drive2sim/recon/data/loaders/phenikaa.py`.

## 1. Đặt dữ liệu

```
data/phenikaa/raw/
├── calib/
│   ├── Camera_Intrinsics.json        thông số trong của 10 camera
│   ├── Sensor_Extrinsics.json        xe → camera (trục OpenCV), cộng IMU, INS
│   └── extrinsic_verification/       ảnh vùng nhìn do Phenikaa-X gửi kèm
├── COMPLEX CITY/
│   ├── camera/CAM_*/<mốc>.jpg        10 camera
│   ├── lidar/<mốc>.laz               LiDAR nóc xe, hệ xe, cường độ 0..255
│   ├── ego_pose/<mốc>.txt            ma trận 3x4 xe → world
│   ├── label/<mốc>.txt               box 3D: x y z dài rộng cao yaw lớp (z là tâm hộp)
│   ├── Summary.txt
│   └── *_projected_3d_box_5x2.mp4
├── COMPLEX URBAN/
└── RESIDENTIAL AREA/
```

`<mốc>` là `<giây>-<nano giây>`, chung cho mọi cảm biến. Calib tải riêng từ Phenikaa-X; converter
tìm `calib/` trong chuỗi, cạnh chuỗi hoặc hai cấp trên, hoặc nhận `--calib <thư mục>`.

## 2. Ingest

```bash
d2s source inspect data/phenikaa/raw
d2s scene ingest data/phenikaa/raw --sequence 'COMPLEX CITY'      # → data/phenikaa_complex-city/
```

Cần `laspy[lazrs]` trong môi trường `drive2sim` (đã có trong `requirements.txt`). Ingest một chuỗi
mất khoảng một phút trên CPU; xong thì tool tự dựng sơ đồ camera và chiếu LiDAR lên ảnh
(`reports/lidar_overlay/`).

| Camera | Chỉ số | Mô hình | Kích thước | Xử lý khi ingest |
|---|---|---|---|---|
| `CAM_P_F`, `CAM_P_FL`, `CAM_P_FR`, `CAM_P_B` | 0–3 | pinhole tầm xa, Brown–Conrady 5 hệ số | 1920×1536 | khử méo về pinhole lý tưởng, JPEG 95 |
| `CAM_P_L`, `CAM_P_R` | 4–5 | pinhole 128°, rational 8 hệ số | 1920×1536 | khử méo như trên |
| `CAM_F_F`, `CAM_F_L`, `CAM_F_R`, `CAM_F_B` | 6–9 | mắt cá Kannala–Brandt 4 hệ số | 1920×1080 | giữ ảnh gốc (hard link), vẽ theo mô hình ống kính |

Camera thiếu ảnh ở một số mốc bị bỏ (RESIDENTIAL AREA còn 6 camera: thiếu `CAM_F_F`, `CAM_F_R`,
`CAM_P_FL`, và `CAM_P_R` chỉ có 27 ảnh). World của cảnh là hệ xe ở khung đầu (COMPLEX URBAN có pose
toàn cục, các chuỗi kia bắt đầu từ gốc).

## 3. Đã kiểm

- Chiếu LiDAR lên ảnh qua loader của tool, cả ba chuỗi, mọi camera: điểm khớp thân cây, cạnh nhà,
  lan can; 0 cảnh báo của `projection_checks`. Vật đang chạy (xe máy) lệch nhẹ do LiDAR và camera
  không chụp cùng lúc.
- Box: `z` là tâm hộp (mỗi box chứa trung vị 55 điểm LiDAR; coi `z` là đáy thì còn 32).

## 4. Giới hạn

- **Box chưa có ID theo dõi**: cảnh chưa có `instances/`, manifest ghi `cuboids: none`, nên
  OmniRe, Street GS (cần box) bị khoá; PVG, DeformableGS, XSIM dựng được nền. Cần bước tracking.
- **Chưa có mask thân xe**: nắp capo trong `CAM_P_F` và thân xe ở mép ảnh mắt cá vẫn được dùng.
- **Skew của camera mắt cá bị bỏ** (`K[0, 1]` khoảng 0,4–0,5; lệch dưới 1 pixel).
- `CAM_P_L` và `CAM_P_R` có thông số trong giống hệt nhau trong file calib của Phenikaa-X; cần
  hỏi lại.
- Gốc toạ độ của điểm LiDAR là gốc hệ xe (gần mặt đất), không phải tâm cảm biến: tia LiDAR mà
  loader dựng xuất phát từ gốc hệ xe.
