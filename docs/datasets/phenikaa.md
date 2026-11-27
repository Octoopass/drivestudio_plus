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

## 3. Mask thân xe

Tám camera thấy thân xe: capo trong `CAM_P_F`, hai mảnh mép capo sát đáy `CAM_P_FR`, gương trong
`CAM_P_L`, `CAM_P_R`, thân xe ở mép bốn camera mắt cá. `CAM_P_FL` và `CAM_P_B` không thấy thân xe
(ghi `none` trong `ego_masks/status.json`). Mask nằm ở `drive2sim/assets/ego_masks/phenikaa/<camera>.png` (trắng: thân xe), làm
một lần trên ảnh đã ingest (pinhole đã khử méo theo calib hiện tại):

1. Ảnh trung vị của khoảng 120 khung trên cả ba chuỗi: cảnh bị xoá, thân xe đứng yên còn lại.
2. Tách thân xe đỏ và phần tối (gương, chân gương) chạm mép ảnh hoặc vành ống kính.
3. Lấy các mảng đó làm điểm gợi ý cho SAM2.1 (`facebook/sam2.1-hiera-large`) trên ảnh trung vị;
   mask cuối là hợp của hai kết quả (SAM2 thêm chân gương `CAM_P_R` và dải thân xe sát vành ống
   kính mắt cá).
4. Xem lại bằng mắt trên cả ba chuỗi.

Vành tối và góc tối của camera mắt cá không nằm trong mask: loader tự dò góc tia bắt đầu vùng tối
(`dark_border_angle`) và loại mọi pixel ngoài góc đó.
Converter chép mask vào `data/<cảnh>/ego_masks/`; loader dùng nó để bỏ thân xe khỏi loss, khỏi
phần chấm điểm và bỏ điểm LiDAR rơi lên thân xe. Mask chỉ đúng với bộ calib này: Phenikaa-X đổi
calib thì phải làm lại mask (converter từ chối mask sai kích thước ảnh).

Dataset khác cũng dùng cách này: đặt `ego_masks/<camera_id>.png` (đúng kích thước ảnh trong
`images/`) vào thư mục cảnh thì mọi loader ưu tiên nó hơn mask kèm theo tool.

## 4. Tracking box

Nhãn `label/` là box 3D theo từng khung, không có ID theo dõi. Converter nối chúng thành vật
(`drive2sim/ingest/phenikaa_tracking.py`) rồi ghi `instances/` và `dynamic_masks/` như KITTI-360,
nên OmniRe và Street GS dựng được xe, người. Cách nối:

1. Dự đoán vị trí mỗi vật ở khung sau theo vận tốc không đổi.
2. Ghép box mới với vật bằng Hungarian trên khoảng cách mặt đất, cùng lớp, trong ngưỡng Car 2,5 m,
   Rider 1,5 m, Pedestrian 1,0 m.
3. Vật mất dấu quá 3 khung thì đóng; khung hụt được nội suy; yaw lật 180° được sửa; kích thước lấy
   trung vị; vật ngắn hơn 5 khung bị bỏ.

Lớp: Car → Car (node cứng), Rider → Cyclist (node biến dạng), Pedestrian → Pedestrian.

| Chuỗi | Box nhãn | Vật | Car / Cyclist / Pedestrian | Box bị bỏ | Độ dài trung vị |
|---|---|---|---|---|---|
| COMPLEX CITY | 32 818 | 451 | 172 / 240 / 39 | 92 | 56 khung |
| COMPLEX URBAN | 21 951 | 329 | 91 / 219 / 19 | 366 | 54 khung |
| RESIDENTIAL AREA | 19 992 | 203 | 72 / 88 / 43 | 55 | 107 khung |

Kiểm trên COMPLEX CITY: vẽ box kèm ID lên ảnh qua nhiều khung, ID giữ đúng vật; tốc độ giữa hai
khung trung vị 0,7 m/s, 99% dưới 12,5 m/s, 3 trên 32 347 bước nhảy quá 25 m/s (nghi đổi ID). Số
liệu của từng cảnh ở `reports/ingest.json`, mục `tracking`. Rủi ro: hai xe máy sát nhau cùng lớp có
thể đổi ID cho nhau.

## 5. Đã kiểm

- Chiếu LiDAR lên ảnh qua loader của tool, cả ba chuỗi, mọi camera: điểm khớp thân cây, cạnh nhà,
  lan can; 0 cảnh báo của `projection_checks`. Vật đang chạy (xe máy) lệch nhẹ do LiDAR và camera
  không chụp cùng lúc.
- `Sensor_Extrinsics.json` là LiDAR → camera; điểm `.laz` ở hệ xe (trùng hệ LiDAR), nên converter
  dùng nó như xe → camera, đúng như devkit của Phenikaa-X.
- Box: `z` là tâm hộp (mỗi box chứa trung vị 55 điểm LiDAR; coi `z` là đáy thì còn 32).

## 6. Giới hạn

- **Skew của camera mắt cá bị bỏ** (`K[0, 1]` khoảng 0,4–0,5; lệch dưới 1 pixel).
- `CAM_P_L` và `CAM_P_R` có thông số trong giống hệt nhau trong file calib của Phenikaa-X.
- Gốc toạ độ của điểm LiDAR là gốc hệ xe (gần mặt đất), không phải tâm cảm biến: tia LiDAR mà
  loader dựng xuất phát từ gốc hệ xe.
- Chưa có SMPL cho người đi bộ: người đi bộ không có `humanpose/smpl.pkl`, stage train tắt SMPL.
- Mask vật động (`dynamic_masks/`) chỉ vẽ vật có mọi góc box nằm trong ảnh (như script KITTI của
  DriveStudio): vật sát xe, bị cắt ở mép ảnh, không có trong mask.
