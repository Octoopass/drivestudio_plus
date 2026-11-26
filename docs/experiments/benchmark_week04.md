# Benchmark tuần 04 trên KITTI-360 chuỗi 0003

Ma trận chạy (17 run ưu tiên 1, khoảng 25 GPU-giờ trên RTX 4090 thuê) nằm ở Sheet "Ma trận chạy benchmark tuần 04" của nhóm; số thứ tự run (#1, #2, ...) dưới đây khớp tab **Ma trận** của Sheet. Yêu cầu R1–R8 và quy tắc xếp hạng: `docs/drive2sim/ARCHITECTURE.md` mục 2.17, CONTRACTS mục 6.5.

Script chuẩn bị: `drive2sim/scripts/setup_benchmark_week04.py` (chỉ đọc và ghi file trong thư mục làm việc, không chạy GPU).

## 1. Cảnh và split

Hai cảnh, cả hai từ notebook dữ liệu gốc `d2s-raw-kitti360-0003`:

| Cảnh | Ingest | Khung |
|---|---|---|
| `kitti360_2013_05_28_drive_0003_sync` | đã có (`d2s kaggle ingest ... --num-frames 200`) | 200 khung, khung gốc 1–220, khoảng 124 m |
| `kitti360_0003_long` | `d2s kaggle ingest <người dùng>/d2s-raw-kitti360-0003 --sequence 2013_05_28_drive_0003_sync --start-frame 167 --scene kitti360_0003_long` | đoạn liền khung gốc 167–1030 (864 khung, 1.289 m theo ARCHITECTURE mục 2.17; số khung thật kiểm sau khi ingest) |

Mọi split dùng `every_nth_frame:8`.

| Split | Cảnh | Camera (chỉ test) | Đoạn khung | Dùng cho |
|---|---|---|---|---|
| `every8_stereo` | 200 khung | `image_00`, `image_01` (`image_01`) | cả cảnh | A1: track A, hiệu chỉnh seed |
| `every8_stereo_fisheye2` | 200 khung | + `image_02`, `image_03` (`image_01`) | cả cảnh | A2 |
| `every8_mixed3` | 200 khung | `image_00`, `image_02`, `image_03` | cả cảnh | track C (Instant NuRec) |
| `every8_stereo_a3` | dài | `image_00`, `image_01` (`image_01`) | khung gốc 560–779 | A3 |
| `every8_pinhole2` | dài | `image_00`, `image_01` | cả cảnh | track B, MCGS-SLAM |
| `every8_stereo_c1` … `c6` | dài | `image_00`, `image_01` (`image_01`) | 6 đoạn ~153 khung, chồng 10 khung | track B, drivestudio chia chunk |

Lý do của các lựa chọn khác thường:
- **A3 khung gốc 560–779**: chứa người đi bộ duy nhất của chuỗi 0003 (khung 629–662) và xe chạy, nên đo được PSNR vùng người mà A1 không có, vẫn giữ cặp stereo cho R2. Script đổi số hiệu gốc sang chỉ số khung qua `frame_ids.txt` và in độ đổi hướng của xe trong đoạn (từ `ego_pose/`); dưới 30 độ thì đoạn không có rẽ: dời đoạn trong khoảng khung gốc 560–1030 nhưng giữ khung 629–662 (sửa `A3_SOURCE_RANGE`). Chưa kiểm vì cảnh dài chưa ingest.
- **Track C không có `image_01`**: `prep.ncore` chưa bỏ camera chỉ test của split (chỉ `prep.mcgs` bỏ), nên đưa `image_01` vào split sẽ cho NuRec thấy camera dùng để đo R2. R2 của track C vì vậy không đo. `prep.ncore` cũng chuyển **mọi khung** của `frame_range`, kể cả khung test (`docs/methods/nurec.md`: NuRec dựng từ mọi khung của clip), nên R1 của track C đo trên ảnh NuRec đã nhận làm đầu vào, không so được với R1 của track A.
- **MCGS-SLAM không có camera chỉ test**: MCGS cần ít nhất hai camera pinhole để train, và `prep.mcgs` từ chối camera chỉ test, nên R2 không áp dụng cho run này. `prep.mcgs` mặc định `held_out: true` (bỏ khung test khỏi đầu vào).

## 2. Lệnh

Trên máy thuê, trong thư mục làm việc (gốc repo, hoặc `D2S_WORKDIR`) đã có hai cảnh:

```bash
python -m drive2sim.scripts.setup_benchmark_week04 splits --dry-run   # xem trước
python -m drive2sim.scripts.setup_benchmark_week04 splits             # tạo split còn thiếu
python -m drive2sim.scripts.setup_benchmark_week04 commands           # in lệnh d2s run start
# ... chạy các run ...
python -m drive2sim.scripts.setup_benchmark_week04 definition         # ghi benchmarks/week04.yaml
d2s benchmark benchmarks/week04.yaml --out benchmarks/week04.json
```

`commands` in một lệnh cho mỗi run, đúng cấu hình mà `definition` dùng để nhận ra run (lệnh đã kiểm bằng `d2s approach check` trên cảnh 200 khung, trừ hai lệnh trên cảnh dài). Ô của run thắng (#9, #16, #17) in dạng `<thắng ...>`: điền approach và `--set` của run thắng. Run chạy thử (#1–#3, `--quick`) không vào file định nghĩa.

`definition` xếp run đã xong (`status: succeeded`, không `--quick`) vào nhóm theo cảnh, split, approach, lựa chọn (`approach.options`) và tham số `seed`, `lidar` của `train.drivestudio`:

| Nhóm | Split | Run nhận vào |
|---|---|---|
| A-LB | `every8_stereo` | OmniRe 3DGS + FastGS seed 0; BezierGS |
| A-L | `every8_stereo` | PVG, DeformableGS renderer 3DGS (có LiDAR) |
| A-0 | `every8_stereo` | PVG, DeformableGS renderer 3DGS với `lidar=false` |
| A-LB | `every8_stereo_fisheye2` | OmniRe 3DGUT + FastGS; OmniRe 3DGEER + FastGS |
| A-L | `every8_stereo_fisheye2` | PVG, DeformableGS với renderer 3DGUT (#12) |
| A-LB, A-L | `every8_stereo_a3` | OmniRe (mọi cấu hình); PVG, DeformableGS |
| B | `every8_pinhole2` | MCGS-SLAM |
| C | `every8_mixed3` | Instant NuRec |
| `calibration` | `every8_stereo` | OmniRe 3DGS + FastGS seed 0, 1, 2 (#4, #7, #8) |

Nhiều run khớp một ô thì lấy run mới nhất và in các run bị bỏ. Nhóm chưa có run thì bỏ khỏi file; dưới hai seed thì không có khối `calibration` (`d2s benchmark` dùng ngưỡng hoà mặc định 0,3 dB).

## 3. Chưa làm được trong đợt này

- **#12 (A-L với fisheye, A2)**: từ 26/11 PVG và DeformableGS có renderer 3DGUT, 3DGEER (tắt CamPose), nên #12 chạy được với `--set renderer=3dgut --set campose=false`; chưa chạy thật trên GPU. A2 A-L vì vậy so method thắng A-L ở cấu hình khác A1 (không CamPose).
- **R3 của PVG, DeformableGS**: `export_ply.py` ghi tham số gốc của node `Background` (`_means`, `_opacities`). Với PVG đó là tâm chưa cộng dao động và độ đục chưa nhân suy giảm theo thời gian, nên mọi Gaussian (cả vệt của xe chạy) hiện cùng lúc; với DeformableGS là tâm chưa qua mạng biến dạng. Từ 26/11 `d2s benchmark` đánh dấu các run này `export_static: false` (bảng ghi `†`) và bỏ bước R3 khi phân xử các ô A-L, A-0; R3 cũng đổi từ export gap sang `ply_psnr` (PSNR tuyệt đối của `.ply` trên vùng nền tĩnh), gap chỉ báo kèm (CONTRACTS mục 6.5). Còn lại: viết cách xuất nền tĩnh đúng cho hai method (PVG lọc theo hệ số tĩnh `rho` và ghi ở một thời điểm; DeformableGS biến dạng ở một thời điểm) để R3 dùng lại được.
- **#17 trong bảng xếp hạng**: `d2s benchmark` chưa có nhóm "chunked" gộp `frames.jsonl` của các chunk (ARCHITECTURE mục 2.17), nên 6 run chunk chạy được nhưng chưa vào `week04.yaml`.
- **#14**: chạy thử Instant NuRec trên clip mẫu NCore, ngoài cảnh KITTI-360, nên không có lệnh ở đây.
