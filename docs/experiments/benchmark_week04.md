# Benchmark tuần 04: Waymo (pinhole) và KITTI-360 chuỗi 0003

Ma trận chạy (11 run ưu tiên 1, khoảng 27 GPU-giờ trên RTX 4090 thuê, ước lượng; đã bỏ các run chạy thử R2, hiệu chỉnh seed và chạy thử NuRec) nằm ở Sheet "Ma trận chạy benchmark tuần 04" của nhóm; số thứ tự run (#1–#14, đánh số lại 26/11) dưới đây khớp tab **Ma trận** của Sheet. Yêu cầu R1–R8 và quy tắc xếp hạng: `docs/drive2sim/ARCHITECTURE.md` mục 2.17, CONTRACTS mục 6.5.

Script chuẩn bị: `drive2sim/scripts/setup_benchmark_week04.py` (chỉ đọc và ghi file trong thư mục làm việc, không chạy GPU).

## 1. Cảnh và split

Ba cảnh. **Pinhole xếp hạng trên Waymo (W1), không trên KITTI-360** (quyết định 26/11/2026): pinhole của KITTI-360 chỉ train một camera trước (`image_00`; `image_01` giữ để đo R2), không giống cách các method và dataset khác dùng nhiều camera quanh xe. W1 train cả 5 camera Waymo nhưng không có camera nào để giữ lại đo R2 (camera Waymo chồng nhau rất ít), nên R2 không áp dụng ở các nhóm W1 và phân xử hoà đi thẳng sang R3 rồi thời gian train. KITTI-360 giữ cho R2 (A1, chạy thử), fisheye (A2), cảnh có người (A3), quãng dài (B2) và track C. Đợt sau: Argoverse 2 (7 camera vòng và cặp stereo trước, có thể giữ một camera stereo để đo R2; cần thêm camera stereo vào preprocess và loader).

| Cảnh | Ingest | Khung |
|---|---|---|
| `waymo_segment-10455472356147194054_1560_000_1580_000_with_camera_labels` (W1) | tải tfrecord (`python -m drive2sim.recon.preprocess.waymo.download --scene_ids 23`, cần tài khoản Google đã nhận điều khoản Waymo), rồi `d2s scene ingest <thư mục> --sequence segment-10455472356147194054_1560_000_1580_000_with_camera_labels` | dùng khung 0–149 (theo `waymo_example_scenes.txt` của DriveStudio); 5 camera nạp ở 1/2 độ phân giải (`configs/datasets/waymo/5cams.yaml`). Chưa kiểm nội dung cảnh: ít xe, người thì đổi sang scene khác trong danh sách đó |
| `kitti360_2013_05_28_drive_0003_sync` | đã có (`d2s kaggle ingest ... --num-frames 200`) | 200 khung, khung gốc 1–220, khoảng 124 m |
| `kitti360_0003_long` | `d2s kaggle ingest <người dùng>/d2s-raw-kitti360-0003 --sequence 2013_05_28_drive_0003_sync --start-frame 167 --scene kitti360_0003_long` | đoạn liền khung gốc 167–1030 (864 khung, 1.289 m theo ARCHITECTURE mục 2.17; số khung thật kiểm sau khi ingest) |

Mọi split dùng `every_nth_frame:8`.

| Split | Cảnh | Camera (chỉ test) | Đoạn khung | Dùng cho |
|---|---|---|---|---|
| `every8_5cams` | W1 | FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT | khung 0–149 | W1: A-LB, A-L, A-0 |
| `every8_stereo` | 200 khung | `image_00`, `image_01` (`image_01`) | cả cảnh | A1 (không còn run riêng) |
| `every8_stereo_fisheye2` | 200 khung | + `image_02`, `image_03` (`image_01`) | cả cảnh | A2 |
| `every8_mixed3` | dài | `image_00`, `image_02`, `image_03` (như A2, không có `image_01`) | cả cảnh | track C (Instant NuRec, #9). Cảnh 864 khung cần khoảng 48 đoạn NuRec (18 khung một đoạn) mà `train.nurec` giới hạn 8 đoạn: phải nâng giới hạn hoặc chạy theo 6 chunk của B2 trước khi chạy |
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

`commands` in một lệnh cho mỗi run, đúng cấu hình mà `definition` dùng để nhận ra run (cấu hình đã kiểm bằng `d2s approach check` trên cảnh KITTI-360 200 khung; lệnh trên W1 và cảnh dài chưa kiểm vì chưa có hai cảnh này). Ô của run thắng (#4, #10, #11, #12) in dạng `<thắng ...>`: điền approach và `--set` của run thắng. Run chạy thử (`--quick`) không vào file định nghĩa.

`definition` xếp run đã xong (`status: succeeded`, không `--quick`) vào nhóm theo cảnh, split, approach, lựa chọn (`approach.options`) và tham số `seed`, `lidar` của `train.drivestudio`:

| Nhóm | Split | Run nhận vào |
|---|---|---|
| A-LB | `every8_5cams` (W1) | OmniRe 3DGS + FastGS seed 0; BezierGS |
| A-L | `every8_5cams` (W1) | PVG, DeformableGS renderer 3DGS (có LiDAR) |
| A-0 | `every8_5cams` (W1) | PVG, DeformableGS renderer 3DGS với `lidar=false` |
| A-LB | `every8_stereo_fisheye2` | OmniRe 3DGUT + FastGS; OmniRe 3DGEER + FastGS |
| A-L | `every8_stereo_fisheye2` | PVG, DeformableGS với renderer 3DGUT (#7) |
| A-LB, A-L | `every8_stereo_a3` | OmniRe (mọi cấu hình); PVG, DeformableGS |
| B | `every8_pinhole2` | MCGS-SLAM |
| C | `every8_mixed3` | Instant NuRec |
| `calibration` | `every8_5cams` (W1) | OmniRe 3DGS + FastGS seed 0, 1, 2 nếu có (ma trận không còn run hiệu chỉnh seed: dùng ngưỡng hoà mặc định 0,3 dB) |

Nhiều run khớp một ô thì lấy run mới nhất và in các run bị bỏ. Nhóm chưa có run thì bỏ khỏi file; dưới hai seed thì không có khối `calibration` (`d2s benchmark` dùng ngưỡng hoà mặc định 0,3 dB).

## 3. Kế hoạch chạy trên GPU thuê

Chạy trên **một máy vast.ai 2× RTX 4090** (lý do và cấu hình máy ở cuối mục). Các run ưu tiên 1 chia ba đợt theo run thắng mà đợt sau cần; mỗi đợt là một hàng đợi, mỗi GPU một worker lấy run kế tiếp (`envs/bench_queue.sh`, run dài trước). GPU-giờ là ước lượng của Sheet (RTX 4090), sửa lại sau run đầu.

| Đợt | Run | Cần trước | GPU-giờ | Thời gian trên 2 GPU |
|---|---|---|---|---|
| 0. Chuẩn bị | cài env (main, MCGS, NuRec), ingest W1 và cảnh dài, split | — | ~0 (CPU, mạng) | ~2 giờ |
| 1 | #1, #2, #3, #5, #6, #8 (+ #9 khi đã chọn cách chia đoạn NuRec; + #13 nếu đã port BezierGS) | đợt 0 | 13,6 (+2,5) | ~7,3 giờ |
| 2 | #4, #7 (+ #12 ưu tiên 2) | thắng A-L giữa #2 và #3 | 4,9 (+1,7) | ~2,9 giờ (3,7 với #12) |
| 3 | #10, 6 chunk của #11 | cấu hình thắng A-LB sau #1–#7 | 6,2 | ~3,3 giờ |

Tổng khoảng 25 GPU-giờ cho ưu tiên 1 trừ #9 (27,2 nếu có #9), khoảng 14 giờ chạy và 17 giờ thuê máy kể cả chuẩn bị và hai lần chọn run thắng.

#5, #6 (A-LB fisheye) không đợi #1 như cột "Phụ thuộc" của Sheet: cấu hình đã cố định, nên chạy ngay ở đợt 1 cho đỡ phí GPU; nếu #1 lỗi vì môi trường thì #5, #6 cũng lỗi sớm, `approach check` ở đầu hàng đợi bắt phần lớn lỗi cấu hình.

### Các bước

Trên máy mình (thư mục repo, nhánh `feat/bench-week04`); `VAST_HOST`, `VAST_PORT` lấy ở nút Connect của máy thuê:

```bash
export VAST_HOST=root@<ip> VAST_PORT=<cổng>
bash envs/vast.sh check          # 2 GPU, không phải RTX 50xx, có nvcc, đủ đĩa
bash envs/vast.sh setup          # env drive2sim, ~30–60 phút; xong khi setup.log có SETUP_DONE
bash envs/vast.sh setup-extra    # MCGS-SLAM, Instant NuRec; xong khi setup-extra.log có SETUP_EXTRA_DONE
D2S_WORKDIR="<repo chính drivestudio_plus>" D2S_SMPL_MODEL=<file SMPL> \
  bash envs/vast.sh push-scene kitti360_2013_05_28_drive_0003_sync   # cảnh 200 khung (A2) và split của nó
```

Ingest trên máy thuê (mạng nhanh hơn; cần `~/.kaggle/kaggle.json` và tài khoản Google đã nhận điều khoản Waymo):

```bash
bash envs/vast.sh exec 'd2s kaggle ingest <người dùng>/d2s-raw-kitti360-0003 --sequence 2013_05_28_drive_0003_sync --start-frame 167 --scene kitti360_0003_long'
# Waymo: gcloud auth login cần tương tác, nên ssh vào máy (ssh -p $VAST_PORT $VAST_HOST), cài gcloud CLI nếu image chưa có, rồi:
#   gcloud auth login --no-launch-browser
#   cd /workspace/d2s && python -m drive2sim.recon.preprocess.waymo.download --scene_ids 23 --target_dir data/waymo/raw
bash envs/vast.sh exec 'd2s scene ingest data/waymo/raw --sequence segment-10455472356147194054_1560_000_1580_000_with_camera_labels'
bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 splits'   # xem độ đổi hướng của A3
bash envs/vast.sh exec 'd2s scene lidar-check kitti360_0003_long'                    # tương tự cho W1
```

Chạy thử 10 phút trước đợt 1 để bắt lỗi môi trường trên GPU (run `--quick` không vào bảng xếp hạng):

```bash
bash envs/vast.sh run omnire --scene waymo_segment-10455472356147194054_1560_000_1580_000_with_camera_labels --split every8_5cams --set fastgs=true --quick
```

Ba đợt:

```bash
bash envs/vast.sh wave 1                       # thêm --nurec khi đã chọn cách chia đoạn, --p2 cho #13
bash envs/vast.sh wave-status 1                # QUEUE_DONE trong logs/wave1.out là xong
bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml'
bash envs/vast.sh wave 2 --al-winner pvg       # hoặc deformablegs, theo nhóm A-L ở trên; --p2 thêm #12
bash envs/vast.sh wave 3 --alb-winner "omnire renderer=3dgs fastgs=true"   # cấu hình thắng A-LB
bash envs/vast.sh pull                         # kéo runs/, cache/ về; rồi definition + d2s benchmark ở máy mình
```

Run lỗi được ghi vào `logs/queue-wave<N>/failed`; chạy lại: `bash envs/vast.sh exec 'nohup bash envs/bench_queue.sh logs/queue-wave<N>/failed > logs/retry.out 2>&1 &'`. Cập nhật cột "Trạng thái", "Run ID" của Sheet sau mỗi đợt. Kéo kết quả về (`pull`) sau mỗi đợt, rồi xoá máy thuê ngay khi xong vì dữ liệu có license riêng.

### GPU nên thuê

- **Loại:** RTX 4090 24 GB. Env dùng torch 2.4.1+cu121 nên RTX 50xx (Blackwell) không chạy; 3090 chậm hơn khoảng 1,5–2 lần (ước lượng) mà giá chỉ rẻ hơn một chút; A100, H100 đắt gấp 2–4 lần mà các run đã đo chỉ dùng dưới 9 GB VRAM (OmniRe nuScenes 6,9 GB, MCGS 8,8 GB trên T4); W1 ảnh lớn hơn nhưng vẫn trong 24 GB (ước lượng, xem VRAM đỉnh của #1).
- **Số lượng: 2 GPU trên cùng một máy.** Một máy thì cài env, ingest một lần và dùng chung cache (mask, SMPL). Đợt 2 chỉ có 2 run, và giữa các đợt máy chờ nhóm chọn run thắng, nên 4 GPU nhanh hơn (khoảng 12 giờ thuê) nhưng tốn hơn (khoảng 4 × 12 = 48 GPU-giờ trả tiền cho 25 GPU-giờ dùng). 1 GPU rẻ nhất nhưng mất khoảng 30 giờ. Chỉ chọn 4 GPU nếu cần kết quả trong một ngày.
- **Cấu hình máy:** RAM ≥ 64 GB, ≥ 16 vCPU, đĩa ≥ 200 GB (ba env, cảnh dài, Waymo, khoảng 20 run), mạng ≥ 500 Mbps, driver hỗ trợ CUDA ≥ 12.1, độ tin cậy ≥ 0,98; image `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`; chế độ riêng tư.
- **Chi phí:** ở giá trung vị RTX 4090 trên vast.ai khoảng 0,5 USD/giờ mỗi GPU (giá xem ngày 26/11/2026, thay đổi theo giờ), 2 GPU × ~17 giờ ≈ 17 USD; 4 GPU ≈ 24 USD; 1 GPU ≈ 15 USD.

## 4. Chưa làm được trong đợt này

- **#7 (A-L với fisheye, A2)**: từ 26/11 PVG và DeformableGS có renderer 3DGUT, 3DGEER (tắt CamPose), nên #7 chạy được với `--set renderer=3dgut --set campose=false`; chưa chạy thật trên GPU. A2 A-L vì vậy so method thắng A-L ở cấu hình khác A1 (không CamPose).
- **R3 của PVG, DeformableGS**: `export_ply.py` ghi tham số gốc của node `Background` (`_means`, `_opacities`). Với PVG đó là tâm chưa cộng dao động và độ đục chưa nhân suy giảm theo thời gian, nên mọi Gaussian (cả vệt của xe chạy) hiện cùng lúc; với DeformableGS là tâm chưa qua mạng biến dạng. Từ 26/11 `d2s benchmark` đánh dấu các run này `export_static: false` (bảng ghi `†`) và bỏ bước R3 khi phân xử các ô A-L, A-0; R3 cũng đổi từ export gap sang `ply_psnr` (PSNR tuyệt đối của `.ply` trên vùng nền tĩnh), gap chỉ báo kèm (CONTRACTS mục 6.5). PVG đã có cách xuất nền tĩnh: chỉ giữ Gaussian sống từ 20 khung trở lên (tham số gốc của PVG là mỗi Gaussian ở đỉnh của nó, nên không cần chọn thời điểm). **Kiểm trên run #2 (PVG trên W1) trước khi bỏ `pvg` khỏi `TIME_VARYING_METHODS`:** xem `time_varying` trong `eval.ply/background/scene.json` (số giữ, số bỏ, phân vị độ dài đời), ảnh render `.ply` không còn vệt xe, và `ply_psnr` không kém `native_psnr` quá khoảng 2 dB; nếu vẫn kém thì giữ cờ và ghi R3 là không áp dụng cho PVG. DeformableGS còn lại: biến dạng ở một thời điểm, làm sau khi có checkpoint của #3.
- **#11 trong bảng xếp hạng**: `d2s benchmark` chưa có nhóm "chunked" gộp `frames.jsonl` của các chunk (ARCHITECTURE mục 2.17), nên 6 run chunk chạy được nhưng chưa vào `week04.yaml`.
- **#14 (SplatAD)**: chưa có approach trong tool, nên không có lệnh ở đây.
