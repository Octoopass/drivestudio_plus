# Benchmark tuần 04: Waymo (pinhole) và KITTI-360 chuỗi 0003

Ma trận chạy (11 run ưu tiên 1, khoảng 27 GPU-giờ trên RTX 4090 thuê, ước lượng; đã bỏ các run chạy thử R2, hiệu chỉnh seed và chạy thử NuRec) nằm ở Sheet "Ma trận chạy benchmark tuần 04" của nhóm; số thứ tự run (#1–#14, đánh số lại 26/11) dưới đây khớp tab **Ma trận** của Sheet. Yêu cầu R1–R8 và quy tắc xếp hạng: `docs/drive2sim/ARCHITECTURE.md` mục 2.17, CONTRACTS mục 6.5.

Script chuẩn bị: `drive2sim/scripts/setup_benchmark_week04.py` (chỉ đọc và ghi file trong thư mục làm việc, không chạy GPU).

**Cập nhật 27/11/2026 tối (mục 6):** thắng A-LB chốt OmniRe 3DGEER + FastGS; W1 đổi sang một segment Waymo đi xa hơn (chọn trong 8 ứng viên High-speed của NOTR bằng `waymo-stats`, đặt `D2S_W1_INDEX`), dùng cả segment; #8 (MCGS-SLAM) chuyển sang W1; #11 thành một run trên cả cảnh dài; thêm run thử một khối (#15) so với 2 chunk (#16) trên 320 khung; mọi run còn lại gộp vào một đợt 3; R3 tạm bỏ qua; #9 NuRec bỏ, #13, #14 làm nếu thừa thời gian. Các mục 1–5 dưới đây mô tả W1 cũ (chỉ số 23) và giữ làm ghi chép.

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
| A-L | `every8_stereo_fisheye2` | PVG, DeformableGS với renderer 3DGEER (#7; renderer fisheye chốt 27/11) |
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
| 2 | #4, #7, chạy lại #6, #8 (+ #12 ưu tiên 2) | thắng A-L giữa #2 và #3 (PVG, chốt 27/11) | ~4–5 (+1,7) | 1 GPU, xem `benchmark_week04_wave2.md` |
| 3 | #10, 6 chunk của #11 | cấu hình thắng A-LB sau #1–#7 | 6,2 | ~3,3 giờ |

Tổng khoảng 25 GPU-giờ cho ưu tiên 1 trừ #9 (27,2 nếu có #9), khoảng 14 giờ chạy và 17 giờ thuê máy kể cả chuẩn bị và hai lần chọn run thắng.

#5, #6 (A-LB fisheye) không đợi #1 như cột "Phụ thuộc" của Sheet: cấu hình đã cố định, nên chạy ngay ở đợt 1 cho đỡ phí GPU; nếu #1 lỗi vì môi trường thì #5, #6 cũng lỗi sớm, `approach check` ở đầu hàng đợi bắt phần lớn lỗi cấu hình.

### Các bước

`envs/week04_vast.sh` làm từ lúc thuê máy tới lúc chạy được đợt 1; phần trên máy thuê ở `envs/week04_remote.sh`. Chạy trên máy mình, ở gốc repo, nhánh `feat/bench-week04`. Cần: `vastai` CLI có API key, khoá SSH công khai đã thêm ở console.vast.ai, token Kaggle của tài khoản đọc được notebook `kascnte/d2s-raw-kitti360-0003`, file `SMPL_NEUTRAL.pkl`, tài khoản Google đã nhận điều khoản Waymo.

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=2 verified=true rentable=true reliability>0.98 cpu_ram>=64 disk_space>=200 inet_down>=500' -o dph_total
bash envs/week04_vast.sh rent <offer id>     # thuê, chờ SSH, ghi .vast.env (các lệnh sau tự đọc)
bash envs/week04_vast.sh bootstrap           # rsync, git, uv, Kaggle CLI, gcloud; kiểm GPU
bash envs/week04_vast.sh setup               # nền: env drive2sim (~30–60 phút), rồi MCGS, NuRec, venv Waymo
SMPL_PKL=<đường dẫn SMPL_NEUTRAL.pkl> bash envs/week04_vast.sh creds
bash envs/week04_vast.sh data-kitti          # nền: tải raw KITTI-360 0003 (~8 GB), ingest A2 (200 khung) và cảnh dài
bash envs/week04_vast.sh waymo-auth          # tương tác: gcloud auth login
bash envs/week04_vast.sh data-waymo          # nền, sau khi setup xong (cần venv Waymo): tải segment W1, ingest
bash envs/week04_vast.sh status              # mốc *_DONE của từng bước
bash envs/week04_vast.sh finish              # split, lidar-check, in hàng đợi đợt 1
bash envs/week04_vast.sh smoke               # run --quick OmniRe trên W1 (~10 phút), không vào bảng xếp hạng
```

Cảnh KITTI-360 trên máy mình là cảnh ingest trên Kaggle (`d2s kaggle ingest`): ảnh, LiDAR nằm ở output notebook Kaggle, máy chỉ giữ file nhỏ, nên không chép được sang máy thuê và `d2s kaggle ingest` cũng không cho run chạy trên máy thuê. Vì vậy `data-kitti` tải raw rồi ingest lại bằng `d2s scene ingest` trên máy thuê, với cùng tham số (cảnh 200 khung: `--num-frames 200`), và dừng nếu `frame_ids.txt` khác cảnh đã dùng. Waymo cần TensorFlow 2.11 và `waymo-open-dataset`, không hợp với env chính, nên ingest bằng venv riêng `.venv-waymo`.

Ba đợt (đợt 2 theo `docs/experiments/benchmark_week04_wave2.md`: một GPU, `BENCH_GPUS=0`, chạy lại #6, #8):

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

- **#7 (A-L với fisheye, A2)**: từ 26/11 PVG và DeformableGS có renderer 3DGUT, 3DGEER (tắt CamPose); từ 27/11 #7 chạy với `--set renderer=3dgeer --set campose=false` (renderer fisheye chốt là 3DGEER); đã chạy thật ngày 27/11 (run `fa62`: 19,24 dB, thấp bất thường, xem `benchmark_week04_wave2_results.md` mục 4). A2 A-L vì vậy so method thắng A-L ở cấu hình khác A1 (không CamPose).
- **R3 của PVG, DeformableGS**: `export_ply.py` ghi tham số gốc của node `Background` (`_means`, `_opacities`). Với PVG đó là tâm chưa cộng dao động và độ đục chưa nhân suy giảm theo thời gian, nên mọi Gaussian (cả vệt của xe chạy) hiện cùng lúc; với DeformableGS là tâm chưa qua mạng biến dạng. Từ 26/11 `d2s benchmark` đánh dấu các run này `export_static: false` (bảng ghi `†`) và bỏ bước R3 khi phân xử các ô A-L, A-0; R3 cũng đổi từ export gap sang `ply_psnr` (PSNR tuyệt đối của `.ply` trên vùng nền tĩnh), gap chỉ báo kèm (CONTRACTS mục 6.5). PVG đã có cách xuất nền tĩnh: chỉ giữ Gaussian sống từ 20 khung trở lên (tham số gốc của PVG là mỗi Gaussian ở đỉnh của nó, nên không cần chọn thời điểm). **Kiểm trên run #2 (PVG trên W1) trước khi bỏ `pvg` khỏi `TIME_VARYING_METHODS`:** xem `time_varying` trong `eval.ply/background/scene.json` (số giữ, số bỏ, phân vị độ dài đời), ảnh render `.ply` không còn vệt xe, và `ply_psnr` không kém `native_psnr` quá khoảng 2 dB; nếu vẫn kém thì giữ cờ và ghi R3 là không áp dụng cho PVG. DeformableGS còn lại: biến dạng ở một thời điểm, làm sau khi có checkpoint của #3.
- **#11 trong bảng xếp hạng**: `d2s benchmark` chưa có nhóm "chunked" gộp `frames.jsonl` của các chunk (ARCHITECTURE mục 2.17), nên 6 run chunk chạy được nhưng chưa vào `week04.yaml`.
- **#14 (SplatAD)**: chưa có approach trong tool, nên không có lệnh ở đây.

## 5. Đợt 2 đã chạy (27/11/2026)

Kết quả đầy đủ ở `benchmark_week04_wave2_results.md`. Tóm tắt:

- Máy: vast.ai 1× RTX 4090 (offer `62286745`), một worker. Thuê khoảng 6,5 giờ, tiêu khoảng 4,1 USD (giá thật 0,575 USD/giờ gồm phí lưu trữ và mạng; offer ghi 0,43), nhiều hơn ước lượng 3–4 USD vì ba lần chạy lại #8 và các run chẩn đoán.
- Run: #4 (`76e9`, 29,02 dB), #6 (`7a64`, 22,61 dB, `image_00` 25,37 dB hơn #5), #7 (`fa62`, 19,24 dB), #8 (`3902`, 10,25 dB, ATE 21,1 m).
- Lỗi gặp và cách sửa: 3DGEER điểm chính lệch tâm (đạt trên GPU); MCGS 2 thư mục (đạt một phần); `lidar=false` PVG chia 0 (`single.py`, thời gian điểm ngẫu nhiên); MCGS `--buffer` nhỏ (`train.mcgs` 0.1.1); MCGS mốc thời gian camera trùng ở chuỗi dài hơn 500 khung (`install_mcgs.sh` sửa `500*i`); đường truyền về máy mình 300 KB/s (up thẳng lên Drive bằng `rclone`, `D2S_PULL_LIGHT=1` cho `vast.sh pull`, `envs/vast_autopull.sh`); `waymo-auth` không có tty (FIFO).
- Chưa chạy đợt 3 (chờ K chốt `--alb-winner`, cần nạp thêm credit).

## 6. Quyết định 27/11/2026 tối: đợt 3 gộp, đổi W1, #11 một run, run thử chunk

Các quyết định của K sau khi xem kết quả đợt 2 (bản đầu lúc 19:26 có đợt 3 và đợt 4 riêng, #11 chia 6 chunk; K đổi lúc 20:11, mục này ghi bản sau):

- **Thắng A-LB chốt OmniRe 3DGEER + FastGS** (`omnire renderer=3dgeer fastgs=true`, `A_LB_CHOSEN` trong script). Thắng A-L là PVG (`A_L_CHOSEN`). Hai giá trị này là mặc định của `--alb-winner`, `--al-winner`, và `commands` in sẵn lệnh của run thắng.
- **Đổi W1 sang segment Waymo đi xa hơn.** W1 cũ (chỉ số 23, khung 0–149) chỉ có quãng đường ngắn. Segment Waymo nào cũng dài khoảng 20 giây, nên "đi xa" nghĩa là xe chạy nhanh. Ứng viên là 8 segment nhóm High-speed của Diverse-56 trong NOTR (EmerNeRF): chỉ số 2, 41, 46, 62, 71, 73, 82, 83 của `waymo_train_list.txt` (`WAYMO_W1_CANDIDATES`). W1 mới dùng **cả segment** (khoảng 198 khung), không cắt 150 khung. Chọn W1:
  1. `bash envs/week04_vast.sh data-waymo`: tải và ingest mọi ứng viên (đặt `W1_CANDIDATES="41 46"` để chỉ tải vài cái), xong thì in `waymo-stats`.
  2. `python -m drive2sim.scripts.setup_benchmark_week04 waymo-stats`: bảng quãng đường, tốc độ, độ đổi hướng, số xe, người, vật chạy, quãng đường dài trước. Chọn segment đi xa mà vẫn có xe chạy và người (A-LB cần box có vật động).
  3. `export D2S_W1_INDEX=<chỉ số>` ở **máy mình** (hoặc thêm vào `.vast.env`). `envs/vast.sh` chuyển biến này sang mọi lệnh trên máy thuê; `definition` ở máy mình cũng cần nó vì nhận run theo `scene_id`. Chưa đặt thì W1 là ứng viên đầu (chỉ số 2), **tạm thời**.
- **#11 là một run thường trên cả cảnh dài**, thay 6 chunk: OmniRe 3DGEER + FastGS, 30.000 bước như mọi run, split `every8_stereo_long` (864 khung, train `image_00`, `image_01` chỉ test). Mỗi ảnh được ít bước tối ưu hơn một run cảnh ngắn khoảng 5–6 lần; đó là cách chạy "như bình thường" mà K chọn, không tăng số bước. Ảnh nạp lên GPU ở 1/2 độ phân giải (float32, khoảng 2,7 GB cho 1728 ảnh) cộng mask, depth: dự kiến vừa 24 GB, chưa đo.
- **Run thử chia chunk** trên 320 khung đầu của cảnh dài (khoảng 480 m): **#15** một khối (`every8_stereo_t320`, khung 0–319) so với **#16** hai chunk (`every8_stereo_t320_c1` khung 0–175, `_c2` khung 144–319), cùng cấu hình thắng A-LB, mỗi run 30.000 bước. Hai chunk chồng 32 khung (khoảng 48 m; theo cách lõi cộng đệm của Hierarchical 3DGS, project doc `claude/hierarchical-3dgs-long-drive.md`), khung đầu mỗi chunk là bội của 8 nên tập khung test sau khi gộp trùng tập test của #15 và đoạn chồng có 3 khung test chung. Không thử một khối với số bước gấp đôi: lịch densify của OmniRe theo bước tuyệt đối (`stop_split_at: 15000`), nên tăng `num_iters` không tăng densify theo.
- **Gộp đợt 3 và đợt 4 thành một đợt 3:** #1, #2, #3, #4 (PVG `lidar=false`), #8 trên W1 mới; #10 (A3); #11; #15; #16 (2 run). `--p2` thêm #12 (PVG trên A3) và #13 (BezierGS trên W1). Khoảng 8,7 GPU-giờ (ước lượng theo run chạy một mình ở đợt 1, 2). Phải chọn W1 trước khi dựng hàng đợi. Run W1 cũ (`c40a`, `76e9`, …) nằm trên cảnh khác nên `definition` tự bỏ.
- **#8 (MCGS-SLAM) chuyển sang W1**, ba camera trước (`--set prep.mcgs.cameras=FRONT,FRONT_LEFT,FRONT_RIGHT`; cùng độ phân giải 1920×1280, hai camera bên 1920×886). Split `every8_pinhole2` trên cảnh dài bỏ. Run `3902` (10,25 dB trên cảnh dài) giữ làm ghi chép.
- **R3 tạm bỏ qua** (benchmark có thể đổi lại): không điều tra gap pinhole, không kiểm PVG trong `TIME_VARYING_METHODS` ở đợt này.
- **Tool:** SplatAD (#14) và BezierGS (#13) chỉ làm nếu thừa thời gian; Instant NuRec (#9) đã có người test nên bỏ khỏi benchmark này.

Nhóm `B` của file định nghĩa có ba nhóm một run (`W1 MCGS-SLAM`, `B2 OmniRe một run cả cảnh`, `thử: một khối 320 khung`) và một mục `chunked` (#16). Cập nhật Sheet: thêm #15, #16; #11 đổi từ 6 chunk sang một run.

### Nhóm chunked (#16)

File định nghĩa `d2s-benchmark/0.3` có danh sách `chunked`: mỗi mục là một lần dựng chia chunk (`label`, `scene_id`, `chunks: [{split_id, run_id}, …]`). `definition` ghi mục #16 khi cả hai chunk đã xong. `d2s benchmark` (`drive2sim/engine/chunked.py`) gộp `frames.jsonl` các chunk:

- chỉ số khung đưa về chỉ số của cảnh (DriveStudio ghi chỉ số tính từ khung đầu của split), vị trí lấy từ `ego_pose/` của cảnh (vị trí camera của mỗi chunk tính trong hệ riêng của chunk đó);
- đoạn chồng chia đôi: khung đến giữa đoạn chồng thuộc chunk trước, còn lại thuộc chunk sau, nên mỗi ảnh test đếm một lần;
- báo PSNR, SSIM, LPIPS, Avg gộp, R2 trên `image_01`, quãng đường, tổng thời gian train, s/100 m, MB `.ply`/100 m, đoạn hỏng trên trục gộp;
- với mỗi chỗ nối: PSNR các khung test trong ±24 khung quanh giữa đoạn chồng, mức kém hơn trung vị cả lần dựng, và PSNR của hai chunk trên khung test chung của đoạn chồng.

Mục chunked không xếp hạng cùng `groups`: so tay số gộp của #16 với dòng #15 (cùng tập khung test). Chưa chạy trên run thật. Kiểu TypeScript của Studio (`src/api/generated/`) chưa sinh lại: chạy `npm run types`.

### Lệnh

```bash
export D2S_W1_INDEX=<chỉ số đã chọn>                 # sau khi xem waymo-stats
bash envs/week04_vast.sh data-waymo                  # nếu máy chưa có: tải, ingest ứng viên, in waymo-stats
bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 splits'
BENCH_GPUS=0 bash envs/vast.sh wave 3                # mọi run còn lại; --p2 thêm #12, #13
bash envs/vast.sh pull
python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml
```

Chưa kiểm trên máy thật: `prep.mcgs` với ba camera Waymo và VRAM ở độ phân giải gốc; ingest các segment ứng viên; VRAM và thời gian của #11 trên 864 khung; `approach check` trên W1 mới (`bench_queue.sh` kiểm đầu hàng đợi).

## 7. Đợt 3 đã chạy (27–28/11/2026)

Kết quả, so chunk với một khối và lỗi gặp: `benchmark_week04_wave3_results.md`.

- **W1 đã chọn: chỉ số 83**, `segment-11839652018869852123_2565_000_2585_000`, 198 khung, 575 m,
  29,2 m/s, 13 xe chạy, không có người, trời sáng. Chọn bằng `waymo-stats` trong 8 ứng viên (41 bị
  lóa nắng, 46 ngắn hơn 8 m). `WAYMO_INDEX` mặc định giờ là 83 (`WAYMO_W1_CHOSEN`); không cần
  `D2S_W1_INDEX` nữa. `EST_GPU_HOURS` sửa theo số đo.
- Đã chạy đủ #1, #2, #3, #4, #8, #10, #11, #15, #16 (hai chunk). #11 chạy thành công ở lần thứ 3,
  sau hai bản sửa trong `drive2sim/recon/` (FastGS prune `RigidNodes` về 0 điểm; chưa commit).
- #8 MCGS-SLAM trên W1: ATE 78,6 m, PSNR 13,2 dB, dưới ngưỡng của plan; chưa đổi cấu hình.
