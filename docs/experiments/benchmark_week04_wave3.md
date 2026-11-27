# Plan cho Claude Code: đợt 3 benchmark tuần 04 trên vast.ai

Plan này để giao cho Claude Code chạy trên máy của K (có mạng, `vastai`, `ssh`, khoá SSH). Trước khi làm, Claude Code đọc hết file này, `AGENTS.md` và mục 6 của `docs/experiments/benchmark_week04.md`, rồi làm lần lượt từng giai đoạn. Mỗi giai đoạn có điều kiện xong và điểm phải dừng hỏi K. Hai plan trước (`benchmark_week04_claude_code.md`, `benchmark_week04_wave2.md`) chỉ dùng để tra chi tiết; chỗ nào khác nhau thì file này đúng.

Tài liệu liên quan: `docs/experiments/benchmark_week04_wave2_results.md` (kết quả và lỗi đợt 2), project doc `claude/benchmark-week04-decisions-27-11.md` và `claude/hierarchical-3dgs-long-drive.md`. Script: `envs/week04_vast.sh` (chạy ở máy mình), `envs/week04_remote.sh` (chạy ở máy thuê), `envs/vast.sh` (SSH, rsync, chạy đợt), `envs/bench_queue.sh` (hàng đợi), `drive2sim/scripts/setup_benchmark_week04.py` (split, hàng đợi, file định nghĩa, `waymo-stats`).

## 0. Quy tắc

- **Tiền:** máy tính tiền từ lúc `rent` tới lúc `destroy`. Mỗi lúc chỉ một máy. Máy không được ngồi không quá 1 giờ: xong đợt hoặc phải chờ K quyết thì báo K ngay. Trần cả tuần 04 là 25 USD. Đợt 2 đã tiêu khoảng 4,1 USD và credit còn 0,58 USD. Hỏi K chi phí đợt 1 và số credit vừa nạp, rồi ghi lại trần còn lại. Ước lượng ở mục 1 vượt trần còn lại thì dừng hỏi trước khi thuê.
- **Hỏi K trước khi:** thuê máy (đưa offer, giá/giờ), chọn W1 (mục 4), huỷ máy, sửa code trong `drive2sim/` (AGENTS.md mục 2.1), push.
- **Bí mật:** không in, không ghi vào repo, không commit token Kaggle, khoá SSH, file SMPL. `.vast.env` đã có trong `.gitignore`.
- **Dữ liệu có license riêng** (KITTI-360, Waymo, SMPL): chỉ để trên máy thuê ở chế độ riêng tư, huỷ máy khi xong. Kết quả đi về Drive bằng `rclone` (`envs/vast_autopull.sh`, `D2S_PULL_LIGHT=1`), như đợt 2.
- **Lỗi script:** sửa trên nhánh `feat/bench-week04-wave3`, chạy `ruff` và test liên quan, commit kèm mô tả lỗi thật đã gặp. Không lách lỗi bằng lệnh tay rồi bỏ đó. Không commit `runs/`, `cache/`, `data/`, `.vast.env`. Không đụng `AGENTS.md` (thay đổi chưa commit của K; đừng dùng `git commit -a`), `.claude/`, `docs/superpowers/`.
- **Ngoài phạm vi đợt này** (K quyết 27/11): R3 (không điều tra gap `.ply`, không sửa `TIME_VARYING_METHODS`), #9 Instant NuRec (đã có người test), #14 SplatAD. #12 và #13 chỉ chạy khi còn thời gian và tiền (mục 7).
- **Theo dõi bước chạy nền** bằng `bash envs/week04_vast.sh status` và `bash envs/vast.sh wave-status 3`, cách nhau 10–20 phút. Không poll liên tục.

## 1. Trạng thái lúc giao (27/11/2026 tối)

- Repo `~/Projects/Phase 3/drivestudio_plus`, nhánh `feat/bench-week04-wave3` (tách từ `origin/drive2sim` `d7b73b2`), **chưa push**. Máy thuê nhận code bằng rsync từ máy mình, nên không cần push để chạy.
- Quyết định của K (`benchmark_week04.md` mục 6):
  - **Thắng A-LB: `omnire renderer=3dgeer fastgs=true`**, thắng A-L: PVG. Cả hai là mặc định của `queue` và `commands`.
  - **W1 đổi** sang một segment Waymo đi xa hơn, dùng cả segment (~198 khung). Ứng viên là 8 segment nhóm High-speed của NOTR: chỉ số 2, 41, 46, 62, 71, 73, 82, 83. Chọn bằng `waymo-stats`, rồi đặt `D2S_W1_INDEX`. Chưa đặt thì W1 tạm là chỉ số 2: **không dựng hàng đợi khi chưa đặt**.
  - **#8 MCGS-SLAM chuyển sang W1**, ba camera trước. MCGS-SLAM trên Waymo qua tool **chưa chạy thật lần nào**.
  - **#11 là một run thường trên cả cảnh dài** (864 khung, 30.000 bước), không chia chunk nữa.
  - **Run thử chia chunk:** #15 (một khối, khung 0–319) so với #16 (hai chunk 0–175 và 144–319, chồng 32 khung) trên cảnh dài, cùng cấu hình.
  - **Một đợt 3 duy nhất** gồm mọi run còn lại.
- Hàng đợi (`setup_benchmark_week04 queue --wave 3`, xếp run dài trước):

| # | Approach, `--set` | Cảnh, split | Ô | GPU-giờ (ước lượng) |
|---|---|---|---|---|
| 1 | omnire `fastgs=true` | W1 mới, `every8_5cams` | W1 A-LB | 1,5 |
| 11 | omnire `renderer=3dgeer fastgs=true` | `kitti360_0003_long`, `every8_stereo_long` | B, một run | 1,3 |
| 2 | pvg | W1 mới | W1 A-L | 1,0 |
| 3 | deformablegs | W1 mới | W1 A-L | 1,0 |
| 15 | như #11 | `kitti360_0003_long`, `every8_stereo_t320` | thử: một khối | 0,8 |
| 10 | như #11 | `kitti360_0003_long`, `every8_stereo_a3` | A3 A-LB | 0,7 |
| 16 ×2 | như #11 | `every8_stereo_t320_c1`, `_c2` | thử: chunked | 0,65 mỗi chunk |
| 8 | mcgs `prep.mcgs.cameras=FRONT,FRONT_LEFT,FRONT_RIGHT` | W1 mới | B | 0,6 |
| 4 | pvg `train.drivestudio.lidar=false` | W1 mới | W1 A-0 | 0,5 |

  Tổng khoảng 8,7 GPU-giờ. Số lấy từ run chạy một mình ở đợt 1, 2 (`EST_GPU_HOURS`): số bước train cố định 30.000 nên cảnh nhiều khung hơn chủ yếu chỉ làm lâu phần eval. Sửa lại sau run đầu. Chạy một GPU lần lượt thì mất khoảng 11–12 giờ thuê kể cả dựng máy, khoảng 7 USD ở 0,575 USD/giờ (giá thật đợt 2).
- **Một GPU, một worker** là mặc định. Lý do: ở đợt 1, container vast.ai bị giới hạn khoảng 61 lõi CPU và hai run song song bị throttle (`benchmark_week04_wave2.md` mục 1). Chỉ dùng máy 2 GPU khi K muốn xong nhanh, và phải kiểm throttle như mục 5.3.
- Máy đợt 2 (offer `62286745`) có thể chưa huỷ, và đã có env, cảnh KITTI-360, venv Waymo.

## 2. Chuẩn bị trên máy mình (K làm, Claude Code kiểm)

| Mục | Kiểm bằng |
|---|---|
| Code | `git status -sb` đang ở `feat/bench-week04-wave3`, chỉ có `AGENTS.md` sửa và `.claude/`, `benchmarks/`, `docs/superpowers/` chưa track |
| Test | `make check` (hoặc `uvx ruff check drive2sim tests` và `.venv/bin/python -m pytest tests/drive2sim/engine tests/drive2sim/scripts -q`) qua hết |
| Hàng đợi | `.venv/bin/python -m drive2sim.scripts.setup_benchmark_week04 queue --wave 3` in đúng 10 dòng như bảng mục 1 |
| Máy cũ | `vastai show instances`. Còn máy đợt 2 thì báo K và hỏi dùng lại hay huỷ. Dùng lại thì bỏ qua 3.1–3.2. Rỗng thì thuê mới |
| Credit | `vastai show user` (số dư). Dưới ước lượng mục 1 thì dừng: K nạp thêm |
| vastai, SSH | `vastai show ssh-keys` có khoá trùng `~/.ssh/id_ed25519.pub` |
| Token Kaggle | `test -s ~/.kaggle/access_token`, `stat -c %a` là 600 (không in nội dung) |
| SMPL | `find ~/Downloads -iname '*neutral*.pkl'` ra đúng một file; đặt `SMPL_PKL` |
| Google | tài khoản đã nhận điều khoản Waymo (cần để tải 8 ứng viên) |
| `D2S_W1_INDEX` | **chưa** đặt ở bước này (`echo ${D2S_W1_INDEX:-trống}`); đặt ở mục 4 |

## 3. Giai đoạn A: máy

1. **Máy mới:** tìm máy 1 GPU:
   ```bash
   vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true reliability>0.98 cpu_ram>=64 cpu_cores_effective>=16 disk_space>=250 inet_down>=500 cuda_max_good>=12.1' -o dph_total
   ```
   Lấy đĩa ≥ 250 GB vì 8 segment Waymo thêm khoảng 10–15 GB tfrecord cộng cảnh đã ingest. Bỏ máy ở Trung Quốc. **Dừng: đưa K 2–3 offer (id, nơi, giá/giờ, RAM, số lõi, độ tin cậy) và chờ K chọn.** K muốn 2 GPU thì đổi `num_gpus=2` và `cpu_cores_effective>=32`.
2. `bash envs/week04_vast.sh rent <offer id>`, rồi `bootstrap`, `setup` (nền), `SMPL_PKL=<file> bash envs/week04_vast.sh creds`, `data-kitti` (nền), `waymo-auth` (tương tác: K mở link, dán mã). Xong khi `status` có `SETUP_DONE`, `EXTRAS_DONE`, `KITTI_DONE`, cảnh dài 864 khung. Chi tiết và rủi ro đã biết ở `benchmark_week04_wave2.md` mục 3.
3. **Máy cũ dùng lại:** chép code mới bằng `bash envs/vast.sh setup`. Lệnh này rsync code, ghi `COMMIT`, chạy lại `install_main.sh` với wheel cache. Đợi `SETUP_DONE`, rồi kiểm `bash envs/vast.sh exec 'cat COMMIT'` khớp `git rev-parse HEAD`. Bước cài mất quá 20 phút trên máy đã có env thì thêm lệnh chỉ rsync vào `envs/vast.sh` và commit. Kiểm `gcloud auth list` còn đăng nhập, không thì chạy `waymo-auth`.
4. `bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 splits'`. Các split trên cảnh dài phải `đã có` hoặc `đã tạo`: `every8_stereo_a3`, `every8_stereo_long`, `every8_stereo_t320`, `every8_stereo_t320_c1`, `every8_stereo_t320_c2`. In `KHÔNG có rẽ` ở A3 thì chỉ ghi nhận (đoạn đã chọn vì có người đi bộ). W1 lúc này báo `chưa có cảnh` là đúng.

## 4. Giai đoạn B: chọn W1

1. Tải và ingest 8 ứng viên (chạy song song với bước cài ở 3.2 được, sau `EXTRAS_DONE` vì cần venv Waymo):
   ```bash
   bash envs/week04_vast.sh data-waymo
   ```
   Lệnh này tải và ingest từng segment bằng venv Waymo, xong thì in `waymo-stats` và `WAYMO_DONE`. Một segment lỗi tải hay ingest thì ghi lại và đi tiếp với các segment còn lại; dưới 3 ứng viên dùng được thì dừng, báo K.
2. **Cho GPU khỏi ngồi không trong lúc chờ** (tuỳ chọn, chỉ khi `SETUP_DONE` mà `WAYMO_DONE` còn lâu): chạy trước #11 bằng `bash envs/vast.sh run omnire --scene kitti360_0003_long --split every8_stereo_long --set renderer=3dgeer --set fastgs=true`. Khi dựng hàng đợi ở mục 5, bỏ dòng #11 khỏi file nếu run này đang chạy hoặc đã xong.
3. Lấy bảng:
   ```bash
   bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 waymo-stats'
   ```
   **Dừng: đưa K bảng này** (chỉ số, số khung, quãng đường, tốc độ, độ đổi hướng, số xe, người, vật chạy), kèm đề xuất theo thứ tự:
   - quãng đường dài nhất;
   - có ít nhất vài xe chạy (`vật chạy` > 0), vì ô A-LB cần box có vật động;
   - có người nếu được (OmniRe dựng người bằng SMPL);
   - trời sáng: xem `data/<cảnh>/images/000_0.*` của 2–3 ứng viên đầu bảng, loại cảnh tối, mưa hay lóa nặng.

   Chờ K chọn chỉ số.
4. Đặt W1 ở máy mình, cho cả shell hiện tại lẫn `.vast.env`:
   ```bash
   export D2S_W1_INDEX=<chỉ số>
   grep -q '^D2S_W1_INDEX=' .vast.env || echo "D2S_W1_INDEX=$D2S_W1_INDEX" >> .vast.env
   bash envs/vast.sh exec 'python -c "from drive2sim.scripts import setup_benchmark_week04 as w; print(w.WAYMO_INDEX, w.WAYMO_SCENE)"'
   ```
   Lệnh cuối phải in đúng chỉ số K chọn. `envs/vast.sh` chỉ chuyển biến này sang máy thuê khi nó có trong shell gọi lệnh, hoặc trong `.vast.env` khi `VAST_HOST` chưa đặt.
5. Tạo split W1 mới và kiểm LiDAR:
   ```bash
   bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 splits && d2s scene lidar-check "$(python -c "from drive2sim.scripts import setup_benchmark_week04 as w; print(w.WAYMO_SCENE)")" | tail -n 3'
   ```

## 5. Giai đoạn C: chạy đợt 3

1. Dựng hàng đợi rồi tách #8 ra để chạy trước, vì MCGS-SLAM trên Waymo chưa chạy thật:
   ```bash
   bash envs/vast.sh exec 'mkdir -p logs benchmarks/queues && python -m drive2sim.scripts.setup_benchmark_week04 queue --wave 3 --out benchmarks/queues/wave3.tsv && grep -P "^8\t" benchmarks/queues/wave3.tsv > benchmarks/queues/wave3_mcgs.tsv && grep -vP "^8\t" benchmarks/queues/wave3.tsv > benchmarks/queues/wave3_rest.tsv && wc -l benchmarks/queues/wave3_*.tsv'
   ```
   `wave3_mcgs.tsv` có 1 dòng, `wave3_rest.tsv` có 9 dòng (8 nếu đã bỏ #11 theo 4.2). Mọi dòng W1 phải có `--scene waymo_<segment K chọn>`. Mọi dòng trên `kitti360_0003_long` phải có `--set renderer=3dgeer --set fastgs=true`. Sai thì dừng và kiểm code đã rsync (`COMMIT`) và `D2S_W1_INDEX`.
2. **#8 trước:**
   ```bash
   bash envs/vast.sh exec 'nohup bash envs/bench_queue.sh benchmarks/queues/wave3_mcgs.tsv --gpus 0 > logs/wave3_mcgs.out 2>&1 &'
   ```
   Đạt khi:
   - run `succeeded`;
   - `prep.mcgs/mcgs_prep.json` có `layout` 4 phần tử, bắt đầu bằng `FRONT`, camera đối tác lặp ở cuối;
   - có quỹ đạo và ATE (`mcgs_eval.json`). Mốc so: MCGS-SLAM trên mẫu Waymo 100613 ở Kaggle cho ATE 1,19 m.

   Không đạt thì làm theo bảng:

   | Gặp | Làm |
   |---|---|
   | lỗi kích thước ảnh, `calib.yml`, chỉ số | đọc traceback, sửa ở `drive2sim/external/mcgs` hoặc `stages/prep/mcgs.py` (hỏi K trước khi sửa `drive2sim/`), thêm test, chạy lại |
   | hết VRAM ở 1920×1280 | báo K. Hướng sửa: thêm tham số thu nhỏ ảnh cho `prep.mcgs`, hoặc chỉ dùng FRONT + một camera bên |
   | chạy xong mà ATE > 5 m hay PSNR < 15 dB | báo K kèm ATE, tỉ lệ căn, PSNR từng khung. Không tự đổi cấu hình |

   Lỗi chưa sửa được ngay thì báo K và vẫn chạy bước 3, để GPU không ngồi không.
3. Các run còn lại:
   ```bash
   bash envs/vast.sh exec 'nohup bash envs/bench_queue.sh benchmarks/queues/wave3_rest.tsv --gpus 0 > logs/wave3.out 2>&1 &'
   ```
   `bench_queue.sh` chạy `approach check` mọi dòng trước. Run lỗi thì đọc log trong `logs/queue-wave3_rest/`, sửa nếu là lỗi script hay cấu hình, rồi chạy lại các dòng trong `failed` (`bash envs/bench_queue.sh logs/queue-wave3_rest/failed --gpus 0`).
   - #11 hết VRAM: báo K (giảm độ phân giải lúc nạp hay chuyển `preload_device` sang `cpu` là quyết định của K); không tự quay về chia chunk.
   - **Hai GPU:** chia `wave3_rest.tsv` thành hai file (các dòng KITTI-360 và các dòng W1), mỗi file một worker `--gpus 0` và `--gpus 1`. Sau 10 phút đầu khi hai run chạy cùng lúc, so `s/bước` trong log train với số chạy một mình (OmniRe khoảng 0,135 s/bước, PVG khoảng 0,08). Chậm hơn quá 1,5 lần thì báo K và đề nghị dừng một worker, vì thời gian train sẽ không còn là thời gian thật (R8).
4. Sau run đầu tiên xong: so GPU-giờ thật với ước lượng, tính lại tổng tiền. Vượt trần còn lại thì báo K.
5. Kiểm từng run (`eval.standard/summary.json`, `train.drivestudio/train_stats.json`, `run.json`):

| # | Đạt khi | Nếu không |
|---|---|---|
| 1 | `succeeded`; có PSNR vùng người, xe nếu cảnh có | báo K |
| 2, 3 | `succeeded`; ghi PSNR so với run W1 cũ (`c40a` 29,79 dB, `8355`) chỉ để tham khảo, khác cảnh | báo K nếu lỗi |
| 4 | `succeeded`; `run.json` có `lidar=false` ở `train.drivestudio`, `lidar_usage` rỗng | báo K |
| 8 | như bước 2 | như bước 2 |
| 10 | `succeeded`; `summary.json` có `shifted` (R2 trên `image_01`); có PSNR vùng người (đoạn A3 chứa người đi bộ khung gốc 629–662) | báo K |
| 11 | `succeeded`; ghi VRAM đỉnh, số Gaussian, thời gian train; có `shifted` | báo K (VRAM, lỗi) |
| 15 | `succeeded`; có `shifted` | báo K |
| 16 ×2 | cả hai chunk `succeeded` | chạy lại chunk lỗi; lỗi lặp lại thì báo K |

## 6. Giai đoạn D: bảng benchmark, so chunk với một khối

1. Trên máy thuê:
   ```bash
   bash envs/vast.sh exec 'python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml --out benchmarks/week04.json'
   ```
   Ghi chú của `definition` phải không còn dòng `#16: chưa có chunk`. Nhóm `W1 A-LB`, `W1 A-L`, `W1 A-0`, `W1 MCGS-SLAM`, `A3 A-LB`, `B2 OmniRe một run cả cảnh`, `thử: một khối 320 khung` chỉ chứa run của đợt 3. Nhóm A2 giữ run của đợt 1, 2 (`f4be`, `7a64`, `fa62`).
2. **Kiểm nhóm chunked #16** (code mới, lần đầu chạy trên run thật):

   | Kiểm | Đạt khi | Nếu không |
   |---|---|---|
   | đọc chỉ số khung | không có cảnh báo `không khớp tập test của split` | báo K. Giả định "DriveStudio ghi chỉ số tính từ khung đầu của split" sai: so `frame_index` trong `frames.jsonl` của chunk 2 với `test` của split `every8_stereo_t320_c2` |
   | tập khung test | `frames_measured` của #16 bằng số ảnh test camera `image_00` của #15 (39 khung test × 1 camera đã train) | báo K |
   | quãng đường | `quãng đường` của #16 khớp `distance_m` của #15 (lệch dưới 1%) | báo K |
   | chỗ nối | 1 dòng, đoạn chồng `144–175`, `chồng: trước / sau` có số với 3 khung | báo K |
3. **So chunk với một khối** (câu hỏi của K): đưa một bảng
   - #16 gộp so với #15: PSNR, SSIM, LPIPS, R2 cc, tổng thời gian train, VRAM đỉnh, tổng MB `.ply`;
   - PSNR quanh chỗ nối của #16 (`PSNR quanh`, `kém trung vị`) và PSNR từng khung của #15 ở cùng các khung đó (đọc `frames.jsonl` của #15 quanh khung 144–175);
   - hai chunk trên 3 khung test chung (`chồng: trước / sau`).

   Kèm kết luận ngắn: chênh dưới khoảng 0,3 dB (ngưỡng hoà) thì chưa đủ để nói cách nào hơn, vì mỗi cấu hình chạy một lần.
4. **Dừng: đưa K** các bảng `d2s benchmark` (W1 ba ô, A3, B: MCGS-SLAM trên W1 và #11 trên cảnh dài, bảng so chunk) kèm GPU-giờ và tiền đã tiêu.

## 7. Nếu còn thời gian và tiền (K quyết)

- `queue --wave 3 --p2` thêm hai dòng: #12 (PVG trên A3, khoảng 1 GPU-giờ) và #13 (BezierGS trên W1 mới, khoảng 2 GPU-giờ, chưa đo). Lấy riêng hai dòng đó (`grep -P "^(12|13)\t"`) vào một file hàng đợi. #13 chỉ chạy khi `d2s approach check beziergs --scene <W1> --split every8_5cams` qua.
- SplatAD (#14): chưa có approach. Không làm trong đợt này nếu K không giao riêng.

## 8. Giai đoạn E: thu kết quả, trả máy, cập nhật tài liệu

1. Kéo kết quả về như đợt 2 (`D2S_PULL_LIGHT=1 bash envs/vast.sh pull`, bản đầy đủ lên Drive bằng `rclone`). Trên máy mình, với `D2S_W1_INDEX` đã export:
   ```bash
   python -m drive2sim.scripts.setup_benchmark_week04 definition && d2s benchmark benchmarks/week04.yaml --out benchmarks/week04.json
   ```
   Kết quả phải giống bảng trên máy thuê.
2. Kiểm đủ mọi run ở bảng mục 1 đều `succeeded` và có trong `runs/` hoặc trên Drive. **Hỏi K rồi** `bash envs/week04_vast.sh destroy`; xác nhận `vastai show instances` rỗng.
3. Cập nhật:
   - `docs/experiments/benchmark_week04_wave3_results.md` (file mới, theo mẫu `benchmark_week04_wave2_results.md`): trạng thái từng run, bảng các ô, MCGS-SLAM trên W1, #11 trên cả cảnh dài, so chunk với một khối, lỗi gặp và cách sửa, chi phí thật;
   - `docs/experiments/benchmark_week04.md` mục 6: W1 đã chọn (chỉ số, segment, quãng đường). Đổi mặc định `WAYMO_INDEX` trong `setup_benchmark_week04.py` sang chỉ số đó (sửa docstring, test liên quan) để lần sau không cần biến môi trường. Sửa `EST_GPU_HOURS` theo số đo thật;
   - `docs/experiments/run_plan.md`: thêm các run (run ID, cấu hình, PSNR, SSIM, LPIPS, thời gian train, VRAM đỉnh, GPU);
   - project doc `claude/benchmark-week04-decisions-27-11.md`: thêm W1 đã chọn, kết quả chính, kết luận so chunk;
   - Sheet (K làm, hoặc Claude Code nếu có connector): thêm #15, #16; #11 đổi thành một run; cột Trạng thái, Run ID, GPU-giờ thật.
4. Chạy `make check` trên máy mình (có mạng nên chạy được cả test Studio, test Python 3.9) và `npm run types` để sinh lại kiểu TypeScript cho `d2s-benchmark/0.3`, `d2s-benchresult/0.5`. Commit trên `feat/bench-week04-wave3`. **Hỏi K trước khi push** và trước khi mở PR vào `drive2sim`.
