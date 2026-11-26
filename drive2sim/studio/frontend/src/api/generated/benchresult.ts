// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type TieThresholdDb = number;
export type TieThresholdSource = "default" | "calibration";
export type Warnings = string[];
export type BenchTrack = "A" | "B" | "C";
export type Cell = ("A-LB" | "A-L" | "A-0") | null;
export type SceneId = string;
export type SplitId = string;
export type WinnerRunId = string | null;
export type DecidedBy = ("avg" | "r2" | "r3" | "train_time") | null;
export type RunId = string;
export type ApproachId = string;
export type Rank = number;
export type TiedWithBest = boolean;
export type Psnr = number | null;
export type Ssim = number | null;
export type Lpips = number | null;
export type AvgError = number | null;
export type PsnrPinhole = number | null;
export type PsnrFisheye = number | null;
export type ShiftPsnr = number | null;
export type ShiftLpips = number | null;
export type ShiftPsnrCc = number | null;
export type ShiftLpipsCc = number | null;
export type ShiftCameras = string[];
export type AbsRel = number;
export type Rmse = number;
export type Delta125 = number;
export type ExportGapDb = number | null;
export type ExportApplicable = boolean;
export type PlyGaussians = number | null;
export type PlySizeMb = number | null;
export type TrainTimeS = number | null;
export type ReconTimeS = number | null;
export type PeakVramMb = number | null;
export type RenderFps = number | null;
export type Gpu = string | null;
export type DistanceM = number | null;
export type TrainSPer100M = number | null;
export type PlyMbPer100M = number | null;
export type AteRmseM = number | null;
export type StartM = number;
export type EndM = number;
export type NumFrames = number;
export type MinPsnr = number;
export type FramesMeasured = number;
export type Coverage = number | null;
export type Rows = BenchmarkRunRow[];
export type Warnings1 = string[];
export type Groups = BenchmarkGroup[];

/**
 * Schema của file kết quả benchmark, do ``d2s benchmark --out`` ghi.
 *
 * Attributes:
 *     tie_threshold_db: Ngưỡng PSNR coi hai run là ngang nhau, dB.
 *     tie_threshold_source: ``calibration`` khi ngưỡng đo từ các run lặp theo seed của file định
 *         nghĩa, ``default`` khi dùng ngưỡng mặc định 0,3 dB (thêm từ ``d2s-benchresult/0.2``).
 *     warnings: Cảnh báo chung của cả benchmark, ví dụ không hiệu chỉnh được ngưỡng.
 *     groups: Kết quả từng nhóm, theo thứ tự trong file định nghĩa.
 */
export interface BenchmarkResult {
  schema_version: SchemaVersion;
  tie_threshold_db: TieThresholdDb;
  tie_threshold_source?: TieThresholdSource;
  warnings?: Warnings;
  groups: Groups;
}
/**
 * Kết quả xếp hạng của một nhóm run.
 *
 * Attributes:
 *     bench_track: Track benchmark.
 *     cell: Ô của track A, nếu có.
 *     scene_id: Cảnh.
 *     split_id: Split.
 *     winner_run_id: Run thắng theo quy tắc chọn method; ``None`` nếu nhóm không có run nào có
 *         kết quả đánh giá, hoặc các run ngang nhau không phân xử được (khác GPU, thiếu thời gian
 *         train) hoặc đo bằng protocol không so được.
 *     decided_by: Bước đã chọn người thắng: ``avg`` (chỉ một run ngang run tốt nhất), ``r2``,
 *         ``r3`` hoặc ``train_time``; ``None`` khi không có người thắng (thêm từ
 *         ``d2s-benchresult/0.2``).
 *     rows: Các run, theo thứ hạng.
 *     warnings: Cảnh báo khi so (khác máy, khác protocol, run chưa có kết quả).
 */
export interface BenchmarkGroup {
  bench_track: BenchTrack;
  cell?: Cell;
  scene_id: SceneId;
  split_id: SplitId;
  winner_run_id?: WinnerRunId;
  decided_by?: DecidedBy;
  rows: Rows;
  warnings?: Warnings1;
}
/**
 * Kết quả của một run trong một nhóm benchmark; trường nào chưa đo thì để trống.
 *
 * Attributes:
 *     run_id: Run.
 *     approach_id: Approach của run.
 *     rank: Thứ hạng trong nhóm, bắt đầu từ 1.
 *     tied_with_best: ``True`` nếu PSNR kém run có Avg tốt nhất dưới ngưỡng ngang nhau (gồm
 *         chính run đó).
 *     psnr: PSNR trên tập test, dB.
 *     ssim: SSIM.
 *     lpips: LPIPS.
 *     avg_error: Chỉ số Avg của Mip-NeRF, thấp là tốt.
 *     psnr_pinhole: PSNR trung bình các camera pinhole (R5), nếu có.
 *     psnr_fisheye: PSNR trung bình các camera fisheye (R5), nếu có.
 *     shift_psnr: PSNR thô trên camera chỉ test (R2), nếu split có camera chỉ test.
 *     shift_lpips: LPIPS thô trên camera chỉ test (R2).
 *     shift_psnr_cc: PSNR sau chỉnh màu trên camera chỉ test (R2); số dùng để phân xử hoà.
 *     shift_lpips_cc: LPIPS sau chỉnh màu trên camera chỉ test (R2).
 *     shift_cameras: ``camera_id`` của các camera chỉ test; rỗng nếu split không có.
 *     depth: Độ sâu render so với LiDAR (R4), nếu đã đo.
 *     export_gap_db: Chênh PSNR giữa render gốc và render ``.ply`` (R3), nếu đã đo.
 *     export_applicable: ``False`` khi approach không xuất được ``.ply`` nền tĩnh (R3 không áp
 *         dụng, khác với chưa đo); ``True`` khi có xuất hoặc không biết.
 *     ply_gaussians: Số Gaussian trong ``scene.ply``.
 *     ply_size_mb: Dung lượng ``scene.ply``, megabyte.
 *     train_time_s: Thời gian train, giây.
 *     recon_time_s: Tổng thời gian dựng cảnh từ stage đầu tới khi xuất xong ``scene.ply``, giây
 *         (R8).
 *     peak_vram_mb: VRAM cao nhất khi train, megabyte.
 *     render_fps: Tốc độ render, khung mỗi giây.
 *     gpu: GPU đã chạy run.
 *     distance_m: Quãng đường đường gấp khúc qua các khung test của camera đã train, mét (R6).
 *     train_s_per_100m: Thời gian train chia cho ``distance_m / 100`` (R6); chỉ điền ở track B.
 *     ply_mb_per_100m: Dung lượng ``scene.ply`` chia cho ``distance_m / 100`` (R6); chỉ track B.
 *     ate_rmse_m: ATE do method tự báo (``mcgs_eval.json``), mét; không tính lại (R6).
 *     bad_segments: Các đoạn đường hỏng (R7) theo nhóm camera, khoá ``pinhole`` hoặc ``fisheye``
 *         (chỉ nhóm có ảnh test); chỉ tính trên camera đã train. Rỗng nếu không đo được.
 *     frames_measured: Số cặp (khung, camera đã train) có metric từng ảnh; 0 nếu run chưa có.
 *     coverage: Tỉ lệ điểm ảnh được chấm (``Summary.coverage``, thêm từ ``d2s-benchresult/0.3``);
 *         để trống nếu stage đánh giá không ghi (``eval.standard``).
 */
export interface BenchmarkRunRow {
  run_id: RunId;
  approach_id: ApproachId;
  rank: Rank;
  tied_with_best: TiedWithBest;
  psnr?: Psnr;
  ssim?: Ssim;
  lpips?: Lpips;
  avg_error?: AvgError;
  psnr_pinhole?: PsnrPinhole;
  psnr_fisheye?: PsnrFisheye;
  shift_psnr?: ShiftPsnr;
  shift_lpips?: ShiftLpips;
  shift_psnr_cc?: ShiftPsnrCc;
  shift_lpips_cc?: ShiftLpipsCc;
  shift_cameras?: ShiftCameras;
  depth?: DepthMetrics | null;
  export_gap_db?: ExportGapDb;
  export_applicable?: ExportApplicable;
  ply_gaussians?: PlyGaussians;
  ply_size_mb?: PlySizeMb;
  train_time_s?: TrainTimeS;
  recon_time_s?: ReconTimeS;
  peak_vram_mb?: PeakVramMb;
  render_fps?: RenderFps;
  gpu?: Gpu;
  distance_m?: DistanceM;
  train_s_per_100m?: TrainSPer100M;
  ply_mb_per_100m?: PlyMbPer100M;
  ate_rmse_m?: AteRmseM;
  bad_segments?: BadSegments;
  frames_measured?: FramesMeasured;
  coverage?: Coverage;
}
/**
 * Độ sâu render so với độ sâu LiDAR trên tập test (yêu cầu R4 của benchmark).
 *
 * Chỉ đo ở điểm ảnh có tia LiDAR trúng, hợp lệ và có độ sâu render dương; mỗi ảnh có trọng số
 * bằng nhau. ``drive2sim/recon/rendering/depth_metrics.py`` tính các giá trị này.
 *
 * Attributes:
 *     abs_rel: Sai số tương đối tuyệt đối trung bình, ``|pred - lidar| / lidar``; thấp là tốt.
 *     rmse: Căn bậc hai của sai số bình phương trung bình, mét; thấp là tốt.
 *     delta_1_25: Tỉ lệ điểm có ``max(pred/lidar, lidar/pred) < 1.25``; cao là tốt.
 */
export interface DepthMetrics {
  abs_rel: AbsRel;
  rmse: Rmse;
  delta_1_25: Delta125;
}
export interface BadSegments {
  [k: string]: BadSegment[];
}
/**
 * Một đoạn đường bị hỏng (yêu cầu R7): nhiều ảnh test liền nhau thấp hơn hẳn trung vị.
 *
 * Attributes:
 *     start_m: Quãng đường từ ảnh test đầu tiên tới đầu đoạn, mét (đường gấp khúc qua vị trí
 *         camera của các ảnh test).
 *     end_m: Quãng đường tới cuối đoạn, mét.
 *     num_frames: Số khung test liền nhau trong đoạn.
 *     min_psnr: PSNR thấp nhất trong đoạn, dB.
 */
export interface BadSegment {
  start_m: StartM;
  end_m: EndM;
  num_frames: NumFrames;
  min_psnr: MinPsnr;
}
