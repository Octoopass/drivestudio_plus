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
export type PlyPsnr = number | null;
export type ExportStatic = boolean;
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
export type BenchTrack1 = "A" | "B" | "C";
export type Label = string;
export type SceneId1 = string;
export type ApproachIds = string[];
export type SplitId1 = string;
export type RunId1 = string;
export type ApproachId1 = string;
/**
 * @minItems 2
 * @maxItems 2
 */
export type FrameRange = [unknown, unknown];
export type Psnr1 = number | null;
export type Ssim1 = number | null;
export type Lpips1 = number | null;
export type TrainTimeS1 = number | null;
export type PeakVramMb1 = number | null;
export type PlySizeMb1 = number | null;
export type Gpu1 = string | null;
export type Chunks = ChunkRow[];
export type Psnr2 = number | null;
export type Ssim2 = number | null;
export type Lpips2 = number | null;
export type AvgError1 = number | null;
export type ShiftPsnr1 = number | null;
export type ShiftPsnrCc1 = number | null;
export type DistanceM1 = number | null;
export type TrainTimeS2 = number | null;
export type PeakVramMb2 = number | null;
export type PlySizeMb2 = number | null;
export type Gpu2 = string | null;
export type TrainSPer100M1 = number | null;
export type PlyMbPer100M1 = number | null;
export type FramesMeasured1 = number;
export type Index = number;
export type Overlap = [unknown, unknown] | null;
export type AtM = number | null;
export type WindowFrames = number;
export type NumFrames1 = number;
export type PsnrNear = number | null;
export type DropDb = number | null;
export type OverlapFrames = number;
export type OverlapPsnrBefore = number | null;
export type OverlapPsnrAfter = number | null;
export type Seams = ChunkSeam[];
export type Warnings2 = string[];
export type Chunked = BenchmarkChunkedGroup[];

/**
 * Schema của file kết quả benchmark, do ``d2s benchmark --out`` ghi.
 *
 * Attributes:
 *     tie_threshold_db: Ngưỡng PSNR coi hai run là ngang nhau, dB.
 *     tie_threshold_source: ``calibration`` khi ngưỡng đo từ các run lặp theo seed của file định
 *         nghĩa, ``default`` khi dùng ngưỡng mặc định 0,3 dB (thêm từ ``d2s-benchresult/0.2``).
 *     warnings: Cảnh báo chung của cả benchmark, ví dụ không hiệu chỉnh được ngưỡng.
 *     groups: Kết quả từng nhóm, theo thứ tự trong file định nghĩa.
 *     chunked: Kết quả gộp của từng lần dựng chia chunk (thêm từ ``d2s-benchresult/0.5``).
 */
export interface BenchmarkResult {
  schema_version: SchemaVersion;
  tie_threshold_db: TieThresholdDb;
  tie_threshold_source?: TieThresholdSource;
  warnings?: Warnings;
  groups: Groups;
  chunked?: Chunked;
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
 *     export_gap_db: Chênh PSNR giữa render gốc và render ``.ply`` (R3), nếu đã đo; báo kèm.
 *     ply_psnr: PSNR của render ``.ply`` nền tĩnh trên vùng nền tĩnh (R3), số phân xử hoà thứ hai
 *         (thêm từ ``d2s-benchresult/0.4``).
 *     export_static: ``False`` khi ``.ply`` xuất ra chưa phải nền tĩnh của cảnh đã train (PVG,
 *         DeformableGS: tham số gốc, chưa tính theo thời gian), nên R3 không dùng để phân xử
 *         (thêm từ ``d2s-benchresult/0.4``).
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
  ply_psnr?: PlyPsnr;
  export_static?: ExportStatic;
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
/**
 * Kết quả gộp của một lần dựng chia chunk (thêm từ ``d2s-benchresult/0.5``).
 *
 * Metric gộp lấy trung bình theo ảnh trên mọi ảnh test của camera đã train, mỗi ảnh một lần
 * (luật chia đoạn chồng ở ``ChunkSeam``); cùng cách tính ``overall`` của ``summary.json``.
 * Không xếp hạng cùng các nhóm ``groups``; so tay với run một khối cùng cảnh.
 *
 * Attributes:
 *     bench_track: Track benchmark.
 *     label: Tên hiển thị của file định nghĩa.
 *     scene_id: Cảnh.
 *     approach_ids: Approach của các chunk (thường chỉ một).
 *     chunks: Các chunk theo khung đầu.
 *     psnr: PSNR gộp, dB.
 *     ssim: SSIM gộp.
 *     lpips: LPIPS gộp; ``None`` nếu có ảnh thiếu LPIPS.
 *     avg_error: Avg của Mip-NeRF từ ba số gộp.
 *     shift_psnr: PSNR gộp trên camera chỉ test (R2).
 *     shift_psnr_cc: PSNR sau chỉnh màu gộp trên camera chỉ test (R2).
 *     distance_m: Quãng đường qua các khung test đã gộp, mét, theo ``ego_pose/`` của cảnh.
 *     train_time_s: Tổng thời gian train các chunk, giây; ``None`` nếu thiếu một chunk.
 *     peak_vram_mb: VRAM cao nhất trong các chunk.
 *     ply_size_mb: Tổng dung lượng ``scene.ply`` các chunk; ``None`` nếu thiếu một chunk.
 *     gpu: GPU nếu mọi chunk chạy cùng một loại; ``None`` nếu khác hoặc thiếu.
 *     train_s_per_100m: ``train_time_s`` chia ``distance_m / 100`` (R6).
 *     ply_mb_per_100m: ``ply_size_mb`` chia ``distance_m / 100`` (R6).
 *     frames_measured: Số cặp (khung, camera đã train) có metric sau khi gộp.
 *     bad_segments: Đoạn đường hỏng (R7) trên trục gộp, theo nhóm camera.
 *     seams: PSNR quanh từng chỗ nối.
 *     warnings: Cảnh báo khi gộp (khác approach, khác GPU, chunk hở, thiếu metric).
 */
export interface BenchmarkChunkedGroup {
  bench_track: BenchTrack1;
  label: Label;
  scene_id: SceneId1;
  approach_ids: ApproachIds;
  chunks: Chunks;
  psnr?: Psnr2;
  ssim?: Ssim2;
  lpips?: Lpips2;
  avg_error?: AvgError1;
  shift_psnr?: ShiftPsnr1;
  shift_psnr_cc?: ShiftPsnrCc1;
  distance_m?: DistanceM1;
  train_time_s?: TrainTimeS2;
  peak_vram_mb?: PeakVramMb2;
  ply_size_mb?: PlySizeMb2;
  gpu?: Gpu2;
  train_s_per_100m?: TrainSPer100M1;
  ply_mb_per_100m?: PlyMbPer100M1;
  frames_measured?: FramesMeasured1;
  bad_segments?: BadSegments1;
  seams?: Seams;
  warnings?: Warnings2;
}
/**
 * Một chunk trong kết quả của một lần dựng chia chunk.
 *
 * Attributes:
 *     split_id: Split của chunk.
 *     run_id: Run của chunk.
 *     approach_id: Approach của run.
 *     frame_range: Đoạn khung của chunk trong cảnh, tính cả hai đầu.
 *     psnr: PSNR của run trên tập test của chính chunk (``summary.json``), dB.
 *     ssim: SSIM.
 *     lpips: LPIPS.
 *     train_time_s: Thời gian train, giây.
 *     peak_vram_mb: VRAM cao nhất khi train, megabyte.
 *     ply_size_mb: Dung lượng ``scene.ply``, megabyte.
 *     gpu: GPU đã chạy run.
 */
export interface ChunkRow {
  split_id: SplitId1;
  run_id: RunId1;
  approach_id: ApproachId1;
  frame_range: FrameRange;
  psnr?: Psnr1;
  ssim?: Ssim1;
  lpips?: Lpips1;
  train_time_s?: TrainTimeS1;
  peak_vram_mb?: PeakVramMb1;
  ply_size_mb?: PlySizeMb1;
  gpu?: Gpu1;
}
export interface BadSegments1 {
  [k: string]: BadSegment[];
}
/**
 * PSNR quanh chỗ nối giữa chunk ``index`` và chunk ``index + 1`` (R6).
 *
 * Khung test trong đoạn chồng thuộc chunk có khung đó xa mép hơn (nửa đầu đoạn chồng thuộc
 * chunk trước, nửa sau thuộc chunk sau), nên metric gộp đếm mỗi ảnh một lần.
 *
 * Attributes:
 *     index: Số thứ tự chunk trước chỗ nối, từ 1.
 *     overlap: Đoạn khung chung của hai chunk, tính cả hai đầu; ``None`` nếu hai chunk không
 *         chồng (khi đó chỗ nối là giữa khung cuối chunk trước và khung đầu chunk sau).
 *     at_m: Quãng đường tới chỗ nối, mét (trên trục chung của các khung test đã gộp).
 *     window_frames: Nửa độ rộng của vùng quanh chỗ nối, số khung của cảnh.
 *     num_frames: Số khung test (đã gộp) trong vùng quanh chỗ nối.
 *     psnr_near: PSNR trung bình theo khung trong vùng đó, dB.
 *     drop_db: Trung vị PSNR theo khung của cả lần dựng trừ ``psnr_near``; dương là chỗ nối
 *         kém hơn phần còn lại.
 *     overlap_frames: Số khung test trong đoạn chồng mà cả hai chunk đều đo.
 *     overlap_psnr_before: PSNR của chunk trước trên các khung đó, dB.
 *     overlap_psnr_after: PSNR của chunk sau trên các khung đó, dB.
 */
export interface ChunkSeam {
  index: Index;
  overlap?: Overlap;
  at_m?: AtM;
  window_frames: WindowFrames;
  num_frames: NumFrames1;
  psnr_near?: PsnrNear;
  drop_db?: DropDb;
  overlap_frames?: OverlapFrames;
  overlap_psnr_before?: OverlapPsnrBefore;
  overlap_psnr_after?: OverlapPsnrAfter;
}
