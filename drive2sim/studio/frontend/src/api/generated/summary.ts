// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type Version = string;
export type ImageDtype = "uint8";
export type Resolution = string;
export type Net = string;
export type Package = string;
export type PackageVersion = string;
export type SplitId = string;
export type Psnr = number;
export type Ssim = number;
export type Lpips = number;
export type HumanPsnr = number | null;
export type VehiclePsnr = number | null;
export type DynamicPsnr = number | null;
export type OccupiedPsnr = number | null;
export type TrainTimeS = number | null;
export type PeakVramMb = number | null;
export type NumGaussians = number | null;
export type NumIters = number | null;
export type RenderFps = number | null;
export type RenderResolution = string | null;
export type Speedups = string[];
export type EvalTimeS = number | null;
export type AbsRel = number;
export type Rmse = number;
export type Delta125 = number;
/**
 * @minItems 1
 */
export type Cameras = [string, ...string[]];
export type NumImages = number;
export type Psnr1 = number;
export type Ssim1 = number;
export type Lpips1 = number;
export type PsnrCc = number | null;
export type LpipsCc = number | null;
export type Coverage = number | null;

/**
 * Schema của file ``summary.json``: tóm tắt kết quả đánh giá của một run.
 *
 * Stage ``eval.standard`` ghi file này vào thư mục của nó; Studio và ``d2s report`` đọc nó để
 * lập bảng xếp hạng (CONTRACTS mục 6.3).
 *
 * Attributes:
 *     protocol: Protocol đánh giá.
 *     split_id: Split đã đánh giá.
 *     overall: Metric trên mọi camera.
 *     per_camera: ``camera_id`` → metric của camera đó.
 *     train_time_s: Thời gian train thực, giây: chỉ tính các bước train, cộng qua các lần chạy
 *         tiếp, không tính lúc dừng, nạp checkpoint, đánh giá hay lưu file; để trống nếu không
 *         biết.
 *     peak_vram_mb: VRAM cao nhất khi train, megabyte; để trống nếu không biết.
 *     num_gaussians: Số Gaussian của cả model cuối (mọi node); để trống nếu không đọc được.
 *     num_iters: Số iteration đã train (thêm từ ``d2s-summary/0.3``).
 *     render_fps: Tốc độ render khi đánh giá, khung mỗi giây, chỉ tính thời gian model vẽ ảnh
 *         (thêm từ ``d2s-summary/0.3``).
 *     render_resolution: Độ phân giải ảnh khi đo ``render_fps``, dạng ``<rộng>x<cao>``.
 *     speedups: Thuật toán, kỹ thuật tăng tốc thật sự đã bật khi train, ví dụ ``["fastgs",
 *         "fastergs.fused_adam"]``; rỗng nếu không bật gì hoặc không biết (thêm từ
 *         ``d2s-summary/0.4``). Hai run cùng bật Faster-GS nhưng khác kỹ thuật thì khác ở đây.
 *     eval_time_s: Thời gian đánh giá cuối trong stage train (vẽ ảnh test, tính metric, render
 *         video), giây; phần còn lại của thời gian stage train ngoài ``train_time_s`` và giá trị
 *         này là khởi động, nạp dữ liệu, lưu checkpoint, vẽ ảnh kiểm tra định kỳ (thêm từ
 *         ``d2s-summary/0.5``); để trống nếu không biết.
 *     depth: Độ sâu render so với LiDAR (thêm từ ``d2s-summary/0.6``, protocol 0.4); để trống
 *         nếu cảnh không có LiDAR hoặc run đo bằng protocol cũ.
 *     shifted: Metric trên camera chỉ test (R2, thêm từ ``d2s-summary/0.7``, protocol 0.5); để
 *         trống nếu split không có camera chỉ test. Khi có, ``overall`` chỉ tính trên các ảnh
 *         test của camera đã train.
 *     coverage: Tỉ lệ điểm ảnh được chấm, trung bình trên mọi view test, trong [0, 1] (thêm từ
 *         ``d2s-summary/0.8``). Chỉ ``eval.ncore`` và ``eval.mcgs`` điền: hai stage này chỉ chấm
 *         điểm ảnh ngoài mask thân xe và có alpha > 0,9, nên PSNR của chúng đo trên phần ảnh nhỏ
 *         hơn ``eval.standard``; ``eval.standard`` đo trên mọi điểm hợp lệ nên để trống.
 */
export interface Summary {
  schema_version: SchemaVersion;
  protocol: EvalProtocol;
  split_id: SplitId;
  overall: Metrics;
  per_camera?: PerCamera;
  train_time_s?: TrainTimeS;
  peak_vram_mb?: PeakVramMb;
  num_gaussians?: NumGaussians;
  num_iters?: NumIters;
  render_fps?: RenderFps;
  render_resolution?: RenderResolution;
  speedups?: Speedups;
  eval_time_s?: EvalTimeS;
  depth?: DepthMetrics | null;
  shifted?: ShiftedMetrics | null;
  coverage?: Coverage;
}
/**
 * Protocol đánh giá đã dùng cho một run (SPEC mục 9).
 *
 * Attributes:
 *     version: Phiên bản protocol.
 *     image_dtype: Kiểu ảnh khi đo; hiện chỉ ``uint8`` (ảnh 8 bit).
 *     resolution: Độ phân giải khi đo, ví dụ ``native``.
 *     lpips: Cấu hình LPIPS.
 */
export interface EvalProtocol {
  version: Version;
  image_dtype: ImageDtype;
  resolution: Resolution;
  lpips: LpipsSetting;
}
/**
 * Cấu hình LPIPS đã dùng khi đánh giá; ghi rõ để so sánh được giữa các run (SPEC mục 9).
 *
 * Attributes:
 *     net: Backbone, ví dụ ``alex``.
 *     package: Gói Python đã dùng, ví dụ ``lpips``.
 *     package_version: Phiên bản gói.
 */
export interface LpipsSetting {
  net: Net;
  package: Package;
  package_version: PackageVersion;
}
/**
 * Các metric chất lượng ảnh trên tập test.
 *
 * Attributes:
 *     psnr: PSNR trên toàn ảnh, dB.
 *     ssim: SSIM.
 *     lpips: LPIPS.
 *     human_psnr: PSNR trong vùng người; chỉ có khi cảnh có box.
 *     vehicle_psnr: PSNR trong vùng xe; chỉ có khi cảnh có box.
 *     dynamic_psnr: PSNR trong vùng vật động (mọi người và xe, ``masked_psnr`` của
 *         DriveStudio); chỉ có khi cảnh có box (thêm từ ``d2s-summary/0.3``).
 *     occupied_psnr: PSNR trong vùng không phải bầu trời (``Non-Sky PSNR`` của DriveStudio);
 *         chỉ có khi cảnh có mask bầu trời.
 */
export interface Metrics {
  psnr: Psnr;
  ssim: Ssim;
  lpips: Lpips;
  human_psnr?: HumanPsnr;
  vehicle_psnr?: VehiclePsnr;
  dynamic_psnr?: DynamicPsnr;
  occupied_psnr?: OccupiedPsnr;
}
export interface PerCamera {
  [k: string]: Metrics;
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
/**
 * Metric trên camera chỉ test, không train (yêu cầu R2 của benchmark, protocol 0.5).
 *
 * Đo ở các ảnh test của camera trong ``Split.test_only_cameras``, ví dụ ``image_01`` của
 * KITTI-360 lệch khoảng 0,59 m so với ``image_00``. Camera chưa train nên không có ma trận chỉnh
 * màu học được; ``psnr_cc`` và ``lpips_cc`` đo sau khi khớp một phép affine màu 3x4 cho mỗi ảnh
 * (``drive2sim/recon/rendering/color_match.py``) và là số dùng để phân xử hoà.
 *
 * Attributes:
 *     cameras: ``camera_id`` của các camera chỉ test.
 *     num_images: Số ảnh test đã đo.
 *     psnr: PSNR thô trung bình theo ảnh, dB.
 *     ssim: SSIM trung bình theo ảnh.
 *     lpips: LPIPS thô trung bình theo ảnh.
 *     psnr_cc: PSNR sau chỉnh màu, dB; để trống nếu file metric từng ảnh không có hoặc mọi ảnh
 *         khớp màu thất bại.
 *     lpips_cc: LPIPS sau chỉnh màu; để trống như ``psnr_cc``.
 */
export interface ShiftedMetrics {
  cameras: Cameras;
  num_images: NumImages;
  psnr: Psnr1;
  ssim: Ssim1;
  lpips: Lpips1;
  psnr_cc?: PsnrCc;
  lpips_cc?: LpipsCc;
}
