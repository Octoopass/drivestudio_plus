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
export type OccupiedPsnr = number | null;
export type TrainTimeS = number | null;
export type PeakVramMb = number | null;
export type NumGaussians = number | null;

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
 *     train_time_s: Thời gian train, giây; để trống nếu stage đánh giá không biết (thời gian
 *         của từng stage luôn có trong ``run.json``).
 *     peak_vram_mb: VRAM cao nhất khi train, megabyte; để trống nếu không biết.
 *     num_gaussians: Số Gaussian của model cuối; để trống nếu không đọc được.
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
 *     occupied_psnr: PSNR trong vùng vật động; chỉ có khi cảnh có box.
 */
export interface Metrics {
  psnr: Psnr;
  ssim: Ssim;
  lpips: Lpips;
  human_psnr?: HumanPsnr;
  vehicle_psnr?: VehiclePsnr;
  occupied_psnr?: OccupiedPsnr;
}
export interface PerCamera {
  [k: string]: Metrics;
}
