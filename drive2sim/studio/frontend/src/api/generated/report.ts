// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SceneId = string;
export type CreatedUtc = string;
export type Errors = number;
export type Warnings = number;
export type CheckId = string;
export type Severity = "error" | "warning" | "info";
export type Message = string;
export type CameraId = string | null;
export type FrameIndex = number | null;
export type Checks = Check[];

/**
 * Schema của báo cáo kiểm tra dữ liệu ``data/<scene_id>/reports/validation.json``.
 *
 * Ingest ghi báo cáo này sau khi kiểm tra đồng bộ thời gian, calib, mô hình ống kính, ảnh hỏng
 * hoặc thiếu (SPEC mục 5 bước 2, CONTRACTS mục 3.2); Studio hiện nó ở nhóm A.
 *
 * Attributes:
 *     scene_id: Cảnh được kiểm tra.
 *     created_utc: Thời điểm tạo báo cáo.
 *     summary: Số lỗi và cảnh báo; phải khớp với ``checks``.
 *     checks: Các mục kiểm tra.
 */
export interface ValidationReport {
  schema_version: SchemaVersion;
  scene_id: SceneId;
  created_utc: CreatedUtc;
  summary: ReportSummary;
  checks?: Checks;
}
/**
 * Tóm tắt của báo cáo kiểm tra.
 *
 * Attributes:
 *     errors: Số mục ``error``.
 *     warnings: Số mục ``warning``.
 */
export interface ReportSummary {
  errors: Errors;
  warnings: Warnings;
}
/**
 * Một mục trong báo cáo kiểm tra dữ liệu của cảnh.
 *
 * Attributes:
 *     check_id: Tên phép kiểm tra, dạng ``<nhóm>.<tên>``, ví dụ ``timestamps.camera_sync``.
 *     severity: ``error`` (không chạy được), ``warning`` hoặc ``info``.
 *     message: Mô tả vấn đề bằng tiếng Việt.
 *     camera_id: Camera liên quan, nếu có.
 *     frame_index: Khung liên quan, nếu có.
 *     details: Số liệu thêm, ví dụ ``{"max_offset_us": 6200}``.
 */
export interface Check {
  check_id: CheckId;
  severity: Severity;
  message: Message;
  camera_id?: CameraId;
  frame_index?: FrameIndex;
  details?: Details;
}
export interface Details {
  [k: string]: unknown;
}
