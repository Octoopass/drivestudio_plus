// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type JobId = string;
export type Status = "succeeded" | "failed";
export type DurationS = number;
export type PeakVramMb = number;
export type Code = string;
export type Message = string;

/**
 * Schema của file ``result.json`` worker ghi khi xong việc, kể cả khi thất bại.
 *
 * Định dạng ở CONTRACTS mục 9.1.
 *
 * Attributes:
 *     job_id: Định danh việc, giống ``job.json``.
 *     status: ``succeeded`` hoặc ``failed``.
 *     outputs: Tên đầu ra → đường dẫn; rỗng khi thất bại.
 *     duration_s: Thời gian chạy, giây.
 *     peak_vram_mb: VRAM cao nhất, megabyte.
 *     error: Lỗi; có khi và chỉ khi ``status`` là ``failed``.
 */
export interface JobResult {
  schema_version: SchemaVersion;
  job_id: JobId;
  status: Status;
  outputs?: Outputs;
  duration_s: DurationS;
  peak_vram_mb: PeakVramMb;
  error?: ErrorInfo | null;
}
export interface Outputs {
  [k: string]: string;
}
/**
 * Một lỗi theo hình dạng chung của tool, dùng trong ``run.json``, ``result.json`` và API.
 *
 * Cùng một hình dạng ở mọi nơi để CLI, Studio và worker xử lý lỗi theo một cách (CONTRACTS mục
 * 9.1, 11.3).
 *
 * Attributes:
 *     code: Mã lỗi ``snake_case`` trong ``ERROR_CODES``, để máy phân loại.
 *     message: Thông báo tiếng Việt cho người dùng.
 *     details: Thông tin thêm tuỳ loại lỗi.
 */
export interface ErrorInfo {
  code: Code;
  message: Message;
  details?: Details;
}
export interface Details {
  [k: string]: unknown;
}
