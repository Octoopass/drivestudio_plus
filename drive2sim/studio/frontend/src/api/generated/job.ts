// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type JobId = string;
export type Kind = "stage" | "render";
export type StageId = string | null;
export type OutputDir = string;
export type Env = string;

/**
 * Schema của file ``job.json`` engine giao cho ``d2s-worker`` (CONTRACTS mục 9.1).
 *
 * Worker chạy trong môi trường riêng (codebase ngoài, hoặc tiến trình render của viewer); file
 * này là cách duy nhất engine nói cho worker biết phải làm gì.
 *
 * Attributes:
 *     job_id: Định danh việc, thường là ``<run_id>/<stage_id>``.
 *     kind: ``stage`` (chạy một stage) hoặc ``render`` (giữ model và render cho viewer).
 *     stage_id: Stage cần chạy; chỉ có khi ``kind`` là ``stage``.
 *     params: Tham số đã gộp của stage.
 *     inputs: Tên đầu vào → đường dẫn tương đối với ``D2S_WORKDIR``.
 *     output_dir: Thư mục worker ghi kết quả, tương đối với ``D2S_WORKDIR``.
 *     env: Tên file môi trường trong ``envs/``, không có đuôi ``.yml``.
 */
export interface JobSpec {
  schema_version: SchemaVersion;
  job_id: JobId;
  kind: Kind;
  stage_id?: StageId;
  params?: Params;
  inputs?: Inputs;
  output_dir: OutputDir;
  env: Env;
}
export interface Params {
  [k: string]: unknown;
}
export interface Inputs {
  [k: string]: string;
}
