// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type RunId = string;
export type SceneId = string;
export type SplitId = string;
export type ApproachId = string;
export type Version = string;
export type Preset = string | null;
export type Quick = boolean;
export type Level = "L1" | "L2" | "L3" | "L4";
export type Executor = "local" | "kaggle";
export type Status = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export type CreatedUtc = string | null;
export type LidarUsage = string[];
export type StageId = string;
export type Status1 = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export type CacheHit = boolean;
export type CacheKey = string | null;
export type DurationS = number;
export type ReusedDurationS = number | null;
export type PeakVramMb = number;
export type Code = string;
export type Message = string;
export type Stages = StageRecord[];

/**
 * Schema của file ``d2s/run.json`` trong thư mục run (CONTRACTS mục 6.2).
 *
 * Engine tạo file này khi tạo run và ghi lại sau mỗi lần một stage đổi trạng thái; Studio và
 * ``d2s run status`` đọc nó để hiện tiến độ.
 *
 * Attributes:
 *     run_id: Định danh run, ``YYYYMMDD-HHMMSS-<approach_id>-<4 hex>``.
 *     scene_id: Cảnh của run.
 *     split_id: Split của run.
 *     approach: Approach và lựa chọn đã quyết định.
 *     level: Mức năng lực của approach.
 *     executor: Nơi chạy, ``local`` hoặc ``kaggle``.
 *     status: Trạng thái của cả run.
 *     created_utc: Thời điểm tạo run.
 *     lidar_usage: Các stage dùng LiDAR; rỗng nếu run không dùng LiDAR.
 *     stages: Trạng thái của từng stage theo thứ tự chạy.
 */
export interface RunRecord {
  schema_version: SchemaVersion;
  run_id: RunId;
  scene_id: SceneId;
  split_id: SplitId;
  approach: RunApproach;
  level: Level;
  executor: Executor;
  status: Status;
  created_utc?: CreatedUtc;
  lidar_usage?: LidarUsage;
  stages?: Stages;
}
/**
 * Approach của một run cùng các lựa chọn đã quyết định, ghi trong ``run.json``.
 *
 * Attributes:
 *     approach_id: Approach đã chạy.
 *     version: Phiên bản file approach lúc tạo run.
 *     preset: Preset đã chọn, nếu có.
 *     options: Giá trị cụ thể của mọi mục cấu hình, kể cả mục để ``auto``; Studio lọc và so sánh
 *         run theo trường này.
 *     quick: ``True`` nếu là run chạy thử nhanh (``--quick``): train ít iteration để kiểm tra
 *         approach chạy được từ đầu tới cuối; metric không so được với run đủ iteration
 *         (thêm từ ``d2s-run/0.3``).
 */
export interface RunApproach {
  approach_id: ApproachId;
  version: Version;
  preset?: Preset;
  options?: Options;
  quick?: Quick;
}
export interface Options {
  [k: string]: boolean | number | string;
}
/**
 * Trạng thái và kết quả của một stage trong ``run.json``.
 *
 * Attributes:
 *     stage_id: Stage.
 *     status: ``pending``, ``running``, ``succeeded``, ``failed`` hoặc ``cancelled``.
 *     cache_hit: ``True`` nếu dùng lại kết quả có sẵn trong cache.
 *     cache_key: Khoá cache với stage có cache; ``None`` với stage khác.
 *     duration_s: Thời gian máy thực sự chạy stage này trong run, giây, cộng dồn qua các lần
 *         chạy tiếp; ``0`` khi dùng lại kết quả cũ.
 *     reused_duration_s: Khi dùng lại kết quả từ cache: thời gian lần chạy gốc đã tạo ra kết
 *         quả đó (đọc từ ``artifact.json``); ``None`` nếu không dùng lại hoặc không biết.
 *     peak_vram_mb: VRAM cao nhất, megabyte, lấy lớn nhất qua các lần chạy; ``0`` khi dùng
 *         lại kết quả cũ hoặc không đo được.
 *     params: Tham số đã gộp, để chạy tiếp đúng như lúc tạo run (thêm từ ``d2s-run/0.2``).
 *     error: Lỗi nếu stage ``failed``.
 */
export interface StageRecord {
  stage_id: StageId;
  status: Status1;
  cache_hit?: CacheHit;
  cache_key?: CacheKey;
  duration_s?: DurationS;
  reused_duration_s?: ReusedDurationS;
  peak_vram_mb?: PeakVramMb;
  params?: Params;
  error?: ErrorInfo | null;
}
export interface Params {
  [k: string]: unknown;
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
