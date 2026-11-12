// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type ArtifactType =
  | "scene_subset"
  | "masks"
  | "smpl"
  | "depth_prior"
  | "points_init"
  | "model"
  | "eval_report"
  | "videos"
  | "export_ply"
  | "preview";
export type CacheKey = string | null;
export type StageId = string;
export type Version = string;
export type CodeCommit = string | null;
export type Inputs = string[];
export type CreatedUtc = string;
export type Files = string[];

/**
 * Schema của file ``artifact.json`` mô tả kết quả của một stage (CONTRACTS mục 5.3).
 *
 * Engine ghi file này vào thư mục kết quả của mỗi stage sau khi stage chạy xong. Thư mục cache
 * có ``artifact.json`` nghĩa là kết quả đầy đủ, dùng lại được.
 *
 * Attributes:
 *     artifact_type: Loại artifact, theo bảng CONTRACTS mục 5.4.
 *     cache_key: Khoá cache; bỏ trống với kết quả nằm trong thư mục run (stage không cache).
 *     producer: Stage đã tạo artifact.
 *     params: Tham số đã gộp của stage.
 *     inputs: Khoá của các đầu vào, ví dụ ``["scene:nuscenes_scene-0061@19.9.0"]``.
 *     created_utc: Thời điểm tạo.
 *     files: Các file trong thư mục, đường dẫn tương đối với thư mục đó.
 *     meta: Thông tin thêm tuỳ stage, ví dụ tên model đã dùng.
 */
export interface ArtifactRecord {
  schema_version: SchemaVersion;
  artifact_type: ArtifactType;
  cache_key?: CacheKey;
  producer: Producer;
  params?: Params;
  inputs?: Inputs;
  created_utc: CreatedUtc;
  files?: Files;
  meta?: Meta;
}
/**
 * Stage đã tạo ra một artifact, ghi trong ``artifact.json``.
 *
 * Attributes:
 *     stage_id: Stage đã tạo artifact.
 *     version: Phiên bản của stage lúc tạo.
 *     code_commit: Commit của code lúc tạo, nếu đọc được.
 */
export interface Producer {
  stage_id: StageId;
  version: Version;
  code_commit?: CodeCommit;
}
export interface Params {
  [k: string]: unknown;
}
export interface Meta {
  [k: string]: unknown;
}
