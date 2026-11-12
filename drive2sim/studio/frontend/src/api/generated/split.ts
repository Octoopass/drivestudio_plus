// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SplitId = string;
export type SceneId = string;
export type Strategy = string;
/**
 * @minItems 2
 * @maxItems 2
 */
export type FrameRange = [unknown, unknown];
/**
 * @minItems 1
 */
export type Cameras = [string, ...string[]];
export type Train = [unknown, unknown][];
export type Test = [unknown, unknown][];

/**
 * Schema của file split ``data/<scene_id>/splits/<split_id>.json`` (CONTRACTS mục 3.1).
 *
 * Split ghi ảnh nào dùng để train, ảnh nào dùng để test. Mọi approach đánh giá trên cùng split
 * nên so sánh được với nhau. Tạo bằng ``drive2sim/core/split.py``.
 *
 * Attributes:
 *     split_id: Định danh split, ``snake_case``, ví dụ ``every8_allcams``.
 *     scene_id: Cảnh của split.
 *     strategy: Chiến lược, ``every_nth_frame:N`` hoặc ``custom``.
 *     frame_range: Khung đầu và khung cuối của đoạn dùng, tính cả hai đầu.
 *     cameras: Các camera dùng trong split, không trùng.
 *     train: Các cặp ``[chỉ số khung, camera_id]`` để train.
 *     test: Các cặp ``[chỉ số khung, camera_id]`` để test; không cặp nào nằm ở cả hai phần.
 */
export interface Split {
  schema_version: SchemaVersion;
  split_id: SplitId;
  scene_id: SceneId;
  strategy: Strategy;
  frame_range: FrameRange;
  cameras: Cameras;
  train: Train;
  test: Test;
}
