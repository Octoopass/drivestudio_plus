// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SceneId = string;
export type SplitId = string;
/**
 * @minItems 2
 */
export type Runs = [string, string, ...string[]];
/**
 * @minItems 1
 */
export type Entries = [BenchmarkEntry, ...BenchmarkEntry[]];
export type BenchTrack = "A" | "B" | "C";
export type Cell = ("A-LB" | "A-L" | "A-0") | null;
export type SceneId1 = string;
export type SplitId1 = string;
/**
 * @minItems 1
 */
export type Runs1 = [string, ...string[]];

/**
 * Schema của file định nghĩa benchmark ``benchmarks/<tên>.yaml``.
 *
 * Attributes:
 *     calibration: Các run lặp theo seed để hiệu chỉnh ngưỡng hoà (thêm từ
 *         ``d2s-benchmark/0.2``); thiếu thì dùng ngưỡng mặc định.
 *     entries: Các nhóm run được xếp hạng riêng.
 */
export interface BenchmarkFile {
  schema_version: SchemaVersion;
  calibration?: BenchmarkCalibration | null;
  entries: Entries;
}
/**
 * Một cấu hình train lặp lại với nhiều seed để đo độ dao động của PSNR (yêu cầu R8).
 *
 * Độ lệch chuẩn σ của PSNR các run này đặt ngưỡng hoà của benchmark: ``max(0,3 dB, 2σ)``. Các
 * run phải cùng một cấu hình, chỉ khác ``seed``; tool không kiểm điều đó.
 *
 * Attributes:
 *     scene_id: Cảnh của các run.
 *     split_id: Split của các run.
 *     runs: ``run_id`` của các run, ít nhất hai.
 */
export interface BenchmarkCalibration {
  scene_id: SceneId;
  split_id: SplitId;
  runs: Runs;
}
/**
 * Một nhóm run được xếp hạng với nhau trong benchmark.
 *
 * ``bench_track`` là track của benchmark (A dựng từng cảnh, B quãng đường dài, C feed-forward),
 * không phải loại camera pinhole hay fisheye.
 *
 * Attributes:
 *     bench_track: Track benchmark: ``A``, ``B`` hoặc ``C``.
 *     cell: Ô của track A theo đầu vào: ``A-LB`` (có LiDAR và box), ``A-L`` (chỉ LiDAR),
 *         ``A-0`` (không LiDAR, không box); track khác để trống.
 *     scene_id: Cảnh của các run.
 *     split_id: Split của các run; các run trong một nhóm phải cùng cảnh và split.
 *     runs: ``run_id`` của các run cần so.
 */
export interface BenchmarkEntry {
  bench_track: BenchTrack;
  cell?: Cell;
  scene_id: SceneId1;
  split_id: SplitId1;
  runs: Runs1;
}
