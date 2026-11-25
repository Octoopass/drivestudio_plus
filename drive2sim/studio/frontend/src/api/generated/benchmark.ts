// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
/**
 * @minItems 1
 */
export type Entries = [BenchmarkEntry, ...BenchmarkEntry[]];
export type BenchTrack = "A" | "B" | "C";
export type Cell = ("A-LB" | "A-L" | "A-0") | null;
export type SceneId = string;
export type SplitId = string;
/**
 * @minItems 1
 */
export type Runs = [string, ...string[]];

/**
 * Schema của file định nghĩa benchmark ``benchmarks/<tên>.yaml``.
 *
 * Attributes:
 *     entries: Các nhóm run được xếp hạng riêng.
 */
export interface BenchmarkFile {
  schema_version: SchemaVersion;
  entries: Entries;
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
  scene_id: SceneId;
  split_id: SplitId;
  runs: Runs;
}
