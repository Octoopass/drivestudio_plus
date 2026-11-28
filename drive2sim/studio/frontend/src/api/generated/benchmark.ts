// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SceneId = string;
export type SplitId = string;
/**
 * @minItems 2
 */
export type Runs = [string, string, ...string[]];
export type BenchTrack = "A" | "B" | "C";
export type Cell = ("A-LB" | "A-L" | "A-0") | null;
export type SceneId1 = string;
export type SplitId1 = string;
/**
 * @minItems 1
 */
export type Runs1 = [string, ...string[]];
export type Entries = BenchmarkEntry[];
export type BenchTrack1 = "A" | "B" | "C";
export type Label = string;
export type SceneId2 = string;
/**
 * @minItems 2
 */
export type Chunks = [BenchmarkChunk, BenchmarkChunk, ...BenchmarkChunk[]];
export type SplitId2 = string;
export type RunId = string;
export type Chunked = BenchmarkChunkedEntry[];

/**
 * Schema của file định nghĩa benchmark ``benchmarks/<tên>.yaml``.
 *
 * Attributes:
 *     calibration: Các run lặp theo seed để hiệu chỉnh ngưỡng hoà (thêm từ
 *         ``d2s-benchmark/0.2``); thiếu thì dùng ngưỡng mặc định.
 *     entries: Các nhóm run được xếp hạng riêng.
 *     chunked: Các lần dựng chia chunk (thêm từ ``d2s-benchmark/0.3``); không xếp hạng.
 */
export interface BenchmarkFile {
  schema_version: SchemaVersion;
  calibration?: BenchmarkCalibration | null;
  entries?: Entries;
  chunked?: Chunked;
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
/**
 * Một lần dựng cảnh dài chia chunk, gộp lại như một run (ARCHITECTURE mục 2.17, R6).
 *
 * Các chunk là split có ``frame_range`` chồng nhau trên cùng cảnh, mỗi chunk một run thường;
 * ``d2s benchmark`` gộp metric từng ảnh của các chunk lên một trục quãng đường và đo PSNR quanh
 * chỗ nối. Thêm từ ``d2s-benchmark/0.3``.
 *
 * Attributes:
 *     bench_track: Track benchmark, thường là ``B`` (quãng đường dài).
 *     label: Tên hiển thị, ví dụ ``"#11 OmniRe 3DGEER, 6 chunk"``.
 *     scene_id: Cảnh của mọi chunk.
 *     chunks: Các chunk, ít nhất hai; thứ tự không quan trọng (xếp theo khung đầu).
 */
export interface BenchmarkChunkedEntry {
  bench_track?: BenchTrack1;
  label: Label;
  scene_id: SceneId2;
  chunks: Chunks;
}
/**
 * Một chunk của một lần dựng chia chunk: một run thường trên split của đoạn đó.
 *
 * Attributes:
 *     split_id: Split của chunk (``frame_range`` là đoạn khung của chunk).
 *     run_id: Run đã train trên split đó.
 */
export interface BenchmarkChunk {
  split_id: SplitId2;
  run_id: RunId;
}
