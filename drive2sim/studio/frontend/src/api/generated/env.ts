// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type CodeCommit = string;
export type Python = string;
export type Pytorch = string | null;
export type Cuda = string | null;
export type Gsplat = string | null;
export type Gpu = string | null;
export type Seed = number | null;
export type Executor = "local" | "kaggle" | "colab";
export type Cpu = string | null;
export type CpuCores = number | null;
export type RamGb = number | null;
export type Gpu1 = string | null;
export type GpuCount = number;
export type GpuMemoryMb = number | null;
export type GpuDriver = string | null;
export type Os = string | null;
export type FirstSeenUtc = string | null;
export type Machines = Machine[];

/**
 * Schema của file ``d2s/env.json``: môi trường đã chạy run, để chạy lại được (SPEC mục 11).
 *
 * Engine ghi file khi tạo run (chưa có máy nào) và cập nhật mỗi lần bắt đầu chạy hoặc chạy tiếp,
 * trên đúng máy chạy: với run Kaggle là máy của notebook, không phải máy đã tạo run. Trường nào
 * không xác định được thì bỏ trống (CONTRACTS mục 6.4).
 *
 * Attributes:
 *     code_commit: Commit của code, hoặc ``unknown``.
 *     python: Phiên bản Python.
 *     pytorch: Phiên bản PyTorch.
 *     cuda: Phiên bản CUDA mà PyTorch dùng.
 *     gsplat: Phiên bản gsplat.
 *     gpu: Tên GPU của máy chạy gần nhất.
 *     seed: Seed ngẫu nhiên.
 *     executor: Nơi chạy, ``local``, ``kaggle`` hoặc ``colab``.
 *     machines: Các máy đã chạy run, theo thứ tự; run chạy tiếp trên máy khác (ví dụ phiên
 *         Kaggle mới được cấp GPU khác) có nhiều máy (thêm từ ``d2s-env/0.2``).
 */
export interface EnvRecord {
  schema_version: SchemaVersion;
  code_commit: CodeCommit;
  python: Python;
  pytorch?: Pytorch;
  cuda?: Cuda;
  gsplat?: Gsplat;
  gpu?: Gpu;
  seed?: Seed;
  executor: Executor;
  machines?: Machines;
}
/**
 * Cấu hình phần cứng của một máy đã chạy run (CONTRACTS mục 6.4).
 *
 * Thời gian, VRAM và FPS chỉ so được giữa các run chạy trên cùng loại máy, nên mỗi run ghi lại
 * máy đã chạy nó. Trường nào không đọc được thì để trống.
 *
 * Attributes:
 *     cpu: Tên CPU, ví dụ ``Intel(R) Xeon(R) CPU @ 2.00GHz``.
 *     cpu_cores: Số luồng CPU.
 *     ram_gb: Dung lượng RAM, GB.
 *     gpu: Tên GPU, ví dụ ``Tesla T4``; ``None`` nếu không có GPU NVIDIA.
 *     gpu_count: Số GPU.
 *     gpu_memory_mb: VRAM của mỗi GPU, megabyte.
 *     gpu_driver: Phiên bản driver NVIDIA.
 *     os: Hệ điều hành.
 *     first_seen_utc: Lần đầu máy này chạy run.
 */
export interface Machine {
  cpu?: Cpu;
  cpu_cores?: CpuCores;
  ram_gb?: RamGb;
  gpu?: Gpu1;
  gpu_count?: GpuCount;
  gpu_memory_mb?: GpuMemoryMb;
  gpu_driver?: GpuDriver;
  os?: Os;
  first_seen_utc?: FirstSeenUtc;
}
