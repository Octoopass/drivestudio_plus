// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type CodeCommit = string;
export type Python = string;
export type Pytorch = string | null;
export type Cuda = string | null;
export type Gsplat = string | null;
export type Gpu = string | null;
export type Seed = number | null;
export type Executor = "local" | "kaggle";

/**
 * Schema của file ``d2s/env.json``: môi trường đã chạy run, để chạy lại được (SPEC mục 11).
 *
 * Trường nào không xác định được thì bỏ trống (CONTRACTS mục 6.4).
 *
 * Attributes:
 *     code_commit: Commit của code, hoặc ``unknown``.
 *     python: Phiên bản Python.
 *     pytorch: Phiên bản PyTorch.
 *     cuda: Phiên bản CUDA mà PyTorch dùng.
 *     gsplat: Phiên bản gsplat.
 *     gpu: Tên GPU.
 *     seed: Seed ngẫu nhiên.
 *     executor: Nơi chạy, ``local`` hoặc ``kaggle``.
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
}
