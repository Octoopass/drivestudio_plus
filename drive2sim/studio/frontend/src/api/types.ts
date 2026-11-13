// Kiểu của các phản hồi API không phải là một file trong CONTRACTS. Kiểu của file (manifest, run,
// split...) sinh tự động trong ./generated bằng `npm run types`.
import type { Manifest } from "./generated/manifest";
import type { RunRecord } from "./generated/run";
import type { Summary } from "./generated/summary";
import type { EnvRecord } from "./generated/env";
import type { ValidationReport } from "./generated/report";

export type { Approach } from "./generated/approach";
export type { Manifest } from "./generated/manifest";
export type { RunRecord, StageRecord } from "./generated/run";
export type { Split } from "./generated/split";
export type { Summary } from "./generated/summary";
export type { EnvRecord, Machine } from "./generated/env";

export interface List<T> {
  items: T[];
  total: number;
}

export interface SceneDetail {
  manifest: Manifest;
  validation: ValidationReport | null;
  splits: string[];
}

export interface FrameImages {
  frame_index: number;
  images: { camera_id: string; url: string | null }[];
}

export type OptionValue = boolean | number | string;

export interface CheckResult {
  approach_id: string;
  preset: string | null;
  quick: boolean;
  options: Record<string, OptionValue>;
  lidar_usage: string[];
  stages: { stage_id: string; params: Record<string, unknown>; cache_hit: boolean }[];
  blocked_choices: Record<string, { choice: OptionValue; message: string }[]>;
}

export interface RunDetail extends RunRecord {
  summary: Summary | null;
  env: EnvRecord | null;
  run_dir: string;
}

export interface RunLog {
  engine: string;
  stages: Record<string, string>;
}

export interface Video {
  name: string;
  stage_id: string;
  url: string;
}

export interface ReportRow {
  run_id: string;
  scene_id: string;
  split_id: string;
  approach_id: string;
  options: Record<string, OptionValue>;
  preset: string | null;
  quick: boolean;
  status: string;
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
  avg_error: number | null;
  human_psnr: number | null;
  vehicle_psnr: number | null;
  dynamic_psnr: number | null;
  occupied_psnr: number | null;
  num_iters: number | null;
  train_time_s: number | null;
  recon_time_s: number | null;
  peak_vram_mb: number | null;
  render_fps: number | null;
  num_gaussians: number | null;
  ply_gaussians: number | null;
  ply_size_mb: number | null;
  preview_gaussians: number | null;
  preview_size_mb: number | null;
  gpu: string | null;
  software: string | null;
  stage_times: Record<string, { seconds: number; reused: boolean }>;
}

/** Sự kiện SSE của engine (CONTRACTS mục 11.4). */
export interface RunEvent {
  event: "status" | "progress" | "metric" | "log" | "done";
  run_id: string;
  stage_id?: string | null;
  status?: string;
  fraction?: number;
  message?: string;
  name?: string;
  value?: number;
  step?: number;
  level?: string;
}

/** Việc nền của Studio: ingest, đẩy và tải Kaggle (GET /api/jobs). */
export interface Job {
  job_id: string;
  kind: "ingest" | "kaggle_push" | "kaggle_pull";
  label: string;
  status: "queued" | "running" | "succeeded" | "failed";
  result: string | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
}

/** Kết quả nhận dạng thư mục dữ liệu gốc (POST /api/sources/inspect). */
export interface SourceReport {
  path: string;
  dataset: string | null;
  versions: Record<string, string[]>;
  cameras: string[];
  has_lidar: boolean | null;
  has_cuboids: boolean | null;
  ingest_supported: boolean;
  inventory: Record<string, unknown>[];
  notes: string[];
}

export interface CacheEntry {
  cache_key: string;
  scene_id: string;
  stage_id: string;
  size_bytes: number;
  partial: boolean;
  stale: boolean;
  used_by: string[];
}

export interface CacheList extends List<CacheEntry> {
  size_bytes: number;
}

export interface Splats {
  preview: {
    sh_degree: number;
    rigid_actors: string[];
    box_only_actors: string[];
    omitted: string[];
    background_url: string | null;
    actor_urls: Record<string, string | null>;
    poses_url: string | null;
  } | null;
  ply: string | null;
  /** cameras.json: pose và thông số camera gốc theo khung, cùng hệ toạ độ với các file trên. */
  cameras: string | null;
}

export interface ActorPoses {
  track_ids: string[];
  T_object_world: (number[] | null)[][];
}

export interface KaggleStatus {
  username: string;
  kernel: string;
  attempt: number;
  dataset: string;
  commit: string;
  pushed_utc: string;
  pulled_utc: string | null;
  /** Trạng thái notebook; null khi không hỏi được Kaggle (lý do ở status_error). */
  status: string | null;
  status_error: string | null;
}
