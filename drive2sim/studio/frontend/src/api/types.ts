// Kiểu của các phản hồi API không phải là một file trong CONTRACTS. Kiểu của file (manifest, run,
// split...) sinh tự động trong ./generated bằng `npm run types`.
import type { Manifest } from "./generated/manifest";
import type { RunRecord } from "./generated/run";
import type { Summary } from "./generated/summary";
import type { EnvRecord } from "./generated/env";
import type { ValidationReport } from "./generated/report";
import type { Approach as ApproachFile } from "./generated/approach";
export type { Manifest } from "./generated/manifest";
export type { RunRecord, StageRecord } from "./generated/run";
export type { Split } from "./generated/split";
export type { Summary } from "./generated/summary";
export type { EnvRecord, Machine } from "./generated/env";

// Loại camera một cấu hình nhận: track (pinhole, fisheye) và mô hình camera (resolve.camera_support).
export interface CameraFit {
  tracks: string[];
  camera_models: string[];
}

// File approach như API trả về: nội dung file cộng loại camera của approach, của cấu hình mặc định
// và của từng preset.
export type Approach = ApproachFile & {
  camera_support: CameraFit & { default: CameraFit; presets: Record<string, CameraFit> };
};

export interface List<T> {
  items: T[];
  total: number;
}

export interface SceneDetail {
  manifest: Manifest;
  validation: ValidationReport | null;
  splits: string[];
  /** Camera của từng split (``{split_id: [camera_id]}``), để hiện loại ống kính và xét approach. */
  split_cameras?: Record<string, string[]>;
  /** Notebook ingest trên Kaggle khi cảnh được nạp ở đó (máy không có ảnh); ``null`` nếu cảnh ở máy. */
  kaggle: { username: string; kernel: string; source: string }[] | null;
  /** Cảnh ingest bằng converter cũ: phiên bản đầu ra trong manifest và hiện tại; ``null`` nếu cảnh mới. */
  converter_outdated?: { output_version: number; current_output_version: number } | null;
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
  /** Trạng thái phía Kaggle khi run chưa tải về (runStatus.ts); ``null`` với run khác. */
  remote?: { pushed: boolean; status: string | null; checked_utc: string | null } | null;
  /** Tham số stage train khác mặc định (``report.train_params``), ví dụ ``{"refine_interval": 50}``. */
  train_params?: Record<string, unknown>;
  summary: Summary | null;
  env: EnvRecord | null;
  /** Lịch densify hiệu lực từ ``config.yaml`` của stage train (engine/densify_info.py); ``null`` khi train chưa bắt đầu. */
  densify?: DensifyInfo | null;
  run_dir: string;
}

export type DensifyKey =
  | "warmup_steps"
  | "refine_interval"
  | "densify_grad_thresh"
  | "densify_size_thresh"
  | "cull_alpha_thresh"
  | "reset_alpha_interval"
  | "stop_split_at";

export interface DensifyInfo {
  rule: "default" | "fastgs" | "mcmc";
  /** ``render_mode`` của DriveStudio: ``default`` (3DGS), ``ut`` (3DGUT), ``geer`` (3DGEER). */
  renderer: string;
  /** Đại lượng so với ngưỡng: gradient 2D (có/không absgrad), 3D (3DGUT gốc), screen, geer. */
  statistic: "2d_abs" | "2d" | "3d" | "screen" | "geer";
  nodes: Record<string, Partial<Record<DensifyKey, number>>>;
  fastgs?: Record<string, unknown>;
  mcmc?: {
    refine_start_iter: number;
    refine_stop_iter: number;
    refine_every: number;
    grow_rate: number;
    min_opacity: number;
    noise_lr: number;
    opacity_reg: number;
    scale_reg: number;
    classes: string[];
    caps: Record<string, number>;
  };
}

export interface RunLog {
  engine: string;
  stages: Record<string, string>;
}

export interface Video {
  name: string;
  stage_id: string;
  /** Video gốc qua /files/ (để tải về). */
  url: string;
  /** Bản trình duyệt phát được (H.264, rộng tối đa 1920 px); server cũ không có trường này. */
  preview_url?: string;
  /** `false` khi video còn đang được ghi. */
  ready?: boolean;
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
  /** Cỡ ảnh lúc train và đánh giá, ví dụ "800x450". */
  train_resolution: string | null;
  /** Cỡ ảnh gốc của camera trong split (manifest), ví dụ "1600x900". */
  source_resolution: string | null;
  /** Số ảnh train, số ảnh test theo split; N của every_nth_frame:N. */
  train_images: number | null;
  test_images: number | null;
  test_every: number | null;
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
  speedups: string[];
  stage_times: Record<string, { seconds: number; reused: boolean }>;
  train_params: Record<string, unknown>;
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

/** Việc nền của Studio: ingest, đẩy và tải Kaggle, Colab, Drive (GET /api/jobs). */
export interface Job {
  job_id: string;
  kind: "ingest" | "kaggle_push" | "kaggle_pull" | "colab_push" | "colab_pull" | "colab_update" | "colab_session" | "drive_pull" | "drive_push";
  label: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  result: string | null;
  error: string | null;
  /** Việc biết tiến độ (drive_pull, theo byte); first là phần đã có lúc bắt đầu. */
  progress?: { done: number; total: number; detail: string; at: number; first?: number } | null;
  created_at: number;
  started_at?: number | null;
  finished_at: number | null;
  /** Thư mục uploads/<tên>/ việc ghi vào (drive_pull). */
  target?: string | null;
  /** Huỷ và tải tiếp được (POST /api/jobs/{id}/cancel, /retry). */
  cancellable?: boolean;
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
  errors: string[];
  missing_sweeps: string[];
  candidates: { path: string; dataset: string; relation: "child" | "parent" }[];
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
    /** Phiên bản gói, ví dụ ``d2s-preview/0.2``; gói bản cũ thiếu trời, người, chỉnh màu. */
    schema_version: string;
    sh_degree: number;
    rigid_actors: string[];
    box_only_actors: string[];
    omitted: string[];
    splat_format: "ply" | "spz";
    up_axis: string | null;
    antialiased: boolean;
    humans: string[];
    sky: string | null;
    background_url: string | null;
    actor_urls: Record<string, string | null>;
    poses_url: string | null;
    human_urls: Record<string, { splats: string | null; skin: string | null }>;
    human_poses_url: string | null;
    human_bones_url: string | null;
    sky_url: string | null;
    color_affine_url: string | null;
  } | null;
  ply: string | null;
  /** cameras.json: pose và thông số camera gốc theo khung, cùng hệ toạ độ với các file trên. */
  cameras: string | null;
  /** Run có checkpoint để render GPU không (``false`` với run Instant NuRec). */
  gpu?: boolean;
}

export interface ActorPoses {
  track_ids: string[];
  T_object_world: (number[] | null)[][];
}

/** Job Colab của một run (GET /api/colab/{run_id}; CONTRACTS mục 9.3). */
export interface ColabStatus {
  session: string;
  gpu: string;
  attempt: number;
  job: string;
  commit: string;
  branch: string;
  pushed_utc: string;
  pulled_utc: string | null;
  lost_utc: string | null;
  stopped_utc: string | null;
  /** running, succeeded, failed, died, lost; null khi không hỏi được Colab (lý do ở status_error). */
  status: string | null;
  exit_code: number | null;
  log_tail: string[];
  status_error: string | null;
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
