// Kiểu của các phản hồi API không phải là một file trong CONTRACTS. Kiểu của file (manifest, run,
// split...) sinh tự động trong ./generated bằng `npm run types`.
import type { Manifest } from "./generated/manifest";
import type { RunRecord } from "./generated/run";
import type { Summary } from "./generated/summary";
import type { ValidationReport } from "./generated/report";

export type { Approach } from "./generated/approach";
export type { Manifest } from "./generated/manifest";
export type { RunRecord, StageRecord } from "./generated/run";
export type { Split } from "./generated/split";
export type { Summary } from "./generated/summary";

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
  options: Record<string, OptionValue>;
  lidar_usage: string[];
  stages: { stage_id: string; params: Record<string, unknown>; cache_hit: boolean }[];
  blocked_choices: Record<string, { choice: OptionValue; message: string }[]>;
}

export interface RunDetail extends RunRecord {
  summary: Summary | null;
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
  status: string;
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
  human_psnr: number | null;
  vehicle_psnr: number | null;
  train_time_s: number | null;
  peak_vram_mb: number | null;
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
