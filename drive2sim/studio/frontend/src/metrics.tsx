// Các nhóm metric của một run, dùng chung cho trang run (ô số liệu) và trang so sánh (bảng): hai
// trang luôn cùng nhóm, cùng thứ tự, cùng tên và cùng cách định dạng. Mỗi metric đọc từ một dòng
// của /api/report (ReportRow).
import type { ReportRow } from "./api/types";
import { count, duration, fixed } from "./components/ui";
import { useI18n } from "./i18n";

export type MetricKey = Exclude<keyof ReportRow, "options" | "stage_times">;

export interface MetricDef {
  key: MetricKey;
  /** Khoá i18n của tên metric. */
  label: string;
  /** Khoá i18n của giải thích (icon i), nếu có. */
  hint?: string;
  unit?: string;
  /** Hướng tốt hơn, để tô màu và sắp xếp; không có thì metric chỉ để đọc. */
  better?: "high" | "low";
  format: (row: ReportRow) => string;
  /** Phụ thuộc phần cứng (thời gian, VRAM, FPS): chỉ so được giữa các run cùng GPU. */
  hardware?: boolean;
  /** Đo trên ảnh test của split: chỉ so được giữa các run cùng split. */
  evalSplit?: boolean;
  /** Giá trị số để tô màu, sắp xếp và vẽ biểu đồ. */
  value: (row: ReportRow) => number | null;
}

export interface MetricGroup {
  id: "summary" | "quality" | "region" | "time" | "resources" | "size";
  /** Khoá i18n của tên nhóm và giải thích nhóm. */
  label: string;
  hint: string;
  metrics: MetricDef[];
  /** Cột sắp xếp mặc định khi mở nhóm này trên trang so sánh. */
  defaultSort: MetricKey;
}

const num = (key: MetricKey) => (row: ReportRow) => {
  const value = row[key];
  return typeof value === "number" ? value : null;
};
const digits = (key: MetricKey, n: number) => (row: ReportRow) => fixed(num(key)(row), n);

export const METRIC_GROUPS: MetricGroup[] = [
  {
    id: "quality",
    defaultSort: "avg_error",
    label: "metrics.group.quality",
    hint: "metrics.group.quality_hint",
    metrics: [
      { key: "psnr", evalSplit: true, label: "metrics.psnr", hint: "metrics.hint.psnr", unit: "dB", better: "high", format: digits("psnr", 2), value: num("psnr") },
      { key: "ssim", evalSplit: true, label: "metrics.ssim", hint: "metrics.hint.ssim", better: "high", format: digits("ssim", 3), value: num("ssim") },
      { key: "lpips", evalSplit: true, label: "metrics.lpips", hint: "metrics.hint.lpips", better: "low", format: digits("lpips", 3), value: num("lpips") },
      { key: "avg_error", evalSplit: true, label: "metrics.avg", hint: "metrics.hint.avg", better: "low", format: digits("avg_error", 4), value: num("avg_error") },
    ],
  },
  {
    id: "region",
    defaultSort: "dynamic_psnr",
    label: "metrics.group.region",
    hint: "metrics.group.region_hint",
    metrics: [
      { key: "human_psnr", evalSplit: true, label: "run.result.human", hint: "metrics.hint.human", unit: "dB", better: "high", format: digits("human_psnr", 2), value: num("human_psnr") },
      { key: "vehicle_psnr", evalSplit: true, label: "run.result.vehicle", hint: "metrics.hint.vehicle", unit: "dB", better: "high", format: digits("vehicle_psnr", 2), value: num("vehicle_psnr") },
      { key: "dynamic_psnr", evalSplit: true, label: "run.result.dynamic", hint: "metrics.hint.dynamic", unit: "dB", better: "high", format: digits("dynamic_psnr", 2), value: num("dynamic_psnr") },
      { key: "occupied_psnr", evalSplit: true, label: "metrics.occupied", hint: "metrics.hint.occupied", unit: "dB", better: "high", format: digits("occupied_psnr", 2), value: num("occupied_psnr") },
    ],
  },
  {
    id: "time",
    defaultSort: "train_time_s",
    label: "metrics.group.time",
    hint: "metrics.group.time_hint",
    metrics: [
      { key: "train_time_s", hardware: true, label: "run.result.train_time", hint: "metrics.hint.train_time", better: "low", format: (r) => duration(r.train_time_s), value: num("train_time_s") },
      { key: "recon_time_s", hardware: true, label: "run.result.recon_time", hint: "metrics.hint.recon_time", better: "low", format: (r) => duration(r.recon_time_s), value: num("recon_time_s") },
      { key: "num_iters", label: "run.result.iters", hint: "metrics.hint.iters", format: (r) => count(r.num_iters), value: num("num_iters") },
    ],
  },
  {
    id: "resources",
    defaultSort: "render_fps",
    label: "metrics.group.resources",
    hint: "metrics.group.resources_hint",
    metrics: [
      { key: "peak_vram_mb", hardware: true, label: "run.result.vram", hint: "metrics.hint.vram", unit: "GB", better: "low", format: (r) => (r.peak_vram_mb ? (r.peak_vram_mb / 1024).toFixed(1) : "–"), value: num("peak_vram_mb") },
      { key: "render_fps", hardware: true, label: "run.result.fps", hint: "metrics.hint.fps", unit: "fps", better: "high", format: digits("render_fps", 1), value: num("render_fps") },
      { key: "gpu", label: "metrics.gpu", hint: "metrics.hint.gpu", format: (r) => r.gpu ?? "–", value: () => null },
    ],
  },
  {
    id: "size",
    defaultSort: "num_gaussians",
    label: "metrics.group.size",
    hint: "metrics.group.size_hint",
    metrics: [
      { key: "num_gaussians", label: "run.result.gaussians", hint: "metrics.hint.gaussians", format: (r) => count(r.num_gaussians), value: num("num_gaussians") },
      { key: "ply_gaussians", label: "run.result.ply_gaussians", hint: "metrics.hint.ply_gaussians", format: (r) => count(r.ply_gaussians), value: num("ply_gaussians") },
      { key: "ply_size_mb", label: "run.result.ply_size", hint: "metrics.hint.ply_size", unit: "MB", format: digits("ply_size_mb", 1), value: num("ply_size_mb") },
      { key: "preview_gaussians", label: "metrics.preview_gaussians", hint: "metrics.hint.preview_gaussians", format: (r) => count(r.preview_gaussians), value: num("preview_gaussians") },
      { key: "preview_size_mb", label: "metrics.preview_size", hint: "metrics.hint.preview_size", unit: "MB", format: digits("preview_size_mb", 1), value: num("preview_size_mb") },
    ],
  },
];

/** Metric chính của mỗi nhóm, cho bảng tổng hợp đứng đầu trang so sánh. */
export const SUMMARY_GROUP: MetricGroup = {
  id: "summary",
  defaultSort: "avg_error",
  label: "metrics.group.summary",
  hint: "metrics.group.summary_hint",
  metrics: (["avg_error", "psnr", "ssim", "lpips", "train_time_s", "recon_time_s", "peak_vram_mb", "render_fps", "num_gaussians", "gpu"] as MetricKey[]).map(
    (key) => METRIC_GROUPS.flatMap((g) => g.metrics).find((m) => m.key === key)!,
  ),
};

/** Run đưa vào bảng xếp hạng và biểu đồ: đã xong, train đầy đủ (không phải chạy thử nhanh). */
export function isComparable(row: ReportRow): boolean {
  return row.status === "succeeded" && !row.quick;
}

/** Nhóm có ít nhất một giá trị trong các dòng đang xem; nhóm trống không hiện. */
export function visibleGroups(rows: ReportRow[]): MetricGroup[] {
  return METRIC_GROUPS.filter((group) => group.metrics.some((m) => rows.some((row) => m.format(row) !== "–")));
}

/** Mũi tên theo quy ước của paper: ↑ là cao là tốt, ↓ là thấp là tốt; không có hướng thì không hiện. */
export function BetterArrow({ better }: { better?: "high" | "low" }) {
  const { t } = useI18n();
  if (!better) return null;
  const text = t(better === "high" ? "metrics.better.high" : "metrics.better.low");
  return (
    <span className="better-arrow" title={text} aria-label={text}>
      {better === "high" ? "↑" : "↓"}
    </span>
  );
}
