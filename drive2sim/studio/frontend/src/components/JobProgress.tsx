// Thanh tiến độ của một việc nền (GET /api/jobs): việc báo progress (tải từ Drive) thì hiện phần
// trăm, tốc độ và thời gian còn lại tính từ started_at; việc không báo thì thanh chạy liên tục.
// JobActions: nút Huỷ (đang chờ, đang chạy) và Tải tiếp (đã huỷ, lỗi) cho việc huỷ được.
import { Play, Square } from "lucide-react";
import { useState } from "react";
import { asApiError, post, seg } from "../api/client";
import type { Job } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { useToast } from "./ui";

export function jobActive(job: Job | null | undefined): boolean {
  return job?.status === "queued" || job?.status === "running";
}

/** Huỷ hoặc tải tiếp một việc huỷ được; ``onRetried`` nhận việc mới. */
export function JobActions({ job, onRetried }: { job: Job; onRetried?: (job: Job) => void }) {
  const { t } = useI18n();
  const live = useLive();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  if (!job.cancellable) return null;
  const act = async (action: "cancel" | "retry") => {
    setBusy(true);
    try {
      const result = await post<Job>(`/api/jobs/${seg(job.job_id)}/${action}`);
      if (action === "retry") onRetried?.(result);
      live.refreshJobs();
    } catch (err) {
      toast({ kind: "bad", title: t(action === "cancel" ? "job.cancel_failed" : "job.retry_failed"), detail: asApiError(err).message });
    } finally {
      setBusy(false);
    }
  };
  if (jobActive(job)) {
    return (
      <button className="btn sm" type="button" onClick={() => void act("cancel")} disabled={busy}>
        <Square size={15} />
        {t("job.cancel")}
      </button>
    );
  }
  if (job.status === "cancelled" || job.status === "failed") {
    return (
      <button className="btn sm primary" type="button" onClick={() => void act("retry")} disabled={busy}>
        <Play size={15} />
        {t("job.retry")}
      </button>
    );
  }
  return null;
}

function bytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  return `${Math.max(0, value / 1024 ** 2).toFixed(1)} MiB`;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Tốc độ và thời gian còn lại: tính trên phần đã làm trong lần chạy này (bỏ phần có sẵn lúc đầu). */
export function jobEta(job: Job): { pct: number; rate: number | null; left: number | null } | null {
  const p = job.progress;
  if (!p || !p.total) return null;
  const pct = Math.min(100, (100 * p.done) / p.total);
  const first = p.first ?? 0;
  const seconds = job.started_at ? p.at - job.started_at : 0;
  const rate = seconds > 1 && p.done > first ? (p.done - first) / seconds : null;
  return { pct, rate, left: rate ? (p.total - p.done) / rate : null };
}

export function JobProgress({ job }: { job: Job }) {
  const { t } = useI18n();
  const active = job.status === "queued" || job.status === "running";
  const eta = jobEta(job);
  if (!active) return null;
  if (!eta || !job.progress) {
    return (
      <div className="progress indeterminate" style={{ marginTop: 8 }}>
        <i />
      </div>
    );
  }
  return (
    <div className="stack" style={{ gap: 4, marginTop: 8 }}>
      <div className="progress">
        <i style={{ width: `${eta.pct}%` }} />
      </div>
      <span className="small muted ellipsis">
        {t("job.progress", {
          done: bytes(job.progress.done),
          total: bytes(job.progress.total),
          pct: eta.pct.toFixed(0),
          rate: eta.rate ? `${(eta.rate / 1024 ** 2).toFixed(1)} MB/s` : "–",
          left: eta.left !== null ? clock(eta.left) : "–",
        })}
        {job.progress.detail ? ` · ${job.progress.detail}` : ""}
      </span>
    </div>
  );
}
