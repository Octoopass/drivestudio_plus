// Trạng thái trực tiếp dùng chung cho mọi màn hình: một kết nối SSE tới /api/events (sự kiện của
// mọi run) và đọc định kỳ /api/jobs (việc nền: ingest, Kaggle). Thanh trên hiện số việc đang chạy;
// các trang đọc `tick` để tải lại danh sách khi một run đổi trạng thái, thay vì tự hỏi lại server.
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "./api/client";
import type { Job, List, RunEvent } from "./api/types";
import { useI18n } from "./i18n";
import { useToast } from "./components/ui";

export interface RunProgress {
  stage_id: string | null;
  fraction: number | null;
  message: string;
}

interface Live {
  /** Tăng mỗi khi một run đổi trạng thái hoặc một việc nền xong. */
  tick: number;
  /** Trạng thái mới nhất của run theo run_id (chỉ các run có sự kiện từ lúc mở Studio). */
  statuses: Record<string, string>;
  progress: Record<string, RunProgress>;
  jobs: Job[];
  refreshJobs: () => void;
}

const LiveContext = createContext<Live>({ tick: 0, statuses: {}, progress: {}, jobs: [], refreshJobs: () => undefined });

export function LiveProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  const toast = useToast();
  const [tick, setTick] = useState(0);
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState<Record<string, RunProgress>>({});
  const [jobs, setJobs] = useState<Job[]>([]);
  const known = useRef<Record<string, string>>({});
  const openedAt = useRef(Date.now() / 1000);
  const [jobPoll, setJobPoll] = useState(0);

  useEffect(() => {
    const source = new EventSource("/api/events");
    const read = (raw: MessageEvent) => ({ ...JSON.parse(raw.data), event: raw.type }) as RunEvent;
    source.addEventListener("status", (raw) => {
      const event = read(raw as MessageEvent);
      if (event.stage_id == null && event.status) setStatuses((s) => ({ ...s, [event.run_id]: event.status! }));
      setTick((n) => n + 1);
    });
    source.addEventListener("progress", (raw) => {
      const event = read(raw as MessageEvent);
      setProgress((p) => ({
        ...p,
        [event.run_id]: { stage_id: event.stage_id ?? null, fraction: event.fraction ?? null, message: event.message ?? "" },
      }));
    });
    source.addEventListener("done", (raw) => {
      const event = read(raw as MessageEvent);
      setStatuses((s) => ({ ...s, [event.run_id]: event.status ?? "" }));
      setProgress((p) => {
        const next = { ...p };
        delete next[event.run_id];
        return next;
      });
      setTick((n) => n + 1);
      if (event.status === "succeeded") toast({ kind: "ok", title: t("live.run.succeeded"), detail: event.run_id });
      if (event.status === "failed") toast({ kind: "bad", title: t("live.run.failed"), detail: event.run_id });
    });
    return () => source.close();
  }, []); // một kết nối cho cả phiên

  // Việc nền: hỏi nhanh khi đang có việc chạy, thưa khi không có.
  const busy = jobs.some((job) => job.status === "queued" || job.status === "running");
  useEffect(() => {
    let alive = true;
    api<List<Job>>("/api/jobs")
      .then((data) => {
        if (!alive) return;
        for (const job of data.items) {
          const before = known.current[job.job_id];
          // Báo khi việc vừa xong: đã thấy nó trước đó với trạng thái khác, hoặc nó được tạo sau
          // khi mở Studio và xong trước lần hỏi đầu tiên.
          const fresh = before === undefined && job.created_at > openedAt.current;
          if ((fresh || (before && before !== job.status)) && (job.status === "succeeded" || job.status === "failed")) {
            toast({
              kind: job.status === "succeeded" ? "ok" : "bad",
              title: t(`live.job.${job.status}`, { label: job.label }),
              detail: job.error ?? job.result ?? undefined,
            });
            setTick((n) => n + 1);
          }
          known.current[job.job_id] = job.status;
        }
        setJobs(data.items);
      })
      .catch(() => undefined);
    const timer = window.setTimeout(() => setJobPoll((n) => n + 1), busy ? 1500 : 8000);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [jobPoll]);

  const refreshJobs = () => setJobPoll((n) => n + 1);
  return <LiveContext.Provider value={{ tick, statuses, progress, jobs, refreshJobs }}>{children}</LiveContext.Provider>;
}

export function useLive(): Live {
  return useContext(LiveContext);
}
