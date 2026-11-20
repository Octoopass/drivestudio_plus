// Ngăn kéo "Hoạt động": các run đang chạy hoặc đang chờ trên máy này (kèm tiến độ trực tiếp, nút huỷ),
// run gửi lên Kaggle hoặc Colab mà chưa tải kết quả về, và các việc nền từ lúc mở Studio (ingest,
// Kaggle). Run Kaggle/Colab giữ trạng thái "pending" trên máy tới khi tải về, dù notebook đang chạy
// hay đã xong; nên tách riêng, không có nút huỷ (huỷ ở đây không dừng được notebook), trạng thái thật
// xem ở tab Kaggle/Colab của trang run.
import { Square } from "lucide-react";
import { asApiError, post, seg, useApi } from "../api/client";
import type { Job, List } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { href } from "../router";
import { runStatus, type RunItem } from "../runStatus";
import { ago, Drawer, Empty, Status, useToast } from "./ui";
import { clock, JobProgress } from "./JobProgress";

// việc có kết quả là cảnh hoặc run để mở; drive_pull trả thư mục, colab_session không trả gì
const LINKED = new Set<Job["kind"]>(["ingest", "kaggle_push", "kaggle_pull", "colab_push", "colab_pull"]);

export function ActivityDrawer({ onClose }: { onClose: () => void }) {
  const { t, locale } = useI18n();
  const live = useLive();
  const toast = useToast();
  const running = useApi<List<RunItem>>(`/api/runs?status=running&_=${live.tick}`);
  const pending = useApi<List<RunItem>>(`/api/runs?status=pending&_=${live.tick}`);
  const all = [...(running.data?.items ?? []), ...(pending.data?.items ?? [])];
  const runs = all.filter((run) => run.executor === "local");
  const remote = all.filter((run) => run.executor !== "local");

  const cancel = async (runId: string) => {
    try {
      await post(`/api/runs/${seg(runId)}/cancel`);
      toast({ kind: "ok", title: t("run.toast.cancel_requested"), detail: runId });
    } catch (err) {
      toast({ kind: "bad", title: t("run.toast.cancel_failed"), detail: asApiError(err).message });
    }
  };

  return (
    <Drawer title={t("nav.activity.title")} onClose={onClose}>
      <section className="stack" style={{ gap: 10 }}>
        <h3>{t("activity.runs.title")}</h3>
        {runs.length === 0 && <p className="muted">{t("activity.runs.none")}</p>}
        {runs.map((run) => {
          const progress = live.progress[run.run_id];
          return (
            <div key={run.run_id} className="panel" style={{ padding: "12px 15px" }}>
              <div className="row">
                <a href={href({ page: "run", runId: run.run_id })} onClick={onClose} className="id grow ellipsis">
                  {run.run_id}
                </a>
                <Status value={live.statuses[run.run_id] ?? run.status} />
                <button className="btn sm danger" onClick={() => cancel(run.run_id)}>
                  <Square size={15} />
                  {t("run.action.cancel")}
                </button>
              </div>
              {progress && (
                <div className="stack" style={{ gap: 6, marginTop: 10 }}>
                  <div className="small muted ellipsis">
                    {progress.stage_id} {progress.message}
                    {progress.eta != null && ` · ${t("eta.left", { time: clock(progress.eta) })}`}
                  </div>
                  <div className={`progress${progress.fraction == null ? " indeterminate" : ""}`}>
                    <i style={{ width: `${Math.round((progress.fraction ?? 0) * 100)}%` }} />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </section>
      {remote.length > 0 && (
        <section className="stack" style={{ gap: 10 }}>
          <h3>{t("activity.remote.title")}</h3>
          <p className="small muted">{t("activity.remote.hint")}</p>
          {remote.map((run) => (
            <div key={run.run_id} className="panel row" style={{ padding: "12px 15px" }}>
              <a href={href({ page: "run", runId: run.run_id })} onClick={onClose} className="id grow ellipsis">
                {run.run_id}
              </a>
              <span className="small faint">{run.executor === "kaggle" ? "Kaggle" : "Colab"}</span>
              <Status value={runStatus(run, live.statuses[run.run_id])} />
            </div>
          ))}
        </section>
      )}
      <section className="stack" style={{ gap: 10 }}>
        <h3>{t("activity.jobs.title")}</h3>
        {live.jobs.length === 0 ? (
          <Empty title={t("activity.jobs.none")} />
        ) : (
          live.jobs.map((job) => (
            <div key={job.job_id} className="panel" style={{ padding: "12px 15px" }}>
              <div className="row">
                <span className="grow ellipsis">{job.label}</span>
                <Status value={job.status} />
              </div>
              <div className="small faint">
                {t(`activity.kind.${job.kind}`)}, {ago(job.created_at, locale)}
              </div>
              <JobProgress job={job} />
              {job.error && <div className="small" style={{ color: "var(--stop)", marginTop: 8 }}>{job.error}</div>}
              {job.status === "succeeded" && job.result && LINKED.has(job.kind) && (
                <a
                  className="small"
                  onClick={onClose}
                  href={href(job.kind === "ingest" ? { page: "scene", sceneId: job.result } : { page: "run", runId: job.result })}
                >
                  {t("activity.jobs.open", { id: job.result })}
                </a>
              )}
            </div>
          ))
        )}
      </section>
    </Drawer>
  );
}
