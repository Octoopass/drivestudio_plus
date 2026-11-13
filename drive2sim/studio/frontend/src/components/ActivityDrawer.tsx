// Ngăn kéo "Hoạt động": các run đang chạy hoặc đang chờ (kèm tiến độ trực tiếp, nút huỷ) và các việc
// nền từ lúc mở Studio (ingest, Kaggle).
import { Square } from "lucide-react";
import { asApiError, post, seg, useApi } from "../api/client";
import type { List, RunRecord } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { href } from "../router";
import { ago, Drawer, Empty, Status, useToast } from "./ui";

export function ActivityDrawer({ onClose }: { onClose: () => void }) {
  const { t, locale } = useI18n();
  const live = useLive();
  const toast = useToast();
  const running = useApi<List<RunRecord>>(`/api/runs?status=running&_=${live.tick}`);
  const pending = useApi<List<RunRecord>>(`/api/runs?status=pending&_=${live.tick}`);
  const runs = [...(running.data?.items ?? []), ...(pending.data?.items ?? []).filter((r) => r.executor === "local")];

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
              {(job.status === "queued" || job.status === "running") && (
                <div className="progress indeterminate" style={{ marginTop: 10 }}>
                  <i />
                </div>
              )}
              {job.error && <div className="small" style={{ color: "var(--stop)", marginTop: 8 }}>{job.error}</div>}
              {job.status === "succeeded" && job.result && (
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
