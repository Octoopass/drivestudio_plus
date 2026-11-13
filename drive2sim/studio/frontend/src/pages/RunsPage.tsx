// Mọi run, mới nhất trước: lọc theo trạng thái, cảnh và chữ; dòng đang chạy có thanh tiến độ trực
// tiếp; rê chuột vào dòng để huỷ, chạy tiếp hoặc xoá.
import { Cloud, Play, RotateCcw, Search, Square, Trash2, Zap } from "lucide-react";
import { useMemo, useState } from "react";
import { asApiError, del, post, seg, useApi } from "../api/client";
import type { List, Manifest, RunRecord } from "../api/types";
import { ago, Confirm, Empty, ErrorBox, Loading, optionsText, PageHead, Segmented, Status, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { go, href } from "../router";

const STATUSES = ["all", "running", "pending", "succeeded", "failed", "cancelled"] as const;
type Filter = (typeof STATUSES)[number];

export function RunsPage({ sceneId = "", status = "all" }: { sceneId?: string; status?: string }) {
  const { t, locale } = useI18n();
  const live = useLive();
  const toast = useToast();
  const runs = useApi<List<RunRecord>>(`/api/runs?_=${live.tick}`);
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const [query, setQuery] = useState("");
  const [deleting, setDeleting] = useState<RunRecord | null>(null);
  const filter = (STATUSES as readonly string[]).includes(status) ? (status as Filter) : "all";

  const all = useMemo(
    () => (runs.data?.items ?? []).map((r) => ({ ...r, status: (live.statuses[r.run_id] ?? r.status) as RunRecord["status"] })),
    [runs.data, live.statuses],
  );
  const counts = Object.fromEntries(STATUSES.map((s) => [s, s === "all" ? all.length : all.filter((r) => r.status === s).length]));
  const rows = all.filter(
    (r) =>
      (filter === "all" || r.status === filter) &&
      (!sceneId || r.scene_id === sceneId) &&
      `${r.run_id} ${r.approach.approach_id} ${optionsText(r.approach.options ?? {})}`.toLowerCase().includes(query.toLowerCase()),
  );

  const act = async (label: string, path: string) => {
    try {
      await post(path);
      toast({ kind: "ok", title: label });
      runs.reload();
    } catch (err) {
      toast({ kind: "bad", title: label, detail: asApiError(err).message });
    }
  };

  return (
    <div className="page">
      <PageHead
        title={t("runs.page.title")}
        info={t("runs.page.lede")}
        actions={
          <a className="btn primary" href={href({ page: "new" })}>
            <Play size={19} />
            {t("nav.menu.new_run")}
          </a>
        }
      />
      <div className="row wrap" style={{ gap: 12 }}>
        <Segmented<Filter>
          label={t("runs.filter.status")}
          value={filter}
          onChange={(value) => go({ page: "runs", sceneId: sceneId || undefined, status: value === "all" ? undefined : value })}
          options={STATUSES.map((s) => ({ value: s, label: `${s === "all" ? t("runs.filter.all") : t(`common.status.${s}`)} ${counts[s]}` }))}
        />
        <select value={sceneId} onChange={(e) => go({ page: "runs", sceneId: e.target.value || undefined, status: filter === "all" ? undefined : filter })} aria-label={t("runs.table.scene")}>
          <option value="">{t("runs.filter.all_scenes")}</option>
          {(scenes.data?.items ?? []).map((m) => (
            <option key={m.scene_id}>{m.scene_id}</option>
          ))}
        </select>
        <label className="row grow" style={{ position: "relative", minWidth: 252, maxWidth: 452 }}>
          <Search size={18} style={{ position: "absolute", left: 11 }} className="faint" />
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("runs.filter.search")} style={{ paddingLeft: 35, width: "100%" }} />
        </label>
      </div>
      <ErrorBox error={runs.error} />
      <section className="panel">
        {!runs.data && !runs.error && <div className="panel-body"><Loading rows={6} /></div>}
        {runs.data && all.length === 0 && (
          <Empty title={t("runs.empty.title")} action={<a className="btn primary" href={href({ page: "new" })}>{t("nav.menu.new_run")}</a>}>
            <p>{t("runs.empty.text")}</p>
          </Empty>
        )}
        {all.length > 0 && rows.length === 0 && <Empty title={t("runs.empty.filtered")} />}
        {rows.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("runs.table.run")}</th>
                  <th>{t("runs.table.scene")}</th>
                  <th>{t("runs.table.split")}</th>
                  <th>{t("runs.table.options")}</th>
                  <th style={{ minWidth: 188 }}>{t("runs.table.status")}</th>
                  <th className="num">{t("runs.table.created")}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((run) => {
                  const progress = live.progress[run.run_id];
                  const done = (run.stages ?? []).filter((s) => s.status === "succeeded").length;
                  return (
                    <tr key={run.run_id} className="link-row" onClick={() => go({ page: "run", runId: run.run_id })}>
                      <td>
                        <a className="id" href={href({ page: "run", runId: run.run_id })} onClick={(e) => e.stopPropagation()}>
                          {run.run_id}
                        </a>
                        <div className="small faint row" style={{ gap: 6 }}>
                          {run.approach.approach_id}
                          {run.approach.quick && <Zap size={15} aria-label={t("run.quick")} className="warn-ink" />}
                          {run.executor === "kaggle" && <Cloud size={15} aria-label="Kaggle" />}
                        </div>
                      </td>
                      <td className="ellipsis" style={{ maxWidth: 276 }}>{run.scene_id}</td>
                      <td>{run.split_id}</td>
                      <td className="small muted" style={{ maxWidth: 327 }}>{optionsText(run.approach.options ?? {}) || "–"}</td>
                      <td>
                        <Status value={run.status} />
                        {run.status === "running" && (
                          <div className="progress" style={{ marginTop: 7 }} title={progress?.message}>
                            <i style={{ width: `${Math.round(((done + (progress?.fraction ?? 0)) / Math.max(1, (run.stages ?? []).length)) * 100)}%` }} />
                          </div>
                        )}
                      </td>
                      <td className="num muted nowrap">{ago(run.created_utc ?? null, locale)}</td>
                      <td className="actions" onClick={(e) => e.stopPropagation()}>
                        <span className="row-actions row" style={{ gap: 2, justifyContent: "flex-end" }}>
                          {(run.status === "running" || run.status === "pending") && (
                            <button className="btn sm ghost icon" title={t("run.action.cancel")} aria-label={t("run.action.cancel")} onClick={() => act(t("run.toast.cancel_requested"), `/api/runs/${seg(run.run_id)}/cancel`)}>
                              <Square size={16} />
                            </button>
                          )}
                          {(run.status === "failed" || run.status === "cancelled") && run.executor === "local" && (
                            <button className="btn sm ghost icon" title={t("run.action.resume")} aria-label={t("run.action.resume")} onClick={() => act(t("run.toast.resumed"), `/api/runs/${seg(run.run_id)}/resume`)}>
                              <RotateCcw size={16} />
                            </button>
                          )}
                          {run.status !== "running" && (
                            <button className="btn sm ghost icon" title={t("common.action.delete")} aria-label={t("common.action.delete")} onClick={() => setDeleting(run)}>
                              <Trash2 size={16} />
                            </button>
                          )}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {deleting && (
        <Confirm
          title={t("run.delete.title", { run: deleting.run_id })}
          body={t("run.delete.body")}
          confirmLabel={t("common.action.delete")}
          danger
          onClose={() => setDeleting(null)}
          onConfirm={async () => {
            await del(`/api/runs/${seg(deleting.run_id)}`);
            toast({ kind: "ok", title: t("run.delete.done"), detail: deleting.run_id });
            runs.reload();
          }}
        />
      )}
    </div>
  );
}
