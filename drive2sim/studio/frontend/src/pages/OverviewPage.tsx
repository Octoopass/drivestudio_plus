// Tổng quan: việc đang chạy (tiến độ trực tiếp), kết quả tốt nhất của từng cảnh, run gần đây và
// các cảnh. Mục đích là mở Studio ra là biết ngay đang có gì chạy và kết quả đang dẫn đầu.
import { Play, Plus } from "lucide-react";
import { useMemo, useState } from "react";
import { useApi } from "../api/client";
import type { CacheList, List, Manifest, ReportRow, RunRecord } from "../api/types";
import { IngestDrawer } from "../components/IngestDrawer";
import { ago, bytes, Empty, fixed, Loading, PageHead, Panel, Status } from "../components/ui";
import { useI18n } from "../i18n";
import { BetterArrow, isComparable } from "../metrics";
import { useLive } from "../live";
import { go, href } from "../router";
import { SceneCard } from "./ScenesPage";

export function OverviewPage() {
  const { t, locale } = useI18n();
  const live = useLive();
  const runs = useApi<List<RunRecord>>(`/api/runs?_=${live.tick}`);
  const scenes = useApi<List<Manifest>>(`/api/scenes?_=${live.tick}`);
  const report = useApi<List<ReportRow>>(`/api/report?_=${live.tick}`);
  const cache = useApi<CacheList>("/api/cache");
  const [adding, setAdding] = useState(false);

  const all = runs.data?.items ?? [];
  // "Đang chạy" ở mọi chỗ của Studio = run đang chạy + run đang chờ (cùng cách đếm với menu).
  const active = all.filter((r) => ["running", "pending"].includes(live.statuses[r.run_id] ?? r.status));
  const counts = {
    succeeded: all.filter((r) => r.status === "succeeded").length,
    failed: all.filter((r) => r.status === "failed").length,
  };

  // Run tốt nhất của mỗi cặp cảnh và split, theo Avg (gộp PSNR, SSIM, LPIPS; thấp là tốt) như
  // cột xếp hạng mặc định của trang So sánh.
  const leaders = useMemo(() => {
    const best = new Map<string, ReportRow>();
    for (const row of report.data?.items ?? []) {
      if (row.avg_error == null || !isComparable(row)) continue;
      const key = `${row.scene_id}/${row.split_id}`;
      const current = best.get(key);
      if (!current || (current.avg_error ?? Infinity) > row.avg_error) best.set(key, row);
    }
    return [...best.values()].sort((a, b) => (a.avg_error ?? 0) - (b.avg_error ?? 0)).slice(0, 6);
  }, [report.data]);

  return (
    <div className="page">
      <PageHead
        title={t("overview.page.title")}
        info={t("overview.page.lede")}
        actions={
          <>
            <button className="btn" onClick={() => setAdding(true)}>
              <Plus size={20} />
              {t("scenes.action.add")}
            </button>
            <a className="btn primary" href={href({ page: "new" })}>
              <Play size={19} />
              {t("nav.menu.new_run")}
            </a>
          </>
        }
      />

      <section className="panel">
        <div className="stat-row">
          <a className="stat stat-link" href={href({ page: "scenes" })}>
            <div className="label">{t("overview.stat.scenes")}</div>
            <div className="value">{scenes.data?.total ?? "–"}</div>
          </a>
          <a className="stat stat-link" href={href({ page: "runs", status: "succeeded" })}>
            <div className="label">{t("overview.stat.succeeded")}</div>
            <div className="value">{counts.succeeded}</div>
          </a>
          <a className="stat stat-link" href={href({ page: "runs", status: "failed" })}>
            <div className="label">{t("overview.stat.failed")}</div>
            <div className="value" style={{ color: counts.failed ? "var(--stop)" : undefined }}>{counts.failed}</div>
          </a>
          <a className="stat stat-link" href={href({ page: "storage" })}>
            <div className="label">{t("overview.stat.cache")}</div>
            <div className="value">{cache.data ? bytes(cache.data.size_bytes) : "–"}</div>
          </a>
        </div>
      </section>

      <div className="grid-2">
        <Panel title={t("overview.active.title")} flush>
          {!runs.data && <div className="panel-body"><Loading /></div>}
          {runs.data && active.length === 0 && (
            <Empty title={t("overview.active.none")} action={<a className="btn" href={href({ page: "new" })}>{t("nav.menu.new_run")}</a>} />
          )}
          {active.map((run) => {
            const progress = live.progress[run.run_id];
            const status = live.statuses[run.run_id] ?? run.status;
            const done = (run.stages ?? []).filter((s) => s.status === "succeeded").length;
            return (
              <a key={run.run_id} href={href({ page: "run", runId: run.run_id })} style={{ display: "block", padding: "15px 20px", borderBottom: "1px solid var(--line)", textDecoration: "none" }}>
                <div className="row">
                  <span className="id grow ellipsis">{run.run_id}</span>
                  <Status value={status} />
                </div>
                <div className="small muted ellipsis" style={{ margin: "4px 0 8px" }}>
                  {progress ? `${progress.stage_id ?? ""} ${progress.message}` : t("overview.active.stages", { done, total: (run.stages ?? []).length })}
                </div>
                <div className={`progress${status === "running" && progress?.fraction == null ? " indeterminate" : ""}`}>
                  <i style={{ width: `${Math.round(((done + (progress?.fraction ?? 0)) / Math.max(1, (run.stages ?? []).length)) * 100)}%` }} />
                </div>
              </a>
            );
          })}
        </Panel>

        <Panel title={t("overview.leaders.title")} tools={<a className="btn ghost sm" href={href({ page: "compare" })}>{t("overview.leaders.all")}</a>} flush>
          {report.data && leaders.length === 0 && <Empty title={t("overview.leaders.none")} />}
          {leaders.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("runs.table.scene")}</th>
                    <th>{t("runs.table.approach")}</th>
                    <th className="num">Avg <BetterArrow better="low" /></th>
                    <th className="num">PSNR <BetterArrow better="high" /></th>
                  </tr>
                </thead>
                <tbody>
                  {leaders.map((row) => (
                    <tr key={row.run_id} className="link-row" onClick={() => go({ page: "compare", sceneId: row.scene_id, splitId: row.split_id })}>
                      <td>
                        <div className="ellipsis" style={{ maxWidth: 301 }}>{row.scene_id}</div>
                        <div className="small faint">{row.split_id}</div>
                      </td>
                      <td>{row.approach_id}</td>
                      <td className="num">{fixed(row.avg_error, 4)}</td>
                      <td className="num">{fixed(row.psnr)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      <Panel title={t("overview.recent.title")} tools={<a className="btn ghost sm" href={href({ page: "runs" })}>{t("overview.recent.all")}</a>} flush>
        {runs.data && all.length === 0 && <Empty title={t("runs.empty.title")} />}
        {all.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("runs.table.run")}</th>
                  <th>{t("runs.table.scene")}</th>
                  <th>{t("runs.table.approach")}</th>
                  <th>{t("runs.table.status")}</th>
                  <th className="num">{t("runs.table.created")}</th>
                </tr>
              </thead>
              <tbody>
                {all.slice(0, 8).map((run) => (
                  <tr key={run.run_id} className="link-row" onClick={() => go({ page: "run", runId: run.run_id })}>
                    <td className="id">{run.run_id}</td>
                    <td className="ellipsis" style={{ maxWidth: 301 }}>{run.scene_id}</td>
                    <td>{run.approach.approach_id}</td>
                    <td><Status value={live.statuses[run.run_id] ?? run.status} /></td>
                    <td className="num muted">{ago(run.created_utc ?? null, locale)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {(scenes.data?.items.length ?? 0) > 0 && (
        <section className="stack" style={{ gap: 12 }}>
          <div className="row">
            <h2 className="grow">{t("nav.menu.scenes")}</h2>
            <a className="btn ghost sm" href={href({ page: "scenes" })}>{t("overview.scenes.all")}</a>
          </div>
          <div className="cards">
            {scenes.data!.items.slice(0, 4).map((m) => (
              <SceneCard key={m.scene_id} manifest={m} />
            ))}
          </div>
        </section>
      )}
      {adding && <IngestDrawer onClose={() => setAdding(false)} />}
    </div>
  );
}
