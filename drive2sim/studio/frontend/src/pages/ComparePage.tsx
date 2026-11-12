// Nhóm D: so sánh các run trên cùng cảnh và split. Bảng xếp hạng (sắp được theo cột), PSNR theo
// thời gian train và theo VRAM, tải bảng CSV, và video của hai run đặt cạnh nhau.
import { useMemo, useState } from "react";
import { useApi } from "../api/client";
import type { List, Manifest, ReportRow, SceneDetail, Video } from "../api/types";
import { ScatterChart } from "../components/charts";
import { ErrorBox, fixed, Loading, optionsText, Status } from "../components/common";
import { useI18n } from "../i18n";
import { go, href } from "../router";

type SortKey = "psnr" | "ssim" | "lpips" | "train_time_s" | "peak_vram_mb";
const LOWER_IS_BETTER: SortKey[] = ["lpips", "train_time_s", "peak_vram_mb"];

export function ComparePage({ sceneId = "", splitId = "" }: { sceneId?: string; splitId?: string }) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const scene = useApi<SceneDetail>(sceneId ? `/api/scenes/${encodeURIComponent(sceneId)}` : null);
  const query = new URLSearchParams();
  if (sceneId) query.set("scene_id", sceneId);
  if (splitId) query.set("split_id", splitId);
  const report = useApi<List<ReportRow>>(`/api/report?${query}`);
  const [sort, setSort] = useState<SortKey>("psnr");
  const [picked, setPicked] = useState<string[]>([]);

  const rows = useMemo(() => {
    const items = [...(report.data?.items ?? [])];
    const sign = LOWER_IS_BETTER.includes(sort) ? 1 : -1;
    return items.sort((a, b) => {
      const av = a[sort];
      const bv = b[sort];
      if (av == null) return 1;
      if (bv == null) return -1;
      return sign * (av - bv);
    });
  }, [report.data, sort]);

  const byTime = rows
    .filter((r) => r.psnr != null && r.train_time_s != null)
    .map((r) => ({ x: (r.train_time_s ?? 0) / 60, y: r.psnr ?? 0, label: r.run_id, name: r.approach_id }));
  const byVram = rows
    .filter((r) => r.psnr != null && r.peak_vram_mb)
    .map((r) => ({ x: r.peak_vram_mb ?? 0, y: r.psnr ?? 0, label: r.run_id, name: r.approach_id }));
  const mixedSplits = new Set(rows.map((r) => r.split_id)).size > 1;

  const header = (key: SortKey, label: string) => (
    <th className="num" aria-sort={sort === key ? "descending" : undefined}>
      <button onClick={() => setSort(key)}>{label}</button>
    </th>
  );
  const toggle = (runId: string) =>
    setPicked((p) => (p.includes(runId) ? p.filter((id) => id !== runId) : [...p, runId].slice(-2)));

  return (
    <>
      <h1>{t("compare.page.title")}</h1>
      <p className="lede">{t("compare.page.lede")}</p>
      <div className="form-row">
        <label className="field">
          {t("compare.field.scene")}
          <select value={sceneId} onChange={(e) => go({ page: "compare", sceneId: e.target.value || undefined })}>
            <option value="">{t("compare.field.all")}</option>
            {(scenes.data?.items ?? []).map((m) => (
              <option key={m.scene_id}>{m.scene_id}</option>
            ))}
          </select>
        </label>
        <label className="field">
          {t("compare.field.split")}
          <select
            value={splitId}
            disabled={!scene.data}
            onChange={(e) => go({ page: "compare", sceneId, splitId: e.target.value || undefined })}
          >
            <option value="">{t("compare.field.all")}</option>
            {(scene.data?.splits ?? []).map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
        <a className="button" href={`/api/report?${query}&format=csv`} download="d2s_report.csv" style={{ textDecoration: "none" }}>
          {t("compare.export")}
        </a>
      </div>

      <ErrorBox error={report.error} />
      {!report.data && !report.error && <Loading />}
      {report.data && rows.length === 0 && <p className="empty">{t("compare.empty.text")}</p>}
      {mixedSplits && <p className="error">{t("compare.mixed")}</p>}
      {rows.length > 0 && (
        <table>
          <thead>
            <tr>
              <th />
              <th>{t("compare.table.rank")}</th>
              <th>{t("runs.table.run")}</th>
              {!splitId && <th>{t("compare.table.split")}</th>}
              <th>{t("runs.table.approach")}</th>
              <th>{t("runs.table.options")}</th>
              {header("psnr", "PSNR")}
              {header("ssim", "SSIM")}
              {header("lpips", "LPIPS")}
              {header("train_time_s", t("compare.table.time"))}
              {header("peak_vram_mb", t("compare.table.vram"))}
              <th>{t("runs.table.status")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={row.run_id}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={row.run_id}
                    checked={picked.includes(row.run_id)}
                    onChange={() => toggle(row.run_id)}
                  />
                </td>
                <td className="num">{row[sort] == null ? "–" : index + 1}</td>
                <td>
                  <a href={href({ page: "run", runId: row.run_id })}>{row.run_id}</a>
                </td>
                {!splitId && <td>{row.split_id}</td>}
                <td>{row.approach_id}</td>
                <td className="small">{optionsText(row.options)}</td>
                <td className="num">{fixed(row.psnr)}</td>
                <td className="num">{fixed(row.ssim, 3)}</td>
                <td className="num">{fixed(row.lpips, 3)}</td>
                <td className="num">{row.train_time_s == null ? "–" : (row.train_time_s / 60).toFixed(1)}</td>
                <td className="num">{row.peak_vram_mb || "–"}</td>
                <td>
                  <Status value={row.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {rows.length > 0 && (
        <div className="cols" style={{ marginTop: 24 }}>
          <figure style={{ margin: 0 }}>
            <figcaption>{t("compare.chart.time")}</figcaption>
            {byTime.length ? <ScatterChart points={byTime} label={t("compare.chart.time")} xLabel={t("compare.chart.x_time")} /> : <p className="muted small">{t("compare.chart.none")}</p>}
          </figure>
          <figure style={{ margin: 0 }}>
            <figcaption>{t("compare.chart.vram")}</figcaption>
            {byVram.length ? <ScatterChart points={byVram} label={t("compare.chart.vram")} xLabel={t("compare.chart.x_vram")} /> : <p className="muted small">{t("compare.chart.none")}</p>}
          </figure>
        </div>
      )}

      {rows.length > 1 && (
        <section>
          <h2>{t("compare.side.title")}</h2>
          {picked.length < 2 ? <p className="muted">{t("compare.side.hint")}</p> : <SideBySide runIds={picked} />}
        </section>
      )}
    </>
  );
}

function SideBySide({ runIds }: { runIds: string[] }) {
  return (
    <div className="videos">
      {runIds.map((runId) => (
        <FirstVideo key={runId} runId={runId} />
      ))}
    </div>
  );
}

function FirstVideo({ runId }: { runId: string }) {
  const { t } = useI18n();
  const videos = useApi<List<Video>>(`/api/runs/${encodeURIComponent(runId)}/videos`);
  const video = videos.data?.items[0];
  return (
    <figure style={{ margin: 0 }}>
      {video ? <video src={video.url} controls preload="metadata" /> : <p className="muted">{t("run.videos.none")}</p>}
      <figcaption className="small muted">{runId}</figcaption>
    </figure>
  );
}
