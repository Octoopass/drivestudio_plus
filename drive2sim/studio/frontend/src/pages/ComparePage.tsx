// So sánh các run trên cùng cảnh và split: bảng xếp hạng (sắp theo cột, ô metric tô theo bảng màu
// Turbo từ kém tới tốt), PSNR theo thời gian train và theo VRAM, tải CSV, và video của hai run đặt
// cạnh nhau, phát đồng bộ.
import { Download, Pause, Play } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { seg, useApi } from "../api/client";
import type { List, Manifest, ReportRow, SceneDetail, Video } from "../api/types";
import { ScatterChart } from "../components/charts";
import { Empty, ErrorBox, Loading, Note, optionsText, PageHead, Panel, Status, turbo } from "../components/ui";
import { useI18n } from "../i18n";
import { go, href } from "../router";

type SortKey = "psnr" | "ssim" | "lpips" | "train_time_s" | "peak_vram_mb";
const LOWER_IS_BETTER: SortKey[] = ["lpips", "train_time_s", "peak_vram_mb"];

/** Ô metric tô màu: tốt nhất về phía đỏ cam của Turbo, kém nhất về phía xanh. */
function Heat({ value, values, lowerBetter, digits }: { value: number | null; values: number[]; lowerBetter?: boolean; digits: number }) {
  if (value == null) return <span className="faint">–</span>;
  if (values.length < 2) return <span>{value.toFixed(digits)}</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const rank = max === min ? 1 : (value - min) / (max - min);
  const goodness = lowerBetter ? 1 - rank : rank;
  // Chỉ dùng đoạn giữa của Turbo (xanh lơ tới cam vàng): chữ tối luôn đọc được, và đầu tốt không
  // ra đỏ để khỏi lẫn với màu lỗi.
  return (
    <span className="heat" style={{ background: turbo(0.3 + goodness * 0.38) }}>
      {value.toFixed(digits)}
    </span>
  );
}

export function ComparePage({ sceneId = "", splitId = "" }: { sceneId?: string; splitId?: string }) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const scene = useApi<SceneDetail>(sceneId ? `/api/scenes/${seg(sceneId)}` : null);
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
  const column = (key: SortKey) => rows.map((r) => r[key]).filter((v): v is number => v != null && v !== 0);

  const byTime = rows
    .filter((r) => r.psnr != null && r.train_time_s != null)
    .map((r) => ({ x: (r.train_time_s ?? 0) / 60, y: r.psnr ?? 0, label: r.run_id, name: r.approach_id }));
  const byVram = rows
    .filter((r) => r.psnr != null && r.peak_vram_mb)
    .map((r) => ({ x: (r.peak_vram_mb ?? 0) / 1024, y: r.psnr ?? 0, label: r.run_id, name: r.approach_id }));
  const mixedSplits = new Set(rows.map((r) => r.split_id)).size > 1;

  const header = (key: SortKey, label: string) => (
    <th className="num" aria-sort={sort === key ? (LOWER_IS_BETTER.includes(key) ? "ascending" : "descending") : undefined}>
      <button onClick={() => setSort(key)}>
        {label}
        {sort === key && <span aria-hidden="true">{LOWER_IS_BETTER.includes(key) ? "↑" : "↓"}</span>}
      </button>
    </th>
  );
  const toggle = (runId: string) => setPicked((p) => (p.includes(runId) ? p.filter((id) => id !== runId) : [...p, runId].slice(-2)));

  return (
    <div className="page">
      <PageHead
        title={t("compare.page.title")}
        lede={t("compare.page.lede")}
        actions={
          <a className="btn" href={`/api/report?${query}${query.toString() ? "&" : ""}format=csv`} download="d2s_report.csv">
            <Download size={15} />
            {t("compare.export")}
          </a>
        }
      />
      <div className="row wrap">
        <label className="field">
          <span>{t("compare.field.scene")}</span>
          <select value={sceneId} onChange={(e) => go({ page: "compare", sceneId: e.target.value || undefined })}>
            <option value="">{t("compare.field.all")}</option>
            {(scenes.data?.items ?? []).map((m) => (
              <option key={m.scene_id}>{m.scene_id}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>{t("compare.field.split")}</span>
          <select value={splitId} disabled={!scene.data} onChange={(e) => go({ page: "compare", sceneId, splitId: e.target.value || undefined })}>
            <option value="">{t("compare.field.all")}</option>
            {(scene.data?.splits ?? []).map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
      </div>

      <ErrorBox error={report.error} />
      {mixedSplits && <Note kind="warn">{t("compare.mixed")}</Note>}
      <section className="panel">
        {!report.data && !report.error && <div className="panel-body"><Loading rows={4} /></div>}
        {report.data && rows.length === 0 && <Empty title={t("compare.empty.title")}><p>{t("compare.empty.text")}</p></Empty>}
        {rows.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th aria-label={t("compare.side.pick")} />
                  <th className="num">#</th>
                  <th>{t("runs.table.run")}</th>
                  {!splitId && <th>{t("compare.table.split")}</th>}
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
                  <tr key={row.run_id} className={picked.includes(row.run_id) ? "selected" : undefined}>
                    <td>
                      <input type="checkbox" aria-label={row.run_id} checked={picked.includes(row.run_id)} onChange={() => toggle(row.run_id)} />
                    </td>
                    <td className="num faint">{row[sort] == null ? "–" : index + 1}</td>
                    <td>
                      <a className="id" href={href({ page: "run", runId: row.run_id })}>{row.run_id}</a>
                      <div className="small faint">{row.approach_id}</div>
                    </td>
                    {!splitId && <td>{row.split_id}</td>}
                    <td className="small muted" style={{ maxWidth: 240 }}>{optionsText(row.options) || "–"}</td>
                    <td className="num"><Heat value={row.psnr} values={column("psnr")} digits={2} /></td>
                    <td className="num"><Heat value={row.ssim} values={column("ssim")} digits={3} /></td>
                    <td className="num"><Heat value={row.lpips} values={column("lpips")} lowerBetter digits={3} /></td>
                    <td className="num">{row.train_time_s == null ? "–" : (row.train_time_s / 60).toFixed(1)}</td>
                    <td className="num">{row.peak_vram_mb ? (row.peak_vram_mb / 1024).toFixed(1) : "–"}</td>
                    <td><Status value={row.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {rows.length > 0 && (
        <div className="grid-2">
          <Panel title={t("compare.chart.time")}>
            {byTime.length ? <ScatterChart points={byTime} label={t("compare.chart.time")} xLabel={t("compare.chart.x_time")} /> : <p className="muted small">{t("compare.chart.none")}</p>}
          </Panel>
          <Panel title={t("compare.chart.vram")}>
            {byVram.length ? <ScatterChart points={byVram} label={t("compare.chart.vram")} xLabel="GB" /> : <p className="muted small">{t("compare.chart.none")}</p>}
          </Panel>
        </div>
      )}

      {rows.length > 1 && (
        <Panel title={t("compare.side.title")}>
          {picked.length < 2 ? <p className="muted">{t("compare.side.hint")}</p> : <SideBySide runIds={picked} />}
        </Panel>
      )}
    </div>
  );
}

function SideBySide({ runIds }: { runIds: string[] }) {
  const { t } = useI18n();
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  const [playing, setPlaying] = useState(false);
  const toggle = () => {
    const videos = refs.current.filter((v): v is HTMLVideoElement => Boolean(v));
    if (playing) videos.forEach((v) => v.pause());
    else {
      const time = videos[0]?.currentTime ?? 0;
      videos.forEach((v) => {
        v.currentTime = time;
        void v.play();
      });
    }
    setPlaying(!playing);
  };
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div>
        <button className="btn" onClick={toggle}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
          {t(playing ? "compare.side.pause" : "compare.side.play")}
        </button>
      </div>
      <div className="videos">
        {runIds.map((runId, index) => (
          <FirstVideo key={runId} runId={runId} register={(el) => (refs.current[index] = el)} />
        ))}
      </div>
    </div>
  );
}

function FirstVideo({ runId, register }: { runId: string; register: (el: HTMLVideoElement | null) => void }) {
  const { t } = useI18n();
  const videos = useApi<List<Video>>(`/api/runs/${seg(runId)}/videos`);
  const video = videos.data?.items[0];
  return (
    <figure>
      {video ? <video ref={register} src={video.url} controls muted preload="metadata" /> : <p className="muted">{t("run.videos.none")}</p>}
      <figcaption>{runId}</figcaption>
    </figure>
  );
}
