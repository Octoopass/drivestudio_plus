// So sánh các run trên một cảnh: một split, hoặc mọi split (mặc định), có thể gom theo split.
// - Một bảng, chọn nhóm metric bằng tab (tổng hợp, rồi các nhóm trong metrics.ts, cùng thứ tự với
//   trang run); bảng luôn giữ cột run và cấu hình, thêm cột split khi xem mọi split. Bảng trộn
//   nhiều split (không gom) hay nhiều GPU thì cột bị ảnh hưởng có dấu tam giác cảnh báo ở tiêu đề.
//   Mỗi nhóm sắp theo một cột của chính nhóm đó
//   (chọn ở thanh trên bảng hoặc bấm tiêu đề cột), chiều cao → thấp hoặc thấp → cao. Ô metric có
//   hướng tốt hơn được tô theo bảng màu Turbo, từ kém tới tốt.
// - Biểu đồ: chất lượng theo thời gian train, VRAM, tốc độ render, số Gaussian (nối các run trên
//   đường Pareto); PSNR theo vùng; phân rã thời gian dựng cảnh; đường cong train.
// - Video của hai run đặt cạnh nhau, phát đồng bộ.
import { ArrowDownWideNarrow, ArrowUpNarrowWide, ChevronDown, ChevronsUpDown, ChevronUp, Download, Pause, Play, AlertTriangle } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, seg, useApi } from "../api/client";
import type { List, Manifest, ReportRow, SceneDetail, Video } from "../api/types";
import { GroupedBarChart, Legend, MultiLineChart, ScatterChart, seriesColor, StackedBarChart, type Point } from "../components/charts";
import { duration, Empty, ErrorBox, InfoTip, Loading, optionsText, PageHead, Panel, Segmented, Tabs, turbo } from "../components/ui";
import { useI18n } from "../i18n";
import { BetterArrow, isComparable, SUMMARY_GROUP, visibleGroups, type MetricDef, type MetricKey } from "../metrics";
import { go, href } from "../router";

const TAB_KEY = "d2s.compare.tab";
const BY_SPLIT_KEY = "d2s.compare.by_split";
function readPref(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}
function writePref(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Không lưu được (chế độ riêng tư): chỉ mất việc nhớ tab.
  }
}

/** Ô metric tô màu: tốt nhất về phía cam của Turbo, kém nhất về phía xanh. */
function Heat({ text, value, values, better }: { text: string; value: number | null; values: number[]; better?: "high" | "low" }) {
  if (value == null || !better || values.length < 2) return <span className={value == null ? "faint" : undefined}>{text}</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const rank = max === min ? 1 : (value - min) / (max - min);
  const goodness = better === "low" ? 1 - rank : rank;
  // Chỉ dùng đoạn giữa của Turbo (xanh lơ tới cam vàng): chữ tối luôn đọc được, và đầu tốt không
  // ra đỏ để khỏi lẫn với màu lỗi.
  return (
    <span className="heat" style={{ background: turbo(0.3 + goodness * 0.38) }}>
      {text}
    </span>
  );
}

/** Dấu tam giác cảnh báo cạnh tiêu đề cột hoặc biểu đồ; rê chuột để đọc lý do. */
function WarnMark({ text }: { text: string }) {
  return (
    <span className="warn-mark" title={text} aria-label={text}>
      <AlertTriangle size={15} />
    </span>
  );
}

// Nhóm stage cho biểu đồ phân rã thời gian, theo tiền tố stage_id.
const TIME_PARTS = ["prep", "train", "eval", "export"] as const;

export function ComparePage({ sceneId = "", splitId = "" }: { sceneId?: string; splitId?: string }) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  // Luôn so trên một cảnh: chưa chọn thì mở cảnh của run so được mới nhất (run_id bắt đầu bằng
  // thời điểm tạo), không có run nào thì lấy cảnh đầu tiên. Split: một split, hoặc mọi split
  // (mặc định).
  const report = useApi<List<ReportRow>>("/api/report");
  const latest = (report.data?.items ?? []).filter(isComparable).reduce<ReportRow | undefined>((a, r) => (!a || r.run_id > a.run_id ? r : a), undefined);
  const currentScene = sceneId || latest?.scene_id || scenes.data?.items[0]?.scene_id || "";
  const scene = useApi<SceneDetail>(currentScene ? `/api/scenes/${seg(currentScene)}` : null);
  const currentSplit = splitId;
  const query = new URLSearchParams();
  if (currentScene) query.set("scene_id", currentScene);
  if (currentSplit) query.set("split_id", currentSplit);
  const inView = (r: ReportRow) => r.scene_id === currentScene && (!currentSplit || r.split_id === currentSplit);
  // Một bảng, chọn nhóm metric bằng tab (nhớ trong trình duyệt). Mỗi nhóm có cột sắp xếp mặc định;
  // chiều mặc định là "tốt nhất trước" (theo hướng tốt hơn của cột).
  const [groupId, setGroupId] = useState<string>(() => readPref(TAB_KEY, "summary"));
  // Gom theo split: mỗi split một khối, xếp hạng và tô màu trong từng khối.
  const [bySplit, setBySplit] = useState<boolean>(() => readPref(BY_SPLIT_KEY, "0") === "1");
  const [sortKey, setSortKey] = useState<MetricKey | null>(null);
  const [direction, setDirection] = useState<"asc" | "desc" | null>(null);

  // Chỉ run đã xong và train đầy đủ: run đang chạy, lỗi, huỷ hay chạy thử nhanh không có metric
  // so được với các run còn lại.
  const comparable = useMemo(() => (report.data?.items ?? []).filter((r) => inView(r) && isComparable(r)), [report.data, currentScene, currentSplit]);
  const groups = comparable.length ? [SUMMARY_GROUP, ...visibleGroups(comparable)] : [];
  const group = groups.find((g) => g.id === groupId) ?? groups[0];
  const sortable = group ? group.metrics.filter((m) => m.key !== "gpu") : [];
  const sort = sortable.find((m) => m.key === sortKey) ?? sortable.find((m) => m.key === group?.defaultSort) ?? sortable[0];
  const dir = direction ?? (sort?.better === "low" ? "asc" : "desc");
  const rows = useMemo(() => {
    if (!sort) return comparable;
    const sign = dir === "asc" ? 1 : -1;
    // Run thiếu số liệu luôn ở cuối, dù sắp chiều nào.
    return [...comparable].sort((a, b) => {
      const av = sort.value(a);
      const bv = sort.value(b);
      if (av == null) return bv == null ? 0 : 1;
      if (bv == null) return -1;
      return sign * (av - bv);
    });
  }, [comparable, sort, dir]);
  const chooseGroup = (id: string) => {
    setGroupId(id);
    writePref(TAB_KEY, id);
    setSortKey(null);
    setDirection(null);
  };
  const chooseSort = (metric: MetricDef) => {
    if (metric.key === sort?.key) setDirection(dir === "asc" ? "desc" : "asc");
    else {
      setSortKey(metric.key);
      setDirection(null);
    }
  };
  const splits = [...new Set(rows.map((r) => r.split_id))].sort();
  const grouped = bySplit && splits.length > 1;
  const blocks = grouped ? splits.map((id) => ({ split: id, rows: rows.filter((r) => r.split_id === id) })) : [{ split: "", rows }];
  // Bảng trộn nhiều split mà không gom: cột metric đo trên ảnh test (chất lượng ảnh, theo vùng) có
  // dấu cảnh báo, vì hai split khác nhau đánh giá trên các khung test khác nhau.
  const mixedSplit = splits.length > 1 && !grouped;
  const chooseBySplit = (value: boolean) => {
    setBySplit(value);
    writePref(BY_SPLIT_KEY, value ? "1" : "0");
  };
  // Bảng có run chạy trên nhiều GPU: cột phụ thuộc phần cứng có dấu cảnh báo nhỏ ở tiêu đề.
  const gpus = [...new Set(rows.map((r) => r.gpu).filter((g): g is string => !!g))];
  const mixedGpu = gpus.length > 1;
  // Bảng có run chạy bằng phiên bản PyTorch hoặc gsplat khác nhau: mọi metric có thể lệch dù cùng
  // cấu hình, nên cảnh báo một lần ở thanh trên bảng (không gắn vào từng cột).
  const softwares = [...new Set(rows.map((r) => r.software).filter((v): v is string => !!v))];

  return (
    <div className="page">
      <PageHead
        title={t("compare.page.title")}
        info={t("compare.page.lede")}
        actions={
          <a className="btn" href={`/api/report?${query}${query.toString() ? "&" : ""}format=csv`} download="d2s_report.csv">
            <Download size={17} />
            {t("compare.export")}
          </a>
        }
      />
      <div className="row wrap">
        <label className="field">
          <span>{t("compare.field.scene")}</span>
          <select value={currentScene} disabled={!scenes.data?.items.length} onChange={(e) => go({ page: "compare", sceneId: e.target.value })}>
            {(scenes.data?.items ?? []).map((m) => (
              <option key={m.scene_id}>{m.scene_id}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>{t("compare.field.split")}</span>
          <select value={currentSplit} disabled={!scene.data?.splits.length} onChange={(e) => go({ page: "compare", sceneId: currentScene, splitId: e.target.value || undefined })}>
            <option value="">{t("compare.field.all_splits")}</option>
            {(scene.data?.splits ?? []).map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
      </div>

      <ErrorBox error={report.error} />
      {(!scenes.data || !report.data) && !report.error && !scenes.error && <Loading rows={4} />}
      {((scenes.data && !currentScene) || (report.data && rows.length === 0)) && (
        <section className="panel">
          <Empty title={t("compare.empty.title")}>
            <p>{t("compare.empty.text")}</p>
          </Empty>
        </section>
      )}

      {group && sort && (
        <section className="panel">
          <Tabs<string> value={group.id} onChange={chooseGroup} items={groups.map((g) => ({ value: g.id, label: t(g.label) }))} />
          <div className="table-toolbar">
            <span className="small muted row" style={{ gap: 2 }}>
              {t("compare.count", { count: rows.length })}
              <InfoTip>
                {t(group.hint)} {t("compare.arrows")}
              </InfoTip>
            </span>
            {softwares.length > 1 && (
              <span className="tag warn row" style={{ gap: 4 }} title={t("compare.mixed_software", { versions: softwares.join(" | ") })}>
                <AlertTriangle size={14} />
                {t("compare.mixed_software_tag", { count: softwares.length })}
              </span>
            )}
            <span className="grow" />
            {!currentSplit && splits.length > 1 && (
              <Segmented<"mixed" | "split">
                label={t("compare.by_split.label")}
                value={grouped ? "split" : "mixed"}
                onChange={(v) => chooseBySplit(v === "split")}
                options={[
                  { value: "mixed", label: t("compare.by_split.off") },
                  { value: "split", label: t("compare.by_split.on") },
                ]}
              />
            )}
            <label className="field inline">
              <span>{t("compare.sort.by")}</span>
              <select value={sort.key} onChange={(e) => chooseSort(sortable.find((m) => m.key === e.target.value)!)}>
                {sortable.map((m) => (
                  <option key={m.key} value={m.key}>
                    {t(m.label)}
                  </option>
                ))}
              </select>
            </label>
            <Segmented<"desc" | "asc">
              label={t("compare.sort.direction")}
              value={dir}
              onChange={setDirection}
              options={[
                { value: "desc", label: <span className="row" style={{ gap: 6 }}><ArrowDownWideNarrow size={16} />{t("compare.sort.desc")}</span> },
                { value: "asc", label: <span className="row" style={{ gap: 6 }}><ArrowUpNarrowWide size={16} />{t("compare.sort.asc")}</span> },
              ]}
            />
            {sort.better && (
              <span className="tag">{t((sort.better === "low") === (dir === "asc") ? "compare.sort.best_first" : "compare.sort.worst_first")}</span>
            )}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>{t("runs.table.run")}</th>
                  {!currentSplit && !grouped && <th>{t("compare.table.split")}</th>}
                  <th>{t("runs.table.options")}</th>
                  {group.metrics.map((metric) => (
                    <th
                      key={metric.key}
                      className={metric.key === "gpu" ? undefined : "num"}
                      aria-sort={sort.key === metric.key ? (dir === "asc" ? "ascending" : "descending") : undefined}
                    >
                      <span className="th-metric">
                        {metric.key === "gpu" ? (
                          <span>{t(metric.label)}</span>
                        ) : (
                          <button
                            onClick={() => chooseSort(metric)}
                            className={sort.key === metric.key ? "sorted" : undefined}
                            title={t("compare.sort.click")}
                          >
                            {t(metric.label)}
                            {metric.unit && <span className="faint th-unit">({metric.unit})</span>}
                            <BetterArrow better={metric.better} />
                            {/* Ô icon chiều sắp xếp có độ rộng cố định ở mọi cột: bấm cột nào cũng không
                                làm cột đổi độ rộng, bảng không bị xô. */}
                            <span className="sort-icon" aria-hidden="true">
                              {sort.key !== metric.key ? <ChevronsUpDown size={14} /> : dir === "asc" ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                            </span>
                          </button>
                        )}
                        {mixedGpu && metric.hardware && <WarnMark text={t("compare.mixed_gpu", { gpus: gpus.join(", ") })} />}
                        {mixedSplit && metric.evalSplit && <WarnMark text={t("compare.mixed_split", { splits: splits.join(", ") })} />}
                        {metric.hint && <InfoTip>{t(metric.hint)}</InfoTip>}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              {blocks.map((block) => (
              <tbody key={block.split}>
                {grouped && (
                  <tr className="group-row">
                    <td colSpan={3 + group.metrics.length}>
                      {t("compare.table.split")} <b>{block.split}</b>
                      <span className="faint"> · {t("compare.count", { count: block.rows.length })}</span>
                    </td>
                  </tr>
                )}
                {block.rows.map((row, index) => (
                  <tr key={row.run_id}>
                    <td className="num faint">{sort.value(row) == null ? "–" : index + 1}</td>
                    <td>
                      <a className="id" href={href({ page: "run", runId: row.run_id })}>{row.run_id}</a>
                      <div className="small faint row" style={{ gap: 6 }}>
                        {row.approach_id}
                      </div>
                    </td>
                    {!currentSplit && !grouped && <td className="small nowrap">{row.split_id}</td>}
                    <td className="small muted" style={{ maxWidth: 260 }}>{optionsText(row.options) || "–"}</td>
                    {group.metrics.map((metric) => (
                      <td key={metric.key} className={metric.key === "gpu" ? "small nowrap" : sort.key === metric.key ? "num sorted-col" : "num"}>
                        <Heat text={metric.format(row)} value={metric.value(row)} values={block.rows.map((r) => metric.value(r)).filter((v): v is number => v != null)} better={metric.better} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
              ))}
            </table>
          </div>
        </section>
      )}

      {rows.length > 0 && <Charts rows={rows} splits={splits} grouped={grouped} gpus={gpus} />}

      {rows.length > 1 && (
        <Panel title={t("compare.side.title")} info={t("compare.side.info")}>
          <SideBySide rows={rows} />
        </Panel>
      )}
    </div>
  );
}

// Biểu đồ theo đúng cách bảng đang hiện split: trộn chung thì vẽ chung (màu theo approach, một
// đường Pareto) và có dấu cảnh báo ở biểu đồ đo trên ảnh; gom theo split thì màu theo split, mỗi
// split một đường Pareto, cột và đường cong ghi kèm split. Bảng trộn nhiều GPU thì biểu đồ có trục
// thời gian, VRAM hay FPS có dấu cảnh báo.
function Charts({ rows: sorted, splits, grouped, gpus }: { rows: ReportRow[]; splits: string[]; grouped: boolean; gpus: string[] }) {
  const { t } = useI18n();
  const rows = grouped ? [...sorted].sort((a, b) => a.split_id.localeCompare(b.split_id)) : sorted;
  const splitWarn = splits.length > 1 && !grouped ? <WarnMark text={t("compare.chart.mixed_split", { splits: splits.join(", ") })} /> : null;
  const gpuWarn = gpus.length > 1 ? <WarnMark text={t("compare.chart.mixed_gpu", { gpus: gpus.join(", ") })} /> : null;
  const titled = (key: string, ...marks: ReactNode[]) => (
    <span className="row" style={{ gap: 6 }}>
      {t(key)}
      {marks}
    </span>
  );
  const withPsnr = rows.filter((r) => r.psnr != null);
  const scatter = (x: (r: ReportRow) => number | null): Point[] =>
    withPsnr.flatMap((r) => {
      const value = x(r);
      return value != null && value > 0
        ? [{ x: value, y: r.psnr ?? 0, label: r.run_id, name: `${r.approach_id} ${r.run_id.slice(-4)}`, group: grouped ? r.split_id : r.approach_id }]
        : [];
    });
  const plots: { key: string; points: Point[]; xLabel: string; xBetter: "low" | "high"; hardware: boolean }[] = [
    { key: "time", points: scatter((r) => (r.train_time_s ?? 0) / 60), xLabel: t("compare.chart.x_time"), xBetter: "low", hardware: true },
    { key: "vram", points: scatter((r) => (r.peak_vram_mb ?? 0) / 1024), xLabel: "GB", xBetter: "low", hardware: true },
    { key: "fps", points: scatter((r) => r.render_fps), xLabel: "fps", xBetter: "high", hardware: true },
    { key: "gaussians", points: scatter((r) => (r.num_gaussians ?? 0) / 1e6), xLabel: t("compare.chart.x_gaussians"), xBetter: "low", hardware: false },
  ];
  const short = (r: ReportRow) => `${r.approach_id} ${r.run_id.slice(-4)}${grouped ? ` · ${r.split_id}` : ""}`;
  const regionSeries = [t("metrics.psnr"), t("run.result.human"), t("run.result.vehicle"), t("run.result.dynamic")];
  const regionGroups = withPsnr
    .filter((r) => r.human_psnr != null || r.vehicle_psnr != null || r.dynamic_psnr != null)
    .map((r) => ({ label: short(r), values: [r.psnr, r.human_psnr, r.vehicle_psnr, r.dynamic_psnr] }));
  const timeRows = rows
    .filter((r) => r.recon_time_s != null)
    .map((r) => ({
      label: short(r),
      parts: TIME_PARTS.map((part) => {
        const stages = Object.entries(r.stage_times ?? {}).filter(([id]) => id.startsWith(`${part}.`) && id !== "export.preview");
        return { value: stages.reduce((sum, [, v]) => sum + v.seconds, 0), faded: stages.length > 0 && stages.every(([, v]) => v.reused) };
      }),
    }));
  return (
    <>
      <div className="grid-2">
        {plots.map((plot) => (
          <Panel key={plot.key} title={titled(`compare.chart.${plot.key}`, splitWarn, plot.hardware && gpuWarn)} info={t("compare.chart.pareto_hint")}>
            {plot.points.length ? (
              <>
                <ScatterChart
                  points={plot.points}
                  label={t(`compare.chart.${plot.key}`)}
                  xLabel={plot.xLabel}
                  xBetter={plot.xBetter}
                  groups={grouped ? splits : undefined}
                  frontByGroup={grouped}
                />
                {grouped && <Legend items={splits.map((name, i) => ({ name, color: seriesColor(i) }))} />}
              </>
            ) : (
              <p className="muted small">{t("compare.chart.none")}</p>
            )}
          </Panel>
        ))}
        <Panel title={titled("compare.chart.regions", splitWarn)} info={t("compare.chart.regions_hint")}>
          {regionGroups.length ? (
            <>
              <GroupedBarChart groups={regionGroups} seriesNames={regionSeries} label={t("compare.chart.regions")} unit="dB" />
              <Legend items={regionSeries.map((name, i) => ({ name, color: seriesColor(i) }))} />
            </>
          ) : (
            <p className="muted small">{t("compare.chart.none")}</p>
          )}
        </Panel>
        <Panel title={titled("compare.chart.breakdown", gpuWarn)} info={t("compare.chart.breakdown_hint")}>
          {timeRows.length ? (
            <>
              <StackedBarChart rows={timeRows} partNames={TIME_PARTS.map((p) => t(`compare.chart.part.${p}`))} label={t("compare.chart.breakdown")} format={(s) => duration(s)} />
              <Legend items={TIME_PARTS.map((p, i) => ({ name: t(`compare.chart.part.${p}`), color: seriesColor(i) }))} />
            </>
          ) : (
            <p className="muted small">{t("compare.chart.none")}</p>
          )}
        </Panel>
      </div>
      <TrainCurves rows={rows} grouped={grouped} warn={splitWarn} />
    </>
  );
}

/** PSNR trên ảnh train theo bước của nhiều run (tối đa 6), đọc từ metrics.json của từng run. */
function TrainCurves({ rows, grouped, warn }: { rows: ReportRow[]; grouped: boolean; warn: ReactNode }) {
  const { t } = useI18n();
  const shown = rows.filter((r) => r.status === "succeeded" || r.status === "running").slice(0, 6);
  const [curves, setCurves] = useState<Record<string, Point[]>>({});
  const ids = shown.map((r) => r.run_id).join(",");
  useEffect(() => {
    let alive = true;
    void Promise.all(
      shown.map((r) =>
        api<{ series: Record<string, [number, number][]> }>(`/api/runs/${seg(r.run_id)}/curve`)
          .then((d) => [r.run_id, (d.series.psnr ?? []).map(([x, y]) => ({ x, y }))] as const)
          .catch(() => [r.run_id, [] as Point[]] as const),
      ),
    ).then((pairs) => {
      if (alive) setCurves(Object.fromEntries(pairs));
    });
    return () => {
      alive = false;
    };
  }, [ids]);
  const series = shown
    .map((r) => ({ name: `${r.approach_id} ${r.run_id.slice(-4)}${grouped ? ` · ${r.split_id}` : ""}`, points: curves[r.run_id] ?? [] }))
    .filter((s) => s.points.length > 1);
  if (!series.length) return null;
  return (
    <Panel
      title={
        <span className="row" style={{ gap: 6 }}>
          {t("compare.chart.curves")}
          {warn}
        </span>
      }
      info={t("compare.chart.curves_hint")}>
      {/* Biểu đồ vẽ theo viewBox cố định: giới hạn bề rộng để chữ trên trục không phóng quá to. */}
      <div style={{ maxWidth: 820 }}>
        <MultiLineChart series={series} label={t("compare.chart.curves")} xLabel={t("run.curve.step")} />
      </div>
      <Legend items={series.map((s, i) => ({ name: s.name, color: seriesColor(i) }))} />
    </Panel>
  );
}

/** Hai video phát đồng bộ; chọn run cho từng bên ngay trong khung, mặc định hai run đứng đầu bảng. */
function SideBySide({ rows }: { rows: ReportRow[] }) {
  const { t } = useI18n();
  const [left, setLeft] = useState(rows[0]?.run_id ?? "");
  const [right, setRight] = useState(rows[1]?.run_id ?? "");
  const runIds = [left, right];
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
    <div className="stack" style={{ gap: 12 }}>
      <div className="row wrap" style={{ gap: 12, alignItems: "flex-end" }}>
        {[
          { label: t("compare.side.left"), value: left, set: setLeft },
          { label: t("compare.side.right"), value: right, set: setRight },
        ].map((side) => (
          <label key={side.label} className="field">
            <span>{side.label}</span>
            <select value={side.value} onChange={(e) => side.set(e.target.value)}>
              {rows.map((r) => (
                <option key={r.run_id} value={r.run_id}>{`${r.approach_id} ${r.run_id.slice(-4)} (${r.run_id})`}</option>
              ))}
            </select>
          </label>
        ))}
        <button className="btn" onClick={toggle}>
          {playing ? <Pause size={17} /> : <Play size={17} />}
          {t(playing ? "compare.side.pause" : "compare.side.play")}
        </button>
      </div>
      <div className="videos">
        {runIds.map((runId, index) => (
          <FirstVideo key={`${index}-${runId}`} runId={runId} register={(el) => (refs.current[index] = el)} />
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
