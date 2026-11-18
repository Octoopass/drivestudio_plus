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
import { ArrowDownWideNarrow, ArrowUpNarrowWide, ChevronDown, ChevronsUpDown, ChevronUp, Download, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, preferredVideo, seg, useApi } from "../api/client";
import type { List, Manifest, ReportRow, SceneDetail, Video } from "../api/types";
import { GroupedBarChart, Legend, MultiLineChart, ScatterChart, seriesColor, StackedBarChart, type Point } from "../components/charts";
import { count, duration, Empty, ErrorBox, InfoTip, Loading, PageHead, Panel, Segmented, Tabs } from "../components/ui";
import { ConfigText } from "../components/ConfigText";
import { useI18n } from "../i18n";
import { BetterArrow, isComparable, size, SUMMARY_GROUP, visibleGroups, type MetricDef, type MetricKey } from "../metrics";
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

/**
 * Ô metric; số tốt nhất của cột in đậm. So theo số đang hiện (đã làm tròn): mọi ô hiện cùng số tốt
 * nhất đều in đậm; cả cột hiện cùng một số thì không in đậm ô nào.
 */
function Best({ text, value, cells, better }: { text: string; value: number | null; cells: { text: string; value: number }[]; better?: "high" | "low" }) {
  if (value == null) return <span className="faint">{text}</span>;
  const shown = new Set(cells.map((cell) => cell.text));
  if (!better || shown.size < 2) return <span>{text}</span>;
  const pick = better === "high" ? Math.max : Math.min;
  const best = cells.find((cell) => cell.value === pick(...cells.map((c) => c.value)));
  return text === best?.text ? <strong>{text}</strong> : <span>{text}</span>;
}

/** Dấu tam giác cảnh báo cạnh tiêu đề cột hoặc biểu đồ; bấm hoặc rê chuột để đọc lý do. */
function WarnMark({ text }: { text: string }) {
  return (
    <InfoTip warn label={text}>
      {text}
    </InfoTip>
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
  const sortable = group ? group.metrics.filter((m) => !m.text) : [];
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
  // Mỗi khối là các run so được với nhau: cùng số iteration và cùng cỡ ảnh lúc train (thêm cùng
  // split khi gom theo split). Run chạy thử 2.000 bước đứng chung bảng với run 30.000 bước thì
  // xếp hạng vô nghĩa, nên tách khối; khối nhiều iteration hơn đứng trước. Xếp hạng, in đậm số
  // tốt nhất làm trong từng khối.
  const blocks = useMemo(() => {
    const keyOf = (r: ReportRow) => [grouped ? r.split_id : "", r.num_iters ?? -1, r.train_resolution ?? ""] as const;
    const found = new Map<string, { key: ReturnType<typeof keyOf>; rows: ReportRow[] }>();
    for (const row of rows) {
      const key = keyOf(row);
      const id = key.join("|");
      if (!found.has(id)) found.set(id, { key, rows: [] });
      found.get(id)!.rows.push(row);
    }
    return [...found.entries()]
      .sort(([, a], [, b]) => a.key[0].localeCompare(b.key[0]) || b.key[1] - a.key[1] || a.key[2].localeCompare(b.key[2]))
      .map(([id, block]) => ({ id, split: block.key[0], rows: block.rows, sample: block.rows[0] }));
  }, [rows, grouped]);
  // Bảng trộn nhiều split mà không gom: cột metric đo trên ảnh test (chất lượng ảnh, theo vùng) có
  // dấu cảnh báo, vì hai split khác nhau đánh giá trên các khung test khác nhau.
  const mixedSplit = splits.length > 1 && !grouped;
  const chooseBySplit = (value: boolean) => {
    setBySplit(value);
    writePref(BY_SPLIT_KEY, value ? "1" : "0");
  };
  // Biểu đồ (chất lượng theo thời gian, Pareto...) chỉ vẽ các run có nhiều iteration nhất: run
  // chạy thử ít bước vừa nhanh vừa kém nên làm lệch đường Pareto.
  const chartIters = Math.max(-1, ...rows.map((r) => r.num_iters ?? -1));
  const chartRows = rows.filter((r) => (r.num_iters ?? -1) === chartIters);
  // Bảng có run train trên nhiều GPU: cảnh báo một lần ở thanh trên bảng.
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
            {/* Cảnh báo trộn GPU, trộn split hiện một lần ở đây, không gắn vào từng cột: gắn từng
                cột làm tiêu đề dài ra, lệch với số bên dưới. */}
            {mixedGpu && (
              <span className="tag warn row" style={{ gap: 4 }}>
                {t("compare.mixed_gpu_tag", { count: gpus.length })}
                <InfoTip warn>{t("compare.mixed_gpu", { gpus: gpus.join(", ") })}</InfoTip>
              </span>
            )}
            {mixedSplit && (
              <span className="tag warn row" style={{ gap: 4 }}>
                {t("compare.mixed_split_tag", { count: splits.length })}
                <InfoTip warn>{t("compare.mixed_split", { splits: splits.join(", ") })}</InfoTip>
              </span>
            )}
            {softwares.length > 1 && (
              <span className="tag warn row" style={{ gap: 4 }}>
                {t("compare.mixed_software_tag", { count: softwares.length })}
                <InfoTip warn>{t("compare.mixed_software", { versions: softwares.join(" | ") })}</InfoTip>
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
            <table className="centered">
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>{t("runs.table.run")}</th>
                  {!currentSplit && !grouped && <th>{t("compare.table.split")}</th>}
                  <th>{t("runs.table.options")}</th>
                  {group.metrics.map((metric) => (
                    <th
                      key={metric.key}
                      className={metric.text ? undefined : "num"}
                      aria-sort={sort.key === metric.key ? (dir === "asc" ? "ascending" : "descending") : undefined}
                    >
                      <span className="th-metric">
                        {metric.text ? (
                          <span title={t(metric.label)}>{t(metric.short ?? metric.label)}</span>
                        ) : (
                          <button
                            onClick={() => chooseSort(metric)}
                            className={sort.key === metric.key ? "sorted" : undefined}
                            title={`${t(metric.label)} · ${t("compare.sort.click")}`}
                          >
                            {t(metric.short ?? metric.label)}
                            {metric.unit && <span className="faint th-unit">({metric.unit})</span>}
                            <BetterArrow better={metric.better} />
                            {/* Ô icon chiều sắp xếp có độ rộng cố định ở mọi cột: bấm cột nào cũng không
                                làm cột đổi độ rộng, bảng không bị xô. */}
                            <span className="sort-icon" aria-hidden="true">
                              {sort.key !== metric.key ? <ChevronsUpDown size={14} /> : dir === "asc" ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                            </span>
                          </button>
                        )}
                        {metric.hint && <InfoTip>{t(metric.hint)}</InfoTip>}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              {blocks.map((block) => (
              <tbody key={block.id}>
                <tr className="group-row">
                  <td className="block-head" colSpan={(!currentSplit && !grouped ? 4 : 3) + group.metrics.length}>
                    {grouped && (
                      <>
                        {t("compare.table.split")} <b>{block.split}</b> ·{" "}
                      </>
                    )}
                    <b>{block.sample.num_iters != null ? t("compare.block.iters", { count: count(block.sample.num_iters) }) : t("compare.block.iters_unknown")}</b>
                    {block.sample.train_resolution && (
                      <span>
                        {" · "}
                        {t("compare.block.resolution", { train: size(block.sample.train_resolution), source: size(block.sample.source_resolution) })}
                      </span>
                    )}
                    <span className="faint"> · {t("compare.count", { count: block.rows.length })}</span>
                  </td>
                </tr>
                {block.rows.map((row, index) => (
                  <tr key={row.run_id}>
                    <td className="num faint">{sort.value(row) == null ? "–" : index + 1}</td>
                    <td>
                      <a className="id" href={href({ page: "run", runId: row.run_id })}>{row.run_id}</a>
                      <div className="small faint row" style={{ gap: 6 }}>
                        {row.approach_id}
                        {row.num_iters != null && <span>· {t("compare.row.iters", { count: count(row.num_iters) })}</span>}
                      </div>
                    </td>
                    {!currentSplit && !grouped && <td className="small nowrap">{row.split_id}</td>}
                    <td className="small muted" style={{ maxWidth: 260 }}>
                      <ConfigText approachId={row.approach_id} options={row.options} preset={row.preset} params={row.train_params} compact />
                    </td>
                    {group.metrics.map((metric) => (
                      <td key={metric.key} className={metric.text ? "small nowrap" : sort.key === metric.key ? "num sorted-col" : "num"}>
                        <Best text={metric.format(row)} value={metric.value(row)} cells={block.rows.flatMap((r) => { const v = metric.value(r); return v == null ? [] : [{ text: metric.format(r), value: v }]; })} better={metric.better} />
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

      {chartRows.length > 0 && chartRows.length < rows.length && (
        <p className="small faint">{t("compare.charts.only_iters", { count: count(chartIters) })}</p>
      )}
      {chartRows.length > 0 && <Charts rows={chartRows} splits={splits} grouped={grouped} gpus={gpus} />}

      {rows.length > 1 && (
        <Panel title={t("compare.side.title")} info={t("compare.side.info")}>
          {/* key: đổi cảnh hoặc split thì chọn lại hai run, không giữ run của cảnh cũ. */}
          <SideBySide key={`${currentScene}/${currentSplit}`} rows={rows} />
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
  const video = preferredVideo(videos.data?.items ?? []);
  return (
    <figure>
      {video ? <video ref={register} src={video.url} controls muted preload="metadata" /> : <p className="muted">{t("run.videos.none")}</p>}
      <figcaption>{runId}</figcaption>
    </figure>
  );
}
