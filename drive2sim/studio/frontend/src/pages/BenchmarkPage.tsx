// Benchmark: xếp hạng các run trong từng nhóm của một file benchmarks/<tên>.yaml (CONTRACTS mục 6.5).
// - Chọn file định nghĩa ở trên; kết quả do `GET /api/benchmarks/<tên>` dựng, giống `d2s benchmark`.
// - Mỗi nhóm (track, ô, cảnh, split) là một bảng xếp theo Avg; hàng đầu là người thắng, kèm bước đã
//   quyết định (Avg, R2, R3, thời gian train). Run ngang run tốt nhất đánh dấu ≈.
// - Cột "chưa đo" là dấu –; export gap "n/a" khi method không xuất được .ply (khác chưa đo).
// - Cột chi phí mỗi 100 m, ATE chỉ hiện ở track B; PSNR pinhole và fisheye chỉ hiện khi có số.
import { AlertTriangle, Trophy } from "lucide-react";
import type { ReactNode } from "react";
import { useApi } from "../api/client";
import type { BenchmarkGroup, BenchmarkResult, BenchmarkRunRow } from "../api/generated/benchresult";
import type { List } from "../api/types";
import { duration, Empty, ErrorBox, fixed, InfoTip, Loading, PageHead, Panel } from "../components/ui";
import { useI18n } from "../i18n";
import { go, href } from "../router";

interface Definition {
  name: string;
}

/** Số đoạn hỏng của một run: "0", hoặc "pinhole 1 · fisheye 2" khi có hai nhóm; "–" khi chưa đo. */
function segmentsText(row: BenchmarkRunRow, t: (key: string) => string): string {
  if (!row.frames_measured) return "–";
  const entries = Object.entries(row.bad_segments ?? {});
  if (entries.length === 0) return "0";
  if (entries.length === 1 && entries[0][0] === "all") return String(entries[0][1].length);
  return entries.map(([group, list]) => `${t(`benchmark.camera.${group}`)} ${list.length}`).join(" · ");
}

function GroupTable({ group }: { group: BenchmarkGroup }) {
  const { t } = useI18n();
  const rows = group.rows;
  const trackB = group.bench_track === "B";
  const hasCameraGroups = rows.some((r) => r.psnr_pinhole != null || r.psnr_fisheye != null);
  const hasAte = rows.some((r) => r.ate_rmse_m != null);
  const winner = rows.find((r) => r.run_id === group.winner_run_id);
  const title = `${t("benchmark.group.track", { track: group.bench_track })}${group.cell ? ` · ${group.cell}` : ""}`;

  const headers: { label: string; hint?: string; num?: boolean }[] = [
    { label: "#", num: true },
    { label: t("runs.table.run") },
    { label: t("benchmark.col.approach") },
    { label: "PSNR", hint: t("benchmark.hint.psnr"), num: true },
    { label: "Avg", hint: t("benchmark.hint.avg"), num: true },
    ...(hasCameraGroups
      ? [
          { label: t("benchmark.col.pinhole"), hint: t("benchmark.hint.camera_group"), num: true },
          { label: t("benchmark.col.fisheye"), hint: t("benchmark.hint.camera_group"), num: true },
        ]
      : []),
    { label: "R2", hint: t("benchmark.hint.r2"), num: true },
    { label: "R3", hint: t("benchmark.hint.r3"), num: true },
    { label: "AbsRel", hint: t("benchmark.hint.depth"), num: true },
    { label: t("benchmark.col.segments"), hint: t("benchmark.hint.segments"), num: true },
    ...(trackB
      ? [
          { label: t("benchmark.col.distance"), num: true },
          { label: t("benchmark.col.train_100m"), hint: t("benchmark.hint.per_100m"), num: true },
          { label: t("benchmark.col.ply_100m"), hint: t("benchmark.hint.per_100m"), num: true },
        ]
      : []),
    ...(hasAte ? [{ label: "ATE (m)", hint: t("benchmark.hint.ate"), num: true }] : []),
    { label: t("benchmark.col.train"), hint: t("benchmark.hint.train"), num: true },
    { label: "GPU" },
  ];

  const cell = (value: number | null | undefined, digits = 2): ReactNode =>
    value == null ? <span className="faint">–</span> : fixed(value, digits);

  return (
    <Panel
      title={title}
      sub={`${group.scene_id} · ${group.split_id}`}
      info={t("benchmark.group.info")}
    >
      <div className="stack" style={{ gap: 10 }}>
        <div className="row wrap" style={{ gap: 10, alignItems: "center" }}>
          {winner ? (
            <span>
              <Trophy size={16} style={{ verticalAlign: "-2px" }} /> {t("benchmark.group.winner")}:{" "}
              <a href={href({ page: "run", runId: winner.run_id })}>
                <b>{winner.run_id}</b>
              </a>{" "}
              <span className="faint">({winner.approach_id})</span>
              {group.decided_by && <span className="tag">{t(`benchmark.decided.${group.decided_by}`)}</span>}
            </span>
          ) : (
            <span className="faint">{t("benchmark.group.no_winner")}</span>
          )}
        </div>
        {(group.warnings ?? []).length > 0 && (
          <div className="alert warn" role="status">
            <AlertTriangle size={20} />
            <ul>
              {(group.warnings ?? []).map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="table-wrap">
          <table className="centered">
            <thead>
              <tr>
                {headers.map((h) => (
                  <th key={h.label} className={h.num ? "num" : undefined}>
                    <span className="th-metric">
                      {h.label}
                      {h.hint && <InfoTip>{h.hint}</InfoTip>}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isWinner = row.run_id === group.winner_run_id;
                return (
                  <tr key={row.run_id}>
                    <td className="num">
                      {row.rank}
                      {row.tied_with_best && row.rank > 1 && (
                        <span title={t("benchmark.tied")} aria-label={t("benchmark.tied")}>
                          {" "}
                          ≈
                        </span>
                      )}
                    </td>
                    <td>
                      <a href={href({ page: "run", runId: row.run_id })} className={isWinner ? "strong" : undefined}>
                        {row.run_id}
                      </a>
                    </td>
                    <td>{row.approach_id}</td>
                    <td className="num">{isWinner ? <strong>{cell(row.psnr)}</strong> : cell(row.psnr)}</td>
                    <td className="num">{cell(row.avg_error, 4)}</td>
                    {hasCameraGroups && (
                      <>
                        <td className="num">{cell(row.psnr_pinhole)}</td>
                        <td className="num">{cell(row.psnr_fisheye)}</td>
                      </>
                    )}
                    <td className="num" title={(row.shift_cameras ?? []).join(", ") || undefined}>
                      {cell(row.shift_psnr_cc)}
                    </td>
                    <td className="num">
                      {row.export_applicable ? cell(row.export_gap_db) : <span className="faint" title={t("benchmark.not_applicable")}>n/a</span>}
                    </td>
                    <td className="num">{cell(row.depth?.abs_rel, 3)}</td>
                    <td className="num">{segmentsText(row, t)}</td>
                    {trackB && (
                      <>
                        <td className="num">{cell(row.distance_m, 0)}</td>
                        <td className="num">{cell(row.train_s_per_100m, 0)}</td>
                        <td className="num">{cell(row.ply_mb_per_100m, 1)}</td>
                      </>
                    )}
                    {hasAte && <td className="num">{cell(row.ate_rmse_m, 2)}</td>}
                    <td className="num">{duration(row.train_time_s)}</td>
                    <td>{row.gpu ?? <span className="faint">–</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}

export function BenchmarkPage({ name }: { name?: string }) {
  const { t } = useI18n();
  const list = useApi<List<Definition>>("/api/benchmarks");
  const names = (list.data?.items ?? []).map((item) => item.name);
  // Chưa chọn thì mở file đầu tiên; URL ghi tên đã chọn để tải lại vẫn giữ.
  const current = name ?? names[0];
  const result = useApi<BenchmarkResult>(current ? `/api/benchmarks/${encodeURIComponent(current)}` : null);

  return (
    <div className="page">
      <PageHead title={t("benchmark.page.title")} info={t("benchmark.page.lede")} />
      <ErrorBox error={list.error} />
      {!list.data && !list.error && <Loading rows={3} />}
      {list.data && names.length === 0 && (
        <section className="panel">
          <Empty title={t("benchmark.empty.title")}>
            <p>{t("benchmark.empty.text")}</p>
          </Empty>
        </section>
      )}
      {names.length > 0 && (
        <div className="row wrap">
          <label className="field">
            <span>{t("benchmark.field.file")}</span>
            <select value={current ?? ""} onChange={(e) => go({ page: "benchmark", name: e.target.value })}>
              {names.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        </div>
      )}
      <ErrorBox error={result.error} />
      {current && !result.data && !result.error && <Loading rows={4} />}
      {result.data && (
        <>
          <div className="row wrap" style={{ gap: 10, alignItems: "center" }}>
            <span>
              {t("benchmark.threshold")}: <b>{fixed(result.data.tie_threshold_db, 2)} dB</b>{" "}
              <span className="tag">{t(`benchmark.threshold.${result.data.tie_threshold_source}`)}</span>
            </span>
          </div>
          {(result.data.warnings ?? []).length > 0 && (
            <div className="alert warn" role="status">
              <AlertTriangle size={20} />
              <ul>
                {(result.data.warnings ?? []).map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          {result.data.groups.map((group, index) => (
            <GroupTable key={`${group.bench_track}-${group.cell}-${group.scene_id}-${group.split_id}-${index}`} group={group} />
          ))}
        </>
      )}
    </div>
  );
}
