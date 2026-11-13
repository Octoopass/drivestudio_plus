// Một run: trạng thái và hành động ở đầu trang (huỷ, chạy tiếp, xuất, xem 3D, Kaggle, xoá); làn
// stage cập nhật trực tiếp qua SSE; các tab kết quả, metric trong lúc train, stage, video, log, Kaggle.
import { Box, Cloud, CloudDownload, CloudUpload, Download, RotateCcw, Square, Trash2, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { asApiError, del, post, seg, useApi } from "../api/client";
import type { EnvRecord, KaggleStatus, List, ReportRow, RunDetail, RunEvent, RunLog, Splats, Video } from "../api/types";
import { LineChart, type Point } from "../components/charts";
import { RunArtifacts } from "../components/RunArtifacts";
import { StageRoad } from "../components/StageRoad";
import { ago, Confirm, count, duration, Empty, ErrorBox, fixed, InfoTip, Loading, Note, optionsText, PageHead, Status, Tabs, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { BetterArrow, visibleGroups } from "../metrics";
import { go, href } from "../router";

type Tab = "result" | "metrics" | "stages" | "artifacts" | "videos" | "logs" | "kaggle";

/** Theo dõi sự kiện SSE của một run; gọi onDone khi run kết thúc. */
function useRunEvents(runId: string, onDone: () => void) {
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState<{ stage: string; fraction: number | null; message: string } | null>(null);
  const [metrics, setMetrics] = useState<Record<string, Point[]>>({});

  useEffect(() => {
    const source = new EventSource(`/api/runs/${seg(runId)}/events`);
    const read = (handler: (event: RunEvent) => void) => (raw: MessageEvent) => handler({ ...JSON.parse(raw.data), event: raw.type } as RunEvent);
    source.addEventListener("status", read((e) => setStatuses((s) => ({ ...s, [e.stage_id ?? "__run__"]: e.status ?? "" }))));
    source.addEventListener("progress", read((e) => setProgress({ stage: e.stage_id ?? "", fraction: e.fraction ?? null, message: e.message ?? "" })));
    source.addEventListener(
      "metric",
      read((e) => setMetrics((m) => ({ ...m, [e.name ?? "?"]: [...(m[e.name ?? "?"] ?? []), { x: e.step ?? 0, y: e.value ?? 0 }] }))),
    );
    source.addEventListener(
      "done",
      read(() => {
        source.close();
        setProgress(null);
        onDone();
      }),
    );
    return () => source.close();
  }, [runId]); // onDone đổi mỗi lần render; chỉ mở lại kết nối khi đổi run

  return { statuses, progress, metrics };
}

export function RunPage({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const liveAll = useLive();
  const base = `/api/runs/${seg(runId)}`;
  const run = useApi<RunDetail>(base);
  const log = useApi<RunLog>(`${base}/log`);
  const videos = useApi<List<Video>>(`${base}/videos`);
  const splats = useApi<Splats>(`${base}/splats`);
  // Một dòng của bảng so sánh: cùng cách tính thời gian, số Gaussian, file PLY với trang so sánh.
  const report = useApi<List<ReportRow>>(`/api/report?run_id=${encodeURIComponent(runId)}`);

  const [tab, setTab] = useState<Tab>("result");
  // Đường cong train đọc từ metrics.json: dùng khi không có số liệu trực tiếp (run đã xong).
  const curve = useApi<{ series: Record<string, [number, number][]> }>(tab === "metrics" ? `${base}/curve` : null);
  const curveSeries = Object.entries(curve.data?.series ?? {}).map(
    ([name, rows]) => [name, rows.map(([x, y]) => ({ x, y }))] as [string, Point[]],
  );
  const [logStage, setLogStage] = useState("__engine__");
  const [deleting, setDeleting] = useState(false);
  const refresh = () => {
    run.reload();
    log.reload();
    videos.reload();
    splats.reload();
    report.reload();
  };
  const live = useRunEvents(runId, refresh);
  // Run khác xong hoặc đổi trạng thái (ví dụ việc Kaggle tải về xong) cũng làm mới trang này.
  useEffect(() => {
    if (liveAll.tick) run.reload();
  }, [liveAll.tick]);

  const stages = useMemo(
    () => (run.data?.stages ?? []).map((s) => ({ ...s, status: live.statuses[s.stage_id] ?? s.status })),
    [run.data, live.statuses],
  );
  useEffect(() => {
    if (run.data?.status === "failed") setTab("logs");
  }, [run.data?.status]);

  if (run.error) return <div className="page"><ErrorBox error={run.error} /></div>;
  if (!run.data) return <div className="page"><Loading rows={6} /></div>;
  const data = run.data;
  const status = live.statuses.__run__ ?? data.status;
  const active = status === "running" || status === "pending";
  const failed = stages.filter((s) => s.error);
  const hasSplats = Boolean(splats.data?.preview || splats.data?.ply);

  const action = async (label: string, path: string, body?: unknown) => {
    try {
      await post(path, body);
      toast({ kind: "ok", title: label });
      refresh();
    } catch (err) {
      toast({ kind: "bad", title: label, detail: asApiError(err).message });
    }
  };

  const summary = data.summary;
  const row = report.data?.items[0];
  return (
    <div className="page">
      <PageHead
        title={data.run_id}
        meta={
          <>
            <div className="row wrap" style={{ gap: 12 }}>
              <Status value={status} />
              {data.approach.quick && (
                <span className="tag warn" title={t("run.quick_hint")}>
                  <Zap size={15} />
                  {t("run.quick")}
                </span>
              )}
              {data.executor === "kaggle" && (
                <span className="tag info">
                  <Cloud size={15} />
                  Kaggle
                </span>
              )}
              <span className="small muted">{t("run.meta.created", { when: ago(data.created_utc ?? null, locale) })}</span>
            </div>
            <dl className="facts">
              <div>
                <dt>{t("run.facts.scene")}</dt>
                <dd><a href={href({ page: "scene", sceneId: data.scene_id })}>{data.scene_id}</a></dd>
              </div>
              <div>
                <dt>{t("run.facts.split")}</dt>
                <dd>{data.split_id}</dd>
              </div>
              <div>
                <dt>{t("run.facts.approach")}</dt>
                <dd>{`${data.approach.approach_id} ${data.approach.version}`}</dd>
              </div>
              <div>
                <dt>{t("run.facts.options")}</dt>
                <dd>{optionsText(data.approach.options ?? {}) || "–"}</dd>
              </div>
              <div>
                <dt>{t("run.facts.machine")}</dt>
                <dd className="row" style={{ gap: 2 }}>
                  {machineText(data.env, t("run.facts.no_gpu"))}
                  {(data.env?.machines ?? []).length > 0 && data.env && (
                    <InfoTip>
                      <MachineInfo env={data.env} />
                    </InfoTip>
                  )}
                </dd>
              </div>
            </dl>
          </>
        }
        actions={
          <>
            {!active && (
              <button className="btn ghost icon" onClick={() => setDeleting(true)} title={t("common.action.delete")} aria-label={t("common.action.delete")}>
                <Trash2 size={20} />
              </button>
            )}
            {active && (
              <button className="btn danger" onClick={() => action(t("run.toast.cancel_requested"), `${base}/cancel`)}>
                <Square size={18} />
                {t("run.action.cancel")}
              </button>
            )}
            {(status === "failed" || status === "cancelled") && data.executor === "local" && (
              <button className="btn" onClick={() => action(t("run.toast.resumed"), `${base}/resume`)}>
                <RotateCcw size={19} />
                {t("run.action.resume")}
              </button>
            )}
            {status === "succeeded" && (
              <>
                {/* Run thường đã có cả hai (stage mặc định của approach); khi đó nút chỉ để xuất lại,
                    ví dụ sau khi stage export lỗi hay code export vừa sửa, nên làm mờ đi. */}
                <button
                  className={`btn${splats.data?.ply ? " ghost" : ""}`}
                  title={splats.data?.ply ? t("run.action.reexport_hint") : undefined}
                  onClick={() => action(t("run.toast.export", { kind: "PLY" }), `${base}/export`, { kind: "ply" })}
                >
                  <Download size={19} />
                  {t(splats.data?.ply ? "run.action.reexport_ply" : "run.action.export_ply")}
                </button>
                <button
                  className={`btn${splats.data?.preview ? " ghost" : ""}`}
                  title={splats.data?.preview ? t("run.action.reexport_hint") : undefined}
                  onClick={() => action(t("run.toast.export", { kind: "preview" }), `${base}/export`, { kind: "preview" })}
                >
                  <Download size={19} />
                  {t(splats.data?.preview ? "run.action.reexport_preview" : "run.action.export_preview")}
                </button>
              </>
            )}
            <a className={`btn${hasSplats ? " primary" : ""}`} href={href({ page: "viewer", runId })} aria-disabled={!hasSplats}>
              <Box size={19} />
              {t("run.action.view3d")}
            </a>
          </>
        }
      />

      <StageRoad
        stages={stages.map((s) => ({
          stage_id: s.stage_id,
          status: s.status,
          cached: s.cache_hit && s.status === "succeeded",
          note:
            s.status === "running" && live.progress?.stage === s.stage_id
              ? `${live.progress.fraction != null ? `${Math.round(live.progress.fraction * 100)}% ` : ""}${live.progress.message}`
              : s.cache_hit && s.status === "succeeded"
                ? s.reused_duration_s
                  ? t("run.stages.from_cache_time", { time: duration(s.reused_duration_s) })
                  : t("run.stages.from_cache")
                : s.duration_s
                  ? `${t(`common.status.${s.status}`)}, ${duration(s.duration_s)}`
                  : undefined,
          fraction: live.progress?.stage === s.stage_id ? live.progress.fraction : null,
        }))}
      />

      {failed.map((s) => (
        <div className="alert" role="alert" key={s.stage_id}>
          <div>
            <strong>{s.error?.message}</strong>
            <div className="detail">{t("run.error.where", { log: String(s.error?.details?.log ?? "d2s/log.txt") })}</div>
          </div>
        </div>
      ))}

      <section className="panel">
        <Tabs<Tab>
          value={tab}
          onChange={setTab}
          items={[
            { value: "result", label: t("run.tab.result") },
            { value: "metrics", label: t("run.tab.metrics"), badge: Object.keys(live.metrics).length || null },
            { value: "stages", label: t("run.tab.stages"), badge: stages.length },
            { value: "artifacts", label: t("run.tab.artifacts") },
            { value: "videos", label: t("run.tab.videos"), badge: videos.data?.total || null },
            { value: "logs", label: t("run.tab.logs") },
            ...(data.executor === "kaggle" ? [{ value: "kaggle" as Tab, label: "Kaggle" }] : []),
          ]}
        />
        <div className={`panel-body${tab === "logs" || tab === "stages" ? " flush" : ""}`}>
          {tab === "result" &&
            (!summary ? (
              <Empty title={t("run.result.none")}>
                <p>{t(active ? "run.result.wait" : "run.result.no_eval")}</p>
              </Empty>
            ) : (
              <div className="stack">
                {row &&
                  visibleGroups([row]).map((group) => (
                    <section key={group.id} className="stack" style={{ gap: 8 }}>
                      <h3 className="small muted row" style={{ gap: 2 }}>
                        {t(group.label)}
                        <InfoTip>
                          {t(group.hint)}
                          {group.id === "quality" &&
                            " " +
                              t("run.result.protocol", {
                                net: summary.protocol.lpips.net,
                                package: summary.protocol.lpips.package,
                                version: summary.protocol.lpips.package_version,
                                resolution: summary.protocol.resolution,
                              })}
                        </InfoTip>
                      </h3>
                      <div className="panel">
                        <div className="stat-row">
                          {group.metrics.map((metric) => (
                            <Stat
                              key={metric.key}
                              better={metric.better}
                              label={t(metric.label)}
                              hint={metric.hint ? t(metric.hint, { resolution: summary.render_resolution ?? "–" }) : undefined}
                              value={metric.format(row)}
                              unit={metric.unit}
                            />
                          ))}
                        </div>
                      </div>
                    </section>
                  ))}
                {summary.per_camera && Object.keys(summary.per_camera).length > 0 && (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>camera</th>
                          <th className="num">PSNR</th>
                          <th className="num">SSIM</th>
                          <th className="num">LPIPS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(summary.per_camera as Record<string, { psnr: number; ssim: number; lpips: number }>).map(([camera, m]) => (
                          <tr key={camera}>
                            <td>{camera}</td>
                            <td className="num">{fixed(m.psnr)}</td>
                            <td className="num">{fixed(m.ssim, 3)}</td>
                            <td className="num">{fixed(m.lpips, 3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}

          {tab === "metrics" &&
            (Object.keys(live.metrics).length === 0 ? (
              curveSeries.length > 0 ? (
                <div className="grid-2">
                  {curveSeries.map(([name, points]) => (
                    <figure key={name} style={{ margin: 0 }}>
                      <figcaption className="row">
                        <b className="grow">{t(`run.curve.${name}`)}</b>
                        <span className="dense muted">{name === "gaussians" ? count(points[points.length - 1].y) : points[points.length - 1].y.toFixed(name === "loss" ? 4 : 2)}</span>
                      </figcaption>
                      <LineChart points={points} label={t(`run.curve.${name}`)} xLabel={t("run.curve.step")} />
                    </figure>
                  ))}
                </div>
              ) : (
                <Empty title={t("run.metrics.none")}>
                  <p>{t("run.metrics.hint")}</p>
                </Empty>
              )
            ) : (
              <div className="grid-2">
                {Object.entries(live.metrics).map(([name, points]) => (
                  <figure key={name} style={{ margin: 0 }}>
                    <figcaption className="row">
                      <b className="grow">{name}</b>
                      <span className="dense muted">{points[points.length - 1].y.toFixed(4)}</span>
                    </figcaption>
                    <LineChart points={points} label={name} xLabel="step" />
                  </figure>
                ))}
              </div>
            ))}

          {tab === "stages" && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("run.stages.stage")}</th>
                    <th>{t("run.stages.status")}</th>
                    <th>{t("run.stages.cache")}</th>
                    <th className="num">{t("run.stages.seconds")}</th>
                    <th className="num">{t("run.stages.vram")}</th>
                    <th>{t("run.stages.params")}</th>
                  </tr>
                </thead>
                <tbody>
                  {stages.map((s) => (
                    <tr key={s.stage_id}>
                      <td className="id">{s.stage_id}</td>
                      <td><Status value={s.status} /></td>
                      <td className="small muted">{s.cache_hit ? t("run.stages.reused") : ""}</td>
                      <td className="num">
                        {s.cache_hit ? (
                          <span className="muted" title={t("run.stages.original_hint")}>
                            {s.reused_duration_s ? t("run.stages.original", { time: duration(s.reused_duration_s) }) : "–"}
                          </span>
                        ) : (
                          duration(s.duration_s)
                        )}
                      </td>
                      <td className="num">{s.peak_vram_mb || "–"}</td>
                      <td className="small muted">{optionsText((s.params ?? {}) as Record<string, unknown>) || "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === "artifacts" && <RunArtifacts runId={runId} />}

          {tab === "videos" &&
            ((videos.data?.items ?? []).length === 0 ? (
              <Empty title={t("run.videos.none")} />
            ) : (
              <div className="videos">
                {videos.data!.items.map((video) => (
                  <figure key={video.url}>
                    <video src={video.url} controls preload="metadata" />
                    <figcaption>{`${video.stage_id}/${video.name}`}</figcaption>
                  </figure>
                ))}
              </div>
            ))}

          {tab === "logs" && (
            <>
              <div className="log-tabs">
                <Segmented
                  stages={["__engine__", ...Object.keys(log.data?.stages ?? {})]}
                  value={logStage}
                  onChange={setLogStage}
                  engineLabel={t("run.logs.engine")}
                />
                <button className="btn sm ghost" style={{ marginLeft: "auto" }} onClick={log.reload}>
                  {t("run.logs.reload")}
                </button>
              </div>
              <pre className="log">
                {(logStage === "__engine__" ? log.data?.engine : log.data?.stages[logStage]) || t("run.logs.empty")}
              </pre>
            </>
          )}

          {tab === "kaggle" && <KagglePanel runId={runId} />}
        </div>
      </section>

      {deleting && (
        <Confirm
          title={t("run.delete.title", { run: runId })}
          body={t("run.delete.body")}
          items={[data.run_dir]}
          confirmLabel={t("common.action.delete")}
          danger
          onClose={() => setDeleting(false)}
          onConfirm={async () => {
            await del(base);
            toast({ kind: "ok", title: t("run.delete.done"), detail: runId });
            go({ page: "runs" });
          }}
        />
      )}
    </div>
  );
}

function Segmented({ stages, value, onChange, engineLabel }: { stages: string[]; value: string; onChange: (v: string) => void; engineLabel: string }) {
  return (
    <div className="segmented" role="group">
      {stages.map((stage) => (
        <button key={stage} aria-pressed={stage === value} onClick={() => onChange(stage)}>
          {stage === "__engine__" ? engineLabel : stage}
        </button>
      ))}
    </div>
  );
}

function KagglePanel({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const live = useLive();
  const state = useApi<KaggleStatus>(`/api/kaggle/${seg(runId)}?_=${live.tick}`);
  const job = async (kind: "push" | "pull") => {
    try {
      await post(`/api/kaggle/${seg(runId)}/${kind}`);
      toast({ kind: "ok", title: t(`run.kaggle.${kind}_queued`) });
      live.refreshJobs();
    } catch (err) {
      toast({ kind: "bad", title: t(`run.kaggle.${kind}_queued`), detail: asApiError(err).message });
    }
  };
  return (
    <div className="stack" style={{ gap: 15 }}>
      {state.error && state.error.code !== "kaggle_failed" && <ErrorBox error={state.error} />}
      {state.error?.code === "kaggle_failed" && <Note kind="warn">{state.error.message}</Note>}
      {state.data && (
        <dl className="facts">
          <div>
            <dt>{t("run.kaggle.status")}</dt>
            <dd><Status value={state.data.status} /></dd>
          </div>
          <div>
            <dt>{t("run.kaggle.kernel")}</dt>
            <dd>
              <a href={`https://www.kaggle.com/code/${state.data.kernel}`} target="_blank" rel="noreferrer">{state.data.kernel}</a>
            </dd>
          </div>
          <div>
            <dt>{t("run.kaggle.attempt")}</dt>
            <dd>{state.data.attempt}</dd>
          </div>
          <div>
            <dt>{t("run.kaggle.pushed")}</dt>
            <dd>{ago(state.data.pushed_utc, locale)}</dd>
          </div>
          <div>
            <dt>{t("run.kaggle.pulled")}</dt>
            <dd>{state.data.pulled_utc ? ago(state.data.pulled_utc, locale) : "–"}</dd>
          </div>
        </dl>
      )}
      <div className="row wrap">
        <button className="btn" onClick={() => job("push")}>
          <CloudUpload size={19} />
          {t("run.kaggle.push")}
        </button>
        <button className="btn" onClick={() => job("pull")} disabled={!state.data}>
          <CloudDownload size={19} />
          {t("run.kaggle.pull")}
        </button>
        <InfoTip>{t("run.kaggle.hint")}</InfoTip>
      </div>
    </div>
  );
}

/** Một mục cấu hình số liệu; ``hint`` hiện khi rê chuột vào nhãn. */
function Stat({ label, value, unit, hint, better }: { label: string; value: string; unit?: string; hint?: string; better?: "high" | "low" }) {
  return (
    <div className="stat">
      <div className="label">
        <span className="ellipsis">{label}</span>
        <BetterArrow better={better} />
        {hint && <InfoTip>{hint}</InfoTip>}
      </div>
      <div className="value">
        {value}
        {unit && value !== "–" && <small>{unit}</small>}
      </div>
    </div>
  );
}

/** GPU của các máy đã chạy run, ví dụ "Tesla T4 · 15 GB"; "–" khi run chưa chạy. */
function machineText(env: EnvRecord | null, noGpu: string): string {
  const machines = env?.machines ?? [];
  if (!machines.length) return "–";
  const names = [...new Set(machines.map((m) => (m.gpu ? `${(m.gpu_count ?? 0) > 1 ? `${m.gpu_count}× ` : ""}${m.gpu}` : noGpu)))];
  return names.join(" + ");
}

/** Nội dung icon giải thích: cấu hình từng máy đã chạy run và phiên bản gói. */
function MachineInfo({ env }: { env: EnvRecord }) {
  const { t, locale } = useI18n();
  const machines = env.machines ?? [];
  return (
    <span className="stack" style={{ gap: 10 }}>
      {machines.length > 1 && <span>{t("run.machine.several", { count: machines.length })}</span>}
      {machines.map((m, index) => (
        <span key={index} className="machine">
          {machines.length > 1 && <b>{t("run.machine.since", { when: ago(m.first_seen_utc ?? null, locale) })}</b>}
          <span>GPU: {m.gpu ? `${(m.gpu_count ?? 0) > 1 ? `${m.gpu_count}× ` : ""}${m.gpu}${m.gpu_memory_mb ? `, ${(m.gpu_memory_mb / 1024).toFixed(0)} GB` : ""}` : t("run.facts.no_gpu")}</span>
          {m.gpu_driver && <span>{t("run.machine.driver")}: {m.gpu_driver}</span>}
          <span>CPU: {m.cpu ?? "–"}{m.cpu_cores ? `, ${t("run.machine.threads", { count: m.cpu_cores })}` : ""}</span>
          <span>RAM: {m.ram_gb != null ? `${m.ram_gb} GB` : "–"}</span>
          {m.os && <span>{t("run.machine.os")}: {m.os}</span>}
        </span>
      ))}
      <span className="muted">
        Python {env.python}, PyTorch {env.pytorch ?? "–"}, CUDA {env.cuda ?? "–"}, gsplat {env.gsplat ?? "–"}
      </span>
    </span>
  );
}
