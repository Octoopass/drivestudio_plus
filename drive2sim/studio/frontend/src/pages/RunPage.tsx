// Một run: trạng thái và hành động ở đầu trang (huỷ, chạy tiếp, xuất, xem 3D, Kaggle, xoá); làn
// stage cập nhật trực tiếp qua SSE; các tab kết quả, metric trong lúc train, stage, video, log, Kaggle.
import { Box, Cloud, CloudDownload, CloudUpload, Download, RotateCcw, Square, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { asApiError, del, post, seg, useApi } from "../api/client";
import type { KaggleStatus, List, RunDetail, RunEvent, RunLog, Splats, Video } from "../api/types";
import { LineChart, type Point } from "../components/charts";
import { StageRoad } from "../components/StageRoad";
import { ago, Confirm, duration, Empty, ErrorBox, fixed, Loading, Note, optionsText, PageHead, Status, Tabs, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { go, href } from "../router";

type Tab = "result" | "metrics" | "stages" | "videos" | "logs" | "kaggle";

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
  const [tab, setTab] = useState<Tab>("result");
  const [logStage, setLogStage] = useState("__engine__");
  const [deleting, setDeleting] = useState(false);
  const refresh = () => {
    run.reload();
    log.reload();
    videos.reload();
    splats.reload();
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
  // summary.json có thể không ghi thời gian train và VRAM; lấy từ stage train của run.
  const train = stages.find((s) => s.stage_id.startsWith("train."));
  const trainTime = summary?.train_time_s ?? train?.duration_s ?? null;
  const peakVram = summary?.peak_vram_mb ?? Math.max(0, ...stages.map((s) => s.peak_vram_mb ?? 0));
  return (
    <div className="page">
      <PageHead
        title={data.run_id}
        meta={
          <>
            <div className="row wrap" style={{ gap: 10 }}>
              <Status value={status} />
              {data.executor === "kaggle" && (
                <span className="tag info">
                  <Cloud size={12} />
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
            </dl>
          </>
        }
        actions={
          <>
            {!active && (
              <button className="btn ghost icon" onClick={() => setDeleting(true)} title={t("common.action.delete")} aria-label={t("common.action.delete")}>
                <Trash2 size={16} />
              </button>
            )}
            {active && (
              <button className="btn danger" onClick={() => action(t("run.toast.cancel_requested"), `${base}/cancel`)}>
                <Square size={14} />
                {t("run.action.cancel")}
              </button>
            )}
            {(status === "failed" || status === "cancelled") && data.executor === "local" && (
              <button className="btn" onClick={() => action(t("run.toast.resumed"), `${base}/resume`)}>
                <RotateCcw size={15} />
                {t("run.action.resume")}
              </button>
            )}
            {status === "succeeded" && (
              <>
                <button className="btn" onClick={() => action(t("run.toast.export", { kind: "PLY" }), `${base}/export`, { kind: "ply" })}>
                  <Download size={15} />
                  {t("run.action.export_ply")}
                </button>
                <button className="btn" onClick={() => action(t("run.toast.export", { kind: "preview" }), `${base}/export`, { kind: "preview" })}>
                  <Download size={15} />
                  {t("run.action.export_preview")}
                </button>
              </>
            )}
            <a className={`btn${hasSplats ? " primary" : ""}`} href={href({ page: "viewer", runId })} aria-disabled={!hasSplats}>
              <Box size={15} />
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
                ? t("run.stages.from_cache")
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
                <div className="panel">
                  <div className="stat-row">
                    <div className="stat">
                      <div className="label">PSNR</div>
                      <div className="value">{fixed(summary.overall.psnr)}<small>dB</small></div>
                    </div>
                    <div className="stat">
                      <div className="label">SSIM</div>
                      <div className="value">{fixed(summary.overall.ssim, 3)}</div>
                    </div>
                    <div className="stat">
                      <div className="label">LPIPS</div>
                      <div className="value">{fixed(summary.overall.lpips, 3)}</div>
                    </div>
                    {summary.overall.human_psnr != null && (
                      <div className="stat">
                        <div className="label">{t("run.result.human")}</div>
                        <div className="value">{fixed(summary.overall.human_psnr)}</div>
                      </div>
                    )}
                    {summary.overall.vehicle_psnr != null && (
                      <div className="stat">
                        <div className="label">{t("run.result.vehicle")}</div>
                        <div className="value">{fixed(summary.overall.vehicle_psnr)}</div>
                      </div>
                    )}
                    <div className="stat">
                      <div className="label">{t("run.result.train_time")}</div>
                      <div className="value">{duration(trainTime)}</div>
                    </div>
                    <div className="stat">
                      <div className="label">{t("run.result.vram")}</div>
                      <div className="value">{peakVram ? (peakVram / 1024).toFixed(1) : "–"}<small>GB</small></div>
                    </div>
                  </div>
                </div>
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
                <p className="muted small">
                  {t("run.result.protocol", {
                    net: summary.protocol.lpips.net,
                    package: summary.protocol.lpips.package,
                    version: summary.protocol.lpips.package_version,
                    resolution: summary.protocol.resolution,
                  })}
                </p>
              </div>
            ))}

          {tab === "metrics" &&
            (Object.keys(live.metrics).length === 0 ? (
              <Empty title={t("run.metrics.none")}>
                <p>{t("run.metrics.hint")}</p>
              </Empty>
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
                      <td className="num">{duration(s.duration_s)}</td>
                      <td className="num">{s.peak_vram_mb || "–"}</td>
                      <td className="small muted">{optionsText((s.params ?? {}) as Record<string, unknown>) || "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

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
    <div className="stack" style={{ gap: 12 }}>
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
          <CloudUpload size={15} />
          {t("run.kaggle.push")}
        </button>
        <button className="btn" onClick={() => job("pull")} disabled={!state.data}>
          <CloudDownload size={15} />
          {t("run.kaggle.pull")}
        </button>
      </div>
      <p className="muted small">{t("run.kaggle.hint")}</p>
    </div>
  );
}
