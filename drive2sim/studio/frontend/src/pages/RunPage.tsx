// Một run: trạng thái và hành động ở đầu trang (huỷ, chạy tiếp, xuất, xem 3D, Kaggle, Colab, xoá);
// làn stage cập nhật trực tiếp qua SSE; các tab kết quả, metric trong lúc train, stage, video, log,
// Kaggle, Colab.
import { Box, Car, Cloud, Gamepad2, Sparkles, CloudDownload, CloudUpload, Download, Power, RefreshCw, RotateCcw, Square, Trash2, Zap, Trophy, Plus } from "lucide-react";
import { useRef, useEffect, useMemo, useState } from "react";
import { asApiError, del, playableUrl, post, seg, useApi } from "../api/client";
import type { ColabStatus, KaggleStatus, List, ReportRow, RunDetail, RunEvent, RunLog, Splats, Video } from "../api/types";
import { LineChart, type Point } from "../components/charts";
import { RunArtifacts } from "../components/RunArtifacts";
import { MachineInfo, machineText } from "../components/RunInfo";
import { RunConfig, RunProgress } from "../components/RunLive";
import { StageRoad } from "../components/StageRoad";
import { ago, Confirm, count, duration, Empty, ErrorBox, fixed, InfoTip, Loading, Note, PageHead, Status, Tabs, useToast } from "../components/ui";
import { ConfigText, paramsLabel, useApproaches } from "../components/ConfigText";
import { SpeedupText } from "../components/SpeedupText";
import { DensifyText } from "../components/DensifyInfo";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { BetterArrow, visibleGroups } from "../metrics";
import { go, href } from "../router";
import { runStatus } from "../runStatus";
import { etaSeconds, trackProgress, type EtaTrack } from "../eta";
import { clock } from "../components/JobProgress";

type Tab = "result" | "metrics" | "stages" | "artifacts" | "videos" | "logs" | "kaggle" | "colab";

/**
 * Theo dõi sự kiện SSE của một run; gọi onDone khi run kết thúc. Kết nối đóng sau "done", và
 * trạng thái từ SSE thắng dữ liệu tải từ API, nên sau khi bấm chạy tiếp, xuất lại hay huỷ phải
 * gọi `restart` để mở lại kết nối với trạng thái mới; không thì trang kẹt ở trạng thái cũ (vẫn
 * hiện "lỗi" và nút "Chạy tiếp", không có nút Huỷ) tới khi tải lại trang.
 */
function useRunEvents(runId: string, onDone: () => void) {
  const [generation, setGeneration] = useState(0);
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState<{ stage: string; fraction: number | null; message: string; eta: number | null } | null>(null);
  const track = useRef<EtaTrack | undefined>(undefined);
  const [metrics, setMetrics] = useState<Record<string, Point[]>>({});

  useEffect(() => {
    const source = new EventSource(`/api/runs/${seg(runId)}/events`);
    const read = (handler: (event: RunEvent) => void) => (raw: MessageEvent) => handler({ ...JSON.parse(raw.data), event: raw.type } as RunEvent);
    source.addEventListener("status", read((e) => setStatuses((s) => ({ ...s, [e.stage_id ?? "__run__"]: e.status ?? "" }))));
    source.addEventListener(
      "progress",
      read((e) => {
        track.current = trackProgress(track.current, e.stage_id ?? "", e.fraction);
        setProgress({ stage: e.stage_id ?? "", fraction: e.fraction ?? null, message: e.message ?? "", eta: etaSeconds(track.current) });
      }),
    );
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
  }, [runId, generation]); // onDone đổi mỗi lần render; mở lại kết nối khi đổi run hoặc restart

  const restart = () => {
    setStatuses({});
    setProgress(null);
    setMetrics({});
    setGeneration((g) => g + 1);
  };
  return { statuses, progress, metrics, restart };
}

export function RunPage({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const approaches = useApproaches();
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
  // Hộp "Tối ưu tiếp" (stage train.nurec_refine): số bước và cách thêm Gaussian.
  const [refine, setRefine] = useState<{ iters: number; strategy: string } | null>(null);
  // Hộp "Chạy trong CARLA" (stage render.carla): lựa chọn lần trước được nhớ trong trình duyệt.
  const [carla, setCarlaState] = useState<CarlaSettings | null>(null);
  const setCarla = (next: CarlaSettings | null) => {
    setCarlaState(next);
    if (next) localStorage.setItem(CARLA_KEY, JSON.stringify(next));
  };
  // Tên đường cong: nhãn dịch nếu có (psnr, loss...), không thì giữ tên metric stage gửi lên.
  const curveLabel = (name: string) => (t(`run.curve.${name}`) === `run.curve.${name}` ? name : t(`run.curve.${name}`));
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

  // Thời gian stage train gồm cả khởi động, lưu checkpoint, đánh giá cuối và render video, không chỉ
  // các bước train: chia ra để người đọc biết phần nào tốn bao lâu (summary.json: train_time_s,
  // eval_time_s; run cũ chưa có eval_time_s thì gộp phần còn lại).
  const trainBreakdown = (total: number | undefined): string | null => {
    const train = data?.summary?.train_time_s;
    if (!train || !total || train > total) return null;
    const evaluation = data?.summary?.eval_time_s;
    if (evaluation != null && train + evaluation <= total) {
      return t("run.stages.breakdown", { train: duration(train), other: duration(total - train - evaluation), eval: duration(evaluation) });
    }
    return t("run.stages.breakdown_old", { train: duration(train), other: duration(total - train) });
  };
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
      live.restart();
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
              <Status value={runStatus(data, live.statuses.__run__)} />
              {data.approach.quick && (
                <span className="tag warn row" style={{ gap: 4 }}>
                  <Zap size={15} />
                  {t("run.quick")}
                  <InfoTip>{t("run.quick_hint")}</InfoTip>
                </span>
              )}
              {data.executor !== "local" && (
                <span className="tag info">
                  <Cloud size={15} />
                  {data.executor === "kaggle" ? "Kaggle" : "Colab"}
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
                <dd>
                  <ConfigText approachId={data.approach.approach_id} options={data.approach.options ?? {}} preset={data.approach.preset} params={data.train_params} />
                  {summary?.speedups && summary.speedups.length > 0 && (
                    <div>
                      <SpeedupText speedups={summary.speedups} />
                    </div>
                  )}
                </dd>
              </div>
              <div>
                <dt>{t("run.facts.densify")}</dt>
                <dd>
                  <DensifyText info={data.densify} />
                </dd>
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
                {/* Đã có file thì nút xuất lại nằm ở stage export trong tab Artifact (hiếm khi cần);
                    ở đây chỉ hiện nút cho file chưa có. */}
                {!splats.data?.ply && (
                  <button className="btn" onClick={() => action(t("run.toast.export", { kind: "PLY" }), `${base}/export`, { kind: "ply" })}>
                    <Download size={19} />
                    {t("run.action.export_ply")}
                  </button>
                )}
                {!splats.data?.preview && (
                  <button className="btn" onClick={() => action(t("run.toast.export", { kind: "preview" }), `${base}/export`, { kind: "preview" })}>
                    <Download size={19} />
                    {t("run.action.export_preview")}
                  </button>
                )}
                {/* run NuRec tạo trước khi có stage video: thêm video đi dọc đoạn đường */}
                {data.approach.approach_id === "nurec" && !data.stages?.some((s) => s.stage_id === "render.nurec_video") && (
                  <button className="btn" onClick={() => action(t("run.toast.export", { kind: "video" }), `${base}/export`, { kind: "video" })} title={t("run.action.export_video_help")}>
                    <Download size={19} />
                    {t("run.action.export_video")}
                  </button>
                )}
                {data.approach.approach_id === "nurec" && data.executor === "local" && !data.stages?.some((s) => s.stage_id === "train.nurec_refine" && Number(s.params?.iters ?? 0) > 0) && (
                  <button className="btn" onClick={() => setRefine({ iters: 30000, strategy: "none" })} title={t("run.action.refine_help")}>
                    <Sparkles size={19} />
                    {t("run.action.refine")}
                  </button>
                )}
                {data.approach.approach_id === "nurec" && data.executor === "local" && (
                  <a className="btn" href={href({ page: "drive", runId })} title={t("run.action.drive_help")}>
                    <Gamepad2 size={19} />
                    {t("run.action.drive")}
                  </a>
                )}
                {data.approach.approach_id === "nurec" && data.executor === "local" && (
                  <button className="btn" onClick={() => setCarlaState(loadCarlaSettings())} title={t("run.action.carla_help")}>
                    <Car size={19} />
                    {t("run.action.carla")}
                  </button>
                )}
              </>
            )}
            <a className="btn ghost" href={href({ page: "compare", sceneId: data.scene_id, splitId: data.split_id })}>
              <Trophy size={18} />
              {t("run.action.compare")}
            </a>
            <a className="btn ghost" href={href({ page: "new", sceneId: data.scene_id, splitId: data.split_id })}>
              <Plus size={18} />
              {t("run.action.new_on_split")}
            </a>
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
          lidar: (data.lidar_usage ?? []).includes(s.stage_id),
          cached: s.cache_hit && s.status === "succeeded",
          note:
            s.status === "running" && live.progress?.stage === s.stage_id
              ? `${live.progress.fraction != null ? `${Math.round(live.progress.fraction * 100)}% ` : ""}${live.progress.message}${
                  live.progress.eta != null ? ` · ${t("eta.left", { time: clock(live.progress.eta) })}` : ""
                }`
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
            ...(data.executor === "colab" ? [{ value: "colab" as Tab, label: "Colab" }] : []),
          ]}
        />
        <div className={`panel-body${tab === "logs" || tab === "stages" ? " flush" : ""}`}>
          {tab === "result" && (
            <div className="stack">
              {active && <RunProgress runId={runId} data={data} stages={stages} progress={live.progress} />}
            {!summary ? (
              active ? null : (
                <Empty title={t("run.result.none")}>
                  <p>{t("run.result.no_eval")}</p>
                </Empty>
              )
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
                          <th>{t("common.camera")}</th>
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
            )}
              <RunConfig data={data} stages={stages} approach={approaches?.[data.approach.approach_id]} />
            </div>
          )}

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
                      <LineChart points={points} label={curveLabel(name)} xLabel={t("run.curve.step")} />
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
                      <b className="grow">{curveLabel(name)}</b>
                      <span className="dense muted">{points[points.length - 1].y.toFixed(4)}</span>
                    </figcaption>
                    <LineChart points={points} label={t(`run.curve.${name}`)} xLabel={t("run.curve.step")} />
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
                          <span className="muted row" style={{ gap: 2, justifyContent: "flex-end" }}>
                            {s.reused_duration_s ? t("run.stages.original", { time: duration(s.reused_duration_s) }) : "–"}
                            <InfoTip>{t("run.stages.original_hint")}</InfoTip>
                          </span>
                        ) : s.stage_id === "eval.standard" && s.status === "succeeded" ? (
                          <span className="muted row" style={{ gap: 2, justifyContent: "flex-end" }}>
                            {t("run.stages.eval_reads")}
                            <InfoTip>{t("run.stages.eval_reads_hint")}</InfoTip>
                          </span>
                        ) : (
                          <>
                            {duration(s.duration_s)}
                            {s.stage_id.startsWith("train.") && trainBreakdown(s.duration_s) && (
                              <div className="small muted" style={{ whiteSpace: "normal", textAlign: "right" }}>
                                {trainBreakdown(s.duration_s)}
                              </div>
                            )}
                          </>
                        )}
                      </td>
                      <td className="num">
                        {s.peak_vram_mb || (s.stage_id.startsWith("train.") && data.summary?.peak_vram_mb) || (
                          <span className="muted" title={t("run.stages.vram_unknown")}>
                            –
                          </span>
                        )}
                      </td>
                      <td className="small muted">{paramsLabel(approaches?.[data.approach.approach_id], (s.params ?? {}) as Record<string, unknown>) || "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === "artifacts" && <RunArtifacts
              runId={runId}
              onReexport={
                status === "succeeded"
                  ? (kind) => action(t("run.toast.export", { kind: kind === "ply" ? "PLY" : "preview" }), `${base}/export`, { kind })
                  : undefined
              }
            />}

          {tab === "videos" &&
            ((videos.data?.items ?? []).length === 0 ? (
              <Empty title={t("run.videos.none")} />
            ) : (
              <div className="videos">
                {videos.data!.items.map((video) => (
                  <figure key={video.url}>
                    {video.ready === false ? (
                      <p className="muted">{t("run.videos.writing")}</p>
                    ) : (
                      <video src={playableUrl(video)} controls preload="metadata" />
                    )}
                    <figcaption>
                      {`${video.stage_id}/${video.name}`}{" "}
                      <a href={video.url} download>
                        {t("run.videos.original")}
                      </a>
                    </figcaption>
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
          {tab === "colab" && <ColabPanel runId={runId} />}
        </div>
      </section>

      {refine && (
        <Confirm
          title={t("run.refine.title")}
          body={
            <div className="stack" style={{ gap: 12 }}>
              <p>{t("run.refine.body")}</p>
              <CarlaNumber label={t("run.refine.iters")} value={refine.iters} min={1000} max={200000} step={1000} onChange={(iters) => setRefine({ ...refine, iters })} />
              <label className="field">
                <span>{t("run.refine.strategy")}</span>
                <select value={refine.strategy} onChange={(e) => setRefine({ ...refine, strategy: e.target.value })}>
                  {["none", "mcmc"].map((s) => (
                    <option key={s} value={s}>
                      {t(`run.refine.strategy.${s}`)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          }
          confirmLabel={t("run.refine.start")}
          onClose={() => setRefine(null)}
          onConfirm={async () => {
            await post(`${base}/export`, { kind: "refine", params: { iters: refine.iters, strategy: refine.strategy } });
            toast({ kind: "ok", title: t("run.toast.export", { kind: "refine" }) });
            live.restart();
            refresh();
          }}
        />
      )}

      {carla && (
        <Confirm
          title={t("run.carla.title")}
          body={
            <div className="stack" style={{ gap: 12 }}>
              <p>{t("run.carla.body")}</p>
              <label className="field">
                <span>{t("run.carla.ego")}</span>
                <select value={carla.ego} onChange={(e) => setCarla({ ...carla, ego: e.target.value })}>
                  {["trajectory", "replay", "autopilot"].map((ego) => (
                    <option key={ego} value={ego}>
                      {t(`run.carla.ego.${ego}`)}
                    </option>
                  ))}
                </select>
              </label>
              <CarlaNumber label={t("run.carla.seconds")} value={carla.seconds} min={0} max={600} step={5} onChange={(seconds) => setCarla({ ...carla, seconds })} />
              <label className="field">
                <span>{t("run.carla.quality")}</span>
                <select value={carla.ratio} onChange={(e) => setCarla({ ...carla, ratio: Number(e.target.value) })}>
                  {[0.25, 0.5, 0.75, 1].map((ratio) => (
                    <option key={ratio} value={ratio}>
                      {t(`run.carla.quality.${ratio * 100}`)}
                    </option>
                  ))}
                </select>
                <span className="hint">{t("run.carla.quality_hint")}</span>
              </label>
              <div className="row wrap" style={{ gap: 12 }}>
                <CarlaNumber label={t("run.carla.video_height")} value={carla.height} min={120} max={1080} step={60} onChange={(height) => setCarla({ ...carla, height })} />
                <CarlaNumber label={t("run.carla.fps")} value={carla.fps} min={5} max={60} step={5} onChange={(fps) => setCarla({ ...carla, fps })} />
              </div>
              <label className="chip-check" title={t("drive.objects_help")}>
                <input type="checkbox" checked={carla.objects} onChange={(e) => setCarla({ ...carla, objects: e.target.checked })} />
                {t("drive.objects")}
              </label>
            </div>
          }
          confirmLabel={t("run.carla.start")}
          onClose={() => setCarla(null)}
          onConfirm={async () => {
            await post(`${base}/export`, { kind: "carla", params: { ego: carla.ego, max_seconds: carla.seconds, resolution_ratio: carla.ratio, video_height: carla.height, fps: carla.fps, objects: carla.objects } });
            toast({ kind: "ok", title: t("run.toast.export", { kind: "CARLA" }) });
            live.restart();
            refresh();
          }}
        />
      )}

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

type CarlaSettings = { ego: string; seconds: number; ratio: number; height: number; fps: number; objects: boolean };
const CARLA_KEY = "d2s.carla.settings";
const CARLA_DEFAULTS: CarlaSettings = { ego: "trajectory", seconds: 30, ratio: 0.5, height: 540, fps: 30, objects: true };

// Lựa chọn lần trước của hộp CARLA; giá trị hỏng hoặc thiếu thì lấy mặc định.
function loadCarlaSettings(): CarlaSettings {
  try {
    return { ...CARLA_DEFAULTS, ...JSON.parse(localStorage.getItem(CARLA_KEY) ?? "{}") };
  } catch {
    return CARLA_DEFAULTS;
  }
}

function CarlaNumber({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))} />
    </label>
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
            <dd className="row" style={{ gap: 4 }}>
              {state.data.status ? <Status value={state.data.status} /> : <span className="muted">{t("run.kaggle.status_unknown")}</span>}
              {state.data.status_error && <InfoTip warn>{state.data.status_error}</InfoTip>}
            </dd>
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

function ColabPanel({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const live = useLive();
  // Mỗi lần hỏi là một lệnh colab exec vài giây trên máy chạy Studio, nên chỉ hỏi khi mở tab, khi
  // một run đổi trạng thái, hoặc khi bấm Làm mới.
  const state = useApi<ColabStatus>(`/api/colab/${seg(runId)}?_=${live.tick}`);
  const job = async (kind: "push" | "pull") => {
    try {
      await post(`/api/colab/${seg(runId)}/${kind}`);
      toast({ kind: "ok", title: t(`run.colab.${kind}_queued`) });
      live.refreshJobs();
    } catch (err) {
      toast({ kind: "bad", title: t(`run.colab.${kind}_queued`), detail: asApiError(err).message });
    }
  };
  const stop = async () => {
    try {
      await post(`/api/colab/${seg(runId)}/stop`);
      toast({ kind: "ok", title: t("run.colab.stopped") });
      state.reload();
    } catch (err) {
      toast({ kind: "bad", title: t("run.colab.stop"), detail: asApiError(err).message });
    }
  };
  const data = state.data;
  return (
    <div className="stack" style={{ gap: 15 }}>
      {state.error && state.error.code !== "colab_failed" && <ErrorBox error={state.error} />}
      {state.error?.code === "colab_failed" && <Note kind="warn">{state.error.message}</Note>}
      {data?.status === "lost" && <Note kind="warn">{data.status_error}</Note>}
      {data && (
        <dl className="facts">
          <div>
            <dt>{t("run.colab.status")}</dt>
            <dd className="row" style={{ gap: 4 }}>
              {data.status ? <Status value={data.status} /> : <span className="muted">{t("run.kaggle.status_unknown")}</span>}
              {data.status_error && data.status !== "lost" && <InfoTip warn>{data.status_error}</InfoTip>}
            </dd>
          </div>
          <div>
            <dt>{t("run.colab.session")}</dt>
            <dd>
              {data.session} ({data.gpu})
            </dd>
          </div>
          <div>
            <dt>{t("run.kaggle.attempt")}</dt>
            <dd>{data.attempt}</dd>
          </div>
          <div>
            <dt>{t("run.kaggle.pushed")}</dt>
            <dd>{ago(data.pushed_utc, locale)}</dd>
          </div>
          <div>
            <dt>{t("run.kaggle.pulled")}</dt>
            <dd>{data.pulled_utc ? ago(data.pulled_utc, locale) : "–"}</dd>
          </div>
        </dl>
      )}
      {data && data.log_tail.length > 0 && <pre className="log">{data.log_tail.join("\n")}</pre>}
      <div className="row wrap">
        <button className="btn" onClick={() => job("push")}>
          <CloudUpload size={19} />
          {t("run.colab.push")}
        </button>
        <button className="btn" onClick={() => job("pull")} disabled={!data}>
          <CloudDownload size={19} />
          {t("run.kaggle.pull")}
        </button>
        <button className="btn" onClick={() => state.reload()} disabled={state.loading}>
          <RefreshCw size={19} className={state.loading ? "spin" : undefined} />
          {t("run.colab.refresh")}
        </button>
        <button className="btn" onClick={() => void stop()} disabled={!data || data.status === "lost"}>
          <Power size={19} />
          {t("run.colab.stop")}
        </button>
        <InfoTip>{t("run.colab.hint")}</InfoTip>
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
