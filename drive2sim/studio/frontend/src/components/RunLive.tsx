// Tab "Kết quả" của trang run khi run đang chạy: tiến trình (từng stage, thời gian, log cuối, tải
// GPU, cảnh báo khi stage có vẻ treo) và bảng cấu hình của run (approach, tham số từng stage, máy).
import { AlertTriangle, Check, CircleDashed, Cpu, Database, FileText, Loader2, Settings2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { seg, useApi } from "../api/client";
import type { Approach, RunDetail } from "../api/types";
import { useI18n } from "../i18n";
import { clock } from "./JobProgress";
import { MachineInfo, machineText } from "./RunInfo";
import { ago, duration } from "./ui";

interface LiveGpu {
  index: number;
  name: string;
  util: number;
  memory_used_mb: number;
  memory_total_mb: number;
}

interface RunLiveState {
  active: boolean;
  stage_id: string | null;
  log_tail: string;
  log_idle_s: number | null;
  /** Thanh tqdm cuối trong log stage đang chạy, ví dụ 5/8 đoạn. */
  log_progress: { label: string; done: number; total: number } | null;
  gpus: LiveGpu[];
}

export interface LiveStage {
  stage_id: string;
  status: string;
  cache_hit?: boolean;
  duration_s?: number;
  started_utc?: string | null;
  params?: Record<string, unknown>;
}

export interface LiveProgress {
  stage: string;
  fraction: number | null;
  message: string;
  eta: number | null;
}

/** Log không có dòng mới quá chừng này (giây) và cuối log là lỗi: coi là treo. */
const STALL_WITH_ERROR_S = 180;
/** Log im quá chừng này mà mọi GPU gần như rảnh: cũng coi là treo. */
const STALL_IDLE_S = 900;

/** Đồng hồ cập nhật mỗi giây, để thời gian đã chạy tăng dần không cần tải lại. */
function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function StageIcon({ stage }: { stage: LiveStage }) {
  if (stage.cache_hit && stage.status === "succeeded") return <Database size={16} />;
  if (stage.status === "succeeded") return <Check size={16} />;
  if (stage.status === "failed" || stage.status === "cancelled") return <X size={16} />;
  if (stage.status === "running") return <Loader2 size={16} className="spin" />;
  return <CircleDashed size={16} />;
}

/** Tiến trình của run đang chạy. */
export function RunProgress({
  runId,
  data,
  stages,
  progress,
}: {
  runId: string;
  data: RunDetail;
  stages: LiveStage[];
  progress: LiveProgress | null;
}) {
  const { t, locale } = useI18n();
  const now = useNow();
  const live = useApi<RunLiveState>(`/api/runs/${seg(runId)}/live`, 4000);
  const done = stages.filter((s) => s.status === "succeeded").length;
  const running = stages.find((s) => s.status === "running");
  // Stage báo tiến độ qua engine thì dùng số đó; chưa báo (hoặc mới báo 0) thì đọc thanh tqdm trong log.
  const bar = running && live.data?.stage_id === running.stage_id ? live.data.log_progress : null;
  const barFraction = bar && bar.total > 0 ? bar.done / bar.total : null;
  const reported = running && progress?.stage === running.stage_id ? progress.fraction : null;
  const fraction = reported ? reported : (barFraction ?? reported);
  const overall = stages.length ? (done + (fraction ?? 0)) / stages.length : 0;
  const created = data.created_utc ? Date.parse(data.created_utc) : NaN;
  const since = (iso: string | null | undefined) => {
    const ms = iso ? Date.parse(iso) : NaN;
    return Number.isNaN(ms) ? null : Math.max(0, (now - ms) / 1000);
  };

  const state = live.data;
  const tail = state?.log_tail ?? "";
  const lastLines = tail.trimEnd().split("\n").slice(-8).join("\n");
  const endsInError = /Traceback|Error|Exception|Killed/.test(lastLines);
  const idle = state?.log_idle_s ?? null;
  const gpuIdle = (state?.gpus ?? []).every((g) => g.util < 5);
  const stalled =
    state?.active && state.stage_id && idle != null && ((endsInError && idle > STALL_WITH_ERROR_S) || (gpuIdle && idle > STALL_IDLE_S));

  return (
    <div className="stack">
      {stalled && (
        <div className="alert" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>{t("run.live.stalled_title", { stage: state!.stage_id! })}</strong>
            <div className="detail">
              {t(endsInError ? "run.live.stalled_error" : "run.live.stalled_idle", { time: duration(idle) })}
            </div>
          </div>
        </div>
      )}

      <section className="panel live-panel">
        <div className="live-head">
          <h3>{t("run.live.title")}</h3>
          <span className="live-big dense">{Math.round(overall * 100)}%</span>
          <span className="muted small">{t("run.live.overall", { done, total: stages.length })}</span>
          <span className="muted small" style={{ marginLeft: "auto" }}>
            {Number.isNaN(created) ? "" : t("run.live.elapsed", { time: clock((now - created) / 1000) })}
          </span>
        </div>
        <div className="progress lg" aria-label={t("run.live.title")}>
          <i style={{ width: `${Math.round(overall * 100)}%` }} />
        </div>
        <ol className="live-stages">
          {stages.map((s, index) => {
            const isRunning = s.status === "running";
            const elapsed = isRunning ? since(s.started_utc) : null;
            const reportedOwn = isRunning && progress?.stage === s.stage_id ? progress : null;
            const own =
              isRunning && !reportedOwn?.fraction && barFraction != null
                ? {
                    stage: s.stage_id,
                    fraction: barFraction,
                    message: `${reportedOwn?.message ? `${reportedOwn.message} · ` : ""}${bar!.label} ${bar!.done}/${bar!.total}`,
                    eta: null,
                  }
                : reportedOwn;
            return (
              <li key={s.stage_id} className={`live-stage ${s.status}`}>
                <span className="live-icon"><StageIcon stage={s} /></span>
                <span className="live-name">
                  <span className="muted dense small">{index + 1}.</span> <b>{s.stage_id}</b>
                </span>
                <span className="live-note small">
                  {isRunning ? (
                    <>
                      {own?.message || t("common.status.running")}
                      {own?.fraction != null && (
                        <span className="progress" style={{ marginTop: 6 }}>
                          <i style={{ width: `${Math.round(own.fraction * 100)}%` }} />
                        </span>
                      )}
                    </>
                  ) : s.cache_hit && s.status === "succeeded" ? (
                    t("run.stages.from_cache")
                  ) : (
                    t(`common.status.${s.status}`)
                  )}
                </span>
                <span className="live-time dense small">
                  {isRunning
                    ? [
                        own?.fraction != null ? `${Math.round(own.fraction * 100)}%` : null,
                        elapsed != null ? clock(elapsed) : null,
                        own?.eta != null ? t("eta.left", { time: clock(own.eta) }) : null,
                      ]
                        .filter(Boolean)
                        .join(" · ") || "…"
                    : s.duration_s
                      ? duration(s.duration_s)
                      : s.status === "pending"
                        ? ""
                        : "–"}
                </span>
              </li>
            );
          })}
        </ol>
      </section>

      <div className="grid-2 live-grid">
        <section className="panel live-panel">
          <div className="live-head">
            <h3 className="row" style={{ gap: 8 }}>
              <FileText size={17} />
              {t("run.live.log", { stage: state?.stage_id ?? t("run.logs.engine") })}
            </h3>
            {idle != null && (
              <span className={`small ${stalled ? "warn-ink" : "muted"}`} style={{ marginLeft: "auto" }}>
                {t("run.live.log_idle", { time: ago(now / 1000 - idle, locale) })}
              </span>
            )}
          </div>
          <pre className="log live-log">{tail || t("run.logs.empty")}</pre>
        </section>

        <section className="panel live-panel">
          <div className="live-head">
            <h3 className="row" style={{ gap: 8 }}>
              <Cpu size={17} />
              {t("run.live.gpu")}
            </h3>
          </div>
          {(state?.gpus ?? []).length === 0 ? (
            <p className="muted small">{t(state?.active === false ? "run.live.not_active" : "run.live.no_gpu")}</p>
          ) : (
            <div className="stack" style={{ gap: 12 }}>
              {state!.gpus.map((g) => (
                <div key={g.index} className="gpu-row">
                  <div className="row small" style={{ gap: 8 }}>
                    <b className="dense">GPU {g.index}</b>
                    <span className="muted ellipsis grow">{g.name}</span>
                    <span className="dense">{g.util}%</span>
                  </div>
                  <span className="progress" title={t("run.live.util")}>
                    <i style={{ width: `${g.util}%` }} />
                  </span>
                  <div className="row small muted" style={{ justifyContent: "space-between" }}>
                    <span>{t("run.live.vram")}</span>
                    <span className="dense">
                      {(g.memory_used_mb / 1024).toFixed(1)} / {(g.memory_total_mb / 1024).toFixed(0)} GB
                    </span>
                  </div>
                  <span className="progress vram">
                    <i style={{ width: `${Math.round((g.memory_used_mb / Math.max(g.memory_total_mb, 1)) * 100)}%` }} />
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function formatValue(value: unknown, on: string, off: string): string {
  if (value === true) return on;
  if (value === false) return off;
  if (value === null || value === undefined || value === "") return "–";
  if (typeof value === "number") return Number.isInteger(value) && Math.abs(value) >= 10000 ? value.toLocaleString() : String(value);
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Một mục cấu hình bằng nhãn của file approach khi có, không thì tên gốc. */
function Entry({ approach, name, value }: { approach?: Approach; name: string; value: unknown }) {
  const { t } = useI18n();
  const spec = approach?.options?.[name] ?? approach?.retired_options?.[name];
  const index = spec ? spec.choices.findIndex((choice) => choice === value) : -1;
  const shown = spec?.labels?.[index] ?? formatValue(value, t("run.config.on"), t("run.config.off"));
  return (
    <>
      <dt title={name}>{spec?.label ?? name}</dt>
      <dd>{shown}</dd>
    </>
  );
}

/** Bảng cấu hình của run: thông tin chung, cấu hình approach, tham số từng stage, máy chạy. */
export function RunConfig({ data, stages, approach }: { data: RunDetail; stages: LiveStage[]; approach?: Approach }) {
  const { t, locale } = useI18n();
  const options = (data.approach.options ?? {}) as Record<string, unknown>;
  const lidar = data.lidar_usage ?? [];
  return (
    <section className="panel live-panel">
      <div className="live-head">
        <h3 className="row" style={{ gap: 8 }}>
          <Settings2 size={17} />
          {t("run.config.title")}
        </h3>
      </div>
      <div className="cfg-grid">
        <div className="cfg-card">
          <h4>{t("run.config.run")}</h4>
          <dl className="kv">
            <dt>{t("run.facts.scene")}</dt>
            <dd>{data.scene_id}</dd>
            <dt>{t("run.facts.split")}</dt>
            <dd>{data.split_id}</dd>
            <dt>{t("run.facts.approach")}</dt>
            <dd>{`${data.approach.approach_id} ${data.approach.version}${data.approach.quick ? ` · ${t("run.quick")}` : ""}`}</dd>
            <dt>{t("run.config.level")}</dt>
            <dd>{data.level}</dd>
            <dt>{t("run.config.executor")}</dt>
            <dd>{data.executor}</dd>
            <dt>{t("run.config.created")}</dt>
            <dd>{ago(data.created_utc ?? null, locale)}</dd>
          </dl>
        </div>
        <div className="cfg-card">
          <h4>{t("run.facts.options")}</h4>
          {Object.keys(options).length === 0 ? (
            <p className="muted small">{t("run.config.defaults")}</p>
          ) : (
            <dl className="kv">
              {Object.entries(options).map(([name, value]) => (
                <Entry key={name} approach={approach} name={name} value={value} />
              ))}
            </dl>
          )}
        </div>
        <div className="cfg-card">
          <h4>{t("run.facts.machine")}</h4>
          <p className="small" style={{ margin: "0 0 8px" }}>{machineText(data.env, t("run.facts.no_gpu"))}</p>
          {data.env && (data.env.machines ?? []).length > 0 && (
            <div className="small muted machine-box">
              <MachineInfo env={data.env} />
            </div>
          )}
        </div>
        {stages.map((s) => {
          const params = Object.entries(s.params ?? {});
          return (
            <div key={s.stage_id} className={`cfg-card stage ${s.status}`}>
              <h4>
                <StageIcon stage={s} />
                <span className="ellipsis">{s.stage_id}</span>
                {lidar.includes(s.stage_id) && <span className="tag lidar-tag">{t("run.stages.lidar")}</span>}
              </h4>
              {params.length === 0 ? (
                <p className="muted small">{t("run.config.defaults")}</p>
              ) : (
                <dl className="kv">
                  {params.map(([name, value]) => (
                    <Entry key={name} approach={approach} name={name} value={value} />
                  ))}
                </dl>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
