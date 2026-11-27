// Nút cài môi trường ngay trên máy chạy Studio (CONTRACTS mục 11.1, /api/environment/install):
// chạy envs/install_<env>.sh bằng Python đang chạy Studio, hiện bước đang chạy, thời gian, log cuối;
// dừng được, lỗi thì cài tiếp từ bước lỗi. Cài xong phải mở lại Studio để nạp gói mới.
import { Download, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, asApiError, post } from "../api/client";
import { useI18n } from "../i18n";
import { clock } from "./JobProgress";
import { ErrorBox } from "./ui";

interface Install {
  state: "idle" | "running" | "succeeded" | "failed" | "cancelled";
  env: string | null;
  python: string;
  step: number | null;
  steps: number | null;
  label: string | null;
  from_step: number | null;
  started_at: number | null;
  step_started_at: number | null;
  finished_at: number | null;
  log: string;
  log_path: string | null;
}

export function EnvInstall({ env, onDone }: { env: string; onDone?: () => void }) {
  const { t } = useI18n();
  const [install, setInstall] = useState<Install | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const running = install?.state === "running";

  useEffect(() => {
    let stop = false;
    const load = () =>
      api<Install>("/api/environment/install")
        .then((result) => {
          if (stop) return;
          setInstall((before) => {
            if (before?.state === "running" && result.state === "succeeded") onDone?.();
            return result;
          });
        })
        .catch((err) => !stop && setError(asApiError(err)));
    void load();
    const timer = window.setInterval(() => {
      setNow(Date.now() / 1000);
      if (running) void load();
    }, 3000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [running, onDone]);

  const start = async (fromStep: number) => {
    const key =
      env === "nurec" || env === "xsim" ? `env.install.confirm_${env}` : "env.install.confirm";
    if (!window.confirm(t(key, { env, python: install?.python ?? "" }))) return;
    setError(null);
    try {
      setInstall(await post<Install>("/api/environment/install", { env, from_step: fromStep }));
    } catch (err) {
      setError(asApiError(err));
    }
  };
  const cancel = async () => {
    try {
      setInstall(await post<Install>("/api/environment/install/cancel"));
    } catch (err) {
      setError(asApiError(err));
    }
  };

  const step = install?.step ?? 0;
  const steps = install?.steps ?? 0;
  const resumable = (install?.state === "failed" || install?.state === "cancelled") && step > 1;
  const pct = steps ? (100 * Math.max(0, step - 1)) / steps : 0;
  return (
    <div className="stack" style={{ gap: 6 }}>
      {running ? (
        <>
          <div className="row" style={{ gap: 8, alignItems: "center" }}>
            <span className="small grow">
              {step
                ? t("env.install.step", { step, steps, label: install?.label ?? "" })
                : t("env.install.starting")}
            </span>
            <button className="btn sm" type="button" onClick={() => void cancel()}>
              <Square size={14} />
              {t("env.install.cancel")}
            </button>
          </div>
          <div className={steps ? "progress" : "progress indeterminate"}>
            <i style={steps ? { width: `${pct}%` } : undefined} />
          </div>
          <span className="small muted">
            {t("env.install.elapsed", {
              total: clock(now - (install?.started_at ?? now)),
              step: clock(now - (install?.step_started_at ?? now)),
            })}
          </span>
        </>
      ) : (
        <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
          {resumable && (
            <button className="btn sm primary" type="button" onClick={() => void start(step)}>
              <Play size={14} />
              {t("env.install.resume", { step })}
            </button>
          )}
          <button className={`btn sm${resumable ? "" : " primary"}`} type="button" onClick={() => void start(1)}>
            <Download size={14} />
            {t("env.install.start")}
          </button>
          {install?.state === "succeeded" && <span className="small ok-ink">{t("env.install.done")}</span>}
          {install?.state === "failed" && <span className="small bad-ink">{t("env.install.failed", { step })}</span>}
          {install?.state === "cancelled" && <span className="small muted">{t("env.install.cancelled")}</span>}
        </div>
      )}
      {install && install.state !== "idle" && install.log && (
        <details open={install.state === "failed"}>
          <summary className="small" style={{ cursor: "pointer" }}>
            {t("env.install.log", { path: install.log_path ?? "" })}
          </summary>
          <pre className="small" style={{ maxHeight: 220, overflow: "auto", whiteSpace: "pre-wrap" }}>
            {install.log.split("\n").slice(-60).join("\n")}
          </pre>
        </details>
      )}
      <ErrorBox error={error} />
    </div>
  );
}
