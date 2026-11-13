// Thiết lập Kaggle ngay trong trang Chạy mới: kiểm tra lệnh kaggle, khoá API, tài khoản, giờ GPU
// còn lại và code đã push chưa (GET /api/kaggle/setup/status), và nhập khoá API
// (POST /api/kaggle/setup/token). Khoá chỉ gửi lên server chạy trên máy này; server ghi vào thư
// mục cấu hình của Kaggle và không bao giờ trả khoá lại.
import { CheckCircle2, CircleAlert, KeyRound, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { api, asApiError, post, type ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { ErrorBox, InfoTip, useToast } from "./ui";

export interface KaggleSetupStatus {
  ready: boolean;
  cli: { installed: boolean; version: string | null };
  credentials: { source: "env" | "access_token" | "kaggle.json" | "oauth" | null; config_dir: string };
  account: { ok: boolean; username: string | null; error: string | null };
  gpu: { used_h: number | null; remaining_h: number | null; total_h: number | null; refresh_at: string | null } | null;
  code: { ok: boolean; commit: string | null; error: string | null; dirty: boolean };
}

function Check({ state, label, children }: { state: "ok" | "bad" | "warn"; label: string; children: ReactNode }) {
  const Icon = state === "ok" ? CheckCircle2 : state === "warn" ? CircleAlert : XCircle;
  return (
    <li className={`check ${state}`}>
      <Icon size={17} aria-hidden="true" />
      <span className="check-label">{label}</span>
      <span className="check-value">{children}</span>
    </li>
  );
}

/** Ô kiểm tra và nhập khoá Kaggle; gọi `onStatus` mỗi lần có kết quả kiểm tra mới. */
export function KaggleSetup({ onStatus }: { onStatus?: (status: KaggleSetupStatus | null) => void }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const [status, setStatus] = useState<KaggleSetupStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<ApiError | null>(null);
  const [editing, setEditing] = useState(false);
  const [token, setToken] = useState("");
  const [username, setUsername] = useState("");
  const [replace, setReplace] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<ApiError | null>(null);

  const runCheck = async () => {
    setChecking(true);
    setCheckError(null);
    try {
      const result = await api<KaggleSetupStatus>("/api/kaggle/setup/status");
      setStatus(result);
      onStatus?.(result);
      if (!result.credentials.source) setEditing(true);
    } catch (err) {
      setCheckError(asApiError(err));
      onStatus?.(null);
    } finally {
      setChecking(false);
    }
  };
  useEffect(() => {
    void runCheck();
  }, []);

  const legacy = token.trim() !== "" && !token.trim().startsWith("KGAT_");
  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await post("/api/kaggle/setup/token", { token, username: legacy ? username : null, replace });
      setToken("");
      setEditing(false);
      toast({ kind: "ok", title: t("kaggle.setup.saved") });
      await runCheck();
    } catch (err) {
      setSaveError(asApiError(err));
    } finally {
      setSaving(false);
    }
  };

  const gpu = status?.gpu;
  const refresh = gpu?.refresh_at ? new Date(gpu.refresh_at).toLocaleString(locale, { weekday: "short", hour: "2-digit", minute: "2-digit" }) : "";
  return (
    <div className="kaggle-setup">
      <div className="row wrap" style={{ gap: 8 }}>
        <span className={`tag ${status?.ready ? "go" : status ? "warn" : ""}`}>
          {checking ? t("kaggle.setup.checking") : status?.ready ? t("kaggle.setup.ready") : status ? t("kaggle.setup.not_ready") : "–"}
        </span>
        <span className="grow" />
        <button className="btn sm" onClick={() => void runCheck()} disabled={checking}>
          <RefreshCw size={15} className={checking ? "spin" : undefined} />
          {t("kaggle.setup.check")}
        </button>
        <button className="btn sm" onClick={() => setEditing((e) => !e)}>
          <KeyRound size={15} />
          {t("kaggle.setup.key")}
        </button>
      </div>
      <ErrorBox error={checkError} />
      {status && (
        <ul className="checks">
          <Check state={status.cli.installed ? "ok" : "bad"} label={t("kaggle.setup.cli")}>
            {status.cli.installed ? `Kaggle CLI ${status.cli.version}` : <code>pip install kaggle</code>}
          </Check>
          <Check state={status.credentials.source ? "ok" : "bad"} label={t("kaggle.setup.credentials")}>
            {status.credentials.source ? t(`kaggle.setup.source.${status.credentials.source}`) : t("kaggle.setup.no_key")}
            <InfoTip>{t("kaggle.setup.config_dir", { dir: status.credentials.config_dir })}</InfoTip>
          </Check>
          <Check state={status.account.ok ? "ok" : "bad"} label={t("kaggle.setup.account")}>
            {status.account.ok ? status.account.username : (status.account.error ?? "–")}
          </Check>
          <Check state={gpu && (gpu.remaining_h ?? 0) > 0 ? "ok" : "bad"} label={t("kaggle.setup.gpu")}>
            {gpu ? (
              <span className="stack" style={{ gap: 4 }}>
                <span>{t("kaggle.setup.gpu_value", { remaining: gpu.remaining_h?.toFixed(1) ?? "–", total: gpu.total_h?.toFixed(0) ?? "–" })}</span>
                <span className="progress" aria-hidden="true">
                  <i style={{ width: `${Math.round(((gpu.remaining_h ?? 0) / (gpu.total_h || 1)) * 100)}%` }} />
                </span>
                {refresh && <span className="small muted">{t("kaggle.setup.refresh", { when: refresh })}</span>}
              </span>
            ) : (
              "–"
            )}
          </Check>
          <Check state={!status.code.ok ? "bad" : status.code.dirty ? "warn" : "ok"} label={t("kaggle.setup.code")}>
            {status.code.ok ? t("kaggle.setup.commit", { commit: status.code.commit ?? "" }) : status.code.error}
            {status.code.ok && status.code.dirty && <InfoTip>{t("kaggle.setup.dirty")}</InfoTip>}
          </Check>
        </ul>
      )}
      {editing && (
        <form
          className="stack key-form"
          style={{ gap: 10 }}
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <label className="field">
            <span>
              {t("kaggle.setup.token")}
              <InfoTip>{t("kaggle.setup.token_help")}</InfoTip>
            </span>
            <input type="password" autoComplete="off" spellCheck={false} value={token} onChange={(e) => setToken(e.target.value)} placeholder="KGAT_…" />
          </label>
          {legacy && (
            <label className="field">
              <span>{t("kaggle.setup.username")}</span>
              <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="off" />
            </label>
          )}
          {status?.credentials.source && (
            <label className="row small" style={{ gap: 8 }}>
              <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
              {t("kaggle.setup.replace")}
            </label>
          )}
          <div className="row" style={{ gap: 8 }}>
            <button className="btn primary sm" type="submit" disabled={!token.trim() || saving}>
              {t(saving ? "kaggle.setup.saving" : "kaggle.setup.save")}
            </button>
            <a className="small" href="https://www.kaggle.com/settings" target="_blank" rel="noreferrer">
              {t("kaggle.setup.get_key")}
            </a>
          </div>
          <ErrorBox error={saveError} />
        </form>
      )}
    </div>
  );
}
