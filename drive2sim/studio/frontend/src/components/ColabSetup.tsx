// Thiết lập Colab ngay trong trang Chạy mới: kiểm tra lệnh colab, đăng nhập Google, compute unit,
// file SMPL trên máy và code đã push chưa (GET /api/colab/setup/status). Đăng nhập không làm được từ
// trang này: lệnh colab mở trang đồng ý của Google từ terminal, nên ô này chỉ chỉ lệnh cần chạy.
import { CheckCircle2, CircleAlert, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { api, asApiError, type ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { ErrorBox, InfoTip } from "./ui";

export interface ColabSetupStatus {
  ready: boolean;
  cli: { installed: boolean; version: string | null };
  account: { ok: boolean; error: string | null; hint: string | null };
  gpu: { sessions: { name: string; hardware: string }[]; balance: number | null; rate_per_h: number | null; choices: string[] };
  smpl: { found: boolean; path: string | null };
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

/** Ô kiểm tra Colab; gọi `onStatus` mỗi lần có kết quả kiểm tra mới. */
export function ColabSetup({ onStatus, needsSmpl }: { onStatus?: (status: ColabSetupStatus | null) => void; needsSmpl?: boolean }) {
  const { t } = useI18n();
  const [status, setStatus] = useState<ColabSetupStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<ApiError | null>(null);

  const runCheck = async () => {
    setChecking(true);
    setCheckError(null);
    try {
      const result = await api<ColabSetupStatus>("/api/colab/setup/status");
      setStatus(result);
      onStatus?.(result);
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

  const gpu = status?.gpu;
  return (
    <div className="kaggle-setup">
      <div className="row wrap" style={{ gap: 8 }}>
        <span className={`tag ${status?.ready ? "go" : status ? "warn" : ""}`}>
          {checking ? t("colab.setup.checking") : status?.ready ? t("colab.setup.ready") : status ? t("colab.setup.not_ready") : "–"}
        </span>
        <span className="grow" />
        <button className="btn sm" onClick={() => void runCheck()} disabled={checking}>
          <RefreshCw size={15} className={checking ? "spin" : undefined} />
          {t("colab.setup.check")}
        </button>
      </div>
      <ErrorBox error={checkError} />
      {status && (
        <ul className="checks">
          <Check state={status.cli.installed ? "ok" : "bad"} label={t("colab.setup.cli")}>
            {status.cli.installed ? `Colab CLI ${status.cli.version}` : <code>uv tool install google-colab-cli --with keyrings.alt</code>}
          </Check>
          <Check state={status.account.ok ? "ok" : "bad"} label={t("colab.setup.account")}>
            {status.account.ok ? t("colab.setup.signed_in") : (status.account.error ?? "–")}
            {!status.account.ok && status.cli.installed && <InfoTip>{status.account.hint ?? t("colab.setup.login_help")}</InfoTip>}
          </Check>
          <Check state={status.account.ok ? "ok" : "warn"} label={t("colab.setup.gpu")}>
            {gpu && status.account.ok ? (
              <span className="stack" style={{ gap: 4 }}>
                <span>
                  {gpu.sessions.length
                    ? t("colab.setup.sessions", { list: gpu.sessions.map((s) => `${s.name} (${s.hardware})`).join(", ") })
                    : t("colab.setup.no_session")}
                </span>
                {gpu.balance !== null && <span className="small muted">{t("colab.setup.balance", { balance: gpu.balance.toFixed(1) })}</span>}
              </span>
            ) : (
              "–"
            )}
            <InfoTip>{t("colab.setup.gpu_help")}</InfoTip>
          </Check>
          <Check state={status.smpl.found ? "ok" : needsSmpl ? "bad" : "warn"} label={t("colab.setup.smpl")}>
            {status.smpl.found ? t("colab.setup.smpl_found") : t("colab.setup.smpl_missing")}
            <InfoTip>{status.smpl.path ?? t("colab.setup.smpl_help")}</InfoTip>
          </Check>
          <Check state={!status.code.ok ? "bad" : status.code.dirty ? "warn" : "ok"} label={t("colab.setup.code")}>
            {status.code.ok ? t("kaggle.setup.commit", { commit: status.code.commit ?? "" }) : status.code.error}
            {status.code.ok && status.code.dirty && <InfoTip>{t("colab.setup.dirty")}</InfoTip>}
          </Check>
        </ul>
      )}
    </div>
  );
}
