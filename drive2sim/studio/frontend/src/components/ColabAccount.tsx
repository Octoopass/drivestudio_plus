// Tài khoản Google và phiên bản lệnh colab, nằm trong ô thiết lập Colab (CONTRACTS mục 9.3, 11.1):
// xem tài khoản đang dùng, chuyển sang tài khoản đã đăng nhập, thêm tài khoản (mở trang đăng nhập
// Google, dán mã xác thực Google hiện ra), bỏ tài khoản khỏi danh sách; xem bản google-colab-cli và
// cập nhật (việc nền colab_update). Phiên Colab và Drive đều theo tài khoản đang dùng.
import { ExternalLink, RefreshCw, UserPlus } from "lucide-react";
import { useEffect, useState } from "react";
import { api, asApiError, del, post, seg, type ApiError } from "../api/client";
import type { Job } from "../api/types";
import { useI18n } from "../i18n";
import { ErrorBox, InfoTip } from "./ui";

interface Accounts {
  accounts: string[];
  active: string | null;
}

interface Login {
  state: "idle" | "waiting_code" | "checking" | "succeeded" | "failed";
  url: string | null;
  email: string | null;
  error: string | null;
}

interface CliVersions {
  version: string | null;
  latest: string | null;
  up_to_date: boolean;
}

/** Ô tài khoản Google; gọi `onChanged` sau khi đổi hoặc thêm tài khoản để ô thiết lập kiểm tra lại. */
export function ColabAccount({ onChanged }: { onChanged?: () => void }) {
  const { t } = useI18n();
  const [accounts, setAccounts] = useState<Accounts | null>(null);
  const [selected, setSelected] = useState("");
  const [login, setLogin] = useState<Login | null>(null);
  const [code, setCode] = useState("");
  const [cli, setCli] = useState<CliVersions | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const run = async (what: string, work: () => Promise<void>) => {
    setBusy(what);
    setError(null);
    try {
      await work();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(null);
    }
  };

  const loadAccounts = async () => {
    const data = await api<Accounts>("/api/colab/accounts");
    setAccounts(data);
    setSelected(data.active ?? data.accounts[0] ?? "");
  };
  const loadCli = async () => setCli(await api<CliVersions>("/api/colab/cli"));

  useEffect(() => {
    void run("load", async () => {
      await loadAccounts();
      setLogin(await api<Login>("/api/colab/login"));
    });
    void loadCli().catch(() => undefined);
  }, []);

  const switchTo = () =>
    run("use", async () => {
      setAccounts(await post<Accounts>(`/api/colab/accounts/${seg(selected)}/use`));
      onChanged?.();
    });
  const forget = () =>
    run("forget", async () => {
      const data = await del<Accounts>(`/api/colab/accounts/${seg(selected)}`);
      setAccounts(data);
      setSelected(data.active ?? data.accounts[0] ?? "");
    });
  const startLogin = () =>
    run("login", async () => {
      setCode("");
      setLogin(await post<Login>("/api/colab/login"));
    });
  const submitCode = () =>
    run("code", async () => {
      const result = await post<Login>("/api/colab/login/code", { code: code.trim() });
      setLogin(result);
      if (result.state === "succeeded") {
        await loadAccounts();
        onChanged?.();
      }
    });
  const cancelLogin = () => run("cancel", async () => setLogin(await post<Login>("/api/colab/login/cancel")));
  const update = () =>
    run("update", async () => {
      let job = await post<Job>("/api/colab/cli/update");
      while (job.status === "queued" || job.status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        job = await api<Job>(`/api/jobs/${seg(job.job_id)}`);
      }
      if (job.status === "failed") throw new Error(job.error ?? t("colab.account.update_failed"));
      await loadCli();
    });

  const waiting = login?.state === "waiting_code" && login.url?.startsWith("https://accounts.google.com/");
  const other = Boolean(selected) && selected !== accounts?.active;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
        <label className="field grow">
          <span>
            {t("colab.account.active")}
            <InfoTip>{t("colab.account.help")}</InfoTip>
          </span>
          <select value={selected} onChange={(e) => setSelected(e.target.value)} disabled={!accounts?.accounts.length}>
            {!accounts?.accounts.length && <option value="">{t("colab.account.none")}</option>}
            {accounts?.accounts.map((email) => (
              <option key={email} value={email}>
                {email === accounts.active ? t("colab.account.current", { email }) : email}
              </option>
            ))}
          </select>
        </label>
        {other && (
          <button className="btn primary sm" onClick={() => void switchTo()} disabled={busy !== null}>
            {t("colab.account.use")}
          </button>
        )}
        {other && (
          <button className="btn sm" onClick={() => void forget()} disabled={busy !== null}>
            {t("colab.account.forget")}
          </button>
        )}
        <button className="btn sm" onClick={() => void startLogin()} disabled={busy !== null || waiting}>
          <UserPlus size={15} />
          {t("colab.account.add")}
        </button>
      </div>
      {waiting && login?.url && (
        <form
          className="stack key-form"
          onSubmit={(event) => {
            event.preventDefault();
            void submitCode();
          }}
        >
          <p className="small">{t("colab.account.login_steps")}</p>
          <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
            <a className="btn primary sm" href={login.url} target="_blank" rel="noreferrer">
              <ExternalLink size={15} />
              {t("colab.account.open_google")}
            </a>
            <label className="field grow">
              <span>{t("colab.account.code")}</span>
              <input value={code} onChange={(e) => setCode(e.target.value)} autoComplete="off" spellCheck={false} placeholder="4/0A…" />
            </label>
            <button className="btn primary sm" type="submit" disabled={code.trim().length < 10 || busy !== null}>
              {t(busy === "code" ? "colab.account.checking" : "colab.account.confirm")}
            </button>
            <button className="btn sm" type="button" onClick={() => void cancelLogin()} disabled={busy === "code"}>
              {t("colab.account.cancel")}
            </button>
          </div>
        </form>
      )}
      {login?.state === "failed" && <p className="small muted">{t("colab.account.login_failed")}</p>}
      <div className="row wrap small" style={{ gap: 8 }}>
        <span className="muted">
          {cli?.version
            ? cli.latest
              ? t(cli.up_to_date ? "colab.account.cli_latest" : "colab.account.cli_outdated", { version: cli.version, latest: cli.latest })
              : t("colab.account.cli_version", { version: cli.version })
            : t("colab.account.cli_unknown")}
        </span>
        <button className="btn sm" onClick={() => void run("cli", loadCli)} disabled={busy !== null}>
          <RefreshCw size={15} className={busy === "cli" ? "spin" : undefined} />
          {t("colab.account.cli_check")}
        </button>
        {cli?.version && cli.latest && !cli.up_to_date && (
          <button className="btn primary sm" onClick={() => void update()} disabled={busy !== null}>
            {t(busy === "update" ? "colab.account.updating" : "colab.account.cli_update")}
          </button>
        )}
      </div>
      <ErrorBox error={error} />
    </div>
  );
}
