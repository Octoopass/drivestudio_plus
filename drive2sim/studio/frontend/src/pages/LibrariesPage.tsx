// Thư viện: những gì Studio cần trên máy này (môi trường chính, Instant NuRec, renderer CUDA, SDK
// dataset, công cụ), trạng thái từng mục và nút cài (GET /api/libraries, CONTRACTS mục 11.1), như
// bảng thư viện của Fisheye Fusion Lab. Mỗi lúc một việc cài; tiến độ, log, nút dừng ở đầu trang.
// "Cài tất cả" (POST /api/libraries/install-all) cài lần lượt mọi mục còn thiếu, trừ mục bulk=false.
import { CheckCircle2, CircleDashed, Download, ExternalLink, PackageCheck, Square, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, asApiError, post, seg } from "../api/client";
import { clock } from "../components/JobProgress";
import { ErrorBox, Loading, PageHead } from "../components/ui";
import { useI18n } from "../i18n";

interface QueueItem {
  id: string;
  name: string;
  result: "waiting" | "running" | "installed" | "present" | "skipped" | "failed" | "cancelled";
  reason: string | null;
}

interface Install {
  state: "idle" | "running" | "succeeded" | "failed" | "cancelled";
  all: { state: "running" | "succeeded" | "failed" | "cancelled"; current: string | null; items: QueueItem[] } | null;
  env: string | null;
  step: number | null;
  steps: number | null;
  label: string | null;
  started_at: number | null;
  log: string;
  log_path: string | null;
}

interface Item {
  id: string;
  group: string;
  name: string;
  desc: string;
  links: [string, string][];
  size: string;
  minutes: number;
  requires: string[];
  installable: boolean;
  bulk: boolean;
  bulk_note?: string;
  status: "ok" | "partial" | "missing";
  found: string | null;
}

interface Catalog {
  groups: string[];
  items: Item[];
  install: Install;
}

function StatusIcon({ status }: { status: Item["status"] }) {
  if (status === "ok") return <CheckCircle2 size={17} className="ok-ink" />;
  if (status === "partial") return <CircleDashed size={17} />;
  return <XCircle size={17} className="bad-ink" />;
}

export function LibrariesPage() {
  const { t } = useI18n();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [install, setInstall] = useState<Install | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const running = install?.state === "running" || install?.all?.state === "running";

  const load = () =>
    api<Catalog>("/api/libraries")
      .then((result) => {
        setCatalog(result);
        setInstall(result.install);
      })
      .catch((err) => setError(asApiError(err)));

  useEffect(() => {
    void load();
  }, []);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setNow(Date.now() / 1000);
      api<Install>("/api/environment/install")
        .then((result) => {
          setInstall(result);
          if (result.state !== "running") void load();
        })
        .catch((err) => setError(asApiError(err)));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [running]);

  const start = async (item: Item) => {
    if (!window.confirm(t("libraries.confirm", { name: item.name, size: item.size, minutes: item.minutes }))) return;
    setError(null);
    try {
      setInstall(await post<Install>(`/api/libraries/${seg(item.id)}/install`));
    } catch (err) {
      setError(asApiError(err));
    }
  };
  const startAll = async () => {
    const todo = (catalog?.items ?? []).filter((item) => item.bulk && item.status !== "ok");
    if (!window.confirm(t("libraries.all.confirm", { count: todo.length, names: todo.map((item) => item.name).join(", ") }))) return;
    setError(null);
    try {
      setInstall(await post<Install>("/api/libraries/install-all"));
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

  if (!catalog) return error ? <ErrorBox error={error} /> : <Loading />;
  const names = Object.fromEntries(catalog.items.map((item) => [item.id, item]));
  return (
    <div className="page">
      <PageHead
        title={t("libraries.page.title")}
        info={t("libraries.page.lede")}
        actions={
          <button className="btn primary" type="button" onClick={() => void startAll()} disabled={running || !catalog.items.some((item) => item.bulk && item.status !== "ok")} title={t("libraries.all.help")}>
            <PackageCheck size={17} />
            {t("libraries.all.button")}
          </button>
        }
      />
      {install && install.state !== "idle" && (
        <div className="panel stack" style={{ gap: 6, padding: "12px 15px" }}>
          <div className="row" style={{ gap: 8, alignItems: "center" }}>
            <b className="grow">
              {t(`libraries.install.${install.state}`, { name: names[install.env ?? ""]?.name ?? install.env ?? "" })}
              {install.step ? ` · ${t("env.install.step", { step: install.step, steps: install.steps ?? "?", label: install.label ?? "" })}` : ""}
            </b>
            {running && install.started_at && <span className="small muted">{clock(now - install.started_at)}</span>}
            {running && (
              <button className="btn sm" type="button" onClick={() => void cancel()}>
                <Square size={14} />
                {t("env.install.cancel")}
              </button>
            )}
          </div>
          {running && (
            <div className={install.steps ? "progress" : "progress indeterminate"}>
              <i style={install.steps ? { width: `${(100 * Math.max(0, (install.step ?? 1) - 1)) / install.steps}%` } : undefined} />
            </div>
          )}
          {install.all && (
            <ul className="small stack" style={{ gap: 2, margin: 0, paddingLeft: 18 }}>
              {install.all.items.map((item) => (
                <li key={item.id} className={item.result === "failed" ? "bad-ink" : item.result === "running" ? "" : "muted"}>
                  {item.name}: {t(`libraries.all.result.${item.result}`)}
                  {item.reason ? ` (${item.reason})` : ""}
                </li>
              ))}
            </ul>
          )}
          {install.log && (
            <details open={install.state === "failed"}>
              <summary className="small" style={{ cursor: "pointer" }}>
                {t("env.install.log", { path: install.log_path ?? "" })}
              </summary>
              <pre className="small" style={{ maxHeight: 220, overflow: "auto", whiteSpace: "pre-wrap" }}>
                {install.log.split("\n").slice(-60).join("\n")}
              </pre>
            </details>
          )}
        </div>
      )}
      <ErrorBox error={error} />
      {catalog.groups.map((group) => (
        <section key={group} className="stack" style={{ gap: 10 }}>
          <h3>{group}</h3>
          <div className="grid" style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" }}>
            {catalog.items
              .filter((item) => item.group === group)
              .map((item) => {
                const blocked = item.requires.filter((id) => names[id]?.status !== "ok");
                return (
                  <div key={item.id} className="panel stack" style={{ gap: 6, padding: "12px 15px" }}>
                    <div className="row" style={{ gap: 8, alignItems: "center" }}>
                      <StatusIcon status={item.status} />
                      <b className="grow">{item.name}</b>
                      <span className="small faint">
                        {item.size} · ~{item.minutes} {t("libraries.minutes")}
                      </span>
                    </div>
                    <span className="small muted">{item.desc}</span>
                    {item.found && <span className="small faint ellipsis">{item.found}</span>}
                    <div className="row wrap small" style={{ gap: 10, alignItems: "center" }}>
                      {item.links.map(([label, url]) => (
                        <a key={url} href={url} target="_blank" rel="noreferrer">
                          <ExternalLink size={13} /> {label}
                        </a>
                      ))}
                      <span className="grow" />
                      {item.status !== "ok" && item.installable && (
                        <button
                          className="btn sm primary"
                          type="button"
                          onClick={() => void start(item)}
                          disabled={running || blocked.length > 0}
                          title={blocked.length ? t("libraries.requires", { names: blocked.map((id) => names[id]?.name ?? id).join(", ") }) : undefined}
                        >
                          <Download size={14} />
                          {t("libraries.install")}
                        </button>
                      )}
                      {item.status !== "ok" && !item.installable && <span className="small faint">{t("libraries.manual")}</span>}
                      {item.installable && !item.bulk && <span className="small faint" title={item.bulk_note}>{t("libraries.all.excluded")}</span>}
                    </div>
                  </div>
                );
              })}
          </div>
        </section>
      ))}
    </div>
  );
}
