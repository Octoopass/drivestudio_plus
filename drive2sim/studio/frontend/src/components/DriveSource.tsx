// Lấy dữ liệu gốc từ Google Drive qua một phiên Colab CPU (CONTRACTS mục 9.3, 11.1): chuẩn bị phiên,
// kết nối Drive (lần đầu Google hỏi quyền: mở trang đồng ý rồi bấm "Đã đồng ý"), duyệt thư mục trong
// MyDrive, rồi tải thư mục đã chọn về uploads/<tên>/ (việc nền drive_pull, có tiến độ). Tải xong thì
// gọi onPulled với đường dẫn trên máy chạy Studio để ô nạp cảnh xem thư mục đó.
import { ChevronRight, CloudDownload, ExternalLink, Folder, HardDrive, Power } from "lucide-react";
import { useState } from "react";
import { ApiError, api, asApiError, post, seg } from "../api/client";
import type { Job } from "../api/types";
import { useI18n } from "../i18n";
import { JobProgress } from "./JobProgress";
import { ErrorBox, InfoTip } from "./ui";

interface Mount {
  state: "idle" | "needs_consent" | "running" | "succeeded" | "failed";
  session: string | null;
  url: string | null;
  log: string;
}

interface Listing {
  path: string;
  mounted: boolean;
  exists: boolean;
  dirs: string[];
  files: [string, number][];
  truncated: boolean;
}

const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;

function size(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KiB`;
}

function folderName(path: string): string {
  const last = path.split("/").filter(Boolean).pop() ?? "drive";
  return last.replace(/[^A-Za-z0-9_.-]/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80) || "drive";
}

async function waitJob(job: Job, onUpdate: (job: Job) => void): Promise<Job> {
  onUpdate(job);
  while (job.status === "queued" || job.status === "running") {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    job = await api<Job>(`/api/jobs/${seg(job.job_id)}`);
    onUpdate(job);
  }
  return job;
}

export function DriveSource({ onPulled }: { onPulled: (pathAbs: string) => void }) {
  const { t } = useI18n();
  const [session, setSession] = useState("d2s-drive");
  const [mount, setMount] = useState<Mount | null>(null);
  const [listing, setListing] = useState<Listing | null>(null);
  const [name, setName] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const run = async (key: string, action: () => Promise<void>) => {
    setBusy(key);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(null);
    }
  };

  const browse = (path: string) =>
    run("browse", async () => {
      const result = await api<Listing>(`/api/colab/drive/browse?session=${seg(session)}&path=${seg(path)}`);
      setListing(result);
      setName(folderName(result.path));
    });

  const prepare = () =>
    run("session", async () => {
      const done = await waitJob(await post<Job>("/api/colab/drive/session", { session }), setJob);
      if (done.status === "failed") throw new ApiError(502, "colab_failed", done.error ?? "");
      setJob(null);
      const result = await post<Mount>("/api/colab/drive/mount", { session });
      setMount(result);
      if (result.state === "succeeded") await browseNow("");
    });

  // browse bên trong một run() khác: gọi thẳng, không bọc lại
  const browseNow = async (path: string) => {
    const result = await api<Listing>(`/api/colab/drive/browse?session=${seg(session)}&path=${seg(path)}`);
    setListing(result);
    setName(folderName(result.path));
  };

  const confirm = () =>
    run("confirm", async () => {
      const result = await post<Mount>("/api/colab/drive/mount/confirm");
      setMount(result);
      if (result.state === "succeeded" || result.state === "running") await browseNow("");
    });

  const check = () =>
    run("check", async () => {
      const result = await api<Mount>("/api/colab/drive/mount");
      setMount(result);
      if (result.state === "succeeded") await browseNow("");
    });

  const stop = () =>
    run("stop", async () => {
      await post("/api/colab/drive/session/stop", { session });
      setMount(null);
      setListing(null);
    });

  const pull = () =>
    run("pull", async () => {
      if (!listing) return;
      const done = await waitJob(await post<Job>("/api/colab/drive/pull", { session, path: listing.path, name }), setJob);
      if (done.status === "failed") throw new ApiError(502, "colab_failed", done.error ?? "");
      if (done.result) onPulled(done.result);
    });

  const crumbs = listing?.path ? listing.path.split("/") : [];
  const consent = mount?.state === "needs_consent" && mount.url?.startsWith("https://");
  return (
    <details className="stack" style={{ gap: 8 }}>
      <summary className="small" style={{ cursor: "pointer" }}>
        {t("drive.title")}
        <InfoTip>{t("drive.help")}</InfoTip>
      </summary>
      <div className="row wrap" style={{ gap: 8, alignItems: "flex-end", marginTop: 8 }}>
        <label className="field grow">
          <span>{t("drive.session")}</span>
          <input value={session} onChange={(e) => setSession(e.target.value)} pattern="[A-Za-z0-9_\-]{1,40}" disabled={Boolean(busy)} spellCheck={false} />
        </label>
        <button className="btn sm primary" type="button" onClick={() => void prepare()} disabled={Boolean(busy) || !/^[A-Za-z0-9_-]{1,40}$/.test(session)}>
          <HardDrive size={15} />
          {busy === "session" ? t("drive.connecting") : t("drive.connect")}
        </button>
        {(mount || listing) && (
          <button className="btn sm" type="button" onClick={() => void stop()} disabled={Boolean(busy)} title={t("drive.stop_help")}>
            <Power size={15} />
            {t("drive.stop")}
          </button>
        )}
      </div>
      {busy === "session" && job && <JobProgress job={job} />}
      {consent && (
        <div className="row wrap" style={{ gap: 8 }}>
          <span className="small grow">{t("drive.consent")}</span>
          <a className="btn sm" href={mount.url ?? undefined} target="_blank" rel="noreferrer">
            <ExternalLink size={15} />
            {t("drive.consent_open")}
          </a>
          <button className="btn sm primary" type="button" onClick={() => void confirm()} disabled={Boolean(busy)}>
            {t("drive.consent_done")}
          </button>
        </div>
      )}
      {mount?.state === "running" && !listing?.mounted && (
        <div className="row wrap" style={{ gap: 8 }}>
          <span className="small grow muted">{t("drive.mounting")}</span>
          <button className="btn sm" type="button" onClick={() => void check()} disabled={Boolean(busy)}>
            {t("drive.check")}
          </button>
        </div>
      )}
      {mount?.state === "failed" && <pre className="small muted" style={{ whiteSpace: "pre-wrap" }}>{mount.log}</pre>}
      {listing && !listing.mounted && <span className="small muted">{t("drive.not_mounted")}</span>}
      {listing?.mounted && (
        <div className="stack" style={{ gap: 6 }}>
          <div className="row wrap small" style={{ gap: 4, alignItems: "center" }}>
            <button className="btn sm" type="button" onClick={() => void browse("")} disabled={Boolean(busy)}>
              MyDrive
            </button>
            {crumbs.map((part, i) => (
              <span key={i} className="row" style={{ gap: 4, alignItems: "center" }}>
                <ChevronRight size={13} />
                <button className="btn sm" type="button" onClick={() => void browse(crumbs.slice(0, i + 1).join("/"))} disabled={Boolean(busy)}>
                  {part}
                </button>
              </span>
            ))}
          </div>
          <div className="panel" style={{ maxHeight: 220, overflow: "auto", padding: "6px 10px" }}>
            {!listing.exists && <span className="small muted">{t("drive.missing")}</span>}
            {listing.dirs.map((dir) => (
              <button
                key={dir}
                type="button"
                className="row small"
                style={{ gap: 6, width: "100%", background: "none", border: 0, padding: "3px 0", cursor: "pointer", color: "inherit", textAlign: "left" }}
                onClick={() => void browse(listing.path ? `${listing.path}/${dir}` : dir)}
                disabled={Boolean(busy)}
              >
                <Folder size={14} />
                <span className="grow ellipsis">{dir}</span>
              </button>
            ))}
            {listing.files.map(([file, bytes]) => (
              <div key={file} className="row small muted" style={{ gap: 6, padding: "3px 0" }}>
                <span className="grow ellipsis">{file}</span>
                <span>{size(bytes)}</span>
              </div>
            ))}
            {listing.exists && !listing.dirs.length && !listing.files.length && <span className="small muted">{t("drive.empty")}</span>}
            {listing.truncated && <span className="small faint">{t("drive.truncated")}</span>}
          </div>
          {listing.path && (
            <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
              <label className="field grow">
                <span>{t("drive.target")}</span>
                <input value={name} onChange={(e) => setName(e.target.value)} disabled={Boolean(busy)} spellCheck={false} />
              </label>
              <button className="btn sm primary" type="button" onClick={() => void pull()} disabled={Boolean(busy) || !NAME.test(name)}>
                <CloudDownload size={15} />
                {t("drive.pull", { path: listing.path })}
              </button>
            </div>
          )}
        </div>
      )}
      {busy === "pull" && job && <JobProgress job={job} />}
      <ErrorBox error={error} />
    </details>
  );
}
