// Lấy dữ liệu gốc từ Google Drive qua một phiên Colab CPU (CONTRACTS mục 9.3, 11.1): chuẩn bị phiên,
// kết nối Drive (lần đầu Google hỏi quyền: mở trang đồng ý rồi bấm "Đã đồng ý"), duyệt thư mục trong
// MyDrive, rồi tải thư mục đã chọn về uploads/<tên>/ (việc nền drive_pull, có tiến độ). Tải xong thì
// gọi onPulled với đường dẫn trên máy chạy Studio để ô nạp cảnh xem thư mục đó.
// Mỗi thư mục mở ra được nhận dạng ở nền (/api/colab/drive/inspect, cùng bộ nhận dạng với đường dẫn
// trên máy): dataset, chuỗi kèm dung lượng, thư mục con chứa dataset. KITTI-360 chọn được một chuỗi
// để chỉ tải file chuỗi đó cần; dataset khác chọn được vài mục trong thư mục bằng ô tick.
import { ChevronRight, CloudDownload, ExternalLink, Folder, HardDrive, Power, ScanSearch } from "lucide-react";
import { useRef, useState } from "react";
import { ApiError, api, asApiError, post, seg } from "../api/client";
import type { Job, SourceReport } from "../api/types";
import { useI18n } from "../i18n";
import { JobProgress } from "./JobProgress";
import { ErrorBox, InfoTip, Note } from "./ui";

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

interface Inspection {
  path: string;
  mounted: boolean;
  exists: boolean;
  truncated: boolean;
  report: SourceReport | null;
  sequence_bytes: Record<string, number>;
  pull_sets: Record<string, string[]>;
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
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [sequence, setSequence] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const shownPath = useRef<string | null>(null);

  // nhận dạng ở nền; kết quả của thư mục đã rời đi thì bỏ
  const inspectNow = async (path: string) => {
    setInspecting(true);
    try {
      const result = await api<Inspection>(`/api/colab/drive/inspect?session=${seg(session)}&path=${seg(path)}`);
      if (shownPath.current === result.path) setInspection(result);
    } catch {
      if (shownPath.current === path) setInspection(null);
    } finally {
      if (shownPath.current === path) setInspecting(false);
    }
  };

  const show = (result: Listing) => {
    shownPath.current = result.path;
    setListing(result);
    setName(folderName(result.path));
    setInspection(null);
    setSequence("");
    setPicked([]);
    if (result.mounted && result.exists) void inspectNow(result.path);
    else setInspecting(false);
  };

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
      show(await api<Listing>(`/api/colab/drive/browse?session=${seg(session)}&path=${seg(path)}`));
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
    show(await api<Listing>(`/api/colab/drive/browse?session=${seg(session)}&path=${seg(path)}`));
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
      setInspection(null);
      shownPath.current = null;
    });

  // phần sẽ tải: file của chuỗi đã chọn cộng các mục đã tick; rỗng là cả thư mục
  const include = [...(sequence ? inspection?.pull_sets[sequence] ?? [] : []), ...picked];
  const toggle = (item: string) => setPicked((now) => (now.includes(item) ? now.filter((x) => x !== item) : [...now, item]));

  const pull = () =>
    run("pull", async () => {
      if (!listing) return;
      const body = { session, path: listing.path, name, include };
      const done = await waitJob(await post<Job>("/api/colab/drive/pull", body), setJob);
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
              <div key={dir} className="row small" style={{ gap: 6, alignItems: "center" }}>
                {listing.path && (
                  <input type="checkbox" checked={picked.includes(`${dir}/`)} onChange={() => toggle(`${dir}/`)} disabled={Boolean(busy)} title={t("drive.pick")} />
                )}
                <button
                  type="button"
                  className="row grow"
                  style={{ gap: 6, background: "none", border: 0, padding: "3px 0", cursor: "pointer", color: "inherit", textAlign: "left", minWidth: 0 }}
                  onClick={() => void browse(listing.path ? `${listing.path}/${dir}` : dir)}
                  disabled={Boolean(busy)}
                >
                  <Folder size={14} />
                  <span className="grow ellipsis">{dir}</span>
                </button>
              </div>
            ))}
            {listing.files.map(([file, bytes]) => (
              <div key={file} className="row small muted" style={{ gap: 6, padding: "3px 0", alignItems: "center" }}>
                {listing.path && (
                  <input type="checkbox" checked={picked.includes(file)} onChange={() => toggle(file)} disabled={Boolean(busy)} title={t("drive.pick")} />
                )}
                <span className="grow ellipsis">{file}</span>
                <span>{size(bytes)}</span>
              </div>
            ))}
            {listing.exists && !listing.dirs.length && !listing.files.length && <span className="small muted">{t("drive.empty")}</span>}
            {listing.truncated && <span className="small faint">{t("drive.truncated")}</span>}
          </div>
          {inspecting && (
            <span className="small muted row" style={{ gap: 6, alignItems: "center" }}>
              <ScanSearch size={14} />
              {t("drive.inspecting")}
            </span>
          )}
          {inspection?.report && <Detected inspection={inspection} sequence={sequence} setSequence={setSequence} browse={browse} busy={Boolean(busy)} />}
          {listing.path && (
            <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
              <label className="field grow">
                <span>{t("drive.target")}</span>
                <input value={name} onChange={(e) => setName(e.target.value)} disabled={Boolean(busy)} spellCheck={false} />
              </label>
              <button className="btn sm primary" type="button" onClick={() => void pull()} disabled={Boolean(busy) || !NAME.test(name)}>
                <CloudDownload size={15} />
                {sequence && !picked.length
                  ? t("drive.pull_sequence", { sequence })
                  : include.length
                    ? t("drive.pull_some", { count: picked.length + (sequence ? 1 : 0) })
                    : t("drive.pull", { path: listing.path })}
              </button>
            </div>
          )}
          {listing.path && include.length > 0 && <span className="small faint">{t("drive.pull_some_help")}</span>}
        </div>
      )}
      {busy === "pull" && job && <JobProgress job={job} />}
      <ErrorBox error={error} />
    </details>
  );
}

// Kết quả nhận dạng một thư mục trên Drive: dataset, chuỗi, thư mục con chứa dataset; KITTI-360 thì
// chọn một chuỗi để chỉ tải file của nó.
function Detected({
  inspection,
  sequence,
  setSequence,
  browse,
  busy,
}: {
  inspection: Inspection;
  sequence: string;
  setSequence: (value: string) => void;
  browse: (path: string) => void;
  busy: boolean;
}) {
  const { t } = useI18n();
  const report = inspection.report;
  if (!report) return null;
  const sequences = Object.keys(inspection.pull_sets);
  const count = Object.values(report.versions).reduce((sum, names) => sum + names.length, 0);
  return (
    <div className="panel stack" style={{ gap: 6, padding: "8px 10px" }}>
      {report.dataset ? (
        <span className="small row wrap" style={{ gap: 6, alignItems: "center" }}>
          <ScanSearch size={14} />
          <strong>{t("drive.detected", { dataset: report.dataset })}</strong>
          {count > 0 && <span className="muted">{t("drive.detected_sequences", { count })}</span>}
          {report.has_lidar && <span className="muted">· LiDAR</span>}
          {report.cameras.length > 0 && <span className="muted">· {t("drive.detected_cameras", { count: report.cameras.length })}</span>}
        </span>
      ) : (
        !report.candidates.length && <span className="small muted">{t("drive.not_detected")}</span>
      )}
      {report.dataset && !report.ingest_supported && <Note kind="warn">{t("drive.not_supported")}</Note>}
      {report.errors.map((message) => (
        <Note key={message} kind="warn">{message}</Note>
      ))}
      {report.candidates.length > 0 && (
        <div className="stack" style={{ gap: 4 }}>
          <span className="small muted">{t("drive.candidates")}</span>
          {report.candidates.map((candidate) => (
            <button key={candidate.path} type="button" className="btn sm" style={{ justifyContent: "flex-start" }} onClick={() => browse(candidate.path)} disabled={busy}>
              <Folder size={14} />
              <span className="ellipsis">
                {candidate.path} · {candidate.dataset}
              </span>
            </button>
          ))}
        </div>
      )}
      {sequences.length > 0 && (
        <label className="field">
          <span>
            {t("drive.sequence")}
            <InfoTip>{t("drive.sequence_help")}</InfoTip>
          </span>
          <select value={sequence} onChange={(e) => setSequence(e.target.value)} disabled={busy}>
            <option value="">{t("drive.sequence_all")}</option>
            {sequences.map((name) => (
              <option key={name} value={name}>
                {name} ({size(inspection.sequence_bytes[name] ?? 0)})
              </option>
            ))}
          </select>
        </label>
      )}
      {inspection.truncated && <span className="small faint">{t("drive.inspect_truncated")}</span>}
    </div>
  );
}
