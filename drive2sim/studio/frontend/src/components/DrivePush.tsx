// Đẩy file của run (PLY, scene.usdz cho CARLA, video) lên Google Drive qua một phiên Colab CPU đã
// mount Drive, như ô lấy dữ liệu từ Drive nhưng ngược chiều: kết nối Drive (lần đầu Google hỏi
// quyền), chọn file và thư mục trong MyDrive, rồi đẩy ở nền (việc drive_push, có tiến độ, huỷ và
// đẩy tiếp được). Tên file trên Drive có thêm 4 ký tự cuối của run ID để file các run không đè nhau.
import { CloudUpload, ExternalLink, HardDrive, Power } from "lucide-react";
import { useState } from "react";
import { ApiError, api, asApiError, post } from "../api/client";
import type { Job } from "../api/types";
import { useI18n } from "../i18n";
import { waitJob, type Mount } from "./DriveSource";
import { JobActions, JobProgress } from "./JobProgress";
import { ErrorBox, bytes } from "./ui";

export interface PushItem {
  kind: string;
  name: string;
  path: string;
  size_bytes: number;
}

const SESSION = /^[A-Za-z0-9_-]{1,40}$/;

export function DrivePush({ runId, items }: { runId: string; items: PushItem[] }) {
  const { t } = useI18n();
  const [session, setSession] = useState("d2s-drive");
  const [mount, setMount] = useState<Mount | null>(null);
  const [mounted, setMounted] = useState<boolean | null>(null);
  const [folder, setFolder] = useState(`drive2sim/${runId}`);
  const [picked, setPicked] = useState<string[]>(() => items.filter((i) => i.kind !== "video").map((i) => i.path));
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [done, setDone] = useState<string | null>(null);

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
  const checkMounted = async () => {
    const listing = await api<{ mounted: boolean }>(`/api/colab/drive/browse?session=${encodeURIComponent(session)}&path=`);
    setMounted(listing.mounted);
  };
  const connect = () =>
    run("session", async () => {
      const created = await waitJob(await post<Job>("/api/colab/drive/session", { session }), setJob);
      if (created.status === "failed") throw new ApiError(502, "colab_failed", created.error ?? "");
      setJob(null);
      const result = await post<Mount>("/api/colab/drive/mount", { session });
      setMount(result);
      if (result.state === "succeeded") await checkMounted();
    });
  const confirm = () =>
    run("confirm", async () => {
      const result = await post<Mount>("/api/colab/drive/mount/confirm");
      setMount(result);
      if (result.state === "succeeded" || result.state === "running") await checkMounted();
    });
  const check = () => run("check", checkMounted);
  const stop = () =>
    run("stop", async () => {
      await post("/api/colab/drive/session/stop", { session });
      setMount(null);
      setMounted(null);
    });
  const push = () =>
    run("push", async () => {
      setDone(null);
      const body = { session, paths: picked, folder, prefix: runId.slice(-4) };
      const finished = await waitJob(await post<Job>("/api/colab/drive/push", body), setJob);
      if (finished.status === "cancelled") return;
      if (finished.status === "failed") throw new ApiError(502, "colab_failed", finished.error ?? "");
      setDone(finished.result);
    });
  const toggle = (path: string) => setPicked((now) => (now.includes(path) ? now.filter((p) => p !== path) : [...now, path]));
  const consent = mount?.state === "needs_consent" && mount.url?.startsWith("https://");
  const total = items.filter((i) => picked.includes(i.path)).reduce((sum, i) => sum + i.size_bytes, 0);
  return (
    <div className="stack" style={{ gap: 8, marginTop: 8 }}>
      <p className="small muted">{t("drive_push.help")}</p>
      <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
        <label className="field grow">
          <span>{t("drive.session")}</span>
          <input value={session} onChange={(e) => setSession(e.target.value)} disabled={Boolean(busy)} spellCheck={false} />
        </label>
        <button className="btn sm" type="button" onClick={() => void connect()} disabled={Boolean(busy) || !SESSION.test(session)}>
          <HardDrive size={15} />
          {busy === "session" ? t("drive.connecting") : t("drive.connect")}
        </button>
        <button className="btn sm" type="button" onClick={() => void check()} disabled={Boolean(busy) || !SESSION.test(session)}>
          {t("drive.check")}
        </button>
        <button className="btn sm" type="button" onClick={() => void stop()} disabled={Boolean(busy) || !SESSION.test(session)} title={t("drive_push.stop_help")}>
          <Power size={15} />
          {t("drive.stop")}
        </button>
      </div>
      {busy === "session" && job && <JobProgress job={job} />}
      {consent && (
        <div className="row wrap" style={{ gap: 8 }}>
          <span className="small grow">{t("drive.consent")}</span>
          <a className="btn sm" href={mount?.url ?? undefined} target="_blank" rel="noreferrer">
            <ExternalLink size={15} />
            {t("drive.consent_open")}
          </a>
          <button className="btn sm primary" type="button" onClick={() => void confirm()} disabled={Boolean(busy)}>
            {t("drive.consent_done")}
          </button>
        </div>
      )}
      {mount?.state === "running" && !mounted && <span className="small muted">{t("drive.mounting")}</span>}
      {mount?.state === "failed" && <pre className="small muted" style={{ whiteSpace: "pre-wrap" }}>{mount.log}</pre>}
      {mounted === false && <span className="small muted">{t("drive.not_mounted")}</span>}
      {mounted && <span className="small">{t("drive_push.mounted")}</span>}
      <div className="stack" style={{ gap: 4 }}>
        {items.map((item) => (
          <label key={item.path} className="row small" style={{ gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={picked.includes(item.path)} onChange={() => toggle(item.path)} disabled={Boolean(busy)} />
            <span className="grow ellipsis">
              {t(`downloads.kind.${item.kind}`)} <span className="muted">{item.name}</span>
            </span>
            <span className="muted nowrap">{bytes(item.size_bytes)}</span>
          </label>
        ))}
      </div>
      <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
        <label className="field grow">
          <span>{t("drive_push.folder")}</span>
          <input value={folder} onChange={(e) => setFolder(e.target.value)} disabled={Boolean(busy)} spellCheck={false} />
        </label>
        <button className="btn sm primary" type="button" onClick={() => void push()} disabled={Boolean(busy) || !picked.length || !folder.trim() || !SESSION.test(session)}>
          <CloudUpload size={15} />
          {t("drive_push.start", { count: picked.length, size: bytes(total) })}
        </button>
      </div>
      {busy === "push" && job && (
        <div className="row" style={{ gap: 8, alignItems: "center" }}>
          <div className="grow">
            <JobProgress job={job} />
          </div>
          <JobActions job={job} />
        </div>
      )}
      {done && <span className="small">{t("drive_push.done", { path: done })}</span>}
      {error && <ErrorBox error={error} />}
    </div>
  );
}
