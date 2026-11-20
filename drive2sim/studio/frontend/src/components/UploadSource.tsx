// Tải dữ liệu gốc từ máy người dùng lên uploads/<tên>/ của thư mục làm việc (CONTRACTS mục 11.1),
// cho Studio chạy trên máy khác (GPU thuê, server lab qua port forward). Chọn thư mục hoặc file;
// mỗi file gửi theo khối 8 MiB, tải lại cùng tên thì đi tiếp phần còn thiếu. Tải xong thì gọi
// onUploaded với đường dẫn trên máy chạy Studio để ô nạp cảnh xem thư mục đó.
import { FileUp, FolderUp, X } from "lucide-react";
import { useRef, useState } from "react";
import { ApiError, asApiError, del, post, seg } from "../api/client";
import { useI18n } from "../i18n";
import { ErrorBox, InfoTip } from "./ui";

const CHUNK = 8 * 1024 * 1024;

interface PlanFile {
  path: string;
  size: number;
  offset: number;
  done: boolean;
}

interface Plan {
  name: string;
  path_abs: string;
  files: PlanFile[];
  skipped: string[];
  needed_bytes: number;
}

interface Picked {
  file: File;
  path: string;
}

function gib(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MiB`;
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s >= 3600 ? `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Đường dẫn tương đối của file: với thư mục bỏ tên thư mục gốc (trùng tên lần tải). */
function relative(file: File): string {
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return rel ? rel.split("/").slice(1).join("/") : file.name;
}

export function UploadSource({ onUploaded }: { onUploaded: (pathAbs: string) => void }) {
  const { t } = useI18n();
  const folderInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<Picked[]>([]);
  const [name, setName] = useState("");
  const [progress, setProgress] = useState<{ sent: number; total: number; file: string; rate: number } | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const cancelled = useRef(false);

  const choose = (list: FileList | null, folder: boolean) => {
    if (!list?.length) return;
    const files = Array.from(list).map((file) => ({ file, path: folder ? relative(file) : file.name }));
    setPicked(files);
    const root = folder ? (list[0] as File & { webkitRelativePath?: string }).webkitRelativePath?.split("/")[0] : "";
    setName((root || "upload").replace(/[^A-Za-z0-9_.-]/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80) || "upload");
    setError(null);
    setProgress(null);
  };

  const plan = () =>
    post<Plan>("/api/sources/uploads", { name, files: picked.map((p) => ({ path: p.path, size: p.file.size })) });

  const start = async () => {
    cancelled.current = false;
    setError(null);
    try {
      let current = await plan();
      const byPath = new Map(picked.map((p) => [p.path, p.file]));
      const total = current.needed_bytes;
      let sent = 0;
      const began = performance.now();
      setProgress({ sent: 0, total, file: "", rate: 0 });
      for (let i = 0; i < current.files.length; i++) {
        const entry = current.files[i];
        const file = byPath.get(entry.path);
        if (!file || entry.done) continue;
        let offset = entry.offset;
        while (offset < entry.size) {
          if (cancelled.current) throw new ApiError(0, "cancelled", t("upload.cancelled"));
          const blob = file.slice(offset, Math.min(entry.size, offset + CHUNK));
          const query = `path=${seg(entry.path)}&offset=${offset}&total=${entry.size}`;
          const response = await fetch(`/api/sources/uploads/${seg(name)}/chunk?${query}`, { method: "POST", body: blob });
          if (response.status === 409) {
            // máy chủ có số byte khác dự kiến (tab khác cũng đang tải): hỏi lại rồi đi tiếp
            current = await plan();
            offset = current.files.find((f) => f.path === entry.path)?.offset ?? 0;
            continue;
          }
          if (!response.ok) {
            const body = (await response.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
            throw new ApiError(response.status, body?.error?.code ?? "internal", body?.error?.message ?? response.statusText);
          }
          const result = (await response.json()) as { offset: number };
          sent += result.offset - offset;
          offset = result.offset;
          const seconds = (performance.now() - began) / 1000;
          setProgress({ sent, total, file: entry.path, rate: seconds > 0 ? sent / seconds : 0 });
        }
      }
      setProgress({ sent: total, total, file: "", rate: 0 });
      onUploaded(current.path_abs);
    } catch (err) {
      setError(asApiError(err));
    }
  };

  const discard = async () => {
    try {
      await del(`/api/sources/uploads/${seg(name)}`);
      setPicked([]);
      setProgress(null);
    } catch (err) {
      setError(asApiError(err));
    }
  };

  const size = picked.reduce((sum, p) => sum + p.file.size, 0);
  const running = progress !== null && progress.sent < progress.total;
  const pct = progress && progress.total ? Math.min(100, (100 * progress.sent) / progress.total) : 0;
  const left = progress && progress.rate > 0 ? (progress.total - progress.sent) / progress.rate : null;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        <span className="small">
          {t("upload.title")}
          <InfoTip>{t("upload.help")}</InfoTip>
        </span>
        <span className="grow" />
        <button className="btn sm" type="button" onClick={() => folderInput.current?.click()} disabled={running}>
          <FolderUp size={15} />
          {t("upload.pick_folder")}
        </button>
        <button className="btn sm" type="button" onClick={() => fileInput.current?.click()} disabled={running}>
          <FileUp size={15} />
          {t("upload.pick_files")}
        </button>
        <input
          ref={folderInput}
          type="file"
          multiple
          hidden
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(e) => choose(e.target.files, true)}
        />
        <input ref={fileInput} type="file" multiple hidden onChange={(e) => choose(e.target.files, false)} />
      </div>
      {picked.length > 0 && (
        <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
          <label className="field grow">
            <span>{t("upload.name")}</span>
            <input value={name} onChange={(e) => setName(e.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}" disabled={running} spellCheck={false} />
          </label>
          <span className="small muted">{t("upload.selected", { count: picked.length, size: gib(size) })}</span>
          {running ? (
            <button className="btn sm" type="button" onClick={() => (cancelled.current = true)}>
              <X size={15} />
              {t("upload.stop")}
            </button>
          ) : (
            <button className="btn primary sm" type="button" onClick={() => void start()} disabled={!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(name)}>
              {t(progress ? "upload.resume" : "upload.start")}
            </button>
          )}
          {!running && progress && (
            <button className="btn sm" type="button" onClick={() => void discard()}>
              {t("upload.discard")}
            </button>
          )}
        </div>
      )}
      {progress && (
        <div className="stack" style={{ gap: 4 }}>
          <div className="progress">
            <i style={{ width: `${pct}%` }} />
          </div>
          <span className="small muted">
            {progress.sent >= progress.total
              ? t("upload.done")
              : t("upload.progress", {
                  sent: gib(progress.sent),
                  total: gib(progress.total),
                  rate: progress.rate ? `${(progress.rate / 1024 ** 2).toFixed(1)} MB/s` : "–",
                  left: left !== null ? clock(left) : "–",
                  file: progress.file,
                })}
          </span>
        </div>
      )}
      <ErrorBox error={error} />
    </div>
  );
}
