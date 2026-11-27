// Tải dữ liệu gốc từ máy người dùng lên uploads/<tên>/ của thư mục làm việc (CONTRACTS mục 11.1),
// cho Studio chạy trên máy khác (GPU thuê, server lab qua port forward). Chọn thư mục hoặc file;
// mỗi file gửi theo khối 8 MiB qua hàng đợi chung (uploadQueue), nên đóng ngăn này việc tải vẫn
// chạy và trang Lưu trữ quản lý được. Tải lại cùng tên thì đi tiếp phần còn thiếu. Tải xong thì
// gọi onUploaded với đường dẫn trên máy chạy Studio để ô nạp cảnh xem thư mục đó.
import { FileUp, FolderUp, Play, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { type ApiError, asApiError } from "../api/client";
import { useI18n } from "../i18n";
import {
  clock,
  deleteUpload,
  gib,
  isActive,
  NAME_RE,
  pickedFrom,
  type Picked,
  startUpload,
  stopUpload,
  type UploadTask,
  useUploads,
} from "../uploadQueue";
import { ErrorBox, InfoTip } from "./ui";

/** Thanh tiến độ và dòng trạng thái của một việc tải. */
export function UploadProgress({ task }: { task: UploadTask }) {
  const { t } = useI18n();
  const pct = task.total ? Math.min(100, (100 * task.sent) / task.total) : task.status === "done" ? 100 : 0;
  const left = task.rate > 0 ? (task.total - task.sent) / task.rate : null;
  const text =
    task.status === "done"
      ? t("upload.finished")
      : task.status === "stopping"
        ? t("upload.stopping")
        : task.status === "paused"
          ? t("upload.paused", { sent: gib(task.sent), total: gib(task.total) })
          : task.status === "failed"
            ? t("upload.failed")
            : t("upload.progress", {
                sent: gib(task.sent),
                total: gib(task.total),
                rate: task.rate ? `${(task.rate / 1024 ** 2).toFixed(1)} MB/s` : "–",
                left: left !== null ? clock(left) : "–",
                file: task.file,
              });
  return (
    <div className="stack" style={{ gap: 4, minWidth: 0 }}>
      <div className={`progress${task.status === "running" && !task.total ? " indeterminate" : ""}`}>
        <i style={{ width: `${pct}%` }} />
      </div>
      <span className="small muted ellipsis">{text}</span>
    </div>
  );
}

export function UploadSource({ onUploaded }: { onUploaded?: (pathAbs: string) => void }) {
  const { t } = useI18n();
  const folderInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<Picked[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<ApiError | null>(null);
  const [started, setStarted] = useState(false);
  const task = useUploads().find((x) => x.name === name);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const choose = (list: FileList | null, folder: boolean) => {
    if (!list?.length) return;
    const { picked: files, suggested } = pickedFrom(list, folder);
    setPicked(files);
    setName(suggested);
    setError(null);
    setStarted(false);
  };

  const start = async () => {
    setError(null);
    setStarted(true);
    try {
      const result = await startUpload(name, started ? undefined : picked);
      if (result.status === "done" && mounted.current) onUploaded?.(result.pathAbs);
    } catch (err) {
      setError(asApiError(err));
    }
  };

  const discard = async () => {
    try {
      await deleteUpload(name);
      setPicked([]);
      setStarted(false);
    } catch (err) {
      setError(asApiError(err));
    }
  };

  const size = picked.reduce((sum, p) => sum + p.file.size, 0);
  const active = isActive(task);
  const shown = started && task;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        <span className="small">
          {t("upload.title")}
          <InfoTip>{t("upload.help")}</InfoTip>
        </span>
        <span className="grow" />
        <button className="btn sm" type="button" onClick={() => folderInput.current?.click()} disabled={active}>
          <FolderUp size={15} />
          {t("upload.pick_folder")}
        </button>
        <button className="btn sm" type="button" onClick={() => fileInput.current?.click()} disabled={active}>
          <FileUp size={15} />
          {t("upload.pick_files")}
        </button>
        <input
          ref={folderInput}
          type="file"
          multiple
          hidden
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(e) => {
            choose(e.target.files, true);
            e.target.value = "";
          }}
        />
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            choose(e.target.files, false);
            e.target.value = "";
          }}
        />
      </div>
      {picked.length > 0 && (
        <div className="row wrap" style={{ gap: 8, alignItems: "flex-end" }}>
          <label className="field grow">
            <span>{t("upload.name")}</span>
            <input value={name} onChange={(e) => setName(e.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}" disabled={active || started} spellCheck={false} />
          </label>
          <span className="small muted">{t("upload.selected", { count: picked.length, size: gib(size) })}</span>
          {active ? (
            <button className="btn sm" type="button" onClick={() => stopUpload(name)} disabled={task?.status === "stopping"}>
              <Square size={15} />
              {t("upload.stop")}
            </button>
          ) : (
            task?.status !== "done" && (
              <button className="btn primary sm" type="button" onClick={() => void start()} disabled={!NAME_RE.test(name)}>
                <Play size={15} />
                {t(shown ? "upload.resume" : "upload.start")}
              </button>
            )
          )}
          {shown && !active && task.status !== "done" && (
            <button className="btn sm" type="button" onClick={() => void discard()}>
              <Trash2 size={15} />
              {t("upload.discard")}
            </button>
          )}
        </div>
      )}
      {shown && <UploadProgress task={task} />}
      <ErrorBox error={error ?? (shown ? task.error : null)} />
    </div>
  );
}
