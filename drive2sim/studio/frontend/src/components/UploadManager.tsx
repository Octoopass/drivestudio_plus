// Quản lý dữ liệu gốc đã tải lên uploads/ (trang Lưu trữ): thêm bộ mới, dừng việc đang tải, tải
// tiếp (tab này còn giữ file thì bấm là chạy; không thì chọn lại thư mục, phần đã có không gửi lại),
// xoá. Cảnh nào đọc dữ liệu gốc từ một bộ (raw_source.json, Instant NuRec cần) thì hiện ra và hộp
// thoại xoá cảnh báo trước. Bộ lấy từ Google Drive (việc nền drive_pull) hiện tiến độ của việc đó,
// huỷ được, và tải tiếp được với nguồn Drive đã nhớ kể cả sau khi khởi động lại Studio.
import { CloudDownload, FileUp, FolderUp, Play, Square, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { asApiError, post, seg, useApi } from "../api/client";
import type { Job } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { href } from "../router";
import { deleteUpload, forgetUpload, hasPicked, isActive, pickedFrom, startUpload, stopUpload, type UploadTask, useUploads } from "../uploadQueue";
import { ago, bytes, Confirm, Empty, ErrorBox, Loading, useToast } from "./ui";
import { JobActions, jobActive, JobProgress } from "./JobProgress";
import { UploadProgress, UploadSource } from "./UploadSource";

interface UploadItem {
  name: string;
  path: string;
  path_abs: string;
  files: number;
  size_bytes: number;
  partial: number;
  partial_bytes: number;
  modified: string;
  used_by: string[];
  origin: "drive" | "browser";
}

interface UploadList {
  items: UploadItem[];
  total: number;
}

interface Row {
  name: string;
  item?: UploadItem;
  task?: UploadTask;
  /** việc tải từ Drive mới nhất ghi vào thư mục này */
  job?: Job;
}

// Studio cũ (chưa khởi động lại sau khi cập nhật) chưa trả các trường mới.
function normalize(item: UploadItem | undefined): UploadItem | undefined {
  return item && { ...item, used_by: item.used_by ?? [], partial_bytes: item.partial_bytes ?? 0, origin: item.origin ?? "browser" };
}

export function UploadManager({ focus }: { focus?: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const tasks = useUploads();
  const live = useLive();
  const driveJobs = live.jobs.filter((job) => job.kind === "drive_pull" && job.target);
  const anyActive = tasks.some(isActive) || driveJobs.some(jobActive);
  const list = useApi<UploadList>("/api/sources/uploads", anyActive ? 3000 : undefined);
  const [removing, setRemoving] = useState<Row | null>(null);
  const [repick, setRepick] = useState<{ name: string; folder: boolean } | null>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  // Việc tải vừa dừng hoặc xong thì đọc lại danh sách (số file, dung lượng, phần dở).
  const signature = [...tasks.map((x) => `${x.name}:${x.status}`), ...driveJobs.map((j) => `${j.job_id}:${j.status}`)].join("|");
  const reload = list.reload;
  useEffect(() => reload(), [signature, reload]);

  const resume = async (name: string, picked?: ReturnType<typeof pickedFrom>["picked"]) => {
    const result = await startUpload(name, picked).catch((err) => {
      toast({ kind: "bad", title: t("upload.failed"), detail: asApiError(err).message });
      return null;
    });
    if (result?.status === "done") toast({ kind: "ok", title: t("datasets.done", { name }) });
  };

  // Mở từ Hoạt động (#/storage?focus=<tên>): cuộn tới dòng của bộ đó và làm nổi một lúc.
  const focusRow = useRef<HTMLTableRowElement>(null);
  const hasFocusRow = Boolean(focus && list.data);
  useEffect(() => {
    if (hasFocusRow) focusRow.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [focus, hasFocusRow]);

  const resumeDrive = async (name: string) => {
    try {
      await post<Job>(`/api/colab/drive/pull/${seg(name)}/resume`);
      live.refreshJobs();
    } catch (err) {
      toast({ kind: "bad", title: t("job.retry_failed"), detail: asApiError(err).message });
    }
  };

  const onRepicked = (files: FileList | null) => {
    if (!files?.length || !repick) return;
    const { picked } = pickedFrom(files, repick.folder);
    const name = repick.name;
    setRepick(null);
    void resume(name, picked);
  };

  const askRepick = (name: string, folder: boolean) => {
    setRepick({ name, folder });
    (folder ? folderInput : fileInput).current?.click();
  };

  const names = new Set([...(list.data?.items ?? []).map((i) => i.name), ...tasks.map((x) => x.name), ...driveJobs.filter(jobActive).map((j) => j.target!)]);
  const rows: Row[] = [...names].sort().map((name) => ({
    name,
    item: normalize(list.data?.items.find((i) => i.name === name)),
    task: tasks.find((x) => x.name === name),
    // live.jobs mới nhất trước
    job: driveJobs.find((j) => j.target === name),
  }));
  const total = (list.data?.items ?? []).reduce((sum, i) => sum + i.size_bytes, 0);

  return (
    <section className="panel">
      <div className="panel-body stack" style={{ gap: 12 }}>
        <div className="row wrap" style={{ gap: 8, alignItems: "baseline" }}>
          <h2 style={{ margin: 0 }}>{t("datasets.title")}</h2>
          <span className="small muted">{t("datasets.summary", { count: list.data?.total ?? 0, size: bytes(total) })}</span>
        </div>
        <p className="small muted" style={{ margin: 0 }}>{t("datasets.lede")}</p>
        <UploadSource onUploaded={() => list.reload()} />
        <ErrorBox error={list.error} />
      </div>
      <input
        ref={folderInput}
        type="file"
        multiple
        hidden
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
        onChange={(e) => {
          onRepicked(e.target.files);
          e.target.value = "";
        }}
      />
      <input
        ref={fileInput}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          onRepicked(e.target.files);
          e.target.value = "";
        }}
      />
      {!list.data && !list.error && <div className="panel-body"><Loading /></div>}
      {list.data && rows.length === 0 && <Empty title={t("datasets.empty.title")}><p>{t("datasets.empty.text")}</p></Empty>}
      {rows.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("datasets.table.name")}</th>
                <th className="num">{t("datasets.table.size")}</th>
                <th style={{ width: "100%" }}>{t("datasets.table.status")}</th>
                <th className="num">{t("datasets.table.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ name, item, task, job }) => {
                const pulling = jobActive(job);
                const active = isActive(task) || pulling;
                const fromDrive = item?.origin === "drive";
                const partial = (item?.partial ?? 0) > 0 || task?.status === "paused" || task?.status === "failed";
                const canResume = !active && partial && !fromDrive && hasPicked(name);
                return (
                  <tr key={name} ref={name === focus ? focusRow : undefined} className={name === focus ? "flash" : undefined}>
                    <td>
                      <div className="row" style={{ gap: 6 }}>
                        <span className="id">{name}</span>
                        {fromDrive && <span className="tag info">{t("datasets.tag.drive")}</span>}
                      </div>
                      {item && <div className="small muted ellipsis" style={{ maxWidth: 260 }} title={item.path_abs}>{item.path} · {ago(item.modified, locale)}</div>}
                      {item && item.used_by.length > 0 && (
                        <div className="row wrap" style={{ gap: 4, marginTop: 4, maxWidth: 300 }} title={t("datasets.table.used_by")}>
                          {item.used_by.slice(0, 3).map((sceneId) => (
                            <a key={sceneId} className="tag" style={{ fontSize: 12.5 }} href={href({ page: "scene", sceneId })}>{sceneId}</a>
                          ))}
                          {item.used_by.length > 3 && <span className="tag" style={{ fontSize: 12.5 }}>+{item.used_by.length - 3}</span>}
                        </div>
                      )}
                    </td>
                    <td className="num" style={{ whiteSpace: "nowrap" }}>
                      {item ? bytes(item.size_bytes) : "–"}
                      {item && <div className="small muted">{t("datasets.files", { count: item.files })}</div>}
                    </td>
                    <td style={{ maxWidth: 0, minWidth: 180 }}>
                      {pulling && job ? (
                        <>
                          <span className="small muted">{job.status === "queued" ? t("common.status.queued") : t("activity.kind.drive_pull")}</span>
                          <JobProgress job={job} />
                        </>
                      ) : task && (active || task.status !== "done") ? (
                        <UploadProgress task={task} />
                      ) : partial ? (
                        <span className="tag warn">{t("datasets.tag.partial", { count: item?.partial ?? 0, size: bytes(item?.partial_bytes ?? 0) })}</span>
                      ) : (
                        <span className="tag go">{t("datasets.tag.complete")}</span>
                      )}
                      {!pulling && job?.status === "failed" && partial && <div className="small" style={{ color: "var(--stop)" }}>{job.error}</div>}
                      {task?.error && <div className="small" style={{ color: "var(--stop)" }}>{task.error.message}</div>}
                    </td>
                    <td className="num" style={{ whiteSpace: "nowrap" }}>
                      <span className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
                        {pulling && job && <JobActions job={job} />}
                        {!active && partial && fromDrive && (
                          <button className="btn primary sm" type="button" onClick={() => void resumeDrive(name)} title={t("datasets.action.resume_drive_help")}>
                            <CloudDownload size={15} />
                            {t("datasets.action.resume_drive")}
                          </button>
                        )}
                        {isActive(task) && (
                          <button className="btn sm" type="button" onClick={() => stopUpload(name)} disabled={task?.status === "stopping"} title={t("upload.stop")}>
                            <Square size={15} />
                            {t("upload.stop")}
                          </button>
                        )}
                        {canResume && (
                          <button className="btn primary sm" type="button" onClick={() => void resume(name)}>
                            <Play size={15} />
                            {t("upload.resume")}
                          </button>
                        )}
                        {!active && partial && !fromDrive && !canResume && (
                          <>
                            <button className="btn sm" type="button" onClick={() => askRepick(name, true)} title={t("datasets.action.repick_folder_help")}>
                              <FolderUp size={15} />
                              {t("datasets.action.repick_folder")}
                            </button>
                            <button className="btn icon sm" type="button" onClick={() => askRepick(name, false)} title={t("datasets.action.repick_files")} aria-label={t("datasets.action.repick_files")}>
                              <FileUp size={15} />
                            </button>
                          </>
                        )}
                        {task && !item && !isActive(task) && (
                          <button className="btn icon sm" type="button" onClick={() => forgetUpload(name)} title={t("datasets.action.forget")} aria-label={t("datasets.action.forget")}>
                            <X size={15} />
                          </button>
                        )}
                        {item && (
                          <button className="btn icon sm danger" type="button" onClick={() => setRemoving({ name, item, task, job })} title={t("datasets.action.delete")} aria-label={t("datasets.action.delete")}>
                            <Trash2 size={15} />
                          </button>
                        )}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {removing?.item && (
        <Confirm
          title={t("datasets.delete.title", { name: removing.name })}
          body={
            removing.item.used_by.length
              ? t("datasets.delete.body_used", { size: bytes(removing.item.size_bytes), count: removing.item.used_by.length })
              : t("datasets.delete.body", { size: bytes(removing.item.size_bytes), files: removing.item.files })
          }
          items={removing.item.used_by.length ? removing.item.used_by : [removing.item.path_abs]}
          confirmLabel={t(isActive(removing.task) || jobActive(removing.job) ? "datasets.delete.confirm_running" : "datasets.delete.confirm")}
          danger
          onClose={() => setRemoving(null)}
          onConfirm={async () => {
            await deleteUpload(removing.name);
            live.refreshJobs();
            toast({ kind: "ok", title: t("datasets.delete.done", { name: removing.name, size: bytes(removing.item!.size_bytes) }) });
            list.reload();
          }}
        />
      )}
    </section>
  );
}
