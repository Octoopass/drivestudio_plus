// Hàng đợi tải dữ liệu gốc lên uploads/<tên>/ (CONTRACTS mục 11.1), dùng chung cho cả Studio: đóng
// ngăn nạp cảnh hay đổi trang thì việc tải vẫn chạy, trang Lưu trữ dừng, tải tiếp hoặc xoá được.
// File đã chọn chỉ nằm trong bộ nhớ của tab này: tải lại trang thì chọn lại thư mục để tải tiếp,
// phần đã có trên máy chạy Studio không gửi lại (plan_upload trả offset của từng file).
import { useSyncExternalStore } from "react";
import { ApiError, asApiError, del, post, seg } from "./api/client";

export const CHUNK = 8 * 1024 * 1024;
export const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;

export type UploadStatus = "running" | "stopping" | "paused" | "done" | "failed";

export interface Picked {
  file: File;
  path: string;
}

export interface UploadTask {
  name: string;
  status: UploadStatus;
  /** byte đã gửi trong lần chạy này / byte còn thiếu lúc bắt đầu */
  sent: number;
  total: number;
  file: string;
  rate: number;
  error: ApiError | null;
  pathAbs: string;
  count: number;
  size: number;
}

interface Plan {
  name: string;
  path_abs: string;
  files: { path: string; size: number; offset: number; done: boolean }[];
  needed_bytes: number;
}

const tasks = new Map<string, UploadTask>();
const picks = new Map<string, Picked[]>();
const runs = new Map<string, Promise<UploadTask>>();
const aborts = new Map<string, AbortController>();
const listeners = new Set<() => void>();
let snapshot: UploadTask[] = [];

function emit() {
  snapshot = [...tasks.values()];
  listeners.forEach((listener) => listener());
}

function patch(name: string, change: Partial<UploadTask>) {
  const task = tasks.get(name);
  if (!task) return;
  tasks.set(name, { ...task, ...change });
  emit();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Mọi việc tải của tab này (đang chạy, đã dừng, xong, lỗi). */
export function useUploads(): UploadTask[] {
  return useSyncExternalStore(subscribe, () => snapshot);
}

export function isActive(task: UploadTask | undefined): boolean {
  return task?.status === "running" || task?.status === "stopping";
}

/** Tab này còn giữ file đã chọn cho lần tải ``name`` (tải tiếp được mà không chọn lại). */
export function hasPicked(name: string): boolean {
  return picks.has(name);
}

// Rời trang khi đang tải thì trình duyệt hỏi lại (phần đã gửi vẫn còn, nhưng phải chọn lại file).
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", (event) => {
    if ([...tasks.values()].some(isActive)) event.preventDefault();
  });
}

async function run(name: string, list: Picked[]): Promise<UploadTask> {
  const abort = new AbortController();
  aborts.set(name, abort);
  const byPath = new Map(list.map((p) => [p.path, p.file]));
  const plan = () => post<Plan>("/api/sources/uploads", { name, files: list.map((p) => ({ path: p.path, size: p.file.size })) });
  try {
    let current = await plan();
    const total = current.needed_bytes;
    let sent = 0;
    const began = performance.now();
    patch(name, { total, pathAbs: current.path_abs });
    for (const entry of current.files) {
      const file = byPath.get(entry.path);
      if (!file || entry.done) continue;
      let offset = entry.offset;
      while (offset < entry.size) {
        if (abort.signal.aborted) throw new DOMException("stopped", "AbortError");
        const blob = file.slice(offset, Math.min(entry.size, offset + CHUNK));
        const query = `path=${seg(entry.path)}&offset=${offset}&total=${entry.size}`;
        const response = await fetch(`/api/sources/uploads/${seg(name)}/chunk?${query}`, { method: "POST", body: blob, signal: abort.signal });
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
        patch(name, { sent, file: entry.path, rate: seconds > 0 ? sent / seconds : 0 });
      }
    }
    patch(name, { status: "done", sent: total, file: "", rate: 0 });
  } catch (err) {
    // Huỷ giữa chừng một khối: máy chủ chỉ ghi khi nhận đủ thân, nên phần đã có vẫn liền mạch.
    if (abort.signal.aborted) patch(name, { status: "paused", rate: 0 });
    else patch(name, { status: "failed", rate: 0, error: asApiError(err) });
  } finally {
    aborts.delete(name);
    runs.delete(name);
  }
  return tasks.get(name)!;
}

/**
 * Bắt đầu hoặc tải tiếp lần tải ``name``. ``picked`` thay file đã giữ (chọn lại thư mục sau khi tải
 * lại trang); không có thì dùng file đã chọn trước. Trả trạng thái khi việc tải dừng.
 */
export function startUpload(name: string, picked?: Picked[]): Promise<UploadTask> {
  const running = runs.get(name);
  if (running) return running;
  if (picked) picks.set(name, picked);
  const list = picks.get(name);
  if (!list?.length) return Promise.reject(new ApiError(0, "invalid", "chưa chọn file"));
  tasks.set(name, {
    name,
    status: "running",
    sent: 0,
    total: 0,
    file: "",
    rate: 0,
    error: null,
    pathAbs: tasks.get(name)?.pathAbs ?? "",
    count: list.length,
    size: list.reduce((sum, p) => sum + p.file.size, 0),
  });
  emit();
  const promise = run(name, list);
  runs.set(name, promise);
  return promise;
}

/** Dừng ngay (huỷ khối đang gửi); bấm tải tiếp thì đi tiếp từ phần đã có. */
export function stopUpload(name: string) {
  const abort = aborts.get(name);
  if (!abort) return;
  patch(name, { status: "stopping" });
  abort.abort();
}

/** Dừng việc tải (nếu đang chạy) rồi xoá thư mục ``uploads/<tên>/`` trên máy chạy Studio. */
export async function deleteUpload(name: string) {
  const running = runs.get(name);
  if (running) {
    stopUpload(name);
    await running;
  }
  await del(`/api/sources/uploads/${seg(name)}`);
  forgetUpload(name);
}

/** Bỏ lần tải khỏi danh sách của tab này (không đụng tới dữ liệu đã tải). */
export function forgetUpload(name: string) {
  if (runs.has(name)) return;
  tasks.delete(name);
  picks.delete(name);
  emit();
}

/** Đường dẫn tương đối của file: với thư mục bỏ tên thư mục gốc (trùng tên lần tải). */
export function relativePath(file: File): string {
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return rel ? rel.split("/").slice(1).join("/") : file.name;
}

/** Đổi danh sách file trình duyệt chọn thành Picked, kèm tên gợi ý (tên thư mục gốc). */
export function pickedFrom(list: FileList, folder: boolean): { picked: Picked[]; suggested: string } {
  const picked = Array.from(list).map((file) => ({ file, path: folder ? relativePath(file) : file.name }));
  const root = folder ? (list[0] as File & { webkitRelativePath?: string }).webkitRelativePath?.split("/")[0] : "";
  const suggested = (root || "upload").replace(/[^A-Za-z0-9_.-]/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80) || "upload";
  return { picked, suggested };
}

export function gib(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MiB`;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s >= 3600 ? `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
