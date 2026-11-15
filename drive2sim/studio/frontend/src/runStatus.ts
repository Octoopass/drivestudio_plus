// Trạng thái run hiện trên Studio. Run Kaggle giữ "pending" trên máy tới khi tải kết quả về, dù
// notebook đang chạy hay đã xong; Studio đọc trạng thái notebook lần hỏi gần nhất (trường "remote"
// của /api/runs, lấy từ kaggle.json) để hiện đúng:
//   chưa đẩy lên            → not_pushed
//   đang chờ máy Kaggle     → pending
//   đang chạy trên Kaggle   → running
//   xong, chưa tải về       → remote_done
//   lỗi, huỷ, chưa tải về   → remote_failed
//   chưa hỏi được Kaggle    → remote_unknown (ví dụ run đẩy bằng tài khoản Kaggle khác)
import type { RunRecord } from "./api/types";

export interface RemoteState {
  pushed: boolean;
  status: string | null;
  checked_utc: string | null;
}

export type RunItem = RunRecord & { remote?: RemoteState | null };

export function runStatus(run: RunItem, live?: string): string {
  const status = live ?? run.status;
  const remote = run.remote;
  if (status !== "pending" || !remote) return status;
  if (!remote.pushed) return "not_pushed";
  switch (remote.status) {
    case "running":
      return "running";
    case "queued":
    case "new":
      return "pending";
    case "complete":
      return "remote_done";
    case "error":
    case "cancel_requested":
    case "cancel_acknowledged":
      return "remote_failed";
    default:
      return "remote_unknown";
  }
}
