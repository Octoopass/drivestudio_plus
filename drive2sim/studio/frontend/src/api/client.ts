// Gọi API của Studio (CONTRACTS mục 11). Lỗi luôn có dạng {"error": {code, message, details}};
// ApiError mang đủ ba trường để giao diện hiện đúng thông báo.
import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  const text = await response.text();
  // Lỗi không lường trước của server trả chữ thường ("Internal Server Error"), không phải JSON:
  // đọc JSON sau khi biết là lỗi, để báo đúng mã HTTP thay vì "SyntaxError: Unexpected token".
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    if (response.ok) throw new ApiError(response.status, "internal", `phản hồi không phải JSON: ${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    const error = (data as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null)?.error ?? {};
    throw new ApiError(response.status, error.code ?? "internal", error.message ?? (text || response.statusText), error.details);
  }
  return data as T;
}

export function post<T>(path: string, body: unknown = {}): Promise<T> {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function del<T>(path: string): Promise<T> {
  return api<T>(path, { method: "DELETE" });
}

/** Đổi lỗi bất kỳ thành ApiError để hiện bằng ErrorBox. */
export function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, "internal", String(err));
}

/** Mã hoá một đoạn đường dẫn. */
export const seg = encodeURIComponent;

/** Đọc một đường dẫn API; đổi đường dẫn thì đọc lại; null thì không đọc. ``pollMs`` thì đọc lại
 * định kỳ (không hiện trạng thái đang tải khi đọc lại). */
export function useApi<T>(path: string | null, pollMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [version, setVersion] = useState(0);

  // Đổi đường dẫn (hoặc thành null) thì bỏ dữ liệu cũ: không thì trang không mount lại (ví dụ Xem
  // 3D khi bỏ chọn run) vẫn hiện dữ liệu của đường dẫn trước tới khi có phản hồi mới.
  useEffect(() => {
    setData(null);
    setError(null);
  }, [path]);

  useEffect(() => {
    if (path === null) return;
    let cancelled = false;
    setLoading(true);
    api<T>(path)
      .then((result) => !cancelled && (setData(result), setError(null)))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err : new ApiError(0, "internal", String(err))))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [path, version]);

  useEffect(() => {
    if (!pollMs || path === null) return;
    const timer = window.setInterval(() => setVersion((v) => v + 1), pollMs);
    return () => window.clearInterval(timer);
  }, [path, pollMs]);

  const reload = useCallback(() => setVersion((v) => v + 1), []);
  return { data, error, loading, reload };
}

/**
 * Video nên mở mặc định trong danh sách video của một run: ảnh render của tập test, rồi của cả
 * đoạn, rồi video bất kỳ. Thứ tự chữ cái đặt bản đồ sai khác (`..._error_maps.mp4`) lên đầu, nên
 * lấy phần tử đầu tiên sẽ ra bản đồ sai khác thay vì ảnh render.
 */
export function preferredVideo<V extends { name: string }>(items: V[]): V | undefined {
  const score = (name: string) =>
    (/error/i.test(name) ? 10 : 0) + (/rgbs?\.mp4$|_rgbs?_|rgb/i.test(name) ? 0 : 2) + (/test_set/i.test(name) ? 0 : 1);
  return [...items].sort((a, b) => score(a.name) - score(b.name) || a.name.localeCompare(b.name))[0];
}
