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
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const error = data?.error ?? {};
    throw new ApiError(response.status, error.code ?? "internal", error.message ?? text, error.details);
  }
  return data as T;
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

/** Đọc một đường dẫn API; đổi đường dẫn thì đọc lại; null thì không đọc. */
export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [version, setVersion] = useState(0);

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

  const reload = useCallback(() => setVersion((v) => v + 1), []);
  return { data, error, loading, reload };
}
