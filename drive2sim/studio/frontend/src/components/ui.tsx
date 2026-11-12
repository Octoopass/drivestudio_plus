// Thành phần giao diện dùng ở nhiều màn hình: panel, nhãn trạng thái, hộp lỗi, khung đang tải,
// trạng thái trống, tab, nút chọn, hộp thoại xác nhận, ngăn kéo bên phải, thông báo nổi.
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { ApiError } from "../api/client";
import { useI18n } from "../i18n";

export function Panel({
  title,
  sub,
  tools,
  children,
  flush,
  id,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  tools?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  id?: string;
}) {
  return (
    <section className="panel" id={id}>
      {(title || tools) && (
        <header className="panel-head">
          {title && <h2>{title}</h2>}
          {sub && <span className="sub">{sub}</span>}
          {tools && <div className="tools">{tools}</div>}
        </header>
      )}
      <div className={`panel-body${flush ? " flush" : ""}`}>{children}</div>
    </section>
  );
}

export function PageHead({ title, lede, actions, meta }: { title: ReactNode; lede?: ReactNode; actions?: ReactNode; meta?: ReactNode }) {
  return (
    <div className="page-head">
      <div className="titles">
        <h1>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
        {meta}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

export function Status({ value }: { value: string }) {
  const { t } = useI18n();
  return <span className={`status ${value}`}>{t(`common.status.${value}`)}</span>;
}

export function Skeleton({ height = 16, width = "100%" }: { height?: number; width?: number | string }) {
  return <div className="skeleton" style={{ height, width }} aria-hidden="true" />;
}

export function Loading({ rows = 3 }: { rows?: number }) {
  const { t } = useI18n();
  return (
    <div className="stack" style={{ gap: 10, padding: 4 }} role="status" aria-label={t("common.state.loading")}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} height={18} width={`${90 - i * 12}%`} />
      ))}
    </div>
  );
}

export function Empty({ title, children, action }: { title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children}
      {action}
    </div>
  );
}

/** Hộp lỗi API. Ở tiếng Anh, tiêu đề dịch theo error.code; thông báo gốc tiếng Việt vẫn hiện. */
export function ErrorBox({ error }: { error: ApiError | null | undefined }) {
  const { t, locale } = useI18n();
  if (!error) return null;
  const reasons = (error.details?.reasons as { message: string }[] | undefined) ?? [];
  return (
    <div className="alert" role="alert">
      <XCircle size={16} />
      <div>
        <strong>{locale === "en" ? t(`errors.${error.code}`) : error.message}</strong>
        {locale === "en" && <div className="detail">{error.message}</div>}
        {reasons.length > 0 && (
          <ul>
            {reasons.map((reason) => (
              <li key={reason.message}>{reason.message}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function Note({ kind = "info", children }: { kind?: "info" | "warn"; children: ReactNode }) {
  return (
    <div className={`alert ${kind}`}>
      {kind === "warn" ? <AlertTriangle size={16} /> : <Info size={16} />}
      <div>{children}</div>
    </div>
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  items,
}: {
  value: T;
  onChange: (value: T) => void;
  items: { value: T; label: ReactNode; badge?: ReactNode }[];
}) {
  return (
    <div className="tabs" role="tablist">
      {items.map((item) => (
        <button
          key={item.value}
          role="tab"
          aria-selected={item.value === value}
          onClick={() => onChange(item.value)}
        >
          {item.label}
          {item.badge !== undefined && item.badge !== null && <span className="badge">{item.badge}</span>}
        </button>
      ))}
    </div>
  );
}

export function Segmented<T extends string | number | boolean>({
  value,
  options,
  onChange,
  label,
  disabled,
}: {
  value: T | undefined;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  onChange: (value: T) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          aria-pressed={option.value === value}
          disabled={disabled || option.disabled}
          title={option.title}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** Bắt phím Escape để đóng lớp phủ. */
function useEscape(onClose: () => void) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);
}

/** Hộp thoại xác nhận trước một việc khó hoàn tác (xoá). Liệt kê những gì sẽ bị ảnh hưởng. */
export function Confirm({
  title,
  body,
  items,
  confirmLabel,
  danger,
  onConfirm,
  onClose,
}: {
  title: ReactNode;
  body?: ReactNode;
  items?: string[];
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => Promise<void> | void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  useEscape(onClose);
  useEffect(() => confirmRef.current?.focus(), []);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (err) {
      setError(err as ApiError);
      setBusy(false);
    }
  };
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-label={typeof title === "string" ? title : undefined}>
        <div className="dialog-body">
          <h2>{title}</h2>
          {body && <p className="muted">{body}</p>}
          {items && items.length > 0 && (
            <ul>
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          <ErrorBox error={error} />
        </div>
        <div className="dialog-foot">
          <button className="btn" onClick={onClose}>
            {t("common.action.cancel")}
          </button>
          <button ref={confirmRef} className={`btn ${danger ? "danger solid" : "primary"}`} onClick={run} disabled={busy}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export function Drawer({
  title,
  onClose,
  children,
  foot,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  foot?: ReactNode;
}) {
  const { t } = useI18n();
  useEscape(onClose);
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true">
        <header className="drawer-head">
          <h2 className="grow">{title}</h2>
          <button className="btn ghost icon" onClick={onClose} aria-label={t("common.action.close")}>
            <X size={16} />
          </button>
        </header>
        <div className="drawer-body">{children}</div>
        {foot && <footer className="drawer-foot">{foot}</footer>}
      </aside>
    </div>
  );
}

// ------------------------------------------------------------------ thông báo nổi

interface Toast {
  id: number;
  kind: "ok" | "bad";
  title: string;
  detail?: string;
}

const ToastContext = createContext<(toast: Omit<Toast, "id">) => void>(() => undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((toast: Omit<Toast, "id">) => {
    const id = Date.now() + Math.random();
    setToasts((list) => [...list.slice(-3), { ...toast, id }]);
    window.setTimeout(() => setToasts((list) => list.filter((item) => item.id !== id)), toast.kind === "bad" ? 9000 : 4500);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.kind}`} role={toast.kind === "bad" ? "alert" : "status"}>
            {toast.kind === "ok" ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
            <div className="grow">
              <div>{toast.title}</div>
              {toast.detail && <div className="detail">{toast.detail}</div>}
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}

// ------------------------------------------------------------------ định dạng

export function yesNo(value: boolean | null | undefined, t: (key: string) => string): string {
  if (value === null || value === undefined) return t("common.field.unknown");
  return t(value ? "common.field.yes" : "common.field.no");
}

export function fixed(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "–" : value.toFixed(digits);
}

export function optionsText(options: Record<string, unknown>): string {
  return Object.entries(options)
    .map(([name, value]) => `${name}=${String(value)}`)
    .join(", ");
}

export function bytes(size: number): string {
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "–";
  if (seconds < 60) return `${seconds.toFixed(0)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

/** Thời gian đã qua kể từ một mốc, dạng ngắn theo ngôn ngữ đang dùng. */
export function ago(date: string | number | null | undefined, locale: string): string {
  if (date === null || date === undefined) return "–";
  const ms = typeof date === "number" ? date * 1000 : Date.parse(date);
  const seconds = (ms - Date.now()) / 1000;
  const format = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const steps: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, "second"],
    [3600, "minute"],
    [86400, "hour"],
    [604800, "day"],
    [Infinity, "week"],
  ];
  const divisors = [1, 60, 3600, 86400, 604800];
  const index = steps.findIndex(([limit]) => Math.abs(seconds) < limit);
  return format.format(Math.round(seconds / divisors[index]), steps[index][1]);
}

/** Bảng màu Turbo (Google), xấp xỉ đa thức; t trong [0, 1]. Dùng tô ô metric. */
export function turbo(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const r = 34.61 + x * (1172.33 - x * (10793.56 - x * (33300.12 - x * (38394.49 - x * 14825.05))));
  const g = 23.31 + x * (557.33 + x * (1225.33 - x * (3574.96 - x * (1073.77 + x * 707.56))));
  const b = 27.2 + x * (3211.1 - x * (15327.97 - x * (27814 - x * (22569.18 - x * 6838.66))));
  const clamp = (v: number) => Math.round(Math.min(255, Math.max(0, v)));
  return `rgb(${clamp(r)}, ${clamp(g)}, ${clamp(b)})`;
}
