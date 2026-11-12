// Các thành phần nhỏ dùng ở nhiều màn hình: nhãn trạng thái, hộp lỗi, dòng đang tải, định dạng số.
import type { ApiError } from "../api/client";
import { useI18n } from "../i18n";

export function Status({ value }: { value: string }) {
  const { t } = useI18n();
  return <span className={`status ${value}`}>{t(`common.status.${value}`)}</span>;
}

export function Loading() {
  const { t } = useI18n();
  return <p className="muted">{t("common.state.loading")}</p>;
}

/** Hiện lỗi API. Ở tiếng Anh, tiêu đề dịch theo error.code; thông báo gốc tiếng Việt vẫn hiện. */
export function ErrorBox({ error }: { error: ApiError | null }) {
  const { t, locale } = useI18n();
  if (!error) return null;
  const reasons = (error.details?.reasons as { message: string }[] | undefined) ?? [];
  return (
    <div className="error" role="alert">
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
  );
}

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
