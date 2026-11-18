// Cấu hình của một run, viết như trang Chạy mới: tên preset (nếu cấu hình khớp một preset), cộng
// các mục cấu hình khác giá trị của preset hay mặc định, bằng nhãn trong file approach
// ("Renderer: 3DGEER"). Thay cho dạng thô "renderer=3dgeer, campose=false, fastgs=false".
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Approach, List, OptionValue } from "../api/types";
import { useI18n } from "../i18n";

let cache: Promise<Record<string, Approach>> | null = null;

/** Mọi file approach, đọc một lần cho cả trang (nhiều dòng bảng dùng chung). */
function useApproaches(): Record<string, Approach> | null {
  const [data, setData] = useState<Record<string, Approach> | null>(null);
  useEffect(() => {
    cache ??= api<List<Approach>>("/api/approaches").then((list) =>
      Object.fromEntries(list.items.map((item) => [item.approach_id, item])),
    );
    let alive = true;
    cache.then((value) => alive && setData(value)).catch(() => (cache = null));
    return () => {
      alive = false;
    };
  }, []);
  return data;
}

function same(a: unknown, b: unknown): boolean {
  return typeof a === typeof b && a === b;
}

/** Tham số stage train khác mặc định (``train_params`` của API), ví dụ ``refine_interval=50``. */
function paramsText(params: Record<string, unknown> | undefined): string[] {
  return Object.entries(params ?? {}).map(([name, value]) => `${name}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`);
}

export function ConfigText({
  approachId,
  options,
  preset,
  params,
  compact = false,
}: {
  approachId: string;
  options: Record<string, OptionValue>;
  preset?: string | null;
  /** Tham số stage train khác mặc định: hai run cùng cấu hình approach vẫn phân biệt được. */
  params?: Record<string, unknown>;
  /** Dạng ngắn cho ô bảng hẹp: bỏ tên mục ("3DGUT" thay cho "Renderer: 3DGUT"), mục bật/tắt chỉ
   * ghi tên khi bật, bỏ phần trong ngoặc; câu đầy đủ nằm ở tooltip. */
  compact?: boolean;
}) {
  const { t } = useI18n();
  const approaches = useApproaches();
  const approach = approaches?.[approachId];
  const extra = paramsText(params);
  if (!approach) {
    const raw = [...Object.entries(options).map(([name, value]) => `${name}=${String(value)}`), ...extra].join(", ");
    return <>{raw || "–"}</>;
  }
  const defaults = Object.fromEntries(Object.entries(approach.options ?? {}).map(([name, spec]) => [name, spec.default]));
  const presetValues = (name: string) => ({ ...defaults, ...(approach.presets?.[name] ?? {}) });
  const matches = (values: Record<string, unknown>) => Object.entries(options).every(([name, value]) => same(values[name], value));
  // Preset ghi trong run, hoặc preset nào có đúng cấu hình này (người dùng tự bấm ra cùng tổ hợp).
  const name = preset ?? Object.keys(approach.presets ?? {}).find((key) => matches(presetValues(key))) ?? null;
  const base = name ? presetValues(name) : defaults;
  const changed = Object.entries(options)
    .filter(([key, value]) => !same(base[key], value))
    .map(([key, value]) => {
      const spec = approach.options?.[key];
      const index = spec ? spec.choices.findIndex((choice) => same(choice, value)) : -1;
      const label = spec?.labels?.[index] ?? String(value);
      const full = `${spec?.label ?? key}: ${label}`;
      if (!compact) return { full, short: full };
      const title = (spec?.label ?? key).replace(/\s*\(.*?\)/g, "");
      const short = typeof value === "boolean" ? (value ? title : t("config.off", { name: title })) : label.replace(/\s*\(.*?\)/g, "");
      return { full, short };
    });
  const head = name ?? (changed.length ? null : t("config.default"));
  const fullText = [head, ...changed.map((c) => c.full), ...extra].filter(Boolean).join(" · ");
  if (!compact) return <>{fullText}</>;
  const shortText = [head, ...changed.map((c) => c.short), ...extra].filter(Boolean).join(" · ");
  return <span title={fullText}>{shortText}</span>;
}
