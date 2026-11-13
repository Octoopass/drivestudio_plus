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

export function ConfigText({ approachId, options, preset }: { approachId: string; options: Record<string, OptionValue>; preset?: string | null }) {
  const { t } = useI18n();
  const approaches = useApproaches();
  const approach = approaches?.[approachId];
  if (!approach) {
    const raw = Object.entries(options).map(([name, value]) => `${name}=${String(value)}`).join(", ");
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
      return `${spec?.label ?? key}: ${label}`;
    });
  const parts = [name ?? (changed.length ? null : t("config.default")), ...changed].filter(Boolean);
  return <>{parts.join(" · ")}</>;
}
