// Cấu hình của một run: renderer, rồi các mục cấu hình khác giá trị mặc định, bằng nhãn trong file
// approach
// ("Renderer: 3DGEER"). Thay cho dạng thô "renderer=3dgeer, campose=false, fastgs=false".
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Approach, List, OptionValue } from "../api/types";
import { useI18n } from "../i18n";

let cache: Promise<Record<string, Approach>> | null = null;

/** Mọi file approach, đọc một lần cho cả trang (nhiều dòng bảng dùng chung). */
export function useApproaches(): Record<string, Approach> | null {
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

/** Mục cấu hình luôn hiện trong ô cấu hình, kể cả khi bằng giá trị mặc định. */
const ALWAYS_SHOWN = ["renderer"];

function same(a: unknown, b: unknown): boolean {
  return typeof a === typeof b && a === b;
}

/** Một tham số viết bằng nhãn của file approach khi có ("FastGS: Bật"), không thì ``tên=giá trị``.
 * Dùng chung cho mọi chỗ hiện tham số, để cùng một mục không lúc ``fastgs`` lúc ``FastGS``. */
export function paramLabel(approach: Approach | undefined, name: string, value: unknown): string {
  const spec = approach?.options?.[name] ?? approach?.retired_options?.[name];
  if (spec) {
    const index = spec.choices.findIndex((choice) => same(choice, value));
    return `${spec.label ?? name}: ${spec.labels?.[index] ?? String(value)}`;
  }
  return `${name}=${typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}`;
}

/** Danh sách tham số viết bằng ``paramLabel``, nối bằng dấu phẩy. */
export function paramsLabel(approach: Approach | undefined, params: Record<string, unknown> | undefined): string {
  return Object.entries(params ?? {}).map(([name, value]) => paramLabel(approach, name, value)).join(", ");
}

/** Tham số stage train khác mặc định (``train_params`` của API), ví dụ ``refine_interval=50``. */
function paramsText(approach: Approach | undefined, params: Record<string, unknown> | undefined): string[] {
  return Object.entries(params ?? {}).map(([name, value]) => paramLabel(approach, name, value));
}

export function ConfigText({
  approachId,
  options,
  params,
  compact = false,
}: {
  approachId: string;
  options: Record<string, OptionValue>;
  /** Preset ghi trong run; giữ để các trang gọi không phải đổi, không còn dùng để hiển thị. */
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
  const extra = paramsText(approach, params);
  if (!approach) {
    // Chưa đọc xong file approach: chưa có nhãn, tạm để trống thay vì hiện tên thô.
    if (approaches === null) return <>…</>;
    const raw = [...Object.entries(options).map(([name, value]) => paramLabel(undefined, name, value)), ...extra].join(", ");
    return <>{raw || "–"}</>;
  }
  const defaults = Object.fromEntries(Object.entries(approach.options ?? {}).map(([name, spec]) => [name, spec.default]));
  // So với giá trị mặc định và hiện nhãn từng mục, không hiện tên preset: run khớp đúng một preset
  // (ví dụ "fastgs") trước đây hiện tên kỹ thuật viết thường, khác run cùng mục mà không khớp preset.
  const base = defaults;
  // Mục đã gỡ (retired_options): run cũ còn ghi nó; so với giá trị mặc định cũ, nhãn lấy từ đó.
  const retired = approach.retired_options ?? {};
  const baseOf = (key: string) => (key in retired ? retired[key].default : base[key]);
  const specOf = (key: string) => approach.options?.[key] ?? retired[key];
  // Mục luôn hiện kể cả khi bằng mặc định (ô trống dễ đọc nhầm khi so run), đặt lên đầu.
  const always = (key: string) => ALWAYS_SHOWN.includes(key);
  const changed = Object.entries(options)
    .filter(([key, value]) => always(key) || !same(baseOf(key), value))
    .sort(([a], [b]) => Number(always(b)) - Number(always(a)))
    .map(([key, value]) => {
      const spec = specOf(key);
      const index = spec ? spec.choices.findIndex((choice) => same(choice, value)) : -1;
      const label = spec?.labels?.[index] ?? String(value);
      const full = `${spec?.label ?? key}: ${label}`;
      if (!compact) return { full, short: full };
      const title = (spec?.label ?? key).replace(/\s*\(.*?\)/g, "");
      const short = typeof value === "boolean" ? (value ? title : t("config.off", { name: title })) : label.replace(/\s*\(.*?\)/g, "");
      return { full, short, first: always(key) };
    });
  const firsts = changed.filter((c) => c.first);
  const rest = changed.filter((c) => !c.first);
  const head = rest.length ? null : t("config.default");
  // Renderer đứng đầu, rồi các mục khác: mọi dòng của bảng cùng thứ tự.
  const text = (pick: (c: (typeof changed)[number]) => string) =>
    [...firsts.map(pick), head, ...rest.map(pick), ...extra].filter(Boolean).join(" · ");
  const fullText = text((c) => c.full);
  if (!compact) return <>{fullText}</>;
  const shortText = text((c) => c.short);
  return <span title={fullText}>{shortText}</span>;
}
