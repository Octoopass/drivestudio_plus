// Tìm nhanh (Ctrl+K): gõ vài chữ để nhảy tới một màn hình, một cảnh hoặc một run. Phím lên/xuống
// chọn, Enter mở, Escape đóng.
import { Boxes, CornerDownLeft, ListChecks, Navigation } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useApi } from "../api/client";
import type { List, Manifest, RunRecord } from "../api/types";
import { useI18n } from "../i18n";
import { go, type Route } from "../router";

interface Item {
  id: string;
  label: string;
  kind: string;
  icon: ReactNode;
  route: Route;
}

const PAGES: { key: string; route: Route }[] = [
  { key: "nav.menu.overview", route: { page: "overview" } },
  { key: "nav.menu.scenes", route: { page: "scenes" } },
  { key: "nav.menu.new_run", route: { page: "new" } },
  { key: "nav.menu.runs", route: { page: "runs" } },
  { key: "nav.menu.compare", route: { page: "compare" } },
  { key: "nav.menu.viewer", route: { page: "viewer" } },
  { key: "nav.menu.storage", route: { page: "storage" } },
];

export function CommandPalette({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const runs = useApi<List<RunRecord>>("/api/runs");
  useEffect(() => input.current?.focus(), []);

  const items = useMemo<Item[]>(() => {
    const all: Item[] = [
      ...PAGES.map((p) => ({ id: p.key, label: t(p.key), kind: t("palette.kind.page"), icon: <Navigation size={19} />, route: p.route })),
      ...(scenes.data?.items ?? []).map((m) => ({
        id: m.scene_id,
        label: m.scene_id,
        kind: t("palette.kind.scene"),
        icon: <Boxes size={19} />,
        route: { page: "scene", sceneId: m.scene_id } as Route,
      })),
      ...(runs.data?.items ?? []).map((r) => ({
        id: r.run_id,
        label: `${r.run_id}`,
        kind: `${t("palette.kind.run")} ${t(`common.status.${r.status}`)}`,
        icon: <ListChecks size={19} />,
        route: { page: "run", runId: r.run_id } as Route,
      })),
    ];
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    return all.filter((item) => words.every((w) => `${item.label} ${item.kind}`.toLowerCase().includes(w))).slice(0, 40);
  }, [query, scenes.data, runs.data, t]);

  useEffect(() => setCursor(0), [query]);

  const open = (item: Item | undefined) => {
    if (!item) return;
    go(item.route);
    onClose();
  };
  const onKey = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowDown") setCursor((c) => Math.min(items.length - 1, c + 1));
    else if (event.key === "ArrowUp") setCursor((c) => Math.max(0, c - 1));
    else if (event.key === "Enter") open(items[cursor]);
    else if (event.key === "Escape") onClose();
    else return;
    event.preventDefault();
  };

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-modal="true" aria-label={t("nav.search.button")}>
        <input
          ref={input}
          type="search"
          value={query}
          placeholder={t("palette.placeholder")}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKey}
          role="combobox"
          aria-expanded="true"
          aria-controls="palette-list"
        />
        {items.length === 0 ? (
          <div className="none">{t("palette.none")}</div>
        ) : (
          <ul id="palette-list" role="listbox">
            {items.map((item, index) => (
              <li
                key={`${item.kind}:${item.id}`}
                role="option"
                aria-selected={index === cursor}
                onMouseEnter={() => setCursor(index)}
                onClick={() => open(item)}
              >
                {item.icon}
                <span className="ellipsis">{item.label}</span>
                <span className="kind">{item.kind}</span>
                {index === cursor && <CornerDownLeft size={16} className="faint" />}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
