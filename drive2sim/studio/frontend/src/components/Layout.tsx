// Khung trang: thanh điều hướng bên trái, thanh trên có đường dẫn trang, nút hoạt động (run đang
// chạy và việc nền), ô tìm nhanh (Ctrl+K), nút đổi giao diện sáng/tối và ngôn ngữ.
import {
  Box,
  Boxes,
  Gauge,
  HardDrive,
  Languages,
  LayoutDashboard,
  ListChecks,
  Monitor,
  Moon,
  Play,
  Search,
  Sun,
  Trophy,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useApi } from "../api/client";
import type { List, RunRecord } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { href, type Page, type Route } from "../router";
import { ActivityDrawer } from "./ActivityDrawer";
import { InfoTip } from "./ui";
import { CommandPalette } from "./CommandPalette";

interface MenuItem {
  key: string;
  route: Route;
  pages: Page[];
  icon: ReactNode;
}

const ICON = 17;
const MENU: { group: string | null; items: MenuItem[] }[] = [
  {
    group: null,
    items: [{ key: "nav.menu.overview", route: { page: "overview" }, pages: ["overview"], icon: <LayoutDashboard size={ICON} /> }],
  },
  {
    group: "nav.group.data",
    items: [{ key: "nav.menu.scenes", route: { page: "scenes" }, pages: ["scenes", "scene"], icon: <Boxes size={ICON} /> }],
  },
  {
    group: "nav.group.experiments",
    items: [
      { key: "nav.menu.new_run", route: { page: "new" }, pages: ["new"], icon: <Play size={ICON} /> },
      { key: "nav.menu.runs", route: { page: "runs" }, pages: ["runs", "run"], icon: <ListChecks size={ICON} /> },
      { key: "nav.menu.compare", route: { page: "compare" }, pages: ["compare"], icon: <Trophy size={ICON} /> },
    ],
  },
  {
    group: "nav.group.results",
    items: [
      { key: "nav.menu.viewer", route: { page: "viewer" }, pages: ["viewer"], icon: <Box size={ICON} /> },
      { key: "nav.menu.storage", route: { page: "storage" }, pages: ["storage"], icon: <HardDrive size={ICON} /> },
    ],
  },
];

type Theme = "system" | "light" | "dark";
const THEME_KEY = "d2s.theme";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem(THEME_KEY) as Theme) || "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // trình duyệt chặn lưu: vẫn đổi trong phiên này
    }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "dark" : t === "dark" ? "light" : "system"));
  return [theme, next];
}

function Crumbs({ route }: { route: Route }) {
  const { t } = useI18n();
  const parent = (page: Page, label: string) => (
    <>
      <a href={href({ page } as Route)}>{label}</a>
      <span aria-hidden="true">/</span>
    </>
  );
  return (
    <nav className="crumbs" aria-label={t("nav.crumbs")}>
      {route.page === "scene" && (
        <>
          {parent("scenes", t("nav.menu.scenes"))}
          <span className="here ellipsis">{route.sceneId}</span>
        </>
      )}
      {route.page === "run" && (
        <>
          {parent("runs", t("nav.menu.runs"))}
          <span className="here ellipsis">{route.runId}</span>
        </>
      )}
      {route.page !== "scene" && route.page !== "run" && <span className="here">{t(`nav.menu.${route.page === "new" ? "new_run" : route.page}`)}</span>}
    </nav>
  );
}

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  const { t, locale, setLocale } = useI18n();
  const live = useLive();
  const [theme, nextTheme] = useTheme();
  const [palette, setPalette] = useState(false);
  const [activity, setActivity] = useState(false);
  // "Đang chạy" = đang chạy + đang chờ, cùng cách đếm với trang Tổng quan và ngăn Hoạt động.
  const running = useApi<List<RunRecord>>(`/api/runs?status=running&_=${live.tick}`);
  const pending = useApi<List<RunRecord>>(`/api/runs?status=pending&_=${live.tick}`);
  const studio = useApi<{ workdir: string; demo: boolean }>("/api/studio");
  const runningCount = (running.data?.total ?? 0) + (pending.data?.total ?? 0);
  const activeJobs = live.jobs.filter((job) => job.status === "queued" || job.status === "running").length;
  const busy = runningCount + activeJobs;

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPalette((open) => !open);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => {
    document.title = `${t(`nav.menu.${route.page === "new" ? "new_run" : route.page === "scene" ? "scenes" : route.page === "run" ? "runs" : route.page}`)} · drive2sim Studio`;
  }, [route, locale]);

  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
  return (
    <div className="shell">
      <nav className="nav" aria-label="drive2sim">
        <a className="brand" href="#/">
          <span className="brand-mark" aria-hidden="true">
            <Gauge size={20} />
          </span>
          <span className="brand-name">
            drive<b>2</b>sim
          </span>
        </a>
        {MENU.map((section) => (
          <div key={section.group ?? "top"} style={{ display: "contents" }}>
            {section.group && <div className="nav-group">{t(section.group)}</div>}
            {section.items.map((item) => (
              <a key={item.key} href={href(item.route)} aria-current={item.pages.includes(route.page) ? "page" : undefined} title={t(item.key)}>
                {item.icon}
                <span className="label-long">{t(item.key)}</span>
                {item.key === "nav.menu.runs" && runningCount > 0 && <span className="count label-long">{runningCount}</span>}
              </a>
            ))}
          </div>
        ))}
      </nav>
      <div className="main">
        <header className="topbar">
          <Crumbs route={route} />
          {studio.data?.demo && (
            <span className="demo-badge">
              {t("nav.demo.label")}
              <InfoTip>{t("nav.demo.info", { workdir: studio.data.workdir })}</InfoTip>
            </span>
          )}
          <button className="search-button" onClick={() => setPalette(true)}>
            <Search size={19} />
            <span className="label">{t("nav.search.button")}</span>
            <kbd>Ctrl K</kbd>
          </button>
          <button className={`activity${busy ? " busy" : ""}`} onClick={() => setActivity(true)} aria-label={t("nav.activity.title")}>
            <span className="pulse" />
            {busy ? t("nav.activity.busy", { count: busy }) : t("nav.activity.idle")}
          </button>
          <button className="btn ghost icon" onClick={nextTheme} title={t(`nav.theme.${theme}`)} aria-label={t(`nav.theme.${theme}`)}>
            <ThemeIcon size={20} />
          </button>
          <button
            className="btn ghost sm"
            onClick={() => setLocale(locale === "vi" ? "en" : "vi")}
            title={t("nav.menu.language")}
          >
            <Languages size={19} />
            {locale === "vi" ? "VI" : "EN"}
          </button>
        </header>
        <main className="content">{children}</main>
      </div>
      {palette && <CommandPalette onClose={() => setPalette(false)} />}
      {activity && <ActivityDrawer onClose={() => setActivity(false)} />}
    </div>
  );
}
