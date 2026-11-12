// Khung trang: thanh bên tối màu nhựa đường với bốn mục điều hướng và nút đổi ngôn ngữ.
import type { ReactNode } from "react";
import { useI18n } from "../i18n";
import { href, type Route } from "../router";

const MENU: { key: string; route: Route; pages: Route["page"][] }[] = [
  { key: "nav.menu.scenes", route: { page: "scenes" }, pages: ["scenes", "scene"] },
  { key: "nav.menu.new_run", route: { page: "new" }, pages: ["new"] },
  { key: "nav.menu.runs", route: { page: "runs" }, pages: ["runs", "run"] },
  { key: "nav.menu.compare", route: { page: "compare" }, pages: ["compare"] },
];

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  const { t, locale, setLocale } = useI18n();
  return (
    <div className="app">
      <nav className="sidebar" aria-label="drive2sim">
        <div className="brand">
          drive<span>2</span>sim
        </div>
        {MENU.map((item) => (
          <a
            key={item.key}
            href={href(item.route)}
            aria-current={item.pages.includes(route.page) ? "page" : undefined}
          >
            {t(item.key)}
          </a>
        ))}
        <div className="spacer" />
        <div className="lang" role="group" aria-label={t("nav.menu.language")}>
          {(["vi", "en"] as const).map((code) => (
            <button key={code} aria-pressed={locale === code} onClick={() => setLocale(code)}>
              {code === "vi" ? "Tiếng Việt" : "English"}
            </button>
          ))}
        </div>
      </nav>
      <main>{children}</main>
    </div>
  );
}
