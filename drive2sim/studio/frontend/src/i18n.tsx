// Hai ngôn ngữ của Studio (CONTRACTS mục 10.6): mặc định tiếng Việt, chuyển tiếng Anh bằng nút ở
// thanh bên; lựa chọn lưu trong trình duyệt. Thông báo lỗi từ API chỉ có tiếng Việt; khi đang ở
// tiếng Anh, giao diện dịch phần tiêu đề lỗi theo error.code.
import { createContext, useContext, useState, type ReactNode } from "react";
import en from "./locales/en.json";
import vi from "./locales/vi.json";

export type Locale = "vi" | "en";
const dictionaries: Record<Locale, Record<string, string>> = { vi, en };
const STORAGE_KEY = "d2s.locale";

function initialLocale(): Locale {
  try {
    return localStorage.getItem(STORAGE_KEY) === "en" ? "en" : "vi";
  } catch {
    return "vi";
  }
}

interface I18n {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18n | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);
  const setLocale = (next: Locale) => {
    setLocaleState(next);
    document.documentElement.lang = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // trình duyệt chặn lưu: vẫn đổi ngôn ngữ trong phiên này
    }
  };
  const t = (key: string, vars: Record<string, string | number> = {}) => {
    const template = dictionaries[locale][key] ?? dictionaries.vi[key] ?? key;
    return template.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? `{${name}}`));
  };
  return <I18nContext.Provider value={{ locale, setLocale, t }}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18n {
  const value = useContext(I18nContext);
  if (!value) throw new Error("useI18n cần I18nProvider");
  return value;
}
