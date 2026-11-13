// Điểm vào của giao diện Studio. Font đóng gói kèm (không tải từ mạng), vì Studio có thể
// chạy trên máy GPU không có internet.
// Bộ chữ của P-087: Plus Jakarta Sans cho chữ giao diện, IBM Plex Mono cho nhãn kỹ thuật và giá trị
// (ID, số liệu, thời điểm). Cả hai có bộ chữ tiếng Việt, giấy phép SIL OFL 1.1.
import "@fontsource/plus-jakarta-sans/400.css";
import "@fontsource/plus-jakarta-sans/500.css";
import "@fontsource/plus-jakarta-sans/600.css";
import "@fontsource/plus-jakarta-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ToastProvider } from "./components/ui";
import { I18nProvider } from "./i18n";
import { LiveProvider } from "./live";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <ToastProvider>
        <LiveProvider>
          <App />
        </LiveProvider>
      </ToastProvider>
    </I18nProvider>
  </StrictMode>,
);
