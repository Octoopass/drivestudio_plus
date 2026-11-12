// Điểm vào của giao diện Studio. Font Barlow đóng gói kèm (không tải từ mạng), vì Studio có thể
// chạy trên máy GPU không có internet.
import "@fontsource/barlow/400.css";
import "@fontsource/barlow/500.css";
import "@fontsource/barlow/600.css";
import "@fontsource/barlow-semi-condensed/400.css";
import "@fontsource/barlow-semi-condensed/500.css";
import "@fontsource/barlow-semi-condensed/600.css";
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
