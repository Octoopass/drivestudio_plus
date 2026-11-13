// Điểm vào của giao diện Studio. Font Be Vietnam Pro đóng gói kèm (không tải từ mạng), vì Studio có thể
// chạy trên máy GPU không có internet.
import "@fontsource/be-vietnam-pro/400.css";
import "@fontsource/be-vietnam-pro/500.css";
import "@fontsource/be-vietnam-pro/600.css";
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
