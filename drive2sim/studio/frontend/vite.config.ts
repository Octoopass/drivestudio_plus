// Cấu hình build giao diện Studio. Khi phát triển (npm run dev hoặc make dev), chuyển /api và
// /files sang server Studio đang chạy bằng `d2s studio`, ở cổng D2S_STUDIO_PORT (mặc định 8000).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const studio = `http://127.0.0.1:${process.env.D2S_STUDIO_PORT ?? "8000"}`;

export default defineConfig({
  plugins: [react()],
  // Trang xem 3D nạp lười một gói khoảng 3,3 MB (three.js và Spark kèm WASM); các trang khác không
  // tải gói này, nên nâng ngưỡng cảnh báo thay vì chia nhỏ.
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 3500 },
  server: {
    proxy: {
      "/api": studio,
      "/files": studio,
    },
  },
});
