// Cấu hình build giao diện Studio. Khi phát triển (npm run dev), chuyển /api và /files sang
// server Studio đang chạy bằng `d2s studio` ở cổng 8000.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
    },
  },
});
