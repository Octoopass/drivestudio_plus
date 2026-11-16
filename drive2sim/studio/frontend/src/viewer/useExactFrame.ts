// Chế độ "Kết hợp" của trang Xem 3D: kéo, xoay thì trình duyệt vẽ Gói xem trước (mượt); camera đứng
// yên khoảng 0,3 giây thì xin máy chạy Studio vẽ đúng khung đó bằng renderer gốc của model (frame
// server, cùng code với viewer GPU và video của stage train) rồi đè ảnh lên. Camera hay khung đổi
// thì ảnh đè ẩn ngay, chờ lần dừng sau.
import { useEffect, useRef, useState, type RefObject } from "react";
import { api, post, seg } from "../api/client";
import type { SplatViewer } from "./engine";

/** Camera đứng yên bao lâu thì xin ảnh chính xác (ms). */
const SETTLE_MS = 300;
/** Ảnh chính xác cao tối đa bao nhiêu điểm ảnh: cỡ train thường 450–900, vẽ to hơn không nét hơn. */
const MAX_HEIGHT = 900;
const POLL_MS = 1500;

export type ExactState = "off" | "loading" | "ready" | "rendering" | "error";

interface GpuStatus {
  state: "stopped" | "starting" | "running" | "exited";
  kind: string | null;
  run_id: string | null;
  log: string;
}

export function useExactFrame(
  viewer: RefObject<SplatViewer | null>,
  runId: string | undefined,
  active: boolean,
  frame: number,
  playing: boolean,
) {
  const [state, setState] = useState<ExactState>("off");
  const [error, setError] = useState<string | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [renderMs, setRenderMs] = useState<number | null>(null);
  const request = useRef(0);
  const timer = useRef<number | undefined>(undefined);
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const playingRef = useRef(playing);
  playingRef.current = playing;

  // Bật frame server cho run, chờ nó nạp model xong.
  useEffect(() => {
    if (!active || !runId) {
      setState("off");
      return;
    }
    let alive = true;
    setState("loading");
    setError(null);
    const poll = async () => {
      try {
        await post(`/api/runs/${seg(runId)}/frame-server`, {});
        while (alive) {
          const status = await api<GpuStatus>("/api/gpu-viewer");
          if (status.kind !== "frames" || status.run_id !== runId) throw new Error("frame server đã bị thay");
          if (status.state === "running") {
            if (alive) setState("ready");
            return;
          }
          if (status.state === "exited" || status.state === "stopped") throw new Error(status.log.split("\n").slice(-3).join("\n"));
          await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        }
      } catch (err) {
        if (alive) {
          setState("error");
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    void poll();
    return () => {
      alive = false;
    };
  }, [active, runId]);

  // Mỗi lần camera hay khung đổi: ẩn ảnh cũ, hẹn xin ảnh mới khi đứng yên.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !runId || (state !== "ready" && state !== "rendering")) {
      if (v) v.onCameraChange = undefined;
      setImage(null);
      return;
    }
    const schedule = () => {
      request.current += 1;
      setImage(null);
      window.clearTimeout(timer.current);
      if (playingRef.current) return;
      timer.current = window.setTimeout(() => void renderNow(), SETTLE_MS);
    };
    const renderNow = async () => {
      const id = request.current;
      const view = v.viewSize();
      if (!view.width || !view.height) return;
      const height = Math.min(MAX_HEIGHT, Math.round(view.height));
      const width = Math.round((height * view.width) / view.height);
      const { c2w, K } = v.renderRequest(width, height);
      setState("rendering");
      try {
        const response = await fetch(`/api/runs/${seg(runId)}/render-frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ c2w, K, width, height, frame: frameRef.current }),
        });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        if (id !== request.current) return; // camera đã đổi trong lúc vẽ
        setImage((old) => {
          if (old) URL.revokeObjectURL(old);
          return URL.createObjectURL(blob);
        });
        setRenderMs(Number(response.headers.get("X-Render-Ms")) || null);
        setState("ready");
      } catch (err) {
        if (id === request.current) {
          setState("error");
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    v.onCameraChange = schedule;
    schedule();
    return () => {
      v.onCameraChange = undefined;
      window.clearTimeout(timer.current);
    };
  }, [state === "ready" || state === "rendering", runId, frame, playing]);

  return { state, error, image, renderMs };
}
