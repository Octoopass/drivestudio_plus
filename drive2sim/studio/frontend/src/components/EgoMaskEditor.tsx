// Công cụ vẽ mask thân xe (drive2sim/studio/api/ego_mask_tool.py): chọn camera, bấm điểm lên ảnh
// trung vị (chuột trái: thân xe, chuột phải: không phải), SAM2.1 tách mask nháp, người dùng xem rồi
// lưu vào ego_masks/<camera>.png của cảnh. Loader dùng mask này để bỏ thân xe khỏi loss, khỏi chấm
// điểm và khỏi bước chiếu LiDAR.
import { Ban, Check, Eraser, LoaderCircle, MousePointer2, Save, Trash2, Undo2 } from "lucide-react";
import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from "react";
import { asApiError, del, post, seg, useApi, type ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { ErrorBox, InfoTip, Loading } from "./ui";

interface EgoMaskCamera {
  camera_id: string;
  /** mask: có mask; none: không thấy thân xe; pending: chưa ai xác nhận (drive2sim/ingest/ego_masks.py). */
  state: "mask" | "none" | "pending";
  source: "scene" | "tool" | "dataset" | "confirmed" | null;
  has_mask: boolean;
  coverage: number | null;
  has_draft: boolean;
}

interface EgoMaskStatus {
  sam2_ready: boolean;
  sam2_hint: string;
  cameras: EgoMaskCamera[];
}

type Point = [number, number, 0 | 1];

const pct = (v: number | null | undefined) => (v == null ? "–" : `${(100 * v).toFixed(1)}%`);

export function EgoMaskEditor({ sceneId }: { sceneId: string }) {
  const { t } = useI18n();
  const status = useApi<EgoMaskStatus>(`/api/scenes/${seg(sceneId)}/egomask`);
  const [camera, setCamera] = useState<string | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [draft, setDraft] = useState<{ coverage: number; version: number } | null>(null);
  const [savedVersion, setSavedVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState("segmenting");
  const [mode, setMode] = useState<"points" | "erase">("points");
  const [radius, setRadius] = useState(24);
  const stroke = useRef<[number, number][]>([]);
  const preview = useRef<HTMLCanvasElement>(null);
  const [imageError, setImageError] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  const cameras = status.data?.cameras ?? [];
  const pending = cameras.filter((c) => c.state === "pending");
  // Mở sẵn camera đầu tiên còn chưa xử lý, để người dùng đi lần lượt hết các camera.
  const current = camera ?? pending[0]?.camera_id ?? cameras[0]?.camera_id ?? null;
  const info = cameras.find((c) => c.camera_id === current);
  const base = current ? `/api/scenes/${seg(sceneId)}/egomask/${seg(current)}` : "";

  useEffect(() => {
    setPoints([]);
    setDraft(null);
    setSize(null);
    setError(null);
    setImageError(false);
    setMode("points");
  }, [current]);

  const run = async (next: Point[]) => {
    setPoints(next);
    if (!next.some((p) => p[2] === 1)) {
      setDraft(null);
      return;
    }
    setOperation("segmenting");
    setBusy(true);
    setError(null);
    try {
      const out = await post<{ coverage: number }>(`${base}/segment`, { points: next });
      setDraft({ coverage: out.coverage, version: Date.now() });
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };

  const click = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (mode !== "points" || busy || !size || !status.data?.sam2_ready) return;
    const box = event.currentTarget.getBoundingClientRect();
    const x = Math.min(size.w - 1, Math.max(0, Math.round(((event.clientX - box.left) / box.width) * size.w)));
    const y = Math.min(size.h - 1, Math.max(0, Math.round(((event.clientY - box.top) / box.height) * size.h)));
    void run([...points, [x, y, event.button === 2 ? 0 : 1]]);
  };

  const draw = (event: PointerEvent<HTMLCanvasElement>) => {
    if (busy || !size || mode !== "erase" || !(draft || info?.has_mask)) return;
    if (event.type === "pointerdown") {
      if (event.button !== 0) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      stroke.current = [];
    } else if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const box = event.currentTarget.getBoundingClientRect();
    const point: [number, number] = [
      Math.max(0, Math.min(size.w - 1, (event.clientX - box.left) / box.width * size.w)),
      Math.max(0, Math.min(size.h - 1, (event.clientY - box.top) / box.height * size.h)),
    ];
    const context = event.currentTarget.getContext("2d");
    const previous = stroke.current.at(-1) ?? point;
    stroke.current.push(point);
    if (context) {
      context.strokeStyle = "rgba(255,255,255,0.8)";
      context.lineWidth = radius * 2;
      context.lineCap = "round";
      context.beginPath();
      context.moveTo(...previous);
      context.lineTo(...point);
      context.stroke();
      context.beginPath();
      context.arc(...point, radius, 0, Math.PI * 2);
      context.fillStyle = "rgba(255,255,255,0.45)";
      context.fill();
    }
  };

  const finishStroke = async (event: PointerEvent<HTMLCanvasElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    const next = stroke.current;
    stroke.current = [];
    if (!next.length) return;
    setOperation("erasing");
    setBusy(true);
    setError(null);
    try {
      const out = await post<{ coverage: number }>(`${base}/erase`, { points: next, radius });
      setDraft({ coverage: out.coverage, version: Date.now() });
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
      preview.current?.getContext("2d")?.clearRect(0, 0, size!.w, size!.h);
    }
  };

  const act = async (fn: () => Promise<unknown>) => {
    setOperation("saving");
    setBusy(true);
    setError(null);
    try {
      await fn();
      setPoints([]);
      setDraft(null);
      setSavedVersion(Date.now());
      status.reload();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };

  if (status.error) return <ErrorBox error={status.error} />;
  if (!status.data) return <Loading rows={2} />;
  const overlay = draft
    ? `${base}/overlay?draft=true&v=${draft.version}`
    : info?.has_mask
      ? `${base}/overlay?v=${savedVersion}`
      : null;

  return (
    <div className="stack" style={{ gap: 10 }}>
      <h3 className="row" style={{ gap: 6, margin: 0, fontSize: 17 }}>
        {t("scene.egomask.title")}
        <InfoTip>{t("scene.egomask.hint")}</InfoTip>
      </h3>
      {!status.data.sam2_ready && <p className="small muted">{t("scene.egomask.no_sam2", { hint: status.data.sam2_hint })}</p>}
      <p className="small" style={pending.length ? { color: "var(--warn)" } : undefined}>
        {pending.length ? t("scene.egomask.pending", { count: pending.length, total: cameras.length }) : t("scene.egomask.all_done")}
      </p>
      <ErrorBox error={error} />
      {current && (
        <>
          <p className="small muted" style={{ margin: 0 }}>{t("scene.egomask.how_to")}</p>
          <div className="egomask-layout">
            <aside><div className="egomask-cameras" aria-label={t("scene.egomask.choose_camera")}>
        {cameras.map((c) => (
          <button key={c.camera_id} type="button" className="egomask-camera" disabled={busy} aria-pressed={c.camera_id === current} onClick={() => setCamera(c.camera_id)}>
            <span className="egomask-camera-name">{c.camera_id}{c.camera_id === current && <Check size={16} />}</span>
            <span className="muted" style={c.state === "pending" ? { color: "var(--warn)" } : undefined}>
              {" "}
              {c.state === "pending"
                ? t("scene.egomask.state_pending")
                : c.state === "none"
                  ? t("scene.egomask.state_none")
                  : c.has_mask
                    ? pct(c.coverage)
                    : t(`scene.egomask.source_${c.source}`)}
            </span>
          </button>
        ))}
      </div></aside>
            <div className="egomask-workspace">
          <div
            className="egomask-image"
            aria-busy={busy || (!size && !imageError)}
            style={{ cursor: busy ? "progress" : "crosshair" }}
            onClick={click}
            onContextMenu={click}
          >
            <img
              key={base}
              src={`${base}/median`}
              onError={() => setImageError(true)}
              alt={current}
              style={{ width: "100%", display: "block" }}
              draggable={false}
              onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
            {overlay && <img src={overlay} alt="" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} draggable={false} />}
            {mode === "erase" && size && <canvas ref={preview} width={size.w} height={size.h} className="egomask-canvas" onPointerDown={draw} onPointerMove={draw} onPointerUp={(e) => void finishStroke(e)} onPointerCancel={(e) => { stroke.current = []; e.currentTarget.getContext("2d")?.clearRect(0, 0, size.w, size.h); }} />}
            {(busy || !size || imageError) && <div className="egomask-loading" role={imageError ? "alert" : "status"} aria-live="polite">
              {!imageError && <LoaderCircle size={22} className="egomask-spinner" />}
              <span>{t(`scene.egomask.${imageError ? "image_error" : busy ? operation : "loading_image"}`)}</span>
            </div>}
            {size && mode === "points" &&
              points.map(([x, y, label], i) => (
                <span
                  key={i}
                  style={{
                    position: "absolute",
                    left: `${(100 * x) / size.w}%`,
                    top: `${(100 * y) / size.h}%`,
                    width: 12,
                    height: 12,
                    marginLeft: -6,
                    marginTop: -6,
                    borderRadius: "50%",
                    border: "2px solid white",
                    background: label === 1 ? "#22c55e" : "#ef4444",
                    pointerEvents: "none",
                  }}
                />
              ))}
          </div>
            </div>
          </div>
          <div className="row wrap egomask-actions" style={{ gap: 8, alignItems: "center" }}>
            <div className="egomask-modes" role="group" aria-label={t("scene.egomask.tools")}>
              <button type="button" className="btn sm" disabled={busy} aria-pressed={mode === "points"} onClick={() => setMode("points")}><MousePointer2 size={15} /> {t("scene.egomask.points_tool")}</button>
              <button type="button" className="btn sm" disabled={busy || !(draft || info?.has_mask)} aria-pressed={mode === "erase"} onClick={() => setMode("erase")}><Eraser size={15} /> {t("scene.egomask.erase_tool")}</button>
            </div>
            <span className="small egomask-summary">
              {draft
                ? t("scene.egomask.draft", { coverage: pct(draft.coverage) })
                : info?.has_mask
                  ? t("scene.egomask.saved", { coverage: pct(info.coverage) })
                  : info?.state === "mask"
                    ? t(`scene.egomask.from_${info.source}`)
                    : info?.state === "none"
                      ? t("scene.egomask.is_none")
                      : t("scene.egomask.none")}
            </span>
            <button className="btn sm" disabled={busy || mode === "erase" || points.length === 0} onClick={() => void run(points.slice(0, -1))}>
              <Undo2 size={15} />
              {t("scene.egomask.undo")}
            </button>
            <button className="btn sm" disabled={busy || mode === "erase" || points.length === 0} onClick={() => void run([])}>
              <Eraser size={15} />
              {t("scene.egomask.clear")}
            </button>
            <button className="btn sm" disabled={busy || !draft} onClick={() => void act(() => post(`${base}/save`, {}))}>
              <Save size={15} />
              {t("scene.egomask.save")}
            </button>
            <button
              className="btn sm"
              disabled={busy || info?.state === "none" || (info?.state === "mask" && !info.has_mask)}
              title={t("scene.egomask.mark_none_hint")}
              onClick={() => void act(() => post(`${base}/none`, {}))}
            >
              <Ban size={15} />
              {t("scene.egomask.mark_none")}
            </button>
            <button className="btn sm" disabled={busy || (!info?.has_mask && info?.source !== "confirmed")} onClick={() => void act(() => del(base))}>
              <Trash2 size={15} />
              {t("scene.egomask.delete")}
            </button>
          </div>
          {mode === "erase" && <div className="egomask-guide">
            <label>{t("scene.egomask.brush_size")} <input type="range" min="3" max="128" value={radius} disabled={busy} onChange={(e) => setRadius(Number(e.target.value))} /> {radius * 2} px</label>
            <span>{t("scene.egomask.erase_hint")}</span>
          </div>}
        </>
      )}
    </div>
  );
}
