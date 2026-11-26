// Lái thử: lái tay xe ego trong CARLA (vật lý, bản đồ dựng theo đoạn đường) và nhìn bằng ảnh Gaussian
// của run NuRec. Studio bật phiên ở nền (POST /api/runs/{id}/drive): CARLA nếu chưa chạy, server render
// PLY mới nhất của run, vòng lái. Trang gửi lệnh lái 20 lần/giây theo phím đang giữ
// (POST /api/drive/control) và tải ảnh mới nhất liên tục (GET /api/drive/frame/{gaussian,carla}).
// Hai ảnh cạnh nhau: bên trái là ảnh Gaussian từ camera của xe, bên phải là thế giới 3D của chính CARLA
// (camera đặt sau xe) để thấy xe đang ở đâu so với con đường.
import { Car, Gauge, Keyboard, Play, RotateCcw, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, asApiError, del, post, seg } from "../api/client";
import { Note, PageHead, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { href } from "../router";

interface DriveInfo {
  phase?: string;
  camera?: string;
  cameras?: string[];
  speed_kmh?: number;
  sim_time_s?: number;
  render_fps?: number;
  carla_fps?: number;
  sim_fps?: number;
  gpus?: { index: number; util: number; memory_used_mb: number; memory_total_mb: number; roles: string[] }[];
  ratio?: number;
  control?: { reverse?: boolean };
}

interface DriveStatus {
  state: "idle" | "starting" | "running" | "error";
  run_id: string | null;
  message: string;
  info: DriveInfo | null;
  log_tail: string;
}

const KEYS = {
  up: ["KeyW", "ArrowUp"],
  down: ["KeyS", "ArrowDown"],
  left: ["KeyA", "ArrowLeft"],
  right: ["KeyD", "ArrowRight"],
};
const STEER_MAX = 0.7;
const STEER_STEP = 0.06;
const CONTROL_MS = 50;

// Tải ảnh mới nhất của một loại liên tục khi phiên đang chạy; trả URL blob để gắn vào <img>.
function useLiveFrame(kind: "gaussian" | "carla", active: boolean): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!active) return;
    let stop = false;
    let current: string | null = null;
    (async () => {
      while (!stop) {
        try {
          const res = await fetch(`/api/drive/frame/${kind}`, { cache: "no-store" });
          if (res.status === 200) {
            const next = URL.createObjectURL(await res.blob());
            if (stop) {
              URL.revokeObjectURL(next);
              break;
            }
            setUrl(next);
            if (current) URL.revokeObjectURL(current);
            current = next;
            continue;
          }
        } catch {
          // mạng chập chờn: thử lại sau
        }
        await new Promise((r) => setTimeout(r, 250));
      }
    })();
    return () => {
      stop = true;
      if (current) URL.revokeObjectURL(current);
    };
  }, [kind, active]);
  return active ? url : null;
}

export function DrivePage({ runId }: { runId?: string }) {
  const { t } = useI18n();
  const toast = useToast();
  const [status, setStatus] = useState<DriveStatus | null>(null);
  const [camera, setCamera] = useState("");
  const [ratio, setRatio] = useState(0.5);
  const [fps, setFps] = useState(20);
  const [reverse, setReverse] = useState(false);
  const [focused, setFocused] = useState(true);
  const pressed = useRef(new Set<string>());
  const steer = useRef(0);
  const reverseRef = useRef(false);
  const sending = useRef(false);

  const running = status?.state === "running" && status.run_id === runId;
  const otherRun = status && status.state !== "idle" && status.run_id && status.run_id !== runId;
  const gaussian = useLiveFrame("gaussian", running);
  const carla = useLiveFrame("carla", running);

  // Trạng thái phiên: nhanh khi đang bật, chậm hơn khi đang lái (chỉ cần tốc độ, khung/giây).
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const next = await api<DriveStatus>("/api/drive");
        if (!stop) setStatus(next);
      } catch {
        // Studio đang khởi động lại: lần sau
      }
      if (!stop) setTimeout(tick, 700);
    };
    tick();
    return () => {
      stop = true;
    };
  }, []);

  // Phím: giữ W/S/A/D hoặc mũi tên, Space phanh tay, Q đổi số lùi, R về đầu đường.
  useEffect(() => {
    if (!running) return;
    const typing = (e: KeyboardEvent) => e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement;
    const down = (e: KeyboardEvent) => {
      if (typing(e)) return;
      if (e.code === "KeyQ" && !e.repeat) {
        reverseRef.current = !reverseRef.current;
        setReverse(reverseRef.current);
      } else if (e.code === "KeyR" && !e.repeat) {
        post("/api/drive/reset").catch(() => undefined);
      }
      if ([...Object.values(KEYS).flat(), "Space"].includes(e.code)) e.preventDefault();
      pressed.current.add(e.code);
    };
    const up = (e: KeyboardEvent) => pressed.current.delete(e.code);
    const blur = () => {
      pressed.current.clear();
      setFocused(false);
    };
    const focus = () => setFocused(true);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    window.addEventListener("focus", focus);
    const any = (codes: string[]) => codes.some((c) => pressed.current.has(c));
    const timer = setInterval(() => {
      const left = any(KEYS.left);
      const right = any(KEYS.right);
      if (left && !right) steer.current = Math.max(-STEER_MAX, steer.current - STEER_STEP);
      else if (right && !left) steer.current = Math.min(STEER_MAX, steer.current + STEER_STEP);
      else steer.current = Math.abs(steer.current) < 0.03 ? 0 : steer.current * 0.6;
      const boost = pressed.current.has("ShiftLeft") || pressed.current.has("ShiftRight");
      const body = {
        throttle: any(KEYS.up) ? (boost ? 1 : 0.6) : 0,
        brake: any(KEYS.down) ? 1 : 0,
        steer: Math.round(steer.current * 100) / 100,
        reverse: reverseRef.current,
        hand_brake: pressed.current.has("Space"),
      };
      if (sending.current) return; // lệnh trước chưa về: bỏ lượt này, không dồn hàng
      sending.current = true;
      post("/api/drive/control", body)
        .catch(() => undefined)
        .finally(() => {
          sending.current = false;
        });
    }, CONTROL_MS);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      window.removeEventListener("focus", focus);
      pressed.current.clear();
    };
  }, [running]);

  const start = async () => {
    if (!runId) return;
    try {
      setStatus(await post<DriveStatus>(`/api/runs/${seg(runId)}/drive`, { camera, ratio, fps }));
      reverseRef.current = false;
      setReverse(false);
    } catch (err) {
      toast({ kind: "bad", title: t("drive.start"), detail: asApiError(err).message });
    }
  };
  const stop = async () => {
    try {
      setStatus(await del<DriveStatus>("/api/drive"));
    } catch (err) {
      toast({ kind: "bad", title: t("drive.stop"), detail: asApiError(err).message });
    }
  };

  const info = running ? status?.info : null;
  const cameras = status?.info?.cameras ?? [];
  const busy = status?.state === "starting" && status.run_id === runId;
  return (
    <div className="page">
      <PageHead
        title={t("drive.title")}
        info={t("drive.help")}
        meta={
          runId ? (
            <span className="small muted">
              {t("drive.run")} <a href={href({ page: "run", runId })}>{runId}</a>
            </span>
          ) : null
        }
        actions={
          <>
            {running && (
              <button className="btn" onClick={() => post("/api/drive/reset").catch(() => undefined)}>
                <RotateCcw size={18} />
                {t("drive.reset")}
              </button>
            )}
            {(running || busy) && (
              <button className="btn danger" onClick={stop}>
                <Square size={18} />
                {t("drive.stop")}
              </button>
            )}
            {!running && !busy && (
              <button className="btn primary" onClick={start} disabled={!runId}>
                <Play size={18} />
                {t("drive.start")}
              </button>
            )}
          </>
        }
      />

      {!runId && <Note kind="warn">{t("drive.no_run")}</Note>}
      {otherRun && <Note kind="warn">{t("drive.other_run", { run: status?.run_id ?? "" })}</Note>}

      {!running && !busy && runId && (
        <section className="panel drive-settings">
          <label className="field">
            <span>{t("drive.camera")}</span>
            <select value={camera} onChange={(e) => setCamera(e.target.value)}>
              <option value="">{t("drive.camera_front")}</option>
              {cameras.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>{t("drive.ratio")}</span>
            <select value={ratio} onChange={(e) => setRatio(Number(e.target.value))}>
              {[0.25, 0.5, 0.75, 1].map((r) => (
                <option key={r} value={r}>
                  {Math.round(r * 100)}%
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>{t("drive.fps")}</span>
            <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
              {[10, 15, 20, 30].map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </label>
          <p className="small muted">{t("drive.settings_hint")}</p>
        </section>
      )}

      {busy && (
        <Note>
          <span className="row" style={{ gap: 8 }}>
            <Car size={18} className="spin" />
            {status?.message || t("drive.starting")}
          </span>
        </Note>
      )}
      {status?.state === "error" && status.run_id === runId && (
        <Note kind="warn">
          <strong>{t("drive.failed")}</strong> {status.message}
          {status.log_tail && <pre className="drive-log">{status.log_tail}</pre>}
        </Note>
      )}

      {running && (
        <>
          <div className="drive-hud">
            <span className="drive-speed">
              <Gauge size={20} />
              {(info?.speed_kmh ?? 0).toFixed(0)} km/h
            </span>
            <span className={`tag ${reverse ? "warn" : "info"}`}>{reverse ? t("drive.gear_r") : t("drive.gear_d")}</span>
            <span className="small muted">{t("drive.fps_now", { fps: (info?.render_fps ?? 0).toFixed(0) })}</span>
            <span className="small muted">{t("drive.carla_fps", { fps: (info?.carla_fps ?? 0).toFixed(0) })}</span>
            <span className="small muted" title={t("drive.sim_fps_help")}>
              {t("drive.sim_fps", { fps: (info?.sim_fps ?? 0).toFixed(0) })}
            </span>
            {(info?.gpus ?? []).map((g) => (
              <span key={g.index} className={`tag ${g.util >= 90 ? "warn" : "info"}`} title={t("drive.gpu_help")}>
                {t("drive.gpu", {
                  index: g.index,
                  roles: g.roles.map((r) => t(`drive.role_${r}`)).join(" + "),
                  util: g.util,
                  used: (g.memory_used_mb / 1024).toFixed(1),
                  total: (g.memory_total_mb / 1024).toFixed(0),
                })}
              </span>
            ))}
            <span className="small muted">{t("drive.sim_time", { s: (info?.sim_time_s ?? 0).toFixed(0) })}</span>
            <span className="small muted">{info?.camera}</span>
            {!focused && <span className="tag warn">{t("drive.click_to_focus")}</span>}
          </div>
          <div className="drive-grid">
            <figure className="panel drive-view">
              <figcaption>{t("drive.view_gaussian")}</figcaption>
              {gaussian ? <img src={gaussian} alt={t("drive.view_gaussian")} /> : <div className="drive-empty">{t("drive.waiting")}</div>}
            </figure>
            <figure className="panel drive-view">
              <figcaption>{t("drive.view_carla")}</figcaption>
              {carla ? <img src={carla} alt={t("drive.view_carla")} /> : <div className="drive-empty">{t("drive.waiting")}</div>}
            </figure>
          </div>
          <p className="small muted row" style={{ gap: 8 }}>
            <Keyboard size={16} />
            {t("drive.keys")}
          </p>
        </>
      )}
    </div>
  );
}
