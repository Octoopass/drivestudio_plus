// Sơ đồ camera trên xe và quỹ đạo xe nhìn từ trên xuống (SPEC mục 8A), vẽ từ reports/rig.json.
//
// rig.json do drive2sim/recon/tools/scene_rig.py ghi từ chính pose mà loader DriveStudio đưa cho
// trainer (không tự đọc file pose của từng dataset), trong hệ x tới, y trái, z lên (mét). Trên sơ
// đồ, đầu xe hướng lên trên: x của file là trục dọc đi lên, y là trục ngang sang trái.
import { RefreshCw } from "lucide-react";
import { useState } from "react";
import { asApiError, post, seg, useApi, type ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { lensOf } from "../lens";
import { Empty, ErrorBox, InfoTip, Loading } from "./ui";

export interface RigCamera {
  camera_id: string;
  model: string | null;
  position_m: [number, number, number];
  yaw_deg: number;
  pitch_deg: number;
  hfov_deg: number | null;
}

export interface TrajectoryPoint {
  frame: number;
  x: number;
  y: number;
  z: number;
  yaw_deg: number;
}

export interface Rig {
  version: number;
  scene_id: string;
  reference_camera: string;
  forward_source: "motion" | "camera";
  cameras: RigCamera[];
  trajectory: TrajectoryPoint[];
}

// Toạ độ vẽ: đầu xe hướng lên. (x tới, y trái) → (ngang = -y, dọc = -x).
const sx = (y: number) => -y;
const sy = (x: number) => -x;
const rad = (deg: number) => (deg * Math.PI) / 180;

/** Điểm trên sơ đồ theo hướng yaw (độ, 0 là phía trước, dương là sang trái). */
function toward(cx: number, cy: number, yawDeg: number, r: number): [number, number] {
  return [cx - r * Math.sin(rad(yawDeg)), cy - r * Math.cos(rad(yawDeg))];
}

/** Hình quạt góc nhìn của một camera. */
function wedge(cx: number, cy: number, yaw: number, hfov: number, r: number): string {
  const span = Math.min(hfov, 359.9);
  const [x1, y1] = toward(cx, cy, yaw + span / 2, r);
  const [x2, y2] = toward(cx, cy, yaw - span / 2, r);
  const large = span > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
}

function niceLength(target: number): number {
  const power = 10 ** Math.floor(Math.log10(Math.max(target, 1e-6)));
  return [1, 2, 5, 10].map((k) => k * power).find((v) => v >= target) ?? power * 10;
}

export function RigDiagram({ rig }: { rig: Rig }) {
  const { t } = useI18n();
  const xs = rig.cameras.map((c) => sx(c.position_m[1]));
  const ys = rig.cameras.map((c) => sy(c.position_m[0]));
  // Thân xe: bao các camera, nới thêm; tối thiểu cỡ một xe con (1,8 x 4,5 m).
  const halfW = Math.max(0.9, Math.max(...xs.map(Math.abs)) + 0.25);
  const halfL = Math.max(2.25, Math.max(...ys.map(Math.abs)) + 0.6);
  const reach = halfL * 0.9;
  const pad = reach + 0.6;
  const box = { x: -halfW - pad, y: -halfL - pad, w: 2 * (halfW + pad), h: 2 * (halfL + pad) };
  // Nhãn đặt ở đầu hình quạt; hai camera nhìn cùng hướng (cặp stereo) thì nhãn sau dời xuống
  // một dòng cho khỏi đè nhau.
  const labels: [number, number][] = [];
  for (const c of rig.cameras) {
    let [lx, ly] = toward(sx(c.position_m[1]), sy(c.position_m[0]), c.yaw_deg, reach + 0.35);
    while (labels.some(([x, y]) => Math.abs(x - lx) < 1.1 && Math.abs(y - ly) < 0.3)) ly += 0.36;
    labels.push([lx, ly]);
  }
  return (
    <figure className="rig-figure">
      <svg viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} role="img" aria-label={t("scene.rig.diagram")}>
        <rect x={-halfW} y={-halfL} width={2 * halfW} height={2 * halfL} rx={0.5} className="rig-car" />
        <path d={`M 0 ${-halfL + 0.35} l -0.35 0.55 l 0.7 0 Z`} className="rig-arrow" />
        {rig.cameras.map((c, index) => {
          const cx = sx(c.position_m[1]);
          const cy = sy(c.position_m[0]);
          const fisheye = c.model ? lensOf(c.model) === "fisheye" : (c.hfov_deg ?? 0) > 150;
          const [lx, ly] = labels[index];
          return (
            <g key={c.camera_id}>
              {c.hfov_deg !== null && (
                <path d={wedge(cx, cy, c.yaw_deg, c.hfov_deg, reach)} className={fisheye ? "rig-fov fisheye" : "rig-fov"} />
              )}
              <circle cx={cx} cy={cy} r={0.09} className="rig-cam" />
              <text x={lx} y={ly} className="rig-label" textAnchor="middle" dominantBaseline="middle">
                {c.camera_id}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="small muted">{t("scene.rig.diagram_caption")}</figcaption>
    </figure>
  );
}

export function TrajectoryMap({ rig, frame, onFrame }: { rig: Rig; frame: number; onFrame: (frame: number) => void }) {
  const { t } = useI18n();
  const points = rig.trajectory;
  if (!points.length) return null;
  const px = points.map((p) => sx(p.y));
  const py = points.map((p) => sy(p.x));
  const minX = Math.min(...px);
  const maxX = Math.max(...px);
  const minY = Math.min(...py);
  const maxY = Math.max(...py);
  const size = Math.max(maxX - minX, maxY - minY, 10);
  const pad = size * 0.08;
  const box = { x: minX - pad, y: minY - pad, w: maxX - minX + 2 * pad, h: maxY - minY + 2 * pad };
  const unit = size / 100; // cỡ nét, chữ theo cỡ bản đồ
  const current = points[Math.min(frame, points.length - 1)];
  const [hx, hy] = toward(sx(current.y), sy(current.x), current.yaw_deg, unit * 6);
  const scale = niceLength(size / 5);
  const length = points.slice(1).reduce((sum, p, i) => sum + Math.hypot(p.x - points[i].x, p.y - points[i].y), 0);
  return (
    <figure className="rig-figure">
      <svg viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} role="img" aria-label={t("scene.rig.trajectory")}>
        <polyline points={points.map((_, i) => `${px[i]},${py[i]}`).join(" ")} className="rig-path" style={{ strokeWidth: unit * 0.8 }} />
        <circle cx={px[0]} cy={py[0]} r={unit * 1.5} className="rig-start" />
        <rect x={px[px.length - 1] - unit * 1.4} y={py[py.length - 1] - unit * 1.4} width={unit * 2.8} height={unit * 2.8} className="rig-end" />
        <line x1={sx(current.y)} y1={sy(current.x)} x2={hx} y2={hy} className="rig-heading" style={{ strokeWidth: unit * 0.9 }} />
        <circle cx={sx(current.y)} cy={sy(current.x)} r={unit * 2} className="rig-now" />
        <g transform={`translate(${box.x + pad * 0.5} ${box.y + box.h - pad * 0.45})`}>
          <line x1={0} y1={0} x2={scale} y2={0} className="rig-scale" style={{ strokeWidth: unit * 0.6 }} />
          <text x={scale / 2} y={-unit * 1.5} textAnchor="middle" className="rig-label" style={{ fontSize: unit * 2.4 }}>
            {`${scale} m`}
          </text>
        </g>
      </svg>
      <div className="row wrap" style={{ gap: 10, alignItems: "center" }}>
        <input
          type="range"
          className="grow"
          min={0}
          max={points.length - 1}
          value={Math.min(frame, points.length - 1)}
          onChange={(e) => onFrame(Number(e.target.value))}
          aria-label={t("scene.rig.frame")}
        />
        <span className="num small nowrap">{t("scene.frames.frame", { frame: current.frame, last: points.length - 1 })}</span>
      </div>
      <figcaption className="small muted">
        {t("scene.rig.trajectory_caption", { length: Math.round(length), climb: Math.round(Math.max(...points.map((p) => p.z)) - Math.min(...points.map((p) => p.z))) })}
      </figcaption>
    </figure>
  );
}

/** Khối sơ đồ camera và quỹ đạo của trang cảnh; tạo rig.json khi chưa có. */
export function SceneRig({ sceneId }: { sceneId: string }) {
  const { t } = useI18n();
  const data = useApi<{ rig: Rig | null }>(`/api/scenes/${seg(sceneId)}/rig`);
  const [frame, setFrame] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      await post(`/api/scenes/${seg(sceneId)}/rig`, {});
      data.reload();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };
  if (data.error) return <ErrorBox error={data.error} />;
  if (!data.data) return <Loading rows={2} />;
  const rig = data.data.rig;
  const button = (
    <button className="btn sm" onClick={generate} disabled={busy}>
      <RefreshCw size={15} className={busy ? "spin" : undefined} />
      {t(busy ? "scene.rig.generating" : rig ? "scene.rig.regenerate" : "scene.rig.generate")}
    </button>
  );
  return (
    <div className="stack" style={{ gap: 10 }}>
      <ErrorBox error={error} />
      {!rig ? (
        <Empty title={t("scene.rig.none")} action={button}>
          {t("scene.rig.none_hint")}
        </Empty>
      ) : (
        <>
          <div className="row wrap" style={{ gap: 8, justifyContent: "space-between" }}>
            <span className="small muted row" style={{ gap: 4 }}>
              {t(rig.forward_source === "motion" ? "scene.rig.forward_motion" : "scene.rig.forward_camera", { camera: rig.reference_camera })}
              <InfoTip>{t("scene.rig.source_hint")}</InfoTip>
            </span>
            {button}
          </div>
          <div className="rig-grid">
            <RigDiagram rig={rig} />
            <TrajectoryMap rig={rig} frame={frame} onFrame={setFrame} />
          </div>
        </>
      )}
    </div>
  );
}
