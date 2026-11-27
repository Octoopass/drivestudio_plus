// Ảnh LiDAR chồng lên ảnh camera của một cảnh, để người xem kiểm calib bằng mắt (không phải bước
// duyệt: ingest, train không chờ). Ảnh và số liệu do drive2sim/recon/tools/lidar_overlay.py ghi
// (reports/lidar_overlay/, reports/lidar_check.json); báo cáo kiểm tra có các mục projection.*.
import { RefreshCw } from "lucide-react";
import { useState } from "react";
import { asApiError, post, seg, useApi, type ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { Empty, ErrorBox, InfoTip, Loading } from "./ui";

interface LidarCamera {
  camera_id: string;
  frames: number;
  points_per_frame?: number;
  points_per_megapixel?: number;
  median_depth_m?: number | null;
  p95_depth_m?: number | null;
}

interface LidarPayload {
  check: { frames: number[]; cameras: LidarCamera[] } | null;
  overlays: { frame: number; camera_id: string; url: string }[];
}

export function LidarOverlay({ sceneId, kaggle, onChange }: { sceneId: string; kaggle: boolean; onChange: () => void }) {
  const { t } = useI18n();
  const data = useApi<LidarPayload>(`/api/scenes/${seg(sceneId)}/lidar`);
  const [frame, setFrame] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      await post(`/api/scenes/${seg(sceneId)}/lidar`, {});
      data.reload();
      onChange();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };
  if (data.error) return <ErrorBox error={data.error} />;
  if (!data.data) return <Loading rows={2} />;
  const { check, overlays } = data.data;
  const frames = check?.frames ?? [];
  const current = frame ?? frames[0] ?? null;
  const shown = overlays.filter((o) => o.frame === current);
  const button = !kaggle && (
    <button className="btn sm" onClick={generate} disabled={busy}>
      <RefreshCw size={15} className={busy ? "spin" : undefined} />
      {t(busy ? "scene.lidar.running" : check ? "scene.lidar.rerun" : "scene.lidar.run")}
    </button>
  );
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row wrap" style={{ gap: 8, justifyContent: "space-between" }}>
        <h3 className="row" style={{ gap: 6, margin: 0, fontSize: 17 }}>
          {t("scene.lidar.title")}
          <InfoTip>{t("scene.lidar.hint")}</InfoTip>
        </h3>
        {button}
      </div>
      <ErrorBox error={error} />
      {!check ? (
        <Empty title={t("scene.lidar.none")}>{t(kaggle ? "scene.lidar.none_kaggle" : "scene.lidar.none_hint")}</Empty>
      ) : (
        <>
          <div className="table-wrap">
            <table className="dense">
              <thead>
                <tr>
                  <th>{t("common.camera")}</th>
                  <th className="num">{t("scene.lidar.points")}</th>
                  <th className="num">{t("scene.lidar.density")}</th>
                  <th className="num">{t("scene.lidar.median_depth")}</th>
                  <th className="num">{t("scene.lidar.p95_depth")}</th>
                </tr>
              </thead>
              <tbody>
                {check.cameras.map((c) => (
                  <tr key={c.camera_id}>
                    <td className="id">{c.camera_id}</td>
                    <td className="num">{c.points_per_frame != null ? Math.round(c.points_per_frame).toLocaleString() : "–"}</td>
                    <td className="num">{c.points_per_megapixel != null ? Math.round(c.points_per_megapixel).toLocaleString() : "–"}</td>
                    <td className="num">{c.median_depth_m != null ? `${c.median_depth_m.toFixed(1)} m` : "–"}</td>
                    <td className="num">{c.p95_depth_m != null ? `${c.p95_depth_m.toFixed(1)} m` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {overlays.length === 0 ? (
            <p className="small muted">{t("scene.lidar.no_images")}</p>
          ) : (
            <>
              <div className="row wrap" style={{ gap: 6 }}>
                <span className="small muted">{t("scene.lidar.frame")}</span>
                {frames.map((f) => (
                  <button key={f} type="button" className="btn sm" aria-pressed={f === current} onClick={() => setFrame(f)}>
                    {f}
                  </button>
                ))}
              </div>
              <div className="frames">
                {shown.map((o) => (
                  <figure key={o.url}>
                    <a href={o.url} target="_blank" rel="noreferrer">
                      <img src={o.url} alt={`${o.camera_id}, ${o.frame}`} loading="lazy" />
                    </a>
                    <figcaption>{o.camera_id}</figcaption>
                  </figure>
                ))}
              </div>
              <p className="small muted">{t("scene.lidar.how_to_read")}</p>
            </>
          )}
        </>
      )}
    </div>
  );
}
