// Thêm cảnh từ dữ liệu gốc (như `d2s source inspect` rồi `d2s scene ingest`): nhập đường dẫn thư
// mục trên máy chạy Studio, xem tool nhận ra gì, chọn phiên bản và chuỗi, rồi ingest ở nền. Kết quả
// hiện trong ngăn "Hoạt động" và thông báo nổi khi xong.
import { FolderSearch } from "lucide-react";
import { useMemo, useState } from "react";
import { asApiError, post, type ApiError } from "../api/client";
import type { Job, SourceReport } from "../api/types";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { Drawer, ErrorBox, Note, useToast, yesNo } from "./ui";

const PATH_KEY = "d2s.ingest.path";

export function IngestDrawer({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const toast = useToast();
  const live = useLive();
  const [path, setPath] = useState(() => {
    try {
      return localStorage.getItem(PATH_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [report, setReport] = useState<SourceReport | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [version, setVersion] = useState("");
  const [sequence, setSequence] = useState("");
  const [filter, setFilter] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [interpolate, setInterpolate] = useState(4);

  const inspect = async () => {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      localStorage.setItem(PATH_KEY, path);
    } catch {
      // không lưu được đường dẫn: bỏ qua
    }
    try {
      const result = await post<SourceReport>("/api/sources/inspect", { path });
      setReport(result);
      const versions = Object.keys(result.versions);
      setVersion(versions[0] ?? "");
      setSequence(versions.length ? result.versions[versions[0]][0] ?? "" : "");
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };

  const sequences = useMemo(() => {
    const all = report?.versions[version] ?? [];
    return filter ? all.filter((s) => s.toLowerCase().includes(filter.toLowerCase())) : all;
  }, [report, version, filter]);

  const ingest = async () => {
    setBusy(true);
    setError(null);
    try {
      const job = await post<Job>("/api/sources/ingest", {
        path,
        source_type: report?.dataset,
        version: version || null,
        sequence,
        scene_id: sceneId || null,
        interpolate_n: interpolate,
      });
      toast({ kind: "ok", title: t("ingest.toast.queued"), detail: job.label });
      live.refreshJobs();
      onClose();
    } catch (err) {
      setError(asApiError(err));
      setBusy(false);
    }
  };

  const defaultId = report?.dataset && sequence ? `${report.dataset}_${sequence.toLowerCase().replace(/[^a-z0-9_-]/g, "-")}` : "";
  const versions = Object.keys(report?.versions ?? {});

  return (
    <Drawer
      title={t("ingest.title")}
      onClose={onClose}
      foot={
        <>
          <button className="btn" onClick={onClose}>
            {t("common.action.cancel")}
          </button>
          <button className="btn primary" onClick={ingest} disabled={!report?.ingest_supported || !sequence || busy}>
            {t("ingest.action.start")}
          </button>
        </>
      }
    >
      <p className="muted">{t("ingest.lede")}</p>
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          inspect();
        }}
      >
        <label className="field grow">
          <span>{t("ingest.field.path")}</span>
          <input type="text" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/data/nuscenes" autoFocus spellCheck={false} />
        </label>
        <button className="btn" type="submit" disabled={!path || busy} style={{ alignSelf: "flex-end" }}>
          <FolderSearch size={15} />
          {t("ingest.action.inspect")}
        </button>
      </form>
      <ErrorBox error={error} />

      {report && (
        <>
          <dl className="facts">
            <div>
              <dt>{t("ingest.fact.dataset")}</dt>
              <dd>{report.dataset ?? t("ingest.fact.unknown")}</dd>
            </div>
            <div>
              <dt>{t("ingest.fact.cameras")}</dt>
              <dd>{report.cameras.length || "–"}</dd>
            </div>
            <div>
              <dt>LiDAR</dt>
              <dd>{yesNo(report.has_lidar, t)}</dd>
            </div>
            <div>
              <dt>{t("ingest.fact.cuboids")}</dt>
              <dd>{yesNo(report.has_cuboids, t)}</dd>
            </div>
          </dl>
          {report.notes.map((note) => (
            <Note key={note} kind={report.ingest_supported ? "info" : "warn"}>
              {note}
            </Note>
          ))}
          {!report.ingest_supported && report.inventory.length > 0 && (
            <div className="table-wrap panel">
              <table>
                <tbody>
                  {report.inventory.map((item, index) => (
                    <tr key={index}>
                      <td>{String(item.path ?? item.suffix ?? "")}</td>
                      <td className="num">{String(item.count ?? "")}</td>
                      <td className="muted">{Array.isArray(item.resolution) ? item.resolution.join("×") : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {report.ingest_supported && (
            <div className="stack" style={{ gap: 12 }}>
              {versions.length > 1 && (
                <label className="field">
                  <span>{t("ingest.field.version")}</span>
                  <select
                    value={version}
                    onChange={(e) => {
                      setVersion(e.target.value);
                      setSequence(report.versions[e.target.value][0] ?? "");
                    }}
                  >
                    {versions.map((v) => (
                      <option key={v} value={v}>
                        {v || t("ingest.field.no_version")} ({report.versions[v].length})
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="field">
                <span>{t("ingest.field.sequence", { count: report.versions[version]?.length ?? 0 })}</span>
                {(report.versions[version]?.length ?? 0) > 12 && (
                  <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={t("ingest.field.filter")} />
                )}
                <select size={Math.min(8, Math.max(3, sequences.length))} value={sequence} onChange={(e) => setSequence(e.target.value)} style={{ height: "auto", padding: 4 }}>
                  {sequences.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <div className="form-grid">
                <label className="field">
                  <span>{t("ingest.field.scene_id")}</span>
                  <input type="text" value={sceneId} onChange={(e) => setSceneId(e.target.value)} placeholder={defaultId} spellCheck={false} />
                </label>
                {report.dataset === "nuscenes" && (
                  <label className="field">
                    <span>{t("ingest.field.interpolate")}</span>
                    <input type="number" min={0} value={interpolate} onChange={(e) => setInterpolate(Number(e.target.value))} />
                    <span className="hint">{t("ingest.field.interpolate_hint")}</span>
                  </label>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </Drawer>
  );
}
