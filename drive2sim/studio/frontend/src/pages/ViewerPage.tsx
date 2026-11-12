// Xem 3D trong trình duyệt bằng Spark: mở gói xem trước của một run (nền tĩnh và từng xe, phát lại
// xe theo khung), mở scene.ply của run, hoặc kéo thả một file splat bất kỳ trên máy (.ply, .spz,
// .splat, .ksplat, .sog); file kéo thả chỉ đọc trong trình duyệt, không gửi lên server.
import { Box, Crosshair, FileUp, Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, seg, useApi } from "../api/client";
import type { ActorPoses, List, RunRecord, Splats } from "../api/types";
import { ErrorBox, Note, PageHead, Segmented } from "../components/ui";
import { useI18n } from "../i18n";
import { go } from "../router";
import { SplatViewer, type SplatSource, type UpAxis } from "../viewer/engine";

const ACCEPT = ".ply,.spz,.splat,.ksplat,.sog,.zip";
type Source = "preview" | "ply" | "file";

export function ViewerPage({ runId }: { runId?: string }) {
  const { t } = useI18n();
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<SplatViewer | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const runs = useApi<List<RunRecord>>("/api/runs?status=succeeded");
  const splats = useApi<Splats>(runId ? `/api/runs/${seg(runId)}/splats` : null);
  const [source, setSource] = useState<Source | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [up, setUp] = useState<UpAxis>("z");
  const [sh, setSh] = useState(3);
  const [progress, setProgress] = useState<number | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poses, setPoses] = useState<ActorPoses | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [webgl, setWebgl] = useState(true);

  useEffect(() => {
    if (!host.current) return;
    try {
      viewer.current = new SplatViewer(host.current);
    } catch {
      setWebgl(false);
    }
    return () => viewer.current?.dispose();
  }, []);

  // Chọn sẵn nguồn tốt nhất khi run có file xem được.
  useEffect(() => {
    if (!splats.data) return;
    setSource(splats.data.preview ? "preview" : splats.data.ply ? "ply" : null);
  }, [splats.data]);

  // Nạp nguồn đã chọn.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !source) return;
    let alive = true;
    const run = async () => {
      setError(null);
      setCount(null);
      setPoses(null);
      setProgress(0);
      let sources: SplatSource[] = [];
      let actorIds: string[] = [];
      let loadedPoses: ActorPoses | null = null;
      if (source === "file" && file) {
        sources = [{ name: file.name, bytes: await file.arrayBuffer() }];
      } else if (source === "ply" && splats.data?.ply) {
        sources = [{ name: "scene.ply", url: splats.data.ply }];
      } else if (source === "preview" && splats.data?.preview) {
        const preview = splats.data.preview;
        if (preview.background_url) sources.push({ name: "background.ply", url: preview.background_url });
        for (const [trackId, url] of Object.entries(preview.actor_urls)) if (url) sources.push({ name: trackId, url });
        actorIds = Object.keys(preview.actor_urls);
        if (preview.poses_url) loadedPoses = await api<ActorPoses>(preview.poses_url);
      }
      if (!sources.length) return;
      const total = await v.load(sources, actorIds, (fraction) => alive && setProgress(fraction));
      if (!alive || total === null) return;
      v.setMaxSh(sh);
      setCount(total);
      setProgress(null);
      if (loadedPoses) {
        setPoses(loadedPoses);
        setFrame(0);
        v.setPoses(loadedPoses.track_ids, loadedPoses.T_object_world[0] ?? []);
      }
    };
    run().catch((err) => alive && (setError(String(err)), setProgress(null)));
    return () => {
      alive = false;
    };
  }, [source, file, splats.data]);

  useEffect(() => viewer.current?.setUpAxis(up), [up]);
  useEffect(() => viewer.current?.setMaxSh(sh), [sh]);
  useEffect(() => {
    if (poses) viewer.current?.setPoses(poses.track_ids, poses.T_object_world[frame] ?? []);
  }, [frame, poses]);
  useEffect(() => {
    if (!playing || !poses) return;
    const last = poses.T_object_world.length - 1;
    const timer = window.setInterval(() => setFrame((f) => (f >= last ? 0 : f + 1)), 100);
    return () => window.clearInterval(timer);
  }, [playing, poses]);

  const openFile = (picked: File | undefined) => {
    if (!picked) return;
    setFile(picked);
    setSource("file");
    setUp("-y");
  };

  const preview = splats.data?.preview;
  const nothing = !source || (source === "file" && !file);
  return (
    <div className="page" style={{ maxWidth: "none" }}>
      <PageHead title={t("viewer.page.title")} lede={t("viewer.page.lede")} />
      <div className="viewer">
        <aside className="viewer-side">
          <label className="field">
            <span>{t("viewer.field.run")}</span>
            <select value={runId ?? ""} onChange={(e) => go({ page: "viewer", runId: e.target.value || undefined })}>
              <option value="">{t("viewer.field.no_run")}</option>
              {(runs.data?.items ?? []).map((run) => (
                <option key={run.run_id}>{run.run_id}</option>
              ))}
            </select>
          </label>
          <ErrorBox error={splats.error} />
          {runId && splats.data && !splats.data.preview && !splats.data.ply && (
            <Note kind="warn">{t("viewer.note.no_export")}</Note>
          )}
          {runId && splats.data && (splats.data.preview || splats.data.ply) && (
            <div className="field">
              <span>{t("viewer.field.source")}</span>
              <Segmented<Source>
                label={t("viewer.field.source")}
                value={source ?? undefined}
                onChange={setSource}
                options={[
                  { value: "preview", label: t("viewer.source.preview"), disabled: !splats.data.preview },
                  { value: "ply", label: "scene.ply", disabled: !splats.data.ply },
                ]}
              />
            </div>
          )}
          <div
            className="drop"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              openFile(e.dataTransfer.files[0]);
            }}
          >
            <FileUp size={18} />
            <span className="small">{t("viewer.drop.text")}</span>
            <button className="btn sm" onClick={() => fileInput.current?.click()}>
              {t("viewer.drop.button")}
            </button>
            <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(e) => openFile(e.target.files?.[0])} />
            {file && source === "file" && <span className="small id ellipsis" style={{ maxWidth: "100%" }}>{file.name}</span>}
          </div>
          <div className="field">
            <span>{t("viewer.field.up")}</span>
            <Segmented<UpAxis>
              label={t("viewer.field.up")}
              value={up}
              onChange={setUp}
              options={[
                { value: "z", label: "Z+" },
                { value: "-y", label: "Y−" },
                { value: "y", label: "Y+" },
              ]}
            />
            <span className="hint">{t("viewer.field.up_hint")}</span>
          </div>
          <div className="field">
            <span>{t("viewer.field.sh")}</span>
            <Segmented<number> label={t("viewer.field.sh")} value={sh} onChange={setSh} options={[0, 1, 2, 3].map((n) => ({ value: n, label: String(n) }))} />
            <span className="hint">{t("viewer.field.sh_hint")}</span>
          </div>
          {preview && source === "preview" && (preview.box_only_actors.length > 0 || preview.omitted.length > 0) && (
            <p className="small muted">{t("viewer.note.omitted", { items: [...preview.omitted, ...preview.box_only_actors.slice(0, 3)].join(", ") })}</p>
          )}
          <p className="small faint">{t("viewer.help")}</p>
        </aside>
        <div
          className={`viewer-stage${dragging ? " dragging" : ""}`}
          ref={host}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            openFile(e.dataTransfer.files[0]);
          }}
        >
          {!webgl && (
            <div className="viewer-empty">
              <div className="box">{t("viewer.no_webgl")}</div>
            </div>
          )}
          {webgl && nothing && (
            <div className="viewer-empty">
              <div className="box">
                <Box size={30} />
                <strong style={{ color: "#e6eaee" }}>{t("viewer.empty.title")}</strong>
                <span className="small">{t("viewer.empty.text")}</span>
              </div>
            </div>
          )}
          {!nothing && (
            <div className="hud">
              {progress !== null && <span className="tag">{t("viewer.hud.loading", { percent: Math.round(progress * 100) })}</span>}
              {count !== null && <span className="tag">{t("viewer.hud.count", { count: count.toLocaleString() })}</span>}
              {poses && <span className="tag">{t("viewer.hud.actors", { count: poses.track_ids.length })}</span>}
              <button className="btn sm" style={{ background: "rgba(10,13,16,.7)", color: "#e6eaee", borderColor: "rgba(255,255,255,.14)" }} onClick={() => viewer.current?.frameAll()}>
                <Crosshair size={13} />
                {t("viewer.hud.recenter")}
              </button>
            </div>
          )}
          {error && (
            <div style={{ position: "absolute", left: 12, right: 12, top: 52 }}>
              <div className="alert" role="alert">{error}</div>
            </div>
          )}
          {poses && poses.T_object_world.length > 1 && (
            <div className="hud-bottom">
              <button className="btn icon sm" onClick={() => setPlaying((p) => !p)} aria-label={t(playing ? "scene.frames.pause" : "scene.frames.play")}>
                {playing ? <Pause size={14} /> : <Play size={14} />}
              </button>
              <input type="range" min={0} max={poses.T_object_world.length - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label={t("viewer.hud.frame")} />
              <span className="dense nowrap">{t("scene.frames.frame", { frame, last: poses.T_object_world.length - 1 })}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
