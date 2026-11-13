// Xem 3D: bốn mode cho một run, cộng kéo thả file.
// - Gói xem trước: Spark vẽ trong trình duyệt nền tĩnh và từng xe, phát lại xe theo khung.
// - Cảnh tĩnh: Spark vẽ scene.ply của run.
// - Video: phát các video MP4 mà stage train đã render theo camera gốc (đầy đủ mọi phần của model).
// - Render GPU: kiểm tra máy (GET /api/runs/{id}/render-check), rồi Studio bật viewer GPU của
//   DriveStudio (viser, POST /api/runs/{id}/gpu-viewer) và trang nhúng nó; viewer vẽ từ checkpoint
//   bằng renderer gốc nên đủ mọi phần của model.
// File kéo thả (.ply, .spz, .splat, .ksplat, .sog) chỉ đọc trong trình duyệt, không gửi lên server.
import { ArrowLeft, Box, CheckCircle2, ChevronLeft, Crosshair, FileUp, Maximize2, Minimize2, Pause, Play, Power, SlidersHorizontal, XCircle, Video as VideoIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, asApiError, del, post, preferredVideo, seg, useApi, type ApiError } from "../api/client";
import type { ActorPoses, List, RunRecord, Splats, Video } from "../api/types";
import { ErrorBox, InfoTip, Note, PageHead, Segmented } from "../components/ui";
import { useI18n } from "../i18n";
import { go } from "../router";
import { SplatViewer, type SceneCamera, type SplatSource, type UpAxis } from "../viewer/engine";

/** Trục lên ghi trong phần đầu file PLY do tool xuất ("comment drive2sim up=+z ..."), nếu có. */
async function headerUpAxis(picked: File): Promise<UpAxis | null> {
  const head = await picked.slice(0, 65536).text();
  const match = /comment drive2sim up=([+-])([xyz])/.exec(head);
  if (!match) return null;
  return (match[1] === "-" ? `-${match[2]}` : match[2]) as UpAxis;
}

/** Camera mở mặc định: camera trước (tên có "front" hoặc "0"), không thì camera đầu tiên. */
function defaultCamera(cameras: SceneCamera[]): number {
  const index = cameras.findIndex((camera) => /front|^0$|_0$/i.test(camera.name));
  return index >= 0 ? index : 0;
}

const ACCEPT = ".ply,.spz,.splat,.ksplat,.sog,.zip";
type Source = "preview" | "ply" | "video" | "gpu" | "file";
const SPLAT_SOURCES: Source[] = ["preview", "ply", "file"];
// Tốc độ phát, khung dữ liệu mỗi giây: nuScenes trong DriveStudio khoảng 10 khung mỗi giây.
const PLAYBACK_FPS = 10;
// Bảng điều khiển thu gọn hay mở, nhớ theo trình duyệt (chỉ là tiện lợi; lỗi bộ nhớ thì bỏ qua).
const PANEL_KEY = "d2s.viewer.panel";

function readPanelOpen(): boolean {
  try {
    return window.localStorage.getItem(PANEL_KEY) !== "closed";
  } catch {
    return true;
  }
}

function formatBytes(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

interface GpuViewerStatus {
  state: "stopped" | "starting" | "running" | "exited";
  run_id: string | null;
  url: string | null;
  exit_code: number | null;
  log: string;
}

interface RenderCheck {
  ready: boolean;
  checks: { id: string; ok: boolean; value: string | null }[];
}

export function ViewerPage({ runId }: { runId?: string }) {
  const { t } = useI18n();
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<SplatViewer | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const runs = useApi<List<RunRecord>>("/api/runs?status=succeeded");
  const splats = useApi<Splats>(runId ? `/api/runs/${seg(runId)}/splats` : null);
  const videos = useApi<List<Video>>(runId ? `/api/runs/${seg(runId)}/videos` : null);
  const [source, setSource] = useState<Source | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [up, setUp] = useState<UpAxis>("z");
  // Camera gốc của run (cameras.json): mở đúng góc của video, nút về góc camera, bám camera khi phát.
  const [cameras, setCameras] = useState<SceneCamera[] | null>(null);
  const [cameraIndex, setCameraIndex] = useState(0);
  const [follow, setFollow] = useState(false);
  const [sh, setSh] = useState(3);
  const [progress, setProgress] = useState<number | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poses, setPoses] = useState<ActorPoses | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [webgl, setWebgl] = useState(true);
  // Màn chọn nguồn: run đang chọn và file đã kéo vào, chưa mở cho tới khi bấm nút.
  const [pickRun, setPickRun] = useState<string>("");
  const [pickFile, setPickFile] = useState<File | null>(null);
  const [panelOpen, setPanelOpen] = useState(readPanelOpen);
  const [fullscreen, setFullscreen] = useState(false);
  const full = useRef<HTMLDivElement>(null);
  // Đang xem (màn hình lớn) khi đã chọn run (qua URL) hoặc đã mở file; còn lại là màn chọn nguồn.
  const immersive = !!runId || (source === "file" && !!file);

  // Viewer chỉ tồn tại trong màn xem: tạo khi vào, trả context WebGL khi quay ra màn chọn.
  useEffect(() => {
    if (!immersive || !host.current) return;
    try {
      viewer.current = new SplatViewer(host.current);
      // Nhấp đúp bay tới một điểm thì thôi bám camera, nếu không camera bị kéo về ngay khung sau.
      viewer.current.onFly = () => setFollow(false);
    } catch {
      setWebgl(false);
    }
    return () => {
      viewer.current?.dispose();
      viewer.current = null;
    };
  }, [immersive]);

  useEffect(() => {
    const onChange = () => setFullscreen(!!full.current && document.fullscreenElement === full.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void full.current?.requestFullscreen?.();
  };
  const togglePanel = () =>
    setPanelOpen((open) => {
      try {
        window.localStorage.setItem(PANEL_KEY, open ? "closed" : "open");
      } catch {
        // bộ nhớ trình duyệt bị chặn: chỉ không nhớ được lựa chọn
      }
      return !open;
    });

  // Chọn sẵn mode tốt nhất mà run có: gói xem trước, rồi cảnh tĩnh, rồi video, cuối cùng Render GPU
  // (luôn chọn được để xem máy còn thiếu gì).
  useEffect(() => {
    if (!splats.data || !videos.data) return;
    setSource(splats.data.preview ? "preview" : splats.data.ply ? "ply" : videos.data.items.length ? "video" : "gpu");
    setVideoUrl(preferredVideo(videos.data.items)?.url ?? "");
  }, [splats.data, videos.data, runId]); // runId: mở lại cùng run thì dữ liệu cache không đổi

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
      let loadedCameras: SceneCamera[] | null = null;
      if (source !== "file" && splats.data?.cameras) {
        loadedCameras = await api<{ cameras: SceneCamera[] }>(splats.data.cameras)
          .then((data) => data.cameras)
          .catch(() => null);
      }
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
      // Trục lên: file của tool ghi sẵn trong phần đầu PLY; run của tool là hệ DriveStudio (Z+);
      // file lạ thì đoán từ hình dạng cảnh (mặt đất phẳng, nhà và cây mọc lên).
      const axis: UpAxis =
        source === "file" && file ? ((await headerUpAxis(file)) ?? v.estimateUpAxis() ?? "-y") : "z";
      if (!alive) return;
      v.setUpAxis(axis);
      setUp(axis);
      setCameras(loadedCameras);
      if (loadedCameras?.length) {
        const index = defaultCamera(loadedCameras);
        setCameraIndex(index);
        const camera = loadedCameras[index];
        v.setCameraPose(camera.c2w[0], camera.K[0][4], camera.height);
      }
      v.setMaxSh(shRef.current);
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
  // Bậc màu đọc qua ref khi tải xong: người dùng có thể đổi trong lúc file lớn đang tải.
  const shRef = useRef(sh);
  shRef.current = sh;
  useEffect(() => viewer.current?.setMaxSh(sh), [sh]);
  // Bỏ chọn run (về "không chọn"): dọn cảnh của run cũ, trừ khi đang xem file kéo thả.
  useEffect(() => {
    if (runId || source === "file") return;
    viewer.current?.clear();
    setSource(null);
    setPoses(null);
    setCount(null);
    setPlaying(false);
  }, [runId]); // chỉ chạy khi đổi run, không chạy khi đổi nguồn xem
  // Đứng yên (kéo thanh khung): đặt đúng pose của khung số nguyên.
  useEffect(() => {
    if (poses && !playing) viewer.current?.setPoses(poses.track_ids, poses.T_object_world[frame] ?? []);
  }, [frame, poses, playing]);

  const openFile = (picked: File | undefined) => {
    if (!picked) return;
    setFile(picked);
    setSource("file");
    setCameras(null);
    setFollow(false);
  };

  // Về đúng góc của camera gốc ở khung đang xem; khi "bám camera", tự đi theo khung lúc phát.
  const camera = cameras?.[cameraIndex];
  const goToCamera = (at: number = frame) => {
    if (!camera) return;
    const index = Math.min(at, camera.c2w.length - 1);
    viewer.current?.setCameraPose(camera.c2w[index], camera.K[index][4], camera.height);
  };
  useEffect(() => {
    if (follow && !playing) goToCamera(frame);
  }, [frame, follow, cameraIndex, playing]); // goToCamera đọc state mới nhất mỗi lần render

  // Đang phát: mỗi khung hình của trình duyệt đặt xe (và camera, khi bám camera) ở vị trí nội suy
  // giữa hai khung dữ liệu liền nhau, nên chuyển động liền thay vì nhảy 10 lần mỗi giây. Thanh
  // khung chỉ cập nhật khi qua khung số nguyên mới. Ảnh giữa hai khung là ảnh nội suy, không phải
  // ảnh camera thật; muốn so với ảnh gốc thì dừng ở một khung.
  const followRef = useRef(follow);
  followRef.current = follow;
  const cameraRef = useRef(camera);
  cameraRef.current = camera;
  const frameRef = useRef(frame);
  frameRef.current = frame;
  useEffect(() => {
    if (!playing || !poses) return;
    const last = poses.T_object_world.length - 1;
    let time = frameRef.current >= last ? 0 : frameRef.current;
    let previous = performance.now();
    let handle = 0;
    const tick = (now: number) => {
      time += ((now - previous) / 1000) * PLAYBACK_FPS;
      previous = now;
      if (time > last) time = 0;
      const index = Math.floor(time);
      const next = Math.min(index + 1, last);
      const alpha = time - index;
      viewer.current?.setPoses(poses.track_ids, poses.T_object_world[index] ?? [], poses.T_object_world[next] ?? [], alpha);
      const cam = cameraRef.current;
      if (followRef.current && cam) {
        const a = Math.min(index, cam.c2w.length - 1);
        const b = Math.min(next, cam.c2w.length - 1);
        viewer.current?.setCameraPose(cam.c2w[a], cam.K[a][4], cam.height, cam.c2w[b], alpha);
      }
      if (index !== frameRef.current) setFrame(index);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, poses]);

  const splatMode = !!source && SPLAT_SOURCES.includes(source);
  const loading = !!runId && !source && !splats.error;
  const videoItems = videos.data?.items ?? [];
  const doneRuns = [...(runs.data?.items ?? [])].sort((a, b) => (b.created_utc ?? "").localeCompare(a.created_utc ?? ""));
  const chosenRun = pickRun || doneRuns[0]?.run_id || "";

  const openPicked = () => {
    if (!pickFile) return;
    openFile(pickFile);
  };
  const leave = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    setPlaying(false);
    setFile(null);
    setSource(null);
    setCameras(null);
    setFollow(false);
    if (runId) go({ page: "viewer" });
  };

  // Màn 1: chọn nguồn. Chọn run hoặc kéo file vào trước, bấm nút mới mở màn xem.
  if (!immersive) {
    return (
      <div className="page">
        <PageHead title={t("viewer.page.title")} info={t("viewer.page.lede")} />
        <div className="viewer-launch">
          <section className="card launch-card">
            <h3>{t("viewer.launch.run_title")}</h3>
            <p className="small muted">{t("viewer.launch.run_text")}</p>
            <ErrorBox error={runs.error} />
            {runs.data && doneRuns.length === 0 && <p className="small muted">{t("viewer.launch.no_runs")}</p>}
            {doneRuns.length > 0 && (
              <div className="launch-runs" role="radiogroup" aria-label={t("viewer.field.run")}>
                {doneRuns.map((run) => (
                  <button
                    key={run.run_id}
                    type="button"
                    role="radio"
                    aria-checked={chosenRun === run.run_id}
                    className={chosenRun === run.run_id ? "active" : undefined}
                    onClick={() => setPickRun(run.run_id)}
                    onDoubleClick={() => go({ page: "viewer", runId: run.run_id })}
                  >
                    <b className="id ellipsis">{run.run_id}</b>
                    <span className="small muted ellipsis">
                      {run.scene_id} · {run.approach.approach_id}
                      {run.approach.preset ? ` / ${run.approach.preset}` : ""}
                    </span>
                  </button>
                ))}
              </div>
            )}
            <div className="row">
              <button className="btn primary" disabled={!chosenRun} onClick={() => go({ page: "viewer", runId: chosenRun })}>
                <Box size={17} />
                {t("viewer.launch.open_run")}
              </button>
            </div>
          </section>
          <section
            className={`card launch-card launch-drop${dragging ? " dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              setPickFile(e.dataTransfer.files[0] ?? null);
            }}
          >
            <h3>{t("viewer.launch.file_title")}</h3>
            <div className="drop">
              <FileUp size={29} />
              <span className="small">{t("viewer.drop.text")}</span>
              <button className="btn sm" onClick={() => fileInput.current?.click()}>
                {t("viewer.drop.button")}
              </button>
              <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(e) => setPickFile(e.target.files?.[0] ?? null)} />
              {pickFile && (
                <span className="small id ellipsis" style={{ maxWidth: "100%" }}>
                  {pickFile.name} · {formatBytes(pickFile.size)}
                </span>
              )}
            </div>
            <div className="row">
              <button className="btn primary" disabled={!pickFile} onClick={openPicked}>
                <Box size={17} />
                {t("viewer.launch.open_file")}
              </button>
            </div>
          </section>
        </div>
      </div>
    );
  }

  // Màn 2: xem. Cảnh chiếm cả vùng nội dung; mọi thanh phụ nằm nổi, trong suốt, bên trong cảnh.
  const title = source === "file" && file ? file.name : runId;
  return (
    <div
      ref={full}
      className={`viewer-full${dragging ? " dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        openFile(e.dataTransfer.files[0]);
      }}
    >
      <div className="viewer-stage" ref={host} />
      {source === "video" && videoUrl && (
        <div className="viewer-overlay">
          <video key={videoUrl} src={videoUrl} controls autoPlay loop muted />
        </div>
      )}
      {source === "gpu" && runId && (
        <div className="viewer-overlay">
          <RenderCheckPanel runId={runId} />
        </div>
      )}
      {!webgl && (
        <div className="viewer-empty">
          <div className="box">{t("viewer.no_webgl")}</div>
        </div>
      )}
      {loading && (
        <div className="viewer-empty">
          <div className="box">{t("viewer.hud.preparing")}</div>
        </div>
      )}

      <div className="vbar vbar-left glass">
        <button className="btn sm ghost" onClick={leave} title={t("viewer.launch.back")}>
          <ArrowLeft size={16} />
          <span className="hide-narrow">{t("viewer.launch.back")}</span>
        </button>
        <b className="id ellipsis vtitle" title={title ?? undefined}>
          {title}
        </b>
        <button className="btn icon sm ghost" onClick={toggleFullscreen} aria-label={t(fullscreen ? "viewer.hud.exit_fullscreen" : "viewer.hud.fullscreen")} title={t(fullscreen ? "viewer.hud.exit_fullscreen" : "viewer.hud.fullscreen")}>
          {fullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
        </button>
      </div>

      {splatMode && (
        <div className="vbar vbar-right">
          {progress !== null && <span className="tag glass">{t("viewer.hud.loading", { percent: Math.round(progress * 100) })}</span>}
          {count !== null && <span className="tag glass">{t("viewer.hud.count", { count: count.toLocaleString() })}</span>}
          {poses && <span className="tag glass">{t("viewer.hud.actors", { count: poses.track_ids.length })}</span>}
          <button className="btn sm glass" onClick={() => viewer.current?.frameAll()}>
            <Crosshair size={16} />
            {t("viewer.hud.recenter")}
          </button>
        </div>
      )}

      {panelOpen ? (
        <aside className="vpanel glass" aria-label={t("viewer.panel.title")}>
          <header>
            <SlidersHorizontal size={16} />
            <b>{t("viewer.panel.title")}</b>
            <button className="btn icon sm ghost" onClick={togglePanel} aria-label={t("viewer.panel.collapse")} title={t("viewer.panel.collapse")}>
              <ChevronLeft size={17} />
            </button>
          </header>
          <div className="vpanel-body">
            <ErrorBox error={splats.error} />
            {runId && splats.data && (
              <div className="field">
                <span>
                  {t("viewer.field.source")}
                  <InfoTip>{t("viewer.mode.hint")}</InfoTip>
                </span>
                <div className="mode-list compact" role="radiogroup" aria-label={t("viewer.field.source")}>
                  {(
                    [
                      ["preview", !!splats.data.preview],
                      ["ply", !!splats.data.ply],
                      ["video", videoItems.length > 0],
                      ["gpu", true],
                    ] as [Source, boolean][]
                  ).map(([value, available]) => (
                    <button
                      key={value}
                      type="button"
                      role="radio"
                      aria-checked={source === value}
                      className={source === value ? "active" : undefined}
                      disabled={!available}
                      title={available ? t(`viewer.mode.${value}.sub`) : t(`viewer.mode.${value}.missing`)}
                      onClick={() => setSource(value)}
                    >
                      <b>{t(`viewer.mode.${value}`)}</b>
                      <span className="small muted">{available ? t(`viewer.mode.${value}.sub`) : t(`viewer.mode.${value}.missing`)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {source === "video" && videoItems.length > 1 && (
              <label className="field">
                <span>{t("viewer.field.video")}</span>
                <select value={videoUrl} onChange={(e) => setVideoUrl(e.target.value)}>
                  {videoItems.map((v) => (
                    <option key={v.url} value={v.url}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {splatMode && (
              <>
                {cameras && cameras.length > 0 && (
                  <div className="field">
                    <span>
                      {t("viewer.field.camera")}
                      <InfoTip>{t("viewer.field.camera_hint")}</InfoTip>
                    </span>
                    <select value={cameraIndex} onChange={(e) => setCameraIndex(Number(e.target.value))} aria-label={t("viewer.field.camera")}>
                        {cameras.map((item, index) => (
                          <option key={item.name} value={index}>
                            {item.name}
                          </option>
                        ))}
                    </select>
                    <div className="row wrap" style={{ gap: 8 }}>
                      <button className="btn sm" onClick={() => goToCamera()}>
                        <VideoIcon size={15} />
                        {t("viewer.camera.go")}
                      </button>
                      <label className="chip-check">
                      <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
                        {t("viewer.camera.follow")}
                      </label>
                    </div>
                  </div>
                )}
                <div className="field">
                  <span>
                    {t("viewer.field.up")}
                    <InfoTip>{t("viewer.field.up_hint")}</InfoTip>
                  </span>
                  <Segmented<UpAxis>
                    label={t("viewer.field.up")}
                    value={up}
                    onChange={setUp}
                    options={[
                      { value: "z", label: "Z+" },
                      { value: "-z", label: "Z−" },
                      { value: "y", label: "Y+" },
                      { value: "-y", label: "Y−" },
                      { value: "x", label: "X+" },
                      { value: "-x", label: "X−" },
                    ]}
                  />
                </div>
                <div className="field">
                  <span>
                    {t("viewer.field.sh")}
                    <InfoTip>{t("viewer.field.sh_hint")}</InfoTip>
                  </span>
                  <Segmented<number> label={t("viewer.field.sh")} value={sh} onChange={setSh} options={[0, 1, 2, 3].map((n) => ({ value: n, label: String(n) }))} />
                </div>
              </>
            )}
          </div>
        </aside>
      ) : (
        <button className="btn sm glass vpanel-open" onClick={togglePanel} aria-label={t("viewer.panel.expand")}>
          <SlidersHorizontal size={16} />
          {t("viewer.panel.title")}
        </button>
      )}

      {error && (
        <div className="viewer-error">
          <div className="alert" role="alert">
            {error}
          </div>
        </div>
      )}
      {splatMode && poses && poses.T_object_world.length > 1 && (
        <div className="hud-bottom glass">
          <button className="btn icon sm" onClick={() => setPlaying((p) => !p)} aria-label={t(playing ? "scene.frames.pause" : "scene.frames.play")}>
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <input type="range" min={0} max={poses.T_object_world.length - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label={t("viewer.hud.frame")} />
          <span className="dense nowrap">{t("scene.frames.frame", { frame, last: poses.T_object_world.length - 1 })}</span>
        </div>
      )}
    </div>
  );
}

/** Mode Render GPU: kiểm tra máy; đủ điều kiện thì bật viewer GPU và nhúng nó vào khung xem. */
function RenderCheckPanel({ runId }: { runId: string }) {
  const { t } = useI18n();
  const check = useApi<RenderCheck>(`/api/runs/${seg(runId)}/render-check`);
  const [viewer, setViewer] = useState<GpuViewerStatus | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const mine = viewer && viewer.run_id === runId ? viewer : null;

  // Hỏi trạng thái viewer: lần đầu, rồi mỗi giây khi đang khởi động.
  useEffect(() => {
    let alive = true;
    const poll = () => api<GpuViewerStatus>("/api/gpu-viewer").then((v) => alive && setViewer(v)).catch(() => undefined);
    void poll();
    if (mine?.state !== "starting") return () => void (alive = false);
    const timer = window.setInterval(poll, 1000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [mine?.state]);

  const start = async () => {
    setError(null);
    try {
      setViewer(await post<GpuViewerStatus>(`/api/runs/${seg(runId)}/gpu-viewer`, {}));
    } catch (err) {
      setError(asApiError(err));
    }
  };
  const stop = async () => {
    try {
      setViewer(await del<GpuViewerStatus>("/api/gpu-viewer"));
    } catch (err) {
      setError(asApiError(err));
    }
  };

  if (mine?.state === "running" && mine.url) {
    return (
      <div className="gpu-frame">
        <iframe src={mine.url} title={t("viewer.mode.gpu")} />
        <button className="btn sm gpu-stop" onClick={() => void stop()}>
          <Power size={16} />
          {t("viewer.gpu.stop")}
        </button>
      </div>
    );
  }
  return (
    <div className="box render-check">
      <strong>{t("viewer.gpu.title")}</strong>
      {check.error && <ErrorBox error={check.error} />}
      {!check.data && !check.error && <span className="small muted">{t("viewer.gpu.checking")}</span>}
      {check.data && (
        <>
          <ul>
            {check.data.checks.map((item) => (
              <li key={item.id}>
                {item.ok ? <CheckCircle2 size={18} className="ok-ink" /> : <XCircle size={18} className="bad-ink" />}
                <span>
                  <b>{t(`viewer.gpu.check.${item.id}`)}</b>
                  <span className="small muted"> {item.ok ? item.value : t(`viewer.gpu.fix.${item.id}`)}</span>
                </span>
              </li>
            ))}
          </ul>
          <ErrorBox error={error} />
          {mine?.state === "starting" && <Note kind="info">{t("viewer.gpu.starting")}</Note>}
          {mine?.state === "exited" && (
            <>
              <Note kind="warn">{t("viewer.gpu.exited", { code: mine.exit_code ?? "?" })}</Note>
              {mine.log && <pre className="log small">{mine.log}</pre>}
            </>
          )}
          {check.data.ready ? (
            <div className="row" style={{ gap: 8 }}>
              <button className="btn primary" onClick={() => void start()} disabled={mine?.state === "starting"}>
                <Power size={17} />
                {t("viewer.gpu.start")}
              </button>
              {mine?.state === "starting" && (
                <button className="btn" onClick={() => void stop()}>
                  {t("viewer.gpu.stop")}
                </button>
              )}
            </div>
          ) : (
            <>
              <Note kind="warn">{t("viewer.gpu.not_ready")}</Note>
              <button className="btn sm" onClick={check.reload}>
                {t("viewer.gpu.recheck")}
              </button>
            </>
          )}
        </>
      )}
    </div>
  );
}
