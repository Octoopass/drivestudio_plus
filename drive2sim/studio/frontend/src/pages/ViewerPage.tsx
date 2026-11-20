// Xem 3D: bốn mode cho một run, cộng kéo thả file.
// - Gói xem trước: Spark vẽ trong trình duyệt nền tĩnh và từng xe, phát lại xe theo khung.
// - Cảnh tĩnh: Spark vẽ scene.ply của run.
// - Video: phát các video MP4 mà stage train đã render theo camera gốc (đầy đủ mọi phần của model).
// - Render GPU: kiểm tra máy (GET /api/runs/{id}/render-check), rồi Studio bật viewer GPU của
//   DriveStudio (viser, POST /api/runs/{id}/gpu-viewer) và trang nhúng nó; viewer vẽ từ checkpoint
//   bằng renderer gốc nên đủ mọi phần của model.
// File kéo thả (.ply, .spz, .splat, .ksplat, .sog) chỉ đọc trong trình duyệt, không gửi lên server.
import { ArrowLeft, Box, CheckCircle2, Crosshair, FileUp, Info, Maximize2, Minimize2, Pause, Play, Power, SlidersHorizontal, XCircle, Video as VideoIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, asApiError, del, post, playableUrl, preferredVideo, seg, useApi, type ApiError } from "../api/client";
import type { ActorPoses, List, RunRecord, Splats, Video } from "../api/types";
import { ErrorBox, InfoTip, Note, PageHead, Segmented } from "../components/ui";
import { RoadScene } from "../components/Illustration";
import { RunInfo } from "../components/RunInfo";
import { useI18n } from "../i18n";
import { go } from "../router";
import { useExactFrame } from "../viewer/useExactFrame";
import { DEFAULT_FOV, novelTrajectory, SplatViewer, upAxisFromCameras, type HumanMotion, type SceneCamera, type SplatSource, type UpAxis } from "../viewer/engine";

/** Khoảng của thanh trượt (góc nhìn dọc, độ): như nhau với mọi camera, vì thanh trượt luôn điều khiển
 * một camera pinhole ảo; muốn đúng camera gốc thì dùng ô "Dùng tiêu cự của camera gốc". */
const FOV_MIN = 10;
const FOV_MAX = 170;

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

type CameraPath = "free" | "camera" | "novel";

const ACCEPT = ".ply,.spz,.splat,.ksplat,.sog,.zip";
type Source = "preview" | "ply" | "video" | "gpu" | "file";
const SPLAT_SOURCES: Source[] = ["preview", "ply", "file"];
// Tốc độ phát, khung dữ liệu mỗi giây: nuScenes trong DriveStudio khoảng 10 khung mỗi giây.
const PLAYBACK_FPS = 10;
/** Phiên bản gói xem trước mà code xuất hiện tại ghi (``export_preview.SCHEMA_VERSION``). */
const PREVIEW_SCHEMA = "d2s-preview/0.2";
/** Chờ xuất lại gói tối đa bao lâu, hỏi lại mỗi bao lâu (ms). */
const REEXPORT_TIMEOUT_MS = 10 * 60 * 1000;
const REEXPORT_POLL_MS = 3000;
// Thanh bên: các tab xếp dọc thành icon; tab đang mở (hoặc thu gọn) nhớ theo trình duyệt (chỉ là
// tiện lợi; lỗi bộ nhớ thì bỏ qua). Giá trị cũ "open" là tab điều khiển.
const PANEL_KEY = "d2s.viewer.panel";
type PanelTab = "controls" | "info";
type PanelState = PanelTab | "closed";

function readPanel(): PanelState {
  try {
    const value = window.localStorage.getItem(PANEL_KEY);
    return value === "closed" || value === "info" ? value : "controls";
  } catch {
    return "controls";
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
  // Xuất lại gói xem trước bản cũ (run train bằng code trước khi gói có trời, người, chỉnh màu).
  const [reexport, setReexport] = useState<{ state: "idle" | "running" | "error"; error?: string }>({ state: "idle" });
  // Camera khi phát: tự do, bám camera gốc đang chọn, hoặc đi theo đường novel (như video novel).
  // Mặc định bám camera gốc: ở "tự do" camera đứng yên ở khung 0 trong khi xe ghi hình đi mất vài
  // trăm mét, xe và người ra khỏi tầm nhìn, nên bấm phát trông như cảnh đứng yên. Tự kéo, xoay
  // camera thì chuyển sang tự do (onFly).
  const [path, setPath] = useState<CameraPath>("camera");
  // Ma trận chỉnh màu 3×4 của từng ảnh (khung, camera) trong gói xem trước.
  const [affine, setAffine] = useState<{ data: Float32Array; cameras: number } | null>(null);
  const follow = path !== "free";
  const setFollow = (value: boolean) => setPath(value ? "camera" : "free");
  const [sh, setSh] = useState(3);
  // Góc nhìn dọc người xem chọn (độ); null là theo camera gốc (thử nghiệm, ARCHITECTURE mục 2.13).
  const [fov, setFov] = useState<number | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poses, setPoses] = useState<ActorPoses | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [webgl, setWebgl] = useState(true);
  // FPS thật của viewer trong trình duyệt, đọc mỗi nửa giây.
  const [rate, setRate] = useState<{ fps: number; idle: boolean } | null>(null);
  // Màn chọn nguồn: run đang chọn và file đã kéo vào, chưa mở cho tới khi bấm nút.
  const [pickRun, setPickRun] = useState<string>("");
  const [pickFile, setPickFile] = useState<File | null>(null);
  const [panel, setPanel] = useState<PanelState>(readPanel);
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
  // Bấm icon của tab đang mở thì thu gọn, icon khác thì mở tab đó.
  const pickPanel = (tab: PanelTab) =>
    setPanel((current) => {
      const next: PanelState = current === tab ? "closed" : tab;
      try {
        window.localStorage.setItem(PANEL_KEY, next);
      } catch {
        // bộ nhớ trình duyệt bị chặn: chỉ không nhớ được lựa chọn
      }
      return next;
    });

  // Chọn sẵn mode tốt nhất mà run có: gói xem trước, rồi cảnh tĩnh, rồi video, cuối cùng Render GPU
  // (luôn chọn được để xem máy còn thiếu gì).
  useEffect(() => {
    if (!splats.data || !videos.data) return;
    setSource(splats.data.preview ? "preview" : splats.data.ply ? "ply" : videos.data.items.length ? "video" : "gpu");
    const first = preferredVideo(videos.data.items);
    setVideoUrl(first ? playableUrl(first) : "");
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
      setAffine(null);
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
        const ext = preview.splat_format;
        if (preview.background_url) sources.push({ name: `background.${ext}`, url: preview.background_url });
        for (const [trackId, url] of Object.entries(preview.actor_urls)) if (url) sources.push({ name: trackId, url });
        actorIds = Object.keys(preview.actor_urls);
        if (preview.poses_url) loadedPoses = await api<ActorPoses>(preview.poses_url);
      }
      if (!sources.length) return;
      let total = await v.load(sources, actorIds, (fraction) => alive && setProgress(fraction));
      if (!alive || total === null) return;
      // Gói xem trước: cách làm mờ theo lúc train, trời, người uốn theo xương.
      const preview = source === "preview" ? splats.data?.preview : null;
      v.setAntialiased(preview?.antialiased ?? true);
      if (preview) {
        await v.setSky(preview.sky_url);
        const humans = Object.entries(preview.human_urls)
          .filter(([, urls]) => urls.splats && urls.skin)
          .map(([name, urls]) => ({ name, url: urls.splats as string, skinUrl: urls.skin as string }));
        if (preview.color_affine_url && loadedCameras?.length) {
          const buffer: ArrayBuffer = await (await fetch(preview.color_affine_url)).arrayBuffer();
          const data = new Float32Array(buffer);
          if (!alive) return;
          setAffine({ data, cameras: loadedCameras.length });
        }
        if (humans.length && preview.human_poses_url && preview.human_bones_url) {
          const [motion, bones] = await Promise.all([
            api<Omit<HumanMotion, "bones">>(preview.human_poses_url),
            fetch(preview.human_bones_url).then((response) => response.arrayBuffer()),
          ]);
          const added = await v.loadHumans(humans, { ...motion, bones: new Float32Array(bones) });
          if (!alive || added === null) return;
          total += added;
        }
      }
      // Trục lên: run của tool suy từ pose camera (hệ world DriveStudio tuỳ dataset: nuScenes -y,
      // Waymo z); file kéo thả đọc phần đầu PLY do tool ghi; file lạ thì đoán từ hình dạng cảnh
      // (mặt đất phẳng, nhà và cây mọc lên).
      const axis: UpAxis =
        source === "file" && file
          ? ((await headerUpAxis(file)) ?? v.estimateUpAxis() ?? "-y")
          : ((loadedCameras?.length ? upAxisFromCameras(loadedCameras) : null) ?? v.estimateUpAxis() ?? "z");
      if (!alive) return;
      v.setUpAxis(axis);
      setUp(axis);
      setCameras(loadedCameras);
      if (loadedCameras?.length) {
        const index = defaultCamera(loadedCameras);
        setCameraIndex(index);
        const camera = loadedCameras[index];
        v.setFovOverride(fovRef.current);
        v.setCameraPose(camera.c2w[0], camera.K[0], camera.width, camera.height);
      }
      v.setMaxSh(shRef.current);
      setCount(total);
      setProgress(null);
      if (loadedPoses) {
        setPoses(loadedPoses);
        setFrame(0);
        v.setPoses(loadedPoses.track_ids, loadedPoses.T_object_world[0] ?? []);
        v.setHumanFrame(0);
      }
    };
    run().catch((err) => alive && (setError(String(err)), setProgress(null)));
    return () => {
      alive = false;
    };
  }, [source, file, splats.data]);

  useEffect(() => viewer.current?.setUpAxis(up), [up]);
  useEffect(() => {
    if (!immersive) return;
    const timer = window.setInterval(() => setRate(viewer.current?.renderRate() ?? null), 500);
    return () => window.clearInterval(timer);
  }, [immersive]);
  // Bậc màu đọc qua ref khi tải xong: người dùng có thể đổi trong lúc file lớn đang tải.
  const shRef = useRef(sh);
  shRef.current = sh;
  useEffect(() => viewer.current?.setMaxSh(sh), [sh]);
  const fovRef = useRef(fov);
  fovRef.current = fov;
  useEffect(() => viewer.current?.setFovOverride(fov), [fov]);
  // Góc nhìn dọc của camera gốc đang chọn khi đưa K vào phép chiếu pinhole (fisheye ra khoảng 119°).
  const nativeFov = (() => {
    const camera = cameras?.[cameraIndex];
    if (!camera) return DEFAULT_FOV;
    return Math.round((2 * Math.atan(camera.height / (2 * camera.K[0][4])) * 180) / Math.PI);
  })();
  // Tiêu cự (pixel) ứng với góc nhìn dọc, theo chiều cao ảnh lúc train của camera đang chọn: cùng đơn
  // vị với K trong cameras.json (image_02 KITTI-360: 700 px, fy 208). f = (H / 2) / tan(góc / 2).
  const focalText = (deg: number) => {
    const height = cameras?.[cameraIndex]?.height;
    if (!height) return `${deg}°`;
    const focal = height / 2 / Math.tan((deg * Math.PI) / 360);
    return `f ≈ ${Math.round(focal)} px · ${deg}°`;
  };
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
    if (poses && !playing) {
      viewer.current?.setPoses(poses.track_ids, poses.T_object_world[frame] ?? []);
      viewer.current?.setHumanFrame(frame);
    }
  }, [frame, poses, playing]);

  const openFile = (picked: File | undefined) => {
    if (!picked) return;
    setFile(picked);
    setSource("file");
    setCameras(null);
    setFollow(false);
  };

  // Về đúng góc của camera gốc ở khung đang xem; khi "bám camera", tự đi theo khung lúc phát.
  // Đường novel dùng nội tham số của camera đầu tiên, như DriveStudio render video novel.
  const novel = useMemo<SceneCamera | null>(
    () => (cameras?.length ? { ...cameras[0], name: "novel", c2w: novelTrajectory(cameras[0].c2w) } : null),
    [cameras],
  );
  const camera = path === "novel" && novel ? novel : cameras?.[cameraIndex];
  const goToCamera = (at: number = frame) => {
    if (!camera) return;
    const index = Math.min(at, camera.c2w.length - 1);
    viewer.current?.setCameraPose(camera.c2w[index], camera.K[index], camera.width, camera.height);
  };
  // "Căn lại": về đúng góc lúc mới mở (camera mặc định, ở khung đang xem); không có camera gốc (file
  // kéo thả) thì nhìn bao quát cả cảnh.
  const recenter = () => {
    if (!cameras?.length) return viewer.current?.frameAll();
    const index = defaultCamera(cameras);
    setCameraIndex(index);
    const start = cameras[index];
    const at = Math.min(frame, start.c2w.length - 1);
    viewer.current?.setCameraPose(start.c2w[at], start.K[at], start.width, start.height);
  };
  useEffect(() => {
    if (follow && !playing) goToCamera(frame);
  }, [frame, path, cameraIndex, playing]); // goToCamera đọc state mới nhất mỗi lần render

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
      // Mốc giờ của requestAnimationFrame là lúc khung hình bắt đầu, có thể sớm hơn
      // performance.now() lấy ngay trước đó: không cho thời gian lùi, nếu không khung đầu thành −1
      // (c2w[−1] không có, setCameraPose ném lỗi và vòng phát dừng hẳn).
      time += (Math.max(0, now - previous) / 1000) * PLAYBACK_FPS;
      previous = Math.max(previous, now);
      if (time > last) time = 0;
      const index = Math.floor(time);
      const next = Math.min(index + 1, last);
      const alpha = time - index;
      viewer.current?.setPoses(poses.track_ids, poses.T_object_world[index] ?? [], poses.T_object_world[next] ?? [], alpha);
      viewer.current?.setHumanFrame(index, next, alpha);
      const cam = cameraRef.current;
      if (followRef.current && cam) {
        const a = Math.min(index, cam.c2w.length - 1);
        const b = Math.min(next, cam.c2w.length - 1);
        viewer.current?.setCameraPose(cam.c2w[a], cam.K[a], cam.width, cam.height, cam.c2w[b], alpha);
      }
      if (index !== frameRef.current) setFrame(index);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, poses]);

  // Camera quyết định ma trận chỉnh màu: camera đang chọn, hoặc camera đầu (đường novel dựng từ đó).
  const colorCamera = path === "novel" ? 0 : cameraIndex;
  const exact = useExactFrame(viewer, runId, source === "preview", frame, playing, colorCamera);
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    if (!affine) return v.setColorAffine(null);
    const frames = affine.data.length / (affine.cameras * 12);
    const offset = (Math.min(frame, frames - 1) * affine.cameras + Math.min(colorCamera, affine.cameras - 1)) * 12;
    v.setColorAffine(affine.data.subarray(offset, offset + 12));
  }, [affine, frame, colorCamera]);
  // Tiến trình GPU (viewer của Render GPU, frame server của Kết hợp) giữ model trong VRAM: tắt khi
  // rời chế độ, rời trang hay đóng tab, để GPU nhỏ (4 GB) còn chỗ cho lần bật sau.
  useEffect(() => {
    const kind = source === "gpu" ? "viewer" : source === "preview" ? "frames" : null;
    if (!kind) return;
    const stop = () => void fetch(`/api/gpu-viewer?kind=${kind}`, { method: "DELETE", keepalive: true }).catch(() => undefined);
    window.addEventListener("pagehide", stop);
    return () => {
      window.removeEventListener("pagehide", stop);
      stop();
    };
  }, [source]);
  const stalePreview = source === "preview" && !!splats.data?.preview && splats.data.preview.schema_version !== PREVIEW_SCHEMA;
  const startReexport = async () => {
    if (!runId) return;
    setReexport({ state: "running" });
    try {
      await post(`/api/runs/${seg(runId)}/export`, { kind: "preview" });
    } catch (err) {
      setReexport({ state: "error", error: asApiError(err).message });
    }
  };
  // Đang xuất lại: hỏi lại gói tới khi đúng phiên bản (khi đó cảnh tự nạp lại), quá lâu thì báo lỗi.
  useEffect(() => {
    if (reexport.state !== "running") return;
    if (!stalePreview) return setReexport({ state: "idle" });
    const started = Date.now();
    const timer = window.setInterval(() => {
      if (Date.now() - started > REEXPORT_TIMEOUT_MS) setReexport({ state: "error", error: t("viewer.stale.timeout") });
      else splats.reload();
    }, REEXPORT_POLL_MS);
    return () => window.clearInterval(timer);
  }, [reexport.state, stalePreview]);
  useEffect(() => setReexport({ state: "idle" }), [runId]);
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
    setFollow(true); // run mở sau lại bám camera gốc như mặc định
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
            <div className="launch-illo">
              <RoadScene width={300} />
            </div>
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
  // Tab "Thông tin" chỉ có khi đang xem một run (file kéo thả không có cấu hình, kết quả đo).
  const panelTabs: { tab: PanelTab; icon: typeof Info }[] = [
    { tab: "controls", icon: SlidersHorizontal },
    ...(runId ? [{ tab: "info" as PanelTab, icon: Info }] : []),
  ];
  const activePanel: PanelTab | null = panel === "closed" ? null : panelTabs.some((item) => item.tab === panel) ? panel : "controls";
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
      {source === "preview" && exact.image && <img className="exact-frame" src={exact.image} alt="" />}
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
          {count !== null && rate && (
            <span className="tag glass fps" title={t("viewer.hud.fps_hint")}>
              {rate.idle ? t("viewer.hud.fps_idle") : t("viewer.hud.fps", { fps: rate.fps })}
            </span>
          )}
          {poses && <span className="tag glass">{t("viewer.hud.actors", { count: poses.track_ids.length })}</span>}
          {source === "preview" && ["loading", "rendering", "ready"].includes(exact.state) && (
            <span className="tag glass" title={t("viewer.exact.hint")}>
              {t(`viewer.exact.${exact.state}`, { ms: exact.renderMs ?? "–" })}
            </span>
          )}
          {stalePreview && (
            <button
              className="btn sm glass"
              onClick={() => void startReexport()}
              disabled={reexport.state === "running"}
              title={reexport.state === "error" ? t("viewer.stale.error", { error: reexport.error ?? "" }) : t("viewer.stale.hint")}
            >
              {t(reexport.state === "running" ? "viewer.stale.running" : reexport.state === "error" ? "viewer.stale.retry" : "viewer.stale.button")}
            </button>
          )}
          <button className="btn sm glass" onClick={recenter} title={t("viewer.hud.recenter_hint")}>
            <Crosshair size={16} />
            {t("viewer.hud.recenter")}
          </button>
        </div>
      )}

      <nav className="vrail glass" aria-label={t("viewer.panel.rail")}>
        {panelTabs.map(({ tab, icon: Icon }) => (
          <button
            key={tab}
            type="button"
            className={`btn icon sm ghost${activePanel === tab ? " active" : ""}`}
            aria-pressed={activePanel === tab}
            aria-label={t(`viewer.panel.${tab}`)}
            title={t(activePanel === tab ? "viewer.panel.collapse" : `viewer.panel.${tab}`)}
            onClick={() => pickPanel(tab)}
          >
            <Icon size={18} />
          </button>
        ))}
      </nav>
      {activePanel && (
        <aside className="vpanel glass" aria-label={t(`viewer.panel.${activePanel}`)}>
          <header>
            <b>{t(`viewer.panel.${activePanel}`)}</b>
          </header>
          {activePanel === "controls" && (
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
                    <option key={v.url} value={playableUrl(v)}>
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
                    </div>
                    <span className="small muted row" style={{ gap: 2 }}>
                      {t("viewer.camera.path")}
                      <InfoTip>{t("viewer.camera.path_hint")}</InfoTip>
                    </span>
                    <Segmented<CameraPath>
                      label={t("viewer.camera.path")}
                      value={path}
                      onChange={setPath}
                      options={[
                        { value: "free", label: t("viewer.camera.path.free") },
                        { value: "camera", label: t("viewer.camera.path.camera") },
                        { value: "novel", label: t("viewer.camera.path.novel") },
                      ]}
                    />
                  </div>
                )}
                <div className="field">
                  <span>
                    {t("viewer.field.fov")}
                    <InfoTip>{t("viewer.field.fov_hint")}</InfoTip>
                  </span>
                  <div className="row" style={{ gap: 8 }}>
                    <input
                      type="range"
                      min={FOV_MIN}
                      max={FOV_MAX}
                      step={1}
                      value={Math.min(FOV_MAX, Math.max(FOV_MIN, fov ?? nativeFov))}
                      onChange={(e) => setFov(Number(e.target.value))}
                      aria-label={t("viewer.field.fov")}
                    />
                    <span className="small">{focalText(fov ?? nativeFov)}</span>
                  </div>
                  <label className="row small" style={{ gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={fov === null}
                      onChange={(e) => setFov(e.target.checked ? null : Math.min(FOV_MAX, Math.max(FOV_MIN, nativeFov)))}
                    />
                    {t("viewer.fov.native")}
                  </label>
                </div>
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
          )}
          {activePanel === "info" && runId && (
            <div className="vpanel-body">
              <RunInfo runId={runId} />
            </div>
          )}
        </aside>
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
