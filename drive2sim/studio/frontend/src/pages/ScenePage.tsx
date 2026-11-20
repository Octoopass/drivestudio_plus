// Một cảnh: dữ kiện chính và hành động ở đầu trang; các tab ảnh đồng bộ của mọi camera (kéo thanh
// thời gian, phím trái/phải, nút phát), split (tạo, xoá, chạy), camera, báo cáo kiểm tra, và các
// run trên cảnh.
import { Pause, Play, RefreshCw, Trash2, Plus } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, asApiError, del, post, seg, useApi, type ApiError } from "../api/client";
import type { FrameImages, List, SceneDetail, Split } from "../api/types";
import { ago, Confirm, Empty, ErrorBox, InfoTip, Loading, Note, PageHead, Panel, Status, Tabs, useToast, yesNo } from "../components/ui";
import { useI18n } from "../i18n";
import { lensGroup, lensOf, lensSummary, type Lens } from "../lens";
import { SceneRig } from "../components/SceneRig";
import { LidarOverlay } from "../components/LidarOverlay";
import { useLive } from "../live";
import { go, href } from "../router";
import { runStatus, type RunItem } from "../runStatus";

type Tab = "frames" | "splits" | "cameras" | "checks" | "runs";

export function ScenePage({ sceneId }: { sceneId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const live = useLive();
  const scene = useApi<SceneDetail>(`/api/scenes/${seg(sceneId)}`);
  const runs = useApi<List<RunItem>>(`/api/runs?scene_id=${seg(sceneId)}&_=${live.tick}`);
  const [tab, setTab] = useState<Tab>("frames");
  const [deleting, setDeleting] = useState(false);

  if (scene.error) return <div className="page"><ErrorBox error={scene.error} /></div>;
  if (!scene.data) return <div className="page"><Loading rows={5} /></div>;
  const { manifest, validation, splits, kaggle } = scene.data;
  const outdated = scene.data.converter_outdated ?? null;
  const splitCameras = scene.data.split_cameras ?? {};

  const revalidate = async () => {
    try {
      const report = await post<{ summary: { errors: number; warnings: number } }>(`/api/scenes/${seg(sceneId)}/validate`);
      toast({ kind: report.summary.errors ? "bad" : "ok", title: t("scene.checks.summary", report.summary) });
      scene.reload();
    } catch (err) {
      toast({ kind: "bad", title: t("scene.checks.failed"), detail: asApiError(err).message });
    }
  };

  const checkCount = validation ? validation.summary.errors + validation.summary.warnings : null;
  return (
    <div className="page">
      <PageHead
        title={manifest.scene_id}
        meta={
          <dl className="facts">
            <div>
              <dt>{t("scene.facts.source")}</dt>
              <dd>{`${manifest.source.dataset} ${manifest.source.sequence_id}`}</dd>
            </div>
            <div>
              <dt>{t("scene.facts.frames")}</dt>
              <dd>{manifest.timeline.num_frames}</dd>
            </div>
            <div>
              <dt>{t("scene.facts.cameras")}</dt>
              <dd>{manifest.cameras.length}</dd>
            </div>
            <div>
              <dt>LiDAR</dt>
              <dd>{yesNo(manifest.available.lidar, t)}</dd>
            </div>
            <div>
              <dt>{t("scene.facts.cuboids")}</dt>
              <dd>{manifest.available.cuboids}</dd>
            </div>
            <div>
              <dt>{t("scene.facts.converter")}</dt>
              <dd>{manifest.source.converter}</dd>
            </div>
          </dl>
        }
        actions={
          <>
            <button className="btn ghost icon" onClick={() => setDeleting(true)} title={t("scene.action.delete")} aria-label={t("scene.action.delete")}>
              <Trash2 size={20} />
            </button>
            <button className="btn" onClick={revalidate}>
              <RefreshCw size={19} />
              {t("scene.action.validate")}
            </button>
            {/* Cảnh chưa có split thì chưa chạy được: việc tiếp theo là tạo split. */}
            {splits.length === 0 ? (
              <button className="btn primary" onClick={() => setTab("splits")}>
                <Plus size={19} />
                {t("scene.action.create_split")}
              </button>
            ) : (
              <a className="btn primary" href={href({ page: "new", sceneId })}>
                <Play size={19} />
                {t("nav.menu.new_run")}
              </a>
            )}
          </>
        }
      />

      {/* Cảnh ingest bằng converter cũ: chỉ cảnh báo, run trên cảnh vẫn chạy được. */}
      {outdated && (
        <Note kind="warn">
          <strong>{t("scene.outdated.title", { converter: manifest.source.converter })}</strong>
          <div className="detail">
            {t("scene.outdated.detail", { recorded: outdated.output_version, current: outdated.current_output_version })}
          </div>
        </Note>
      )}

      <section className="panel">
        <Tabs<Tab>
          value={tab}
          onChange={setTab}
          items={[
            { value: "frames", label: t("scene.tab.frames") },
            { value: "splits", label: t("scene.tab.splits"), badge: splits.length },
            { value: "cameras", label: t("scene.tab.cameras"), badge: manifest.cameras.length },
            { value: "checks", label: t("scene.tab.checks"), badge: checkCount },
            { value: "runs", label: t("scene.tab.runs"), badge: runs.data?.total },
          ]}
        />
        <div className="panel-body">
          {tab === "frames" &&
            (kaggle ? (
              <div className="stack" style={{ gap: 8 }}>
                <p>{t("scene.kaggle.title")}</p>
                <p className="small muted">{t("scene.kaggle.detail", { kernel: kaggle.map((k) => k.kernel).join(", ") })}</p>
              </div>
            ) : (
              <Frames sceneId={sceneId} numFrames={manifest.timeline.num_frames} splits={splits} />
            ))}
          {tab === "splits" && (
            <Splits
              sceneId={sceneId}
              cameras={manifest.cameras.map((c) => ({ id: c.camera_id, model: c.model }))}
              numFrames={manifest.timeline.num_frames}
              splits={splits}
              splitCameras={splitCameras}
              onChange={scene.reload}
            />
          )}
          {tab === "cameras" && (
            <div className="stack" style={{ gap: 10 }}>
            <SceneRig sceneId={sceneId} />
            <p className="small muted row wrap" style={{ gap: 6 }}>
              {(() => {
                const group = lensGroup(manifest.cameras.map((c) => c.model));
                return group && <span className="tag">{t(`lens.${group}`)}</span>;
              })()}
              {t("lens.scene_summary", { count: manifest.cameras.length, parts: lensSummary(manifest.cameras.map((c) => c.model), t) })}
              <InfoTip>{t("lens.hint")}</InfoTip>
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>{t("common.camera")}</th>
                    <th>{t("scene.cameras.lens")}</th>
                    <th>{t("scene.cameras.model")}</th>
                    <th>{t("scene.cameras.resolution")}</th>
                    <th>{t("scene.cameras.shutter")}</th>
                  </tr>
                </thead>
                <tbody>
                  {manifest.cameras.map((c, index) => (
                    <tr key={c.camera_id}>
                      <td className="faint">{index}</td>
                      <td className="id">{c.camera_id}</td>
                      <td>
                        <span className="tag">{t(`lens.${lensOf(c.model)}`)}</span>
                      </td>
                      <td>
                        {t(`lens.model.${c.model}`) !== `lens.model.${c.model}` ? t(`lens.model.${c.model}`) : c.model} <code className="faint">{c.model}</code>
                      </td>
                      <td>{`${c.resolution[0]}×${c.resolution[1]}`}</td>
                      <td>{c.shutter}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            </div>
          )}
          {tab === "checks" &&
            (!validation ? (
              <Empty title={t("scene.checks.none")} action={<button className="btn" onClick={revalidate}>{t("scene.action.validate")}</button>} />
            ) : (
              <div className="stack" style={{ gap: 12 }}>
                <p>{t("scene.checks.summary", { errors: validation.summary.errors, warnings: validation.summary.warnings })}</p>
                <div className="table-wrap">
                  <table>
                    <tbody>
                      {(validation.checks ?? []).map((check, i) => (
                        <tr key={i}>
                          <td className="nowrap">
                            <span className={`status check-${check.severity}`}>{check.check_id}</span>
                          </td>
                          <td className="muted nowrap">{check.camera_id ?? ""}</td>
                          <td>{check.message}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {manifest.available.lidar && <LidarOverlay sceneId={sceneId} kaggle={Boolean(kaggle)} onChange={scene.reload} />}
              </div>
            ))}
          {tab === "runs" &&
            (runs.data?.total ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>{t("runs.table.run")}</th>
                      <th>{t("runs.table.split")}</th>
                      <th>{t("runs.table.approach")}</th>
                      <th>{t("runs.table.status")}</th>
                      <th className="num">{t("runs.table.created")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.items.map((run) => (
                      <tr key={run.run_id} className="link-row" onClick={() => go({ page: "run", runId: run.run_id })}>
                        <td className="id">{run.run_id}</td>
                        <td>{run.split_id}</td>
                        <td>{run.approach.approach_id}</td>
                        <td><Status value={runStatus(run, live.statuses[run.run_id])} /></td>
                        <td className="num muted">{ago(run.created_utc ?? null, locale)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty title={t("runs.empty.title")} action={<a className="btn" href={href({ page: "new", sceneId })}>{t("nav.menu.new_run")}</a>} />
            ))}
        </div>
      </section>

      {deleting && (
        <Confirm
          title={t("scene.delete.title", { scene: sceneId })}
          body={t("scene.delete.body", { runs: runs.data?.total ?? 0, splits: splits.length })}
          confirmLabel={t("scene.action.delete")}
          danger
          onClose={() => setDeleting(false)}
          onConfirm={async () => {
            const result = await del<{ removed: string[] }>(`/api/scenes/${seg(sceneId)}`);
            toast({ kind: "ok", title: t("scene.delete.done"), detail: result.removed.join(", ") });
            go({ page: "scenes" });
          }}
        />
      )}
    </div>
  );
}

// Tốc độ phát xem trước: các cảnh trong tool đều ~10 Hz (nuScenes nội suy lên 10 Hz, KITTI-360
// 10 Hz); manifest chưa ghi tần số.
const PLAY_FPS = 10;
// Cửa sổ trượt quanh khung đang hiện: tải sẵn WINDOW khung phía trước và WINDOW khung phía sau
// (lùi, tua lại cũng hiện ngay), bỏ các khung ra khỏi cửa sổ. Ảnh giải mã chiếm width × height × 4
// byte: nuScenes 6 ảnh 1600×900 ≈ 35 MB mỗi khung, nên 11 khung (66 ảnh) ≈ 380 MB; KITTI-360
// 4 ảnh ≈ 20 MB mỗi khung, 11 khung (44 ảnh) ≈ 220 MB. Dùng chung cho mọi dataset.
const WINDOW = 5;

/**
 * Tải trước khung ảnh theo cửa sổ trượt ``[khung đang hiện − WINDOW, + WINDOW]`` (vòng về đầu
 * khi tới cuối): đường dẫn ảnh (``/frames/{i}``) rồi mọi ảnh của khung, giải mã sẵn
 * (``img.decode``), để khi đổi khung trình duyệt vẽ ngay.
 *
 * ``slide(khung)`` (gọi mỗi lần đổi khung) dời cửa sổ: bỏ tham chiếu các khung đã ra ngoài để
 * trình duyệt thu hồi bộ nhớ, rồi tải các khung mới vào, khung gần trước, phía trước trước phía
 * sau. Rời trang hoặc đổi cảnh thì bỏ hết.
 */
function useFramePreloader(sceneId: string, numFrames: number) {
  return useMemo(() => {
    const pending = new Map<number, Promise<FrameImages>>();
    const decoded = new Map<number, { data: FrameImages; imgs: HTMLImageElement[] }>();
    let current = 0;
    const inWindow = (index: number) => {
      const ahead = (index - current + numFrames) % numFrames;
      return ahead <= WINDOW || numFrames - ahead <= WINDOW;
    };
    const load = (index: number): Promise<FrameImages> => {
      const hit = decoded.get(index);
      if (hit) return Promise.resolve(hit.data);
      const running = pending.get(index);
      if (running) return running;
      const promise = api<FrameImages>(`/api/scenes/${seg(sceneId)}/frames/${index}`)
        .then(async (data) => {
          const imgs = data.images.filter((image) => image.url).map((image) => {
            const img = new Image();
            img.src = image.url as string;
            return img;
          });
          await Promise.all(imgs.map((img) => img.decode().catch(() => undefined)));
          // Trong lúc tải, cửa sổ có thể đã trượt đi: khung ngoài cửa sổ thì không giữ.
          if (inWindow(index)) decoded.set(index, { data, imgs });
          return data;
        })
        .finally(() => pending.delete(index));
      pending.set(index, promise);
      return promise;
    };
    const slide = (index: number) => {
      current = index;
      for (const key of [...decoded.keys()]) if (!inWindow(key)) decoded.delete(key);
      for (let k = 1; k <= WINDOW; k++) {
        load((index + k) % numFrames).catch(() => undefined);
        load((index - k + numFrames) % numFrames).catch(() => undefined);
      }
    };
    return { load, slide, isReady: (index: number) => decoded.has(index) };
  }, [sceneId, numFrames]);
}

function Frames({ sceneId, numFrames, splits }: { sceneId: string; numFrames: number; splits: string[] }) {
  const { t } = useI18n();
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [splitId, setSplitId] = useState(splits[0] ?? "");
  // Công cụ kiểm tra dữ liệu (frame_tools.py): LiDAR chồng lên ảnh, ảnh toàn cảnh; chỉ khi dừng phát.
  const [lidar, setLidar] = useState(false);
  const [panorama, setPanorama] = useState(false);
  const [toolError, setToolError] = useState<string | null>(null);
  // Ảnh đang hiện: giữ khung cũ tới khi mọi ảnh của khung mới đã tải và giải mã xong, để không bị
  // trống, nhấp nháy giữa hai khung.
  const [shown, setShown] = useState<FrameImages | null>(null);
  const frames = useFramePreloader(sceneId, numFrames);
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const split = useApi<Split>(splitId ? `/api/scenes/${seg(sceneId)}/splits/${seg(splitId)}` : null);
  const last = numFrames - 1;
  const width = 100 / numFrames;
  const testFrames = new Set((split.data?.test ?? []).map(([f]) => f));
  const trainFrames = new Set((split.data?.train ?? []).map(([f]) => f));
  const kind = testFrames.has(frame) ? "test" : trainFrames.has(frame) ? "train" : null;

  // Khung đang chọn: hiện ngay khi sẵn sàng (kéo thanh trượt nhanh thì chỉ hiện khung cuối).
  useEffect(() => {
    let cancelled = false;
    frames.load(frame).then((data) => !cancelled && frameRef.current === frame && setShown(data)).catch(() => undefined);
    frames.slide(frame);
    return () => {
      cancelled = true;
    };
  }, [frame, frames]);

  // Phát: PLAY_FPS khung/giây; khung kế chưa tải xong thì chờ (chậm lại chứ không giật, không trống).
  // Cửa sổ tải trước do slide() lo mỗi lần đổi khung.
  useEffect(() => {
    if (!playing) return;
    let timer = 0;
    let stopped = false;
    const step = () => {
      const current = frameRef.current;
      const next = current >= last ? 0 : current + 1;
      if (frames.isReady(next)) setFrame(next);
      if (!stopped) timer = window.setTimeout(step, 1000 / PLAY_FPS);
    };
    step();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [playing, last, numFrames, frames]);

  const onKey = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowRight") setFrame((f) => Math.min(last, f + 1));
    else if (event.key === "ArrowLeft") setFrame((f) => Math.max(0, f - 1));
    else if (event.key === " ") setPlaying((p) => !p);
    else return;
    event.preventDefault();
  };

  return (
    <div className="stack" style={{ gap: 15 }} onKeyDown={onKey}>
      <div className="row wrap" style={{ gap: 15 }}>
        <button className="btn icon" onClick={() => setPlaying((p) => !p)} aria-label={t(playing ? "scene.frames.pause" : "scene.frames.play")}>
          {playing ? <Pause size={19} /> : <Play size={19} />}
        </button>
        <div className="grow stack" style={{ gap: 6, minWidth: 276 }}>
          <div
            className="timeline"
            onClick={(e) => {
              const box = e.currentTarget.getBoundingClientRect();
              setFrame(Math.min(last, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * numFrames))));
            }}
            aria-hidden="true"
          >
            {Array.from({ length: numFrames }, (_, f) =>
              testFrames.has(f) || trainFrames.has(f) ? (
                <span key={f} className={testFrames.has(f) ? "test" : "train"} style={{ left: `${f * width}%`, width: `max(${width}%, 2px)` }} />
              ) : null,
            )}
            <span className="cursor" style={{ left: `calc(${(frame + 0.5) * width}% - 1px)` }} />
          </div>
          <input type="range" min={0} max={last} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label={t("scene.frames.frame", { frame, last })} />
        </div>
        <span className="dense nowrap" style={{ minWidth: 113 }}>
          {t("scene.frames.frame", { frame, last })}
          <span className="small faint" style={{ marginLeft: 8 }}>{t("scene.frames.fps", { fps: PLAY_FPS })}</span>
          {kind && <span className={`tag ${kind === "test" ? "warn" : ""}`} style={{ marginLeft: 8 }}>{kind}</span>}
        </span>
        <label className="field" style={{ minWidth: 227 }}>
          <span>{t("scene.frames.split")}</span>
          <select value={splitId} onChange={(e) => setSplitId(e.target.value)}>
            <option value="">{t("scene.frames.none")}</option>
            {splits.map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        <span className="small muted">
          {t("scene.tools.title")}
          <InfoTip>{t("scene.tools.help")}</InfoTip>
        </span>
        <button className={`btn sm${lidar ? " primary" : ""}`} type="button" onClick={() => (setLidar(!lidar), setToolError(null))}>
          {t("scene.tools.lidar")}
        </button>
        <button className={`btn sm${panorama ? " primary" : ""}`} type="button" onClick={() => (setPanorama(!panorama), setToolError(null))}>
          {t("scene.tools.panorama")}
        </button>
        {(lidar || panorama) && playing && <span className="small faint">{t("scene.tools.paused_only")}</span>}
        {toolError && <span className="small bad-ink">{toolError}</span>}
      </div>
      {panorama && !playing && shown && (
        <figure className="stack" style={{ gap: 4 }}>
          <img
            key={`pano-${shown.frame_index}`}
            src={`/api/scenes/${seg(sceneId)}/frames/${shown.frame_index}/panorama`}
            alt={t("scene.tools.panorama")}
            style={{ width: "100%", borderRadius: 8 }}
            onError={() => setToolError(t("scene.tools.failed"))}
          />
          <figcaption className="small muted">{t("scene.tools.panorama_caption")}</figcaption>
        </figure>
      )}
      {split.data && (
        <div className="legend">
          <span><i style={{ background: "var(--line-strong)" }} />{t("scene.frames.train")}</span>
          <span><i style={{ background: "var(--accent)" }} />{t("scene.frames.test")}</span>
          <InfoTip label={t("scene.frames.keys_label")}>{t("scene.frames.keys")}</InfoTip>
        </div>
      )}
      <div className="frames">
        {(shown?.images ?? []).map((image) => (
          <figure key={image.camera_id}>
            {image.url ? (
              <img
                src={
                  lidar && !playing
                    ? `/api/scenes/${seg(sceneId)}/frames/${shown?.frame_index}/lidar?camera=${seg(image.camera_id)}`
                    : image.url
                }
                alt={`${image.camera_id}, ${shown?.frame_index}`}
                onError={() => lidar && setToolError(t("scene.tools.failed"))}
              />
            ) : (
              <div className="missing">{t("scene.frames.missing")}</div>
            )}
            <figcaption>{image.camera_id}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

function Splits({
  sceneId,
  cameras,
  numFrames,
  splits,
  splitCameras,
  onChange,
}: {
  sceneId: string;
  cameras: { id: string; model: string }[];
  numFrames: number;
  splits: string[];
  splitCameras: Record<string, string[]>;
  onChange: () => void;
}) {
  const { t } = useI18n();
  const toast = useToast();
  const [every, setEvery] = useState(8);
  const [chosen, setChosen] = useState<string[]>(cameras.map((c) => c.id));
  const modelOf = (id: string) => cameras.find((c) => c.id === id)?.model ?? "";
  // Camera chia theo loại ống kính: cảnh hỗn hợp (KITTI-360) chọn nhanh được chỉ pinhole, chỉ mắt cá.
  const groups = (["pinhole", "fisheye"] as Lens[])
    .map((lens) => ({ lens, ids: cameras.filter((c) => lensOf(c.model) === lens).map((c) => c.id) }))
    .filter((group) => group.ids.length > 0);
  const chosenModels = chosen.map(modelOf);
  // Đoạn khung (SPEC mục 5 bước 3), tính cả hai đầu; mặc định cả cảnh.
  const [first, setFirst] = useState(0);
  const [last, setLast] = useState(Math.max(0, numFrames - 1));
  const rangeOk = first >= 0 && last < numFrames && first <= last;
  const [error, setError] = useState<ApiError | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const create = async () => {
    setError(null);
    try {
      const whole = first === 0 && last === numFrames - 1;
      const split = await post<Split>(`/api/scenes/${seg(sceneId)}/splits`, {
        strategy: `every_nth_frame:${every}`,
        cameras: chosen,
        ...(whole ? {} : { frame_range: [first, last] }),
      });
      toast({ kind: "ok", title: t("scene.splits.created", { split: split.split_id }) });
      onChange();
    } catch (err) {
      setError(asApiError(err));
    }
  };

  return (
    <div className="grid-2">
      <div className="stack" style={{ gap: 10 }}>
        <h3>{t("scene.splits.title")}</h3>
        {splits.length === 0 && <p className="muted">{t("scene.splits.none")}</p>}
        {splits.map((id) => (
          <div key={id} className="row panel" style={{ padding: "10px 12px" }}>
            <span className="grow stack" style={{ gap: 2, minWidth: 0 }}>
              <span className="id ellipsis">{id}</span>
              {splitCameras[id] && (
                <span className="small muted ellipsis">
                  {lensSummary(splitCameras[id].map(modelOf), t)} · {splitCameras[id].join(", ")}
                </span>
              )}
            </span>
            <a className="btn sm" href={href({ page: "new", sceneId, splitId: id })}>
              <Play size={15} />
              {t("scene.splits.run")}
            </a>
            <a className="btn sm ghost" href={href({ page: "compare", sceneId, splitId: id })}>
              {t("nav.menu.compare")}
            </a>
            <button className="btn sm ghost icon" onClick={() => setDeleting(id)} aria-label={t("scene.splits.delete", { split: id })}>
              <Trash2 size={16} />
            </button>
          </div>
        ))}
      </div>
      <Panel title={t("scene.splits.new")}>
        <div className="stack" style={{ gap: 15 }}>
          <label className="field" style={{ maxWidth: 252 }}>
            <span>
              {t("scene.splits.every")}
              <InfoTip>{t("scene.splits.every_hint", { n: every })}</InfoTip>
            </span>
            <input type="number" min={1} value={every} onChange={(e) => setEvery(Number(e.target.value))} />
          </label>
          <div className="field">
            <span>
              {t("scene.splits.range")}
              <InfoTip>{t("scene.splits.range_hint", { last: numFrames - 1 })}</InfoTip>
            </span>
            <div className="row" style={{ gap: 8, maxWidth: 320 }}>
              <input type="number" min={0} max={numFrames - 1} value={first} onChange={(e) => setFirst(Number(e.target.value))} aria-label={t("scene.splits.range_first")} />
              <span className="muted">–</span>
              <input type="number" min={0} max={numFrames - 1} value={last} onChange={(e) => setLast(Number(e.target.value))} aria-label={t("scene.splits.range_last")} />
            </div>
          </div>
          <div className="field">
            <span>
              {t("scene.splits.cameras")}
              <InfoTip>{t("lens.hint")}</InfoTip>
            </span>
            {groups.length > 1 && (
              <div className="row wrap" style={{ gap: 6 }}>
                <button type="button" className="btn sm" onClick={() => setChosen(cameras.map((c) => c.id))}>
                  {t("scene.splits.quick_all")}
                </button>
                {groups.map((group) => (
                  <button key={group.lens} type="button" className="btn sm" onClick={() => setChosen(group.ids)}>
                    {t(`scene.splits.quick_${group.lens}`)}
                  </button>
                ))}
              </div>
            )}
            {groups.map((group) => (
              <div key={group.lens} className="row wrap" style={{ gap: 6, alignItems: "center" }}>
                <span className="tag">{t(`lens.${group.lens}`)}</span>
                {group.ids.map((camera) => (
                  <label key={camera} className="chip-check">
                    <input
                      type="checkbox"
                      checked={chosen.includes(camera)}
                      onChange={(e) => setChosen(e.target.checked ? [...chosen, camera] : chosen.filter((c) => c !== camera))}
                    />
                    {camera}
                  </label>
                ))}
              </div>
            ))}
            {chosen.length > 0 && (
              <p className="small muted">
                {lensSummary(chosenModels, t)}. {t(chosenModels.some((m) => lensOf(m) === "fisheye") ? "scene.splits.lens_fisheye" : "scene.splits.lens_pinhole")}
              </p>
            )}
          </div>
          <div>
            <button className="btn primary" onClick={create} disabled={chosen.length === 0 || every < 1 || !rangeOk}>
              {t("scene.splits.create")}
            </button>
          </div>
          <ErrorBox error={error} />
        </div>
      </Panel>
      {deleting && (
        <Confirm
          title={t("scene.splits.delete", { split: deleting })}
          body={t("scene.splits.delete_body")}
          confirmLabel={t("common.action.delete")}
          danger
          onClose={() => setDeleting(null)}
          onConfirm={async () => {
            await del(`/api/scenes/${seg(sceneId)}/splits/${seg(deleting)}`);
            toast({ kind: "ok", title: t("scene.splits.deleted", { split: deleting }) });
            onChange();
          }}
        />
      )}
    </div>
  );
}
