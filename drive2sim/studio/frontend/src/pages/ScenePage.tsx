// Một cảnh: dữ kiện chính và hành động ở đầu trang; các tab ảnh đồng bộ của mọi camera (kéo thanh
// thời gian, phím trái/phải, nút phát), split (tạo, xoá, chạy), camera, báo cáo kiểm tra, và các
// run trên cảnh.
import { Pause, Play, RefreshCw, Trash2, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { asApiError, del, post, seg, useApi, type ApiError } from "../api/client";
import type { FrameImages, List, RunRecord, SceneDetail, Split } from "../api/types";
import { ago, Confirm, Empty, ErrorBox, InfoTip, Loading, PageHead, Panel, Status, Tabs, useToast, yesNo } from "../components/ui";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { go, href } from "../router";

type Tab = "frames" | "splits" | "cameras" | "checks" | "runs";

export function ScenePage({ sceneId }: { sceneId: string }) {
  const { t, locale } = useI18n();
  const toast = useToast();
  const live = useLive();
  const scene = useApi<SceneDetail>(`/api/scenes/${seg(sceneId)}`);
  const runs = useApi<List<RunRecord>>(`/api/runs?scene_id=${seg(sceneId)}&_=${live.tick}`);
  const [tab, setTab] = useState<Tab>("frames");
  const [deleting, setDeleting] = useState(false);

  if (scene.error) return <div className="page"><ErrorBox error={scene.error} /></div>;
  if (!scene.data) return <div className="page"><Loading rows={5} /></div>;
  const { manifest, validation, splits } = scene.data;

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
          {tab === "frames" && <Frames sceneId={sceneId} numFrames={manifest.timeline.num_frames} splits={splits} />}
          {tab === "splits" && (
            <Splits
              sceneId={sceneId}
              cameras={manifest.cameras.map((c) => c.camera_id)}
              numFrames={manifest.timeline.num_frames}
              splits={splits}
              onChange={scene.reload}
            />
          )}
          {tab === "cameras" && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>{t("common.camera")}</th>
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
                      <td>{c.model}</td>
                      <td>{`${c.resolution[0]}×${c.resolution[1]}`}</td>
                      <td>{c.shutter}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
                        <td><Status value={live.statuses[run.run_id] ?? run.status} /></td>
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

function Frames({ sceneId, numFrames, splits }: { sceneId: string; numFrames: number; splits: string[] }) {
  const { t } = useI18n();
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [splitId, setSplitId] = useState(splits[0] ?? "");
  const images = useApi<FrameImages>(`/api/scenes/${seg(sceneId)}/frames/${frame}`);
  const split = useApi<Split>(splitId ? `/api/scenes/${seg(sceneId)}/splits/${seg(splitId)}` : null);
  const last = numFrames - 1;
  const width = 100 / numFrames;
  const testFrames = new Set((split.data?.test ?? []).map(([f]) => f));
  const trainFrames = new Set((split.data?.train ?? []).map(([f]) => f));
  const kind = testFrames.has(frame) ? "test" : trainFrames.has(frame) ? "train" : null;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setFrame((f) => (f >= last ? 0 : f + 1)), 400);
    return () => window.clearInterval(timer);
  }, [playing, last]);

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
      {split.data && (
        <div className="legend">
          <span><i style={{ background: "var(--line-strong)" }} />{t("scene.frames.train")}</span>
          <span><i style={{ background: "var(--accent)" }} />{t("scene.frames.test")}</span>
          <InfoTip label={t("scene.frames.keys_label")}>{t("scene.frames.keys")}</InfoTip>
        </div>
      )}
      <div className="frames">
        {(images.data?.images ?? []).map((image) => (
          <figure key={image.camera_id}>
            {image.url ? <img src={image.url} alt={`${image.camera_id}, ${frame}`} /> : <div className="missing">{t("scene.frames.missing")}</div>}
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
  onChange,
}: {
  sceneId: string;
  cameras: string[];
  numFrames: number;
  splits: string[];
  onChange: () => void;
}) {
  const { t } = useI18n();
  const toast = useToast();
  const [every, setEvery] = useState(8);
  const [chosen, setChosen] = useState<string[]>(cameras);
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
            <span className="id grow ellipsis">{id}</span>
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
            <span>{t("scene.splits.cameras")}</span>
            <div className="row wrap">
              {cameras.map((camera) => (
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
