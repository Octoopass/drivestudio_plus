// Nhóm A của một cảnh: tổng quan, camera, ảnh đồng bộ của mọi camera theo khung kèm thanh thời
// gian hiện split, báo cáo kiểm tra, các split và form tạo split.
import { useState } from "react";
import { ApiError, post, useApi } from "../api/client";
import type { FrameImages, SceneDetail, Split } from "../api/types";
import { ErrorBox, Loading, yesNo } from "../components/common";
import { useI18n } from "../i18n";
import { href } from "../router";

export function ScenePage({ sceneId }: { sceneId: string }) {
  const { t } = useI18n();
  const scene = useApi<SceneDetail>(`/api/scenes/${encodeURIComponent(sceneId)}`);
  if (scene.error) return <ErrorBox error={scene.error} />;
  if (!scene.data) return <Loading />;
  const { manifest, validation, splits } = scene.data;
  return (
    <>
      <h1>{manifest.scene_id}</h1>
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
          <dt>{t("scene.facts.track")}</dt>
          <dd>{manifest.track}</dd>
        </div>
        <div>
          <dt>{t("scene.facts.lidar")}</dt>
          <dd>{yesNo(manifest.available.lidar, t)}</dd>
        </div>
        <div>
          <dt>{t("scene.facts.cuboids")}</dt>
          <dd>{manifest.available.cuboids}</dd>
        </div>
      </dl>

      <Frames sceneId={sceneId} numFrames={manifest.timeline.num_frames} splits={splits} />

      <div className="cols">
        <section>
          <h2>{t("scene.cameras.title")}</h2>
          <table>
            <thead>
              <tr>
                <th>camera_id</th>
                <th>{t("scene.cameras.model")}</th>
                <th>{t("scene.cameras.resolution")}</th>
                <th>{t("scene.cameras.shutter")}</th>
              </tr>
            </thead>
            <tbody>
              {manifest.cameras.map((c) => (
                <tr key={c.camera_id}>
                  <td>{c.camera_id}</td>
                  <td>{c.model}</td>
                  <td>{`${c.resolution[0]}×${c.resolution[1]}`}</td>
                  <td>{c.shutter}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section>
          <h2>{t("scene.checks.title")}</h2>
          {!validation && <p className="muted">{t("scene.checks.none")}</p>}
          {validation && (
            <>
              <p>{t("scene.checks.summary", { errors: validation.summary.errors, warnings: validation.summary.warnings })}</p>
              <ul className="small">
                {(validation.checks ?? []).map((check, i) => (
                  <li key={i}>
                    <span className={`status check-${check.severity}`}>
                      {check.check_id}
                    </span>{" "}
                    {check.message}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      </div>

      <Splits sceneId={sceneId} cameras={manifest.cameras.map((c) => c.camera_id)} splits={splits} onCreated={scene.reload} />
    </>
  );
}

function Frames({ sceneId, numFrames, splits }: { sceneId: string; numFrames: number; splits: string[] }) {
  const { t } = useI18n();
  const [frame, setFrame] = useState(0);
  const [splitId, setSplitId] = useState(splits[0] ?? "");
  const images = useApi<FrameImages>(`/api/scenes/${encodeURIComponent(sceneId)}/frames/${frame}`);
  const split = useApi<Split>(splitId ? `/api/scenes/${encodeURIComponent(sceneId)}/splits/${encodeURIComponent(splitId)}` : null);
  const last = numFrames - 1;
  const width = 100 / numFrames;
  const testFrames = new Set((split.data?.test ?? []).map(([f]) => f));
  const trainFrames = new Set((split.data?.train ?? []).map(([f]) => f));
  return (
    <section>
      <h2>{t("scene.frames.title")}</h2>
      <div className="form-row">
        <label className="field" style={{ flex: 1 }}>
          {t("scene.frames.frame", { frame, last })}
          <input type="range" min={0} max={last} value={frame} onChange={(e) => setFrame(Number(e.target.value))} />
        </label>
        <label className="field">
          {t("scene.frames.split")}
          <select value={splitId} onChange={(e) => setSplitId(e.target.value)}>
            <option value="">{t("scene.frames.none")}</option>
            {splits.map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
      </div>
      {split.data && (
        <>
          <div className="timeline" aria-hidden="true">
            {Array.from({ length: numFrames }, (_, f) =>
              testFrames.has(f) || trainFrames.has(f) ? (
                <span key={f} className={testFrames.has(f) ? "test" : "train"} style={{ left: `${f * width}%`, width: `${width}%` }} />
              ) : null,
            )}
            <span className="cursor" style={{ left: `${(frame + 0.5) * width}%` }} />
          </div>
          <div className="legend">
            <span><i style={{ background: "var(--ink-soft)" }} />{t("scene.frames.train")}</span>
            <span><i style={{ background: "var(--lane)" }} />{t("scene.frames.test")}</span>
          </div>
        </>
      )}
      <div className="frames" style={{ marginTop: 12 }}>
        {(images.data?.images ?? []).map((image) => (
          <figure key={image.camera_id}>
            {image.url ? <img src={image.url} alt={`${image.camera_id}, ${frame}`} loading="lazy" /> : <div className="missing">{t("scene.frames.missing")}</div>}
            <figcaption>{image.camera_id}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function Splits({ sceneId, cameras, splits, onCreated }: { sceneId: string; cameras: string[]; splits: string[]; onCreated: () => void }) {
  const { t } = useI18n();
  const [every, setEvery] = useState(10);
  const [chosen, setChosen] = useState<string[]>(cameras);
  const [error, setError] = useState<ApiError | null>(null);
  const [created, setCreated] = useState<string | null>(null);

  const create = async () => {
    setError(null);
    try {
      const split = await post<Split>(`/api/scenes/${encodeURIComponent(sceneId)}/splits`, {
        strategy: `every_nth_frame:${every}`,
        cameras: chosen,
      });
      setCreated(split.split_id);
      onCreated();
    } catch (err) {
      setError(err as ApiError);
    }
  };

  return (
    <section>
      <h2>{t("scene.splits.title")}</h2>
      {splits.length === 0 ? (
        <p className="muted">{t("scene.splits.none")}</p>
      ) : (
        <ul>
          {splits.map((id) => (
            <li key={id}>
              <strong>{id}</strong>{" "}
              <a className="small" style={{ marginLeft: 12 }} href={href({ page: "new", sceneId, splitId: id })}>
                {t("scene.splits.run")}
              </a>
            </li>
          ))}
        </ul>
      )}
      <div className="panel">
        <div className="form-row">
          <label className="field">
            {t("scene.splits.every")}
            <input type="number" min={1} value={every} onChange={(e) => setEvery(Number(e.target.value))} />
          </label>
          <fieldset className="field" style={{ border: "none", padding: 0, margin: 0 }}>
            <legend>{t("scene.splits.cameras")}</legend>
            {cameras.map((camera) => (
              <label key={camera} style={{ marginRight: 12, color: "var(--ink)" }}>
                <input
                  type="checkbox"
                  checked={chosen.includes(camera)}
                  onChange={(e) => setChosen(e.target.checked ? [...chosen, camera] : chosen.filter((c) => c !== camera))}
                />{" "}
                {camera}
              </label>
            ))}
          </fieldset>
          <button className="button primary" onClick={create} disabled={chosen.length === 0 || every < 1}>
            {t("scene.splits.create")}
          </button>
        </div>
        {created && <p>{t("scene.splits.created", { split: created })}</p>}
        <ErrorBox error={error} />
      </div>
    </section>
  );
}
