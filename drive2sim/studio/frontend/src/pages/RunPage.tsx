// Nhóm B và C của một run: làn đường stage cập nhật trực tiếp qua SSE, việc đang làm, metric trong
// lúc train, bảng stage, kết quả trên tập test, video và log.
import { useEffect, useMemo, useState } from "react";
import { ApiError, post, useApi } from "../api/client";
import type { RunDetail, RunEvent, RunLog, Video, List } from "../api/types";
import { LineChart, type Point } from "../components/charts";
import { ErrorBox, fixed, Loading, optionsText, Status } from "../components/common";
import { StageRoad } from "../components/StageRoad";
import { useI18n } from "../i18n";
import { href } from "../router";

const FINISHED = ["succeeded", "failed", "cancelled"];

/** Theo dõi sự kiện SSE của một run; gọi lại onDone khi run kết thúc. */
function useRunEvents(runId: string, onDone: () => void) {
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [message, setMessage] = useState<string>("");
  const [metrics, setMetrics] = useState<Record<string, Point[]>>({});

  useEffect(() => {
    const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    const read = (handler: (event: RunEvent) => void) => (raw: MessageEvent) =>
      handler({ ...JSON.parse(raw.data), event: raw.type } as RunEvent);
    source.addEventListener(
      "status",
      read((e) => setStatuses((s) => ({ ...s, [e.stage_id ?? "__run__"]: e.status ?? "" }))),
    );
    source.addEventListener(
      "progress",
      read((e) => setMessage(`${e.stage_id}: ${Math.round((e.fraction ?? 0) * 100)}% ${e.message ?? ""}`)),
    );
    source.addEventListener(
      "metric",
      read((e) =>
        setMetrics((m) => ({ ...m, [e.name ?? "?"]: [...(m[e.name ?? "?"] ?? []), { x: e.step ?? 0, y: e.value ?? 0 }] })),
      ),
    );
    source.addEventListener(
      "done",
      read(() => {
        source.close();
        onDone();
      }),
    );
    return () => source.close();
  }, [runId]); // onDone đổi mỗi lần render; chỉ mở lại kết nối khi đổi run

  return { statuses, message, metrics };
}

export function RunPage({ runId }: { runId: string }) {
  const { t } = useI18n();
  const run = useApi<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
  const log = useApi<RunLog>(`/api/runs/${encodeURIComponent(runId)}/log`);
  const videos = useApi<List<Video>>(`/api/runs/${encodeURIComponent(runId)}/videos`);
  const [resumeError, setResumeError] = useState<ApiError | null>(null);
  const refresh = () => {
    run.reload();
    log.reload();
    videos.reload();
  };
  const live = useRunEvents(runId, refresh);

  const stages = useMemo(
    () =>
      (run.data?.stages ?? []).map((s) => ({
        ...s,
        status: live.statuses[s.stage_id] ?? s.status,
      })),
    [run.data, live.statuses],
  );

  if (run.error) return <ErrorBox error={run.error} />;
  if (!run.data) return <Loading />;
  const data = run.data;
  const status = live.statuses.__run__ ?? data.status;

  const resume = async () => {
    setResumeError(null);
    try {
      await post(`/api/runs/${encodeURIComponent(runId)}/resume`, {});
      window.location.reload();
    } catch (err) {
      setResumeError(err as ApiError);
    }
  };

  return (
    <>
      <h1>{data.run_id}</h1>
      <p>
        <Status value={status} />
      </p>
      <dl className="facts">
        <div>
          <dt>{t("run.facts.scene")}</dt>
          <dd>
            <a href={href({ page: "scene", sceneId: data.scene_id })}>{data.scene_id}</a>
          </dd>
        </div>
        <div>
          <dt>{t("run.facts.split")}</dt>
          <dd>{data.split_id}</dd>
        </div>
        <div>
          <dt>{t("run.facts.approach")}</dt>
          <dd>{data.approach.approach_id}</dd>
        </div>
        <div>
          <dt>{t("run.facts.options")}</dt>
          <dd>{optionsText(data.approach.options ?? {}) || "–"}</dd>
        </div>
        <div>
          <dt>{t("run.facts.executor")}</dt>
          <dd>{data.executor}</dd>
        </div>
      </dl>

      <StageRoad
        stages={stages.map((s) => ({
          stage_id: s.stage_id,
          status: s.status,
          cached: s.cache_hit,
          note: s.cache_hit && s.status === "succeeded" ? "cache" : undefined,
        }))}
      />
      {!FINISHED.includes(status) && live.message && <p className="muted">{t("run.progress.now", { message: live.message })}</p>}
      {stages
        .filter((s) => s.error)
        .map((s) => (
          <div className="error" role="alert" key={s.stage_id}>
            <strong>{s.error?.message}</strong>
            <div className="detail">{t("run.error.where", { log: String(s.error?.details?.log ?? "d2s/log.txt") })}</div>
          </div>
        ))}
      {(status === "failed" || status === "cancelled") && (
        <button className="button" onClick={resume}>
          {t("run.action.resume")}
        </button>
      )}
      <ErrorBox error={resumeError} />

      <h2>{t("run.result.title")}</h2>
      {!data.summary && <p className="muted">{t("run.result.none")}</p>}
      {data.summary && (
        <>
          <table style={{ maxWidth: 560 }}>
            <thead>
              <tr>
                <th />
                <th className="num">PSNR</th>
                <th className="num">SSIM</th>
                <th className="num">LPIPS</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{t("run.result.overall")}</td>
                <td className="num">{fixed(data.summary.overall.psnr)}</td>
                <td className="num">{fixed(data.summary.overall.ssim, 3)}</td>
                <td className="num">{fixed(data.summary.overall.lpips, 3)}</td>
              </tr>
              {data.summary.overall.human_psnr != null && (
                <tr>
                  <td>{t("run.result.human")}</td>
                  <td className="num">{fixed(data.summary.overall.human_psnr)}</td>
                  <td />
                  <td />
                </tr>
              )}
              {data.summary.overall.vehicle_psnr != null && (
                <tr>
                  <td>{t("run.result.vehicle")}</td>
                  <td className="num">{fixed(data.summary.overall.vehicle_psnr)}</td>
                  <td />
                  <td />
                </tr>
              )}
            </tbody>
          </table>
          <p className="muted small">
            {t("run.result.protocol", {
              net: data.summary.protocol.lpips.net,
              package: data.summary.protocol.lpips.package,
              version: data.summary.protocol.lpips.package_version,
              resolution: data.summary.protocol.resolution,
            })}
          </p>
        </>
      )}

      <h2>{t("run.metrics.title")}</h2>
      {Object.keys(live.metrics).length === 0 ? (
        <p className="muted">{t("run.metrics.none")}</p>
      ) : (
        <div className="cols">
          {Object.entries(live.metrics).map(([name, points]) => (
            <figure key={name} style={{ margin: 0 }}>
              <figcaption>{name}</figcaption>
              <LineChart points={points} label={name} xLabel="step" />
            </figure>
          ))}
        </div>
      )}

      <h2>{t("run.stages.title")}</h2>
      <table>
        <thead>
          <tr>
            <th>{t("run.stages.stage")}</th>
            <th>{t("run.stages.status")}</th>
            <th>{t("run.stages.cache")}</th>
            <th className="num">{t("run.stages.seconds")}</th>
            <th className="num">{t("run.stages.vram")}</th>
            <th>{t("run.stages.error")}</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((s) => (
            <tr key={s.stage_id}>
              <td>{s.stage_id}</td>
              <td>
                <Status value={s.status} />
              </td>
              <td>{s.cache_hit ? "✓" : ""}</td>
              <td className="num">{s.duration_s ? s.duration_s.toFixed(1) : "–"}</td>
              <td className="num">{s.peak_vram_mb || "–"}</td>
              <td className="small">{s.error?.message ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>{t("run.videos.title")}</h2>
      {(videos.data?.items ?? []).length === 0 && <p className="muted">{t("run.videos.none")}</p>}
      <div className="videos">
        {(videos.data?.items ?? []).map((video) => (
          <figure key={video.url} style={{ margin: 0 }}>
            <video src={video.url} controls preload="metadata" />
            <figcaption className="small muted">{`${video.stage_id}/${video.name}`}</figcaption>
          </figure>
        ))}
      </div>

      <h2>{t("run.logs.title")}</h2>
      <details open={status === "failed"}>
        <summary>{t("run.logs.engine")}</summary>
        <pre className="log">{log.data?.engine || t("run.logs.empty")}</pre>
      </details>
      {Object.entries(log.data?.stages ?? {}).map(([stageId, text]) => (
        <details key={stageId}>
          <summary>{stageId}</summary>
          <pre className="log">{text}</pre>
        </details>
      ))}
    </>
  );
}
