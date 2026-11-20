// Thông tin của một run: cấu hình và kết quả đo. Dùng ở tab "Thông tin" của trang xem 3D; trang run
// dùng chung phần mô tả máy chạy.
import { ExternalLink } from "lucide-react";
import { seg, useApi } from "../api/client";
import type { EnvRecord, List, ReportRow, RunDetail } from "../api/types";
import { BetterArrow, visibleGroups } from "../metrics";
import { useI18n } from "../i18n";
import { href } from "../router";
import { ConfigText } from "./ConfigText";
import { DensifyText } from "./DensifyInfo";
import { SpeedupText } from "./SpeedupText";
import { ago, ErrorBox, fixed, InfoTip, Loading } from "./ui";

/** GPU của các máy đã chạy run, ví dụ "Tesla T4 · 15 GB"; "–" khi run chưa chạy. */
export function machineText(env: EnvRecord | null, noGpu: string): string {
  const machines = env?.machines ?? [];
  if (!machines.length) return "–";
  const names = [...new Set(machines.map((m) => (m.gpu ? `${(m.gpu_count ?? 0) > 1 ? `${m.gpu_count}× ` : ""}${m.gpu}` : noGpu)))];
  return names.join(" + ");
}

/** Nội dung icon giải thích: cấu hình từng máy đã chạy run và phiên bản gói. */
export function MachineInfo({ env }: { env: EnvRecord }) {
  const { t, locale } = useI18n();
  const machines = env.machines ?? [];
  return (
    <span className="stack" style={{ gap: 10 }}>
      {machines.length > 1 && <span>{t("run.machine.several", { count: machines.length })}</span>}
      {machines.map((m, index) => (
        <span key={index} className="machine">
          {machines.length > 1 && <b>{t("run.machine.since", { when: ago(m.first_seen_utc ?? null, locale) })}</b>}
          <span>GPU: {m.gpu ? `${(m.gpu_count ?? 0) > 1 ? `${m.gpu_count}× ` : ""}${m.gpu}${m.gpu_memory_mb ? `, ${(m.gpu_memory_mb / 1024).toFixed(0)} GB` : ""}` : t("run.facts.no_gpu")}</span>
          {m.gpu_driver && <span>{t("run.machine.driver")}: {m.gpu_driver}</span>}
          <span>CPU: {m.cpu ?? "–"}{m.cpu_cores ? `, ${t("run.machine.threads", { count: m.cpu_cores })}` : ""}</span>
          <span>RAM: {m.ram_gb != null ? `${m.ram_gb} GB` : "–"}</span>
          {m.os && <span>{t("run.machine.os")}: {m.os}</span>}
        </span>
      ))}
      <span className="muted">
        Python {env.python}, PyTorch {env.pytorch ?? "–"}, CUDA {env.cuda ?? "–"}, gsplat {env.gsplat ?? "–"}
      </span>
    </span>
  );
}

/** Cấu hình và kết quả đo của run, xếp dọc cho bảng hẹp (tab "Thông tin" của trang xem 3D). */
export function RunInfo({ runId }: { runId: string }) {
  const { t } = useI18n();
  const base = `/api/runs/${seg(runId)}`;
  const run = useApi<RunDetail>(base);
  // Cùng nguồn số liệu với trang run và trang so sánh.
  const report = useApi<List<ReportRow>>(`/api/report?run_id=${encodeURIComponent(runId)}`);
  if (run.error) return <ErrorBox error={run.error} />;
  if (!run.data) return <Loading rows={4} />;
  const data = run.data;
  const summary = data.summary;
  const row = report.data?.items[0];
  const perCamera = (summary?.per_camera ?? {}) as Record<string, { psnr: number; ssim: number; lpips: number }>;
  return (
    <div className="run-info">
      <section>
        <h4>{t("viewer.info.config")}</h4>
        <dl className="info-list stacked">
          <dt>{t("run.facts.scene")}</dt>
          <dd>{data.scene_id}</dd>
          <dt>{t("run.facts.split")}</dt>
          <dd>{data.split_id}</dd>
          <dt>{t("run.facts.approach")}</dt>
          <dd>{`${data.approach.approach_id} ${data.approach.version}`}</dd>
          <dt>{t("run.facts.options")}</dt>
          <dd>
            <ConfigText approachId={data.approach.approach_id} options={data.approach.options ?? {}} preset={data.approach.preset} params={data.train_params} />
            {summary?.speedups && summary.speedups.length > 0 && (
              <div>
                <SpeedupText speedups={summary.speedups} />
              </div>
            )}
          </dd>
          <dt>{t("run.facts.densify")}</dt>
          <dd>
            <DensifyText info={data.densify} />
          </dd>
          <dt>{t("run.facts.machine")}</dt>
          <dd className="row" style={{ gap: 2 }}>
            {machineText(data.env, t("run.facts.no_gpu"))}
            {(data.env?.machines ?? []).length > 0 && data.env && (
              <InfoTip>
                <MachineInfo env={data.env} />
              </InfoTip>
            )}
          </dd>
        </dl>
      </section>

      <section>
        <h4>{t("viewer.info.results")}</h4>
        <ErrorBox error={report.error} />
        {!summary ? (
          <p className="small muted">{t("run.result.no_eval")}</p>
        ) : (
          row &&
          visibleGroups([row]).map((group) => (
            <div key={group.id} className="info-group">
              <span className="small muted row" style={{ gap: 2 }}>
                {t(group.label)}
                <InfoTip>{t(group.hint)}</InfoTip>
              </span>
              <dl className="info-list">
                {group.metrics.map((metric) => {
                  const value = metric.format(row);
                  return (
                    <div key={metric.key} className="info-pair">
                      <dt className="row" style={{ gap: 2 }}>
                        {t(metric.label)}
                        <BetterArrow better={metric.better} />
                      </dt>
                      <dd className="num">
                        {value}
                        {metric.unit && value !== "–" && <small> {metric.unit}</small>}
                      </dd>
                    </div>
                  );
                })}
              </dl>
            </div>
          ))
        )}
        {Object.keys(perCamera).length > 0 && (
          <div className="table-wrap">
            <table className="dense">
              <thead>
                <tr>
                  <th>{t("common.camera")}</th>
                  <th className="num">PSNR</th>
                  <th className="num">SSIM</th>
                  <th className="num">LPIPS</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perCamera).map(([camera, m]) => (
                  <tr key={camera}>
                    <td>{camera}</td>
                    <td className="num">{fixed(m.psnr)}</td>
                    <td className="num">{fixed(m.ssim, 3)}</td>
                    <td className="num">{fixed(m.lpips, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <a className="btn sm" href={href({ page: "run", runId })}>
        <ExternalLink size={15} />
        {t("viewer.info.open_run")}
      </a>
    </div>
  );
}
