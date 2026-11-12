// Danh sách mọi run, mới nhất trước.
import { useApi } from "../api/client";
import type { List, RunRecord } from "../api/types";
import { ErrorBox, Loading, optionsText, Status } from "../components/common";
import { useI18n } from "../i18n";
import { go, href } from "../router";

export function RunsPage() {
  const { t } = useI18n();
  const { data, error } = useApi<List<RunRecord>>("/api/runs");
  return (
    <>
      <h1>{t("runs.page.title")}</h1>
      <p className="lede">{t("runs.page.lede")}</p>
      <ErrorBox error={error} />
      {!data && !error && <Loading />}
      {data && data.total === 0 && (
        <div className="empty">
          <p>{t("runs.empty.text")}</p>
          <a className="button primary" href={href({ page: "new" })} style={{ textDecoration: "none" }}>
            {t("runs.empty.action")}
          </a>
        </div>
      )}
      {data && data.total > 0 && (
        <table>
          <thead>
            <tr>
              <th>{t("runs.table.run")}</th>
              <th>{t("runs.table.scene")}</th>
              <th>{t("runs.table.split")}</th>
              <th>{t("runs.table.approach")}</th>
              <th>{t("runs.table.options")}</th>
              <th>{t("runs.table.status")}</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((run) => (
              <tr key={run.run_id} className="link-row" onClick={() => go({ page: "run", runId: run.run_id })}>
                <td>
                  <a href={href({ page: "run", runId: run.run_id })}>{run.run_id}</a>
                </td>
                <td>{run.scene_id}</td>
                <td>{run.split_id}</td>
                <td>{run.approach.approach_id}</td>
                <td className="small">{optionsText(run.approach.options ?? {})}</td>
                <td>
                  <Status value={run.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
