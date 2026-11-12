// Nhóm A: danh sách các cảnh đã chuyển sang dữ liệu chuẩn.
import { useApi } from "../api/client";
import type { List, Manifest } from "../api/types";
import { ErrorBox, Loading, yesNo } from "../components/common";
import { useI18n } from "../i18n";
import { go } from "../router";

export function ScenesPage() {
  const { t } = useI18n();
  const { data, error } = useApi<List<Manifest>>("/api/scenes");
  return (
    <>
      <h1>{t("scenes.page.title")}</h1>
      <p className="lede">{t("scenes.page.lede")}</p>
      <ErrorBox error={error} />
      {!data && !error && <Loading />}
      {data && data.total === 0 && (
        <div className="empty">
          <p>{t("scenes.empty.text")}</p>
          <code>d2s scene ingest &lt;thư mục dữ liệu gốc&gt; --sequence &lt;cảnh&gt;</code>
        </div>
      )}
      {data && data.total > 0 && (
        <table>
          <thead>
            <tr>
              <th>{t("scenes.table.scene")}</th>
              <th>{t("scenes.table.dataset")}</th>
              <th className="num">{t("scenes.table.frames")}</th>
              <th className="num">{t("scenes.table.cameras")}</th>
              <th>{t("scenes.table.lidar")}</th>
              <th>{t("scenes.table.cuboids")}</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((m) => (
              <tr key={m.scene_id} className="link-row" onClick={() => go({ page: "scene", sceneId: m.scene_id })}>
                <td>
                  <a href={`#/scenes/${encodeURIComponent(m.scene_id)}`}>{m.scene_id}</a>
                </td>
                <td>{m.source.dataset}</td>
                <td className="num">{m.timeline.num_frames}</td>
                <td className="num">{m.cameras.length}</td>
                <td>{yesNo(m.available.lidar, t)}</td>
                <td>{m.available.cuboids}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
