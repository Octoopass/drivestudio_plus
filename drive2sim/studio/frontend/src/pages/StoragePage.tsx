// Lưu trữ: kết quả trong cache/ (mask, SMPL... dùng chung giữa các approach) kèm dung lượng và run
// nào còn dùng; nút dọn xem trước những gì sẽ xoá rồi mới xoá.
import { Eraser } from "lucide-react";
import { useState } from "react";
import { asApiError, post, useApi } from "../api/client";
import type { CacheEntry, CacheList } from "../api/types";
import { bytes, Confirm, Empty, ErrorBox, Loading, PageHead, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { href } from "../router";

interface GcResult {
  removed: CacheEntry[];
  freed_bytes: number;
}

export function StoragePage() {
  const { t } = useI18n();
  const toast = useToast();
  const cache = useApi<CacheList>("/api/cache");
  const [plan, setPlan] = useState<GcResult | null>(null);

  const preview = async () => {
    try {
      const result = await post<GcResult>("/api/cache/gc", { dry_run: true });
      if (result.removed.length === 0) toast({ kind: "ok", title: t("storage.gc.nothing") });
      else setPlan(result);
    } catch (err) {
      toast({ kind: "bad", title: t("storage.gc.title"), detail: asApiError(err).message });
    }
  };

  const unused = (cache.data?.items ?? []).filter((e) => (!e.partial && e.used_by.length === 0) || (e.partial && e.stale));
  return (
    <div className="page">
      <PageHead
        title={t("storage.page.title")}
        info={t("storage.page.lede")}
        actions={
          <button className="btn" onClick={preview} disabled={!cache.data?.total}>
            <Eraser size={19} />
            {t("storage.gc.action")}
          </button>
        }
      />
      <ErrorBox error={cache.error} />
      {cache.data && (
        <section className="panel">
          <div className="stat-row">
            <div className="stat">
              <div className="label">{t("storage.stat.entries")}</div>
              <div className="value">{cache.data.total}</div>
            </div>
            <div className="stat">
              <div className="label">{t("storage.stat.size")}</div>
              <div className="value">{bytes(cache.data.size_bytes)}</div>
            </div>
            <div className="stat">
              <div className="label">{t("storage.stat.unused")}</div>
              <div className="value">{bytes(unused.reduce((sum, e) => sum + e.size_bytes, 0))}</div>
            </div>
          </div>
        </section>
      )}
      <section className="panel">
        {!cache.data && !cache.error && <div className="panel-body"><Loading /></div>}
        {cache.data?.total === 0 && <Empty title={t("storage.empty.title")}><p>{t("storage.empty.text")}</p></Empty>}
        {(cache.data?.total ?? 0) > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("storage.table.stage")}</th>
                  <th>{t("storage.table.scene")}</th>
                  <th>{t("storage.table.key")}</th>
                  <th className="num">{t("storage.table.size")}</th>
                  <th>{t("storage.table.used_by")}</th>
                </tr>
              </thead>
              <tbody>
                {cache.data!.items.map((entry) => (
                  <tr key={entry.cache_key + entry.partial}>
                    <td className="id">{entry.stage_id}</td>
                    <td className="ellipsis" style={{ maxWidth: 276 }}>
                      <a href={href({ page: "scene", sceneId: entry.scene_id })}>{entry.scene_id}</a>
                    </td>
                    <td className="small muted">{entry.cache_key.split("/")[1]}</td>
                    <td className="num">{bytes(entry.size_bytes)}</td>
                    <td>
                      {entry.partial ? (
                        <span className={`tag ${entry.stale ? "stop" : "info"}`}>{t(entry.stale ? "storage.tag.stale" : "storage.tag.writing")}</span>
                      ) : entry.used_by.length === 0 ? (
                        <span className="tag warn">{t("storage.tag.unused")}</span>
                      ) : (
                        <span className="row wrap" style={{ gap: 6 }}>
                          {entry.used_by.slice(0, 3).map((runId) => (
                            <a key={runId} className="tag" href={href({ page: "run", runId })}>{runId}</a>
                          ))}
                          {entry.used_by.length > 3 && <span className="tag">+{entry.used_by.length - 3}</span>}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {plan && (
        <Confirm
          title={t("storage.gc.title")}
          body={t("storage.gc.body", { count: plan.removed.length, size: bytes(plan.freed_bytes) })}
          items={plan.removed.map((e) => e.cache_key)}
          confirmLabel={t("storage.gc.confirm")}
          danger
          onClose={() => setPlan(null)}
          onConfirm={async () => {
            const result = await post<GcResult>("/api/cache/gc", {});
            toast({ kind: "ok", title: t("storage.gc.done", { size: bytes(result.freed_bytes) }) });
            cache.reload();
          }}
        />
      )}
    </div>
  );
}
