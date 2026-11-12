// Danh sách cảnh dạng thẻ có ảnh đại diện; nút "Thêm cảnh" mở ngăn kéo ingest.
import { Plus, Search } from "lucide-react";
import { useState } from "react";
import { useApi } from "../api/client";
import type { List, Manifest } from "../api/types";
import { IngestDrawer } from "../components/IngestDrawer";
import { SceneThumb } from "../components/SceneThumb";
import { Empty, ErrorBox, PageHead, Skeleton } from "../components/ui";
import { useI18n } from "../i18n";
import { useLive } from "../live";
import { href } from "../router";

export function SceneCard({ manifest }: { manifest: Manifest }) {
  const { t } = useI18n();
  return (
    <a className="card" href={href({ page: "scene", sceneId: manifest.scene_id })}>
      <div className="thumb">
        <SceneThumb sceneId={manifest.scene_id} />
        <span className="ds">{manifest.source.dataset}</span>
      </div>
      <div className="body">
        <div className="title">{manifest.scene_id}</div>
        <div className="meta">
          <span className="tag">{t("scenes.card.frames", { count: manifest.timeline.num_frames })}</span>
          <span className="tag">{t("scenes.card.cameras", { count: manifest.cameras.length })}</span>
          <span className={`tag${manifest.available.lidar ? " on" : ""}`}>{manifest.available.lidar ? "LiDAR" : t("scenes.card.no_lidar")}</span>
          {manifest.available.cuboids !== "none" && <span className="tag on">{t("scenes.card.cuboids", { kind: manifest.available.cuboids })}</span>}
        </div>
      </div>
    </a>
  );
}

export function ScenesPage() {
  const { t } = useI18n();
  const live = useLive();
  const { data, error } = useApi<List<Manifest>>(`/api/scenes?_=${live.tick}`);
  const [adding, setAdding] = useState(false);
  const [query, setQuery] = useState("");
  const items = (data?.items ?? []).filter((m) => `${m.scene_id} ${m.source.dataset}`.toLowerCase().includes(query.toLowerCase()));
  return (
    <div className="page">
      <PageHead
        title={t("scenes.page.title")}
        lede={t("scenes.page.lede")}
        actions={
          <>
            {data && data.total > 6 && (
              <label className="row" style={{ position: "relative" }}>
                <Search size={14} style={{ position: "absolute", left: 9 }} className="faint" />
                <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("scenes.filter")} style={{ paddingLeft: 28 }} />
              </label>
            )}
            <button className="btn primary" onClick={() => setAdding(true)}>
              <Plus size={16} />
              {t("scenes.action.add")}
            </button>
          </>
        }
      />
      <ErrorBox error={error} />
      {!data && !error && (
        <div className="cards">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} height={210} />
          ))}
        </div>
      )}
      {data && data.total === 0 && (
        <div className="panel">
          <Empty
            title={t("scenes.empty.title")}
            action={
              <button className="btn primary" onClick={() => setAdding(true)}>
                <Plus size={16} />
                {t("scenes.action.add")}
              </button>
            }
          >
            <p>{t("scenes.empty.text")}</p>
          </Empty>
        </div>
      )}
      {items.length > 0 && (
        <div className="cards">
          {items.map((m) => (
            <SceneCard key={m.scene_id} manifest={m} />
          ))}
        </div>
      )}
      {adding && <IngestDrawer onClose={() => setAdding(false)} />}
    </div>
  );
}
