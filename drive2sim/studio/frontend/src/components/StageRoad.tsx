// Danh sách stage của một run vẽ thành một làn đường: mỗi stage một đoạn, vạch đứt giữa các
// đoạn; đoạn đang chạy tô vàng như vạch kẻ đường. Dùng ở màn hình chạy mới (xem trước) và màn hình
// run (trạng thái thật).
import { useI18n } from "../i18n";

export interface RoadStage {
  stage_id: string;
  status: string;
  cached?: boolean;
  note?: string;
}

export function StageRoad({ stages }: { stages: RoadStage[] }) {
  const { t } = useI18n();
  return (
    <ol className="road" aria-label={t("run.stages.title")} style={{ listStyle: "none", padding: 0 }}>
      {stages.map((stage) => (
        <li
          key={stage.stage_id}
          className={`segment ${stage.status} ${stage.cached ? "cached" : ""}`}
          title={stage.stage_id}
        >
          <div className="stage">{stage.stage_id}</div>
          <div className="note">{stage.note ?? t(`common.status.${stage.status}`)}</div>
        </li>
      ))}
    </ol>
  );
}
