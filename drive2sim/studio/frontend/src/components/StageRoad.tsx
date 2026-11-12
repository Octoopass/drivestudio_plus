// Các stage của một run vẽ thành một làn đường: mỗi stage một đoạn, vạch đứt giữa các đoạn. Đoạn xong
// có vạch xanh dưới đáy, lỗi vạch đỏ, đang chạy có vạch vàng chạy dọc và thanh tiến độ. Dùng ở màn
// hình chạy mới (xem trước) và màn hình run (trạng thái thật).
import { Check, CircleDashed, Database, Loader2, X } from "lucide-react";
import { useI18n } from "../i18n";

export interface RoadStage {
  stage_id: string;
  status: string;
  cached?: boolean;
  note?: string;
  fraction?: number | null;
}

function StageIcon({ stage }: { stage: RoadStage }) {
  if (stage.cached) return <Database size={12} />;
  if (stage.status === "succeeded") return <Check size={12} />;
  if (stage.status === "failed") return <X size={12} />;
  if (stage.status === "running") return <Loader2 size={12} />;
  return <CircleDashed size={12} />;
}

export function StageRoad({ stages }: { stages: RoadStage[] }) {
  const { t } = useI18n();
  return (
    <ol className="road" aria-label={t("run.stages.title")}>
      {stages.map((stage) => (
        <li key={stage.stage_id} className={stage.status} title={stage.stage_id}>
          <div className="stage">{stage.stage_id}</div>
          <div className="note">
            <StageIcon stage={stage} />
            <span className="ellipsis">{stage.note ?? t(`common.status.${stage.status}`)}</span>
          </div>
          {stage.status === "running" && stage.fraction != null && (
            <span className="bar" style={{ width: `${Math.round(stage.fraction * 100)}%` }} />
          )}
        </li>
      ))}
    </ol>
  );
}
