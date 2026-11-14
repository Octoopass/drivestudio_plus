// Kỹ thuật tăng tốc thật sự đã bật khi train (summary.json, trường speedups), ví dụ
// "Đã chạy: FastGS · Faster-GS (Adam fused, SSIM fused)". Công tắc trong cấu hình chỉ nói bật hay
// tắt; dòng này nói công tắc đã thật sự làm gì, để hai run cùng bật Faster-GS mà khác kỹ thuật
// không bị coi là giống nhau (CONTRACTS mục 6.3).
import { useI18n } from "../i18n";

export function SpeedupText({ speedups }: { speedups?: string[] | null }) {
  const { t } = useI18n();
  if (!speedups?.length) return null;
  const groups = new Map<string, string[]>();
  for (const item of speedups) {
    const [algo, technique] = item.split(".", 2);
    const list = groups.get(algo) ?? [];
    if (technique) list.push(t(`speedup.${algo}.${technique}`));
    groups.set(algo, list);
  }
  const items = [...groups].map(([algo, techniques]) =>
    techniques.length ? `${t(`speedup.${algo}`)} (${techniques.join(", ")})` : t(`speedup.${algo}`),
  );
  return <span className="small faint">{t("speedup.ran", { items: items.join(" · ") })}</span>;
}
