// Lịch densify hiệu lực của một run (API ``densify``, drive2sim/engine/densify_info.py): densify bắt
// đầu, lặp lại, dừng ở bước nào, ngưỡng bao nhiêu. Ô cấu hình chỉ ghi lựa chọn của approach nên
// không cho thấy 3DGEER densify mỗi 50 bước với ngưỡng 0,01 còn 3DGS mỗi 100 bước với 0,0005.
// Dòng ngắn ở phần thông tin của run; chi tiết từng node nằm trong icon giải thích.
import type { DensifyInfo, DensifyKey } from "../api/types";
import { useI18n } from "../i18n";
import { InfoTip } from "./ui";

const NODES = ["Background", "RigidNodes", "DeformableNodes"] as const;
const KEYS: DensifyKey[] = [
  "warmup_steps",
  "refine_interval",
  "densify_grad_thresh",
  "densify_size_thresh",
  "cull_alpha_thresh",
  "reset_alpha_interval",
  "stop_split_at",
];
const MCMC_KEYS = ["refine_start_iter", "refine_stop_iter", "refine_every", "grow_rate", "min_opacity", "noise_lr", "opacity_reg", "scale_reg"] as const;

function useFormat() {
  const { locale } = useI18n();
  const format = new Intl.NumberFormat(locale === "vi" ? "vi-VN" : "en-US", { maximumSignificantDigits: 6 });
  return (value: unknown) => (typeof value === "number" ? format.format(value) : String(value ?? "–"));
}

/** Giá trị của một khoá theo node: một giá trị nếu mọi node giống nhau, không thì liệt kê theo node. */
function perNode(info: DensifyInfo, key: DensifyKey, fmt: (v: unknown) => string, nodeLabel: (n: string) => string): string {
  const entries = NODES.filter((node) => info.nodes[node]?.[key] !== undefined).map((node) => [node, info.nodes[node]![key]] as const);
  if (!entries.length) return "–";
  if (entries.every(([, value]) => value === entries[0][1])) return fmt(entries[0][1]);
  return entries.map(([node, value]) => `${nodeLabel(node)} ${fmt(value)}`).join(" · ");
}

/** Dạng ngắn cho dòng tóm tắt: giá trị của nền, node khác giá trị đó ghi trong ngoặc, gom theo giá trị
 * ("15.000 (xe, vật biến dạng 20.000)"). */
function perNodeShort(info: DensifyInfo, key: DensifyKey, fmt: (v: unknown) => string, nodeLabel: (n: string) => string): string {
  const entries = NODES.filter((node) => info.nodes[node]?.[key] !== undefined).map((node) => [node, info.nodes[node]![key]] as const);
  if (!entries.length) return "–";
  const main = entries[0][1];
  const others = new Map<unknown, string[]>();
  for (const [node, value] of entries.slice(1)) {
    if (value !== main) others.set(value, [...(others.get(value) ?? []), nodeLabel(node)]);
  }
  if (!others.size) return fmt(main);
  const rest = [...others].map(([value, nodes]) => `${nodes.join(", ")} ${fmt(value)}`).join("; ");
  return `${fmt(main)} (${rest})`;
}

export function DensifyText({ info }: { info: DensifyInfo | null | undefined }) {
  const { t } = useI18n();
  const fmt = useFormat();
  if (!info) return <>–</>;
  const nodeLabel = (node: string) => t(`densify.node.${node}`);
  const statistic = t(`densify.statistic.${info.statistic}`);
  const bg = info.nodes.Background ?? info.nodes[Object.keys(info.nodes)[0]] ?? {};
  let short: string;
  if (info.rule === "mcmc" && info.mcmc) {
    const caps = Object.entries(info.mcmc.caps).map(([node, cap]) => `${nodeLabel(node)} ${fmt(cap)}`).join(", ");
    short = t("densify.short.mcmc", {
      start: fmt(info.mcmc.refine_start_iter),
      stop: fmt(info.mcmc.refine_stop_iter),
      every: fmt(info.mcmc.refine_every),
      caps,
    });
  } else {
    short = t("densify.short.default", {
      rule: t(`densify.rule.${info.rule}`),
      start: fmt(bg.warmup_steps),
      every: perNodeShort(info, "refine_interval", fmt, nodeLabel),
      thresh: perNodeShort(info, "densify_grad_thresh", fmt, nodeLabel),
      statistic,
      stop: perNodeShort(info, "stop_split_at", fmt, nodeLabel),
    });
  }
  return (
    <span className="row" style={{ gap: 2 }}>
      {short}
      <InfoTip>
        <span className="stack" style={{ gap: 8 }}>
          <span>{t(`densify.rule_help.${info.rule}`)}</span>
          {info.rule === "mcmc" && info.mcmc ? (
            <span className="machine">
              <span>{t("densify.mcmc.classes", { nodes: info.mcmc.classes.map(nodeLabel).join(", ") || "–" })}</span>
              {MCMC_KEYS.map((key) => (
                <span key={key}>
                  {t(`densify.mcmc.${key}`)}: {fmt(info.mcmc![key])}
                </span>
              ))}
            </span>
          ) : null}
          {(info.rule !== "mcmc" || (info.mcmc && info.mcmc.classes.length < Object.keys(info.nodes).length)) && (
            <span className="machine">
              <span>{t("densify.statistic_line", { statistic })}</span>
              {KEYS.map((key) => (
                <span key={key}>
                  {t(`densify.key.${key}`)}: {perNode(info, key, fmt, nodeLabel)}
                </span>
              ))}
            </span>
          )}
          {info.rule === "fastgs" && info.fastgs && (
            <span className="muted">
              FastGS: {Object.entries(info.fastgs).map(([key, value]) => `${key}=${fmt(value)}`).join(", ")}
            </span>
          )}
          <span className="muted">{t("densify.source")}</span>
        </span>
      </InfoTip>
    </span>
  );
}
