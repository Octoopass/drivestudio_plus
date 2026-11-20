// Xuất một biểu đồ của trang so sánh thành hình cho báo cáo (SPEC mục 8D): SVG (giữ nét, chèn vào
// LaTeX, Word) hoặc PNG (2x).
//
// Hình xuất luôn theo giao diện sáng, dù người xem đang dùng giao diện tối: lúc xuất, trang tạm đặt
// data-theme="light" rồi đọc màu, cỡ chữ đã tính của từng phần tử và ghi thẳng vào bản sao SVG (hình
// rời không còn CSS của trang). Hình gồm tiêu đề, biểu đồ, chú giải, các cảnh báo (trộn split, khác
// GPU...) và dòng nguồn, để hình không gây hiểu nhầm khi tách khỏi trang.
import { Download } from "lucide-react";
import { type ReactNode, useRef } from "react";
import { useI18n } from "../i18n";

const PROPS = [
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-width",
  "stroke-dasharray",
  "stroke-opacity",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
  "font-family",
  "font-size",
  "font-weight",
  "text-anchor",
  "dominant-baseline",
  "display",
  "visibility",
];
// Chữ trong hình rời: font của trang có thể không có trên máy mở hình, nên kèm font dự phòng.
const FONT_FALLBACK = ", 'DejaVu Sans', Arial, Helvetica, sans-serif";
const NS = "http://www.w3.org/2000/svg";

export interface FigureInfo {
  title: string;
  notes: string[];
  source: string;
  legend: { name: string; color: string }[];
}

function escapeXml(text: string): string {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);
}

/** Chạy ``read`` khi trang đang ở giao diện sáng, rồi trả lại giao diện cũ (cùng một nhịp, không nháy). */
function withLightTheme<T>(read: () => T): T {
  const root = document.documentElement;
  const previous = root.dataset.theme;
  root.dataset.theme = "light";
  try {
    return read();
  } finally {
    if (previous === undefined) delete root.dataset.theme;
    else root.dataset.theme = previous;
  }
}

/** Bản sao của ``svg`` với màu, nét, chữ tính theo giao diện sáng ghi thẳng vào từng phần tử. */
function inlineClone(svg: SVGSVGElement, legendColors: string[]): { clone: SVGSVGElement; colors: Map<string, string> } {
  return withLightTheme(() => {
    const clone = svg.cloneNode(true) as SVGSVGElement;
    const source = [svg, ...svg.querySelectorAll("*")];
    const target = [clone, ...clone.querySelectorAll("*")];
    source.forEach((element, index) => {
      const style = getComputedStyle(element);
      const out = target[index] as SVGElement;
      out.removeAttribute("class");
      const declarations = PROPS.map((name) => {
        let value = style.getPropertyValue(name);
        if (name === "font-family" && value) value += FONT_FALLBACK;
        return value ? `${name}:${value}` : "";
      }).filter(Boolean);
      out.setAttribute("style", declarations.join(";"));
    });
    // Màu chú giải có thể là biến CSS: đổi sang màu thật của giao diện sáng.
    const probe = document.createElement("span");
    document.body.appendChild(probe);
    const colors = new Map<string, string>();
    const resolve = (color: string) => {
      probe.style.color = color;
      colors.set(color, getComputedStyle(probe).color);
    };
    legendColors.forEach(resolve);
    probe.remove();
    return { clone, colors };
  });
}

/** Dựng SVG rời: nền trắng, tiêu đề, biểu đồ, chú giải, cảnh báo, nguồn. */
export function buildFigure(svg: SVGSVGElement, info: FigureInfo): string {
  const { clone, colors } = inlineClone(
    svg,
    info.legend.map((l) => l.color),
  );
  const view = svg.viewBox.baseVal;
  const width = view && view.width ? view.width : svg.clientWidth;
  const height = view && view.height ? view.height : svg.clientHeight;
  const margin = 16;
  const font = "font-family:'Plus Jakarta Sans'" + FONT_FALLBACK;
  const line = 20;
  const titleH = 34;
  // Chú giải: xếp hàng ngang, xuống dòng khi tràn.
  const legend: string[] = [];
  let lx = margin;
  let ly = titleH + height + 22;
  for (const item of info.legend) {
    const w = 22 + item.name.length * 7.6 + 18;
    if (lx + w > width + 2 * margin - margin && lx > margin) {
      lx = margin;
      ly += line;
    }
    const color = colors.get(item.color) ?? item.color;
    legend.push(
      `<rect x="${lx}" y="${ly - 10}" width="14" height="10" rx="2" style="fill:${color}"/>` +
        `<text x="${lx + 20}" y="${ly}" style="${font};font-size:14px;fill:#222">${escapeXml(item.name)}</text>`,
    );
    lx += w;
  }
  // Cảnh báo, nguồn: xuống dòng theo bề rộng hình (ước lượng bề rộng chữ, không đo được ngoài trang).
  const perLine = Math.max(30, Math.floor(width / 7));
  const wrap = (text: string): string[] => {
    const lines: string[] = [];
    let current = "";
    for (const word of text.split(" ")) {
      if (current && (current + " " + word).length > perLine) {
        lines.push(current);
        current = word;
      } else current = current ? `${current} ${word}` : word;
    }
    if (current) lines.push(current);
    return lines;
  };
  let ny = (info.legend.length ? ly : titleH + height) + 26;
  const notes: string[] = [];
  for (const note of info.notes) {
    wrap(`⚠ ${note}`).forEach((part, index) => {
      notes.push(`<text x="${margin + (index ? 14 : 0)}" y="${ny}" style="${font};font-size:13px;fill:#9a5b00">${escapeXml(part)}</text>`);
      ny += 18;
    });
    ny += 4;
  }
  const sourceLines = wrap(info.source);
  const sourceY = ny + 4;
  const total = sourceY + 16 * (sourceLines.length - 1) + 14;
  const inner = new XMLSerializer().serializeToString(clone);
  const placed = inner.replace(/^<svg\b/, `<svg x="${margin}" y="${titleH}" width="${width}" height="${height}"`);
  return [
    `<svg xmlns="${NS}" width="${width + 2 * margin}" height="${total}" viewBox="0 0 ${width + 2 * margin} ${total}">`,
    `<rect width="100%" height="100%" fill="#ffffff"/>`,
    `<text x="${margin}" y="24" style="${font};font-size:17px;font-weight:600;fill:#111">${escapeXml(info.title)}</text>`,
    placed,
    ...legend,
    ...notes,
    ...sourceLines.map((part, index) => `<text x="${margin}" y="${sourceY + 16 * index}" style="${font};font-size:12px;fill:#666">${escapeXml(part)}</text>`),
    `</svg>`,
  ].join("");
}

function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function toPng(svgText: string, scale = 2): Promise<Blob> {
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("canvas 2d unavailable");
    context.scale(scale, scale);
    context.drawImage(image, 0, 0);
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob"))), "image/png"));
  } finally {
    URL.revokeObjectURL(url);
  }
}

function slug(text: string): string {
  return (
    text
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .replace(/đ/g, "d")
      .replace(/Đ/g, "D")
      .replace(/[^A-Za-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .toLowerCase() || "figure"
  );
}

/** Khung chứa một biểu đồ kèm hai nút xuất SVG, PNG. ``info`` mô tả nội dung hình xuất. */
export function Exportable({ info, children }: { info: FigureInfo; children: ReactNode }) {
  const { t } = useI18n();
  const box = useRef<HTMLDivElement>(null);
  const run = async (kind: "svg" | "png") => {
    const svg = box.current?.querySelector("svg");
    if (!svg) return;
    const text = buildFigure(svg, info);
    const name = `${slug(info.title)}.${kind}`;
    if (kind === "svg") download(new Blob([text], { type: "image/svg+xml" }), name);
    else download(await toPng(text), name);
  };
  return (
    <div className="exportable">
      <div ref={box}>{children}</div>
      <div className="export-row">
        <span className="small faint">{t("figure.export")}</span>
        <button type="button" className="btn sm" onClick={() => void run("svg")} title={t("figure.svg_hint")}>
          <Download size={14} />
          SVG
        </button>
        <button type="button" className="btn sm" onClick={() => void run("png")} title={t("figure.png_hint")}>
          <Download size={14} />
          PNG
        </button>
      </div>
    </div>
  );
}
