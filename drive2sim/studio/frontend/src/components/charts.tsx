// Hai biểu đồ SVG nhỏ, không cần thư viện: đường metric theo bước train, và chấm so sánh PSNR theo
// thời gian train hoặc VRAM giữa các run. Màu chấm theo approach, chấm tốt nhất có viền vàng.
export interface Point {
  x: number;
  y: number;
  /** Định danh đầy đủ, hiện khi rê chuột. */
  label?: string;
  /** Tên ngắn ghi cạnh chấm, ví dụ approach. */
  name?: string;
}

const W = 560;
const H = 240;
const PAD = { left: 46, right: 16, top: 14, bottom: 32 };
// Màu phân loại cho approach: phân biệt được ở cả nền sáng và tối, không trùng màu trạng thái.
const SERIES = ["#5ca8ff", "#c38cff", "#2fc6c6", "#ff8fb1", "#a3c95a", "#f0a35e"];

function niceTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + i * step);
}

function scale(values: number[], margin = 0): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * margin;
  return [min - pad, max + pad];
}

function project(points: Point[], margin = 0) {
  const xRange = scale(points.map((p) => p.x), margin);
  const yRange = scale(points.map((p) => p.y), margin);
  const px = (x: number) => PAD.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * (W - PAD.left - PAD.right);
  const py = (y: number) => H - PAD.bottom - ((y - yRange[0]) / (yRange[1] - yRange[0])) * (H - PAD.top - PAD.bottom);
  return { xRange, yRange, px, py };
}

function Axes({ xRange, yRange, xLabel, px, py }: { xRange: [number, number]; yRange: [number, number]; xLabel: string; px: (x: number) => number; py: (y: number) => number }) {
  const yTicks = niceTicks(yRange[0], yRange[1]);
  const xTicks = niceTicks(xRange[0], xRange[1]);
  const digits = (range: [number, number]) => (range[1] - range[0] < 5 ? 2 : range[1] - range[0] < 50 ? 1 : 0);
  return (
    <g>
      {yTicks.map((y) => (
        <g key={`y${y}`}>
          <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={py(y)} y2={py(y)} />
          <text x={PAD.left - 8} y={py(y) + 4} textAnchor="end">
            {y.toFixed(digits(yRange))}
          </text>
        </g>
      ))}
      {xTicks.map((x, i) => (
        <text key={`x${x}`} x={px(x)} y={H - 12} textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}>
          {x.toFixed(digits(xRange))}
          {i === xTicks.length - 1 ? ` ${xLabel}` : ""}
        </text>
      ))}
      <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
    </g>
  );
}

export function LineChart({ points, label, xLabel }: { points: Point[]; label: string; xLabel: string }) {
  const { xRange, yRange, px, py } = project(points);
  const line = points.map((p) => `${px(p.x)},${py(p.y)}`).join(" ");
  const area = `${px(points[0].x)},${H - PAD.bottom} ${line} ${px(points[points.length - 1].x)},${H - PAD.bottom}`;
  const last = points[points.length - 1];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} px={px} py={py} />
      <polygon className="area" points={area} />
      <polyline points={line} />
      <circle cx={px(last.x)} cy={py(last.y)} r={3.5} fill="var(--info)" />
    </svg>
  );
}

export function ScatterChart({ points, label, xLabel }: { points: Point[]; label: string; xLabel: string }) {
  // Chừa 15% lề để chấm nhỏ nhất và lớn nhất không nằm sát trục.
  const { xRange, yRange, px, py } = project(points, 0.15);
  const best = Math.max(...points.map((p) => p.y));
  const names = [...new Set(points.map((p) => p.name ?? ""))];
  const color = (name = "") => SERIES[names.indexOf(name) % SERIES.length];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} px={px} py={py} />
      {points.map((p) => (
        <g key={p.label}>
          {p.y === best && <circle cx={px(p.x)} cy={py(p.y)} r={10} fill="none" stroke="var(--accent)" strokeWidth={2} />}
          <circle className="dot" cx={px(p.x)} cy={py(p.y)} r={6} fill={color(p.name)}>
            <title>{`${p.label}: ${p.y.toFixed(2)}`}</title>
          </circle>
          {p.name && (
            <text className="dot-label" x={px(p.x) + 12} y={py(p.y) + 4}>
              {p.name}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
