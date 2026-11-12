// Hai biểu đồ SVG nhỏ, không cần thư viện: đường metric theo bước train, và chấm so sánh PSNR theo
// thời gian train hoặc VRAM giữa các run.
export interface Point {
  x: number;
  y: number;
  /** Định danh đầy đủ, hiện khi rê chuột. */
  label?: string;
  /** Tên ngắn ghi cạnh chấm, ví dụ approach. */
  name?: string;
}

const W = 520;
const H = 220;
const PAD = { left: 44, right: 12, top: 12, bottom: 30 };

function scale(values: number[], margin = 0): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * margin;
  return [min - pad, max + pad];
}

function Axes({ xRange, yRange, xLabel }: { xRange: [number, number]; yRange: [number, number]; xLabel: string }) {
  return (
    <g>
      <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
      <line className="axis" x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} />
      <text x={PAD.left - 6} y={PAD.top + 10} textAnchor="end">{yRange[1].toFixed(1)}</text>
      <text x={PAD.left - 6} y={H - PAD.bottom} textAnchor="end">{yRange[0].toFixed(1)}</text>
      <text x={PAD.left} y={H - 8}>{xRange[0].toFixed(0)}</text>
      <text x={W - PAD.right} y={H - 8} textAnchor="end">{`${xRange[1].toFixed(0)} ${xLabel}`}</text>
    </g>
  );
}

function project(points: Point[], margin = 0) {
  const xRange = scale(points.map((p) => p.x), margin);
  const yRange = scale(points.map((p) => p.y), margin);
  const px = (x: number) => PAD.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * (W - PAD.left - PAD.right);
  const py = (y: number) => H - PAD.bottom - ((y - yRange[0]) / (yRange[1] - yRange[0])) * (H - PAD.top - PAD.bottom);
  return { xRange, yRange, px, py };
}

export function LineChart({ points, label, xLabel }: { points: Point[]; label: string; xLabel: string }) {
  const { xRange, yRange, px, py } = project(points);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} />
      <polyline points={points.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")} />
    </svg>
  );
}

export function ScatterChart({ points, label, xLabel }: { points: Point[]; label: string; xLabel: string }) {
  // Chừa 15% lề để chấm nhỏ nhất và lớn nhất không nằm sát trục.
  const { xRange, yRange, px, py } = project(points, 0.15);
  const best = Math.max(...points.map((p) => p.y));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} />
      {points.map((p) => (
        <g key={p.label}>
          <circle className={`dot ${p.y === best ? "best" : ""}`} cx={px(p.x)} cy={py(p.y)} r={5}>
            <title>{`${p.label}: ${p.y.toFixed(2)}`}</title>
          </circle>
          {p.name && (
            <text className="dot-label" x={px(p.x) + 8} y={py(p.y) + 4}>
              {p.name}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
