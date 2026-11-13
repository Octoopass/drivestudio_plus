// Biểu đồ SVG nhỏ, không cần thư viện: đường metric theo bước train (một hoặc nhiều run), chấm so
// sánh hai đại lượng giữa các run (kèm đường Pareto: các run không bị run nào khác vượt cả hai
// mặt), cột nhóm và cột chồng. Màu theo approach hoặc theo chuỗi, cùng một bảng màu.
export interface Point {
  x: number;
  y: number;
  /** Định danh đầy đủ, hiện khi rê chuột. */
  label?: string;
  /** Tên ngắn ghi cạnh chấm, ví dụ approach. */
  name?: string;
  /** Nhóm để chọn màu (ví dụ approach); mặc định là `name`. */
  group?: string;
}

const W = 560;
const H = 240;
const PAD = { left: 62, right: 20, top: 18, bottom: 42 };
// Màu phân loại cho approach: phân biệt được ở cả nền sáng và tối, không trùng màu trạng thái.
const SERIES = ["#5ca8ff", "#c38cff", "#2fc6c6", "#ff8fb1", "#a3c95a", "#f0a35e"];

function niceTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + i * step);
}

function scale(values: number[], margin = 0): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  // Đại lượng không âm (thời gian, VRAM, số Gaussian) thì trục không xuống dưới 0.
  const floor = min >= 0 ? 0 : -Infinity;
  if (min === max) return [Math.max(floor, min - 1), max + 1];
  const pad = (max - min) * margin;
  return [Math.max(floor, min - pad), max + pad];
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
  // Số lớn (ví dụ số Gaussian) viết gọn để không tràn lề trái.
  const tick = (value: number, range: [number, number]) =>
    Math.abs(range[1]) >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : Math.abs(range[1]) >= 1e4 ? `${Math.round(value / 1e3)}k` : value.toFixed(digits(range));
  return (
    <g>
      {yTicks.map((y) => (
        <g key={`y${y}`}>
          <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={py(y)} y2={py(y)} />
          <text x={PAD.left - 8} y={py(y) + 6} textAnchor="end">
            {tick(y, yRange)}
          </text>
        </g>
      ))}
      {xTicks.map((x, i) => (
        <text key={`x${x}`} x={px(x)} y={H - 15} textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}>
          {tick(x, xRange)}
        </text>
      ))}
      <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
      {/* Đơn vị trục ngang: ngay trên trục, góc phải, để không đè lên số. */}
      <text className="axis-unit" x={W - PAD.right} y={H - PAD.bottom - 8} textAnchor="end">
        {xLabel}
      </text>
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

/** Các điểm không bị điểm nào khác vượt cả hai mặt (y luôn cao là tốt; x theo `xBetter`). */
export function paretoFront(points: Point[], xBetter: "low" | "high"): Point[] {
  const better = (a: number, b: number) => (xBetter === "low" ? a <= b : a >= b);
  return points
    .filter((p) => !points.some((q) => q !== p && better(q.x, p.x) && q.y >= p.y && (q.x !== p.x || q.y !== p.y)))
    .sort((a, b) => a.x - b.x);
}

export function ScatterChart({
  points,
  label,
  xLabel,
  xBetter,
  groups: groupOrder,
  frontByGroup,
}: {
  points: Point[];
  label: string;
  xLabel: string;
  /** Có thì nối các run trên đường Pareto và đánh dấu chúng; không có thì chỉ đánh dấu run tốt nhất. */
  xBetter?: "low" | "high";
  /** Thứ tự nhóm để chọn màu (khớp với chú thích); mặc định theo thứ tự xuất hiện. */
  groups?: string[];
  /** Mỗi nhóm một đường Pareto riêng (ví dụ mỗi split), thay vì một đường chung. */
  frontByGroup?: boolean;
}) {
  // Chừa 15% lề để chấm nhỏ nhất và lớn nhất không nằm sát trục.
  const { xRange, yRange, px, py } = project(points, 0.15);
  const best = Math.max(...points.map((p) => p.y));
  const groups = groupOrder ?? [...new Set(points.map((p) => p.group ?? p.name ?? ""))];
  const groupOf = (p: Point) => p.group ?? p.name ?? "";
  const fronts = !xBetter
    ? []
    : frontByGroup
      ? groups.map((g) => ({ group: g, points: paretoFront(points.filter((p) => groupOf(p) === g), xBetter) }))
      : [{ group: "", points: paretoFront(points, xBetter) }];
  const marked = (p: Point) => (xBetter ? fronts.some((f) => f.points.includes(p)) : p.y === best);
  // Đẩy nhãn của các chấm gần nhau ra xa theo chiều dọc để không đè lên nhau.
  const labelY = new Map<Point, number>();
  const placed: { x: number; y: number }[] = [];
  for (const p of [...points].sort((a, b) => py(a.y) - py(b.y))) {
    let y = py(p.y) + 5;
    while (placed.some((q) => Math.abs(q.x - px(p.x)) < 170 && Math.abs(q.y - y) < 19)) y += 19;
    placed.push({ x: px(p.x), y });
    labelY.set(p, y);
  }
  const color = (p: Point) => SERIES[Math.max(0, groups.indexOf(groupOf(p))) % SERIES.length];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} px={px} py={py} />
      {fronts.map(
        (front) =>
          front.points.length > 1 && (
            <polyline
              key={front.group}
              className="pareto"
              style={frontByGroup ? { stroke: SERIES[Math.max(0, groups.indexOf(front.group)) % SERIES.length] } : undefined}
              points={front.points.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
            />
          ),
      )}
      {points.map((p) => (
        <g key={p.label}>
          {marked(p) && <circle cx={px(p.x)} cy={py(p.y)} r={10} fill="none" stroke="var(--accent)" strokeWidth={2} />}
          <circle className="dot" cx={px(p.x)} cy={py(p.y)} r={6} fill={color(p)}>
            <title>{`${p.label}: ${p.y.toFixed(2)}`}</title>
          </circle>
          {p.name && (
            // Chấm ở phần phải của biểu đồ thì ghi nhãn sang trái, để không bị cắt ở mép.
            <text className="dot-label" x={px(p.x) + (px(p.x) > W * 0.7 ? -14 : 14)} y={labelY.get(p) ?? py(p.y) + 5} textAnchor={px(p.x) > W * 0.7 ? "end" : "start"}>
              {p.name}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

export function seriesColor(index: number): string {
  return SERIES[index % SERIES.length];
}

/** Chú thích màu cho biểu đồ nhiều chuỗi. */
export function Legend({ items }: { items: { name: string; color: string; faded?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span key={item.name}>
          <i style={{ background: item.color, opacity: item.faded ? 0.45 : 1 }} />
          {item.name}
        </span>
      ))}
    </div>
  );
}

/** Nhiều đường trên cùng trục, ví dụ PSNR theo bước train của nhiều run. */
export function MultiLineChart({ series, label, xLabel }: { series: { name: string; points: Point[] }[]; label: string; xLabel: string }) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const { xRange, yRange, px, py } = project(all);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <Axes xRange={xRange} yRange={yRange} xLabel={xLabel} px={px} py={py} />
      {series.map((s, index) => (
        <polyline key={s.name} points={s.points.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")} style={{ stroke: seriesColor(index) }}>
          <title>{s.name}</title>
        </polyline>
      ))}
    </svg>
  );
}

/** Cột nhóm: mỗi nhóm (ví dụ một run) có một cột cho mỗi chuỗi (ví dụ PSNR toàn ảnh, người, xe). */
export function GroupedBarChart({
  groups,
  seriesNames,
  label,
  unit,
}: {
  groups: { label: string; values: (number | null)[] }[];
  seriesNames: string[];
  label: string;
  unit: string;
}) {
  const values = groups.flatMap((g) => g.values).filter((v): v is number => v != null);
  if (!values.length) return null;
  // Trục y không bắt đầu từ 0: PSNR các vùng khác nhau vài dB, bắt đầu từ 0 sẽ làm mất khác biệt.
  const low = Math.floor(Math.min(...values) - 1);
  const high = Math.ceil(Math.max(...values) + 0.5);
  const py = (y: number) => H - PAD.bottom - ((y - low) / (high - low)) * (H - PAD.top - PAD.bottom);
  const slot = (W - PAD.left - PAD.right) / groups.length;
  const bar = Math.min(22, (slot * 0.8) / seriesNames.length);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {niceTicks(low, high).map((y) => (
        <g key={y}>
          <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={py(y)} y2={py(y)} />
          <text x={PAD.left - 8} y={py(y) + 6} textAnchor="end">
            {y.toFixed(0)}
          </text>
        </g>
      ))}
      {groups.map((group, gi) => {
        const x0 = PAD.left + gi * slot + (slot - bar * seriesNames.length) / 2;
        return (
          <g key={group.label}>
            {group.values.map((value, si) =>
              value == null ? null : (
                <rect key={si} className="bar" x={x0 + si * bar} y={py(value)} width={bar - 2} height={H - PAD.bottom - py(value)} style={{ fill: seriesColor(si) }}>
                  <title>{`${group.label}, ${seriesNames[si]}: ${value.toFixed(2)} ${unit}`}</title>
                </rect>
              ),
            )}
            <text x={PAD.left + gi * slot + slot / 2} y={H - 15} textAnchor="middle">
              {group.label}
            </text>
          </g>
        );
      })}
      <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
    </svg>
  );
}

/** Cột chồng nằm ngang: mỗi hàng (một run) chia thành các phần (ví dụ thời gian từng nhóm stage). */
export function StackedBarChart({
  rows,
  partNames,
  label,
  format,
}: {
  rows: { label: string; parts: { value: number; faded?: boolean }[] }[];
  partNames: string[];
  label: string;
  format: (value: number) => string;
}) {
  const totals = rows.map((r) => r.parts.reduce((a, p) => a + p.value, 0));
  const max = Math.max(...totals, 1);
  const rowH = 30;
  const left = 150;
  const height = PAD.top + rows.length * rowH + 10;
  const sx = (v: number) => (v / max) * (W - left - 120);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={label}>
      {rows.map((row, ri) => {
        let x = left;
        const y = PAD.top + ri * rowH;
        return (
          <g key={row.label}>
            <text x={left - 8} y={y + rowH / 2 + 5} textAnchor="end">
              {row.label}
            </text>
            {row.parts.map((part, pi) => {
              const w = sx(part.value);
              const rect = (
                <rect key={pi} className="bar" x={x} y={y + 4} width={Math.max(0, w - 1)} height={rowH - 10} style={{ fill: seriesColor(pi), opacity: part.faded ? 0.4 : 1 }}>
                  <title>{`${row.label}, ${partNames[pi]}: ${format(part.value)}${part.faded ? " (cache)" : ""}`}</title>
                </rect>
              );
              x += w;
              return rect;
            })}
            <text x={x + 6} y={y + rowH / 2 + 5}>
              {format(totals[ri])}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
