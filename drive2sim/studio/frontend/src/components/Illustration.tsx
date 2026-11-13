// Minh hoạ phẳng cho Studio: một đoạn đường có xe, vạch vàng, cột đèn và vòng cảm biến. Vẽ bằng SVG
// nội tuyến, màu lấy từ biến theme nên hợp cả giao diện sáng và tối. Dùng ở Tổng quan, màn chọn
// nguồn của trang Xem 3D và các trạng thái trống.

/** Cảnh đường phố: xe đang quét cảm biến, nền đồi và bầu trời. `width` tính theo px; cao bằng 0,5 rộng. */
export function RoadScene({ width = 360 }: { width?: number }) {
  return (
    <svg className="illo" viewBox="0 0 360 180" width={width} height={width / 2} aria-hidden="true">
      <defs>
        <clipPath id="illo-frame">
          <rect width="360" height="180" rx="18" />
        </clipPath>
      </defs>
      <g clipPath="url(#illo-frame)">
        <rect width="360" height="180" className="illo-sky" />
        <circle cx="292" cy="46" r="22" className="illo-sun" />
        <path d="M0 118 C 60 86, 110 92, 150 110 S 240 124, 300 100 S 350 96, 360 104 V 180 H 0 Z" className="illo-hill-far" />
        <path d="M0 132 C 50 118, 120 122, 180 136 S 290 150, 360 128 V 180 H 0 Z" className="illo-hill" />
        <path d="M0 180 L 120 126 H 240 L 360 180 Z" className="illo-road" />
        <path d="M180 126 V 180" className="illo-lane" strokeDasharray="10 9" />
        <rect x="246" y="78" width="4" height="50" rx="2" className="illo-post" />
        <rect x="236" y="74" width="18" height="6" rx="3" className="illo-post" />
        <circle cx="236" cy="76" r="4" className="illo-sun" />
        <g transform="translate(150 132)">
          <path d="M16 18 c 2 -10, 8 -16, 20 -16 h 28 c 12 0, 18 6, 22 16 h 8 a6 6 0 0 1 6 6 v 8 a4 4 0 0 1 -4 4 H 0 a4 4 0 0 1 -4 -4 v -8 a6 6 0 0 1 6 -6 Z" className="illo-car" />
          <path d="M26 16 c 3 -7, 8 -10, 14 -10 h 10 v 10 Z M 56 6 h 8 c 8 0, 13 4, 16 10 H 56 Z" className="illo-glass" />
          <circle cx="18" cy="36" r="7" className="illo-wheel" />
          <circle cx="76" cy="36" r="7" className="illo-wheel" />
          <rect x="40" y="-4" width="12" height="7" rx="2" className="illo-sensor" />
        </g>
        <g className="illo-scan" fill="none" strokeDasharray="4 5">
          <path d="M166 114 a 30 30 0 0 1 60 0" />
          <path d="M152 114 a 44 44 0 0 1 88 0" />
          <path d="M138 114 a 58 58 0 0 1 116 0" />
        </g>
      </g>
    </svg>
  );
}

/** Biển báo nhỏ cho trạng thái trống: cột biển và tấm biển tròn, không chữ. */
export function EmptySign({ size = 56 }: { size?: number }) {
  return (
    <svg className="illo" viewBox="0 0 56 56" width={size} height={size} aria-hidden="true">
      <path d="M8 50 Q 28 42, 48 50" className="illo-lane" />
      <rect x="26" y="26" width="4" height="22" rx="2" className="illo-post" />
      <circle cx="28" cy="18" r="13" className="illo-sign" />
      <circle cx="28" cy="18" r="9" className="illo-sign-ring" fill="none" />
    </svg>
  );
}
