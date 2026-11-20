// Ước lượng thời gian còn lại của stage đang chạy từ sự kiện progress (fraction 0..1, CONTRACTS mục
// 11.4): giữ điểm đầu (thời điểm, fraction) khi stage bắt đầu báo tiến độ và điểm mới nhất, rồi
// còn lại = đã chạy × (1 − f) / (f − f0). Tính từ điểm đầu chứ không từ 0, nên run chạy tiếp hoặc
// mở Studio giữa chừng vẫn đúng. Chưa đủ dữ liệu (dưới 5 giây hoặc fraction chưa nhích 0,5%) thì
// không đoán.

export interface EtaTrack {
  key: string;
  t0: number;
  f0: number;
  t: number;
  f: number;
}

/** Cập nhật điểm theo dõi của một stage (`key` = stage_id); stage đổi hoặc fraction lùi thì làm lại. */
export function trackProgress(prev: EtaTrack | undefined, key: string, fraction: number | null | undefined, now = Date.now() / 1000): EtaTrack | undefined {
  if (fraction == null) return prev?.key === key ? prev : undefined;
  if (!prev || prev.key !== key || fraction < prev.f) return { key, t0: now, f0: fraction, t: now, f: fraction };
  return { ...prev, t: now, f: fraction };
}

/** Số giây còn lại ước lượng, hoặc null khi chưa đủ dữ liệu. */
export function etaSeconds(track: EtaTrack | undefined): number | null {
  if (!track) return null;
  const done = track.f - track.f0;
  const elapsed = track.t - track.t0;
  if (done < 0.005 || elapsed < 5) return null;
  return Math.max(0, ((1 - track.f) * elapsed) / done);
}
