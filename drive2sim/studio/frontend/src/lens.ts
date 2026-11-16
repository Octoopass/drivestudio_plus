// Loại ống kính của camera (pinhole hay mắt cá), dùng chung cho trang Cảnh và trang Chạy mới.
// Cùng bảng với CAMERA_TRACK của engine (core/schema.py): cảnh chỉ mô tả từng camera thuộc loại
// nào; split chọn camera nào được dùng; renderer của approach vẽ được loại nào thì engine kiểm theo
// camera của split (CONTRACTS mục 4.2).

export type Lens = "pinhole" | "fisheye";

const LENS: Record<string, Lens> = {
  opencv_pinhole: "pinhole",
  ideal_pinhole: "pinhole",
  opencv_fisheye: "fisheye",
  ftheta: "fisheye",
};

/** Loại ống kính của một mô hình camera; mô hình lạ coi như pinhole. */
export const lensOf = (model: string): Lens => LENS[model] ?? "pinhole";

/** Số camera theo loại ống kính. */
export function lensCounts(models: string[]): Record<Lens, number> {
  const counts: Record<Lens, number> = { pinhole: 0, fisheye: 0 };
  for (const model of models) counts[lensOf(model)] += 1;
  return counts;
}

/** Nhóm của một tập camera: chỉ pinhole, chỉ mắt cá, hỗn hợp; ``null`` khi không có camera. */
export function lensGroup(models: string[]): Lens | "mixed" | null {
  const { pinhole, fisheye } = lensCounts(models);
  if (pinhole && fisheye) return "mixed";
  if (pinhole) return "pinhole";
  if (fisheye) return "fisheye";
  return null;
}

/** Tóm tắt dễ đọc, ví dụ "2 pinhole + 2 mắt cá". */
export function lensSummary(models: string[], t: (key: string, values?: Record<string, string | number>) => string): string {
  const counts = lensCounts(models);
  return (["pinhole", "fisheye"] as Lens[])
    .filter((lens) => counts[lens] > 0)
    .map((lens) => t(`lens.count.${lens}`, { count: counts[lens] }))
    .join(" + ");
}
