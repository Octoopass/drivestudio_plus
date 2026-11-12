// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SourceRun = string;
export type ShDegree = number;
export type RigidActors = string[];
export type BoxOnlyActors = string[];
export type Omitted = string[];

/**
 * Schema của file ``preview.json`` trong gói xem trước cho Spark (CONTRACTS mục 7).
 *
 * Gói xem trước cho phép xem cảnh trong trình duyệt không cần GPU CUDA; file này ghi gói có gì
 * và thiếu gì so với model đầy đủ.
 *
 * Attributes:
 *     source_run: Thư mục run tạo ra gói, tương đối với ``D2S_WORKDIR``.
 *     sh_degree: Bậc spherical harmonics giữ lại, 0–3.
 *     rigid_actors: ``track_id`` của các xe có Gaussian riêng.
 *     box_only_actors: ``track_id`` của actor không rigid, chỉ hiện bằng box.
 *     omitted: Những phần của model không có trong gói, ví dụ ``["sky"]``.
 */
export interface Preview {
  schema_version: SchemaVersion;
  source_run: SourceRun;
  sh_degree: ShDegree;
  rigid_actors?: RigidActors;
  box_only_actors?: BoxOnlyActors;
  omitted?: Omitted;
}
