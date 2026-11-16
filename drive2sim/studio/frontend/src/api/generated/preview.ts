// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SourceRun = string;
export type ShDegree = number;
export type SplatFormat = "ply" | "spz";
export type UpAxis = string | null;
export type Antialiased = boolean;
export type RigidActors = string[];
export type Humans = string[];
export type Sky = string | null;
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
 *     splat_format: Đuôi file Gaussian của gói: ``spz`` (từ bản 0.2) hoặc ``ply`` (gói cũ).
 *     up_axis: Trục lên của hệ world, ví dụ ``-y`` (suy từ pose camera); ``None`` nếu không
 *         biết.
 *     antialiased: Model train với chế độ chống răng cưa của gsplat không; viewer chọn cách làm
 *         mờ splat theo đó.
 *     rigid_actors: ``track_id`` của các xe có Gaussian riêng.
 *     humans: ``track_id`` của các người có Gaussian và skin trong gói.
 *     sky: Tên file cubemap trời trong gói, ``None`` nếu model không có trời.
 *     box_only_actors: ``track_id`` của actor chưa có Gaussian trong gói, chỉ hiện bằng box.
 *     omitted: Những phần của model không có trong gói, ví dụ ``["DeformableNodes"]``.
 */
export interface Preview {
  schema_version: SchemaVersion;
  source_run: SourceRun;
  sh_degree: ShDegree;
  splat_format?: SplatFormat;
  up_axis?: UpAxis;
  antialiased?: Antialiased;
  rigid_actors?: RigidActors;
  humans?: Humans;
  sky?: Sky;
  box_only_actors?: BoxOnlyActors;
  omitted?: Omitted;
}
