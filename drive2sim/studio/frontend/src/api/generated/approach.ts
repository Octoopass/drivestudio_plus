// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type ApproachId = string;
export type Name = string;
export type Version = string;
export type Codebase = string;
export type Track = "pinhole" | "fisheye";
export type Level = "L1" | "L2" | "L3" | "L4";
/**
 * @minItems 1
 */
export type CameraModels = [
  "opencv_pinhole" | "opencv_fisheye" | "ftheta" | "ideal_pinhole",
  ...("opencv_pinhole" | "opencv_fisheye" | "ftheta" | "ideal_pinhole")[]
];
export type Cuboids = "required" | "optional" | "none";
/**
 * @minItems 1
 */
export type SplitStrategies = ["every_nth_frame" | "custom", ...("every_nth_frame" | "custom")[]];
/**
 * @minItems 1
 */
export type Choices = [boolean | number | number | string, ...(boolean | number | number | string)[]];
export type Default = boolean | number | string;
export type Reason = string;
export type Constraints = Constraint[];
/**
 * @minItems 1
 */
export type Stages = [StageEntry, ...StageEntry[]];
export type Stage = string;

/**
 * Schema của file khai báo approach ``drive2sim/approaches/<approach_id>.yaml``.
 *
 * Approach là một cách dựng cảnh: danh sách stage chạy lần lượt, có ô chọn, ràng buộc và
 * preset (CONTRACTS mục 4.1). Schema kiểm tra cả sự nhất quán bên trong file: preset và ràng
 * buộc chỉ dùng ô và lựa chọn có thật, chỗ giữ chỗ trỏ tới ô có thật.
 *
 * Attributes:
 *     approach_id: Định danh ``snake_case``, ví dụ ``omnire``.
 *     name: Tên hiển thị.
 *     version: Phiên bản của file approach.
 *     codebase: ``drivestudio`` hoặc ``external:<tên>``.
 *     track: ``pinhole`` hoặc ``fisheye``.
 *     level: Mức năng lực cao nhất đạt được, ``L1``–``L4``.
 *     requires: Điều kiện dữ liệu.
 *     options: Tên ô → ô chọn.
 *     constraints: Các tổ hợp không hợp lệ.
 *     presets: Tên preset → giá trị của các ô khác mặc định.
 *     stages: Các bước theo thứ tự chạy, ít nhất một.
 */
export interface Approach {
  schema_version: SchemaVersion;
  approach_id: ApproachId;
  name: Name;
  version: Version;
  codebase: Codebase;
  track: Track;
  level: Level;
  requires: Requires;
  options?: Options;
  constraints?: Constraints;
  presets?: Presets;
  stages: Stages;
}
/**
 * Điều kiện dữ liệu mà một approach cần để chạy được (CONTRACTS mục 4.2).
 *
 * Attributes:
 *     camera_models: Các mô hình camera approach nhận trực tiếp, không cần khử méo.
 *     cuboids: Approach cần box 3D không: ``required``, ``optional`` hoặc ``none``.
 *     split_strategies: Các chiến lược split codebase của approach hỗ trợ.
 */
export interface Requires {
  camera_models: CameraModels;
  cuboids: Cuboids;
  split_strategies: SplitStrategies;
}
export interface Options {
  [k: string]: OptionSpec;
}
/**
 * Một ô chọn của approach: các lựa chọn thay thế được cho nhau và giá trị mặc định.
 *
 * Ví dụ ô ``renderer`` có ``choices: [3dgs, 3dgut, 3dgeer]``. ``default`` có thể là ``"auto"``:
 * engine lấy lựa chọn đầu tiên không bị khoá theo dữ liệu (CONTRACTS mục 4.2).
 *
 * Attributes:
 *     choices: Các lựa chọn, ít nhất một, không trùng; thứ tự là thứ tự ưu tiên của ``auto``.
 *     default: Giá trị mặc định: một lựa chọn trong ``choices``, hoặc ``"auto"``.
 */
export interface OptionSpec {
  choices: Choices;
  default: Default;
}
/**
 * Một tổ hợp lựa chọn không hợp lệ trong approach.
 *
 * Khi mọi ô trong ``when`` có đúng giá trị đó thì cấm mọi giá trị trong ``forbid``. Ví dụ
 * ``when: {renderer: 3dgeer}, forbid: {campose: true}`` vì 3DGEER không tối ưu được pose camera.
 *
 * Attributes:
 *     when: Ô → giá trị làm ràng buộc có hiệu lực.
 *     forbid: Ô → giá trị bị cấm khi ``when`` khớp.
 *     reason: Lý do bằng tiếng Việt, hiện cho người dùng khi tổ hợp bị khoá.
 */
export interface Constraint {
  when: When;
  forbid: Forbid;
  reason: Reason;
}
export interface When {
  [k: string]: boolean | number | string;
}
export interface Forbid {
  [k: string]: boolean | number | string;
}
export interface Presets {
  [k: string]: {
    [k: string]: boolean | number | string;
  };
}
/**
 * Một bước trong danh sách ``stages`` của approach.
 *
 * Attributes:
 *     stage: ``stage_id`` (ví dụ ``prep.select``) hoặc chỗ giữ chỗ ``${options.<ô>}``; chỗ giữ
 *         chỗ được thay bằng lựa chọn của ô trước khi chạy.
 *     params: Tham số cho stage; giá trị cũng có thể là chỗ giữ chỗ.
 */
export interface StageEntry {
  stage: Stage;
  params?: Params;
}
export interface Params {
  [k: string]: unknown;
}
