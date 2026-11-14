// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type ApproachId = string;
export type Name = string;
export type Version = string;
export type Codebase = string;
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
export type Label = string | null;
export type Help = string | null;
export type Labels = string[];
export type Descriptions = string[];
export type Group = string | null;
export type CameraModels1 = ("opencv_pinhole" | "opencv_fisheye" | "ftheta" | "ideal_pinhole")[][];
export type Reason = string;
export type Constraints = Constraint[];
export type Label1 = string;
export type Help1 = string | null;
/**
 * @minItems 1
 */
export type Stages = [StageEntry, ...StageEntry[]];
export type Stage = string;

/**
 * Schema của file khai báo approach ``drive2sim/approaches/<approach_id>.yaml``.
 *
 * Approach là một cách dựng cảnh: danh sách stage chạy lần lượt, có mục cấu hình, ràng buộc và
 * preset (CONTRACTS mục 4.1). Schema kiểm tra cả sự nhất quán bên trong file: preset và ràng
 * buộc chỉ dùng mục cấu hình và lựa chọn có thật, chỗ giữ chỗ trỏ tới mục cấu hình có thật.
 *
 * Attributes:
 *     approach_id: Định danh ``snake_case``, ví dụ ``omnire``.
 *     name: Tên hiển thị.
 *     version: Phiên bản của file approach.
 *     codebase: ``drivestudio`` hoặc ``external:<tên>``.
 *     level: Mức năng lực cao nhất đạt được, ``L1``–``L4``.
 *     requires: Điều kiện dữ liệu. Track (pinhole, fisheye) của approach và của từng preset suy
 *         ra từ mô hình camera nhận được (``CAMERA_TRACK``), không ghi riêng; file ``0.1``,
 *         ``0.2`` có trường ``track`` thì trường đó được bỏ qua.
 *     options: Tên mục → mục cấu hình.
 *     constraints: Các tổ hợp không hợp lệ.
 *     presets: Tên preset → giá trị của các mục cấu hình khác mặc định.
 *     preset_help: Tên preset → mô tả ngắn hiện trên Studio (thêm từ ``d2s-approach/0.2``).
 *     groups: Tên nhóm → nhóm mục cấu hình, theo thứ tự hiện trên Studio; chỉ để hiển thị.
 *     stages: Các bước theo thứ tự chạy, ít nhất một.
 */
export interface Approach {
  schema_version: SchemaVersion;
  approach_id: ApproachId;
  name: Name;
  version: Version;
  codebase: Codebase;
  level: Level;
  requires: Requires;
  options?: Options;
  constraints?: Constraints;
  presets?: Presets;
  preset_help?: PresetHelp;
  groups?: Groups;
  stages: Stages;
}
/**
 * Điều kiện dữ liệu mà một approach cần để chạy được (CONTRACTS mục 4.2).
 *
 * Attributes:
 *     camera_models: Các mô hình camera approach nhận trực tiếp, không cần khử méo, với ít
 *         nhất một tổ hợp lựa chọn. Lựa chọn nào chỉ nhận một phần thì ghi ở
 *         ``OptionSpec.camera_models``.
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
 * Một mục cấu hình của approach: các lựa chọn thay thế được cho nhau và giá trị mặc định.
 *
 * Ví dụ mục cấu hình ``renderer`` có ``choices: [3dgs, 3dgut, 3dgeer]``. ``default`` có thể là
 * ``"auto"``: engine lấy lựa chọn đầu tiên không bị khoá theo dữ liệu (CONTRACTS mục 4.2).
 *
 * Attributes:
 *     choices: Các lựa chọn, ít nhất một, không trùng; thứ tự là thứ tự ưu tiên của ``auto``.
 *     default: Giá trị mặc định: một lựa chọn trong ``choices``, hoặc ``"auto"``.
 *     label: Tên hiển thị của mục cấu hình trên Studio, ví dụ ``Renderer``; trống thì dùng tên
 *         mục cấu hình.
 *     help: Giải thích ngắn mục này làm gì, hiện trong icon (i) cạnh tên mục cấu hình.
 *     labels: Tên hiển thị của từng lựa chọn, cùng thứ tự với ``choices``; trống thì hiện giá
 *         trị (thêm từ ``d2s-approach/0.2``, cùng ``label``, ``help``).
 *     descriptions: Giải thích từng lựa chọn, cùng thứ tự với ``choices``; hiện trong icon (i).
 *     group: Nhóm của mục cấu hình trong ``Approach.groups``; Studio hiện các mục cấu hình theo
 *         nhóm.
 *     camera_models: Mô hình camera mỗi lựa chọn nhận, cùng thứ tự với ``choices``; trống thì
 *         mọi lựa chọn nhận đủ ``requires.camera_models``. Ví dụ renderer ``3dgs`` chỉ nhận
 *         ``opencv_pinhole``, còn ``3dgeer`` nhận thêm ``opencv_fisheye``, ``ftheta`` (thêm từ
 *         ``d2s-approach/0.3``).
 */
export interface OptionSpec {
  choices: Choices;
  default: Default;
  label?: Label;
  help?: Help;
  labels?: Labels;
  descriptions?: Descriptions;
  group?: Group;
  camera_models?: CameraModels1;
}
/**
 * Một tổ hợp lựa chọn không hợp lệ trong approach.
 *
 * Khi mọi mục cấu hình trong ``when`` có đúng giá trị đó thì cấm mọi giá trị trong ``forbid``. Ví
 * dụ ``when: {renderer: 3dgeer}, forbid: {campose: true}`` vì 3DGEER không tối ưu được pose
 * camera.
 *
 * Attributes:
 *     when: Mục → giá trị làm ràng buộc có hiệu lực.
 *     forbid: Mục → giá trị bị cấm khi ``when`` khớp.
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
export interface PresetHelp {
  [k: string]: string;
}
export interface Groups {
  [k: string]: OptionGroup;
}
/**
 * Một nhóm mục cấu hình hiện chung một mục trên Studio, ví dụ nhóm thuật toán phụ trợ.
 *
 * Attributes:
 *     label: Tên nhóm.
 *     help: Giải thích nhóm, hiện trong icon (i) cạnh tên nhóm.
 */
export interface OptionGroup {
  label: Label1;
  help?: Help1;
}
/**
 * Một bước trong danh sách ``stages`` của approach.
 *
 * Attributes:
 *     stage: ``stage_id`` (ví dụ ``prep.select``) hoặc chỗ giữ chỗ ``${options.<mục>}``; chỗ giữ
 *         chỗ được thay bằng lựa chọn của mục cấu hình trước khi chạy.
 *     params: Tham số cho stage; giá trị cũng có thể là chỗ giữ chỗ.
 */
export interface StageEntry {
  stage: Stage;
  params?: Params;
}
export interface Params {
  [k: string]: unknown;
}
