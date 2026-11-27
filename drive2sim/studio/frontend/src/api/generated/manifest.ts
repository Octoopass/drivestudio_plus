// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.

export type SchemaVersion = string;
export type SceneId = string;
export type Dataset =
  | "waymo"
  | "nuscenes"
  | "nuplan"
  | "argoverse2"
  | "pandaset"
  | "kitti"
  | "kitti360"
  | "phenikaa"
  | "physicalai"
  | "carla"
  | "colmap"
  | "custom";
export type SequenceId = string;
export type Converter = string;
export type ConverterVersion = string;
export type OutputVersion = number | null;
export type DataLicense = string | null;
export type Track = "pinhole" | "fisheye";
/**
 * @minItems 1
 */
export type Cameras = [ManifestCamera, ...ManifestCamera[]];
export type CameraId = string;
export type Model = "opencv_pinhole" | "opencv_fisheye" | "ftheta" | "ideal_pinhole";
/**
 * @minItems 2
 * @maxItems 2
 */
export type Resolution = [unknown, unknown];
export type Shutter =
  "GLOBAL" | "ROLLING_TOP_TO_BOTTOM" | "ROLLING_LEFT_TO_RIGHT" | "ROLLING_BOTTOM_TO_TOP" | "ROLLING_RIGHT_TO_LEFT";
export type MaxAngle = number | null;
export type Lidars = string[];
export type ReferenceSensor = string;
export type NumFrames = number;
export type Lidar = boolean;
export type Cuboids = "gt" | "autolabel" | "none";
export type CameraLabels = string[];
export type EgocarMask = boolean;
export type HumanFrames = [unknown, unknown][] | null;

/**
 * Schema của file ``manifest.yaml`` đi kèm dữ liệu chuẩn của một cảnh (CONTRACTS mục 2.2).
 *
 * Manifest ghi cảnh có gì và thiếu gì. Converter ghi file này lúc ingest; engine đọc nó để quyết
 * định approach và lựa chọn nào chạy được; Studio đọc nó để hiện tổng quan cảnh.
 *
 * Attributes:
 *     scene_id: Định danh cảnh, ``<dataset>_<sequence_id>``.
 *     source: Nguồn gốc của cảnh.
 *     track: ``pinhole`` hoặc ``fisheye``.
 *     cameras: Các camera, ít nhất một, ``camera_id`` không trùng.
 *     lidars: Id các LiDAR; rỗng nếu cảnh không có LiDAR.
 *     timeline: Dòng thời gian tham chiếu.
 *     available: Những gì cảnh có sẵn.
 */
export interface Manifest {
  schema_version: SchemaVersion;
  scene_id: SceneId;
  source: ManifestSource;
  track: Track;
  cameras: Cameras;
  lidars?: Lidars;
  timeline: Timeline;
  available: Available;
}
/**
 * Nguồn gốc của một cảnh: lấy từ dataset nào, chuỗi nào, chuyển bằng converter nào.
 *
 * Attributes:
 *     dataset: Tên nguồn, ví dụ ``nuscenes``; cũng là tiền tố của ``scene_id``.
 *     sequence_id: Id của chuỗi trong nguồn, ví dụ ``scene-0061``.
 *     converter: Converter đã dùng để tạo thư mục chuẩn, ví dụ ``drivestudio.nuscenes`` (script
 *         tiền xử lý của DriveStudio cho nuScenes).
 *     converter_version: Phiên bản converter, dạng ``<phiên bản drive2sim>+<commit>``; nằm
 *         trong khoá cache của mọi stage nhận cảnh, nên ingest lại bằng code khác thì cache cũ
 *         không bị dùng nhầm.
 *     output_version: Phiên bản đầu ra của converter lúc ingest, số nguyên tăng tay mỗi khi
 *         converter đổi dữ liệu nó ghi ra (``KITTI360_OUTPUT_VERSION``,
 *         ``DRIVESTUDIO_OUTPUT_VERSION`` trong ``drive2sim/ingest/manifest.py``). Khác
 *         ``converter_version`` (đổi theo mọi commit), số này chỉ đổi khi đầu ra đổi, nên dùng
 *         để biết cảnh đã cũ. Thêm từ ``d2s-manifest/1.1``; manifest cũ không có trường này
 *         được coi là phiên bản ``1`` (trừ commit ghi trong ``manifest.py``).
 *     data_license: License của dữ liệu nguồn, nếu biết.
 */
export interface ManifestSource {
  dataset: Dataset;
  sequence_id: SequenceId;
  converter: Converter;
  converter_version: ConverterVersion;
  output_version?: OutputVersion;
  data_license?: DataLicense;
}
/**
 * Một camera trong cảnh, như ghi trong manifest.
 *
 * Thông số chi tiết (intrinsics, hệ số méo, extrinsic) nằm trong kho NCore; manifest chỉ ghi
 * những gì engine cần để khoá approach và Studio cần để hiện tổng quan.
 *
 * Attributes:
 *     camera_id: Id của camera, giữ nguyên như trong nguồn, ví dụ ``CAM_FRONT``.
 *     model: Mô hình ống kính: ``opencv_pinhole``, ``opencv_fisheye``, ``ftheta`` hoặc
 *         ``ideal_pinhole``.
 *     resolution: ``[width, height]`` tính bằng pixel.
 *     shutter: Loại shutter theo tên của NCore, ví dụ ``GLOBAL``.
 *     max_angle: Góc lệch trục lớn nhất (rad) mà mô hình trong ``intrinsics/`` còn đúng, khi
 *         converter phải chuyển gần đúng mô hình gốc của dataset (KITTI-360: Mei khớp sang
 *         OpenCV fisheye tới 1,45 rad); ``None`` khi không có giới hạn như vậy. Thêm sau
 *         ``d2s-manifest/1.1`` mà không tăng phiên bản (không bắt buộc).
 */
export interface ManifestCamera {
  camera_id: CameraId;
  model: Model;
  resolution: Resolution;
  shutter: Shutter;
  max_angle?: MaxAngle;
}
/**
 * Dòng thời gian tham chiếu của cảnh, dùng để đánh chỉ số khung.
 *
 * Khung thứ ``i`` của cảnh là khung thứ ``i`` của ``reference_sensor``; ảnh của camera khác được
 * ghép với khung gần nhất về thời gian (CONTRACTS mục 2.2).
 *
 * Attributes:
 *     reference_sensor: Id của camera hoặc LiDAR làm mốc.
 *     num_frames: Số khung của cảnh, lớn hơn 0.
 */
export interface Timeline {
  reference_sensor: ReferenceSensor;
  num_frames: NumFrames;
}
/**
 * Những gì cảnh có sẵn từ nguồn; engine dựa vào đây để khoá approach (CONTRACTS mục 4.2).
 *
 * Attributes:
 *     lidar: Cảnh có LiDAR không; phải khớp với danh sách ``lidars`` của manifest.
 *     cuboids: Nguồn của box 3D: ``gt`` (nhãn gốc), ``autolabel`` (sinh tự động) hoặc ``none``.
 *     camera_labels: Nhãn theo ảnh có sẵn trong nguồn, ví dụ ``["depth", "segmentation"]``.
 *     egocar_mask: Nguồn có mask thân xe không.
 *     human_frames: Các đoạn khung ``[đầu, cuối]`` (tính cả hai đầu) có ít nhất một box người
 *         (lớp OmniRe dựng bằng SMPL, ``drive2sim/core/instances.py``); ``[]`` nếu cảnh không
 *         có người; ``None`` nếu không biết. Engine dùng để bỏ bước dựng SMPL khi đoạn khung
 *         của split không có người. Thêm từ ``d2s-manifest/1.2``; manifest cũ không có trường
 *         này thì ``Workdir.load_scene`` tự đọc ``instances/`` nếu cảnh có trên máy.
 */
export interface Available {
  lidar: Lidar;
  cuboids: Cuboids;
  camera_labels?: CameraLabels;
  egocar_mask?: EgocarMask;
  human_frames?: HumanFrames;
}
