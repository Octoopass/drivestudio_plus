"""Loader DriveStudio cho cảnh PHENIKAA (Phenikaa-X Perception Data).

Converter ``drive2sim/ingest/phenikaa.py`` ghi cảnh theo cùng khung thư mục KITTI như KITTI-360
(``images/``, ``lidar/``, ``ego_pose/``, ``extrinsics/``, ``intrinsics/``, CONTRACTS mục 2), nên
loader này kế thừa loader KITTI-360 (camera pinhole và mắt cá đọc mô hình từ ``manifest.yaml``,
camera mắt cá vẽ theo mô hình ống kính, không khử méo) và chỉ khác ở hai chỗ:

- Ảnh là JPEG (KITTI-360 là PNG): ảnh mắt cá là ảnh gốc, ảnh pinhole đã khử méo lúc ingest.
- Chưa có mask thân xe cho camera nào: không dùng asset thân xe của KITTI-360.

Mười camera theo thứ tự chỉ số trong cảnh: ``CAM_P_F``, ``CAM_P_FL``, ``CAM_P_FR``, ``CAM_P_B``,
``CAM_P_L``, ``CAM_P_R`` (pinhole), ``CAM_F_F``, ``CAM_F_L``, ``CAM_F_R``, ``CAM_F_B`` (mắt cá);
cảnh ingest một phần camera thì chỉ số theo ``manifest.cameras``.

File này chứa ``PhenikaaCameraData`` và ``PhenikaaPixelSource``. LiDAR dùng ``KITTILiDARSource``.
Config dataset ở ``drive2sim/configs/datasets/phenikaa/``.
"""

from drive2sim.recon.data.loaders.kitti import KITTICameraData
from drive2sim.recon.data.loaders.kitti360 import KITTI360CameraData, KITTI360PixelSource

# Mười camera của PHENIKAA (drive2sim/ingest/phenikaa.py, CAMERAS).
AVAILABLE_CAM_LIST = list(range(10))


class PhenikaaCameraData(KITTI360CameraData):
    """Một camera PHENIKAA: như KITTI-360 nhưng ảnh JPEG và chưa có mask thân xe."""

    def create_all_filelist(self) -> None:
        """Danh sách file như KITTI (ảnh ``.jpg``), không đổi sang ``.png`` như KITTI-360."""
        KITTICameraData.create_all_filelist(self)

    def load_egocar_mask(self) -> None:
        """Chưa có mask thân xe cho PHENIKAA: mọi pixel trong vùng ống kính đều được dùng."""
        self.egocar_mask = None


class PhenikaaPixelSource(KITTI360PixelSource):
    """Nguồn ảnh của cảnh PHENIKAA, dùng ``PhenikaaCameraData`` cho mọi camera."""

    CAMERA_DATA_CLASS = PhenikaaCameraData
    SMPL_CAMERA_LIST = AVAILABLE_CAM_LIST
