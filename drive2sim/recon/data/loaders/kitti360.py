"""Loader DriveStudio cho cảnh KITTI-360, vẽ camera mắt cá theo đúng mô hình ống kính (không khử méo).

Converter ``drive2sim/ingest/kitti360.py`` ghi cảnh KITTI-360 theo khung KITTI của DriveStudio
(``images/<khung>_<chỉ số>.png``, ``lidar/``, ``ego_pose/``, ``extrinsics/``, ``intrinsics/``,
CONTRACTS mục 2). Khác KITTI gốc ở ba chỗ, nên loader này kế thừa loader KITTI
(``drive2sim/recon/data/loaders/kitti.py``) và chỉ sửa các chỗ đó:

- Ảnh là PNG (KITTI gốc là JPG).
- Bốn camera có hai mô hình: ``image_00``, ``image_01`` là ``opencv_pinhole`` (1408x376, ảnh đã
  rectify, 9 số ``fx fy cx cy k1 k2 p1 p2 k3`` với méo bằng 0); ``image_02``, ``image_03`` là
  ``opencv_fisheye`` (1400x1400, 8 số ``fx fy cx cy k1 k2 k3 k4`` theo thứ tự của
  ``cv2.fisheye``). Mô hình và độ phân giải của từng camera đọc từ ``manifest.yaml`` của cảnh
  (cảnh ảo của stage train có symlink tới file này, ``drive2sim/stages/scene_view.py``), không
  đoán theo số lượng số trong file intrinsics.
- Camera mắt cá được giữ nguyên ảnh gốc: tia của từng pixel tính bằng ``camera_model_rays`` với
  mô hình ``fisheye`` và ``radial_coeffs``, ``cam_infos`` mang ``camera_model``,
  ``radial_coeffs`` để trainer (``drive2sim/recon/trainers/base.py``) gọi renderer UT hoặc GEER.
  Pixel có tia lệch trục quá ``MEI_FIT_MAX_ANGLE`` (``drive2sim/core/camera.py``, giới hạn của
  phép khớp Mei sang OpenCV fisheye) bị đánh dấu không hợp lệ (``egocar_masks``,
  ``valid_masks``), nên không góp vào loss, giống mask ``invalid_masks`` của PhysicalAI.

File này chứa:
- ``KITTI360CameraData``: một camera KITTI-360 (pinhole hoặc fisheye).
- ``KITTI360PixelSource``: nguồn ảnh của cảnh, dùng ``KITTI360CameraData`` cho mọi camera.

LiDAR dùng nguyên ``KITTILiDARSource`` (cùng định dạng ``lidar/<khung>.bin``). Config dataset
``drive2sim/configs/datasets/kitti360/`` trỏ tới hai lớp này; stage ``train.drivestudio`` chọn
config theo manifest. Xem ``docs/datasets/kitti360.md``, ARCHITECTURE mục 7.1.

Quy ước pixel: giống loader KITTI và trainer ``train_gsplat_3dgut``, ``cx``, ``cy`` trong file
intrinsics được dùng nguyên (không cộng 0,5); tâm pixel ``i`` nằm ở toạ độ ``i + 0.5``.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE
from drive2sim.core.files import load
from drive2sim.core.schema import Manifest
from drive2sim.recon.data.loaders.kitti import KITTICameraData, KITTIPixelSource
from drive2sim.recon.utils.geometry import camera_model_rays, project_world_camera

MANIFEST_FILE = "manifest.yaml"
# Mô hình camera trong manifest → mô hình camera trainer DriveStudio nhận (recon/trainers/base.py).
TRAINER_CAMERA_MODELS = {"opencv_pinhole": "pinhole", "opencv_fisheye": "fisheye"}
# Bốn camera của KITTI-360 theo thứ tự chỉ số trong cảnh: image_00, image_01, image_02, image_03.
AVAILABLE_CAM_LIST = [0, 1, 2, 3]


class KITTI360CameraData(KITTICameraData):
    """Một camera của cảnh KITTI-360, đọc mô hình ống kính từ manifest.

    ``KITTI360PixelSource`` tạo một đối tượng cho mỗi camera được chọn. Camera pinhole đi đúng
    đường của loader KITTI (tia pinhole của ``CameraData.get_image``); camera fisheye có tia,
    mask vùng hợp lệ và phép chiếu điểm riêng.

    Attributes:
        camera_model: Mô hình camera của trainer, ``"pinhole"`` hoặc ``"fisheye"``.
        radial_coeffs: Bốn hệ số ``k1..k4`` của OpenCV fisheye (không có đơn vị nên không đổi khi
            thu nhỏ ảnh), dạng tensor ``(4,)``; ``None`` với camera pinhole.
    """

    def __init__(self, **kwargs: object) -> None:
        """Đọc mô hình và độ phân giải của camera từ manifest rồi nạp dữ liệu như KITTI.

        Args:
            **kwargs: Tham số của ``CameraData`` (``data_path``, ``cam_id``, ...); ``cam_id`` là
                chỉ số camera trong cảnh, cũng là vị trí của camera trong ``manifest.cameras``.

        Raises:
            ValueError: Nếu manifest không đọc được, chỉ số camera vượt số camera của manifest,
                hoặc mô hình camera không phải ``opencv_pinhole``, ``opencv_fisheye``.
        """
        manifest = load(Path(str(kwargs["data_path"])) / MANIFEST_FILE, Manifest)
        index = int(kwargs["cam_id"])
        if not 0 <= index < len(manifest.cameras):
            raise ValueError(f"manifest có {len(manifest.cameras)} camera, không có chỉ số {index}")
        camera = manifest.cameras[index]
        if camera.model not in TRAINER_CAMERA_MODELS:
            raise ValueError(f"{camera.camera_id}: loader KITTI-360 không đọc được {camera.model}")
        self.camera_model = TRAINER_CAMERA_MODELS[camera.model]
        self.radial_coeffs = None
        self._ray_cache = {}
        width, height = camera.resolution
        # CameraData đọc camera_metadata thay cho DATASETS_CONFIG; original_size là (cao, rộng).
        self.camera_metadata = {
            "camera_name": camera.camera_id,
            "original_size": (height, width),
            "egocar_visible": False,
        }
        super().__init__(**kwargs)

    def create_all_filelist(self) -> None:
        """Lập danh sách file như KITTI nhưng ảnh là PNG (converter chép nguyên PNG của KITTI-360)."""
        super().create_all_filelist()
        self.img_filepaths = np.array(
            [str(Path(path).with_suffix(".png")) for path in self.img_filepaths]
        )

    def load_calibrations(self) -> None:
        """Đọc intrinsics, extrinsics và pose; camera fisheye đọc 8 số và giữ ``radial_coeffs``.

        Camera pinhole dùng nguyên ``KITTICameraData.load_calibrations``. Camera fisheye: ``fx``,
        ``fy``, ``cx``, ``cy`` được nhân theo tỉ lệ ``load_size / original_size`` như pinhole;
        ``k1..k4`` giữ nguyên vì là hệ số của góc tia, không phụ thuộc kích thước ảnh.

        Raises:
            ValueError: Nếu bật ``undistort`` cho camera fisheye (loader chỉ vẽ ảnh gốc), hoặc file
                intrinsics của camera fisheye không có đúng 8 số.
        """
        if self.camera_model == "pinhole":
            super().load_calibrations()
            return
        if self.undistort:
            raise ValueError(
                f"{self.cam_name}: camera fisheye của KITTI-360 chỉ train trên ảnh gốc; "
                "đặt data.pixel_source.undistort=False"
            )
        intrinsic = np.loadtxt(os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt"))
        if intrinsic.shape != (8,):
            raise ValueError(
                f"{self.cam_name}: intrinsics fisheye cần 8 số fx fy cx cy k1 k2 k3 k4, "
                f"nhận {intrinsic.size}"
            )
        scale_x = self.load_size[1] / self.original_size[1]
        scale_y = self.load_size[0] / self.original_size[0]
        fx, fy, cx, cy = intrinsic[:4]
        K = torch.tensor(
            [[fx * scale_x, 0, cx * scale_x], [0, fy * scale_y, cy * scale_y], [0, 0, 1]]
        ).float()
        self.radial_coeffs = torch.from_numpy(intrinsic[4:8]).float()
        self.cam_to_worlds = self.get_camera2worlds(
            self.data_path, self.cam_id, self.start_timestep, self.end_timestep
        )
        self.intrinsics = K.repeat(len(self.cam_to_worlds), 1, 1)
        self.distortions = None

    def to(self, device: torch.device) -> "KITTI360CameraData":
        """Chuyển dữ liệu của camera sang ``device``, gồm cả ``radial_coeffs``; xoá cache tia."""
        super().to(device)
        self.device = device
        if self.radial_coeffs is not None:
            self.radial_coeffs = self.radial_coeffs.to(device)
        self._ray_cache.clear()
        return self

    def camera_rays(self, intrinsics: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        """Tính tia đơn vị trong hệ camera của mọi pixel fisheye và mask pixel hợp lệ.

        Tia lấy từ ``camera_model_rays`` (giải ngược ``theta_d = theta·(1 + k1·θ² + ... + k4·θ⁸)``).
        Pixel hợp lệ khi giải ngược hội tụ và góc tia ``theta`` không vượt ``MEI_FIT_MAX_ANGLE``:
        ngoài góc đó đa thức OpenCV fisheye không còn khớp mô hình Mei gốc. Kết quả được nhớ theo
        kích thước ảnh, vì intrinsics không đổi theo khung.

        Args:
            intrinsics: Ma trận ``K`` 3x3 ứng với ảnh ``height`` x ``width``.
            height: Chiều cao ảnh (pixel).
            width: Chiều rộng ảnh (pixel).

        Returns:
            ``(rays, valid)``: ``rays`` dạng ``(height, width, 3)``, ``valid`` dạng
            ``(height, width)`` kiểu bool.
        """
        key = (height, width)
        if key not in self._ray_cache:
            rays, valid = camera_model_rays(
                height, width, intrinsics, "fisheye", self.radial_coeffs
            )
            valid &= rays[..., 2] >= float(np.cos(MEI_FIT_MAX_ANGLE))
            self._ray_cache[key] = rays, valid
        return self._ray_cache[key]

    def get_image(self, frame_idx: int) -> tuple[dict[str, Tensor], dict[str, object]]:
        """Lấy tia, ảnh, mask của một khung; camera fisheye thay tia pinhole bằng tia fisheye.

        Với camera fisheye: ``viewdirs`` là tia fisheye quay sang hệ world, ``direction_norm``
        bằng 1 (tia đã chuẩn hoá, như PhysicalAI), pixel ngoài vùng hợp lệ có
        ``egocar_masks = 1`` và ``valid_masks = False``; ``cam_infos`` thêm ``camera_model`` và
        ``radial_coeffs``. Camera pinhole trả nguyên kết quả của ``CameraData.get_image``.

        Args:
            frame_idx: Chỉ số khung trong các khung đã nạp.

        Returns:
            ``(image_infos, cam_infos)`` như ``CameraData.get_image``.
        """
        image, camera = super().get_image(frame_idx)
        if self.camera_model == "pinhole":
            return image, camera
        height, width = int(camera["height"]), int(camera["width"])
        rays, valid = self.camera_rays(camera["intrinsics"], height, width)
        rotation = camera["camera_to_world"][:3, :3]
        image["viewdirs"] = torch.einsum("ij,...j->...i", rotation, rays)
        image["direction_norm"] = torch.ones_like(rays[..., :1])
        invalid = ~valid
        if "egocar_masks" in image:
            invalid = invalid | image["egocar_masks"].bool()
        image["egocar_masks"] = invalid.float()
        image["valid_masks"] = ~invalid
        camera.update(camera_model=self.camera_model, radial_coeffs=self.radial_coeffs)
        return image, camera

    def project_points(
        self, world_points: Tensor, frame_idx: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Chiếu điểm world lên ảnh của camera theo đúng mô hình ống kính.

        ``DrivingDataset`` gọi hàm này (nếu camera có) để tô màu và lập depth map từ LiDAR, và để
        bỏ điểm LiDAR không camera nào thấy. Camera fisheye còn bỏ điểm rơi vào pixel ngoài vùng
        hợp lệ (``camera_rays``).

        Args:
            world_points: Điểm trong hệ world của cảnh, dạng ``(N, 3)``.
            frame_idx: Chỉ số khung.

        Returns:
            ``(pixels, depth, valid)``: toạ độ pixel ``(N, 2)`` ở kích thước ``load_size``, độ sâu
            ``z`` trong hệ camera ``(N,)``, và mask điểm nằm trong ảnh ``(N,)``.
        """
        c2w = self.cam_to_worlds[frame_idx]
        intrinsics = self.intrinsics[frame_idx]
        pixels, depth, valid = project_world_camera(
            world_points,
            c2w,
            c2w,
            intrinsics,
            self.WIDTH,
            self.HEIGHT,
            self.camera_model,
            self.radial_coeffs,
        )
        if self.camera_model == "fisheye":
            _, region = self.camera_rays(intrinsics, self.HEIGHT, self.WIDTH)
            selected = valid.nonzero().flatten()
            uv = pixels[selected].long()
            valid[selected] &= region[uv[:, 1], uv[:, 0]]
        return pixels, depth, valid


class KITTI360PixelSource(KITTIPixelSource):
    """Nguồn ảnh của cảnh KITTI-360: như KITTI, nhưng camera là ``KITTI360CameraData``.

    Hộp đối tượng (``instances/``) cùng định dạng KITTI nên dùng nguyên ``load_objects``; SMPL
    (nếu bật) được dựng theo pose của cả bốn camera.
    """

    CAMERA_DATA_CLASS = KITTI360CameraData
    SMPL_CAMERA_LIST = AVAILABLE_CAM_LIST
