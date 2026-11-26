"""Loader chung cho mọi cảnh ghi theo định dạng chuẩn của drive2sim (camera gắn cố định trên xe).

Định dạng chuẩn (CONTRACTS mục 2; converter của KITTI-360, PHENIKAA và dataset mới ghi ra):

- ``images/<khung>_<chỉ số>.<jpg|png>``: đuôi ảnh tự nhận từ file có thật của từng camera;
- ``intrinsics/<chỉ số>.txt``: pinhole 9 số ``fx fy cx cy k1 k2 p1 p2 k3``; mắt cá 8 số
  ``fx fy cx cy k1 k2 k3 k4`` (``cv2.fisheye`` / Kannala–Brandt);
- ``extrinsics/<chỉ số>.txt``: camera → xe (4×4, cố định); ``ego_pose/<khung>.txt``: xe → world;
- ``lidar/<khung>.bin``: ``float32`` x, y, z, cường độ trong hệ xe;
- tuỳ chọn: ``instances/``, ``dynamic_masks/``, ``sky_masks/``, ``ego_masks/<camera>.png``;
- ``manifest.yaml``: tên, mô hình ống kính, độ phân giải, ``max_angle`` của từng camera.

Mọi điều khác nhau giữa các dataset đều đọc từ manifest và file trong cảnh, không viết cứng theo
dataset, nên dataset mới chỉ cần converter. Loader kế thừa loader KITTI của DriveStudio
(``loaders/kitti.py``, cùng khung thư mục) và thêm:

- Mô hình và độ phân giải của từng camera đọc từ ``manifest.yaml`` (cảnh ảo của stage train có
  symlink tới file này, ``drive2sim/stages/scene_view.py``).
- Camera mắt cá giữ nguyên ảnh gốc: tia của từng pixel tính bằng ``camera_model_rays`` với mô hình
  ``fisheye`` và ``radial_coeffs``; ``cam_infos`` mang ``camera_model``, ``radial_coeffs`` để
  trainer (``drive2sim/recon/trainers/base.py``) gọi renderer UT hoặc GEER. Pixel có tia lệch
  trục quá ``max_angle`` (góc nhỏ nhất giữa vùng ảnh còn nội dung, mô hình ống kính còn đúng,
  ``cameras[].max_angle`` của manifest, và giới hạn của renderer: 3DGUT 88 độ) bị đánh dấu không
  hợp lệ (``egocar_masks``, ``valid_masks``), như ``invalid_masks`` của PhysicalAI.
- Mask thân xe: ``ego_masks/<camera>.png`` của cảnh, nếu không có thì asset của tool cho dataset
  (``EGO_MASKS_DIR/<dataset>/``, trừ dataset mà converter đã chép mask vào cảnh,
  ``ingest.ego_masks.COPIED_BY_CONVERTER``). Thân xe bị bỏ khỏi loss, chấm điểm và chiếu LiDAR,
  với cả camera pinhole lẫn mắt cá.
- SMPL (nếu bật) dựng theo pose của mọi camera có trong manifest.

File này chứa:
- ``StandardCameraData``: một camera (pinhole hoặc mắt cá).
- ``StandardPixelSource``: nguồn ảnh của cảnh, dùng ``StandardCameraData`` cho mọi camera.

LiDAR dùng nguyên ``KITTILiDARSource``. Config dataset (``drive2sim/configs/datasets/<dataset>/``)
trỏ tới ``StandardPixelSource``; stage ``train.drivestudio`` chọn config theo manifest. Tên cũ
``loaders.kitti360.KITTI360PixelSource``, ``loaders.phenikaa.PhenikaaPixelSource`` trong config của
run cũ được đổi sang lớp này (``recon/utils/misc.py``, ``LEGACY_CLASSES``).

Quy ước pixel: giống loader KITTI, ``cx``, ``cy`` trong file intrinsics được dùng nguyên (không
cộng 0,5); tâm pixel ``i`` nằm ở toạ độ ``i + 0.5``.
"""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE
from drive2sim.core.files import load
from drive2sim.core.schema import Manifest
from drive2sim.ingest import ego_masks
from drive2sim.recon.data.base.pixel_source import scene_ego_mask
from drive2sim.recon.data.loaders.kitti import KITTICameraData, KITTIPixelSource
from drive2sim.recon.utils.geometry import (
    camera_model_rays,
    dark_border_angle,
    project_world_camera,
)

MANIFEST_FILE = "manifest.yaml"
# Mô hình camera trong manifest → mô hình camera trainer DriveStudio nhận (recon/trainers/base.py).
TRAINER_CAMERA_MODELS = {"opencv_pinhole": "pinhole", "opencv_fisheye": "fisheye"}
IMAGE_SUFFIXES = (".jpg", ".png", ".jpeg")


class StandardCameraData(KITTICameraData):
    """Một camera của cảnh định dạng chuẩn, đọc mô hình ống kính từ manifest.

    ``StandardPixelSource`` tạo một đối tượng cho mỗi camera được chọn. Camera pinhole đi đúng
    đường của loader KITTI (tia pinhole của ``CameraData.get_image``); camera fisheye có tia,
    mask vùng hợp lệ và phép chiếu điểm riêng.

    Attributes:
        camera_model: Mô hình camera của trainer, ``"pinhole"`` hoặc ``"fisheye"``.
        radial_coeffs: Bốn hệ số ``k1..k4`` của OpenCV fisheye (không có đơn vị nên không đổi khi
            thu nhỏ ảnh), dạng tensor ``(4,)``; ``None`` với camera pinhole.
    """

    def __init__(
        self,
        fisheye_max_angle: Optional[float] = None,
        fisheye_renderer_max_angle: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        """Đọc mô hình và độ phân giải của camera từ manifest rồi nạp dữ liệu như KITTI.

        Args:
            fisheye_max_angle: Góc lệch trục lớn nhất (rad) của pixel được dùng, thay cho giới
                hạn của mô hình ống kính và của renderer (``data.pixel_source.fisheye_max_angle``,
                để thử ngưỡng khác); vùng đen dò từ ảnh vẫn bị che.
            fisheye_renderer_max_angle: Giới hạn của renderer (``data.pixel_source.
                fisheye_renderer_max_angle``, stage train đặt cho 3DGUT); ``None`` là không có.
            **kwargs: Tham số của ``CameraData`` (``data_path``, ``cam_id``, ...); ``cam_id`` là
                chỉ số camera trong cảnh, cũng là vị trí của camera trong ``manifest.cameras``.

        Raises:
            ValueError: Nếu manifest không đọc được, chỉ số camera vượt số camera của manifest,
                hoặc mô hình camera không phải ``opencv_pinhole``, ``opencv_fisheye``.
        """
        scene = Path(str(kwargs["data_path"]))
        manifest = load(scene / MANIFEST_FILE, Manifest)
        index = int(kwargs["cam_id"])
        if not 0 <= index < len(manifest.cameras):
            raise ValueError(
                f"manifest có {len(manifest.cameras)} camera, không có chỉ số {index}"
            )
        camera = manifest.cameras[index]
        if camera.model not in TRAINER_CAMERA_MODELS:
            raise ValueError(f"{camera.camera_id}: loader chuẩn không đọc được {camera.model}")
        self.camera_model = TRAINER_CAMERA_MODELS[camera.model]
        self.radial_coeffs = None
        # Ngưỡng che theo góc = nhỏ nhất của: góc mô hình ống kính còn đúng (cameras[].max_angle,
        # converter tính từ calib của từng camera; cảnh ingest trước khi manifest có trường này
        # dùng miền khớp MEI_FIT_MAX_ANGLE), giới hạn của renderer, và góc bắt đầu vùng đen dò từ
        # ảnh (sau khi nạp ảnh, bên dưới). Tham số thử nghiệm thay hai giới hạn đầu.
        lens_limit = camera.max_angle if camera.max_angle is not None else MEI_FIT_MAX_ANGLE
        if fisheye_max_angle is not None:
            self.max_angle = float(fisheye_max_angle)
        elif fisheye_renderer_max_angle is not None:
            self.max_angle = float(min(lens_limit, fisheye_renderer_max_angle))
        else:
            self.max_angle = float(lens_limit)
        self._ray_cache = {}
        width, height = camera.resolution
        # CameraData đọc camera_metadata thay cho DATASETS_CONFIG; original_size là (cao, rộng).
        self.camera_metadata = {
            "camera_name": camera.camera_id,
            "original_size": (height, width),
            "egocar_visible": ego_masks.has_mask(
                scene, manifest.source.dataset, camera.camera_id, index
            ),
        }
        # Asset mask của tool (khi cảnh không có ego_masks/<camera>.png), theo dataset của cảnh.
        self.ego_mask_asset = ego_masks.tool_asset(
            manifest.source.dataset, camera.camera_id, index
        )
        super().__init__(**kwargs)
        self.dark_angle = None
        if self.camera_model == "fisheye" and self.images is not None:
            rays, valid = camera_model_rays(
                self.HEIGHT,
                self.WIDTH,
                self.intrinsics[0],
                "fisheye",
                self.radial_coeffs,
                fisheye_max_angle=math.pi,
            )
            theta = torch.acos(rays[..., 2].clamp(-1, 1))
            self.dark_angle = dark_border_angle(self.images, theta, valid)
            self.max_angle = min(self.max_angle, self.dark_angle)
            self._ray_cache.clear()

    def load_egocar_mask(self) -> None:
        """Nạp mask thân xe cố định theo tên camera trong manifest.

        Thứ tự: mask đi kèm cảnh (``<cảnh>/ego_masks/<camera>.png``, converter chép từ dataset
        hoặc người dùng vẽ trong Studio, cùng không gian ảnh với ``images/``), rồi asset của tool
        cho dataset của cảnh (``ingest.ego_masks.tool_asset``). Converter
        đánh lại chỉ số khi ingest một phần camera, nên ``cam_id`` không dùng để chọn mask. Trắng:
        thân xe, đen: giữ. Nội suy nearest giữ mask nhị phân khi giảm độ phân giải. Không dò
        mask từ ảnh của run, nên vệt sáng hay bóng đổ không làm thay đổi vùng được giám sát.

        Raises:
            ValueError: Nếu mask của cảnh khác kích thước ảnh.
        """
        self.egocar_mask = None
        mask = scene_ego_mask(self.data_path, self.cam_name, self.original_size)
        if mask is None:
            if self.ego_mask_asset is None:
                return
            mask = Image.open(self.ego_mask_asset).convert("L")
        resized = mask.resize((self.load_size[1], self.load_size[0]), Image.Resampling.NEAREST)
        self.egocar_mask = torch.from_numpy(np.array(resized) > 0).float()

    def create_all_filelist(self) -> None:
        """Lập danh sách file như KITTI, đuôi ảnh lấy theo file có thật của camera ở khung đầu.

        Raises:
            FileNotFoundError: Nếu khung đầu của camera không có ảnh đuôi nào trong
                ``IMAGE_SUFFIXES``.
        """
        super().create_all_filelist()
        first = Path(self.img_filepaths[0])
        suffix = next((s for s in IMAGE_SUFFIXES if first.with_suffix(s).is_file()), None)
        if suffix is None:
            raise FileNotFoundError(f"không có ảnh {first.with_suffix('.*')}")
        self.img_filepaths = np.array(
            [str(Path(path).with_suffix(suffix)) for path in self.img_filepaths]
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
                f"{self.cam_name}: camera fisheye chỉ train trên ảnh gốc; "
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

    def to(self, device: torch.device) -> "StandardCameraData":
        """Chuyển dữ liệu của camera sang ``device``, gồm cả ``radial_coeffs``; xoá cache tia."""
        super().to(device)
        self.device = device
        if self.radial_coeffs is not None:
            self.radial_coeffs = self.radial_coeffs.to(device)
        self._ray_cache.clear()
        return self

    def camera_rays(self, intrinsics: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        """Tính tia đơn vị trong hệ camera của mọi pixel fisheye và mask pixel hợp lệ.

        Tia lấy từ ``camera_model_rays`` (giải ngược
        ``theta_d = theta·(1 + k1·θ² + ... + k4·θ⁸)``).
        Pixel hợp lệ khi giải ngược hội tụ và góc tia ``theta`` không vượt ``self.max_angle``
        (``__init__``). Kết quả được nhớ theo
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
                height,
                width,
                intrinsics,
                "fisheye",
                self.radial_coeffs,
                fisheye_max_angle=self.max_angle,
            )
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
            # CameraData.get_image đã đặt valid_masks = ngoài thân xe nếu có mask
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
        hợp lệ (``camera_rays``) hoặc bị thân xe che, trước khi tô màu và khởi tạo từ LiDAR.

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
            max_angle=self.max_angle if self.camera_model == "fisheye" else None,
        )
        region = None
        if self.camera_model == "fisheye":
            _, region = self.camera_rays(intrinsics, self.HEIGHT, self.WIDTH)
        if self.egocar_mask is not None:
            egocar = ~self.egocar_mask.bool().to(valid.device)
            region = egocar if region is None else region & egocar
        if region is not None:
            selected = valid.nonzero().flatten()
            uv = pixels[selected].long()
            valid[selected] &= region[uv[:, 1], uv[:, 0]]
        return pixels, depth, valid


class StandardPixelSource(KITTIPixelSource):
    """Nguồn ảnh của cảnh định dạng chuẩn: như KITTI, nhưng camera là ``StandardCameraData``.

    Hộp đối tượng (``instances/``) cùng định dạng KITTI nên dùng nguyên ``load_objects``; SMPL
    (nếu bật) được dựng theo pose của mọi camera trong manifest.
    """

    CAMERA_DATA_CLASS = StandardCameraData

    def __init__(  # noqa: D107
        self,
        dataset_name: str,
        pixel_data_config,  # noqa: ANN001
        data_path: str,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Lấy danh sách camera dựng SMPL từ manifest rồi nạp như ``KITTIPixelSource``."""
        manifest = load(Path(str(data_path)) / MANIFEST_FILE, Manifest)
        self.SMPL_CAMERA_LIST = list(range(len(manifest.cameras)))
        super().__init__(dataset_name, pixel_data_config, data_path, *args, **kwargs)

    def extra_camera_kwargs(self) -> dict:
        """Ngưỡng góc của camera mắt cá đặt trong config (``fisheye_max_angle``), nếu có."""
        return {
            "fisheye_max_angle": self.data_cfg.get("fisheye_max_angle", None),
            "fisheye_renderer_max_angle": self.data_cfg.get("fisheye_renderer_max_angle", None),
        }
