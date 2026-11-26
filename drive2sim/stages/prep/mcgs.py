"""Stage ``prep.mcgs``: xuất ảnh pinhole và ``calib.yml`` làm đầu vào cho MCGS-SLAM.

MCGS-SLAM đọc mỗi camera từ một thư mục ảnh, ghép các thư mục theo vị trí trong danh sách và theo
số cuối cùng trong tên file (xem ``drive2sim/external/mcgs/calib.py``). Stage này dựng đúng bố
cục đó từ cảnh chuẩn:
- chọn camera pinhole của split, bỏ camera chỉ test (``Split.test_only_cameras``) và camera mắt cá
  (SLAM không nhận);
- mặc định bỏ khung test khỏi mọi thư mục (``held_out``), để điểm đo của ``eval.mcgs`` là điểm trên
  ảnh SLAM chưa thấy;
- bỏ khung thiếu ảnh ở bất kỳ camera nào ở mọi thư mục, vì MCGS-SLAM đòi mọi thư mục cùng số ảnh;
- ghi ``calib.yml`` từ intrinsics, extrinsics của cảnh và ``mcgs_prep.json`` ghi lại các lựa chọn
  trên cho ``train.mcgs`` và ``eval.mcgs``.

File này chứa:
- ``PrepMcgsParams``: tham số.
- ``select_cameras``: chọn camera đưa vào SLAM.
- ``PrepMcgsStage``: stage đăng ký với engine; kết quả là thư mục ``mcgs_input``.

``train.mcgs`` đọc thư mục ra của stage này. Xem ARCHITECTURE mục 2.19, CONTRACTS mục 5.4.
Chưa chạy thật cùng MCGS-SLAM (chiều của ``baseline`` kiểm ở bước chạy thật).
"""

import json
import os
import shutil
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external.mcgs import PINHOLE_MODELS

CALIB_FILE = "calib.yml"
PREP_FILE = "mcgs_prep.json"
IMAGES_DIR = "images"
MIN_CAMERAS = 2  # MCGS-SLAM luôn chạy stereo (giống ``calib.MIN_CAMERAS``, nhưng không cần numpy)
PINHOLE_VALUES = 9  # intrinsics/<chỉ số>.txt của camera pinhole: fx fy cx cy k1 k2 p1 p2 k3


class PrepMcgsParams(BaseModel):
    """Tham số của ``prep.mcgs``.

    Attributes:
        cameras: ``camera_id`` đưa vào SLAM; ``None`` thì dùng mọi camera pinhole của split trừ
            camera chỉ test.
        reference_camera: Camera 0 của SLAM; ``None`` thì camera đầu của danh sách.
        stereo_partner: Camera đối tác stereo; ``None`` thì camera gần ``reference_camera`` nhất.
        held_out: Bỏ khung test khỏi đầu vào của SLAM.
    """

    model_config = ConfigDict(extra="forbid")

    cameras: Optional[list[str]] = None
    reference_camera: Optional[str] = None
    stereo_partner: Optional[str] = None
    held_out: bool = True


def select_cameras(params: PrepMcgsParams, manifest: Manifest, split: Split) -> list[str]:
    """Chọn camera đưa vào SLAM và kiểm tra rig dùng được.

    Args:
        params: Tham số của stage.
        manifest: Manifest của cảnh.
        split: Split của run.

    Returns:
        ``camera_id`` theo thứ tự ưu tiên (camera đầu là tham chiếu mặc định).

    Raises:
        ValueError: Nếu một camera trong tham số không có trong split, là camera chỉ test, không
            phải pinhole, hoặc rig có dưới hai camera.
    """
    from drive2sim.external.mcgs import calib  # cần numpy: nạp khi stage chạy

    models = {camera.camera_id: camera.model for camera in manifest.cameras}
    if params.cameras is None:
        chosen = [
            camera
            for camera in split.cameras
            if camera not in split.test_only_cameras and models[camera] in PINHOLE_MODELS
        ]
    else:
        unknown = [camera for camera in params.cameras if camera not in split.cameras]
        if unknown:
            raise ValueError(
                f"camera {', '.join(unknown)} không có trong split ({', '.join(split.cameras)})"
            )
        test_only = [camera for camera in params.cameras if camera in split.test_only_cameras]
        if test_only:
            raise ValueError(
                f"camera {', '.join(test_only)} là camera chỉ test của split: không được đưa "
                "vào đầu vào SLAM (sẽ làm mất ý nghĩa của phép đo trên camera chưa thấy)"
            )
        chosen = list(params.cameras)
    calib.check_cameras({camera: models[camera] for camera in chosen})
    return chosen


def find_image(scene_root: Path, frame: int, camera_index: int) -> Optional[Path]:
    """Ảnh ``images/<khung:03d>_<chỉ số camera>.<đuôi>`` của cảnh; ``None`` nếu không có."""
    found = sorted((scene_root / IMAGES_DIR).glob(f"{frame:03d}_{camera_index}.*"))
    return found[0] if found else None


def _link_or_copy(source: Path, target: Path) -> None:
    """Symlink ``target`` tới ``source`` (đường dẫn tuyệt đối); không tạo được thì chép file."""
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copyfile(source, target)


def _read_pinhole_values(scene_root: Path, camera_index: int) -> list[float]:
    """Chín số ``fx fy cx cy k1 k2 p1 p2 k3`` của ``intrinsics/<chỉ số>.txt``.

    Raises:
        ValueError: Nếu file không có đúng chín số.
    """
    import numpy as np  # nạp khi stage chạy, để d2s nạp được stage mà không cần numpy

    values = np.atleast_1d(np.loadtxt(scene_root / "intrinsics" / f"{camera_index}.txt"))
    if values.size != PINHOLE_VALUES:
        raise ValueError(f"camera pinhole {camera_index}: cần 9 số, có {values.size}")
    return [float(v) for v in values]


@register_stage("prep.mcgs")
class PrepMcgsStage(Stage):
    """Xuất ảnh pinhole (không có khung test) và ``calib.yml`` cho MCGS-SLAM."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"split": "split"}
    outputs: ClassVar[dict[str, str]] = {"data": "mcgs_input"}
    Params: ClassVar[type[BaseModel]] = PrepMcgsParams

    @classmethod
    def unavailable_reason(
        cls, manifest: Manifest, split: Optional[Split] = None
    ) -> Optional[str]:
        """Khoá approach khi split không có đủ hai camera pinhole đã train (một cặp stereo).

        Kiểm theo split thay vì chờ ``run`` báo lỗi, để Studio và ``d2s run start`` từ chối trước
        khi tốn thời gian; không cần numpy nên gọi được ở môi trường nhẹ.
        """
        if split is None:
            return None
        models = {camera.camera_id: camera.model for camera in manifest.cameras}
        usable = [
            camera
            for camera in split.cameras
            if camera not in split.test_only_cameras and models[camera] in PINHOLE_MODELS
        ]
        if len(usable) < MIN_CAMERAS:
            return (
                f"MCGS-SLAM cần ít nhất {MIN_CAMERAS} camera pinhole đã train (một cặp stereo); "
                f"split {split.split_id} có {len(usable)}"
            )
        return None

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chọn camera và khung, xuất ảnh, ghi ``calib.yml`` và ``mcgs_prep.json``.

        Returns:
            ``{"data": <thư mục stage>}``.

        Raises:
            ValueError: Camera không dùng được cho SLAM (``select_cameras``), hoặc
                ``reference_camera``, ``stereo_partner`` không hợp lệ.
            RuntimeError: Không còn khung nào có đủ ảnh của mọi camera.
        """
        assert isinstance(params, PrepMcgsParams)
        from drive2sim.external.mcgs import calib  # cần numpy: nạp khi stage chạy
        from drive2sim.external.mcgs.views import read_matrix

        manifest = ctx.scene.manifest
        root = ctx.scene.root
        cameras = select_cameras(params, manifest, ctx.split)
        index_of = {camera.camera_id: i for i, camera in enumerate(manifest.cameras)}
        cam_to_ego = {
            camera: read_matrix(root / "extrinsics" / f"{index_of[camera]}.txt")
            for camera in cameras
        }
        reference = params.reference_camera or cameras[0]
        layout = calib.rig_layout(cameras, cam_to_ego, reference, params.stereo_partner)

        # Khung test của camera đã train: camera chỉ test không vào SLAM nên khung của nó không
        # làm mất khung train của các camera khác.
        test_frames = {
            frame for frame, camera in ctx.split.test if camera not in ctx.split.test_only_cameras
        }
        wanted = calib.slam_frames(ctx.split.frame_range, test_frames, params.held_out)
        used = sorted(set(layout))
        kept: list[int] = []
        dropped: list[int] = []
        sources: dict[int, dict[str, Path]] = {}
        for frame in wanted:
            images = {camera: find_image(root, frame, index_of[camera]) for camera in used}
            if all(path is not None for path in images.values()):
                kept.append(frame)
                sources[frame] = {camera: path for camera, path in images.items() if path}
            else:
                dropped.append(frame)
        if not kept:
            raise RuntimeError("không khung nào có đủ ảnh của mọi camera đưa vào MCGS-SLAM")

        ctx.progress(0.1, f"xuất {len(kept)} khung, {len(used)} camera")
        for camera in used:
            directory = ctx.output_dir / IMAGES_DIR / camera
            directory.mkdir(parents=True)
            for frame in kept:
                source = sources[frame][camera]
                _link_or_copy(source, directory / f"{frame:06d}{source.suffix}")
        intrinsics = {c: _read_pinhole_values(root, index_of[c]) for c in used}
        document = calib.calib_document(layout, intrinsics, cam_to_ego)
        (ctx.output_dir / CALIB_FILE).write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=None), encoding="utf-8"
        )
        (ctx.output_dir / PREP_FILE).write_text(
            json.dumps(
                {
                    "layout": layout,
                    "cameras": cameras,
                    "reference_camera": reference,
                    "frames": kept,
                    "dropped_frames": dropped,
                    "held_out": params.held_out,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        ctx.progress(1.0, f"bỏ {len(dropped)} khung thiếu ảnh" if dropped else "xong")
        return {"data": ctx.output_dir}
