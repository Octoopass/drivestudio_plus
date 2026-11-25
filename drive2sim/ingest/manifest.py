"""Sinh ``manifest.yaml`` từ một thư mục cảnh DriveStudio đã tiền xử lý.

Sau khi script tiền xử lý của một dataset ghi xong thư mục cảnh, ingest gọi file này để ghi
manifest: cảnh có những camera nào, độ phân giải, bao nhiêu khung, có LiDAR và box không
(CONTRACTS mục 2.2). Engine dựa vào manifest để khoá approach và chọn config dataset; Studio dựa
vào nó để hiện tổng quan cảnh.

File này chứa:
- ``frame_indices``: các chỉ số khung có ảnh của một camera.
- ``converter_version``: phiên bản converter ghi vào manifest (đổi theo mọi commit; thêm hậu tố
  ``.dirty-<hash8>`` khi mã converter có thay đổi chưa commit).
- ``DRIVESTUDIO_OUTPUT_VERSION``, ``KITTI360_OUTPUT_VERSION``: phiên bản đầu ra của từng
  converter, tăng tay khi dữ liệu converter ghi ra đổi; ``current_output_version`` tra theo tên
  converter.
- ``outdated_output_version``, ``outdated_message``: cảnh đã ingest bằng converter cũ chưa (đầu ra
  ghi trong manifest khác đầu ra hiện tại). Validate, Studio, ``run start``, ``approach check`` và
  ``kaggle ingest`` dùng để cảnh báo hoặc chọn notebook mới (CONTRACTS mục 2.2, 9.4).
- ``build_manifest``: dựng ``Manifest`` từ thư mục cảnh.

Mọi thông tin đọc từ chính các file trong thư mục cảnh (tên file ảnh, header ảnh, thư mục
``lidar/``, ``instances/``), không đọc lại dữ liệu gốc.
"""

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core.instances import human_frame_ranges
from drive2sim.core.schema import CAMERA_TRACK, SCHEMA_VERSIONS, Manifest, ManifestSource
from drive2sim.engine.run import code_commit
from drive2sim.ingest.images import IMAGE_SUFFIXES, read_image_size


def frame_indices(scene_dir: Path, camera_index: int) -> list[int]:
    """Liệt kê các chỉ số khung có ảnh của một camera.

    Ảnh DriveStudio đặt tên ``images/<khung 3 chữ số>_<chỉ số camera>.<đuôi>``.

    Args:
        scene_dir: Thư mục cảnh.
        camera_index: Chỉ số camera của DriveStudio.

    Returns:
        Các chỉ số khung, tăng dần, ví dụ ``[0, 1, 2]``.
    """
    indices = []
    for path in (scene_dir / "images").glob(f"*_{camera_index}.*"):
        stem_frame = path.stem.rsplit("_", 1)[0]
        if path.suffix.lower() in IMAGE_SUFFIXES and stem_frame.isdigit():
            indices.append(int(stem_frame))
    return sorted(indices)


# Mã quyết định dữ liệu converter ghi ra; sửa chưa commit ở đây thì ``converter_version`` thêm
# hậu tố ``.dirty-<hash8>``.
CONVERTER_CODE_PATHS = ("drive2sim/ingest", "drive2sim/recon/preprocess", "drive2sim/core")


def converter_version() -> str:
    """Trả về phiên bản converter dạng ``<phiên bản drive2sim>+<commit>``, ví dụ ``0.1.0+e567e29``.

    Phiên bản này nằm trong khoá cache của mọi stage nhận cảnh: ingest lại bằng code khác thì
    cache cũ không bị dùng nhầm (ARCHITECTURE mục 2.7). Khi mã converter có thay đổi chưa commit,
    chỉ commit thì không phân biệt được hai lần ingest bằng hai bản code khác nhau, nên thêm hậu
    tố ``.dirty-<hash8>`` băm từ phần thay đổi đó, ví dụ ``0.1.0+e567e29.dirty-1a2b3c4d``.
    """
    try:
        package = metadata.version("drive2sim")
    except metadata.PackageNotFoundError:
        package = "0.0.0"
    return f"{package}+{code_commit()}{_uncommitted_suffix()}"


def _uncommitted_suffix() -> str:
    """Trả về ``.dirty-<hash8>`` nếu mã converter có thay đổi chưa commit, ngược lại chuỗi rỗng.

    Băm gồm diff so với ``HEAD`` của các file đã theo dõi và nội dung các file mới chưa theo dõi
    trong ``CONVERTER_CODE_PATHS``. Không có git hoặc không phải repo (cài từ wheel) thì coi như
    không có thay đổi.

    Returns:
        Hậu tố, ví dụ ``".dirty-1a2b3c4d"``, hoặc ``""``.
    """
    repo = paths.PACKAGE_DIR.parent
    try:
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "HEAD", "--", *CONVERTER_CODE_PATHS],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *CONVERTER_CODE_PATHS,
            ],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    if not diff and not untracked:
        return ""
    digest = hashlib.sha256(diff)
    for name in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(name)
        try:
            digest.update((repo / name.decode()).read_bytes())
        except OSError:
            continue
    return f".dirty-{digest.hexdigest()[:8]}"


# Phiên bản đầu ra của converter (``source.output_version``, CONTRACTS mục 2.2). Tăng tay **chỉ
# khi dữ liệu converter ghi ra đổi** (file, toạ độ, intrinsics, mask...), không tăng khi chỉ sửa
# code mà đầu ra giữ nguyên. Cảnh ingest bằng phiên bản nhỏ hơn bị coi là cũ: validate, Studio,
# run start cảnh báo; kaggle ingest đẩy notebook mới thay vì dùng lại notebook cũ. Bảng nằm ở đây
# (không nằm trong từng file converter) vì converter và bộ kiểm tra cùng import file này; đặt
# trong kitti360.py thì validate.py phải import kitti360.py, mà kitti360.py lại import validate.py.
# Mọi converter ``drivestudio.<dataset>`` (script tiền xử lý của DriveStudio) dùng chung một số.
DRIVESTUDIO_OUTPUT_VERSION = 1
# 1: các bản trước 17/11/2026. 2: khớp Mei sang opencv_fisheye viết lại, intrinsics của image_02,
# image_03 đổi (commit 188fb9b, 17/11/2026). 3: manifest ghi cameras[].max_angle của camera mắt cá
# tính từ calib (góc mà OpenCV fisheye còn lệch Mei dưới 0,5 px, khoảng 96 độ, 19/11/2026); cảnh
# cũ không có trường này bị loader che từ 1,45 rad.
KITTI360_OUTPUT_VERSION = 3
# Tên converter KITTI-360 ghi trong manifest (``kitti360.CONVERTER``; không import để tránh vòng).
_KITTI360_CONVERTER = "kitti360"
# PHENIKAA (``phenikaa.CONVERTER``): 1 là bản đầu, 25/11/2026.
PHENIKAA_OUTPUT_VERSION = 1
_PHENIKAA_CONVERTER = "phenikaa"
# Manifest viết trước ``d2s-manifest/1.1`` không có ``output_version``: coi là bản đầu, trừ khi
# ``converter_version`` ghi một commit đã có đầu ra mới nhưng chưa ghi trường này. Commit 188fb9b
# (trên origin/drive2sim) đã viết lại khớp Mei; notebook ingest đẩy ở commit đó ra đầu ra 2.
_FIRST_OUTPUT_VERSION = 1
_UNRECORDED_OUTPUT_VERSIONS: dict[tuple[str, str], int] = {(_KITTI360_CONVERTER, "188fb9b"): 2}


def current_output_version(converter: str) -> Optional[int]:
    """Trả về phiên bản đầu ra hiện tại của một converter.

    Args:
        converter: Tên converter như trong manifest, ví dụ ``kitti360``,
            ``drivestudio.nuscenes``.

    Returns:
        Phiên bản đầu ra, ví dụ ``2``; ``None`` nếu converter không do tool quản lý (cảnh mẫu,
        cảnh tự làm), khi đó không xét cảnh cũ hay mới.
    """
    if converter == _KITTI360_CONVERTER:
        return KITTI360_OUTPUT_VERSION
    if converter == _PHENIKAA_CONVERTER:
        return PHENIKAA_OUTPUT_VERSION
    if converter.startswith("drivestudio."):
        return DRIVESTUDIO_OUTPUT_VERSION
    return None


def outdated_output_version(source: ManifestSource) -> Optional[tuple[int, int]]:
    """Cho biết cảnh có được ingest bằng converter cũ không.

    Cảnh cũ khi phiên bản đầu ra ghi trong manifest khác phiên bản hiện tại của converter đó.
    Manifest không ghi phiên bản (viết trước ``d2s-manifest/1.1``) được coi là phiên bản ``1``:
    converter chưa từng đổi đầu ra (DriveStudio) thì cảnh vẫn mới; converter đã tăng lên ``2``
    (KITTI-360) thì cảnh cũ. Ngoại lệ: commit trong ``converter_version`` nằm trong
    ``_UNRECORDED_OUTPUT_VERSIONS`` (đã có đầu ra mới nhưng chưa ghi trường này) thì lấy phiên bản
    ở đó.

    Args:
        source: Phần ``source`` của manifest.

    Returns:
        ``(phiên bản trong manifest, phiên bản hiện tại)`` nếu cảnh cũ, ví dụ ``(1, 2)``;
        ``None`` nếu cảnh mới hoặc converter không có phiên bản đầu ra.
    """
    current = current_output_version(source.converter)
    commit = source.converter_version.rsplit("+", 1)[-1][:7]
    recorded = source.output_version or _UNRECORDED_OUTPUT_VERSIONS.get(
        (source.converter, commit), _FIRST_OUTPUT_VERSION
    )
    if current is None or recorded == current:
        return None
    return recorded, current


def outdated_message(manifest: Manifest) -> Optional[str]:
    """Câu cảnh báo khi cảnh được ingest bằng converter cũ; ``None`` nếu cảnh mới.

    Chỉ cảnh báo, không chặn: run đã chạy trên cảnh cũ vẫn xem, chạy tiếp được.

    Returns:
        Ví dụ ``"cảnh kitti360_... được ingest bằng converter kitti360 cũ (đầu ra phiên bản 1,
        hiện tại 2); ingest lại để dùng đầu ra mới"``.
    """
    outdated = outdated_output_version(manifest.source)
    if outdated is None:
        return None
    return (
        f"cảnh {manifest.scene_id} được ingest bằng converter {manifest.source.converter} cũ "
        f"(đầu ra phiên bản {outdated[0]}, hiện tại {outdated[1]}); ingest lại để dùng đầu ra mới"
    )


def build_manifest(
    scene_dir: Path,
    scene_id: str,
    dataset: str,
    sequence_id: str,
    camera_ids: list[str],
    converter: str,
    lidar_ids: Optional[list[str]] = None,
    data_license: Optional[str] = None,
    camera_model: str = "opencv_pinhole",
    camera_models: Optional[list[str]] = None,
    camera_max_angles: Optional[list[Optional[float]]] = None,
) -> Manifest:
    """Dựng manifest cho một thư mục cảnh DriveStudio.

    Số khung lấy theo camera đầu tiên. Độ phân giải đọc từ ảnh đầu tiên của từng camera. Cảnh có
    LiDAR khi ``lidar/`` có file ``.bin``; có box khi ``instances/instances_info.json`` không rỗng.
    ``available.human_frames`` là các đoạn khung có box người (``human_frame_ranges``), ``[]``
    khi không có ``instances/``; nhãn có nhưng hỏng thì giữ ``None``. Lúc ingest, thư mục cảnh
    là nguồn đầy đủ nên thiếu thư mục nhãn nghĩa là không có box. Ghi vào manifest để cảnh chỉ
    có trên Kaggle cũng biết split nào không cần dựng SMPL.
    ``source.output_version`` ghi phiên bản đầu ra hiện tại của ``converter``
    (``current_output_version``). Mọi camera ghi cùng mô hình ``camera_model`` và shutter
    ``GLOBAL``; track của cảnh suy ra từ mô hình (``CAMERA_TRACK``): ``opencv_pinhole`` là
    track pinhole, ``ftheta`` (PhysicalAI) là track fisheye. Cảnh có nhiều loại camera
    (KITTI-360: hai pinhole, hai mắt cá) truyền ``camera_models`` theo từng camera; khi đó track
    là ``fisheye`` nếu có ít nhất một camera thuộc track fisheye (CONTRACTS mục 2.2).

    Args:
        scene_dir: Thư mục cảnh đã tiền xử lý.
        scene_id: Định danh cảnh, ví dụ ``nuscenes_scene-0061``.
        dataset: Dataset gốc, ví dụ ``nuscenes``.
        sequence_id: Id của chuỗi trong nguồn, ví dụ ``scene-0061``.
        camera_ids: Tên camera theo đúng thứ tự chỉ số camera của DriveStudio cho dataset đó.
        converter: Tên converter, ví dụ ``drivestudio.nuscenes``.
        lidar_ids: Tên các LiDAR nếu cảnh có LiDAR, ví dụ ``["LIDAR_TOP"]``.
        data_license: License của dữ liệu nguồn.
        camera_model: Mô hình ống kính của ảnh trong ``images/``, chung cho mọi camera.
        camera_models: Mô hình của từng camera, cùng thứ tự với ``camera_ids``; có thì thay cho
            ``camera_model``.
        camera_max_angles: Góc lệch trục lớn nhất (rad) mà mô hình của từng camera còn đúng,
            cùng thứ tự với ``camera_ids`` (``ManifestCamera.max_angle``); converter chuyển gần
            đúng mô hình ống kính của dataset thì khai báo, ``None`` là không giới hạn.

    Returns:
        Manifest đã kiểm tra theo schema.

    Raises:
        ValueError: Nếu camera đầu tiên không có ảnh nào, ảnh đầu của một camera không đọc
            được kích thước, hoặc ``camera_models`` khác số camera.
    """
    models = list(camera_models) if camera_models else [camera_model] * len(camera_ids)
    if len(models) != len(camera_ids):
        raise ValueError("camera_models phải có đúng một mô hình cho mỗi camera")
    max_angles = list(camera_max_angles) if camera_max_angles else [None] * len(camera_ids)
    if len(max_angles) != len(camera_ids):
        raise ValueError("camera_max_angles phải có đúng một giá trị cho mỗi camera")
    frames = frame_indices(scene_dir, 0)
    if not frames:
        raise ValueError(f"{scene_dir}: không có ảnh nào của camera 0 trong images/")
    cameras = []
    for index, camera_id in enumerate(camera_ids):
        images = sorted((scene_dir / "images").glob(f"{frames[0]:03d}_{index}.*"))
        size = read_image_size(images[0]) if images else None
        if size is None:
            raise ValueError(f"{scene_dir}: không đọc được ảnh khung {frames[0]} của {camera_id}")
        cameras.append(
            {
                "camera_id": camera_id,
                "model": models[index],
                "resolution": list(size),
                "shutter": "GLOBAL",
                "max_angle": max_angles[index],
            }
        )
    has_lidar = any((scene_dir / "lidar").glob("*.bin"))
    instances_info = scene_dir / "instances" / "instances_info.json"
    has_cuboids = False
    if instances_info.is_file():
        has_cuboids = bool(json.loads(instances_info.read_text(encoding="utf-8")))
    lidars = list(lidar_ids or []) if has_lidar else []
    return Manifest.model_validate(
        {
            "schema_version": SCHEMA_VERSIONS["manifest"],
            "scene_id": scene_id,
            "source": {
                "dataset": dataset,
                "sequence_id": sequence_id,
                "converter": converter,
                "converter_version": converter_version(),
                "output_version": current_output_version(converter),
                "data_license": data_license,
            },
            "track": (
                "fisheye" if any(CAMERA_TRACK[m] == "fisheye" for m in models) else "pinhole"
            ),
            "cameras": cameras,
            "lidars": lidars,
            "timeline": {"reference_sensor": camera_ids[0], "num_frames": frames[-1] + 1},
            "available": {
                "lidar": bool(lidars),
                "cuboids": "gt" if has_cuboids else "none",
                "human_frames": (
                    human_frame_ranges(scene_dir) if (scene_dir / "instances").exists() else []
                ),
            },
        }
    )
