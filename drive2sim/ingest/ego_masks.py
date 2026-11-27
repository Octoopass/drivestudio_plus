"""Trạng thái mask thân xe của từng camera trong một cảnh: có mask, không thấy thân xe, chưa xử lý.

Camera nào thấy thân xe (capo, gương, cửa) thì cần mask để thân xe không bị dựng lẫn vào cảnh và
không bị tính điểm. Tool phải phân biệt ba trường hợp, vì "không có file mask" có thể là "camera
không thấy thân xe" hoặc "chưa ai xem":

- ``mask``: camera có mask. Nguồn (``source``): ``scene`` (``ego_masks/<camera>.png`` trong cảnh:
  converter chép từ dataset hoặc người dùng vẽ trong Studio), ``tool`` (asset kèm tool cho bộ
  công khai, ``EGO_MASKS_DIR/<dataset>/``), ``dataset`` (mask theo khung của chính dataset, như
  ``invalid_masks/`` của PhysicalAI).
- ``none``: camera không thấy thân xe. Nguồn: ``confirmed`` (ghi trong ``status.json``: người dùng
  xác nhận trong Studio, hoặc converter đã biết) hoặc ``tool`` (bộ công khai mà DriveStudio ghi
  ``egocar_visible: False`` cho camera này, hoặc camera trong ``KNOWN_WITHOUT_BODY``).
- ``pending``: chưa ai xác nhận. Vẫn train được, nhưng nếu camera thấy thân xe thì thân xe bị dựng
  lẫn vào cảnh; kiểm tra cảnh và trang tạo run cảnh báo.

Lựa chọn của người dùng ghi ở ``ego_masks/status.json`` (``{camera_id: "mask" | "none"}``); camera
không có trong file thì suy ra như trên. Loader đọc trạng thái none để bỏ asset mặc định
khi người dùng đã xác nhận.

File này chứa ``STATUS_FILE``, ``KNOWN_WITHOUT_BODY``, ``tool_asset``, ``has_mask``,
``camera_states``, ``set_state``, ``pending_cameras``, ``pending_warning``, ``effective_mask``,
``mask_digest`` và ``snapshot_masks``. Engine dùng digest/snapshot khi tạo và chạy run.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core.schema import Manifest

STATUS_FILE = Path("ego_masks") / "status.json"
# Camera không thấy thân xe của các bộ mà DATASETS_CONFIG của DriveStudio không có.
KNOWN_WITHOUT_BODY = {"kitti360": ("image_00", "image_01")}
# Tên dataset trong manifest → tên thư mục asset và khoá của DATASETS_CONFIG (DriveStudio).
_DRIVESTUDIO_NAMES = {"argoverse2": "argoverse"}


def _read_status(scene_root: Path) -> dict[str, str]:
    path = scene_root / STATUS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if v in ("mask", "none")}


# Dataset mà converter chép mask của tool vào cảnh; loader của chúng không đọc asset của tool
# (``StandardCameraData`` dùng ``tool_asset``), nên asset không tính là mask của cảnh.
COPIED_BY_CONVERTER = ("phenikaa",)


def tool_asset(dataset: str, camera_id: str, index: int) -> Optional[Path]:
    """Asset mask mà loader của dataset đọc trực tiếp (như ``CameraData.load_egocar_mask``)."""
    if dataset in COPIED_BY_CONVERTER:
        return None
    folder = paths.EGO_MASKS_DIR / _DRIVESTUDIO_NAMES.get(dataset, dataset)
    for name in (camera_id, str(index)):
        if (folder / f"{name}.png").is_file():
            return folder / f"{name}.png"
    return None


def _known_without_body(dataset: str, camera_id: str, index: int) -> bool:
    """Bộ công khai mà tool biết camera này không thấy thân xe."""
    from drive2sim.recon.data.dataset_meta import DATASETS_CONFIG

    if camera_id in KNOWN_WITHOUT_BODY.get(dataset, ()):
        return True
    cameras = DATASETS_CONFIG.get(_DRIVESTUDIO_NAMES.get(dataset, dataset))
    return bool(cameras) and index in cameras and not cameras[index]["egocar_visible"]


def has_mask(scene_root: Path, dataset: str, camera_id: str, index: int) -> bool:
    """Camera có mask mà loader dùng: trong cảnh, mask theo khung của dataset, asset của tool."""
    return effective_mask(scene_root, dataset, camera_id, index) is not None or (
        dataset == "physicalai" and (scene_root / "invalid_masks").is_dir()
    )


def camera_states(scene_root: Path, manifest: Manifest) -> list[dict[str, Optional[str]]]:
    """Trạng thái mask thân xe của mọi camera trong cảnh.

    Returns:
        Mỗi camera ``{"camera_id", "state": "mask" | "none" | "pending", "source"}``; ``source``
        là ``None`` khi ``pending``.
    """
    status = _read_status(scene_root)
    dataset = manifest.source.dataset
    states = []
    for index, camera in enumerate(manifest.cameras):
        camera_id = camera.camera_id
        has_scene_mask = (scene_root / "ego_masks" / f"{camera_id}.png").is_file()
        state, source = "pending", None
        if has_scene_mask:
            state, source = "mask", "scene"
        elif status.get(camera_id) == "none":
            state, source = "none", "confirmed"
        elif dataset == "physicalai" and (scene_root / "invalid_masks").is_dir():
            state, source = "mask", "dataset"
        elif tool_asset(dataset, camera_id, index) is not None:
            state, source = "mask", "tool"
        elif _known_without_body(dataset, camera_id, index):
            state, source = "none", "tool"
        states.append({"camera_id": camera_id, "state": state, "source": source})
    return states


def set_state(scene_root: Path, camera_id: str, state: Optional[str]) -> None:
    """Ghi lựa chọn của người dùng cho một camera: ``"mask"``, ``"none"`` hoặc ``None`` (bỏ)."""
    status = _read_status(scene_root)
    if state is None:
        status.pop(camera_id, None)
    elif state in ("mask", "none"):
        status[camera_id] = state
    else:
        raise ValueError(f"trạng thái mask không hợp lệ: {state!r}")
    path = scene_root / STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pending_cameras(
    scene_root: Path, manifest: Manifest, cameras: Optional[list[str]] = None
) -> list[str]:
    """Camera (trong ``cameras`` nếu có) chưa có mask và chưa được xác nhận không thấy thân xe."""
    return [
        item["camera_id"]
        for item in camera_states(scene_root, manifest)
        if item["state"] == "pending" and (cameras is None or item["camera_id"] in cameras)
    ]


def pending_warning(
    scene_root: Path, manifest: Manifest, cameras: Optional[list[str]] = None
) -> Optional[str]:
    """Câu cảnh báo khi tạo run mà camera của split chưa xử lý mask thân xe; ``None`` nếu đủ."""
    pending = pending_cameras(scene_root, manifest, cameras)
    if not pending:
        return None
    return (
        f"camera {', '.join(pending)} chưa có mask thân xe và chưa xác nhận không thấy thân xe; "
        "nếu camera thấy thân xe thì thân xe bị dựng lẫn vào cảnh và tính vào điểm. Vẽ mask hoặc "
        "xác nhận ở Studio (cảnh, tab Kiểm tra, mục Mask thân xe)"
    )


def effective_mask(scene_root: Path, dataset: str, camera_id: str, index: int) -> Optional[Path]:
    """Chọn file mask thân xe hiệu lực theo thứ tự cảnh, xác nhận none, asset.

    Args:
        scene_root: Thư mục cảnh hoặc cảnh ảo của run.
        dataset: Dataset nguồn trong manifest.
        camera_id: Tên camera gốc.
        index: Chỉ số camera trong manifest.

    Returns:
        File PNG được chọn, hoặc None khi không dùng mask thân xe.
    """
    scene_mask = scene_root / "ego_masks" / f"{camera_id}.png"
    if scene_mask.is_file():
        return scene_mask
    if _read_status(scene_root).get(camera_id) == "none":
        return None
    return tool_asset(dataset, camera_id, index)


def mask_digest(scene_root: Path, manifest: Manifest) -> str:
    """Băm mask chính thức theo camera để nhận diện dữ liệu của cache và run.

    Args:
        scene_root: Thư mục cảnh hoặc bản chụp của run.
        manifest: Manifest xác định danh sách camera.

    Returns:
        SHA-256 của tên camera, trạng thái không mask và nội dung PNG hiệu lực.
    """
    digest = hashlib.sha256()
    for index, camera in enumerate(manifest.cameras):
        path = effective_mask(scene_root, manifest.source.dataset, camera.camera_id, index)
        digest.update(camera.camera_id.encode() + b"\0")
        digest.update(path.read_bytes() if path is not None else b"none")
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_masks(scene_root: Path, manifest: Manifest, destination: Path) -> str:
    """Chụp mask vào metadata run và trả digest từ chính các byte đã chụp.

    Args:
        scene_root: Cảnh gốc.
        manifest: Manifest của cảnh.
        destination: Thư mục d2s/scene_masks của run mới.

    Returns:
        Digest nhận diện bản chụp đã ghi.
    """
    destination.mkdir(parents=True)
    status = {}
    for index, camera in enumerate(manifest.cameras):
        path = effective_mask(scene_root, manifest.source.dataset, camera.camera_id, index)
        if path is not None:
            (destination / f"{camera.camera_id}.png").write_bytes(path.read_bytes())
        else:
            status[camera.camera_id] = "none"
    (destination / "status.json").write_text(json.dumps(status), encoding="utf-8")
    digest = hashlib.sha256()
    for camera in manifest.cameras:
        path = destination / f"{camera.camera_id}.png"
        digest.update(camera.camera_id.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"none")
        digest.update(b"\0")
    return digest.hexdigest()
