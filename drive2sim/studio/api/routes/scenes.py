"""Route của nhóm cảnh và split: ``/api/scenes/...`` (CONTRACTS mục 11.1; SPEC mục 8 nhóm A).

File này chứa:
- ``GET /api/scenes``: danh sách cảnh (manifest).
- ``GET /api/scenes/{scene_id}``: manifest, báo cáo kiểm tra, danh sách split của một cảnh và
  cảnh có được ingest bằng converter cũ không.
- ``GET /api/scenes/{scene_id}/frames/{frame_index}``: đường dẫn ảnh của mọi camera ở một khung,
  cho lưới ảnh đồng bộ.
- ``GET /api/scenes/{scene_id}/splits``, ``GET .../splits/{split_id}``: đọc split.
- ``POST /api/scenes/{scene_id}/splits``: tạo split (như ``d2s split create``).
- ``POST /api/scenes/{scene_id}/validate``: kiểm tra lại cảnh, ghi ``reports/validation.json``.
- ``GET /api/scenes/{scene_id}/rig``: sơ đồ camera và quỹ đạo xe (``reports/rig.json``) hoặc
  ``null``; ``POST`` cùng đường dẫn tạo lại file đó (``engine/rig.py``, cần PyTorch).
- ``GET /api/scenes/{scene_id}/lidar``: ảnh chồng LiDAR (url) và số liệu
  ``reports/lidar_check.json`` hoặc ``null``; ``POST`` cùng đường dẫn chiếu lại và kiểm tra lại
  cảnh (cần PyTorch, ảnh trên máy).
- ``/api/scenes/{scene_id}/egomask...``: công cụ vẽ mask thân xe (``ego_mask_tool.py``): trạng thái
  mask từng camera, ảnh trung vị, mask (nháp hoặc đã lưu), tách mask bằng SAM2.1 theo điểm bấm,
  lưu, xoá, xác nhận camera không thấy thân xe.
- ``DELETE /api/scenes/{scene_id}``: xoá cảnh cùng split, cache, run của nó.
- ``DELETE /api/scenes/{scene_id}/splits/{split_id}``: xoá split (còn run dùng thì ``409``).
"""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core import files
from drive2sim.engine import housekeeping, scenes
from drive2sim.engine.rig import (
    LIDAR_OVERLAY_DIR,
    RigError,
    generate_lidar_check,
    generate_rig,
    read_lidar_check,
    read_rig,
)
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE
from drive2sim.ingest.images import IMAGE_SUFFIXES
from drive2sim.ingest.manifest import outdated_output_version
from drive2sim.ingest.reports import revalidate
from drive2sim.ingest.validate import validate_scene
from drive2sim.studio.api import ego_mask_tool, frame_tools
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class CreateSplitBody(BaseModel):
    """Thân của ``POST /api/scenes/{scene_id}/splits``; cùng ý nghĩa với ``d2s split create``.

    Attributes:
        strategy: Chiến lược, ví dụ ``every_nth_frame:8``.
        cameras: Camera dùng; bỏ trống là mọi camera.
        frame_range: Khung đầu và khung cuối; bỏ trống là cả cảnh.
        split_id: Tên split; bỏ trống thì đặt theo CONTRACTS mục 10.1.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str
    cameras: Optional[list[str]] = None
    frame_range: Optional[tuple[int, int]] = None
    split_id: Optional[str] = None


@router.get("")
def list_scenes(request: Request) -> dict:
    """Liệt kê các cảnh đã ingest.

    Returns:
        ``{"items": [manifest, ...], "total": n}``.
    """
    items = [m.model_dump(mode="json") for m in scenes.list_scenes(state_of(request).workdir)]
    return {"items": items, "total": len(items)}


@router.get("/{scene_id}")
def get_scene(scene_id: str, request: Request) -> dict:
    """Trả về manifest, báo cáo kiểm tra và danh sách split của một cảnh.

    Returns:
        ``{"manifest", "validation", "splits", "split_cameras", "kaggle",
        "converter_outdated"}``; ``split_cameras``
        là camera của từng split (``{split_id: [camera_id]}``), để giao diện hiện loại ống kính và
        xét approach theo camera của split; ``validation`` là ``null`` nếu chưa có
        báo cáo; ``kaggle`` là danh sách notebook ingest (``kaggle_scene.json``) nếu cảnh được nạp
        trên Kaggle (máy chỉ có manifest và báo cáo, không có ảnh), ngược lại ``null``;
        ``converter_outdated`` là ``{"output_version", "current_output_version"}`` khi cảnh được
        ingest bằng converter cũ (``outdated_output_version``), ngược lại ``null``.
    """
    workdir = state_of(request).workdir
    scene = workdir.load_scene(scene_id)
    report_path = scene.root / "reports" / "validation.json"
    validation = json.loads(report_path.read_text("utf-8")) if report_path.is_file() else None
    kaggle_path = scene.root / KAGGLE_SCENE_FILE
    kaggle = None
    if kaggle_path.is_file():
        kaggle = json.loads(kaggle_path.read_text("utf-8"))["kernels"]
    splits = scenes.list_splits(workdir, scene_id)
    outdated = outdated_output_version(scene.manifest.source)
    return {
        "manifest": scene.manifest.model_dump(mode="json"),
        "validation": validation,
        "splits": [split.split_id for split in splits],
        "split_cameras": {split.split_id: list(split.cameras) for split in splits},
        "kaggle": kaggle,
        "converter_outdated": (
            {"output_version": outdated[0], "current_output_version": outdated[1]}
            if outdated
            else None
        ),
    }


@router.get("/{scene_id}/frames/{frame_index}")
def get_frame(scene_id: str, frame_index: int, request: Request) -> dict:
    """Trả về đường dẫn ảnh của mọi camera ở một khung, theo thứ tự camera trong manifest.

    Ảnh DriveStudio đặt tên ``images/<khung 3 chữ số>_<chỉ số camera>.<đuôi>``.

    Returns:
        ``{"frame_index", "images": [{"camera_id", "url"}]}``; ``url`` là ``null`` nếu thiếu ảnh.
    """
    workdir = state_of(request).workdir
    scene = workdir.load_scene(scene_id)
    images = []
    for index, camera in enumerate(scene.manifest.cameras):
        found = [
            path
            for path in sorted((scene.root / "images").glob(f"{frame_index:03d}_{index}.*"))
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        url = f"/files/{workdir.relative(found[0])}" if found else None
        images.append({"camera_id": camera.camera_id, "url": url})
    return {"frame_index": frame_index, "images": images}


@router.get("/{scene_id}/frames/{frame_index}/panorama")
def get_panorama(
    scene_id: str, frame_index: int, request: Request, width: int = Query(2048, ge=512, le=4096)
) -> Response:
    """Ảnh toàn cảnh equirectangular ghép từ mọi camera của một khung (``frame_tools.py``).

    Returns:
        PNG; header ``X-D2S-Coverage`` là tỉ lệ pixel có ảnh. Cảnh không có calib cố định theo
        camera, thiếu ảnh: ``400``.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    png, coverage = frame_tools.panorama(scene.root, scene.manifest, frame_index, width)
    return Response(png, media_type="image/png", headers={"X-D2S-Coverage": f"{coverage:.4f}"})


@router.get("/{scene_id}/frames/{frame_index}/lidar")
def get_lidar_overlay(scene_id: str, frame_index: int, camera: str, request: Request) -> Response:
    """Ảnh một camera có điểm LiDAR chiếu lên, tô theo độ sâu (``frame_tools.py``).

    Returns:
        PNG; header ``X-D2S-Points`` là số điểm vẽ được. LiDAR dạng khác KITTI, camera lạ, thiếu
        file: ``400``.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    try:
        png, points = frame_tools.lidar_overlay(scene.root, scene.manifest, frame_index, camera)
    except frame_tools.FrameToolError as error:  # gồm môi trường thiếu numpy, OpenCV
        # Dataset cách nhanh chưa đọc được (calib theo khung, LiDAR dạng khác): dùng ảnh chồng do
        # loader tạo sẵn ở tab Kiểm tra nếu khung này có.
        ready = scene.root / "reports" / LIDAR_OVERLAY_DIR / f"{frame_index:03d}_{camera}.jpg"
        if ready.is_file():
            return Response(
                ready.read_bytes(), media_type="image/jpeg", headers={"X-D2S-Source": "loader"}
            )
        frames = sorted(
            {item["frame"] for item in (read_lidar_check(scene.root) or {}).get("overlays", [])}
        )
        hint = (
            f"; ảnh chồng do loader tạo có ở khung {', '.join(map(str, frames))}" if frames else ""
        )
        raise HTTPException(status_code=400, detail=f"{error}{hint}") from None
    return Response(
        png, media_type="image/png", headers={"X-D2S-Points": str(points), "X-D2S-Source": "fast"}
    )


@router.get("/{scene_id}/splits")
def list_splits(scene_id: str, request: Request) -> dict:
    """Liệt kê split của một cảnh.

    Returns:
        ``{"items": [split, ...], "total": n}``.
    """
    items = [
        s.model_dump(mode="json") for s in scenes.list_splits(state_of(request).workdir, scene_id)
    ]
    return {"items": items, "total": len(items)}


@router.get("/{scene_id}/splits/{split_id}")
def get_split(scene_id: str, split_id: str, request: Request) -> dict:
    """Trả về nội dung một split."""
    return state_of(request).workdir.load_split(scene_id, split_id).model_dump(mode="json")


@router.post("/{scene_id}/splits", status_code=201)
def create_split(scene_id: str, body: CreateSplitBody, request: Request) -> dict:
    """Tạo split ``every_nth_frame:N`` (như ``d2s split create``).

    Returns:
        Split vừa tạo; mã ``201``.
    """
    split = scenes.create_split(
        state_of(request).workdir,
        scene_id,
        body.strategy,
        cameras=body.cameras,
        frame_range=body.frame_range,
        split_id=body.split_id,
    )
    return split.model_dump(mode="json")


@router.get("/{scene_id}/rig")
def rig(scene_id: str, request: Request) -> dict:
    """Sơ đồ camera và quỹ đạo xe của cảnh.

    Returns:
        ``{"rig": nội dung reports/rig.json | null}``.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    return {"rig": read_rig(scene.root)}


@router.post("/{scene_id}/rig")
def make_rig(scene_id: str, request: Request) -> dict:
    """Tạo lại ``reports/rig.json`` (như ``d2s scene rig``); vài chục giây.

    Raises:
        HTTPException: ``500`` kèm phần cuối log nếu không tạo được (thiếu PyTorch, thiếu pose).
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    try:
        rig = generate_rig(scene.root)
    except RigError as error:
        raise HTTPException(status_code=500, detail=str(error)) from None
    if not (scene.root / KAGGLE_SCENE_FILE).is_file():
        revalidate(scene.root)  # projection.camera_tilt đọc tilt_deg của rig.json
    return {"rig": rig}


def _lidar_payload(workdir, scene_root) -> dict:  # noqa: ANN001
    data = read_lidar_check(scene_root)
    folder = scene_root / "reports" / LIDAR_OVERLAY_DIR
    overlays = []
    for item in (data or {}).get("overlays", []):
        path = folder / str(item.get("file", ""))
        if path.is_file():
            overlays.append({**item, "url": f"/files/{workdir.relative(path)}"})
    return {"check": data, "overlays": overlays}


@router.get("/{scene_id}/lidar")
def lidar(scene_id: str, request: Request) -> dict:
    """Ảnh chồng LiDAR và số liệu kiểm tra thô của cảnh.

    Returns:
        ``{"check": nội dung reports/lidar_check.json | null, "overlays": [{"frame",
        "camera_id", "file", "url"}]}``; cảnh ingest trên Kaggle có ``check`` nhưng không có ảnh.
    """
    state = state_of(request)
    scene = state.workdir.load_scene(scene_id)
    return _lidar_payload(state.workdir, scene.root)


@router.post("/{scene_id}/lidar")
def make_lidar(scene_id: str, request: Request) -> dict:
    """Chiếu lại LiDAR lên ảnh (như ``d2s scene lidar-check``) rồi kiểm tra lại cảnh.

    Raises:
        HTTPException: ``409`` nếu cảnh chỉ có trên Kaggle hoặc không có LiDAR; ``500`` kèm log nếu
            không chạy được.
    """
    state = state_of(request)
    scene = state.workdir.load_scene(scene_id)
    if (scene.root / KAGGLE_SCENE_FILE).is_file() or not scene.manifest.available.lidar:
        raise HTTPException(status_code=409, detail="cảnh không có ảnh hoặc LiDAR trên máy")
    try:
        generate_lidar_check(scene.root)
    except RigError as error:
        raise HTTPException(status_code=500, detail=str(error)) from None
    revalidate(scene.root)
    return _lidar_payload(state.workdir, scene.root)


@router.post("/{scene_id}/validate")
def validate(scene_id: str, request: Request) -> dict:
    """Kiểm tra lại cảnh (như ``d2s scene validate``) và ghi báo cáo.

    Returns:
        Báo cáo kiểm tra mới.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    report = validate_scene(scene.root, scene.manifest)
    files.dump(report, scene.root / "reports" / "validation.json")
    return report.model_dump(mode="json")


@router.delete("/{scene_id}")
def delete_scene(scene_id: str, request: Request) -> dict:
    """Xoá cảnh cùng dữ liệu chuẩn, split, cache và run; có run đang chạy thì ``409``.

    Returns:
        ``{"removed": [đường dẫn đã xoá]}``.
    """
    return {"removed": housekeeping.delete_scene(state_of(request).workdir, scene_id)}


@router.delete("/{scene_id}/splits/{split_id}")
def delete_split(scene_id: str, split_id: str, request: Request) -> dict:
    """Xoá một split; còn run dùng split thì ``409``.

    Returns:
        ``{"removed": [đường dẫn đã xoá]}``.
    """
    return {"removed": housekeeping.delete_split(state_of(request).workdir, scene_id, split_id)}


class EgoMaskPoints(BaseModel):
    """Điểm bấm trên ảnh trung vị: ``[x, y, nhãn]`` (pixel ảnh gốc; 1 thân xe, 0 không phải)."""

    model_config = ConfigDict(extra="forbid")
    points: list[tuple[float, float, int]]


class EgoMaskStroke(BaseModel):
    """Nét tẩy có toạ độ pixel ảnh gốc và bán kính cọ."""

    model_config = ConfigDict(extra="forbid")
    points: list[tuple[float, float]] = Field(min_length=1, max_length=20000)
    radius: int = Field(ge=1, le=256)


def _ego_mask_call(  # noqa: ANN202
    request: Request,
    scene_id: str,
    action,  # noqa: ANN001
    revalidate_after: bool = False,
):
    scene = state_of(request).workdir.load_scene(scene_id)
    try:
        result = action(scene.root, scene.manifest)
    except ego_mask_tool.EgoMaskError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    if revalidate_after:  # báo cáo kiểm tra có mục ego_mask.*: cập nhật theo lựa chọn mới
        revalidate(scene.root)
    return result


@router.get("/{scene_id}/egomask")
def get_ego_masks(scene_id: str, request: Request) -> dict:
    """Mask thân xe của từng camera và SAM2 đã sẵn sàng chưa (``ego_mask_tool.status``)."""
    return _ego_mask_call(request, scene_id, ego_mask_tool.status)


@router.get("/{scene_id}/egomask/{camera}/median")
def get_ego_mask_median(scene_id: str, camera: str, request: Request) -> Response:
    """Ảnh trung vị của camera (JPEG, kích thước gốc): nền để bấm điểm."""
    path = _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.median_image(root, manifest, camera),
    )
    return Response(path.read_bytes(), media_type="image/jpeg")


@router.get("/{scene_id}/egomask/{camera}/overlay")
def get_ego_mask_overlay(
    scene_id: str, camera: str, request: Request, draft: bool = False
) -> Response:
    """Mask (nháp hoặc đã lưu) dạng PNG trong suốt; chưa có mask thì ``400``."""
    png = _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.overlay_png(root, manifest, camera, draft),
    )
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/{scene_id}/egomask/{camera}/segment")
def segment_ego_mask(scene_id: str, camera: str, body: EgoMaskPoints, request: Request) -> dict:
    """Tách mask nháp bằng SAM2.1 theo các điểm bấm.

    Returns:
        ``{"coverage": tỉ lệ ảnh mask che}``.
    """
    coverage = _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.segment(root, manifest, camera, body.points),
    )
    return {"coverage": coverage}


@router.post("/{scene_id}/egomask/{camera}/erase")
def erase_ego_mask(scene_id: str, camera: str, body: EgoMaskStroke, request: Request) -> dict:
    """Tẩy một nét thành mask nháp và trả tỉ lệ diện tích còn bị che."""
    coverage = _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.erase(
            root, manifest, camera, body.points, body.radius
        ),
    )
    return {"coverage": coverage}


@router.post("/{scene_id}/egomask/{camera}/save")
def save_ego_mask(scene_id: str, camera: str, request: Request) -> dict:
    """Lưu mask nháp thành ``ego_masks/<camera>.png`` của cảnh."""
    _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.save(root, manifest, camera),
        revalidate_after=True,
    )
    return {"saved": camera}


@router.post("/{scene_id}/egomask/{camera}/none")
def mark_ego_mask_none(scene_id: str, camera: str, request: Request) -> dict:
    """Xác nhận camera không thấy thân xe (xoá mask của cảnh nếu có)."""
    _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.mark_none(root, manifest, camera),
        revalidate_after=True,
    )
    return {"none": camera}


@router.delete("/{scene_id}/egomask/{camera}")
def delete_ego_mask(scene_id: str, camera: str, request: Request) -> dict:
    """Xoá mask thân xe (và mask nháp) của camera."""
    _ego_mask_call(
        request,
        scene_id,
        lambda root, manifest: ego_mask_tool.delete(root, manifest, camera),
        revalidate_after=True,
    )
    return {"deleted": camera}
