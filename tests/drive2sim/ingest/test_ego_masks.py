"""Trạng thái mask thân xe từng camera, cảnh báo khi tạo run, và báo dữ liệu tool bỏ qua."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from drive2sim.core.schema import Manifest
from drive2sim.ingest import ego_masks
from drive2sim.ingest.detect import inspect_source


def manifest(dataset: str, cameras: list[str]) -> Manifest:
    item = MagicMock(spec=Manifest)
    item.source = MagicMock(dataset=dataset)
    item.cameras = [MagicMock(camera_id=c) for c in cameras]
    return item


def states(root: Path, m: Manifest) -> dict[str, tuple]:
    return {s["camera_id"]: (s["state"], s["source"]) for s in ego_masks.camera_states(root, m)}


def test_states_from_scene_tool_and_user(tmp_path: Path) -> None:
    custom = manifest("custom", ["front", "left", "rear"])
    assert states(tmp_path, custom) == {c: ("pending", None) for c in ("front", "left", "rear")}
    (tmp_path / "ego_masks").mkdir()
    (tmp_path / "ego_masks" / "front.png").write_bytes(b"png")
    ego_masks.set_state(tmp_path, "left", "none")
    assert states(tmp_path, custom) == {
        "front": ("mask", "scene"),
        "left": ("none", "confirmed"),
        "rear": ("pending", None),
    }
    assert ego_masks.pending_cameras(tmp_path, custom) == ["rear"]
    assert ego_masks.pending_cameras(tmp_path, custom, ["front", "left"]) == []
    assert ego_masks.pending_warning(tmp_path, custom, ["front"]) is None
    assert "rear" in (ego_masks.pending_warning(tmp_path, custom) or "")
    ego_masks.set_state(tmp_path, "left", None)
    assert states(tmp_path, custom)["left"] == ("pending", None)
    with pytest.raises(ValueError):
        ego_masks.set_state(tmp_path, "left", "maybe")


def test_public_datasets_need_no_confirmation(tmp_path: Path) -> None:
    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    nuscenes = manifest("nuscenes", [*cameras, "CAM_BACK"])
    found = states(tmp_path, nuscenes)
    assert found["CAM_BACK"] == ("mask", "tool") and found["CAM_FRONT"] == ("none", "tool")
    kitti360 = manifest("kitti360", ["image_00", "image_01", "image_02", "image_03"])
    assert ego_masks.pending_cameras(tmp_path, kitti360) == []
    # PHENIKAA: loader chỉ đọc mask trong cảnh, asset của tool không tính.
    assert states(tmp_path, manifest("phenikaa", ["CAM_P_F"]))["CAM_P_F"] == ("pending", None)


def test_unused_data_is_reported(tmp_path: Path) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "a.mp4").write_bytes(b"")
    (tmp_path / "videos" / "b.webm").write_bytes(b"")
    (tmp_path / "sensors" / "RADAR_FRONT").mkdir(parents=True)
    report = inspect_source(tmp_path)
    text = " ".join(report.notes)
    assert "1 file .mp4" in text and "1 file .webm" in text and "radar: 1 thư mục" in text
    assert "ffmpeg" in text  # chỉ có video: hướng dẫn tách khung


def test_effective_mask_priority_and_snapshot_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"default")
    monkeypatch.setattr(ego_masks, "tool_asset", lambda *args: asset)
    item = manifest("custom", ["front"])
    before = ego_masks.mask_digest(scene, item)
    snapshot = tmp_path / "snapshot"
    assert ego_masks.snapshot_masks(scene, item, snapshot) == before
    assert (snapshot / "front.png").read_bytes() == b"default"
    ego_masks.set_state(scene, "front", "none")
    assert ego_masks.effective_mask(scene, "custom", "front", 0) is None
    assert ego_masks.mask_digest(scene, item) != before
    official = scene / "ego_masks" / "front.png"
    official.write_bytes(b"edited")
    assert ego_masks.effective_mask(scene, "custom", "front", 0) == official
    digest = ego_masks.mask_digest(scene, item)
    (scene / "reports").mkdir()
    (scene / "reports" / "draft_front.png").write_bytes(b"unsaved")
    assert ego_masks.mask_digest(scene, item) == digest
    assert (snapshot / "front.png").read_bytes() == b"default"


def test_empty_snapshot_suppresses_future_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ego_masks, "tool_asset", lambda *args: None)
    item = manifest("custom", ["front"])
    snapshot = tmp_path / "view" / "ego_masks"
    digest = ego_masks.snapshot_masks(tmp_path / "scene", item, snapshot)
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"added later")
    monkeypatch.setattr(ego_masks, "tool_asset", lambda *args: asset)
    assert ego_masks.effective_mask(snapshot.parent, "custom", "front", 0) is None
    assert ego_masks.mask_digest(snapshot.parent, item) == digest
