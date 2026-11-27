"""Công cụ vẽ mask thân xe của Studio: trung vị, lưu, xoá, ảnh chồng, trạng thái (không SAM2)."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from drive2sim.core.schema import Manifest  # noqa: E402
from drive2sim.studio.api import ego_mask_tool as tool  # noqa: E402


def make_scene(root: Path) -> Manifest:
    """Cảnh hai camera, mười khung: nền ngẫu nhiên, thân xe đỏ cố định ở đáy ảnh camera 0."""
    from unittest.mock import MagicMock

    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for t in range(10):
        for index in range(2):
            image = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            if index == 0:
                image[40:] = (0, 0, 200)
            cv2.imwrite(str(root / "images" / f"{t:03d}_{index}.png"), image)
    manifest = MagicMock(spec=Manifest)
    manifest.cameras = [MagicMock(camera_id="CAM_A"), MagicMock(camera_id="CAM_B")]
    manifest.source = MagicMock(dataset="custom")
    return manifest


def test_median_save_delete_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("D2S_SAM2_PYTHON", str(tmp_path / "missing" / "python"))
    root = tmp_path / "scene"
    manifest = make_scene(root)
    median = cv2.imread(str(tool.median_image(root, manifest, "CAM_A")))
    assert median.shape == (48, 64, 3) and abs(int(median[45, 10, 2]) - 200) < 12  # JPEG
    with pytest.raises(tool.EgoMaskError, match="không có camera"):
        tool.median_image(root, manifest, "CAM_X")
    status = tool.status(root, manifest)
    assert status["sam2_ready"] is False and "missing" in status["sam2_hint"]
    with pytest.raises(tool.EgoMaskError, match="chưa cài"):
        tool.segment(root, manifest, "CAM_A", [(10.0, 45.0, 1)])
    with pytest.raises(tool.EgoMaskError, match="thân xe"):
        tool.segment(root, manifest, "CAM_A", [(10.0, 5.0, 0)])
    # Mask nháp (như SAM2 ghi) → lưu thành mask của cảnh → ảnh chồng → xoá.
    draft = np.zeros((48, 64), np.uint8)
    draft[40:] = 255
    cv2.imwrite(str(root / tool.TOOL_DIR / "draft_CAM_A.png"), draft)
    tool.save(root, manifest, "CAM_A")
    saved = root / "ego_masks" / "CAM_A.png"
    assert saved.is_file() and not (root / tool.TOOL_DIR / "draft_CAM_A.png").exists()
    camera = tool.status(root, manifest)["cameras"][0]
    assert camera["has_mask"] and camera["coverage"] == pytest.approx(8 / 48)
    png = tool.overlay_png(root, manifest, "CAM_A", False)
    overlay = cv2.imdecode(np.frombuffer(png, np.uint8), -1)
    assert overlay.shape == (48, 64, 4) and overlay[45, 30, 3] > 0 and overlay[10, 30, 3] == 0
    with pytest.raises(tool.EgoMaskError, match="chưa có mask"):
        tool.overlay_png(root, manifest, "CAM_B", False)
    tool.mark_none(root, manifest, "CAM_B")
    states = {c["camera_id"]: c["state"] for c in tool.status(root, manifest)["cameras"]}
    assert states == {"CAM_A": "mask", "CAM_B": "none"}
    tool.delete(root, manifest, "CAM_A")
    tool.delete(root, manifest, "CAM_B")
    assert not saved.exists()
    assert {c["state"] for c in tool.status(root, manifest)["cameras"]} == {"pending"}


def test_eraser_keeps_saved_mask_until_save_and_accumulates_strokes(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    manifest = make_scene(root)
    (root / "ego_masks").mkdir()
    saved = root / "ego_masks" / "CAM_A.png"
    original = np.full((48, 64), 255, np.uint8)
    cv2.imwrite(str(saved), original)
    coverage = tool.erase(root, manifest, "CAM_A", [(5, 10), (50, 10)], 3)
    draft_path = root / tool.TOOL_DIR / "draft_CAM_A.png"
    draft = cv2.imread(str(draft_path), cv2.IMREAD_GRAYSCALE)
    assert 0 < coverage < 1
    assert (draft[10, 5:51] == 0).all()
    assert (cv2.imread(str(saved), 0) == original).all()
    tool.erase(root, manifest, "CAM_A", [(30, 30)], 4)
    updated = cv2.imread(str(draft_path), 0)
    assert updated[10, 20] == 0 and updated[30, 30] == 0 and updated[45, 60] == 255
    tool.save(root, manifest, "CAM_A")
    np.testing.assert_array_equal(cv2.imread(str(saved), 0), updated)


@pytest.mark.parametrize("points,radius", [([], 2), ([(1, 1)], 0), ([(64, 5)], 2)])
def test_eraser_rejects_invalid_strokes_without_changing_saved_mask(
    tmp_path: Path, points: list[tuple[float, float]], radius: int
) -> None:
    root = tmp_path / "scene"
    manifest = make_scene(root)
    (root / "ego_masks").mkdir()
    saved = root / "ego_masks" / "CAM_A.png"
    cv2.imwrite(str(saved), np.full((48, 64), 255, np.uint8))
    before = saved.read_bytes()
    with pytest.raises(tool.EgoMaskError):
        tool.erase(root, manifest, "CAM_A", points, radius)
    assert saved.read_bytes() == before
    assert not (root / tool.TOOL_DIR / "draft_CAM_A.png").exists()


def test_erase_uses_visible_source_and_discard_removes_hidden_draft(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    manifest = make_scene(root)
    (root / "ego_masks").mkdir()
    saved = root / "ego_masks" / "CAM_A.png"
    cv2.imwrite(str(saved), np.full((48, 64), 255, np.uint8))
    tool.erase(root, manifest, "CAM_A", [(10, 10)], 3, source="saved")
    first = tool.status(root, manifest)["cameras"][0]
    assert first["has_draft"] and first["draft_coverage"] < 1
    tool.erase(root, manifest, "CAM_A", [(30, 30)], 3, source="saved")
    draft = root / tool.TOOL_DIR / "draft_CAM_A.png"
    assert cv2.imread(str(draft), 0)[10, 10] == 255
    assert tool.revision(root, "CAM_A") != first["revision"]
    tool.discard_draft(root, manifest, "CAM_A")
    assert not draft.exists() and saved.exists()
    assert not tool.status(root, manifest)["cameras"][0]["has_draft"]
    with pytest.raises(tool.EgoMaskError, match="chưa có mask"):
        tool.erase(root, manifest, "CAM_A", [(10, 10)], 3, source="draft")


def test_segment_replaces_draft_only_after_worker_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "scene"
    manifest = make_scene(root)
    tool.median_image(root, manifest, "CAM_A")
    draft = root / tool.TOOL_DIR / "draft_CAM_A.png"
    cv2.imwrite(str(draft), np.zeros((48, 64), np.uint8))
    original = draft.read_bytes()

    def failed_worker(request: dict) -> dict:
        Path(request["out"]).write_bytes(b"incomplete png")
        return {"ok": False, "error": "worker stopped"}

    monkeypatch.setattr(tool._WORKER, "run", failed_worker)
    with pytest.raises(tool.EgoMaskError, match="worker stopped"):
        tool.segment(root, manifest, "CAM_A", [(10, 45, 1)])
    assert draft.read_bytes() == original

    def successful_worker(request: dict) -> dict:
        assert Path(request["out"]) != draft
        cv2.imwrite(request["out"], np.full((48, 64), 255, np.uint8))
        return {"ok": True, "coverage": 1.0}

    monkeypatch.setattr(tool._WORKER, "run", successful_worker)
    assert tool.segment(root, manifest, "CAM_A", [(10, 45, 1)]) == 1.0
    assert (cv2.imread(str(draft), 0) == 255).all()
