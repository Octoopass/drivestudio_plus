"""Nhận dạng thư mục PHENIKAA và gợi ý lệnh, trên cây thư mục giả (không cần numpy, OpenCV).

Các ca lấy từ lần thử luồng người dùng mới (27/11/2026): camera rỗng hoặc thiếu ảnh, thiếu
``label/``, thiếu LiDAR, calib để ở chỗ khác, tên thư mục camera viết thường. ``inspect`` phải báo
đúng dữ liệu trên đĩa và lệnh gợi ý phải chạy được.
"""

from pathlib import Path
from typing import Optional

import pytest

from drive2sim.cli.main import main
from drive2sim.cli.source import _suggestion
from drive2sim.ingest import phenikaa
from drive2sim.ingest.detect import SourceReport, inspect_source
from drive2sim.ingest.dispatch import IngestRequestError, converter_options, plan_ingest

STAMPS = ["1758620947-799749851", "1758620947-899739742", "1758620947-999741793"]


def make_tree(
    root: Path,
    cameras: Optional[dict[str, int]] = None,
    lidar: bool = True,
    label: bool = True,
    calib: bool = True,
) -> Path:
    """Chuỗi ``SEQ A`` với file rỗng: camera → số mốc có ảnh (mặc định CAM_P_F đủ, CAM_F_L 2)."""
    sequence = root / "SEQ A"
    for t, stamp in enumerate(STAMPS):
        for sub, ext, keep in (("ego_pose", "txt", True), ("lidar", "laz", lidar)):
            if keep:
                (sequence / sub).mkdir(parents=True, exist_ok=True)
                (sequence / sub / f"{stamp}.{ext}").touch()
        if label:
            (sequence / "label").mkdir(exist_ok=True)
            (sequence / "label" / f"{stamp}.txt").touch()
        for camera, count in (cameras or {"CAM_P_F": 3, "CAM_F_L": 2}).items():
            (sequence / "camera" / camera).mkdir(parents=True, exist_ok=True)
            if t < count:
                (sequence / "camera" / camera / f"{stamp}.jpg").touch()
    if calib:
        (root / "calib").mkdir()
        (root / "calib" / phenikaa.INTRINSICS_FILE).write_text("{}")
        (root / "calib" / phenikaa.EXTRINSICS_FILE).write_text("{}")
    return root


def test_inspect_reports_cameras_lidar_and_boxes_found_on_disk(tmp_path: Path) -> None:
    raw = make_tree(tmp_path / "raw", cameras={"CAM_P_F": 3, "CAM_F_L": 2, "CAM_F_F": 0})
    report = inspect_source(raw)
    # Chỉ camera có ảnh, không phải danh sách 10 camera cố định; CAM_F_F rỗng không tính.
    assert report.cameras == ["CAM_P_F", "CAM_F_L"]
    assert report.has_lidar is True and report.has_cuboids is True and not report.errors
    line = next(note for note in report.notes if note.startswith("SEQ A:"))
    assert "camera có đủ ảnh: CAM_P_F" in line
    assert "CAM_F_L (2/3 ảnh)" in line and "CAM_F_F (thư mục ảnh rỗng)" in line
    assert "CAM_P_FL (không có thư mục ảnh)" in line


def test_missing_labels_mean_no_boxes(tmp_path: Path) -> None:
    report = inspect_source(make_tree(tmp_path / "raw", label=False))
    assert report.has_cuboids is False
    assert any("không có label/" in note for note in report.notes)


def test_sequence_without_lidar_is_recognised_with_an_error(tmp_path: Path) -> None:
    report = inspect_source(make_tree(tmp_path / "raw", lidar=False))
    assert report.dataset == "phenikaa" and report.has_lidar is False
    assert any("không có LiDAR" in error and "SEQ A" in error for error in report.errors)


def test_lowercase_camera_folders_are_recognised(tmp_path: Path) -> None:
    report = inspect_source(make_tree(tmp_path / "raw", cameras={"cam_p_f": 3, "cam_f_l": 3}))
    assert report.dataset == "phenikaa" and report.cameras == ["CAM_P_F", "CAM_F_L"]
    dirs = phenikaa.camera_dirs(tmp_path / "raw" / "SEQ A")
    assert dirs["CAM_P_F"].name == "cam_p_f"


def test_unrelated_folder_with_lidar_is_not_phenikaa(tmp_path: Path) -> None:
    # Trước đây glob() rỗng vẫn bị coi là đúng nên chỉ cần lidar/*.laz là nhận nhầm.
    sequence = tmp_path / "raw" / "rig"
    for sub in ("camera/front", "ego_pose", "lidar"):
        (sequence / sub).mkdir(parents=True)
    (sequence / "lidar" / "000.laz").touch()
    assert inspect_source(tmp_path / "raw").dataset is None


def test_missing_calib_suggests_a_runnable_command(tmp_path: Path) -> None:
    raw = make_tree(tmp_path / "raw", calib=False)
    report = inspect_source(raw)
    assert any("--calib" in error for error in report.errors)
    assert report.suggested_args[0] == "--calib"
    assert "--calib <thư mục chứa Camera_Intrinsics.json>" in _suggestion(report)
    # Calib để ở chỗ khác: truyền vào inspect thì hết lỗi, lệnh gợi ý kèm đúng đường dẫn đó.
    other = make_tree(tmp_path / "other")
    fixed = inspect_source(raw, calib=other / "calib")
    assert not fixed.errors and f"--calib {other / 'calib'}" in _suggestion(fixed)


def test_plan_without_sequences_explains_the_problem(tmp_path: Path) -> None:
    report = SourceReport(path=tmp_path, errors=["không có LiDAR"])
    with pytest.raises(IngestRequestError, match="không thấy chuỗi nào của phenikaa"):
        plan_ingest(report, "phenikaa")


@pytest.mark.parametrize("missing", ["calib", "lidar"])
def test_inspect_does_not_offer_an_ingest_command_for_invalid_source(
    tmp_path: Path, capsys: pytest.CaptureFixture, missing: str
) -> None:
    raw = make_tree(tmp_path / "raw", calib=missing != "calib", lidar=missing != "lidar")
    assert main(["source", "inspect", str(raw)]) == 0
    output = capsys.readouterr().out
    assert "lỗi:" in output
    assert "lệnh tiếp theo:" not in output
    assert "khắc phục các lỗi trên" in output


def test_converter_options_shared_by_cli_and_studio(tmp_path: Path) -> None:
    kitti, pk = converter_options("phenikaa", [" cam_p_f ", ""], tmp_path / "calib")
    assert pk.cameras == ("cam_p_f",) and pk.calib_dir == tmp_path / "calib"
    assert kitti.cameras == ("image_00", "image_01", "image_02", "image_03")
    kitti, _ = converter_options("kitti360", ["image_00"], None, 5, 40)
    assert kitti.cameras == ("image_00",) and (kitti.start_frame, kitti.num_frames) == (5, 40)
