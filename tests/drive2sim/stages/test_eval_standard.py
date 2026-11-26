"""Kiểm tra ``eval.standard`` protocol 0.5: tách camera đã train và camera chỉ test.

File kiểm tra:
- ``split_frames_by_role`` chia metric từng ảnh theo camera đã train (R1) và camera chỉ test (R2);
- ``aggregate_frames`` lấy trung bình, bỏ giá trị ``None`` của metric sau chỉnh màu;
- ``make_summary`` điền ``overall`` từ camera đã train và ``shifted`` từ camera chỉ test, giữ
  nguyên đường cũ khi split không có camera chỉ test, và báo lỗi khi thiếu ``frames.jsonl``;
- ``summary.json`` ``d2s-summary/0.6`` (chưa có ``shifted``) vẫn đọc được;
- ``frames_have_color_match`` nhận ra file metric từng ảnh do code đo cũ ghi.
"""

import json
from pathlib import Path

import pytest

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, Summary
from drive2sim.stages.eval.standard import (
    PROTOCOL_VERSION,
    aggregate_frames,
    frames_have_color_match,
    make_summary,
    split_frames_by_role,
)

METRICS = {"psnr": 25.0, "ssim": 0.8, "lpips": 0.2}
CAMERA_IDS = ["image_00", "image_01"]


def frame(camera_index: int, psnr: float, psnr_cc: object = None) -> dict:
    """Một bản ghi metric của ảnh test, như một dòng của ``frames.jsonl``."""
    return {
        "frame_index": 8,
        "camera_index": camera_index,
        "psnr": psnr,
        "ssim": 0.8,
        "lpips": 0.2,
        "psnr_cc": psnr_cc,
        "lpips_cc": 0.25 if psnr_cc is not None else None,
    }


def test_split_frames_by_role() -> None:
    frames = [{"camera_index": 0, "psnr": 26.0}, {"camera_index": 1, "psnr": 22.0}]
    trained, shifted = split_frames_by_role(frames, {1})
    assert [f["psnr"] for f in trained] == [26.0]
    assert [f["psnr"] for f in shifted] == [22.0]


def test_aggregate_frames_skips_missing_colour_match() -> None:
    frames = [
        {"psnr": 20.0, "ssim": 0.7, "lpips": 0.3, "psnr_cc": 22.0, "lpips_cc": 0.25},
        {"psnr": 22.0, "ssim": 0.8, "lpips": 0.2, "psnr_cc": None, "lpips_cc": None},
    ]
    result = aggregate_frames(frames)
    assert result["psnr"] == 21.0 and result["psnr_cc"] == 22.0 and result["lpips_cc"] == 0.25


def test_aggregate_frames_without_colour_match_columns_gives_none() -> None:
    # file frames.jsonl của protocol 0.4 không có psnr_cc, lpips_cc
    result = aggregate_frames([{"psnr": 20.0, "ssim": 0.7, "lpips": 0.3}])
    assert result["psnr"] == 20.0
    assert result["psnr_cc"] is None and result["lpips_cc"] is None


def test_summary_0_6_without_shifted_still_loads(tmp_path: Path) -> None:
    data = {
        "schema_version": "d2s-summary/0.6",
        "protocol": {
            "version": "0.4",
            "image_dtype": "uint8",
            "resolution": "downscale=1",
            "lpips": {"net": "alex", "package": "torchmetrics", "package_version": "1"},
        },
        "split_id": "every8_allcams",
        "overall": {"psnr": 27.1, "ssim": 0.83, "lpips": 0.21},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data))
    summary = files.load(path, Summary)
    assert summary.shifted is None
    assert SCHEMA_VERSIONS["summary"] == "d2s-summary/0.8"


def test_make_summary_with_test_only_camera_fills_shifted(tmp_path: Path) -> None:
    frames = [
        frame(0, 26.0, 26.5),
        frame(0, 28.0, 28.5),
        frame(1, 22.0, 23.0),
        frame(1, 20.0, 21.0),
    ]
    summary = make_summary(
        METRICS,
        "every8_stereo",
        tmp_path,
        camera_ids=CAMERA_IDS,
        frames=frames,
        test_only_cameras=["image_01"],
    )
    assert summary.protocol.version == PROTOCOL_VERSION == "0.5"
    assert summary.overall.psnr == pytest.approx(27.0)  # chỉ image_00
    assert summary.shifted is not None
    assert summary.shifted.cameras == ["image_01"]
    assert summary.shifted.num_images == 2
    assert summary.shifted.psnr == pytest.approx(21.0)
    assert summary.shifted.psnr_cc == pytest.approx(22.0)
    assert summary.shifted.lpips_cc == pytest.approx(0.25)
    path = files.dump(summary, tmp_path / "summary.json")
    assert files.load(path, Summary) == summary


def test_make_summary_without_test_only_camera_keeps_drivestudio_overall(tmp_path: Path) -> None:
    summary = make_summary(METRICS, "every8_allcams", tmp_path, camera_ids=CAMERA_IDS)
    assert summary.overall.psnr == 25.0
    assert summary.shifted is None


def test_make_summary_test_only_camera_needs_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frames.jsonl"):
        make_summary(
            METRICS,
            "every8_stereo",
            tmp_path,
            camera_ids=CAMERA_IDS,
            test_only_cameras=["image_01"],
        )


def test_make_summary_test_only_camera_without_its_images_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_01"):
        make_summary(
            METRICS,
            "every8_stereo",
            tmp_path,
            camera_ids=CAMERA_IDS,
            frames=[frame(0, 26.0, 26.5)],
            test_only_cameras=["image_01"],
        )


def test_frames_have_color_match(tmp_path: Path) -> None:
    new = tmp_path / "new.jsonl"
    new.write_text(json.dumps(frame(0, 26.0, 26.5)) + "\n")
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"camera_index": 0, "psnr": 26.0}) + "\n")
    assert frames_have_color_match(new)
    assert not frames_have_color_match(old)
