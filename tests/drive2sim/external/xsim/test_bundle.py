"""Kiểm tra bộ đọc gói quan sát mà worker XSIM dùng thay cho thư mục cảnh."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from drive2sim.external.xsim.bundle import Observations  # noqa: E402
from tests.drive2sim.external.xsim.bundle_fixture import write_bundle  # noqa: E402


def test_reads_calibration_time_masks_and_lidar(bundle: Path) -> None:
    observations = Observations(bundle)
    camera = observations.cameras[1]
    assert camera.model == "fisheye" and camera.distortion == (0.1, 0.0, 0.0, 0.0)
    assert observations.time_s(4) == pytest.approx(0.1)
    assert observations.source_frame(4) == 102
    assert observations.train_mask(1)[:, :8].all() and not observations.train_mask(1)[:, 8:].any()
    assert observations.eval_mask(1).sum() == 96
    assert observations.lidar(4)["sensor_to_world"][0, 3] == 4
    assert observations.instances() is None and observations.smpl() is None


def test_missing_masks_mean_whole_image(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "b", eval_mask=False)
    (root / "masks" / "train_1.png").unlink()
    observations = Observations(root)
    assert observations.eval_mask(1) is None
    assert observations.train_mask(1).all()


def test_rejects_other_format_and_unordered_time(bundle: Path) -> None:
    index = json.loads((bundle / "observations.json").read_text())
    (bundle / "observations.json").write_text(json.dumps({**index, "version": 2}))
    with pytest.raises(ValueError, match="không phải gói"):
        Observations(bundle)
    index["timestamps_us"] = list(reversed(index["timestamps_us"]))
    (bundle / "observations.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="không tăng"):
        Observations(bundle)
