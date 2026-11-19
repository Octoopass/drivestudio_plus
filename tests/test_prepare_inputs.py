"""prepare_drivestudio: refuses early when KITTI-360 inputs are missing, and rebuilds a leftover raw_ dir."""
import pytest

from fisheye_fusion.pipeline import PipelineManager

SEQ = "2013_05_28_drive_0005_sync"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    k = tmp_path / "kitti360"
    (k / "data_2d_raw").mkdir(parents=True)
    (k / "data_3d_raw").mkdir()
    (k / "ImageSets-rosa-6class").mkdir()
    (k / "calibration").mkdir()
    (k / "data_poses.zip").write_bytes(b"z")
    for s in ("train", "val"):
        (k / "ImageSets-rosa-6class" / f"kitti360_rosa_infos_{s}.pkl").write_bytes(b"p")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    pm = PipelineManager(tmp_path)
    started = {}
    monkeypatch.setattr(pm, "_start", lambda kind, out, steps, env, info: started.update(kind=kind, out=out) or {"job_id": "x"})
    return tmp_path, k, venv, pm, started


CFG = {"dataset": "kitti360", "sequence": SEQ, "start": 0, "frames": 100, "name": "t1", "objects": True}


def test_missing_inputs_listed_before_anything_is_written(ws):
    root, k, venv, pm, started = ws
    (k / "data_2d_raw" / f"{SEQ}_image_00.tar").write_bytes(b"t")
    (k / "data_2d_raw" / f"{SEQ}_image_02.tar.part").write_bytes(b"t")
    with pytest.raises(ValueError) as e:
        pm.prepare_drivestudio(CFG, venv)
    msg = str(e.value)
    assert "image_02.tar" in msg and "image_03.tar" in msg and "data_3d_raw" in msg and "đang tải dở" in msg
    assert not (root / "data" / "kitti360" / "raw_t1").exists() and not started


def test_leftover_raw_dir_is_rebuilt(ws):
    root, k, venv, pm, started = ws
    for c in (0, 2, 3):
        (k / "data_2d_raw" / f"{SEQ}_image_0{c}.tar").write_bytes(b"t")
    (k / "data_3d_raw" / f"{SEQ}.tar").write_bytes(b"l")
    stale = root / "data" / "kitti360" / "raw_t1" / "t1"
    stale.mkdir(parents=True)
    (stale / "half.zarr.itar").write_bytes(b"old")
    pm.prepare_drivestudio(CFG, venv)
    assert started["kind"] == "prepare" and not (stale / "half.zarr.itar").exists()
    (root / "data" / "kitti360" / "processed_t1").mkdir()
    with pytest.raises(ValueError, match="processed_t1"):
        pm.prepare_drivestudio(CFG, venv)
