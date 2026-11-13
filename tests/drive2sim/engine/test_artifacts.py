"""Kiểm tra ``drive2sim/engine/artifacts.py``: liệt kê artifact của run và đọc đường cong train.

- Stage có cache trỏ tới ``cache/``, stage khác trỏ tới thư mục stage trong run; stage chưa chạy
  không có đường dẫn.
- ``metrics.json`` (mỗi dòng một JSON) đọc ra PSNR, tổng loss, tổng số Gaussian; khi chạy tiếp,
  số bước đếm lại từ 0 nhưng trục bước vẫn tăng.
"""

import json
from pathlib import Path

from drive2sim.core.schema import RunApproach, RunRecord, StageRecord
from drive2sim.engine.artifacts import run_artifacts, train_curve
from drive2sim.engine.workdir import Workdir


def test_run_artifacts_locations(tmp_path: Path) -> None:
    work = Workdir(tmp_path)
    key = "nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e"
    (work.cache_dir / key).mkdir(parents=True)
    (work.cache_dir / key / "artifact.json").write_text("{}")
    run_dir = work.run_dir("nuscenes_scene-0061", "omnire", "20261112-081000-omnire-a1b2")
    train = run_dir / "d2s" / "stages" / "train.drivestudio"
    (train / "videos").mkdir(parents=True)
    (train / "videos" / "a.mp4").write_bytes(b"0" * 10)
    (train / "log.txt").write_text("x")
    (run_dir / "d2s" / "run.json").write_text("{}")
    record = RunRecord(
        schema_version="d2s-run/0.3",
        run_id="20261112-081000-omnire-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id="omnire", version="0.1.0"),
        level="L1",
        executor="local",
        status="running",
        stages=[
            StageRecord(stage_id="prep.masks_segformer", status="succeeded", cache_key=key),
            StageRecord(stage_id="train.drivestudio", status="running"),
            StageRecord(stage_id="eval.standard", status="pending"),
        ],
    )
    result = run_artifacts(work, record, run_dir)
    prep, train_stage, evaluation = result["stages"]
    assert prep["location"] == "cache" and prep["path"] == f"cache/{key}"
    assert train_stage["location"] == "run" and train_stage["size_bytes"] == 11
    assert [f["name"] for f in train_stage["files"]] == ["videos", "log.txt"]  # thư mục trước
    assert evaluation["path"] is None and evaluation["files"] == []
    assert [f["name"] for f in result["files"]] == ["run.json"]


def test_train_curve_handles_resume(tmp_path: Path) -> None:
    rows = [
        {"iteration": 0, "train_metrics/psnr": 14.0, "losses/rgb": 0.3, "losses/ssim": 0.1,
         "train_stats/gaussian_num_Background": 100, "train_stats/gaussian_num_RigidNodes": 20},
        {"iteration": 500, "train_metrics/psnr": 20.0, "losses/rgb": 0.2},
        {"iteration": 0, "train_metrics/psnr": 21.0},  # chạy tiếp: đếm lại từ 0
        {"iteration": 500, "train_metrics/psnr": 22.0},
    ]  # fmt: skip
    (tmp_path / "metrics.json").write_text("\n".join(json.dumps(r) for r in rows) + "\nhỏng\n")
    curve = train_curve(tmp_path)
    assert [x for x, _ in curve["psnr"]] == [0, 500, 500, 1000]
    assert curve["loss"][0] == [0.0, 0.4] and curve["gaussians"] == [[0.0, 120.0]]
    assert train_curve(tmp_path / "missing") == {}
