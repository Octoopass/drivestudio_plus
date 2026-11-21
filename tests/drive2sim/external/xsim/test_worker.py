"""Kiểm tra giao tiếp stage với worker XSIM ở cả nhánh thành công và lỗi."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from drive2sim.core import files
from drive2sim.core.schema import JobResult, JobSpec
from drive2sim.external.xsim import worker
from drive2sim.stages import xsim_worker
from drive2sim.stages.train import xsim


def test_stage_validates_current_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "data" / "scene"
    out = tmp_path / "runs" / "train.xsim"
    out.mkdir(parents=True)
    peaks = []
    ctx = SimpleNamespace(
        scene=SimpleNamespace(root=scene),
        output_dir=out,
        run_id="smoke",
        should_stop=lambda: False,
        report_peak_vram=peaks.append,
    )

    def native(job: JobSpec, root: Path, output: Path) -> int:
        assert job.inputs["scene"] == "data/scene"
        assert root == tmp_path
        (output / "checkpoint.pt").write_bytes(b"fixture")
        return 37

    def launch(command: list[str], log: Path, **kwargs: object) -> None:
        assert command[0] == "/worker/python"
        assert not (out / "result.json").exists()
        worker.execute(out / "job.json", out / "result.json", tmp_path)

    monkeypatch.setenv("D2S_XSIM_PYTHON", "/worker/python")
    monkeypatch.setattr(worker, "_train", native)
    monkeypatch.setattr(xsim_worker, "run_logged", launch)
    stage = xsim.TrainXsimStage()
    assert stage.run(ctx, {}, xsim.TrainXsimParams()) == {"model": out}
    assert peaks == [37]
    monkeypatch.setattr(xsim_worker, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="result.json"):
        stage.run(ctx, {}, xsim.TrainXsimParams())
    job = files.load(out / "job.json", JobSpec)
    job.inputs["scene"] = "../wrong"
    files.dump(job, out / "job.json")
    with pytest.raises(ValueError, match="tương đối"):
        worker.execute(out / "job.json", out / "result.json", tmp_path)
    assert files.load(out / "result.json", JobResult).status == "failed"
