"""Kiểm tra giao tiếp stage với worker XSIM và phạm vi của fingerprint checkpoint."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, JobResult, JobSpec
from drive2sim.external.xsim import worker
from drive2sim.stages import xsim_worker
from drive2sim.stages.train import xsim


def test_stage_validates_current_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run có bản chụp mask đọc cảnh qua thư mục ảo, không nằm ở <workdir>/data/<cảnh>: gốc của
    # job vẫn phải là workdir (lỗi đã gặp trên Kaggle: relative_to sai gốc).
    scene = tmp_path / ".runtime" / "smoke" / "scene_view" / "000"
    observations = tmp_path / "cache" / "prep.observations-x"
    out = tmp_path / "runs" / "train.xsim"
    out.mkdir(parents=True)
    observations.mkdir(parents=True)
    (observations / "observations.json").write_text(json.dumps({"smpl": False}))
    peaks, progress = [], []
    ctx = SimpleNamespace(
        scene=SimpleNamespace(root=scene),
        workdir=tmp_path,
        output_dir=out,
        run_id="smoke",
        should_stop=lambda: False,
        report_peak_vram=peaks.append,
        progress=lambda value, message: progress.append(value),
    )

    def native(job: JobSpec, root: Path, output: Path) -> int:
        assert job.inputs == {"observations": "cache/prep.observations-x"}
        assert root == tmp_path
        (output / "checkpoint.pt").write_bytes(b"fixture")
        return 37

    def launch(command: list[str], log: Path, **kwargs: object) -> None:
        assert command[0] == "/worker/python"
        assert not (out / "result.json").exists()
        log.write_text("XSIM step=20/40 loss=0.1\n")
        assert kwargs["should_stop"]() is False
        # Kernel Jupyter của Kaggle đặt MPLBACKEND inline; venv XSIM không có backend đó.
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        worker.execute(out / "job.json", out / "result.json", tmp_path)

    monkeypatch.setenv("D2S_XSIM_PYTHON", "/worker/python")
    monkeypatch.setattr(worker, "_train", native)
    monkeypatch.setattr(xsim_worker, "run_logged", launch)
    stage = xsim.TrainXsimStage()
    params = xsim.TrainXsimParams(num_iters=40)
    assert stage.run(ctx, {"observations": observations}, params) == {"model": out}
    assert peaks == [37] and progress == [0.5]
    monkeypatch.setattr(xsim_worker, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="result.json"):
        stage.run(ctx, {"observations": observations}, params)
    job = files.load(out / "job.json", JobSpec)
    job.inputs["observations"] = "../wrong"
    files.dump(job, out / "job.json")
    with pytest.raises(ValueError, match="tương đối"):
        worker.execute(out / "job.json", out / "result.json", tmp_path)
    assert files.load(out / "result.json", JobResult).status == "failed"


def test_fingerprint_ignores_runtime_params_but_not_model_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "XSIM"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "waymo_default.yaml").write_text("nodes: []\n")
    observations = tmp_path / "cache" / "obs"
    observations.mkdir(parents=True)
    (observations / "artifact.json").write_text("{}")
    (observations / "observations.json").write_text(json.dumps({"format": "x"}))
    monkeypatch.setattr(
        subprocess, "check_output", lambda command, **kwargs: "abc" if kwargs else b""
    )

    def digest(**changes: object) -> str:
        params = xsim.TrainXsimParams(**changes).model_dump()
        job = JobSpec(
            schema_version=SCHEMA_VERSIONS["job"],
            job_id="run/train.xsim",
            kind="stage",
            stage_id="train.xsim",
            params=params,
            inputs={"observations": "cache/obs"},
            output_dir="runs/x",
            env="xsim",
        )
        return worker.fingerprint(job, tmp_path, source)

    base = digest()
    assert digest(num_iters=10, save_every=5, preload_images=True, num_workers=4) == base
    assert digest(lidar_policy="train") != base
    assert digest(seed=1) != base
    assert "scoring.py" not in worker.MODEL_MODULES and "videos.py" not in worker.MODEL_MODULES


def test_smpl_model_is_converted_outside_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations = tmp_path / "work" / "cache" / "obs"
    observations.mkdir(parents=True)
    (observations / "observations.json").write_text(json.dumps({"smpl": True}))
    model = tmp_path / "SMPL_NEUTRAL.pkl"
    model.write_bytes(b"pkl")
    commands = []

    def convert(command: list[str], log: Path, **kwargs: object) -> None:
        commands.append(command)
        target = Path(command[command.index("--target") + 1])
        target.parent.mkdir(parents=True)
        target.write_bytes(b"pth")

    monkeypatch.setenv("D2S_SMPL_MODEL", str(model))
    monkeypatch.setenv("D2S_XSIM_DIR", str(tmp_path / "xsim"))
    monkeypatch.setattr(xsim_worker, "run_logged", convert)
    ctx = SimpleNamespace(output_dir=tmp_path, should_stop=lambda: False)
    env = xsim_worker.smpl_environment(ctx, observations, tmp_path / "work")
    assert env == {"SMPL_PATH": str(tmp_path / "xsim" / "smpl" / "SMPL_NEUTRAL.pth")}
    assert not env["SMPL_PATH"].startswith(str(tmp_path / "work"))
    assert xsim_worker.smpl_environment(ctx, observations, tmp_path / "work") == env
    assert len(commands) == 1


def test_native_config_env_fills_unused_waymo_paths_without_overriding() -> None:
    """Config gốc XSIM đòi WAYMO_ROOT_PATH...: thiếu thì jetcon từ chối đọc (lỗi trên Kaggle)."""
    from drive2sim.external.xsim import NATIVE_CONFIG_ENV, set_native_config_env

    environ = {"WAYMO_CACHE_PATH": "/data/cache"}
    set_native_config_env(environ)
    assert set(NATIVE_CONFIG_ENV) == {
        "WAYMO_ROOT_PATH",
        "WAYMO_CACHE_PATH",
        "WAYMO_SMPL_DATA_PATH",
    }
    assert environ["WAYMO_ROOT_PATH"].startswith("/nonexistent/")
    assert environ["WAYMO_CACHE_PATH"] == "/data/cache"


def test_pytoast_patch_replaces_the_failing_mask_assignment() -> None:
    """Bản vá pytoast bỏ out_quats[invalid_mask, 0] = 1 (IndexError trên Kaggle, PyTorch 2.8)."""
    from drive2sim import paths

    patch = (paths.PACKAGE_DIR / "external" / "xsim" / "patches" / "pytoast.patch").read_text()
    assert "-        out_quats[invalid_mask, 0] = 1" in patch
    assert (
        "+        out_quats = torch.where(invalid_mask.unsqueeze(-1), identity, out_quats)"
        in patch
    )
    install = (paths.PACKAGE_DIR.parent / "envs" / "install_xsim.sh").read_text()
    assert 'apply "$PYTOAST_PATCH"' in install
