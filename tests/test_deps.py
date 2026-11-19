"""Library catalog: status checks against a fake venv, and the install steps each entry plans."""
import os
from pathlib import Path
import sys

import pytest

import fisheye_fusion.deps as deps_mod
from fisheye_fusion.dataset_profiles import public
from fisheye_fusion.deps import CATALOG, Deps
from fisheye_fusion.pipeline import PipelineManager


@pytest.fixture
def env(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    venv = tmp_path / "ds"
    (venv / "bin").mkdir(parents=True)
    os.symlink(sys.executable, venv / "bin" / "python")      # this interpreter: has pytest, lacks gsplat etc.
    monkeypatch.setattr(deps_mod, "nvcc_version", lambda: "12.4")
    monkeypatch.setattr(deps_mod, "_cuda_home", lambda: "/usr/local/cuda-12.4")
    pm = PipelineManager(ws)
    pm.discovery.eval_venv = tmp_path / "no-eval-venv"            # independent of this machine's ~/.fflab
    started = {}
    monkeypatch.setattr(pm, "_start", lambda kind, out, steps, env, info: started.update(kind=kind, steps=steps) or {"job_id": "x"})
    return Deps(ws, pm, venv), started


def test_catalog_is_consistent():
    ids = {d["id"] for d in CATALOG}
    assert len(ids) == len(CATALOG)
    for d in CATALOG:
        assert d["links"] and all(len(link) == 2 for link in d["links"])
        assert set(d.get("requires", [])) <= ids
    assert {p["sdk"] for p in public() if p["sdk"]} <= ids


def test_status_reports_missing_modules(env):
    deps, _ = env
    items = {d["id"]: d for d in deps.status()["items"]}
    assert not items["ds-cuda"]["ok"] and "gsplat" in items["ds-cuda"]["missing"]
    assert items["gsplat-eval"]["detail"] == "chưa có môi trường"
    assert "pip" not in items["sdk-nuscenes"] and items["node"]["installable"] is False


def test_install_plans_steps(env):
    deps, started = env
    with pytest.raises(ValueError, match="trước"):
        deps.install("ds-cuda")                     # ds-env (torch, omegaconf…) is not complete in this venv
    deps.install("ds-env")
    assert started["kind"] == "deps-ds-env" and "--nvcc" in started["steps"][0][1]
    deps.install("uv")
    assert started["steps"][0][1][-1] == "uv"
    with pytest.raises(ValueError):
        deps.install("node")
    with pytest.raises(ValueError):
        deps.install("nope")


def test_deps_setup_pip_args(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("deps_setup", Path(__file__).resolve().parents[1] / "tools" / "deps_setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}
    monkeypatch.setattr(mod, "pip", lambda py, *args, env=None: seen.update(py=py, args=args))
    monkeypatch.setattr(sys, "argv", ["deps_setup.py", "pip", "--venv", "/v", "--", "a==1", "b>=1,<2"])
    assert mod.main() == 0 and seen["args"] == ("a==1", "b>=1,<2") and str(seen["py"]) == "/v/bin/python"
