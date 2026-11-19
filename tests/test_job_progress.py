"""Per-step timing / progress of pipeline jobs, and which Instant NuRec setup steps are planned."""
import sys
import time

import pytest

import fisheye_fusion.pipeline as pl
from fisheye_fusion.pipeline import PipelineManager

STEP = "import sys, time\nfor i in range(1, 5):\n    print(f'@@PROGRESS {i} 4', flush=True); time.sleep(0.05)\nprint('done', flush=True)\n"


def wait(pm, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        s = pm.status(job_id)
        if s["state"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError(pm.status(job_id))


def test_plan_timing_and_progress(tmp_path):
    pm = PipelineManager(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    job = pm._start("demo", out, [("a", [sys.executable, "-c", STEP], 30), ("b", [sys.executable, "-c", "print(1)"])],
                    None, {})
    assert [p["label"] for p in job["plan"]] == ["a", "b"] and job["plan"][0]["est"] == 30
    s = wait(pm, job["job_id"])
    assert s["state"] == "succeeded"
    assert all(p["state"] == "done" and p["ended"] >= p["started"] for p in s["plan"])
    assert "@@PROGRESS" not in s["log"] and "done" in s["log"]
    assert s["plan"][1]["est"] is None and s["ended"] >= s["started"]


def test_busy_step_reports_fraction(tmp_path):
    pm = PipelineManager(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    code = "import time\nprint('@@PROGRESS 25 100', flush=True)\ntime.sleep(2)\n"
    job = pm._start("demo", out, [("slow", [sys.executable, "-c", code], 10)], None, {})
    for _ in range(100):
        plan = pm.status(job["job_id"])["plan"][0]
        if plan.get("frac") is not None:
            break
        time.sleep(0.05)
    assert plan["state"] == "busy" and plan["frac"] == pytest.approx(0.25)
    pm.cancel(job["job_id"])


def test_nurec_setup_plans_only_missing_parts(tmp_path, monkeypatch):
    pm = PipelineManager(tmp_path)
    started = {}
    monkeypatch.setattr(pm, "_start", lambda kind, out, steps, env, info: started.update(kind=kind, steps=steps, info=info) or {"ok": 1})
    pm.nurec_setup(["pa-multiview"])
    assert [s[0].split(" ")[0] for s in started["steps"]] == ["Tải", "Cài", "Tải"]
    assert "pa-multiview" in started["steps"][-1][1]
    repo = tmp_path / "backends" / "instant-nurec"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("")
    (repo / "run_inference.py").write_text("")
    pth = repo / "checkpoints" / "pth"
    pth.mkdir(parents=True)
    (pth / "instant_nurec_pa_multiview_1.1.0.pth").write_bytes(b"x")
    with pytest.raises(ValueError):
        pm.nurec_setup(["pa-multiview"])
    pm.nurec_setup(["pa-multiview", "pq-front"])
    assert len(started["steps"]) == 1 and "pq-front" in started["steps"][0][1][-1]
    with pytest.raises(ValueError):
        pm.nurec_setup(["nope"])
    rep = pm.discovery.report()["nurec"]
    assert rep["source"] and rep["venv"] and "pa-multiview" in rep["available"] and pl.NUREC_WEIGHTS["pq-front"] > 0
