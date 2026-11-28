"""Approach bị tắt (trường ``disabled``, hiện là XSIM): ẩn, chặn tạo, chạy, đẩy run và cài.

Approach khác và run cũ không bị ảnh hưởng.
"""

import json
from pathlib import Path

import pytest

from drive2sim.core import files
from drive2sim.core.schema import Approach, RunRecord
from drive2sim.engine import env_install
from drive2sim.engine.approaches import (
    ApproachDisabledError,
    list_approach_ids,
    load_approach,
)
from drive2sim.engine.resolve import resolve
from drive2sim.engine.run import execute_run
from drive2sim.engine.workdir import Workdir
from tests.drive2sim.samples import MANIFEST, SPLIT


def test_xsim_is_disabled_and_hidden_but_still_loadable() -> None:
    assert "xsim" not in list_approach_ids()
    assert "xsim" in list_approach_ids(include_disabled=True)
    xsim = load_approach("xsim")  # run cũ vẫn đọc được file approach
    assert xsim.disabled and "24 GB" in xsim.disabled
    # Approach khác không bị tắt.
    assert {"omnire", "pvg", "streetgs", "nurec", "mcgs"} <= set(list_approach_ids())


def test_resolve_refuses_a_disabled_approach() -> None:
    from drive2sim.core.schema import Manifest, Split

    with pytest.raises(ApproachDisabledError, match="đã tắt"):
        resolve(
            load_approach("xsim"), Manifest.model_validate(MANIFEST), Split.model_validate(SPLIT)
        )


def test_execute_run_refuses_an_existing_run_of_a_disabled_approach(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "d2s").mkdir(parents=True)
    record = {
        "schema_version": "d2s-run/0.1",
        "run_id": "20261128-000000-xsim-abcd",
        "scene_id": MANIFEST["scene_id"],
        "split_id": SPLIT["split_id"],
        "approach": {"approach_id": "xsim", "version": "0.2.0", "options": {}},
        "level": "L1",
        "executor": "local",
        "status": "failed",
        "stages": [],
    }
    files.dump(RunRecord.model_validate(record), run_dir / "d2s" / "run.json")
    before = (run_dir / "d2s" / "run.json").read_text()
    with pytest.raises(ApproachDisabledError):
        execute_run(run_dir, Workdir(tmp_path / "work"))
    assert (run_dir / "d2s" / "run.json").read_text() == before  # không động vào run cũ


def test_install_of_an_environment_used_only_by_disabled_approaches_is_refused(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "envs").mkdir(parents=True)
    for name in ("install_xsim.sh", "install_main.sh", "install_nurec.sh"):
        (repo / "envs" / name).write_text("exit 0\n")
    flow = env_install.InstallFlow(
        tmp_path / "logs", repo=repo, gpus=lambda: [("GPU", 1, "1")], python_version=(3, 10)
    )
    assert env_install.disabled_env_reason("xsim")
    assert env_install.disabled_env_reason("main") is None
    assert env_install.disabled_env_reason("nurec") is None
    with pytest.raises(ValueError, match="không cài qua tool"):
        flow.preflight("xsim")
    assert flow.preflight("main") == repo / "envs" / "install_main.sh"


def test_disabled_field_is_optional_for_other_approaches() -> None:
    for approach_id in list_approach_ids():
        assert load_approach(approach_id).disabled is None
    assert Approach.model_fields["disabled"].default is None
    assert json.loads(load_approach("omnire").model_dump_json())["disabled"] is None
