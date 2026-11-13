"""Kiểm tra việc tạo và chạy run trong ``drive2sim/engine/run.py`` và ``cache.py``.

File này dùng hai stage giả (``init.toy`` có cache, ``train.toy`` không cache) ghi file thật vào
một thư mục làm việc tạm, và kiểm tra:
- run chạy xong ghi đúng ``run.json``, ``env.json``, ``artifact.json``, file ``done`` và phát đủ
  sự kiện;
- hai run cùng tham số dùng chung cache; đổi tham số thì khoá cache đổi;
- stage lỗi làm run ``failed`` với mã ``stage_failed``, traceback vào ``d2s/log.txt``; chạy tiếp
  sau khi sửa thì chỉ chạy lại stage lỗi và giữ số đo của stage đã xong;
- ngắt bằng Ctrl-C ghi ``cancelled``; stage có cache bị lỗi không để lại thư mục ``.partial``;
- ``CacheStore`` từ chối khi tiến trình khác đang ghi và dọn ``.partial`` của tiến trình đã chết;
- ``Workdir`` tìm được run và báo lỗi dễ hiểu khi thiếu cảnh, split hoặc run.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from drive2sim.core import files
from drive2sim.core.schema import Approach, ArtifactRecord, Manifest, RunRecord, Split
from drive2sim.engine.cache import CacheBusyError, CacheStore
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.resolve import resolve
from drive2sim.engine.run import (
    create_run,
    execute_run,
    request_cancel,
    reset_stage,
    run_is_active,
)
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.engine.workdir import Workdir, WorkdirError
from drive2sim.stages.commands import run_logged
from tests.drive2sim.samples import MANIFEST, SPLIT

APPROACH = {
    "schema_version": "d2s-approach/0.1",
    "approach_id": "toy",
    "name": "Toy",
    "version": "0.1.0",
    "codebase": "drivestudio",
    "track": "pinhole",
    "level": "L1",
    "requires": {
        "camera_models": ["opencv_pinhole"],
        "cuboids": "optional",
        "split_strategies": ["every_nth_frame"],
    },
    "stages": [
        {"stage": "init.toy", "params": {"num_points": 3}},
        {"stage": "train.toy"},
    ],
}

# Số lần mỗi stage thật sự chạy, và cờ để train.toy ném lỗi.
CALLS: dict[str, int] = {}
BEHAVIOUR: dict[str, object] = {}


class InitParams(BaseModel):
    """Tham số của ``init.toy``."""

    model_config = ConfigDict(extra="forbid")
    num_points: int = 1


def init_run(
    self: Stage, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
) -> dict[str, Path]:
    CALLS["init.toy"] = CALLS.get("init.toy", 0) + 1
    out = ctx.output_dir / "points_init.npz"
    out.write_text(f"{params.num_points} points of {ctx.scene.scene_id}")  # type: ignore[attr-defined]
    return {"points": out}


def train_run(
    self: Stage, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
) -> dict[str, Path]:
    CALLS["train.toy"] = CALLS.get("train.toy", 0) + 1
    ctx.progress(0.5, "half")
    ctx.log_metric("loss", 0.1, 10)
    if BEHAVIOUR.get("fail"):
        raise RuntimeError("boom")
    if BEHAVIOUR.get("interrupt"):
        raise KeyboardInterrupt
    if BEHAVIOUR.get("wrong_output"):
        return {"other": ctx.output_dir}
    source = (inputs["points"] / "points_init.npz").read_text()
    out = ctx.output_dir / "model.txt"
    out.write_text(f"trained on {source}")
    return {"model": out}


def make_registry() -> StageRegistry:
    registry = StageRegistry()
    init_body = {
        "version": "0.1.0",
        "cacheable": True,
        "uses_lidar": True,
        "inputs": {"scene": "scene"},
        "outputs": {"points": "points_init"},
        "Params": InitParams,
        "run": init_run,
    }
    train_body = {
        "version": "0.1.0",
        "inputs": {"points": "points_init", "split": "split"},
        "outputs": {"model": "model"},
        "run": train_run,
    }
    registry.register("init.toy")(type("InitToyStage", (Stage,), init_body))
    registry.register("train.toy")(type("TrainToyStage", (Stage,), train_body))
    return registry


@pytest.fixture()
def workdir(tmp_path: Path) -> Workdir:
    CALLS.clear()
    BEHAVIOUR.clear()
    work = Workdir(tmp_path)
    scene_id = MANIFEST["scene_id"]
    files.dump(Manifest.model_validate(MANIFEST), work.scene_dir(scene_id) / "manifest.yaml")
    files.dump(Split.model_validate(SPLIT), work.split_path(scene_id, SPLIT["split_id"]))
    return work


def new_run(work: Workdir, registry: StageRegistry, approach: dict = APPROACH) -> Path:
    scene = work.load_scene(MANIFEST["scene_id"])
    split = work.load_split(MANIFEST["scene_id"], SPLIT["split_id"])
    plan = resolve(Approach.model_validate(approach), scene.manifest, split, registry=registry)
    return create_run(plan, work, scene, split)


def test_run_succeeds_and_writes_contracts_files(workdir: Workdir) -> None:
    registry = make_registry()
    events: list[dict] = []
    run_dir = new_run(workdir, registry)
    record = execute_run(run_dir, workdir, registry, events.append)

    assert record.status == "succeeded"
    assert [s.status for s in record.stages] == ["succeeded", "succeeded"]
    assert record.lidar_usage == ["init.toy"]
    assert record.stages[0].params == {"num_points": 3}
    # run.json trên đĩa khớp kết quả trả về
    assert files.load(run_dir / "d2s" / "run.json", RunRecord) == record

    init_dir = workdir.cache_dir / record.stages[0].cache_key
    artifact = files.load(init_dir / "artifact.json", ArtifactRecord)
    assert artifact.artifact_type == "points_init"
    assert artifact.files == ["points_init.npz"]
    assert artifact.inputs == [f"scene:{MANIFEST['scene_id']}@0.1.0+test"]

    train_dir = run_dir / "d2s" / "stages" / "train.toy"
    assert (train_dir / "done").is_file()
    assert "3 points" in (train_dir / "model.txt").read_text()
    assert (run_dir / "d2s" / "env.json").is_file()

    kinds = [event["event"] for event in events]
    assert kinds[0] == "status" and kinds[-1] == "done"
    assert {"progress", "metric"} <= set(kinds)


def test_quick_flag_written_to_run_json(workdir: Workdir) -> None:
    scene = workdir.load_scene(MANIFEST["scene_id"])
    split = workdir.load_split(MANIFEST["scene_id"], SPLIT["split_id"])
    plan = resolve(
        Approach.model_validate(APPROACH), scene.manifest, split, registry=make_registry()
    )
    plan.quick = True
    run_dir = create_run(plan, workdir, scene, split)
    record = files.load(run_dir / "d2s" / "run.json", RunRecord)
    assert record.approach.quick is True
    assert record.schema_version == "d2s-run/0.3"


def test_cache_shared_between_runs(workdir: Workdir) -> None:
    registry = make_registry()
    first = execute_run(new_run(workdir, registry), workdir, registry)
    record = execute_run(new_run(workdir, registry), workdir, registry)
    assert CALLS == {"init.toy": 1, "train.toy": 2}
    assert record.stages[0].cache_hit is True
    assert record.stages[0].duration_s == 0
    # run sau ghi thời gian lần chạy gốc, đọc từ artifact.json trong cache
    original = record.stages[0].reused_duration_s
    assert original is not None and 0 <= original <= first.stages[0].duration_s


def test_cache_key_changes_with_params(workdir: Workdir) -> None:
    registry = make_registry()
    first = execute_run(new_run(workdir, registry), workdir, registry)
    approach = copy.deepcopy(APPROACH)
    approach["stages"][0]["params"]["num_points"] = 5
    second = execute_run(new_run(workdir, registry, approach), workdir, registry)
    assert first.stages[0].cache_key != second.stages[0].cache_key
    assert CALLS["init.toy"] == 2


def test_failure_then_resume(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    BEHAVIOUR["fail"] = True
    record = execute_run(run_dir, workdir, registry)
    assert record.status == "failed"
    assert record.stages[1].status == "failed"
    assert record.stages[1].error.code == "stage_failed"
    assert "boom" in record.stages[1].error.message
    assert "RuntimeError: boom" in (run_dir / "d2s" / "log.txt").read_text()
    assert not (run_dir / "d2s" / "stages" / "train.toy" / "done").exists()
    init_duration = record.stages[0].duration_s
    failed_train_duration = record.stages[1].duration_s

    BEHAVIOUR.clear()
    record = execute_run(run_dir, workdir, registry)
    assert record.status == "succeeded"
    assert record.stages[1].error is None
    assert CALLS == {"init.toy": 1, "train.toy": 2}
    # stage đã xong ở lần trước giữ nguyên số đo
    assert record.stages[0].duration_s == init_duration
    # stage chạy lại cộng dồn thời gian của lần lỗi và lần chạy tiếp
    assert record.stages[1].duration_s >= failed_train_duration
    assert record.stages[0].cache_hit is False


def test_done_stage_skipped_on_resume(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    execute_run(run_dir, workdir, registry)
    execute_run(run_dir, workdir, registry)
    assert CALLS == {"init.toy": 1, "train.toy": 1}


def test_wrong_outputs_fail_the_stage(workdir: Workdir) -> None:
    registry = make_registry()
    BEHAVIOUR["wrong_output"] = True
    record = execute_run(new_run(workdir, registry), workdir, registry)
    assert record.status == "failed"
    assert "khai báo" in record.stages[1].error.message


def test_interrupt_marks_cancelled(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    BEHAVIOUR["interrupt"] = True
    with pytest.raises(KeyboardInterrupt):
        execute_run(run_dir, workdir, registry)
    record = files.load(run_dir / "d2s" / "run.json", RunRecord)
    assert record.status == "cancelled"
    assert record.stages[1].status == "cancelled"


def test_failed_cacheable_stage_leaves_no_partial(workdir: Workdir) -> None:
    registry = make_registry()

    def broken(*args: object) -> dict[str, Path]:
        raise ValueError("bad init")

    registry.get("init.toy").run = broken  # type: ignore[method-assign]
    record = execute_run(new_run(workdir, registry), workdir, registry)
    assert record.stages[0].status == "failed"
    assert list(workdir.cache_dir.rglob("*.partial")) == []


def test_cache_store_busy_and_stale(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    key = "nuscenes_scene-0061/init.toy@0.1.0-0123abcd"
    partial = store.begin(key)
    # tiến trình khác còn sống đang giữ khoá
    (partial / ".owner").write_text(str(os.getppid()))
    with pytest.raises(CacheBusyError):
        store.begin(key)
    # tiến trình đã chết: thư mục .partial bỏ dở được dọn và tạo lại
    (partial / ".owner").write_text("999999999")
    (partial / "junk").write_text("x")
    partial = store.begin(key)
    assert not (partial / "junk").exists()
    (partial / "artifact.json").write_text(json.dumps({}))
    final = store.commit(partial)
    assert store.lookup(key) == final
    assert not (final / ".owner").exists()


def test_workdir_lookups(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    assert workdir.find_run(run_dir.name) == run_dir
    with pytest.raises(WorkdirError):
        workdir.find_run("20260101-000000-toy-ffff")
    with pytest.raises(WorkdirError, match="ingest"):
        workdir.load_scene("nuscenes_scene-9999")
    with pytest.raises(WorkdirError, match="split create"):
        workdir.load_split(MANIFEST["scene_id"], "every4_front1")


def test_cancel_before_start_and_resume(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    run_record = request_cancel(run_dir)
    assert run_record.status == "pending"  # chưa chạy: yêu cầu nằm chờ trong d2s/cancel
    record = execute_run(run_dir, workdir, registry)
    assert record.status == "cancelled"
    assert CALLS == {}
    assert not (run_dir / "d2s" / "cancel").exists()
    assert not (run_dir / "d2s" / "pid").exists()
    assert execute_run(run_dir, workdir, registry).status == "succeeded"


def test_cancel_stops_running_subprocess(workdir: Workdir) -> None:
    registry = make_registry()

    def slow(self: Stage, ctx: StageContext, inputs: dict, params: BaseModel) -> dict:
        (ctx.output_dir / ".." / ".." / "cancel").resolve().write_text("now")
        run_logged(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            ctx.output_dir / "log.txt",
            should_stop=ctx.should_stop,
        )
        return {}

    registry.get("train.toy").run = slow  # type: ignore[method-assign]
    run_dir = new_run(workdir, registry)
    started = time.monotonic()
    record = execute_run(run_dir, workdir, registry)
    assert time.monotonic() - started < 15
    assert record.status == "cancelled"
    assert record.stages[1].status == "cancelled"
    assert "đã dừng" in (run_dir / "d2s" / "stages" / "train.toy" / "log.txt").read_text()


def test_cancel_stale_running_run(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    record = files.load(run_dir / "d2s" / "run.json", RunRecord)
    record.status = "running"
    record.stages[0].status = "running"
    files.dump(record, run_dir / "d2s" / "run.json")
    (run_dir / "d2s" / "pid").write_text("999999999")  # tiến trình đã chết
    assert not run_is_active(run_dir)
    cancelled = request_cancel(run_dir)
    assert cancelled.status == "cancelled" and cancelled.stages[0].status == "cancelled"


def test_reset_stage_reruns_only_that_stage(workdir: Workdir) -> None:
    registry = make_registry()
    run_dir = new_run(workdir, registry)
    execute_run(run_dir, workdir, registry)
    reset_stage(run_dir, "train.toy", registry)
    record = execute_run(run_dir, workdir, registry)
    assert record.status == "succeeded"
    assert CALLS == {"init.toy": 1, "train.toy": 2}
