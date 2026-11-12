"""Kiểm tra lệnh ``d2s`` qua hàm ``drive2sim.cli.main.main``.

File này dựng một thư mục làm việc tạm có một cảnh mẫu, một thư mục approach tạm có approach giả
``toy`` dùng hai stage giả, rồi kiểm tra:
- ``approach list/show`` đọc được hai approach thật ``omnire`` và ``streetgs``;
- ``split create/list/show`` ghi và đọc split, đặt tên mặc định theo CONTRACTS mục 10.1, từ
  chối tên trùng và camera không có;
- ``approach check`` in danh sách stage và stage nào dùng lại cache;
- ``run start/status/list/resume`` chạy trọn một run giả; ``--set`` đổi được ô và tham số stage;
- ``approach check omnire`` chạy được trên cảnh mẫu với ba stage bọc DriveStudio;
- mã thoát và lỗi dạng JSON: ``2`` khi sai giá trị, ``3`` khi bị khoá, ``1``
  khi không tìm thấy cảnh hoặc chọn Kaggle;
- ``parse_value`` và ``parse_set`` đọc ``--set`` đúng CONTRACTS mục 10.3.
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from drive2sim import paths
from drive2sim.cli.main import main
from drive2sim.cli.output import UsageError, parse_set, parse_value
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.registry import REGISTRY
from drive2sim.engine.stage import Stage, StageContext
from tests.drive2sim.samples import MANIFEST

TOY_APPROACH = """
schema_version: d2s-approach/0.1
approach_id: toy
name: Toy
version: 0.1.0
codebase: drivestudio
track: pinhole
level: L1
requires: {camera_models: [opencv_pinhole], cuboids: optional, split_strategies: [every_nth_frame]}
options:
  init: {choices: [init.cli_lidar, init.cli_depth], default: auto}
stages:
  - {stage: "${options.init}"}
  - {stage: train.cli_toy, params: {steps: 2}}
"""


class ToyParams(BaseModel):
    """Tham số của stage giả ``train.cli_toy``."""

    model_config = ConfigDict(extra="forbid")
    steps: int = 1


def _write(name: str):  # noqa: ANN202
    def run(
        self: Stage, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        out = ctx.output_dir / name
        out.write_text("x")
        return {next(iter(type(self).outputs)): out}

    return run


def _register(stage_id: str, name: str, attrs: dict) -> None:
    if stage_id not in REGISTRY:
        body = {"version": "0.1.0", "run": _write("out.txt"), **attrs}
        REGISTRY.register(stage_id)(type(name, (Stage,), body))


_register(
    "init.cli_lidar",
    "InitCliLidarStage",
    {
        "cacheable": True,
        "uses_lidar": True,
        "inputs": {"s": "scene"},
        "outputs": {"p": "points_init"},
    },
)
_register(
    "init.cli_depth",
    "InitCliDepthStage",
    {"cacheable": True, "inputs": {"s": "scene"}, "outputs": {"p": "points_init"}},
)
_register(
    "train.cli_toy",
    "TrainCliToyStage",
    {
        "inputs": {"p": "points_init", "split": "split"},
        "outputs": {"m": "model"},
        "Params": ToyParams,
    },
)


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files.dump(
        Manifest.model_validate(MANIFEST),
        tmp_path / "data" / MANIFEST["scene_id"] / "manifest.yaml",
    )
    approaches = tmp_path / "approaches"
    approaches.mkdir()
    (approaches / "toy.yaml").write_text(TOY_APPROACH)
    for name in ("omnire.yaml", "streetgs.yaml"):
        (approaches / name).write_text((paths.APPROACHES_DIR / name).read_text())
    monkeypatch.setattr(paths, "APPROACHES_DIR", approaches)
    monkeypatch.setenv("D2S_WORKDIR", str(tmp_path))
    return tmp_path


def run_json(capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, object]:
    code = main([*argv, "--json", "--quiet"])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


SCENE = MANIFEST["scene_id"]


def test_approach_list_and_show(workdir: Path, capsys: pytest.CaptureFixture) -> None:
    code, data = run_json(capsys, "approach", "list")
    assert code == 0
    assert {item["approach_id"] for item in data} == {"omnire", "streetgs", "toy"}
    assert main(["approach", "show", "streetgs"]) == 0
    assert "geer_fastgs: renderer=3dgeer, fastgs=true" in capsys.readouterr().out


def test_split_create_list_show(workdir: Path, capsys: pytest.CaptureFixture) -> None:
    code, data = run_json(
        capsys,
        "split",
        "create",
        SCENE,
        "--strategy",
        "every_nth_frame:4",
        "--cameras",
        "CAM_FRONT",
    )
    assert code == 0
    assert data["split_id"] == "every4_1cams"
    assert (workdir / "data" / SCENE / "splits" / "every4_1cams.json").is_file()
    code, data = run_json(capsys, "split", "create", SCENE, "--strategy", "every_nth_frame:4")
    assert data["split_id"] == "every4_allcams"
    code, data = run_json(capsys, "split", "list", SCENE)
    assert [item["split_id"] for item in data] == ["every4_1cams", "every4_allcams"]
    assert main(["split", "show", SCENE, "every4_1cams"]) == 0
    assert "every4_1cams" in capsys.readouterr().out
    code, data = run_json(
        capsys,
        "split",
        "create",
        SCENE,
        "--strategy",
        "every_nth_frame:4",
        "--cameras",
        "CAM_FRONT",
    )
    assert code == 2 and "đã có" in data["error"]["message"]
    code, data = run_json(
        capsys, "split", "create", SCENE, "--strategy", "every_nth_frame:4", "--cameras", "CAM_X"
    )
    assert code == 2 and "CAM_X" in data["error"]["message"]


@pytest.fixture()
def split_id(workdir: Path) -> str:
    assert (
        main(
            [
                "split",
                "create",
                SCENE,
                "--strategy",
                "every_nth_frame:4",
                "--cameras",
                "CAM_FRONT",
                "--quiet",
            ]
        )
        == 0
    )
    return "every4_1cams"


def test_check_run_status_list_resume(
    workdir: Path, split_id: str, capsys: pytest.CaptureFixture
) -> None:
    capsys.readouterr()
    code, data = run_json(
        capsys, "approach", "check", "toy", "--scene", SCENE, "--split", split_id
    )
    assert code == 0
    assert data["options"] == {"init": "init.cli_lidar"}
    assert [s["cache_hit"] for s in data["stages"]] == [False, False]

    code, data = run_json(
        capsys,
        "run",
        "start",
        "toy",
        "--scene",
        SCENE,
        "--split",
        split_id,
        "--set",
        "init=init.cli_depth",
        "--set",
        "train.cli_toy.steps=5",
    )
    assert code == 0
    assert data["status"] == "succeeded"
    assert data["stages"][1]["params"] == {"steps": 5}
    run_id = data["run_id"]

    code, data = run_json(
        capsys,
        "approach",
        "check",
        "toy",
        "--scene",
        SCENE,
        "--split",
        split_id,
        "--set",
        "init=init.cli_depth",
    )
    assert [s["cache_hit"] for s in data["stages"]] == [True, False]

    code, data = run_json(capsys, "run", "status", run_id)
    assert code == 0 and data["run_id"] == run_id
    code, data = run_json(capsys, "run", "list", "--scene", SCENE)
    assert [item["run_id"] for item in data] == [run_id]
    code, data = run_json(capsys, "run", "resume", run_id)
    assert code == 0 and data["status"] == "succeeded"


def test_exit_codes(workdir: Path, split_id: str, capsys: pytest.CaptureFixture) -> None:
    base = ["--scene", SCENE, "--split", split_id]
    code, data = run_json(capsys, "run", "start", "toy", *base, "--set", "init=bogus")
    assert code == 2 and data["error"]["code"] == "invalid_request"
    code, data = run_json(capsys, "approach", "check", "omnire", *base)
    assert code == 0
    assert [s["stage_id"] for s in data["stages"]] == [
        "prep.masks_segformer",
        "prep.smpl",
        "train.drivestudio",
        "eval.standard",
        "export.ply_world",
    ]
    code, data = run_json(capsys, "run", "start", "toy", *base, "--executor", "kaggle")
    assert code == 1
    code, data = run_json(capsys, "scene", "show", "nuscenes_scene-9999")
    assert code == 1 and data["error"]["code"] == "not_found"
    assert main(["run", "bogus"]) == 2
    capsys.readouterr()

    manifest = dict(MANIFEST, lidars=[], available={"lidar": False, "cuboids": "gt"})
    files.dump(Manifest.model_validate(manifest), workdir / "data" / SCENE / "manifest.yaml")
    code, data = run_json(capsys, "run", "start", "toy", *base, "--set", "init=init.cli_lidar")
    assert code == 3 and data["error"]["code"] == "blocked"
    assert "LiDAR" in data["error"]["details"]["reasons"][0]["message"]


def test_scene_list_and_show(workdir: Path, capsys: pytest.CaptureFixture) -> None:
    code, data = run_json(capsys, "scene", "list")
    assert code == 0 and [item["scene_id"] for item in data] == [SCENE]
    assert main(["scene", "show", SCENE]) == 0
    assert "CAM_FISH" in capsys.readouterr().out


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("10", 10),
        ("0.5", 0.5),
        ("a,b", ["a", "b"]),
        ('"10"', "10"),
        ("3dgeer", "3dgeer"),
    ],
)
def test_parse_value(raw: str, expected: object) -> None:
    assert parse_value(raw) == expected


def test_parse_set() -> None:
    options, stage_params = parse_set(["renderer=3dgeer", "train.drivestudio.num_iters=1000"])
    assert options == {"renderer": "3dgeer"}
    assert stage_params == {"train.drivestudio": {"num_iters": 1000}}
    with pytest.raises(UsageError):
        parse_set(["renderer"])


def test_schema_export(tmp_path: Path) -> None:
    assert main(["schema", "export", "--out", str(tmp_path), "--quiet"]) == 0
    run_schema = json.loads((tmp_path / "run.schema.json").read_text())
    assert run_schema["title"] == "RunRecord"
    assert "stages" in run_schema["properties"]


def test_housekeeping_commands(
    workdir: Path, split_id: str, capsys: pytest.CaptureFixture
) -> None:
    base = ["--scene", SCENE, "--split", split_id]
    code, data = run_json(capsys, "run", "start", "toy", *base)
    run_id = data["run_id"]

    code, data = run_json(capsys, "report", "--scene", SCENE)
    assert code == 0 and data[0]["run_id"] == run_id
    assert main(["report", "--csv"]) == 0
    assert capsys.readouterr().out.startswith("run_id,")

    code, data = run_json(capsys, "cache", "list")
    assert code == 0 and data[0]["used_by"] == [run_id]
    code, data = run_json(capsys, "run", "cancel", run_id)
    assert code == 0 and data["status"] == "succeeded"  # run đã xong: huỷ không đổi gì

    code, data = run_json(capsys, "split", "delete", SCENE, split_id, "--yes")
    assert code == 1 and data["error"]["code"] == "conflict"
    # không có terminal và thiếu --yes thì từ chối xoá
    code, data = run_json(capsys, "run", "delete", run_id)
    assert code == 2 and "--yes" in data["error"]["message"]
    code, data = run_json(capsys, "run", "delete", run_id, "--yes")
    assert code == 0 and data["removed"]
    code, data = run_json(capsys, "cache", "gc", "--yes")
    assert code == 0 and len(data["removed"]) == 1
    code, data = run_json(capsys, "split", "delete", SCENE, split_id, "--yes")
    assert code == 0

    code, data = run_json(capsys, "scene", "validate", SCENE)
    assert data["scene_id"] == SCENE
    code, data = run_json(capsys, "scene", "delete", SCENE, "--yes")
    assert code == 0 and any(path.startswith("data/") for path in data["removed"])
    assert not (workdir / "data" / SCENE).exists()


def test_export_unknown_stage_is_reported(
    workdir: Path, split_id: str, capsys: pytest.CaptureFixture
) -> None:
    code, data = run_json(capsys, "run", "start", "toy", "--scene", SCENE, "--split", split_id)
    code, data = run_json(capsys, "export", "preview", data["run_id"])
    assert code == 1 and "export.preview" in data["error"]["message"]
