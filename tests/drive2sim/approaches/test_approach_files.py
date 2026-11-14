"""Kiểm tra các file approach có sẵn trong ``drive2sim/approaches/``.

File này kiểm tra:
- mọi file approach đọc được và đúng schema ``d2s-approach``;
- tên file trùng ``approach_id``;
- mỗi mô hình camera approach khai báo đều có tổ hợp mục cấu hình nhận được;
- với approach dòng DriveStudio, mọi tổ hợp mục cấu hình mà ràng buộc cho phép đều có file config
  method tương ứng trong ``CONFIG_FILES``, để người dùng không chọn được tổ hợp nào mà lúc chạy
  mới báo thiếu config.
"""

import itertools
from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Approach, contains_value
from drive2sim.engine.resolve import camera_models_for
from drive2sim.stages.train.drivestudio import CONFIG_FILES

APPROACH_FILES = sorted(paths.APPROACHES_DIR.glob("*.yaml"))


def test_approach_files_exist() -> None:
    assert {path.stem for path in APPROACH_FILES} >= {"omnire", "streetgs"}


@pytest.mark.parametrize("path", APPROACH_FILES, ids=lambda path: path.stem)
def test_approach_file_valid(path: Path) -> None:
    approach = files.load(path, Approach)
    assert approach.approach_id == path.stem


def allowed_combinations(approach: Approach) -> list[dict[str, object]]:
    """Liệt kê mọi tổ hợp mục cấu hình không vi phạm ràng buộc nào."""
    names = list(approach.options)
    result = []
    for values in itertools.product(*(approach.options[name].choices for name in names)):
        chosen = dict(zip(names, values))
        violated = any(
            all(
                contains_value([chosen[name]], value)
                for name, value in {**constraint.when, **constraint.forbid}.items()
            )
            for constraint in approach.constraints
        )
        if not violated:
            result.append(chosen)
    return result


@pytest.mark.parametrize("path", APPROACH_FILES, ids=lambda path: path.stem)
def test_allowed_combinations_have_config(path: Path) -> None:
    approach = files.load(path, Approach)
    train = next(entry for entry in approach.stages if entry.stage == "train.drivestudio")
    method = train.params["method"]
    combinations = allowed_combinations(approach)
    assert combinations
    for chosen in combinations:
        key = (method, chosen["renderer"], chosen["campose"], chosen["fastgs"])
        assert key in CONFIG_FILES, f"{path.stem}: không có config cho {chosen}"


@pytest.mark.parametrize("path", APPROACH_FILES, ids=lambda path: path.stem)
def test_every_camera_model_reachable(path: Path) -> None:
    """Mỗi mô hình trong requires.camera_models có một tổ hợp được phép nhận nó.

    Để track của approach không hứa loại camera mà không cấu hình nào chạy được.
    """
    approach = files.load(path, Approach)
    reachable = {
        model
        for chosen in allowed_combinations(approach)
        for model in camera_models_for(approach, chosen)
    }
    assert reachable == set(approach.requires.camera_models)
