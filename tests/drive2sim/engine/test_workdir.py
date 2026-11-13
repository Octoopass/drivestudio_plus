"""Kiểm tra ``Workdir.find_run`` từ chối ``run_id`` sai định dạng.

``run_id`` đi vào mẫu glob; trước khi kiểm tra, ``..`` khớp thư mục cảnh và ``*`` khớp một run
bất kỳ, nên ``d2s run delete ..`` xoá mọi run của một cảnh.
"""

from pathlib import Path

import pytest

from drive2sim.engine.workdir import Workdir, WorkdirError

RUN_ID = "20260101-000000-toy-ab12"


@pytest.mark.parametrize("bad", ["..", ".", "*", "2026*", "../x", "20260101-000000-toy-ab12/.."])
def test_find_run_rejects_bad_ids(tmp_path: Path, bad: str) -> None:
    run_dir = tmp_path / "runs" / "scene" / "toy" / RUN_ID
    run_dir.mkdir(parents=True)
    work = Workdir(tmp_path)
    assert work.find_run(RUN_ID) == run_dir
    with pytest.raises(WorkdirError, match="không hợp lệ"):
        work.find_run(bad)
