"""Kiểm tra ``drive2sim/scripts/make_demo.py``: thư mục mẫu đọc được bằng chính engine của tool.

Mọi file đã được kiểm tra theo schema khi ghi; test này kiểm tra thêm là báo cáo, cache và
thư mục làm việc nhận đúng dữ liệu mẫu, và Studio biết đó là dữ liệu mẫu.
"""

from pathlib import Path

import pytest

from drive2sim.engine.housekeeping import cache_entries
from drive2sim.engine.report import report_rows
from drive2sim.engine.workdir import Workdir
from drive2sim.scripts import make_demo


def test_make_demo_builds_a_readable_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_demo, "write_video", lambda path, seconds=3: False)  # nhanh hơn
    assert make_demo.main([str(tmp_path / "demo")]) == 0
    work = Workdir(tmp_path / "demo")
    assert work.is_demo
    rows = {row.run_id: row for row in report_rows(work)}
    assert len(rows) == len(make_demo.RUNS)
    first = rows["20261112-081000-omnire-a1b2"]
    assert first.psnr == 27.64 and first.recon_time_s and first.ply_gaussians
    assert rows["20261112-093000-omnire-b7c1"].gpu == "Tesla T4 + Tesla P100-PCIE-16GB"
    assert first.software == "PyTorch 2.0.0+cu117, gsplat 1.3.0"
    assert rows["20261112-101500-streetgs-c3d4"].software == "PyTorch 2.4.1+cu121, gsplat 1.5.3"
    assert rows["20261112-120500-omnire-d9e0"].quick is True
    assert first.speedups == ["fastgs"]
    assert rows["20261113-090000-omnire-fa57"].speedups == [
        "fastgs",
        "fastergs.fused_adam",
        "fastergs.fused_ssim",
        "fastergs.fewer_syncs",
    ]
    stale = [e for e in cache_entries(work) if not e.used_by]
    assert [e.cache_key for e in stale] == [
        "nuscenes_scene-0061/prep.masks_segformer@0.1.0-0badf00d"
    ]
    # chạy lại không có --force thì không ghi đè
    assert make_demo.main([str(tmp_path / "demo")]) == 1
    # --force không xoá thư mục làm việc thật (không có file DEMO)
    real = tmp_path / "real"
    (real / "runs").mkdir(parents=True)
    (real / "runs" / "keep.txt").write_text("x")
    assert make_demo.main([str(real), "--force"]) == 1
    assert (real / "runs" / "keep.txt").is_file()
