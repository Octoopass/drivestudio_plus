"""Kiểm tra cách ``drive2sim/engine/report.py`` tính thời gian dựng cảnh và đọc file PLY.

- ``recon_time``: cộng thời gian từ stage đầu tới hết ``export.ply_world``; stage dùng lại cache
  tính thời gian của lần chạy gốc; thiếu số đó hoặc chưa xuất PLY thì trả ``None``.
- ``ply_stats``: đọc số Gaussian ở phần đầu file PLY và dung lượng file.
"""

from pathlib import Path

from drive2sim.core.schema import RunApproach, RunRecord, StageRecord
from drive2sim.engine.report import (
    ReportRow,
    avg_error,
    ply_stats,
    preview_stats,
    recon_time,
    rows_to_csv,
)


def make_record(stages: list[StageRecord]) -> RunRecord:
    return RunRecord(
        schema_version="d2s-run/0.3",
        run_id="20261112-081000-omnire-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id="omnire", version="0.1.0"),
        level="L1",
        executor="local",
        status="succeeded",
        stages=stages,
    )


def stage(stage_id: str, duration: float, **kwargs: object) -> StageRecord:
    return StageRecord(stage_id=stage_id, status="succeeded", duration_s=duration, **kwargs)


def test_recon_time_counts_original_time_of_cached_stages() -> None:
    record = make_record(
        [
            stage("prep.masks_segformer", 0, cache_hit=True, reused_duration_s=810.0),
            stage("prep.smpl", 1920.0),
            stage("train.drivestudio", 4380.0),
            stage("eval.standard", 3.0),
            stage("export.ply_world", 41.0),
            stage("export.preview", 30.0),  # sau khi đã có PLY: không tính
        ]
    )
    assert recon_time(record) == 810 + 1920 + 4380 + 3 + 41


def test_recon_time_unknown() -> None:
    no_ply = make_record([stage("train.drivestudio", 10.0)])
    assert recon_time(no_ply) is None
    old_cache = make_record(
        [stage("prep.smpl", 0, cache_hit=True), stage("export.ply_world", 1.0)]
    )
    assert recon_time(old_cache) is None


def test_ply_stats(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\n"
        b"property float x\nend_header\n"
    )
    ply.write_bytes(header + b"\0" * (2 * 1024 * 1024))
    count, size_mb = ply_stats(ply)
    assert count == 1234 and size_mb == 2.0
    assert ply_stats(tmp_path / "missing.ply") == (None, None)


def test_preview_stats_and_csv(tmp_path: Path) -> None:
    folder = tmp_path / "export.preview"
    (folder / "actors").mkdir(parents=True)
    for path, count in (
        (folder / "background.ply", 1000),
        (folder / "actors" / "rigid_0000.ply", 50),
    ):
        path.write_bytes(f"ply\nelement vertex {count}\nend_header\n".encode())
    assert preview_stats(folder) == (None, None)  # chưa có preview.json: chưa xuất gói
    (folder / "preview.json").write_text("{}")
    count, size_mb = preview_stats(folder)
    assert count == 1050 and size_mb is not None
    times = {"train.drivestudio": {"seconds": 1.0, "reused": False}}
    row = ReportRow(run_id="r", scene_id="s", split_id="x", approach_id="a", stage_times=times)
    header = rows_to_csv([row]).splitlines()[0]
    assert "stage_times" not in header and "preview_gaussians" in header


def test_avg_error_matches_mip_nerf() -> None:
    # Mip-NeRF trên Blender: PSNR 33.09, SSIM 0.961, LPIPS 0.043, Avg báo cáo 0.016
    assert round(avg_error(33.09, 0.961, 0.043), 3) == 0.016
    # tốt hơn ở cả ba metric thì Avg thấp hơn
    assert avg_error(28.3, 0.874, 0.183) < avg_error(24.4, 0.803, 0.271)
