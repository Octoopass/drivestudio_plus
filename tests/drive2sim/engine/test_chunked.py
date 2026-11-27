"""Kiểm tra ``drive2sim/engine/chunked.py``: gộp các chunk của một lần dựng cảnh dài.

- ``to_scene_frames``: chỉ số khung tính từ khung đầu của split (DriveStudio) được cộng
  ``frame_range[0]``; chỉ số đã theo cảnh thì giữ.
- ``owner_ranges``: đoạn chồng chia đôi, mỗi khung thuộc đúng một chunk.
- ``run_chunked``: metric gộp đếm mỗi ảnh một lần, quãng đường theo ``ego_pose`` của cảnh, PSNR
  quanh chỗ nối và so hai chunk trên khung chung, R2 trên camera chỉ test.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drive2sim.core.schema import BenchmarkChunk, BenchmarkChunkedEntry, BenchmarkFile
from drive2sim.engine import chunked
from drive2sim.engine.report import ReportRow


def split_of(split_id: str, frame_range: tuple[int, int], every: int = 4) -> SimpleNamespace:
    """Split giả: khung test mỗi ``every`` khung tính từ khung đầu, camera 1 chỉ test."""
    first, last = frame_range
    test = [(frame, camera) for frame in range(first, last + 1, every) for camera in ("c0", "c1")]
    return SimpleNamespace(
        split_id=split_id, frame_range=frame_range, test=test, test_only_cameras=["c1"]
    )


def test_frames_relative_to_the_split_are_shifted_to_scene_frames() -> None:
    split = split_of("b", (16, 39))
    records = [{"frame_index": 0}, {"frame_index": 4}]
    shifted, warning = chunked.to_scene_frames(records, split)  # type: ignore[arg-type]
    assert [r["frame_index"] for r in shifted] == [16, 20] and warning is None
    absolute, warning = chunked.to_scene_frames(
        [{"frame_index": 20}, {"frame_index": 24}],
        split,  # type: ignore[arg-type]
    )
    assert [r["frame_index"] for r in absolute] == [20, 24] and warning is None
    _, warning = chunked.to_scene_frames([{"frame_index": 3}], split)  # type: ignore[arg-type]
    assert warning is not None


def test_owner_ranges_split_the_overlap_in_half() -> None:
    assert chunked.owner_ranges([(0, 23), (16, 39), (32, 50)]) == [(0, 19), (20, 35), (36, 50)]
    # hai chunk không chồng thì giữ nguyên đoạn
    assert chunked.owner_ranges([(0, 9), (12, 20)]) == [(0, 9), (12, 20)]


def write_scene(root: Path, num_frames: int) -> Path:
    """Cảnh giả: xe đi thẳng 1 m mỗi khung."""
    pose_dir = root / "scene" / "ego_pose"
    pose_dir.mkdir(parents=True)
    for frame in range(num_frames):
        rows = [[1, 0, 0, float(frame)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        (pose_dir / f"{frame:03d}.txt").write_text(
            "\n".join(" ".join(str(v) for v in row) for row in rows)
        )
    return root / "scene"


def write_run(root: Path, run_id: str, records: list[dict]) -> Path:
    stage = root / run_id / "d2s" / "stages" / "eval.standard"
    stage.mkdir(parents=True)
    (stage / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return root / run_id


def records_for(frames: list[int], psnr_of: dict[int, float], default: float) -> list[dict]:
    """Bản ghi cho camera 0 (đã train) và camera 1 (chỉ test) ở các khung (chỉ số của split)."""
    out = []
    for frame in frames:
        out.append(
            {
                "frame_index": frame,
                "camera_index": 0,
                "position": [0.0, 0.0, 0.0],  # hệ riêng của chunk: không được dùng
                "psnr": psnr_of.get(frame, default),
                "ssim": 0.8,
                "lpips": 0.2,
            }
        )
        out.append(
            {
                "frame_index": frame,
                "camera_index": 1,
                "position": [0.0, 0.0, 0.0],
                "psnr": 20.0,
                "psnr_cc": 22.0,
                "ssim": 0.7,
                "lpips": 0.3,
            }
        )
    return out


def make_row(run_id: str, split_id: str, train_time_s: float) -> ReportRow:
    return ReportRow(
        run_id=run_id,
        scene_id="s",
        split_id=split_id,
        approach_id="omnire",
        psnr=24.0,
        train_time_s=train_time_s,
        ply_size_mb=50.0,
        peak_vram_mb=3000,
        gpu="RTX 4090",
    )


def test_run_chunked_merges_two_overlapping_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_dir = write_scene(tmp_path, 40)
    run_a, run_b = "20261128-000000-omnire-aaaa", "20261128-000001-omnire-bbbb"
    # chunk A khung 0–23, frames.jsonl theo chỉ số của split (bắt đầu 0)
    write_run(tmp_path, run_a, records_for([0, 4, 8, 12, 16, 20], {}, 25.0))
    # chunk B khung 16–39: chỉ số 4 của split là khung 20 của cảnh, kém ở đó
    write_run(tmp_path, run_b, records_for([0, 4, 8, 12, 16, 20], {4: 20.0}, 24.0))
    splits = {"c1": split_of("c1", (0, 23)), "c2": split_of("c2", (16, 39))}
    rows = {run_a: make_row(run_a, "c1", 100.0), run_b: make_row(run_b, "c2", 200.0)}
    monkeypatch.setattr(chunked, "report_rows", lambda w, s, split, ids: [rows[ids[0]]])
    monkeypatch.setattr(
        chunked,
        "summary_path",
        lambda run_dir: run_dir / "d2s" / "stages" / "eval.standard" / "summary.json",
    )
    monkeypatch.setattr(chunked, "SEAM_WINDOW_FRAMES", 4)
    cameras = [
        SimpleNamespace(camera_id="c0", model="opencv_pinhole"),
        SimpleNamespace(camera_id="c1", model="opencv_pinhole"),
    ]
    workdir = SimpleNamespace(
        load_split=lambda scene_id, split_id: splits[split_id],
        load_scene=lambda scene_id: SimpleNamespace(manifest=SimpleNamespace(cameras=cameras)),
        run_dir=lambda scene_id, approach_id, run_id: tmp_path / run_id,
        scene_dir=lambda scene_id: scene_dir,
    )
    entry = BenchmarkChunkedEntry(
        label="#11",
        scene_id="s",
        chunks=[  # thứ tự trong file không quan trọng
            BenchmarkChunk(split_id="c2", run_id=run_b),
            BenchmarkChunk(split_id="c1", run_id=run_a),
        ],
    )
    group = chunked.run_chunked(workdir, entry)  # type: ignore[arg-type]
    assert [c.run_id for c in group.chunks] == [run_a, run_b]
    # A sở hữu khung 0–19 (0, 4, 8, 12, 16), B sở hữu 20–39 (20, 24, 28, 32, 36)
    assert group.frames_measured == 10
    assert group.psnr == pytest.approx((25.0 * 5 + 20.0 + 24.0 * 4) / 10)
    assert group.ssim == pytest.approx(0.8) and group.lpips == pytest.approx(0.2)
    assert group.shift_psnr == pytest.approx(20.0)
    assert group.shift_psnr_cc == pytest.approx(22.0)
    assert group.distance_m == pytest.approx(36.0)  # theo ego_pose, không theo vị trí của chunk
    assert group.train_time_s == 300.0 and group.ply_size_mb == 100.0
    assert group.train_s_per_100m == pytest.approx(300.0 / 0.36)
    assert group.gpu == "RTX 4090" and group.peak_vram_mb == 3000
    (seam,) = group.seams
    assert seam.overlap == (16, 23)
    assert seam.num_frames == 2 and seam.psnr_near == pytest.approx(22.5)
    assert seam.drop_db == pytest.approx(24.5 - 22.5)
    assert seam.overlap_frames == 2
    assert seam.overlap_psnr_before == pytest.approx(25.0)
    assert seam.overlap_psnr_after == pytest.approx(22.0)
    assert group.warnings == []


def test_run_chunked_rejects_a_run_outside_the_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunked, "report_rows", lambda *args: [])
    entry = BenchmarkChunkedEntry(
        label="x",
        scene_id="s",
        chunks=[
            BenchmarkChunk(split_id="a", run_id="r1"),
            BenchmarkChunk(split_id="b", run_id="r2"),
        ],
    )
    with pytest.raises(ValueError, match="không có hoặc không thuộc"):
        chunked.run_chunked(SimpleNamespace(), entry)  # type: ignore[arg-type]


def test_benchmark_file_needs_entries_or_chunked() -> None:
    with pytest.raises(ValueError):
        BenchmarkFile(schema_version="d2s-benchmark/0.3")
    parsed = BenchmarkFile(
        schema_version="d2s-benchmark/0.3",
        chunked=[
            BenchmarkChunkedEntry(
                label="x",
                scene_id="s",
                chunks=[
                    BenchmarkChunk(split_id="a", run_id="r1"),
                    BenchmarkChunk(split_id="b", run_id="r2"),
                ],
            )
        ],
    )
    assert parsed.entries == [] and parsed.chunked[0].bench_track == "B"
