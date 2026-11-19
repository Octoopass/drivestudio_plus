"""Test bản xem trước video (``drive2sim/studio/api/preview.py``)."""

import subprocess
from pathlib import Path

import pytest

from drive2sim.studio.api import preview


def test_incomplete_video_is_refused(tmp_path: Path) -> None:
    """Video chưa có hộp moov (đang ghi) thì không tạo bản xem trước."""
    partial = tmp_path / "half.mp4"
    header = bytes.fromhex("0000002066747970") + b"isom" + bytes(20)
    partial.write_bytes(header + b"mdat" + b"x" * 200)
    assert not preview.is_complete(partial)
    with pytest.raises(preview.PreviewError):
        preview.build_preview(partial, tmp_path / "out.mp4")


def test_preview_path_layout(tmp_path: Path) -> None:
    """Bản xem trước nằm ở ``d2s/previews/<stage_id>/<name>``."""
    got = preview.preview_path(tmp_path / "d2s", "train.drivestudio", "videos/a.mp4")
    assert got == tmp_path / "d2s" / "previews" / "train.drivestudio" / "videos" / "a.mp4"


def test_wide_video_is_scaled_and_reused(tmp_path: Path) -> None:
    """Video rộng hơn 1920 điểm ảnh được thu nhỏ; lần gọi sau dùng lại file đã tạo."""
    ffmpeg = preview._ffmpeg()
    if ffmpeg is None:
        pytest.skip("không có ffmpeg")
    source = tmp_path / "wide.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=4224x704:rate=2",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    assert preview.is_complete(source)
    target = tmp_path / "d2s" / "previews" / "wide.mp4"
    out = preview.build_preview(source, target)
    assert out == target and preview.is_complete(out)
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(out)], capture_output=True, text=True, check=False
    )
    assert "1920x320" in probe.stderr
    mtime = out.stat().st_mtime
    assert preview.build_preview(source, target).stat().st_mtime == mtime
