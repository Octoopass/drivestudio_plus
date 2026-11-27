"""Kiểm tra gói output lỗi không để lại gói thiếu làm hỏng đường tải file rời."""

import tarfile
from pathlib import Path

import pytest

from drive2sim.kaggle import packing


def test_pack_failure_removes_archives_and_preserves_loose_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "result.json").write_text("{}")
    cache = tmp_path / "cache" / "scene" / "prep@1"
    cache.mkdir(parents=True)
    (cache / "mask.png").write_bytes(b"mask")
    original_add = tarfile.TarFile.add

    def fail_on_cache(
        archive: tarfile.TarFile, name: Path, *args: object, **kwargs: object
    ) -> None:
        if Path(name) == cache:
            assert (tmp_path / packing.PACK_DIR / packing.RUNS_ARCHIVE).is_file()
            raise OSError("no space left")
        original_add(archive, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", fail_on_cache)
    with pytest.raises(OSError, match="no space left"):
        packing.pack_output(tmp_path)

    assert not (tmp_path / packing.PACK_DIR).exists()
    assert (runs / "result.json").read_text() == "{}"
    assert (cache / "mask.png").read_bytes() == b"mask"
    assert packing.unpack_output(tmp_path) == 0
