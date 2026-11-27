"""Fixture gói quan sát cho test của adapter XSIM."""

from pathlib import Path

import pytest


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Thư mục gói quan sát hợp lệ; bỏ qua test cần nó khi máy thiếu numpy, Pillow."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from tests.drive2sim.external.xsim.bundle_fixture import write_bundle

    return write_bundle(tmp_path / "observations")
