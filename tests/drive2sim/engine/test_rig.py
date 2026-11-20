"""Đọc, tạo ``reports/rig.json`` ở phía không cần PyTorch (``engine/rig.py``)."""

import json
import sys
from pathlib import Path

import pytest

from drive2sim.engine.rig import RigError, generate_rig, read_rig


def test_read_rig_missing_or_broken(tmp_path: Path) -> None:
    assert read_rig(tmp_path) is None
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "rig.json").write_text("{bad")
    assert read_rig(tmp_path) is None
    (tmp_path / "reports" / "rig.json").write_text(json.dumps({"cameras": [], "trajectory": []}))
    assert read_rig(tmp_path) == {"cameras": [], "trajectory": []}


def test_generate_rig_reports_script_failure(tmp_path: Path) -> None:
    fake = tmp_path / "python"
    body = "import sys\nprint('ModuleNotFoundError: torch', file=sys.stderr)\nsys.exit(1)\n"
    fake.write_text(f"#!{sys.executable}\n{body}")
    fake.chmod(0o755)
    with pytest.raises(RigError, match="torch"):
        generate_rig(tmp_path, python=str(fake))
