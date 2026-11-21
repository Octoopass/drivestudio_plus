"""Test Cài tất cả: InstallFlow.start_all, libraries.bulk_ids, pip_constraints, lệnh CLI."""

import sys
from pathlib import Path

import pytest

from drive2sim.cli.main import main
from drive2sim.engine import env_install, libraries


class FakeProcess:
    """Tiến trình cài giả: in một dòng, thoát với mã đã đặt."""

    def __init__(self, command: list[str], code: int) -> None:
        self.args = command
        self.pid = 0
        self.stdout = iter([f"cài {' '.join(command[-1:])}\n"])
        self._code = code

    def poll(self) -> int:
        """Mã thoát (đã xong)."""
        return self._code

    def wait(self) -> int:
        """Mã thoát."""
        return self._code


def make_flow(tmp_path: Path, failing: tuple = ()) -> tuple:
    launched: list[list[str]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        launched.append(command)
        return FakeProcess(command, 1 if any(f in " ".join(command) for f in failing) else 0)

    flow = env_install.InstallFlow(tmp_path / "logs", python=sys.executable, popen=popen)
    return flow, launched


@pytest.fixture
def statuses(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Trạng thái giả của từng mục; mục đã cài xong thì thành ``ok``."""
    state = {item["id"]: "missing" for item in libraries.CATALOG}
    monkeypatch.setattr(
        libraries, "item_status", lambda item: {"status": state[item["id"]], "found": None}
    )
    monkeypatch.setattr(libraries, "pip_constraints", lambda python, path=None: Path("/tmp/c.txt"))
    return state


def test_bulk_ids_skip_risky_and_manual_items() -> None:
    ids = libraries.bulk_ids()
    assert "sdk-waymo" not in ids and "sdk-nuplan" not in ids and "node" not in ids
    assert ids.index("env-main") < ids.index("renderers") and ids.index("uv") < ids.index(
        "colab-cli"
    )


def test_start_all_installs_missing_skips_present_and_continues_after_failure(
    tmp_path: Path, statuses: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses["uv"] = "ok"
    monkeypatch.setattr(libraries, "_which", lambda name: None if name == "cargo" else "/x")
    flow, launched = make_flow(tmp_path, failing=("pykitti",))
    flow.preflight = lambda env: (_ for _ in ()).throw(ValueError("không có GPU NVIDIA"))
    flow.start_all(["env-main", "renderers", "sdk-kitti", "sdk-argoverse", "uv", "anthropic"])
    view = flow.wait()
    result = {i["id"]: (i["result"], i["reason"]) for i in view["all"]["items"]}
    assert result["env-main"] == ("skipped", "không có GPU NVIDIA")
    assert result["renderers"][0] == "skipped" and "Môi trường chính" in result["renderers"][1]
    assert result["sdk-kitti"][0] == "failed"
    assert result["sdk-argoverse"][0] == "skipped" and "cargo" in result["sdk-argoverse"][1]
    assert result["uv"] == ("present", None)
    assert result["anthropic"] == ("installed", None)
    assert view["all"]["state"] == "failed"
    # pip cài với file ràng buộc: không đổi phiên bản gói đã có
    assert launched[-1][-3:] == ["-c", "/tmp/c.txt", "anthropic"]
    assert (tmp_path / "logs" / "install-anthropic.log").is_file()


def test_pip_constraints_pin_installed_versions(tmp_path: Path) -> None:
    path = libraries.pip_constraints(sys.executable, tmp_path / "c.txt")
    lines = path.read_text().splitlines()
    assert any(line.startswith("pydantic==") for line in lines)
    assert all("==" in line and " @ " not in line for line in lines)


def test_cli_install_all_prints_summary(
    tmp_path: Path, statuses: dict, capsys: pytest.CaptureFixture
) -> None:
    for key in statuses:
        statuses[key] = "ok"  # mọi mục đã có: không chạy lệnh cài nào
    code = main(["libraries", "install", "--all", "--workdir", str(tmp_path), "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "env-main         present" in out and "anthropic        present" in out
    assert main(["libraries", "install", "nope", "--workdir", str(tmp_path)]) != 0


def test_remove_stray_tests_package_installed_by_a_dependency(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "gmplot-1.4.1.dist-info").mkdir()
    (tmp_path / "gmplot-1.4.1.dist-info" / "RECORD").write_text(
        "gmplot/__init__.py,sha256=x,1\ntests/__init__.py,sha256=y,0\n"
    )
    assert libraries.remove_stray_packages(tmp_path) == ["tests/ (của gmplot-1.4.1)"]
    assert not (tmp_path / "tests").exists()
    # thư mục không do gói nào cài: giữ nguyên
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "gmplot-1.4.1.dist-info" / "RECORD").write_text("gmplot/__init__.py,sha256=x,1\n")
    assert libraries.remove_stray_packages(tmp_path) == []
    assert (tmp_path / "tests").exists()
