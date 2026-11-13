"""Kiểm tra ``drive2sim/engine/machine.py`` và việc ghi máy vào ``env.json`` khi chạy run.

- ``nvidia-smi`` được đọc đúng (tên, VRAM, driver, số GPU); không có lệnh thì để trống GPU.
- Phiên bản CUDA lấy từ đuôi phiên bản PyTorch (``+cu117`` → ``11.7``).
- ``record_machine`` thêm máy mới vào ``machines`` và không thêm trùng khi chạy tiếp cùng máy.
"""

import subprocess
from pathlib import Path

import pytest

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, EnvRecord, Machine
from drive2sim.engine import machine, run


def fake_smi(stdout: str, code: int = 0):  # noqa: ANN201
    def runner(*args: object, **kwargs: object) -> "subprocess.CompletedProcess[str]":
        return subprocess.CompletedProcess([], code, stdout, "")

    return runner


def test_current_machine_reads_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    smi = "Tesla T4, 15360, 535.104.05\nTesla T4, 15360, 535.104.05\n"
    monkeypatch.setattr(machine.subprocess, "run", fake_smi(smi))
    info = machine.current_machine()
    assert (info.gpu, info.gpu_count, info.gpu_memory_mb) == ("Tesla T4", 2, 15360)
    assert info.gpu_driver == "535.104.05"
    assert info.cpu_cores and info.cpu_cores > 0


def test_current_machine_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(machine.subprocess, "run", missing)
    info = machine.current_machine()
    assert info.gpu is None and info.gpu_count == 0


def test_package_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"torch": "2.0.0+cu117", "gsplat": "1.3.0"}

    def version(name: str) -> str:
        if name not in versions:
            raise machine.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(machine.metadata, "version", version)
    assert machine.package_versions() == {
        "pytorch": "2.0.0+cu117",
        "cuda": "11.7",
        "gsplat": "1.3.0",
    }
    versions["torch"] = "2.4.1+cu121"
    assert machine.package_versions()["cuda"] == "12.1"


def test_record_machine_appends_new_machines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "d2s").mkdir()
    env = EnvRecord(
        schema_version=SCHEMA_VERSIONS["env"], code_commit="abc", python="3.9", executor="kaggle"
    )
    files.dump(env, tmp_path / "d2s" / "env.json")
    current = {"value": Machine(gpu="Tesla T4", gpu_count=1)}
    monkeypatch.setattr(run, "current_machine", lambda: current["value"])

    run.record_machine(tmp_path)
    run.record_machine(tmp_path)  # chạy tiếp trên cùng máy: không thêm trùng
    current["value"] = Machine(gpu="Tesla P100-PCIE-16GB", gpu_count=1)
    run.record_machine(tmp_path)  # phiên Kaggle sau được cấp GPU khác

    saved = files.load(tmp_path / "d2s" / "env.json", EnvRecord)
    assert [m.gpu for m in saved.machines] == ["Tesla T4", "Tesla P100-PCIE-16GB"]
    assert saved.gpu == "Tesla P100-PCIE-16GB" and saved.executor == "kaggle"
    assert all(m.first_seen_utc is not None for m in saved.machines)
