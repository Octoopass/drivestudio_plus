"""Kiểm tra ``drive2sim/engine/environment.py``: ba mức ok, warn, missing và lệnh cài."""

from pathlib import Path
from typing import Optional

import pytest

from drive2sim import paths
from drive2sim.engine import environment


def patch_machine(
    monkeypatch: pytest.MonkeyPatch,
    torch: Optional[str],
    gsplat: Optional[str],
    cuda_ok: bool = True,
) -> None:
    """Giả GPU, phiên bản gói và kết quả hỏi CUDA."""
    cuda = torch.split("+cu")[1][:-1] + "." + torch[-1] if torch and "+cu" in torch else None
    monkeypatch.setattr(environment, "_gpus", lambda: [("Quadro T2000", 4096, "580")])
    monkeypatch.setattr(
        environment,
        "package_versions",
        lambda: {"pytorch": torch, "cuda": cuda, "gsplat": gsplat},
    )
    monkeypatch.setattr(environment, "_cached_torch_cuda", lambda: (cuda_ok, "Quadro T2000"))


def test_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_machine(monkeypatch, "2.4.1+cu121", "1.5.3")
    result = environment.environment_check("drivestudio")
    assert result["level"] == "ok" and result["install"] == "bash envs/install_main.sh"
    patch_machine(monkeypatch, "2.0.0+cu117", "1.3.0")
    result = environment.environment_check("drivestudio")
    assert result["level"] == "warn"
    assert {c["id"]: c["level"] for c in result["checks"]}["gsplat"] == "warn"
    patch_machine(monkeypatch, None, None)
    assert environment.environment_check("drivestudio")["level"] == "missing"


def test_codebase_without_environment() -> None:
    assert environment.environment_check("toy")["env"] is None


def test_expected_versions_match_env_files() -> None:
    expected = environment.EXPECTED["main"]
    envs = paths.PACKAGE_DIR.parent / "envs"
    script = (envs / "install_main.sh").read_text()
    assert f"torch=={expected['pytorch']}+cu{expected['cuda'].replace('.', '')}" in script
    assert f"GSPLAT_VERSION={expected['gsplat']}" in script
    assert "install_main.sh" in (envs / "main.yml").read_text()


def test_smpl_model_lookup_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OmniRe (có prep.smpl) cần SMPL_NEUTRAL.pkl; tìm theo biến môi trường rồi workdir."""
    patch_machine(monkeypatch, "2.4.1+cu121", "1.5.3")
    monkeypatch.delenv("D2S_SMPL_MODEL", raising=False)
    monkeypatch.setattr(paths, "KAGGLE_INPUT", tmp_path / "no-kaggle")
    monkeypatch.setattr(paths, "PACKAGE_DIR", tmp_path / "pkg" / "drive2sim")
    result = environment.environment_check("drivestudio", needs_smpl=True, workdir_root=tmp_path)
    assert {c["id"]: c["level"] for c in result["checks"]}["smpl_model"] == "missing"
    model = tmp_path / "smpl_models" / "SMPL_NEUTRAL.pkl"
    model.parent.mkdir()
    model.write_bytes(b"x")
    assert paths.find_smpl_model(tmp_path) == model.resolve()
    other = tmp_path / "elsewhere.pkl"
    other.write_bytes(b"x")
    monkeypatch.setenv("D2S_SMPL_MODEL", str(other))
    assert paths.find_smpl_model(tmp_path) == other.resolve()
    result = environment.environment_check("drivestudio", needs_smpl=True, workdir_root=tmp_path)
    assert result["level"] == "ok"
