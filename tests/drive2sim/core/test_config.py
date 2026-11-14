"""Test ``drive2sim.core.config``: đọc config có ``extends`` (CONTRACTS mục 10.5).

Kiểm tra từng quy tắc gộp, các ca lỗi, và việc đọc YAML giống OmegaConf (``1e-15`` là số thực).
Khi môi trường có OmegaConf (môi trường train), so thêm với ``OmegaConf.load`` trên mọi config
có sẵn trong repo.
"""

from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.core.config import ConfigError, load_config, merge_config


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_merge_rules() -> None:
    parent = {"a": {"x": 1, "y": [1, 2]}, "b": 2, "c": 3}
    child = {"a": {"y": [9], "z": 3}, "b": None, "d": {"k": 1}}
    assert merge_config(parent, child) == {"a": {"x": 1, "y": [9], "z": 3}, "c": 3, "d": {"k": 1}}
    assert parent == {"a": {"x": 1, "y": [1, 2]}, "b": 2, "c": 3}  # không sửa đối số


def test_extends_multi_level_and_root_nulls(tmp_path: Path) -> None:
    base = "trainer:\n  lr: 0.1\n  lr_final: null\n  steps: [1, 2]\nseed: 0\n"
    write(tmp_path, "base.yaml", base)
    write(tmp_path, "mid.yaml", "extends: base.yaml\ntrainer:\n  steps: [5]\n")
    leaf_text = "extends: mid.yaml\nseed: null\ntrainer:\n  fastgs:\n    enabled: true\n"
    leaf = write(tmp_path, "leaf.yaml", leaf_text)
    assert load_config(leaf) == {
        "trainer": {"lr": 0.1, "lr_final": None, "steps": [5], "fastgs": {"enabled": True}},
    }
    # file gốc giữ nguyên giá trị null của nó
    assert load_config(tmp_path / "base.yaml")["trainer"]["lr_final"] is None


def test_yaml_parsed_like_omegaconf(tmp_path: Path) -> None:
    data = load_config(write(tmp_path, "a.yaml", "eps: 1e-15\nlr: 1.6e-04\nday: 2026-11-14\n"))
    assert data == {"eps": 1e-15, "lr": 1.6e-04, "day": "2026-11-14"}
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, "dup.yaml", "a: 1\na: 2\n"))


@pytest.mark.parametrize(
    "files,match",
    [
        ({"a.yaml": "extends: missing.yaml\n"}, "missing.yaml.*a.yaml"),
        ({"a.yaml": "extends: ../x.yaml\n"}, "cùng thư mục"),
        ({"a.yaml": "extends: b.yaml\n", "b.yaml": "extends: a.yaml\n"}, "vòng"),
        ({"a.yaml": "extends: b.yaml\n", "b.yaml": "x: [1\n"}, "YAML sai.*b.yaml.*a.yaml"),
        ({"a.yaml": "- 1\n- 2\n"}, "mapping"),
    ],
)
def test_errors_name_both_files(tmp_path: Path, files: dict[str, str], match: str) -> None:
    for name, text in files.items():
        write(tmp_path, name, text)
    with pytest.raises(ConfigError, match=match):
        load_config(tmp_path / "a.yaml")


def test_same_as_omegaconf_on_repo_configs() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    for path in sorted(paths.CONFIGS_DIR.rglob("*.yaml")):
        reference = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path), resolve=False)
        if "extends" in reference:
            continue  # OmegaConf không hiểu extends; file này đã được gộp ở test khác
        assert repr(load_config(path)) == repr(reference), path.name


def _diff_paths(parent: dict, child: dict, prefix: str = "") -> set[str]:
    """Đường dẫn các khoá khác nhau giữa hai config đã gộp (thêm, bớt, đổi giá trị hay kiểu)."""
    out = set()
    for key in set(parent) | set(child):
        path = f"{prefix}{key}"
        a, b = parent.get(key), child.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out |= _diff_paths(a, b, path + ".")
        elif key not in parent or key not in child or a != b or type(a) is not type(b):
            out.add(path)
    return out


FASTGS_BLOCK = {"trainer.fastgs"}
GEER_CHANGES = {
    "trainer.render.render_mode",
    "trainer.gaussian_ctrl_general_cfg.refine_interval",
    "trainer.gaussian_ctrl_general_cfg.densify_grad_thresh",
}
# Biến thể → (file cha, các khoá được phép khác). Thêm biến thể mới thì thêm một dòng ở đây.
VARIANTS = {
    "omnire_nocampose.yaml": ("omnire.yaml", {"model.CamPose"}),
    "omnire_geer.yaml": ("omnire_nocampose.yaml", GEER_CHANGES),
    "omnire_fastgs.yaml": ("omnire_nocampose.yaml", FASTGS_BLOCK),
    "streetgs_nocampose.yaml": ("streetgs.yaml", {"model.CamPose"}),
    "streetgs_ut.yaml": ("streetgs_nocampose.yaml", {"trainer.render.render_mode"}),
    "streetgs_geer.yaml": ("streetgs_nocampose.yaml", GEER_CHANGES),
    "streetgs_fastgs.yaml": ("streetgs_nocampose.yaml", FASTGS_BLOCK),
    "streetgs_geer_fastgs.yaml": ("streetgs_geer.yaml", FASTGS_BLOCK),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_differs_only_where_declared(variant: str) -> None:
    """Biến thể chỉ khác file cha ở đúng các khoá đã khai, để so sánh hai run là công bằng."""
    parent, allowed = VARIANTS[variant]
    methods = paths.CONFIGS_DIR / "methods"
    child_cfg, parent_cfg = load_config(methods / variant), load_config(methods / parent)
    assert _diff_paths(parent_cfg, child_cfg) == allowed
    assert list(child_cfg["model"]) == [k for k in parent_cfg["model"] if k in child_cfg["model"]]
