"""Test ``drive2sim.engine.densify_info``: đọc lịch densify hiệu lực từ ``config.yaml`` của run."""

from pathlib import Path
from typing import Optional

import yaml

from drive2sim.engine.densify_info import densify_info, load_densify_info

GENERAL = {
    "warmup_steps": 500,
    "reset_alpha_interval": 3000,
    "refine_interval": 100,
    "densify_grad_thresh": 0.0005,
    "densify_size_thresh": 0.003,
    "cull_alpha_thresh": 0.005,
    "stop_split_at": 15000,
    "sh_degree": 3,
}


def config(
    render: Optional[dict] = None, trainer: Optional[dict] = None, models: Optional[dict] = None
) -> dict:
    return {
        "trainer": {
            "type": "drive2sim.recon.trainers.MultiTrainer",
            "render": {"absgrad": True, **(render or {})},
            "gaussian_ctrl_general_cfg": dict(GENERAL),
            **(trainer or {}),
        },
        "model": models
        or {
            "Background": {"type": "VanillaGaussians"},
            "RigidNodes": {
                "type": "RigidNodes",
                "ctrl": {"stop_split_at": 20000, "cull_scale_thresh": 0.1},
            },
            "SMPLNodes": {"type": "SMPLNodes", "ctrl": {"sh_degree": 1}},
            "Sky": {"type": "EnvLight"},
        },
    }


def test_default_rule_merges_node_ctrl_and_skips_non_densified_nodes() -> None:
    info = densify_info(config())
    assert info["rule"] == "default"
    assert info["renderer"] == "default"
    assert info["statistic"] == "2d_abs"
    assert list(info["nodes"]) == ["Background", "RigidNodes"]
    assert info["nodes"]["Background"]["stop_split_at"] == 15000
    assert info["nodes"]["RigidNodes"]["stop_split_at"] == 20000
    assert info["nodes"]["RigidNodes"]["refine_interval"] == 100
    assert "sh_degree" not in info["nodes"]["Background"]
    assert "fastgs" not in info and "mcmc" not in info


def test_statistic_follows_renderer_and_trainer_type() -> None:
    assert densify_info(config(render={"absgrad": False}))["statistic"] == "2d"
    assert densify_info(config(render={"render_mode": "ut"}))["statistic"] == "3d"
    screen = config(
        render={"render_mode": "ut"},
        trainer={"type": "drive2sim.recon.trainers.screen_grad.ScreenGradMultiTrainer"},
    )
    assert densify_info(screen)["statistic"] == "screen"
    assert densify_info(config(render={"render_mode": "geer"}))["statistic"] == "geer"


def test_fastgs_settings_are_reported() -> None:
    info = densify_info(config(trainer={"fastgs": {"enabled": True, "prune_from": 15000}}))
    assert info["rule"] == "fastgs"
    assert info["fastgs"] == {"prune_from": 15000}


def test_mcmc_uses_module_defaults_and_overrides() -> None:
    info = densify_info(
        config(trainer={"mcmc": {"enabled": True, "cap_max": {"Background": 500000}}})
    )
    assert info["rule"] == "mcmc"
    mcmc = info["mcmc"]
    assert mcmc["refine_start_iter"] == 500
    assert mcmc["refine_stop_iter"] == 25000
    assert mcmc["refine_every"] == 100
    assert mcmc["caps"] == {"Background": 500000, "RigidNodes": 50000}
    # Node không có trong config (DeformableNodes ở đây) không có trần.
    assert "DeformableNodes" not in mcmc["caps"]


def test_mcmc_scalar_cap_and_classes() -> None:
    info = densify_info(
        config(trainer={"mcmc": {"enabled": True, "classes": ["Background"], "cap_max": 2000000}})
    )
    assert info["mcmc"]["caps"] == {"Background": 2000000}
    # RigidNodes không dùng MCMC thì giữ quy tắc gốc.
    assert info["mcmc"]["classes"] == ["Background"]


def test_load_reads_train_stage_config(tmp_path: Path) -> None:
    stage = tmp_path / "d2s" / "stages" / "train.drivestudio"
    stage.mkdir(parents=True)
    assert load_densify_info(tmp_path) is None
    # method_config.yaml được ghi trước khi áp override dòng lệnh: không dùng.
    (stage / "method_config.yaml").write_text(yaml.safe_dump(config()), encoding="utf-8")
    assert load_densify_info(tmp_path) is None
    effective = config()
    effective["trainer"]["gaussian_ctrl_general_cfg"]["refine_interval"] = 50
    (stage / "config.yaml").write_text(yaml.safe_dump(effective), encoding="utf-8")
    info = load_densify_info(tmp_path)
    assert info is not None and info["nodes"]["Background"]["refine_interval"] == 50


def test_load_ignores_broken_yaml(tmp_path: Path) -> None:
    stage = tmp_path / "d2s" / "stages" / "train.drivestudio"
    stage.mkdir(parents=True)
    (stage / "config.yaml").write_text("trainer: [", encoding="utf-8")
    assert load_densify_info(tmp_path) is None
