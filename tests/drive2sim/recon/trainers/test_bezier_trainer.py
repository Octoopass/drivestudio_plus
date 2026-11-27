"""Kiểm tra phần thuần torch của ``BezierTrainer``: loss BezierGS và gom instance cho node.

Không dựng trainer thật (cần gsplat, dataset): các loss là hàm tự do, còn ``_bezier_losses`` được
gọi trên một object giả chỉ có các thuộc tính nó đọc (``gaussian_classes``, ``losses_dict``,
``models``, ``ssim``). Phần render (``rasterize_features``, ``forward``) cần GPU nên chưa kiểm ở
đây; xem ``docs/methods/beziergs.md`` mục "Chạy thật".
"""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")

from omegaconf import OmegaConf  # noqa: E402

from drive2sim.recon.trainers import bezier as bezier_trainer  # noqa: E402
from drive2sim.recon.trainers.base import GSModelType  # noqa: E402


def fake_ssim(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """SSIM giả đơn giản (1 khi hai ảnh trùng) để test không phụ thuộc pytorch_msssim."""
    return 1.0 - (pred - target).abs().mean()


def half_left_mask(size: int = 8) -> "torch.Tensor":
    mask = torch.zeros(size, size)
    mask[:, : size // 2] = 1.0
    return mask


def test_dynamic_render_loss_zero_when_render_matches_masked_gt() -> None:
    torch.manual_seed(0)
    gt = torch.rand(8, 8, 3)
    mask = half_left_mask()
    loss = bezier_trainer.dynamic_render_loss(
        gt * mask[..., None], mask[..., None], gt, mask, fake_ssim, lambda_dssim=0.2
    )
    assert loss < 1e-6


def test_dynamic_render_loss_positive_for_wrong_alpha() -> None:
    torch.manual_seed(0)
    gt = torch.rand(8, 8, 3)
    mask = half_left_mask()
    loss = bezier_trainer.dynamic_render_loss(
        gt * mask[..., None], torch.zeros(8, 8, 1), gt, mask, fake_ssim, lambda_dssim=0.2
    )
    assert loss > 0.1


def test_static_velocity_loss_ignores_dynamic_pixels() -> None:
    mask = half_left_mask()
    on_dynamic = torch.zeros(8, 8, 3)
    on_dynamic[mask > 0] = 5.0
    on_static = torch.zeros(8, 8, 3)
    on_static[mask == 0] = 5.0
    assert bezier_trainer.static_velocity_loss(on_dynamic, mask) == 0
    assert bezier_trainer.static_velocity_loss(on_static, mask) > 0


def test_static_velocity_loss_all_dynamic_returns_zero() -> None:
    loss = bezier_trainer.static_velocity_loss(torch.ones(8, 8, 3), torch.ones(8, 8))
    assert loss == 0 and torch.isfinite(loss)


class FakeDataset:
    """Dataset giả: ``get_init_objects`` trả vật khác nhau cho từng loại node."""

    def __init__(self) -> None:
        self.calls = []

    def get_init_objects(self, cur_node_type: str, **kwargs: object) -> dict:
        """Ghi lại lệnh gọi và trả vật giả theo loại node."""
        self.calls.append((cur_node_type, kwargs))
        if cur_node_type == "RigidNodes":
            return {0: {"node_type": "RigidNodes", "num_pts": 5}}
        return {2: {"node_type": "DeformableNodes", "num_pts": 7}}


def test_collect_bezier_instances_merges_vehicle_and_human_types() -> None:
    dataset = FakeDataset()
    instances = bezier_trainer.collect_bezier_instances(dataset, instance_max_pts=100)
    assert set(instances) == {0, 2}
    assert all(v["node_type"] == "BezierNodes" for v in instances.values())
    deformable_call = [kw for name, kw in dataset.calls if name == "DeformableNodes"][0]
    assert deformable_call["exclude_smpl"] is False
    assert deformable_call["instance_max_pts"] == 100


def make_fake_trainer(with_bezier: bool) -> types.SimpleNamespace:
    classes = {"Background": GSModelType.Background}
    models = {}
    if with_bezier:
        classes["BezierNodes"] = GSModelType.BezierNodes
        models["BezierNodes"] = types.SimpleNamespace(icc_loss=lambda: torch.tensor(0.5))
    losses = OmegaConf.create(
        {
            "rgb": {"w": 0.8},
            "ssim": {"w": 0.2},
            "bezier_dynamic": {"w": 0.1},
            "bezier_velocity": {"w": 1.0},
            "bezier_icc": {"w": 0.01},
        }
    )
    return types.SimpleNamespace(
        gaussian_classes=classes, models=models, losses_dict=losses, ssim=fake_ssim
    )


def make_outputs() -> dict:
    return {
        "bezier_rgb": torch.rand(8, 8, 3),
        "bezier_opacity": torch.rand(8, 8, 1),
        "velocity_map": torch.rand(8, 8, 3),
    }


def test_losses_skipped_without_bezier_class() -> None:
    trainer = make_fake_trainer(with_bezier=False)
    assert bezier_trainer.BezierTrainer._bezier_losses(trainer, {}, {}) == {}


def test_bezier_losses_have_weighted_keys() -> None:
    trainer = make_fake_trainer(with_bezier=True)
    image_infos = {"pixels": torch.rand(8, 8, 3), "dynamic_masks": half_left_mask()}
    losses = bezier_trainer.BezierTrainer._bezier_losses(trainer, make_outputs(), image_infos)
    assert set(losses) == {"bezier_dynamic_loss", "bezier_velocity_loss", "bezier_icc_loss"}
    assert torch.isclose(losses["bezier_icc_loss"], torch.tensor(0.005))


def test_missing_dynamic_masks_raises_clear_error() -> None:
    trainer = make_fake_trainer(with_bezier=True)
    image_infos = {"pixels": torch.rand(8, 8, 3)}
    with pytest.raises(ValueError, match="load_dynamic_mask"):
        bezier_trainer.BezierTrainer._bezier_losses(trainer, make_outputs(), image_infos)
    with pytest.raises(ValueError, match="load_dynamic_mask"):
        bezier_trainer.BezierTrainer._bezier_losses(
            trainer, make_outputs(), {**image_infos, "dynamic_masks": None}
        )
