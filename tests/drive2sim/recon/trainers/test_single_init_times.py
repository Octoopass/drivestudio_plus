"""Kiểm tra thời gian khởi tạo của điểm ngẫu nhiên trong ``SingleTrainer`` (PVG không dùng LiDAR).

Gồm: khi có LiDAR thì điểm ngẫu nhiên giữ thời gian 0 như bản gốc; khi không có LiDAR
(``train.drivestudio lidar=false``) thì thời gian của điểm ngẫu nhiên rải đều trong [0, 1], để
ở mọi khung vẫn còn Gaussian sống (PVG lọc ``marginal_t > 0.05``). Trước khi sửa, mọi điểm có
thời gian 0, khung xa 0 không còn Gaussian nào và gsplat chia cho 0 (SIGFPE, run #4 của đợt 2).
"""

from types import SimpleNamespace
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from omegaconf import OmegaConf  # noqa: E402

from drive2sim.recon.models.gaussians import PeriodicVibrationGaussians  # noqa: E402
from drive2sim.recon.trainers.single import SingleTrainer  # noqa: E402


class _FakePvgModel(PeriodicVibrationGaussians):
    """PVG giả: bỏ qua ``__init__`` nặng, chỉ ghi lại thời gian khởi tạo nhận được."""

    def __init__(self) -> None:  # noqa: D107
        self.init_times: Optional[torch.Tensor] = None

    def create_from_pcd(
        self, init_means: "torch.Tensor", init_colors: "torch.Tensor", init_times: "torch.Tensor"
    ) -> None:  # noqa: D102
        self.init_times = init_times


class _FakeDataset:
    """Dataset giả: mọi điểm ngẫu nhiên đều nhìn thấy, LiDAR cho 10 điểm có thời gian 0,5."""

    def check_pts_visibility(self, points: "torch.Tensor") -> "torch.Tensor":  # noqa: D102
        return torch.ones(points.shape[0], dtype=torch.bool)

    def get_lidar_samples(
        self, device: object, **kwargs: object
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":  # noqa: D102
        return torch.zeros(10, 3), torch.zeros(10, 3), torch.full((10, 1), 0.5)


def _run_init(init_config: dict[str, object]) -> _FakePvgModel:
    model = _FakePvgModel()
    trainer = SimpleNamespace(
        gaussian_classes={"Background": None},
        model_config={"Background": {"init": OmegaConf.create(init_config)}},
        models={"Background": model},
        device="cpu",
        scene_radius=10.0,
        scene_origin=torch.zeros(3),
    )
    SingleTrainer.init_gaussians_from_dataset(trainer, _FakeDataset())
    return model


def test_random_points_get_spread_times_without_lidar() -> None:
    model = _run_init({"near_randoms": 200, "far_randoms": 200})
    times = model.init_times
    assert times.shape[0] > 0
    # trải đều: có điểm sống gần cả đầu và cuối đoạn thời gian, không dồn về 0
    assert times.min() < 0.1 and times.max() > 0.9
    assert times.mean() == pytest.approx(0.5, abs=0.1)


def test_random_points_keep_time_zero_next_to_lidar() -> None:
    model = _run_init({"from_lidar": {"num_samples": 10}, "near_randoms": 200, "far_randoms": 200})
    times = model.init_times
    assert times.shape[0] > 10
    assert torch.all(times[:10] == 0.5)  # điểm LiDAR giữ thời gian của chúng
    assert torch.all(times[10:] == 0.0)  # điểm ngẫu nhiên giữ thời gian 0 như bản gốc
