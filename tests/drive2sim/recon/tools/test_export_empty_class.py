"""Kiểm tra ``gaussians_from_state`` khi một lớp Gaussian không còn điểm nào (CPU).

FastGS có thể prune hết Gaussian của ``RigidNodes`` trên cảnh dài; checkpoint khi đó có tensor
0 phần tử và ``reshape(len(means), -1)`` báo "ambiguous" (run ``20261127-225543-omnire-938c``,
stage ``export.preview``).
"""

import pytest

torch = pytest.importorskip("torch")

from drive2sim.recon.tools.export_ply import gaussians_from_state  # noqa: E402


def test_gaussians_from_state_accepts_zero_gaussians() -> None:
    state = {
        "_means": torch.zeros(0, 3),
        "_scales": torch.zeros(0, 3),
        "_quats": torch.zeros(0, 4),
        "_features_dc": torch.zeros(0, 3),
        "_features_rest": torch.zeros(0, 15, 3),
        "_opacities": torch.zeros(0, 1),
    }

    gaussians = gaussians_from_state(state)

    assert gaussians["means"].shape == (0, 3)
    assert gaussians["f_dc"].shape == (0, 3)
    assert gaussians["f_rest"].shape == (0, 45)
    assert gaussians["opac"].shape == (0, 1)
