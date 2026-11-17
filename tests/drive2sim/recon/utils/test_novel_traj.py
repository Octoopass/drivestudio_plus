"""Novel-view trajectory generation for splits without camera 0."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from drive2sim.recon.utils.camera import get_interp_novel_trajectories  # noqa: E402


def _poses(n: int) -> "torch.Tensor":
    p = torch.eye(4).repeat(n, 1, 1)
    p[:, 0, 3] = torch.arange(n, dtype=torch.float32)
    return p


def test_front_center_interp_without_camera_zero_uses_lowest_camera() -> None:
    # KITTI-360 fisheye-only split: camera ids 2 and 3, no camera 0.
    traj = get_interp_novel_trajectories(
        "kitti360", "0003", {2: _poses(20), 3: _poses(20)}, "front_center_interp", 10
    )
    assert traj.shape == (10, 4, 4)
