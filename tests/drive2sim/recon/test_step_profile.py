"""Test phần chia nhóm kernel của ``drive2sim/recon/utils/step_profile.py`` (không cần GPU).

Phần chạy torch.profiler đã thử trên GPU bằng script chạy tay (plan Faster-GS, bước D1).
"""

from pathlib import Path

import pytest

from drive2sim.recon.utils.step_profile import StepProfiler, categorize


@pytest.mark.parametrize(
    "name,category",
    [
        ("void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(...)", "rasterizer"),
        ("void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(...)", "rasterizer"),
        ("void cub::CUB_200700_750_NS::DeviceRadixSortOnesweepKernel<...>", "rasterizer"),
        ("void at::native::(anonymous namespace)::multi_tensor_apply_kernel<...>", "optimizer"),
        ("void fusedssimCUDA<...>(...)", "ssim"),
        ("void at::native::vectorized_elementwise_kernel<4, ...>", "other"),
    ],
)
def test_categorize(name: str, category: str) -> None:
    assert categorize(name) == category


def test_profiler_is_off_without_steps(tmp_path: Path) -> None:
    profiler = StepProfiler([], 30, str(tmp_path))
    for step in range(5):
        profiler.begin_step(step)
        profiler.end_step(step)
    assert list(tmp_path.iterdir()) == []
