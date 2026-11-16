"""Gói skin người cho Spark: 4 xương mạnh nhất, giữ tổng trọng số, mã hoá như ``setSplatBones``."""

import pytest

torch = pytest.importorskip("torch")

from drive2sim.recon.tools.export_preview import _skin  # noqa: E402


def test_skin_keeps_top4_and_total_weight() -> None:
    weights = torch.zeros(2, 24)
    weights[0, [3, 7, 9, 12, 20]] = torch.tensor([0.5, 0.3, 0.1, 0.06, 0.04])
    weights[1, 5] = 1.0
    packed = _skin(weights).astype(int).reshape(2, 4)
    bones, values = packed >> 8, (packed & 255) / 255
    assert set(bones[0]) == {3, 7, 9, 12}  # xương 20 (yếu nhất) bị bỏ
    assert abs(values[0].sum() - 1.0) < 4 / 255  # phần của xương 20 chia lại cho 4 xương giữ
    assert bones[1, 0] == 5 and values[1, 0] == 1.0
