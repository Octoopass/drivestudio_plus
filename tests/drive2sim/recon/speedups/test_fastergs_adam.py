"""Fused Adam nhận được tham số không liền mạch trong bộ nhớ (cần PyTorch và GPU).

DriveStudio tạo ``sh_dc`` bằng ``features[:, 0, :]``; ``torch._fused_adam_`` báo "must have same
dtype, device, and layout" nếu không đổi sang bản liền mạch trước.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cần GPU", allow_module_level=True)

from drive2sim.recon.speedups.fastergs import FasterGS  # noqa: E402


def test_fused_adam_takes_sliced_and_generator_params() -> None:
    features = torch.randn(100, 16, 3, device="cuda")
    sh_dc = torch.nn.Parameter(features[:, 0, :])
    assert not sh_dc.is_contiguous()
    module = torch.nn.Linear(3, 2).cuda()
    groups = [{"params": [sh_dc], "lr": 1e-2}, {"params": module.parameters(), "lr": 1e-2}]
    fastergs = FasterGS({"enabled": True})
    assert fastergs.prepare_adam_params(groups) == 1
    assert sh_dc.is_contiguous() and torch.equal(sh_dc.data, features[:, 0, :])
    assert len(groups[1]["params"]) == 2  # generator đã đổi sang list, không bị dùng hết
    optimizer = torch.optim.Adam(groups, eps=1e-15, **fastergs.adam_kwargs())
    (sh_dc.sum() + module(sh_dc).sum()).backward()
    optimizer.step()  # trước khi sửa: RuntimeError của _fused_adam_


def test_prepare_is_a_no_op_when_fused_adam_is_off() -> None:
    features = torch.randn(10, 16, 3, device="cuda")
    sh_dc = torch.nn.Parameter(features[:, 0, :])
    assert FasterGS(None).prepare_adam_params([{"params": [sh_dc]}]) == 0
    assert not sh_dc.is_contiguous()
