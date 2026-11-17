"""Kiểm tra hướng nhìn SMPL khi camera trùng điểm tạm của người vắng mặt (CPU).

SH CUDA tự chuẩn hoá đầu vào nên hướng bằng 0 vẫn lỗi dù chỉ chặn mẫu số ở Python. Test gọi
get_gaussians thật, thay SH bằng phép chuẩn hoá thứ hai trên CPU và không cần mô hình SMPL.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")
pytest.importorskip("smplx")

import drive2sim.recon.models.nodes.smpl as smpl_module  # noqa: E402
from drive2sim.recon.models.nodes.smpl import SMPLNodes  # noqa: E402


def test_absent_person_at_camera_origin_has_finite_sh_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SMPLNodes.__new__(SMPLNodes)
    torch.nn.Module.__init__(node)
    node._means = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    node._quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    node._scales = torch.nn.Parameter(torch.zeros(2, 1))
    node._opacities = torch.nn.Parameter(torch.zeros(2, 1))
    node._features_dc = torch.nn.Parameter(torch.zeros(2, 3))
    node._features_rest = torch.nn.Parameter(torch.zeros(2, 3, 3))
    node.instances_fv = torch.tensor([[True, False]])
    node.point_ids = torch.tensor([[0], [1]])
    node.cur_frame = 0
    node.step = 30000
    node.ball_gaussians = True
    node.ctrl_cfg = SimpleNamespace(sh_degree=1, sh_degree_interval=1000)
    monkeypatch.setattr(node, "transform_means", lambda means: means)
    captured = []

    def renormalizing_sh(
        degree: int, directions: "torch.Tensor", colors: "torch.Tensor"
    ) -> "torch.Tensor":
        captured.append(directions)
        return directions / directions.norm(dim=-1, keepdim=True)

    monkeypatch.setattr(smpl_module, "spherical_harmonics", renormalizing_sh)
    result = node.get_gaussians(SimpleNamespace(camtoworlds=torch.eye(4)))
    assert all(torch.isfinite(value).all() for value in result.values())
    expected = node._means[0] / node._means[0].norm()
    assert torch.equal(captured[0][0], expected)
    assert torch.equal(captured[0][1], torch.tensor([0.0, 0.0, 1.0]))
    assert result["_opacities"][1] == 0  # Người vắng mặt vẫn không góp màu vào ảnh.
