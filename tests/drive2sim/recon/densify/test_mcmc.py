"""Densify bằng MCMC (drive2sim/recon/densify/mcmc.py) trên một node giả, chạy trên CPU."""

import pytest

torch = pytest.importorskip("torch")

from drive2sim.recon.densify.mcmc import MCMC, relocation  # noqa: E402


class FakeNode:
    """Đủ thuộc tính như node Gaussian của OmniRe: tham số, ``point_ids``, ``get_*``."""

    def __init__(self, n: int, prefix: str = "RigidNodes#", opacities: list = None) -> None:
        self.class_prefix = prefix
        self._means = torch.nn.Parameter(torch.randn(n, 3))
        self._features_dc = torch.nn.Parameter(torch.rand(n, 3))
        self._features_rest = torch.nn.Parameter(torch.zeros(n, 15, 3))
        opac = torch.tensor(opacities) if opacities is not None else torch.full((n,), 0.5)
        self._opacities = torch.nn.Parameter(torch.logit(opac).reshape(n, 1))
        self._scales = torch.nn.Parameter(torch.log(torch.full((n, 3), 0.1)))
        self._quats = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
        self.point_ids = torch.arange(n).reshape(n, 1)  # mỗi Gaussian một vật, để thấy rõ chép
        self.step = 0

    @property
    def num_points(self) -> int:
        """Số Gaussian."""
        return self._means.shape[0]

    @property
    def get_opacity(self) -> torch.Tensor:
        """Độ đục sau sigmoid."""
        return torch.sigmoid(self._opacities)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Kích thước sau exp."""
        return torch.exp(self._scales)

    @property
    def get_quats(self) -> torch.Tensor:
        """Quaternion đã chuẩn hoá."""
        return torch.nn.functional.normalize(self._quats, dim=-1)

    def groups(self) -> list[dict]:
        """Nhóm tham số đặt tên như optimizer của OmniRe."""
        names = ["xyz", "sh_dc", "sh_rest", "opacity", "scaling", "rotation"]
        attrs = ["_means", "_features_dc", "_features_rest", "_opacities", "_scales", "_quats"]
        return [
            {"params": [getattr(self, a)], "name": self.class_prefix + n, "lr": 1e-3}
            for n, a in zip(names, attrs)
        ]


def adam_with_state(node: FakeNode) -> torch.optim.Adam:
    """Adam chia nhóm theo tên như OmniRe, đã có trạng thái khác 0 sau một bước."""
    optimizer = torch.optim.Adam(node.groups())
    loss = sum(getattr(node, a).sum() for a in ("_means", "_opacities", "_scales"))
    loss = loss + node._features_dc.sum() + node._features_rest.sum() + node._quats.sum()
    loss.backward()
    optimizer.step()
    return optimizer


def state_of(optimizer: torch.optim.Optimizer, name: str) -> dict:
    group = next(g for g in optimizer.param_groups if g["name"] == name)
    return optimizer.state[group["params"][0]]


def test_relocation_one_copy_keeps_gaussian() -> None:
    """Chia thành 1 bản (không chia) thì độ đục, kích thước giữ nguyên."""
    o = torch.tensor([0.3, 0.9])
    s = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
    new_o, new_s = relocation(o, s, torch.tensor([1, 1]))
    assert torch.allclose(new_o, o) and torch.allclose(new_s, s, atol=1e-6)
    # 2 bản: tổng độ phủ 1 - (1-o')^2 = o
    new_o, _ = relocation(o, s, torch.tensor([2, 2]))
    assert torch.allclose(1 - (1 - new_o) ** 2, o, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="so với kernel CUDA của gsplat")
def test_relocation_matches_gsplat_kernel() -> None:
    gsplat = pytest.importorskip("gsplat.relocation")
    import math

    binoms = torch.zeros(51, 51)
    for n in range(51):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    o = torch.rand(1000).cuda() * 0.98 + 0.01
    s = torch.rand(1000, 3).cuda()
    ratios = torch.randint(1, 20, (1000,)).cuda()
    ref_o, ref_s = gsplat.compute_relocation(o, s, ratios.clone(), binoms.cuda())
    new_o, new_s = relocation(o, s, ratios)
    assert torch.allclose(new_o, ref_o, atol=1e-5)
    assert torch.allclose(new_s, ref_s, rtol=1e-3, atol=1e-5)


def test_relocate_moves_dead_and_keeps_ids_and_count() -> None:
    torch.manual_seed(0)
    node = FakeNode(6, opacities=[0.9, 0.8, 0.001, 0.7, 0.002, 0.6])
    optimizer = adam_with_state(node)
    mcmc = MCMC({"enabled": True})
    mcmc.relocate(node, optimizer, node.get_opacity.flatten() <= mcmc.min_opacity)
    assert node.num_points == 6
    assert (node.get_opacity.flatten() > mcmc.min_opacity).all()
    for dead in (2, 4):  # Gaussian chết thành bản sao của một Gaussian sống, kể cả point_ids
        same = (node._means.data == node._means.data[dead]).all(-1)
        same[dead] = False
        src = int(same.nonzero()[0])
        assert node.point_ids[dead] == node.point_ids[src]
        assert state_of(optimizer, "RigidNodes#xyz")["exp_avg"][dead].abs().sum() == 0


def test_add_grows_to_cap_and_extends_optimizer() -> None:
    torch.manual_seed(0)
    node = FakeNode(20)
    optimizer = adam_with_state(node)
    mcmc = MCMC({"enabled": True, "cap_max": {"RigidNodes": 22}})
    mcmc.step("RigidNodes", node, step=600, optimizer=optimizer)  # 5 % của 20 = 21 < 22
    assert node.num_points == 21
    mcmc.step("RigidNodes", node, step=700, optimizer=optimizer)  # 22,05 → kẹp vào 22
    assert node.num_points == 22
    mcmc.step("RigidNodes", node, step=800, optimizer=optimizer)  # đủ ngân sách: không thêm
    assert node.num_points == 22
    assert node.point_ids.shape == (22, 1)
    for group in optimizer.param_groups:  # optimizer giữ đúng tham số mới, trạng thái đủ dài
        param = group["params"][0]
        assert param.shape[0] == 22 and optimizer.state[param]["exp_avg"].shape[0] == 22
    # tham số thật sự là tham số của node: một bước Adam nữa chạy được
    node._means.sum().backward()
    optimizer.step()


def test_step_outside_refine_window_only_adds_noise() -> None:
    # Nhiễu của gsplat chỉ đáng kể khi độ đục dưới khoảng 0,005: hệ số sigmoid(100·((1−o)−0,995)).
    node = FakeNode(10, opacities=[0.999] * 5 + [0.001] * 5)
    optimizer = adam_with_state(node)
    before = node._means.data.clone()
    mcmc = MCMC({"enabled": True})
    mcmc.step("RigidNodes", node, step=100, optimizer=optimizer)  # trước refine_start_iter
    assert node.num_points == 10
    moved = (node._means.data - before).norm(dim=-1)
    assert moved[5:].min() > 0  # Gaussian gần trong suốt bị xê dịch
    assert moved[:5].max() < moved[5:].min() * 1e-3  # Gaussian đậm gần như đứng yên


def test_disabled_and_classes() -> None:
    assert not MCMC(None).applies("Background")
    mcmc = MCMC({"enabled": True})
    assert mcmc.applies("Background") and mcmc.applies("DeformableNodes")
    assert not mcmc.applies("SMPLNodes")  # người: không densify
    losses = mcmc.regularization(FakeNode(4))
    assert set(losses) == {"mcmc_opacity_reg", "mcmc_scale_reg"}
