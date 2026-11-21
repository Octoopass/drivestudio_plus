"""Kiểm tra resume Adam trên CPU khi thứ tự nhóm và Parameter thay đổi."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from drive2sim.external.xsim.optimization import capture_optimization, restore_optimization


def test_resume_binds_new_parameters_and_matches_next_step() -> None:
    torch = pytest.importorskip("torch")

    class Nodes(torch.nn.Module):
        def __init__(self, order: list[str]) -> None:
            super().__init__()
            self.order = order
            self.weights = torch.nn.ParameterDict(
                {k: torch.nn.Parameter(torch.ones(i + 1)) for i, k in enumerate(("x", "y"))}
            )

        def init_optimization(self) -> None:
            self.opt = torch.optim.Adam(
                [{"name": k, "params": [self.weights[k]]} for k in self.order]
            )
            self.schedulers = [SimpleNamespace(param=k, _step=0) for k in self.order]

        def advance(self) -> None:
            self.opt.zero_grad()
            sum(p.square().sum() for p in self.parameters()).backward()
            self.opt.step()

    original = Nodes(["x", "y"])
    original.init_optimization()
    original.advance()
    original.schedulers[0]._step = 7
    resumed = Nodes(["y", "x"])
    resumed.init_optimization()
    resumed.weights["x"] = torch.nn.Parameter(original.weights["x"].detach().clone())
    resumed.load_state_dict(original.state_dict())
    restore_optimization(resumed, deepcopy(capture_optimization(original)))
    assert resumed.schedulers[1]._step == 7
    original.advance()
    resumed.advance()
    for key in original.weights:
        torch.testing.assert_close(original.weights[key], resumed.weights[key])
