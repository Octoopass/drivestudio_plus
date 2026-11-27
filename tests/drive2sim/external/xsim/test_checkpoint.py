"""Kiểm tra checkpoint của worker XSIM bằng node và sampler nhỏ trên CPU."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_checkpoint_restores_rng_sampler_and_resized_buffer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from drive2sim.external.xsim.checkpoint import restore_checkpoint, save_checkpoint

    class Nodes(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("step", torch.tensor(12))
            self.register_buffer("grad_sum", torch.arange(3), persistent=False)

        def init_optimization(self) -> None:
            pass

    def draw() -> tuple:
        return random.random(), np.random.rand(), torch.rand(()).item()

    nodes = Nodes()
    sampler = SimpleNamespace(error_buffer=torch.tensor([0.3]))
    subset = SimpleNamespace(image_indices=[(2, 0)], img_sampler=sampler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "configuration", nodes, subset)
    expected = draw()
    nodes.grad_sum = torch.zeros(1)
    sampler.error_buffer.zero_()
    nodes.step.zero_()
    assert restore_checkpoint(path, "configuration", nodes, subset) == 12
    assert nodes.grad_sum.tolist() == [0, 1, 2]
    assert sampler.error_buffer.item() == pytest.approx(0.3)
    assert draw() == expected
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint(path, "other", nodes, subset)
