"""Chứng minh scorer XSIM chấm trên ảnh thật và mask đánh giá của gói, đủ cặp test."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from drive2sim.external.xsim.bundle import Observations  # noqa: E402
from drive2sim.external.xsim.scoring import score_renders  # noqa: E402


def lpips(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean()


def test_scores_masked_pixels_and_requires_every_test_render(bundle: Path, tmp_path: Path) -> None:
    observations = Observations(bundle)
    renders = tmp_path / "renders"
    renders.mkdir()
    gt = np.asarray(Image.open(observations.image_path(4, 1)))
    wrong = gt.copy()
    wrong[:, 8:] = 255  # chỉ sai ngoài mask đánh giá
    Image.fromarray(wrong).save(renders / "004_1.png")
    overall, cameras, count = score_renders(observations, renders, {(4, "image_02")}, lpips)
    assert count == 1 and overall.psnr == float("inf") and cameras["image_02"].lpips == 0
    with pytest.raises(ValueError, match="thiếu ảnh render"):
        score_renders(observations, renders, {(3, "image_02"), (4, "image_02")}, lpips)
