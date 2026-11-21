"""Chứng minh scorer XSIM kiểm đủ cặp test và lấy trung bình theo ảnh."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from drive2sim.external.xsim.scoring import score_pairs  # noqa: E402


def test_score_requires_exact_pairs_and_weights_images_equally() -> None:
    gt = torch.zeros(32, 32, 3)
    mask = torch.ones(32, 32, dtype=torch.bool)
    samples = [(f, s, gt + v, gt, mask) for f, s, v in [(0, 2, 0.2), (1, 2, 0.3), (0, 3, 0.6)]]
    names = {2: "left", 3: "right"}
    expected = {(f, names[s]) for f, s, *_ in samples}

    def scorer(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean()

    overall, cameras = score_pairs(samples, expected, names, scorer)
    assert overall.psnr == pytest.approx((2 * cameras["left"].psnr + cameras["right"].psnr) / 3)
    with pytest.raises(ValueError, match="thiếu"):
        score_pairs(samples[:-1], expected, names, scorer)
    with pytest.raises(ValueError, match="lặp"):
        score_pairs(samples + samples[:1], expected, names, scorer)
