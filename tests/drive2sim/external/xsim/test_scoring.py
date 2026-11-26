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


def test_shifted_cameras_are_scored_separately_from_overall(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys
    from types import SimpleNamespace

    from drive2sim.external.xsim.scoring import main

    index_path = bundle / "observations.json"
    index = json.loads(index_path.read_text())
    camera = {**index["cameras"][0], "index": 0, "camera_id": "image_00"}
    index["cameras"].append(camera)
    index["camera_indices"]["image_00"] = 0
    index["split"].update(
        cameras=["image_00", "image_02"],
        train=[[3, "image_00"]],
        test=[[4, "image_00"], [4, "image_02"]],
        test_only_cameras=["image_02"],
    )
    index_path.write_text(json.dumps(index))
    poses = np.stack([np.eye(4), np.eye(4)])
    np.savez(bundle / "cameras.npz", **{"0": poses, "1": poses})
    for frame in (3, 4):
        (bundle / "images" / f"{frame:03d}_0.png").write_bytes(
            (bundle / "images" / f"{frame:03d}_1.png").read_bytes()
        )
    renders = tmp_path / "renders"
    renders.mkdir()
    Image.fromarray(np.full((12, 16, 3), 100, np.uint8)).save(renders / "004_0.png")
    (renders / "004_1.png").write_bytes((bundle / "images" / "004_1.png").read_bytes())

    class FakeLpips(torch.nn.Module):
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (prediction - target).abs().mean()

    monkeypatch.setitem(
        sys.modules,
        "torchmetrics.image.lpip",
        SimpleNamespace(LearnedPerceptualImagePatchSimilarity=lambda **kwargs: FakeLpips()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "scores.json"
    main(["--observations", str(bundle), "--renders", str(renders), "--out", str(output)])
    scores = json.loads(output.read_text())
    assert scores["overall"]["psnr"] == scores["per_camera"]["image_00"]["psnr"]
    assert scores["shifted"]["cameras"] == ["image_02"]
    assert scores["shifted"]["psnr"] == float("inf")
    assert scores["shifted"]["psnr_cc"] is None  # ảnh đồng màu không khớp affine được
    assert scores["count"] == 2
