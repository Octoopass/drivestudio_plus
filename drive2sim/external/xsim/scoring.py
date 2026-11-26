"""Chấm ảnh render của XSIM trong môi trường chính, như ``eval.standard`` chấm DriveStudio.

``score_renders`` đọc ảnh render 8-bit worker ghi và ảnh thật của gói quan sát, nhân mask đánh giá
của gói cho cả hai ảnh, rồi gọi ``compute_image_metrics`` với cùng LPIPS (torchmetrics, AlexNet,
``normalize=True``) mà trainer DriveStudio dùng. Ảnh thật, mask và phiên bản thư viện đo vì thế
giống hệt các approach DriveStudio. Stage ``eval.xsim`` chạy ``python -m
drive2sim.external.xsim.scoring`` thành tiến trình con. Xem SPEC mục 9.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np
import torch
from PIL import Image

from drive2sim.core.schema import Metrics, ShiftedMetrics
from drive2sim.external.xsim.bundle import Observations
from drive2sim.recon.rendering.color_match import apply_color_affine, fit_color_affine
from drive2sim.recon.rendering.image_metrics import compute_image_metrics


def _read(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float() / 255


def score_renders(
    observations: Observations,
    renders: Path,
    test: set[tuple[int, str]],
    lpips: Callable,
    device: str = "cpu",
    color_matched: bool = False,
) -> tuple[Metrics, dict[str, Metrics], int]:
    """Đo metric trên đúng các cặp test; báo lỗi khi thiếu ảnh render.

    Args:
        observations: Gói quan sát của run.
        renders: Thư mục ``<frame>_<camera>.png`` do worker ghi.
        test: Các cặp (khung, tên camera) của split test.
        lpips: Module LPIPS nhận hai tensor BCHW trong [0, 1].
        device: Thiết bị tính LPIPS.
        color_matched: Khớp affine màu theo protocol chung; bỏ ảnh có phép khớp suy biến.

    Returns:
        Metric chung theo ảnh, metric từng camera, và số ảnh được chấm.

    Raises:
        ValueError: Nếu thiếu ảnh render, sai kích thước, hoặc không ảnh nào có pixel hợp lệ.
    """
    indices = observations.camera_indices
    values: dict[str, list[tuple]] = defaultdict(list)
    for frame, name in sorted(test):
        camera = indices[name]
        path = renders / f"{frame:03d}_{camera}.png"
        if not path.is_file():
            raise ValueError(f"thiếu ảnh render của cặp test ({frame}, {name})")
        rgb = _read(path).to(device)
        gt = _read(observations.image_path(frame, camera)).to(device)
        if rgb.shape != gt.shape:
            raise ValueError(f"ảnh render ({frame}, {name}) khác kích thước ảnh thật")
        mask = observations.eval_mask(camera)
        valid = None
        if mask is not None:
            valid = torch.from_numpy(mask.copy()).to(device)
            rgb, gt = rgb * valid[..., None], gt * valid[..., None]
        if color_matched:
            rgb_array, gt_array = rgb.cpu().numpy(), gt.cpu().numpy()
            valid_array = valid.cpu().numpy() if valid is not None else None
            affine = fit_color_affine(rgb_array, gt_array, valid_array)
            if affine is None:
                continue
            rgb = torch.from_numpy(apply_color_affine(rgb_array, affine)).to(
                device=device, dtype=gt.dtype
            )
        result = compute_image_metrics(rgb, gt, valid, lpips)
        if result is not None:
            values[name].append(result)
    if not values:
        raise ValueError("không ảnh test nào có pixel hợp lệ")

    def average(rows: list[tuple]) -> Metrics:
        """Trung bình từng metric, mỗi ảnh một trọng số."""
        psnr, ssim, perceptual = map(mean, zip(*rows))
        return Metrics(psnr=psnr, ssim=ssim, lpips=perceptual)

    rows = [row for items in values.values() for row in items]
    return average(rows), {name: average(items) for name, items in values.items()}, len(rows)


def main(argv: Optional[list[str]] = None) -> None:
    """Chấm các cặp test của split ghi trong gói, ghi ``overall``, ``per_camera``, ``count``."""
    parser = argparse.ArgumentParser(description="Chấm ảnh render XSIM trên gói quan sát")
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    observations = Observations(args.observations)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).eval().to(device)
    test = {(int(frame), str(name)) for frame, name in observations.split["test"]}
    test_only = set(observations.split.get("test_only_cameras", []))
    shifted_pairs = {pair for pair in test if pair[1] in test_only}
    with torch.no_grad():
        overall, per_camera, count = score_renders(
            observations, args.renders, test - shifted_pairs, lpips, device
        )
        shifted = None
        if shifted_pairs:
            raw, shifted_cameras, shifted_count = score_renders(
                observations, args.renders, shifted_pairs, lpips, device
            )
            per_camera.update(shifted_cameras)
            corrected = None
            try:
                corrected, _, _ = score_renders(
                    observations, args.renders, shifted_pairs, lpips, device, color_matched=True
                )
            except ValueError as error:
                if str(error) != "không ảnh test nào có pixel hợp lệ":
                    raise
            shifted = ShiftedMetrics(
                cameras=sorted(shifted_cameras),
                num_images=shifted_count,
                psnr=raw.psnr,
                ssim=raw.ssim,
                lpips=raw.lpips,
                psnr_cc=corrected.psnr if corrected is not None else None,
                lpips_cc=corrected.lpips if corrected is not None else None,
            )
            count += shifted_count
    payload = {
        "overall": overall.model_dump(),
        "per_camera": {name: metrics.model_dump() for name, metrics in per_camera.items()},
        "count": count,
        "shifted": shifted.model_dump() if shifted is not None else None,
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
