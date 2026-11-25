"""Kiểm tra phần không cần GPU của ``recon/tools/eval_ply.py``.

- ``load_ply_gaussians`` đọc lại đúng file ``.ply`` mà ``export_ply.py`` ghi (bố cục SH theo kênh
  màu, hàm kích hoạt của độ mờ đục và kích thước);
- ``camera_group`` chia camera thành pinhole và fisheye;
- ``read_native_psnr`` lấy PSNR nền tĩnh của render gốc từ ``frames.jsonl``, bỏ ảnh ``null``.
Phần vẽ bằng gsplat chưa chạy thật ở đây. Cần PyTorch và plyfile nên bị bỏ qua ở ``.venv`` nhẹ.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("plyfile")

from drive2sim.recon.tools.eval_ply import (  # noqa: E402
    camera_group,
    load_ply_gaussians,
    read_native_psnr,
    renderer_mode,
)
from drive2sim.recon.tools.export_ply import gaussians_from_state, write_ply  # noqa: E402


def test_load_ply_round_trips_export(tmp_path: Path) -> None:
    torch.manual_seed(0)
    num, rest = 6, 15  # SH bậc 3: 1 hệ số DC và 15 hệ số còn lại mỗi kênh màu
    state = {
        "_means": torch.randn(num, 3),
        "_scales": torch.randn(num, 3) * 0.1 - 2.0,
        "_quats": torch.randn(num, 4),
        "_features_dc": torch.randn(num, 1, 3),
        "_features_rest": torch.randn(num, rest, 3),
        "_opacities": torch.randn(num, 1),
    }
    path = tmp_path / "scene.ply"
    write_ply(gaussians_from_state(state), str(path))
    loaded = load_ply_gaussians(str(path))
    assert loaded["sh"].shape == (num, rest + 1, 3)
    assert torch.allclose(loaded["sh"][:, 0], state["_features_dc"][:, 0], atol=1e-6)
    assert torch.allclose(loaded["sh"][:, 1:], state["_features_rest"], atol=1e-6)
    assert torch.allclose(loaded["means"], state["_means"], atol=1e-6)
    assert torch.allclose(loaded["scales"], torch.exp(state["_scales"]), atol=1e-6)
    assert torch.allclose(loaded["opacities"], torch.sigmoid(state["_opacities"][:, 0]), atol=1e-6)
    norms = loaded["quats"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(num), atol=1e-5)


def test_camera_group() -> None:
    assert camera_group("pinhole") == "pinhole"
    assert camera_group("fisheye") == "fisheye" and camera_group("ftheta") == "fisheye"


def test_read_native_psnr_skips_missing_static_psnr(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    records = [
        {"frame_index": 8, "camera_index": 0, "psnr_static": 25.5},
        {"frame_index": 8, "camera_index": 1, "psnr_static": None},
        {"frame_index": 16, "camera_index": 0, "psnr_static": 24.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n\n")
    assert read_native_psnr(str(path)) == {(8, 0): 25.5, (16, 0): 24.0}


@pytest.mark.parametrize(
    "camera_model,has_distortion,expected",
    [
        ("pinhole", False, "default"),  # renderer chung: 3DGS gốc
        ("fisheye", False, "ut"),  # renderer chung cho camera mắt cá: 3DGUT
        ("ftheta", False, "ut"),
        # pinhole có hệ số méo (Waymo): chế độ default của gsplat không vẽ được méo, giữ chế độ
        # lúc train (UT hoặc GEER) thay vì ép default rồi lỗi
        ("pinhole", True, None),
    ],
)
def test_renderer_mode(camera_model: str, has_distortion: bool, expected: object) -> None:
    assert renderer_mode(camera_model, has_distortion) == expected
