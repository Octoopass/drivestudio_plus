"""Khung từng camera trong video ghép và bản đồ sai khác (``recon/rendering/videos.py``)."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from drive2sim.recon.rendering.videos import (  # noqa: E402
    ERROR_MAP_MAX,
    INVALID_PIXEL_GREY,
    error_map_image,
    tile_boxes,
    write_video_layout,
)
from drive2sim.recon.utils.visualization import get_layout  # noqa: E402


def test_tile_boxes_follow_dataset_layout() -> None:
    # nuScenes: trước giữa, trước-trái bên trái, trước-phải bên phải
    shapes = [(45, 80)] * 3
    names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    boxes = {b["name"]: b for b in tile_boxes(get_layout("nuscenes"), shapes, names)}
    assert boxes["CAM_FRONT_LEFT"]["x"] < boxes["CAM_FRONT"]["x"] < boxes["CAM_FRONT_RIGHT"]["x"]
    assert all(b["y"] == 0 for b in boxes.values())


def test_tile_boxes_handle_resized_tiles() -> None:
    # KITTI-360 đổi mọi ảnh về cùng chiều rộng: pinhole thấp ở hàng trên, mắt cá vuông ở dưới
    shapes = [(38, 140), (70, 70)]
    boxes = {
        b["name"]: b for b in tile_boxes(get_layout("kitti360"), shapes, ["image_00", "image_02"])
    }
    assert boxes["image_00"]["y"] == 0
    assert boxes["image_02"]["y"] >= boxes["image_00"]["height"] - 1
    assert abs(boxes["image_02"]["width"] - boxes["image_02"]["height"]) <= 1


def test_layout_file_holds_boxes_and_scales(tmp_path: Path) -> None:
    frames = [np.zeros((45, 80, 3), np.float32)] * 3
    names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    write_video_layout(str(tmp_path / "test_set_5.mp4"), get_layout("nuscenes"), frames, names, 2)
    data = json.loads((tmp_path / "test_set_5_layout.json").read_text())
    assert data["fps"] == 2 and len(data["cameras"]) == 3
    assert data["error_map"]["max"] == ERROR_MAP_MAX


def test_error_map_scale_is_fixed_and_invalid_pixels_grey() -> None:
    gt = np.zeros((2, 3, 3), np.float32)
    rgb = gt.copy()
    rgb[0, 1] = ERROR_MAP_MAX  # sai bằng ngưỡng: màu cuối của thang
    rgb[0, 2] = 1.0  # vượt ngưỡng: cũng màu cuối, không đổi thang theo khung
    valid = np.ones((2, 3), bool)
    valid[1, 2] = False
    image = error_map_image(rgb, gt, valid)
    assert image.shape == (2, 3, 3)
    assert np.allclose(image[0, 1], image[0, 2])
    assert image[0, 0].sum() < image[0, 1].sum()
    assert np.allclose(image[1, 2], INVALID_PIXEL_GREY)
    # Cùng sai số cho cùng màu dù khung khác có sai số lớn hơn (DriveStudio chuẩn hoá theo khung).
    other = error_map_image(np.full_like(gt, 0.5), gt, None)
    assert np.allclose(error_map_image(rgb, gt, None)[0, 1], other[0, 0])
