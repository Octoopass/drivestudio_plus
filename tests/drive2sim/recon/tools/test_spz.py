"""Bộ ghi SPZ: đọc lại đúng trong sai số lượng tử hoá, và đúng bố cục byte mà Spark ghi.

Bố cục byte được so với file SPZ do chính ``writeSpz`` của Spark 2.4.0 ghi từ cùng dữ liệu (đã
làm bằng tay trong trình duyệt ngày 16/11/2026): header, rồi SH theo hệ số rồi theo kênh màu.
"""

import gzip
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drive2sim.recon.tools.spz import read_spz, write_spz  # noqa: E402


def _gaussians(n: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(n, 4))
    return {
        "means": rng.uniform(-300, 300, (n, 3)),
        "scales": rng.uniform(-6, 1, (n, 3)),
        "quats": quats / np.linalg.norm(quats, axis=1, keepdims=True),
        "f_dc": rng.uniform(-2, 2, (n, 3)),
        "f_rest": rng.uniform(-0.5, 0.5, (n, 45)),
        "opac": rng.uniform(-4, 4, (n, 1)),
    }


def test_round_trip_within_quantisation(tmp_path: Path) -> None:
    g = _gaussians()
    write_spz(g, tmp_path / "a.spz")
    r = read_spz(tmp_path / "a.spz")
    assert np.abs(r["means"] - g["means"]).max() < 2.5e-4  # 12 bit phần lẻ: 0,24 mm
    assert np.abs(r["scales"] - g["scales"]).max() <= 1 / 32 + 1e-9
    dot = np.abs((r["quats"] * g["quats"]).sum(1))
    assert np.degrees(2 * np.arccos(np.clip(dot, 0, 1))).max() < 0.5
    assert np.abs(r["f_rest"] - g["f_rest"]).max() <= 0.07  # SH bậc 2–3 giữ 4 bit
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    assert np.abs(sig(r["opac"]) - sig(g["opac"])).max() <= 1 / 255 + 1e-9


def test_layout_matches_spark(tmp_path: Path) -> None:
    n = 4
    g = {
        "means": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
        "scales": np.full((n, 3), -3.0),
        "quats": np.tile([1.0, 0, 0, 0], (n, 1)),
        "f_dc": np.zeros((n, 3)),
        "f_rest": np.tile((np.arange(45) - 22) / 32, (n, 1)),
        "opac": np.zeros((n, 1)),
    }
    write_spz(g, tmp_path / "a.spz")
    data = gzip.decompress((tmp_path / "a.spz").read_bytes())
    assert struct.unpack_from("<IIIBBB", data) == (0x5053474E, 3, 4, 3, 12, 0)
    sh = list(data[16 + n * 20 : 16 + n * 20 + 12])
    assert sh == [40, 104, 160, 48, 104, 168, 48, 112, 168, 48, 112, 176]  # như Spark ghi


def test_wide_scene_keeps_positions_in_range(tmp_path: Path) -> None:
    g = _gaussians()
    g["means"][0] = [2655.4, 0, 0]  # nền nuScenes có điểm xa tới 2,6 km
    write_spz(g, tmp_path / "a.spz")
    r = read_spz(tmp_path / "a.spz")
    assert np.abs(r["means"][0, 0] - 2655.4) < 1e-3
