"""Ghi Gaussian ra file SPZ (định dạng nén của Niantic, Spark đọc được), chỉ cần numpy.

File PLY lưu mỗi Gaussian bằng 59 số float32 (236 byte với SH bậc 3); SPZ lượng tử hoá từng
trường xuống 1–3 byte rồi nén gzip, nhỏ hơn khoảng 10 lần. Bố cục theo bản 2 của định dạng
(``load-spz.cc`` của nianticlabs/spz), bản 3:

- header 16 byte: ``magic`` 0x5053474e, ``version`` 3, ``numPoints``, ``shDegree``,
  ``fractionalBits``, ``flags``, ``reserved`` (little-endian);
- vị trí: 3 số nguyên có dấu 24 bit mỗi Gaussian, ``round(x * 2^fractionalBits)``;
- độ đục: ``sigmoid(opacity) * 255``;
- màu (hệ số SH bậc 0): ``f_dc * 0,15 * 255 + 127,5``;
- kích thước: ``(log scale + 10) * 16``;
- hướng: quaternion đã chuẩn hoá (thứ tự x, y, z, w) gói "smallest-three" vào 4 byte: 2 bit chỉ
  thành phần lớn nhất, ba thành phần còn lại mỗi cái 10 bit (1 bit dấu, 9 bit độ lớn chia
  sqrt(1/2)). Bản 2 chỉ giữ x, y, z mỗi số 8 bit, lệch trung bình 1° và tới 12° (đo trên nền
  run ``f8a6``), nên dùng bản 3;
- SH bậc cao: theo hệ số rồi theo kênh màu, ``round(sh * 128) + 128``, bậc 1 giữ 5 bit, bậc 2–3
  giữ 4 bit.

Toạ độ ghi nguyên như đầu vào (không đổi hệ trục), giống ``write_ply``: Spark đọc PLY và SPZ đều
không đổi trục, nên hai file hiện giống nhau.

File này chứa ``write_spz``, ``read_spz`` (đọc lại để kiểm tra).
"""

import gzip
import struct

import numpy as np

MAGIC = 0x5053474E
VERSION = 3
SQRT1_2 = 0.7071067811865476
COLOR_SCALE = 0.15
SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15}


def _u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _quantize_sh(values: np.ndarray, bits: int) -> np.ndarray:
    bucket = 1 << (8 - bits)
    q = np.round(values * 128.0) + 128.0
    q = np.floor((q + bucket / 2) / bucket) * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def _pack_quats(xyzw: np.ndarray) -> np.ndarray:
    """Gói quaternion (x, y, z, w) "smallest-three" thành 4 byte mỗi Gaussian."""
    largest = np.abs(xyzw).argmax(1)
    negate = xyzw[np.arange(len(xyzw)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        skip = largest == i
        value = xyzw[:, i]
        negbit = ((value < 0) ^ negate).astype(np.uint32)
        mag = np.floor(511.0 * (np.abs(value) / SQRT1_2) + 0.5).astype(np.uint32)
        packed = (comp << 10) | (negbit << 9) | np.minimum(mag, 511)
        comp = np.where(skip, comp, packed)
    return comp.astype("<u4").view(np.uint8).reshape(-1, 4)


def _unpack_quats(raw: np.ndarray) -> np.ndarray:
    """Ngược lại ``_pack_quats``: trả quaternion (w, x, y, z)."""
    comp = raw.astype(np.uint8).copy().view("<u4")[:, 0].astype(np.int64)
    largest = comp >> 30
    xyzw = np.zeros((len(comp), 4))
    for i in (3, 2, 1, 0):
        here = largest != i
        mag = (comp & 511) / 511.0 * SQRT1_2
        sign = np.where((comp >> 9) & 1, -1.0, 1.0)
        xyzw[:, i] = np.where(here, sign * mag, 0.0)
        comp = np.where(here, comp >> 10, comp)
    rest = np.clip(1.0 - (xyzw**2).sum(1), 0.0, None)
    xyzw[np.arange(len(comp)), largest] = np.sqrt(rest)
    return np.concatenate([xyzw[:, 3:], xyzw[:, :3]], 1)


def fractional_bits(means: np.ndarray, wanted: int = 12) -> int:
    """Số bit phần lẻ của vị trí: 12 (0,24 mm) nếu đủ chỗ, ít hơn khi cảnh rộng quá 2 km."""
    largest = float(np.abs(means).max()) if len(means) else 0.0
    bits = wanted
    while bits > 0 and largest * (1 << bits) >= (1 << 23) - 1:
        bits -= 1
    return bits


def write_spz(g: dict, path: str) -> None:
    """Ghi các Gaussian ``g`` (cùng dạng ``gaussians_from_state``) ra ``path``.

    Args:
        g: ``means`` (N,3), ``scales`` (N,3, log), ``quats`` (N,4, wxyz), ``f_dc`` (N,3),
            ``f_rest`` (N, 3K, thứ tự INRIA: hết kênh R rồi G, B), ``opac`` (N,1, logit); tensor
            torch hoặc mảng numpy.
        path: File ra, thường đuôi ``.spz``.
    """
    arrays = {
        k: np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, np.float64)
        for k, v in g.items()
    }
    means, n = arrays["means"], len(arrays["means"])
    coeffs = arrays["f_rest"].shape[1] // 3
    degree = {v: k for k, v in SH_DIM.items()}[coeffs]
    bits = fractional_bits(means)

    fixed = np.round(means * (1 << bits)).astype(np.int64) & 0xFFFFFF
    positions = np.stack([(fixed >> s) & 0xFF for s in (0, 8, 16)], -1).astype(np.uint8)
    alphas = _u8(255.0 / (1.0 + np.exp(-arrays["opac"][:, 0])))
    colors = _u8(arrays["f_dc"] * (COLOR_SCALE * 255.0) + 127.5)
    scales = _u8((arrays["scales"] + 10.0) * 16.0)
    wxyz = arrays["quats"] / np.linalg.norm(arrays["quats"], axis=1, keepdims=True)
    rotations = _pack_quats(np.concatenate([wxyz[:, 1:], wxyz[:, :1]], 1))
    parts = [positions, alphas, colors, scales, rotations]
    if coeffs:
        sh = arrays["f_rest"].reshape(n, 3, coeffs).transpose(0, 2, 1)  # (N, hệ số, kênh)
        packed = np.empty(sh.shape, np.uint8)
        packed[:, :3] = _quantize_sh(sh[:, :3], 5)
        packed[:, 3:] = _quantize_sh(sh[:, 3:], 4)
        parts.append(packed)
    header = struct.pack("<IIIBBBB", MAGIC, VERSION, n, degree, bits, 0, 0)
    body = b"".join(part.tobytes() for part in parts)
    with gzip.open(path, "wb", compresslevel=6) as file:
        file.write(header + body)


def read_spz(path: str) -> dict:
    """Đọc lại file do ``write_spz`` ghi (để test và đo sai số lượng tử hoá)."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    magic, version, n, degree, bits, _, _ = struct.unpack_from("<IIIBBBB", data)
    assert magic == MAGIC and version == VERSION, "không phải SPZ bản 3"
    offset = 16

    def take(count: int, shape: tuple) -> np.ndarray:
        nonlocal offset
        chunk = np.frombuffer(data, np.uint8, count, offset).reshape(shape)
        offset += count
        return chunk.astype(np.float64)

    raw = take(n * 9, (n, 3, 3)).astype(np.int64)
    fixed = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alphas = take(n, (n,)) / 255.0
    colors = take(n * 3, (n, 3))
    scales = take(n * 3, (n, 3))
    quats = _unpack_quats(take(n * 4, (n, 4)))
    coeffs = SH_DIM[degree]
    sh = take(n * coeffs * 3, (n, coeffs, 3)) if coeffs else np.zeros((n, 0, 3))
    eps = 1e-6
    return {
        "means": fixed / (1 << bits),
        "opac": np.log(np.clip(alphas, eps, 1 - eps) / np.clip(1 - alphas, eps, 1))[:, None],
        "f_dc": (colors - 127.5) / (COLOR_SCALE * 255.0),
        "scales": scales / 16.0 - 10.0,
        "quats": quats,
        "f_rest": ((sh - 128.0) / 128.0).transpose(0, 2, 1).reshape(n, coeffs * 3),
    }
