"""Đọc kích thước ảnh JPEG và PNG, ghi mặt nạ PNG, không cần thư viện xử lý ảnh.

Ingest, ``source inspect`` và bộ kiểm tra cần biết độ phân giải của ảnh camera nhưng chạy trong
môi trường nhẹ (không có Pillow, OpenCV). Kích thước nằm sẵn trong phần đầu file: PNG ở khối
``IHDR``, JPEG ở khối ``SOF``. File này chứa:
- ``IMAGE_SUFFIXES``: các đuôi file được coi là ảnh.
- ``read_image_size``: trả về ``(rộng, cao)`` của một ảnh, hoặc ``None`` nếu không đọc được.
- ``write_mask_png``: ghi mặt nạ xám 8 bit (0 hoặc 255) gồm các hình chữ nhật, như mặt nạ vật
  động mà script tiền xử lý của DriveStudio ghi.
"""

import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Các marker SOF của JPEG chứa kích thước ảnh (baseline, progressive, ...); bỏ DHT (C4), JPG (C8),
# DAC (CC) vì chúng không phải SOF dù cùng dải giá trị.
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _png_size(data: bytes) -> Optional[tuple[int, int]]:
    """Đọc kích thước từ khối ``IHDR`` ngay sau chữ ký PNG."""
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _jpeg_size(path: Path) -> Optional[tuple[int, int]]:
    """Duyệt các khối của JPEG tới khối ``SOF`` đầu tiên rồi đọc kích thước."""
    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = file.read(1)
            while byte == b"\xff":  # bỏ các byte đệm 0xFF trước marker
                byte = file.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # marker không có độ dài
                continue
            length_bytes = file.read(2)
            if len(length_bytes) < 2:
                return None
            (length,) = struct.unpack(">H", length_bytes)
            if marker in _JPEG_SOF_MARKERS:
                header = file.read(5)
                if len(header) < 5:
                    return None
                height, width = struct.unpack(">HH", header[1:5])
                return width, height
            file.seek(length - 2, 1)


def read_image_size(path: Path) -> Optional[tuple[int, int]]:
    """Đọc kích thước của một ảnh JPEG hoặc PNG.

    Args:
        path: Đường dẫn file ảnh.

    Returns:
        ``(rộng, cao)`` tính bằng pixel, ví dụ ``(1600, 900)``; ``None`` nếu file không tồn tại,
        không phải JPEG/PNG hoặc bị hỏng.
    """
    try:
        if path.suffix.lower() == ".png":
            with path.open("rb") as file:
                return _png_size(file.read(24))
        if path.suffix.lower() in (".jpg", ".jpeg"):
            return _jpeg_size(path)
    except OSError:
        return None
    return None


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_mask_png(
    path: Path, width: int, height: int, rects: Sequence[tuple[int, int, int, int]]
) -> None:
    """Ghi mặt nạ PNG xám 8 bit: 255 trong các hình chữ nhật ``(u0, v0, u1, v1)``, 0 ở ngoài.

    ``u1``, ``v1`` không tính (như ``mask[v0:v1, u0:u1] = 255``); toạ độ bị kẹp vào ảnh.
    """
    blank = bytes(width + 1)  # mỗi dòng PNG bắt đầu bằng byte bộ lọc 0
    rows = [blank] * height
    for u0, v0, u1, v1 in rects:
        u0, u1 = max(0, u0), min(width, u1)
        for v in range(max(0, v0), min(height, v1)):
            row = bytearray(rows[v])
            row[1 + u0 : 1 + u1] = b"\xff" * max(0, u1 - u0)
            rows[v] = bytes(row)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = zlib.compress(b"".join(rows), 6)
    path.write_bytes(
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", data)
        + _png_chunk(b"IEND", b"")
    )
