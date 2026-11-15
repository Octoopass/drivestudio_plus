"""Tên notebook và dataset mà tool tạo trên Kaggle (CONTRACTS mục 10.1).

Tên đọc là hiểu: ``d2s-<loại>-<cảnh hoặc nguồn>-...``, chỉ dùng chữ thường, số và ``-`` (slug của
Kaggle), tối đa 50 ký tự (giới hạn tiêu đề notebook và dataset của Kaggle):

- Notebook train: ``d2s-<cảnh>-<approach>-<biến thể>-<4 hex của run>-<lần đẩy>``, ví dụ
  ``d2s-nuscenes-0061-omnire-3dgs-fastgs-453a-1``.
- Dataset cảnh upload từ máy: ``d2s-scene-<cảnh>-<commit converter>``, ví dụ
  ``d2s-scene-nuscenes-0061-96876ad``.
- Notebook ingest: ``d2s-ingest-<nguồn>[-<số của chuỗi gốc>]`` hoặc ``d2s-ingest-<--scene>``, ví
  dụ ``d2s-ingest-nuscenes-mini-0061``.
- Dataset dữ liệu gốc (người dùng tự tạo): ``d2s-raw-<dataset>-<phần>``, ví dụ
  ``d2s-raw-kitti360-0003``.
- Dataset mô hình có bản quyền riêng (người dùng tự tạo): ``d2s-model-<tên>``, ví dụ
  ``d2s-model-smpl``.

``<cảnh>`` là ``<dataset>-<số cuối của chuỗi gốc>`` (``nuscenes_scene-0061`` → ``nuscenes-0061``,
``kitti360_2013_05_28_drive_0003_sync`` → ``kitti360-0003``). ``<biến thể>``: lựa chọn của mọi
mục chọn (renderer luôn có mặt) cộng các mục bật/tắt khác mặc định (``variant_label``). Notebook
train không có chữ loại, vì tên đã có approach và renderer; các loại khác có chữ ``scene``,
``ingest``, ``raw``. Tên quá 50 ký tự thì bỏ tiền tố ``d2s-`` của notebook train trước, rồi bỏ
nguyên từ ở cuối phần biến thể (thông tin đầy đủ nằm trong ``run.json``).

File này chứa ``slug``, ``scene_label``, ``variant_label``, ``train_kernel``, ``scene_dataset``,
``ingest_kernel``.
"""

import re
from collections.abc import Mapping
from typing import Optional

MAX_LENGTH = 50
_DIGITS = re.compile(r"(\d+)\D*$")


def slug(text: str) -> str:
    """Chữ thường, ``a-z 0-9 -``, không có ``-`` lặp hay ở hai đầu."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def scene_label(scene_id: str) -> str:
    """Tên ngắn của cảnh: ``<dataset>-<số cuối của chuỗi gốc>``.

    ``scene_id`` có dạng ``<dataset>_<chuỗi gốc>`` (mục 10.1). Chuỗi gốc không có số thì giữ cả
    chuỗi.
    """
    dataset, _, sequence = scene_id.partition("_")
    match = _DIGITS.search(sequence)
    return slug(f"{dataset}-{match.group(1) if match else sequence}")


def variant_label(options: Mapping[str, object], defaults: Mapping[str, object]) -> str:
    """Biến thể của run, theo thứ tự mục trong ``options``.

    Mục chọn (ví dụ renderer) luôn ghi lựa chọn (``3dgs``, ``3dgeer``), để hai run khác cách render
    luôn khác tên. Mục bật/tắt chỉ ghi khi khác mặc định: bật ghi tên mục (``fastgs``), tắt ghi
    ``no-<tên>``. Không có gì để ghi thì là ``base``.
    """
    parts = []
    for name, value in options.items():
        if isinstance(value, bool):
            if value != defaults.get(name):
                parts.append(name if value else f"no-{name}")
        else:
            parts.append(str(value))
    return slug("-".join(parts)) or "base"


def _fit(head: str, middle: str, tail: str) -> str:
    """Ghép ``head-middle-tail``; quá ``MAX_LENGTH`` thì bỏ nguyên từ cuối của ``middle``."""
    words = middle.split("-") if middle else []
    while True:
        name = "-".join(part for part in (head, "-".join(words), tail) if part)
        if len(name) <= MAX_LENGTH or not words:
            return name[:MAX_LENGTH].rstrip("-")
        words.pop()


def train_kernel(
    scene_id: str,
    run_id: str,
    attempt: int,
    options: Mapping[str, object],
    defaults: Mapping[str, object],
) -> str:
    """Tên notebook train của một lần đẩy run (không có người dùng).

    Args:
        scene_id: Cảnh của run.
        run_id: ``<YYYYMMDD-HHMMSS>-<approach_id>-<4 hex>``.
        attempt: Lần đẩy thứ mấy (1, 2...).
        options: Lựa chọn của các mục cấu hình trong ``run.json``.
        defaults: Lựa chọn mặc định của approach.
    """
    approach_id, short_id = run_id.split("-", 2)[2].rsplit("-", 1)
    head = f"d2s-{scene_label(scene_id)}-{slug(approach_id)}"
    variant = variant_label(options, defaults)
    tail = f"{short_id}-{attempt}"
    if len(f"{head}-{variant}-{tail}") > MAX_LENGTH:
        head = head[len("d2s-") :]  # thiếu chỗ thì bỏ tiền tố trước, giữ đủ biến thể
    return _fit(head, variant, tail)


def scene_dataset(scene_id: str, converter_version: str) -> str:
    """Tên dataset chứa cảnh upload từ máy: đổi converter thì là dataset mới.

    ``converter_version`` có dạng ``<phiên bản>+<commit>``; lấy commit (hoặc cả chuỗi nếu không
    có ``+``).
    """
    version = slug(converter_version.rsplit("+", 1)[-1])
    return _fit(f"d2s-scene-{scene_label(scene_id)}", "", version)


def ingest_kernel(scene: Optional[str], sequence: Optional[str], source: str) -> str:
    """Tên notebook ingest: theo ``--scene`` nếu có, không thì theo nguồn và ``--sequence``.

    Args:
        scene: ``scene_id`` người dùng đặt (``--scene``), hoặc ``None``.
        sequence: Chuỗi gốc (``--sequence``), hoặc ``None`` khi nguồn chỉ có một chuỗi.
        source: Dataset hoặc notebook nguồn, ``<người dùng>/<tên>``.
    """
    if scene:
        return _fit("d2s-ingest", slug(scene), "")
    # Bỏ các chữ không nói lên nội dung: tiền tố d2s, chữ raw của dataset dữ liệu gốc.
    name = re.sub(r"(^d2s-|^raw-|-raw$)", "", slug(source.split("/")[-1]))
    name = re.sub(r"^raw-", "", name)
    middle = name
    if sequence:
        match = _DIGITS.search(sequence)
        middle = f"{name}-{match.group(1) if match else slug(sequence)}"
    return _fit("d2s-ingest", middle, "")
