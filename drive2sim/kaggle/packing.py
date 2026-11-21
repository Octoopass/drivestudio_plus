r"""Gói output của notebook Kaggle thành vài file ``.tar`` để máy tải về nhanh.

``kaggle kernels output`` tải từng file một, mỗi file một lượt gọi API (khoảng 0,7 giây mỗi file
đo ngày 21/11/2026). Output của một run có hàng nghìn file nhỏ (cache mask SegFormer của một cảnh
mới: hơn 11.000 file, 51 MB), nên tải mất vài giờ dù dung lượng nhỏ. Cuối notebook, hàm
``pack_output`` gói ``runs/`` thành một file và mỗi mục cache thành một file trong
``d2s-pack/``; máy tải các file này rồi ``unpack_output`` bung ra đúng cây thư mục cũ.

File rời trong output vẫn giữ nguyên: notebook sau gắn output này làm input và đọc thẳng run,
cache từ đó (``bundle.materialize``). Xem ARCHITECTURE mục 2.8, CONTRACTS mục 9.2.
Mỗi mục cache một file riêng để máy bỏ qua mục đã có
(``pack_file_pattern``), như ``executor.skip_local_cache`` với file rời.

Chạy trên Kaggle (ô cuối của ``templates/run.ipynb``)::

    python -m drive2sim.kaggle.packing /kaggle/working
"""

import argparse
import re
import shutil
import sys
import tarfile
from pathlib import Path

from drive2sim.engine.cache import PARTIAL_SUFFIX

PACK_DIR = "d2s-pack"
RUNS_ARCHIVE = "runs.tar"


def pack_output(working: Path) -> list[Path]:
    """Gói ``runs/`` và từng mục ``cache/<cảnh>/<mục>/`` của ``working`` vào ``working/d2s-pack/``.

    Gói không nén (``.tar``): ảnh, mask, checkpoint đã nén sẵn hoặc nén kém, nén thêm chỉ tốn
    thời gian notebook. Symlink được giữ là symlink (notebook đã xoá chúng trước bước này). Mục
    cache ghi dở (``.partial``) bị bỏ, như ``remote.merge_output``. Nếu gói lỗi, xoá toàn bộ
    gói đã tạo để máy chuyển sang tải file rời, không nhận nhầm một bộ gói thiếu là đầy đủ.

    Args:
        working: Thư mục làm việc chứa ``runs/`` và ``cache/`` của notebook.

    Returns:
        Các file ``.tar`` đã tạo.

    Raises:
        OSError: Nếu không đọc được đầu vào hoặc không ghi được gói, ví dụ hết chỗ trống.
        tarfile.TarError: Nếu không tạo được gói tar.
    """
    pack = working / PACK_DIR
    shutil.rmtree(pack, ignore_errors=True)
    pack.mkdir(parents=True)
    made = []
    try:
        runs = working / "runs"
        if runs.is_dir():
            target = pack / RUNS_ARCHIVE
            with tarfile.open(target, "w") as archive:
                archive.add(runs, arcname="runs")
            made.append(target)
        for entry in sorted((working / "cache").glob("*/*")):
            if not entry.is_dir() or entry.name.endswith(PARTIAL_SUFFIX):
                continue
            scene = entry.parent.name
            target = pack / "cache" / scene / f"{entry.name}.tar"
            target.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target, "w") as archive:
                archive.add(entry, arcname=f"cache/{scene}/{entry.name}")
            made.append(target)
    except (OSError, tarfile.TarError):
        # Chỉ giữ bộ gói đầy đủ; notebook vẫn giữ file rời để pull tải theo đường cũ.
        shutil.rmtree(pack)
        raise
    return made


def pack_file_pattern(local_cache: Path) -> str:
    """Tạo biểu thức chỉ tải gói output, bỏ gói của mục cache máy đã có.

    Args:
        local_cache: Thư mục ``cache/`` của máy nhận output.

    Returns:
        Biểu thức chính quy truyền vào ``kaggle kernels output --file-pattern``.
    """
    entries = sorted(p for p in local_cache.glob("*/*") if p.is_dir())
    names = "|".join(re.escape(f"{PACK_DIR}/cache/{p.parent.name}/{p.name}.tar") for p in entries)
    skip = f"(?!.*(?:{names})$)" if names else ""
    return f"^{skip}(?:.*/)?{re.escape(PACK_DIR)}/"


def unpack_output(output: Path) -> int:
    """Bung mọi gói trong ``<output>/**/d2s-pack/`` ra thư mục chứa ``d2s-pack``, rồi xoá gói.

    Sau bước này ``output`` có cây ``runs/``, ``cache/`` như khi tải file rời, nên
    ``remote.merge_output`` dùng được như cũ.

    Args:
        output: Thư mục chứa output tải từ Kaggle.

    Returns:
        Số gói đã bung.

    Raises:
        ValueError: Nếu một gói có đường dẫn ra ngoài thư mục đích hoặc là thiết bị đặc biệt.
    """
    count = 0
    for pack in sorted(output.glob(f"**/{PACK_DIR}")):
        if not pack.is_dir():
            continue
        destination = pack.parent
        for path in sorted(pack.rglob("*.tar")):
            with tarfile.open(path) as archive:
                _check_members(archive, destination)
                archive.extractall(destination)
            count += 1
        shutil.rmtree(pack)
    return count


def _check_members(archive: tarfile.TarFile, destination: Path) -> None:
    """Kiểm tra đường dẫn, link và loại file trước khi bung một gói tar.

    Args:
        archive: Gói tar đã mở để đọc.
        destination: Thư mục đích dùng để xác định giới hạn đường dẫn cho phép.

    Raises:
        ValueError: Nếu có đường dẫn hoặc link trỏ ra ngoài đích, hoặc file thiết bị.
    """
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"gói có đường dẫn ra ngoài thư mục đích: {member.name}")
        if member.isdev():
            raise ValueError(f"gói có file thiết bị: {member.name}")
        if member.issym() or member.islnk():
            linked = (target.parent / member.linkname).resolve()
            if member.islnk():
                linked = (root / member.linkname).resolve()
            if root != linked and root not in linked.parents:
                raise ValueError(f"gói có link trỏ ra ngoài thư mục đích: {member.name}")


def main() -> int:
    """Gói output của thư mục làm việc trên dòng lệnh và in số gói, dung lượng.

    Returns:
        Mã thoát 0 khi đóng gói thành công.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("working", type=Path)
    args = parser.parse_args()
    made = pack_output(args.working)
    size = sum(p.stat().st_size for p in made)
    print(f"d2s-pack: {len(made)} gói, {size / 1e6:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
