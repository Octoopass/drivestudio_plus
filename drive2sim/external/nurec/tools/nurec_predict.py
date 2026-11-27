"""Chạy ``run_inference.py`` của Instant NuRec mà không để các lượt ghi đè PLY của nhau.

Dùng::

    python nurec_predict.py <các tham số của run_inference.py>

NuRec dựng mỗi lượt 8 đoạn (``predict_batch_size``), và với ``--merge`` mỗi lượt ghi PLY đã ghép
ra cùng một tên ``<sequence>.ply``: clip hơn 8 đoạn chỉ còn đoạn đường của lượt cuối. Dựng cả
clip trong một lượt thì bước ghép hết VRAM (29 đoạn: hơn 47 GB). Script này đổi tên đầu ra của
mỗi lượt ngay khi lượt đó ghi xong thành ``<sequence>_partNN.*``; stage ``train.nurec`` ghép các
phần lại thành một PLY cho cả clip.

``--no-voxelize`` (của drive2sim, không chuyển cho NuRec): vẫn ghép các đoạn (``--merge``) nhưng
bỏ bước voxel hoá lớp tĩnh, nên giữ mọi Gaussian model dự đoán thay vì gộp về ``--n-gaussians``.
NuRec gắn voxel hoá với ``--merge`` (``PrimitiveMergeConfig.enable_voxelization = merge``), nên
script thay ``KelvinPrimitiveMerge._maybe_voxelize_static_layer`` bằng hàm trả nguyên đầu vào.

Chạy bằng Python của venv NuRec, thư mục làm việc là thư mục mã NuRec.
"""

from __future__ import annotations

import sys
from pathlib import Path

PART_TAG = "_part"
NO_VOXELIZE = "--no-voxelize"


def rename_outputs(ply_dir: Path, part: int) -> list[Path]:
    """Đổi ``<sequence>.<đuôi>`` trong từng thư mục sequence thành ``<sequence>_partNN.<đuôi>``.

    Returns:
        Các file sau khi đổi tên.
    """
    renamed = []
    for sequence_dir in sorted(p for p in ply_dir.glob("*") if p.is_dir()):
        stem = sequence_dir.name
        for path in sorted(sequence_dir.iterdir()):
            if path.name.startswith(stem + "."):
                target = path.with_name(f"{stem}{PART_TAG}{part:02d}{path.name[len(stem) :]}")
                path.rename(target)
                renamed.append(target)
    return renamed


def main() -> int:
    sys.path.insert(0, str(Path.cwd()))
    from instant_nurec import cli
    from instant_nurec.model import system as system_module

    cls = system_module.GaussiansInstantNuRecSystem
    original = cls.on_predict_batch_end
    parts = [0]

    def on_predict_batch_end(self, outputs, batch) -> None:
        original(self, outputs, batch)
        renamed = rename_outputs(Path(self.out_dir) / self.run_id / "ply", parts[0])
        print(f"Part {parts[0]:02d}: {len(renamed)} file", flush=True)
        parts[0] += 1

    cls.on_predict_batch_end = on_predict_batch_end
    args = sys.argv[1:]
    if NO_VOXELIZE in args:
        from instant_nurec.predict import primitive_merge

        args = [a for a in args if a != NO_VOXELIZE]
        primitive_merge.KelvinPrimitiveMerge._maybe_voxelize_static_layer = (
            lambda self, primitive: primitive
        )
        print("Không giới hạn số Gaussian: bỏ bước voxel hoá khi ghép", flush=True)
    return cli.main(args)


if __name__ == "__main__":
    sys.exit(main())
