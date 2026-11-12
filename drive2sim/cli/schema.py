"""Lệnh ``d2s schema export``: xuất JSON Schema của mọi loại file trong CONTRACTS.

Giao diện Studio viết bằng TypeScript sinh kiểu dữ liệu từ các file JSON Schema này
(``npm run types`` trong ``drive2sim/studio/frontend/``), nên tên và kiểu trường ở frontend luôn
khớp với ``drive2sim/core/schema.py``, không có bảng ánh xạ nào phải giữ tay (CONTRACTS mục 11.2).

File này chứa ``register`` (thêm lệnh vào parser) và ``export_schemas`` (ghi mỗi loại file một
file ``<loại>.schema.json``).
"""

import argparse
import json
from pathlib import Path

from drive2sim.cli.output import emit, info
from drive2sim.core.schema import MODELS_BY_KIND


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``schema`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("schema", help="xuất JSON Schema của các file")
    actions = group.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export", parents=[common], help="ghi <loại>.schema.json")
    export.add_argument("--out", type=Path, required=True, help="thư mục ghi file")
    export.set_defaults(handler=export_schemas)


def export_schemas(args: argparse.Namespace) -> int:
    """Ghi JSON Schema của từng loại file (``manifest``, ``run``...) vào ``--out``.

    Args:
        args: Tham số dòng lệnh, có ``out``.

    Returns:
        Mã thoát ``0``.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, model in sorted(MODELS_BY_KIND.items()):
        path = args.out / f"{kind}.schema.json"
        schema = model.model_json_schema()
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path.name)
    info(args, f"đã ghi {len(written)} file vào {args.out}")
    emit(args, written, "\n".join(written))
    return 0
