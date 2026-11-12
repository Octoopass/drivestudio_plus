"""Phần dùng chung của mọi lệnh ``d2s``: in kết quả, in tiến độ, đọc giá trị ``--set``.

Quy tắc in theo CONTRACTS mục 10.4: kết quả có cấu trúc (bảng, hoặc JSON khi có ``--json``) ra
**stdout**; log, tiến độ, cảnh báo ra **stderr**. Nhờ vậy ``d2s run list --json | jq`` luôn nhận
JSON sạch.

File này chứa:
- ``UsageError``: lỗi do người dùng gõ sai giá trị (mã thoát ``2``).
- ``CommandError``: lỗi đã lường trước khi chạy lệnh, ví dụ tính năng chưa có (mã thoát ``1``).
- ``emit``, ``format_table``, ``format_value``, ``info``: in kết quả và thông báo.
- ``event_printer``: tạo hàm nhận sự kiện của engine và in tiến độ ra stderr.
- ``parse_value``, ``parse_set``: đọc giá trị ``--set <khoá>=<giá trị>`` theo CONTRACTS mục 10.3.
"""

import argparse
import json
import sys
from collections.abc import Sequence

import yaml

from drive2sim.engine.run import EventSink


class UsageError(ValueError):
    """Lỗi do người dùng gõ sai giá trị trên dòng lệnh; ứng với mã thoát ``2``."""


class CommandError(RuntimeError):
    """Lỗi đã lường trước khi chạy một lệnh (ví dụ tính năng chưa có); ứng với mã thoát ``1``.

    Attributes:
        code: Mã lỗi trong bảng ``ERROR_CODES`` (CONTRACTS mục 11.3).
    """

    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


def info(args: argparse.Namespace, message: str) -> None:
    """In một thông báo cho người đọc ra stderr, trừ khi có ``--quiet``.

    Args:
        args: Tham số dòng lệnh đã phân tích.
        message: Thông báo tiếng Việt.
    """
    if not getattr(args, "quiet", False):
        print(message, file=sys.stderr)


def format_value(value: object) -> str:
    """Viết một giá trị cho người đọc, cùng cách viết với YAML và ``--set``.

    Args:
        value: Giá trị bất kỳ.

    Returns:
        ``"true"``/``"false"`` cho boolean, ``str(value)`` cho giá trị khác.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Dựng một bảng chữ căn cột cho đầu ra không có ``--json``.

    Args:
        headers: Tên cột.
        rows: Các dòng; mỗi dòng cùng số phần tử với ``headers``.

    Returns:
        Bảng nhiều dòng; chỉ có dòng tiêu đề nếu ``rows`` rỗng.
    """
    cells = [[str(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[column]) for row in cells) for column in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in cells]
    return "\n".join(line.rstrip() for line in lines)


def emit(args: argparse.Namespace, data: object, text: str) -> None:
    """In kết quả của một lệnh ra stdout: JSON khi có ``--json``, chữ khi không.

    Args:
        args: Tham số dòng lệnh đã phân tích.
        data: Kết quả dạng JSON được (dict, list...), cùng tên trường với file trong CONTRACTS.
        text: Kết quả dạng chữ cho người đọc.
    """
    if getattr(args, "json", False):
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


def event_printer(args: argparse.Namespace) -> EventSink:
    """Tạo hàm nhận sự kiện của engine và in tiến độ ra stderr.

    Mỗi sự kiện một dòng, ví dụ ``[train.drivestudio] running`` hoặc
    ``[train.drivestudio] 50% đang train``. Metric chỉ in khi có ``--verbose``; không in gì khi có
    ``--quiet``.

    Args:
        args: Tham số dòng lệnh đã phân tích.

    Returns:
        Hàm nhận một sự kiện dạng ``{"event": ..., ...}`` (CONTRACTS mục 11.4).
    """

    def sink(event: dict[str, object]) -> None:
        kind = event.get("event")
        where = event.get("stage_id") or event.get("run_id")
        if kind == "status":
            info(args, f"[{where}] {event['status']}")
        elif kind == "progress":
            percent = round(float(event["fraction"]) * 100)  # type: ignore[arg-type]
            info(args, f"[{where}] {percent}% {event.get('message', '')}".rstrip())
        elif kind == "log":
            info(args, f"[{where}] {event['level']}: {event['message']}")
        elif kind == "metric" and getattr(args, "verbose", False):
            info(args, f"[{where}] {event['name']}={event['value']} (bước {event['step']})")

    return sink


def parse_value(raw: str) -> object:
    """Đọc giá trị của ``--set`` theo CONTRACTS mục 10.3.

    ``true``/``false`` thành boolean, ``10`` thành số nguyên, ``0.5`` thành số thực, chuỗi có dấu
    phẩy thành danh sách chuỗi, phần còn lại là chuỗi. Đặt trong ngoặc kép để giữ nguyên chuỗi,
    ví dụ ``"10"`` hoặc ``"a,b"``.

    Args:
        raw: Phần sau dấu ``=``.

    Returns:
        Giá trị đã đổi kiểu, ví dụ ``True``, ``10``, ``["a", "b"]``, ``"3dgeer"``.

    Raises:
        UsageError: Nếu giá trị là cấu trúc YAML phức tạp (mapping), không hỗ trợ.
    """
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    value = yaml.safe_load(raw) if raw else ""
    if isinstance(value, dict):
        raise UsageError(f"giá trị {raw!r} không hỗ trợ; dùng số, chuỗi, true/false hoặc a,b,c")
    return value


def parse_set(pairs: Sequence[str]) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Tách các ``--set`` thành lựa chọn cho ô và tham số cho stage.

    Khoá không có dấu chấm là tên ô (``--set renderer=3dgeer``). Khoá có dấu chấm là
    ``<stage_id>.<tham số>`` (``--set train.drivestudio.num_iters=1000``); phần sau dấu chấm cuối
    là tên tham số.

    Args:
        pairs: Các chuỗi ``<khoá>=<giá trị>`` theo thứ tự trên dòng lệnh; khoá trùng thì giá trị
            sau đè giá trị trước.

    Returns:
        Bộ ``(tên ô → giá trị, stage_id → {tham số → giá trị})``.

    Raises:
        UsageError: Nếu một chuỗi không có dấu ``=`` hoặc khoá rỗng.
    """
    options: dict[str, object] = {}
    stage_params: dict[str, dict[str, object]] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        key = key.strip()
        if not separator or not key:
            raise UsageError(f"--set {pair!r}: cần dạng <khoá>=<giá trị>")
        value = parse_value(raw.strip())
        if "." in key:
            stage_id, _, param = key.rpartition(".")
            stage_params.setdefault(stage_id, {})[param] = value
        else:
            options[key] = value
    return options, stage_params
