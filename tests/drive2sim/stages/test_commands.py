"""Kiểm tra phân biệt lỗi import cuối lệnh với lỗi đã được xử lý trước đó."""

from drive2sim.stages.commands import find_import_error


def test_only_terminal_import_error_changes_error_kind() -> None:
    assert find_import_error("Traceback:\nModuleNotFoundError: No module named 'numpy'\n")
    assert find_import_error("ImportError: optional backend\nValueError: bad input") is None
    assert find_import_error("") is None
