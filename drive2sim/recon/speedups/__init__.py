"""Thuật toán phụ trợ gắn vào trainer DriveStudio, bật/tắt bằng config (docs/methods/).

Mỗi thuật toán một module: ``fastgs`` (densify và prune theo nhiều góc nhìn). Module này không
import gì để ``fastgs_rules`` dùng được ở nơi không có PyTorch (test trên CPU).
"""
