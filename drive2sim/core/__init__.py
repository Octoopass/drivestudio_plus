"""Package ``drive2sim.core``: phần nền dùng chung, không nạp model, không cần GPU.

Gồm ``identifiers`` (định dạng và tạo định danh), ``schema`` (schema của mọi file trong
CONTRACTS), ``files`` (đọc ghi file qua schema) và ``split`` (tạo split train/test). Mọi package
khác của tool dựa vào đây (ARCHITECTURE mục 4.1).
"""
