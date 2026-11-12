"""Package ``drive2sim``: dựng lại cảnh lái xe thành cảnh 3D Gaussian Splatting để mô phỏng.

Các package con:
- ``recon``: mã tái dựng lấy từ DriveStudio (tiền xử lý, loader, model, trainer, render).
- ``core``: định danh, schema của mọi file, đọc ghi file, tạo split.
- ``engine``: đăng ký stage, quyết định danh sách stage của approach, cache, chạy run.
- ``stages``, ``ingest``, ``kaggle``, ``sim``, ``studio``, ``cli``, ``external``: các phần còn
  lại của tool, xem ARCHITECTURE mục 3.

Thiết kế ở ``docs/drive2sim/`` (SPEC, ARCHITECTURE, CONTRACTS); quy tắc viết code ở ``AGENTS.md``.
"""
