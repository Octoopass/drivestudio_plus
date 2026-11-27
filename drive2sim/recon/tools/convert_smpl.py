"""drive2sim: lưu lại mô hình SMPL ``.pkl`` thành file ``torch.save`` mà XSIM đọc.

XSIM đọc mô hình SMPL từ biến ``SMPL_PATH`` dưới dạng dict các mảng đã lưu bằng ``torch.save``.
File ``.pkl`` chính thức chứa đối tượng chumpy, mà venv XSIM không có chumpy, nên bước chuyển đổi
chạy ở môi trường chính. Kết quả đặt ngoài ``cache/`` và ``runs/``: file hơn 100 MB và chứa trọng
số SMPL có bản quyền, nên không nằm trong artifact hay output notebook.

Cách dùng::

    python -m drive2sim.recon.tools.convert_smpl --source SMPL_NEUTRAL.pkl --target out.pth
"""

import argparse
import inspect
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")


def convert_smpl_model(source: Path, target: Path) -> None:
    """Đọc ``source`` (pickle chính thức) rồi ghi ``target`` đủ file hoặc không ghi gì."""
    # chumpy gọi inspect.getargspec, hàm Python 3.11 đã bỏ.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    with open(source, "rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    # J_regressor là ma trận thưa scipy (có toarray), giữ nguyên; còn lại đổi sang numpy.
    model = {k: raw[k] if hasattr(raw[k], "toarray") else np.asarray(raw[k]) for k in KEYS}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(handle)
    try:
        torch.save(model, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    convert_smpl_model(args.source, args.target)


if __name__ == "__main__":
    main()
