"""tools/pickle_compat.py reads pickles that name NumPy 2's numpy._core modules."""
import importlib.util
import io
import pickle
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("pickle_compat", Path(__file__).resolve().parents[1] / "tools" / "pickle_compat.py")
pickle_compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pickle_compat)


def test_reads_numpy2_module_paths():
    obj = {"data_list": [{"lidar_path": "x/2013_05_28_drive_0005_sync/0.bin", "pose": np.arange(6, dtype=np.float32).reshape(2, 3)}]}
    raw = pickle.dumps(obj, protocol=0)                    # text protocol: module paths are plain lines
    if b"numpy._core" not in raw:                           # written by NumPy 1.x: make it look like NumPy 2
        raw = raw.replace(b"numpy.core", b"numpy._core")
    back = pickle_compat.load(io.BytesIO(raw))
    assert np.array_equal(back["data_list"][0]["pose"], obj["data_list"][0]["pose"])
    assert pickle_compat.load(io.BytesIO(pickle.dumps(obj)))["data_list"][0]["lidar_path"].endswith("0.bin")
