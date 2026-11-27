"""Đọc file pickle của bên thứ ba mà không chạy mã lạ: chỉ cho phép một danh sách kiểu cố định.

Pickle có thể gọi bất kỳ hàm Python nào lúc nạp, nên nạp file tải từ Internet bằng ``pickle.load``,
``joblib.load`` hay ``torch.load`` (``weights_only=False``) là chạy mã của người khác. File
``humanpose/smpl.pkl`` mà DriveStudio phát hành (``docs/datasets/human_pose.md``) là pickle của
dict chứa tensor PyTorch; khi liệt kê bằng ``pickletools`` (không thực thi), nó chỉ tham chiếu ba
kiểu: ``collections.OrderedDict``, ``torch._utils._rebuild_tensor_v2`` và
``torch.storage._load_from_bytes``. Kiểu cuối tự gọi ``torch.load`` không giới hạn, nên ở đây được
thay bằng ``torch.load(..., weights_only=True)`` (chế độ chỉ nạp tensor của PyTorch). Mọi kiểu khác
bị từ chối.

Metadata ROSA của KITTI-360 (``ImageSets-rosa-6class/*.pkl``, box 3D) là pickle của dict, list
và mảng numpy; liệt kê bằng ``pickletools`` chỉ thấy ``numpy.ndarray``, ``numpy.dtype`` và
``numpy._core.multiarray._reconstruct`` (NumPy 2 ghi ``numpy._core``, NumPy 1 là ``numpy.core``).
``load_numpy_pickle`` chỉ cho các kiểu đó cùng ``scalar``.

File này chứa ``UnsafePickleError``, ``load_tensor_pickle``, ``load_numpy_pickle``.
"""

import importlib
import io
import pickle
from collections import OrderedDict
from typing import Any, BinaryIO


class UnsafePickleError(pickle.UnpicklingError):
    """File pickle tham chiếu kiểu ngoài danh sách cho phép."""


def _load_storage(data: bytes) -> Any:  # noqa: ANN401 (tensor hoặc storage của PyTorch)
    import torch

    return torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")


class _TensorUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 (giao diện của pickle)
        import torch._utils

        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): _load_storage,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise UnsafePickleError(f"pickle tham chiếu {module}.{name}, không cho phép") from None


def load_tensor_pickle(data: bytes) -> Any:  # noqa: ANN401 (nội dung tuỳ file)
    """Nạp pickle chỉ gồm dict, list, số, chuỗi và tensor PyTorch.

    Raises:
        UnsafePickleError: Nếu file tham chiếu kiểu nào khác.
    """
    return _TensorUnpickler(io.BytesIO(data)).load()


_NUMPY_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class _NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 (giao diện của pickle)
        if (module, name) not in _NUMPY_GLOBALS:
            raise UnsafePickleError(f"pickle tham chiếu {module}.{name}, không cho phép")
        try:
            return getattr(importlib.import_module(module), name)
        except ImportError:  # NumPy 1 không có numpy._core
            legacy = module.replace("numpy._core", "numpy.core")
            return getattr(importlib.import_module(legacy), name)


def load_numpy_pickle(stream: BinaryIO) -> Any:  # noqa: ANN401 (nội dung tuỳ file)
    """Nạp pickle chỉ gồm dict, list, số, chuỗi và mảng numpy.

    Raises:
        UnsafePickleError: Nếu file tham chiếu kiểu nào khác.
        pickle.UnpicklingError: Nếu file không phải pickle.
    """
    return _NumpyUnpickler(stream).load()
