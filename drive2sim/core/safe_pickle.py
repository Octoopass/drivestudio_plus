"""Đọc file pickle của bên thứ ba mà không chạy mã lạ: chỉ cho phép một danh sách kiểu cố định.

Pickle có thể gọi bất kỳ hàm Python nào lúc nạp, nên nạp file tải từ Internet bằng ``pickle.load``,
``joblib.load`` hay ``torch.load`` (``weights_only=False``) là chạy mã của người khác. File
``humanpose/smpl.pkl`` mà DriveStudio phát hành (``docs/datasets/human_pose.md``) là pickle của
dict chứa tensor PyTorch; khi liệt kê bằng ``pickletools`` (không thực thi), nó chỉ tham chiếu ba
kiểu: ``collections.OrderedDict``, ``torch._utils._rebuild_tensor_v2`` và
``torch.storage._load_from_bytes``. Kiểu cuối tự gọi ``torch.load`` không giới hạn, nên ở đây được
thay bằng ``torch.load(..., weights_only=True)`` (chế độ chỉ nạp tensor của PyTorch). Mọi kiểu khác
bị từ chối.

File này chứa ``UnsafePickleError``, ``load_tensor_pickle``.
"""

import io
import pickle
from collections import OrderedDict
from typing import Any


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
