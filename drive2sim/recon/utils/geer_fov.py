"""Sửa cận góc nhìn của 3DGEER cho camera pinhole có điểm chính lệch tâm, không sửa gsplat-geer.

``gsplat.geer.camera.get_camera_tanfov`` (submodule ``third_party/gsplat-geer`` @f6eeee9) trả
cận đối xứng ``W/(2fx), H/(2fy)`` cho camera pinhole không méo. ``computePBF`` trong CUDA kẹp mọi
Gaussian vào ``[-tanfov, tanfov]``, nên khi điểm chính lệch tâm thì phía xa điểm chính không bao
giờ được vẽ. KITTI-360 ``image_00`` (``cx = 682,0`` trên 1408, ``cy = 238,8`` trên 376) mất ~25
hàng trên và ~18 cột phải ở nửa độ phân giải: run #6 tuần 04 (``393f``, OmniRe 3DGEER) chỉ được
19,8 dB trên ``image_00`` so với 25,0 dB của 3DGUT, và ngang 3DGUT khi bỏ hai dải đó.

Repo gsplat-geer không sửa được từ dự án, nên drive2sim thay hàm mà ``gsplat.geer._preprocess``
gọi (tra theo tên lúc gọi) bằng bản lấy khoảng cách tới mép ảnh xa điểm chính nhất. Điểm chính ở
giữa ảnh cho đúng giá trị cũ; camera méo, fisheye, FTheta vẫn đi hàm gốc. Chỉ đổi Python, không
phải build lại CUDA, nên áp dụng được cho wheel đã build sẵn (Kaggle, máy thuê).

File này chứa:
- ``pinhole_tanfov``: cận ``(tanfovx, tanfovy)`` phủ cả ảnh pinhole.
- ``install``: thay hàm trong ``gsplat.geer._preprocess`` (gọi nhiều lần không sao).

``recon/trainers/base.py`` gọi ``install`` trong ``_init_renderer`` khi gsplat có ``with_geer``.
"""

import functools
from typing import Callable

import numpy as np

PATCHED_FLAG = "_drive2sim_offcentre_tanfov"  # thuộc tính đánh dấu hàm đã thay


def pinhole_tanfov(
    K: object, width: float, height: float, fov_mod: float = 1.0
) -> tuple[float, float]:
    """Cận tan góc nhìn đối xứng phủ cả ảnh pinhole, kể cả khi điểm chính lệch tâm.

    Args:
        K: Ma trận nội tại ``(..., 3, 3)`` của đúng một camera (tensor torch hoặc mảng numpy).
        width: Chiều rộng ảnh (pixel).
        height: Chiều cao ảnh (pixel).
        fov_mod: Hệ số nhân tiêu cự như hàm gốc.

    Returns:
        ``(max(cx, W - cx) / fx, max(cy, H - cy) / fy)``.
    """
    if hasattr(K, "detach"):
        K = K.detach().cpu()
    k = np.asarray(K, dtype=np.float64).reshape(-1, 3, 3)[0]
    fx, fy = k[0, 0] * fov_mod, k[1, 1] * fov_mod
    cx, cy = k[0, 2], k[1, 2]
    return float(max(cx, width - cx) / fx), float(max(cy, height - cy) / fy)


def _wrap(original: Callable) -> Callable:
    @functools.wraps(original)
    def get_camera_tanfov(
        camera_model: str,
        Ks: object,
        width: float,
        height: float,
        fov_mod: float = 1,
        radial_coeffs: object = None,
        tangential_coeffs: object = None,
        thin_prism_coeffs: object = None,
        **kwargs: object,
    ) -> tuple:
        distorted = (
            radial_coeffs is not None
            or tangential_coeffs is not None
            or thin_prism_coeffs is not None
        )
        if camera_model == "pinhole" and not distorted:
            return pinhole_tanfov(Ks, width, height, fov_mod)
        return original(
            camera_model,
            Ks,
            width,
            height,
            fov_mod,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            **kwargs,
        )

    setattr(get_camera_tanfov, PATCHED_FLAG, True)
    return get_camera_tanfov


def install() -> bool:
    """Thay ``get_camera_tanfov`` mà ``gsplat.geer._preprocess`` gọi.

    Returns:
        ``True`` nếu hàm đang dùng là bản đã sửa; ``False`` nếu gsplat không có GEER.
    """
    try:
        from gsplat.geer import _preprocess
    except ImportError:
        return False
    current = getattr(_preprocess, "get_camera_tanfov", None)
    if current is None:
        return False
    if not getattr(current, PATCHED_FLAG, False):
        _preprocess.get_camera_tanfov = _wrap(current)
    return True
