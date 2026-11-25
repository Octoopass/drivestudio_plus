"""Tính chỉ số khung và chỉ số ảnh train/test của DriveStudio, kể cả camera chỉ test.

File này chứa:
- ``split_indices``: hàm thuần Python (không cần PyTorch) tách các khung thành train và test
  theo ``test_image_stride`` và đổi sang chỉ số ảnh ``khung * số_camera + camera``.

``SceneDataset.split_train_test`` (``drive2sim/recon/data/driving_dataset.py``) gọi hàm này khi
nạp dataset. Tách ra file riêng để test được ở môi trường nhẹ, không GPU. Quy tắc khung test khớp
``every_nth_frame_pairs`` của ``drive2sim/core/split.py`` (CONTRACTS mục 3.1); camera chỉ test là
``Split.test_only_cameras`` đã được ``train.drivestudio`` đổi sang vị trí trong
``data.pixel_source.cameras``.
"""

from collections.abc import Sequence


def split_indices(
    num_timesteps: int,
    num_cams: int,
    stride: int,
    test_only_cams: Sequence[int] = (),
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Chia các khung và các ảnh của dataset thành train và test.

    Khung test là ``stride, 2 * stride, ...`` (không có khung đầu, vì khung đầu cần cho train);
    ``stride == 0`` nghĩa là không có khung test. Cách chia này giống ``split_train_test`` gốc
    của DriveStudio. Camera chỉ test không có ảnh nào trong ``train_indices``: ở khung train
    ảnh của nó bị bỏ, ở khung test ảnh của nó vẫn nằm trong ``test_indices``.

    Args:
        num_timesteps: Số khung của dataset.
        num_cams: Số camera, tức độ dài ``data.pixel_source.cameras``.
        stride: ``test_image_stride``; ``0`` là không có khung test.
        test_only_cams: Vị trí (trong ``data.pixel_source.cameras``, không phải chỉ số camera của
            manifest) của các camera chỉ test; mặc định rỗng.

    Returns:
        Bộ ``(train_timesteps, test_timesteps, train_indices, test_indices)``. Hai danh sách đầu
        là chỉ số khung; hai danh sách sau là chỉ số ảnh ``khung * num_cams + camera`` theo thứ
        tự khung rồi camera.
    """
    test_timesteps = list(range(stride, num_timesteps, stride)) if stride != 0 else []
    test_set = set(test_timesteps)
    train_timesteps = [t for t in range(num_timesteps) if t not in test_set]
    test_only = set(test_only_cams)
    train_indices = [
        t * num_cams + cam
        for t in train_timesteps
        for cam in range(num_cams)
        if cam not in test_only
    ]
    test_indices = [t * num_cams + cam for t in test_timesteps for cam in range(num_cams)]
    return train_timesteps, test_timesteps, train_indices, test_indices
