"""SMPL của một người tại keyframe box (``provider.smpl_for_keyframes``).

Cần venv XSIM (``toast``, ``xsim``): máy khác bỏ qua. Lỗi đã gặp trên Kaggle: gọi
``toast.sample_trajectory`` với query hai chiều (24 khớp, K khung) làm pytoast báo IndexError ở
``out_quats[invalid_mask, 0] = 1``.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("toast")
pytest.importorskip("xsim")

from drive2sim.external.xsim.provider import smpl_for_keyframes  # noqa: E402


def rotations_about_z(angles: np.ndarray) -> np.ndarray:
    """Ma trận quay quanh trục z, (F, 24, 3, 3), mọi khớp cùng góc của khung."""
    c, s = np.cos(angles), np.sin(angles)
    mtx = np.zeros((len(angles), 3, 3), dtype=np.float32)
    mtx[:, 0, 0], mtx[:, 0, 1], mtx[:, 1, 0], mtx[:, 1, 1], mtx[:, 2, 2] = c, -s, s, c, 1
    return np.repeat(mtx[:, None], 24, axis=1)


def test_keyframes_with_gaps_keep_valid_poses_and_fill_the_rest() -> None:
    frames = 10
    rotations = rotations_about_z(np.linspace(0.0, 0.9, frames))
    betas = np.arange(frames * 10, dtype=np.float32).reshape(frames, 10)
    valid = np.array([False, True, True, False, False, True, True, True, False, False])
    keyframes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    data = smpl_for_keyframes(rotations, betas, valid, keyframes)
    assert data is not None
    assert tuple(data.joint_q.shape) == (frames, 24, 4)
    assert torch.isfinite(data.joint_q).all()
    # Khung có SMPL giữ nguyên tư thế; betas lấy từ khung có SMPL đầu tiên.
    import toast

    with toast.use_backend("torch"):
        expected = toast.Quat.from_matrix(torch.from_numpy(rotations)).std().data
    kept = torch.from_numpy(valid)
    assert torch.allclose(data.joint_q[kept], expected[kept], atol=1e-5)
    assert torch.equal(data.betas, torch.from_numpy(betas[1]))


def test_person_without_any_smpl_keyframe_returns_none() -> None:
    rotations = rotations_about_z(np.zeros(4))
    data = smpl_for_keyframes(
        rotations, np.zeros((4, 10), np.float32), np.zeros(4, bool), torch.tensor([0, 2])
    )
    assert data is None
