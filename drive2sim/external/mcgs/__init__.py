"""MCGS-SLAM chạy như một codebase ngoài của drive2sim (ARCHITECTURE mục 2.19).

MCGS-SLAM (Cao và cộng sự, ICRA 2026, giấy phép MIT) là SLAM đa camera dùng 3D Gaussian Splatting:
bundle adjustment nhiều camera ước lượng pose và độ sâu, rồi dựng bản đồ Gaussian. Nó chạy trong
conda env riêng (PyTorch 2.1.2, CUDA 11.8, mã nguồn chốt revision) qua tiến trình con; mã của nó
không nằm trong package này.

Nơi cài: biến ``D2S_MCGS_DIR``, mặc định ``~/.cache/drive2sim/mcgs-slam``; cài bằng
``bash envs/install_mcgs.sh``.

File này chứa:
- ``REVISION``, ``REPOSITORY``: commit chốt của repo (``envs/install_mcgs.sh`` dùng cùng giá trị).
- ``mcgs_dir``, ``mcgs_python``: đường dẫn bản cài.
- ``OUTPUT_TRAJECTORY``, ``OUTPUT_GAUSSIANS``, ``OUTPUT_METRICS``: tên file ``demo.py`` ghi ra.
- ``POSE_SCALE``: hệ số co mà bản đồ Gaussian dùng so với quỹ đạo.
- ``PINHOLE_MODELS``: mô hình camera mà SLAM nhận.

Căn quỹ đạo về hệ của cảnh và đo ATE ở ``align.py`` (cần numpy, chỉ nạp khi stage chạy).
"""

import os
from pathlib import Path

MCGS_DIR_ENV = "D2S_MCGS_DIR"
REPOSITORY = "https://github.com/mcgs-slam/mcgs-slam"
REVISION = "147eb3ac66684bed347cf3412b3e393d7c6c8171"
# demo.py: mcgs.save_kf_poses (TUM: timestamp tx ty tz qx qy qz qw, pose camera 0 → world, chỉ
# keyframe), gs.finalize (PLY 3DGS), eval_rendering_kf (PSNR, SSIM, LPIPS trên keyframe đã train).
OUTPUT_TRAJECTORY = "traj_mcgs.txt"
OUTPUT_GAUSSIANS = "3dgs_final.ply"
OUTPUT_METRICS = "psnr/after_opt/final_result_kf.json"
# Mcgs._scale_poses: poses và độ sâu đưa vào bản đồ Gaussian được nhân 0,2 (mcgs.py), trong khi
# traj_mcgs.txt giữ pose chưa co. Hệ của 3dgs_final.ply = hệ quỹ đạo × POSE_SCALE.
POSE_SCALE = 0.2
# MCGS-SLAM chỉ nhận camera pinhole (không có fisheye); đặt ở đây, không phải ``views.py``, để
# stage kiểm tra camera được mà không cần nạp numpy.
PINHOLE_MODELS = ("opencv_pinhole", "ideal_pinhole")


def mcgs_dir() -> Path:
    """Thư mục mã nguồn MCGS-SLAM (``D2S_MCGS_DIR`` hoặc ``~/.cache/drive2sim/mcgs-slam``)."""
    default = Path.home() / ".cache" / "drive2sim" / "mcgs-slam"
    return Path(os.environ.get(MCGS_DIR_ENV, default)).expanduser()


def mcgs_python() -> Path:
    """Python của conda env MCGS-SLAM (``<thư mục>/env/bin/python``)."""
    return mcgs_dir() / "env" / "bin" / "python"
