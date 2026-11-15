"""Chiếu box người của cảnh KITTI-360 lên ảnh, cho bước dựng SMPL (``smpl/run.py``).

Cảnh KITTI-360 (``drive2sim/ingest/kitti360.py``) ghi theo khung thư mục KITTI của DriveStudio,
nên dùng lại nguyên hàm ``project_human_boxes`` của KITTI (``preprocess/kitti/human.py``): cùng
``ego_pose/``, ``extrinsics/``, ``intrinsics/``, ``instances/``, cùng lớp người (Pedestrian).

Chỉ chạy trên hai camera pinhole (``image_00``, ``image_01``, chỉ số 0, 1), như nhánh KITTI:
phép chiếu box dùng ma trận pinhole, và 4D-Humans được huấn luyện trên ảnh thường. Tư thế SMPL
nằm trong hệ toạ độ thế giới nên camera mắt cá (2, 3) vẫn vẽ được người khi train; người chỉ hiện
ở camera mắt cá (hai bên hông) thì không có SMPL và được dựng bằng node biến dạng.
"""

from drive2sim.recon.preprocess.kitti.human import project_human_boxes  # noqa: F401

CAMERA_LIST = [0, 1]
