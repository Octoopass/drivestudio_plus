"""Chiếu box người của cảnh PHENIKAA lên ảnh, cho bước dựng SMPL (``smpl/run.py``).

Cảnh PHENIKAA (``drive2sim/ingest/phenikaa.py``) ghi theo khung thư mục KITTI của DriveStudio như
KITTI-360: cùng ``ego_pose/``, ``extrinsics/``, ``intrinsics/`` (``fx fy cx cy`` rồi hệ số méo),
``instances/`` và lớp người ``Pedestrian``. Nên dùng lại nguyên hàm ``project_human_boxes`` của
KITTI (``preprocess/kitti/human.py``).

Số camera và thứ tự camera khác nhau giữa các cảnh PHENIKAA, nên file này không có
``CAMERA_LIST`` cố định: stage ``prep.smpl`` truyền chỉ số các camera pinhole qua ``--cameras``.
Như KITTI-360, chỉ dùng camera pinhole: phép chiếu box dùng ma trận pinhole, và 4D-Humans được
huấn luyện trên ảnh thường. Tư thế SMPL nằm trong hệ toạ độ thế giới nên camera mắt cá vẫn vẽ
được người khi train; người chỉ hiện ở camera mắt cá thì không có SMPL và được dựng bằng node
biến dạng.
"""

from drive2sim.recon.preprocess.kitti.human import project_human_boxes  # noqa: F401

CAMERA_LIST = None  # bắt buộc truyền --cameras
