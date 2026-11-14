"""Dữ liệu mẫu dùng chung cho nhiều file test.

File này chứa manifest và split hợp lệ của một cảnh nuScenes giả, viết dưới dạng dict giống hệt
nội dung file ``manifest.yaml`` và ``splits/<split_id>.json`` (CONTRACTS mục 2.2, 3.1). Test nào
cần một cảnh hợp lệ thì import từ đây rồi chép và sửa phần cần thử, thay vì tự viết lại cả
manifest, để khi schema đổi chỉ phải sửa một chỗ.

Cuối file có tham số Mei thật của camera mắt cá ``image_02`` của KITTI-360, dùng chung cho test của
phép khớp Mei → OpenCV fisheye và của converter KITTI-360.
"""

MANIFEST = {
    "schema_version": "d2s-manifest/1.0",
    "scene_id": "nuscenes_scene-0061",
    "source": {
        "dataset": "nuscenes",
        "sequence_id": "scene-0061",
        "converter": "drivestudio.nuscenes",
        "converter_version": "0.1.0+test",
    },
    "track": "pinhole",
    "cameras": [
        {
            "camera_id": "CAM_FRONT",
            "model": "opencv_pinhole",
            "resolution": [1600, 900],
            "shutter": "GLOBAL",
        },
        {
            "camera_id": "CAM_FISH",
            "model": "opencv_fisheye",
            "resolution": [1600, 900],
            "shutter": "GLOBAL",
        },
    ],
    "lidars": ["LIDAR_TOP"],
    "timeline": {"reference_sensor": "CAM_FRONT", "num_frames": 10},
    "available": {"lidar": True, "cuboids": "gt"},
}

SPLIT = {
    "schema_version": "d2s-split/0.1",
    "split_id": "every8_front1",
    "scene_id": "nuscenes_scene-0061",
    "strategy": "every_nth_frame:8",
    "frame_range": [0, 9],
    "cameras": ["CAM_FRONT"],
    "train": [[0, "CAM_FRONT"]],
    "test": [[8, "CAM_FRONT"]],
}

# Tham số Mei của image_02 trong calib thật của KITTI-360 (calibration/image_02.yaml).
MEI_02 = {"xi": 2.2134047507854890, "k1": 1.6798235660113681e-02, "k2": 1.6548773243373522}
GAMMA_02 = (1.3363220825849971e03, 1.3357883350012958e03)
# Hệ số numpy.linalg.lstsq cho MEI_02 (cùng cách của teammate, chạy trong môi trường có numpy).
NUMPY_COEFFS_02 = [
    -0.009477485565102724,
    0.009264348574303914,
    -0.001678306810050497,
    -0.00045523033695660204,
]
