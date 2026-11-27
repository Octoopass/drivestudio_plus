"""(drive2sim) Clip NCore → file cảnh ``.usdz`` cho phần NuRec của CARLA 1.0.

    <python môi trường NuRec> ncore_carla_scene.py clip.json --out scene.usdz [--scene-id ID]
        [--cameras a,b] [--rig-height 1.65] [--lane-width 3.5] [--scene-root DIR]

CARLA 1.0 (``PythonAPI/examples/nvidia/nurec``) đọc cảnh từ một file zip đuôi ``.usdz`` gồm:

- ``map.xodr``: bản đồ OpenDRIVE, CARLA dựng thế giới vật lý từ đây
  (``client.generate_opendrive_world``);
- ``rig_trajectories.json``: quỹ đạo rig của xe ego và hiệu chỉnh từng camera
  (``T_sensor_rig``, mô hình F-theta);
- ``sequence_tracks.json``: quỹ đạo xe, người khác; CARLA tạo một actor (xe hoặc người có kích
  thước gần nhất, có vật lý và va chạm) cho mỗi track nhãn xe hoặc ``person``;
- ``data_info.json``: ``sequence_id`` và khoảng thời gian của cảnh.

Ảnh camera do server gRPC render (``drive2sim.sim.server.sensorsim``); server đọc thêm
``drive2sim_cameras.json`` (thông số ống kính đúng như lúc dựng: pinhole hoặc Kannala-Brandt khớp từ
F-theta, giống ``ncore_eval_export.py``). CARLA bỏ qua file lạ trong zip.

Hệ toạ độ: world của file cảnh là world của clip NCore (cũng là hệ của PLY Instant NuRec), không có
``geoReference`` nên CARLA đặt bản đồ và quỹ đạo cùng một hệ. Rig của KITTI-360 nằm ở camera
``image_00`` (cao khoảng 1,65 m); CARLA đặt xe ego ở gốc rig nên file cảnh dời rig xuống mặt đường
(``--rig-height``), camera vẫn đúng chỗ vì ``T_sensor_rig`` dời ngược lại.

Xe, người khác (``--scene-root``): lấy từ box 3D của cảnh drive2sim (``instances/``, cùng box
StreetGS dùng), mỗi vật thể một track. Box nằm trong world của cảnh drive2sim; đổi sang world của
clip bằng ``T = rig(f) · R_cam→FLU · (ego(f) · cam0→ego)⁻¹`` (pose camera ``image_00`` ở hai hệ;
không đổi theo khung, sai lệch cỡ 1 mm), lấy trung bình trên các khung có trong clip. Vật thể đứng
yên (tâm box dời dưới ``STATIC_MOVE_M``) có mặt từ ``STATIC_BEFORE_US`` trước box đầu tới
``STATIC_AFTER_US`` sau box cuối (vật gần cuối đường thì giữ mãi, cho lúc lái tay quá thời lượng
clip); vật thể chuyển động chỉ có mặt trong khoảng khung có box. Cảnh không có ``instances/`` hoặc thiếu
``clip.json`` của ``prep.ncore`` cạnh file clip thì để trống như trước. Ảnh Gaussian không đổi: xe,
người vẫn là phần "đóng băng" trong PLY; actor chỉ có trong thế giới vật lý và góc nhìn của CARLA.

Bản đồ: một con đường bám quỹ đạo xe (đoạn thẳng mỗi ~2 m, độ cao theo quỹ đạo), xe chạy giữa
làn phải, thêm một làn ngược chiều và lề hai bên. Chỉ gần đúng hình học đường thật; đủ để CARLA có
mặt đường cho vật lý và Traffic Manager.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from upath import UPath

from ncore.data.v4 import (
    CameraSensorComponent,
    IntrinsicsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)

CAMERAS_FILE = "drive2sim_cameras.json"
POLY_DEGREE = 5
CLIP_INFO = "clip.json"  # của prep.ncore: số khung gốc của clip và của đoạn split
# OpenCV camera (x phải, y xuống, z trước) -> rig FLU, như kitti360_to_ncore.R_CAM_TO_FLU
R_CAM_TO_FLU = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
# Nhãn box của drive2sim (instances_info.json) -> nhãn track CARLA (constants.VEHICLE_LABELS, "person")
LABELS = {"Car": "automobile", "Truck": "heavy_truck", "Bus": "bus", "Pedestrian": "person",
          "Cyclist": "rider"}
STATIC_MOVE_M = 1.0  # tâm box dời ít hơn chừng này trên cả track: coi là đứng yên
FOREVER_US = 24 * 3600 * 1_000_000  # vật đứng yên cuối đường còn đó cả khi lái tay quá clip
# Vật đứng yên chỉ có mặt quanh lúc xe đi ngang: từ STATIC_BEFORE_US trước box đầu tới
# STATIC_AFTER_US sau box cuối. Tạo hết từ đầu (hàng trăm actor một lúc) làm CARLA nghẽn và ngắt.
STATIC_BEFORE_US = 10_000_000
STATIC_AFTER_US = 20_000_000


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def angle_poly_from_radius(f_of_theta, max_angle: float) -> tuple[list[float], list[float]]:
    """Đa thức F-theta (góc → bán kính, bán kính → góc) bậc ``POLY_DEGREE`` khớp một hàm r(θ)."""
    theta = np.linspace(0.0, max_angle, 400)
    radius = f_of_theta(theta)

    def fit(x, y):  # đa thức không có hằng số (đi qua gốc): hệ số x^0..x^POLY_DEGREE
        a = np.stack([x**k for k in range(1, POLY_DEGREE + 1)], 1)
        coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
        return [0.0, *coeffs.tolist()]

    return fit(theta, radius), fit(radius, theta)


def kb_from_ftheta(params: FThetaCameraModelParameters) -> tuple[float, list[float]]:
    """Kannala-Brandt (f, k1..k4) khớp đa thức thuận của F-theta, như ncore_eval_export.ftheta_as_kb."""
    t = np.linspace(1e-3, float(params.max_angle), 2000)
    r = np.polynomial.polynomial.polyval(t, params.angle_to_pixeldist_poly.astype(np.float64))
    a = np.stack([t ** (2 * i) for i in range(5)], 1)
    sol, *_ = np.linalg.lstsq(a, r / t, rcond=None)
    return float(sol[0]), (sol[1:] / sol[0]).tolist()


def camera_entry(cam_id: str, params, resolution: list[int]) -> tuple[dict, dict]:
    """(``camera_model`` theo định dạng F-theta của CARLA, thông số ống kính đúng cho server)."""
    w, h = resolution
    if isinstance(params, FThetaCameraModelParameters):
        c, d, e = (float(x) for x in params.linear_cde)
        pp = [float(params.principal_point[0]), float(params.principal_point[1])]
        ftheta = {
            "principal_point": pp,
            "reference_poly": "PIXELDIST_TO_ANGLE"
            if str(params.reference_poly).endswith("PIXELDIST_TO_ANGLE")
            else "ANGLE_TO_PIXELDIST",
            "pixeldist_to_angle_poly": params.pixeldist_to_angle_poly.astype(float).tolist(),
            "angle_to_pixeldist_poly": params.angle_to_pixeldist_poly.astype(float).tolist(),
            "max_angle": float(params.max_angle),
            "linear_cde": [c, d, e],
        }
        f, radial = kb_from_ftheta(params)
        exact = {
            "model": "fisheye",
            # F-theta: tâm ảnh theo quy ước tâm pixel; gsplat theo góc pixel như NCore OpenCV
            "K": [[f * c, 0, pp[0] + 0.5], [0, f, pp[1] + 0.5], [0, 0, 1]],
            "radial": radial,
        }
    else:
        fx, fy = (float(v) for v in params.focal_length)
        cx, cy = (float(v) for v in params.principal_point)
        if isinstance(params, OpenCVFisheyeCameraModelParameters):
            radial = [float(v) for v in params.radial_coeffs[:4]]

            def radius(theta):
                t2 = theta * theta
                return fx * theta * (1 + radial[0] * t2 + radial[1] * t2**2 + radial[2] * t2**3 + radial[3] * t2**4)

            max_angle = float(getattr(params, "max_angle", 0) or math.pi / 2)
            model = "fisheye"
        else:
            radial = [0.0, 0.0, 0.0, 0.0]

            def radius(theta):
                return fx * np.tan(theta)

            corner = max(math.hypot(x - cx, y - cy) for x in (0, w) for y in (0, h))
            max_angle = math.atan(corner / fx)
            model = "pinhole"
        forward, backward = angle_poly_from_radius(radius, max_angle)
        ftheta = {
            "principal_point": [cx - 0.5, cy - 0.5],
            "reference_poly": "ANGLE_TO_PIXELDIST",
            "pixeldist_to_angle_poly": backward,
            "angle_to_pixeldist_poly": forward,
            "max_angle": max_angle,
            "linear_cde": [fy / fx if fx else 1.0, 0.0, 0.0],
        }
        exact = {"model": model, "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "radial": radial}
    calibration_model = {
        "type": "ftheta",
        "parameters": {
            "resolution": [int(w), int(h)],
            "shutter_type": "GLOBAL",
            "external_distortion_parameters": None,
            **ftheta,
        },
    }
    exact.update({"resolution": [int(w), int(h)], "logical_id": cam_id})
    return calibration_model, exact


def resample_path(points: np.ndarray, step: float) -> np.ndarray:
    """Điểm cách đều ``step`` mét theo chiều dài quỹ đạo (đã làm mượt), bỏ chỗ xe đứng yên."""
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])
    s, points = s[keep], points[keep]
    if len(points) < 2:
        raise ValueError("quỹ đạo xe quá ngắn để dựng bản đồ")
    targets = np.arange(0.0, s[-1], step)
    if s[-1] - targets[-1] > 0.25 * step:
        targets = np.append(targets, s[-1])
    out = np.stack([np.interp(targets, s, points[:, k]) for k in range(3)], 1)
    if len(out) >= 5:  # trung bình trượt 5 điểm cho đường cong mượt, giữ hai đầu
        kernel = np.ones(5) / 5
        for k in range(3):
            smooth = np.convolve(np.pad(out[:, k], 2, mode="edge"), kernel, mode="valid")
            out[2:-2, k] = smooth[2:-2]
    return out


def opendrive(points: np.ndarray, name: str, lane_width: float, shoulder: float) -> str:
    """Một con đường bám ``points`` (mặt đường), xe chạy giữa làn phải (làn -1)."""
    xy, z = points[:, :2], points[:, 2]
    d = np.diff(xy, axis=0)
    lengths = np.linalg.norm(d, axis=1)
    headings = np.arctan2(d[:, 1], d[:, 0])
    s = np.concatenate([[0.0], np.cumsum(lengths)])
    geometry, elevation = [], []
    for i, length in enumerate(lengths):
        geometry.append(
            f'      <geometry s="{s[i]:.4f}" x="{xy[i, 0]:.4f}" y="{xy[i, 1]:.4f}" '
            f'hdg="{headings[i]:.6f}" length="{length:.4f}"><line/></geometry>'
        )
        slope = (z[i + 1] - z[i]) / length
        elevation.append(f'      <elevation s="{s[i]:.4f}" a="{z[i]:.4f}" b="{slope:.6f}" c="0" d="0"/>')

    def lane(lane_id: int, kind: str, width: float) -> str:
        return (
            f'            <lane id="{lane_id}" type="{kind}" level="false"><link/>'
            f'<width sOffset="0" a="{width}" b="0" c="0" d="0"/>'
            f'<roadMark sOffset="0" type="{"solid" if kind != "driving" else "broken"}" '
            f'weight="standard" color="standard" width="0.12"/></lane>'
        )

    total = s[-1]
    return "\n".join([
        '<?xml version="1.0" standalone="yes"?>',
        "<OpenDRIVE>",
        f'  <header revMajor="1" revMinor="4" name="{escape(name)}" version="1.0" vendor="drive2sim"/>',
        f'  <road name="{escape(name)}" length="{total:.4f}" id="1" junction="-1">',
        "    <link/>",
        '    <type s="0" type="town"><speed max="50" unit="km/h"/></type>',
        "    <planView>",
        *geometry,
        "    </planView>",
        "    <elevationProfile>",
        *elevation,
        "    </elevationProfile>",
        "    <lateralProfile/>",
        "    <lanes>",
        # tâm đường lệch sang trái nửa làn: quỹ đạo xe nằm giữa làn phải
        f'      <laneOffset s="0" a="{lane_width / 2:.3f}" b="0" c="0" d="0"/>',
        '      <laneSection s="0">',
        "        <left>",
        lane(2, "shoulder", shoulder),
        lane(1, "driving", lane_width),
        "        </left>",
        '        <center><lane id="0" type="none" level="false"><link/>'
        '<roadMark sOffset="0" type="broken" weight="standard" color="standard" width="0.12"/></lane></center>',
        "        <right>",
        lane(-1, "driving", lane_width),
        lane(-2, "shoulder", shoulder),
        "        </right>",
        "      </laneSection>",
        "    </lanes>",
        "  </road>",
        "</OpenDRIVE>",
        "",
    ])


def orthonormal(rotation: np.ndarray) -> np.ndarray:
    """Ma trận quay gần nhất (SVD), bỏ sai số làm tròn của box."""
    u, _, vt = np.linalg.svd(rotation)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def quat_xyzw(r: np.ndarray) -> list[float]:
    """Ma trận quay 3x3 -> quaternion [qx, qy, qz, qw]."""
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return [float(v) for v in q]


def scene_to_clip_world(scene_root: Path, clip_info: dict, rig_world: np.ndarray) -> tuple[np.ndarray, dict]:
    """Phép đổi world của cảnh drive2sim -> world của clip và chỉ số pose rig của từng khung cảnh.

    Returns:
        ``(T, rig_index)``: ``T`` 4x4; ``rig_index`` = khung trong cảnh -> chỉ số trong ``rig_world``
        (chỉ các khung có trong clip).
    """
    position = {raw: i for i, raw in enumerate(clip_info["frame_ids"])}
    first = clip_info["scene_frames"][0]
    rig_index = {first + k: position[raw] for k, raw in enumerate(clip_info["scene_frame_ids"])
                 if raw in position}
    cam0_ego = np.loadtxt(scene_root / "extrinsics" / "0.txt")
    samples = []
    for frame, index in list(rig_index.items())[:: max(1, len(rig_index) // 50)]:
        ego = np.loadtxt(scene_root / "ego_pose" / f"{frame:03d}.txt")
        samples.append(rig_world[index] @ R_CAM_TO_FLU @ np.linalg.inv(ego @ cam0_ego))
    samples = np.asarray(samples)
    t = np.eye(4)
    t[:3, :3] = orthonormal(samples[:, :3, :3].mean(0))
    t[:3, 3] = samples[:, :3, 3].mean(0)
    spread = float(np.abs(samples[:, :3, 3] - t[:3, 3]).max())
    print(f"world cảnh -> world clip: lệch tối đa {spread * 1000:.1f} mm trên {len(samples)} khung")
    if spread > 0.5:
        raise SystemExit("pose của cảnh và clip không khớp (lệch hơn 0,5 m): không đặt được box")
    return t, rig_index


def scene_tracks(scene_root: Path, clip_info: dict, rig_world: np.ndarray, rig_ts: np.ndarray) -> dict:
    """``sequence_tracks.json`` từ ``instances/`` của cảnh (xem docstring của file)."""
    info_path = scene_root / "instances" / "instances_info.json"
    if not info_path.is_file():
        print("cảnh không có instances/: không có xe, người khác")
        return empty_tracks()
    t, rig_index = scene_to_clip_world(scene_root, clip_info, rig_world)
    data = empty_tracks()["dummy_chunk_id"]
    tracks, dims = data["tracks_data"], data["cuboidtracks_data"]["cuboids_dims"]
    counts: dict[str, int] = {}
    for key, item in json.loads(info_path.read_text()).items():
        label = LABELS.get(item.get("class_name"))
        notes = item.get("frame_annotations", {})
        if label is None:
            continue
        frames = [(f, m, s) for f, m, s in zip(notes["frame_idx"], notes["obj_to_world"], notes["box_size"])
                  if f in rig_index]
        if not frames:
            continue
        poses, stamps = [], []
        for frame, obj_to_world, _ in frames:
            m = t @ np.asarray(obj_to_world, dtype=np.float64)
            poses.append([*m[:3, 3].tolist(), *quat_xyzw(orthonormal(m[:3, :3]))])
            stamps.append(int(rig_ts[rig_index[frame]]))
        centres = np.asarray([p[:3] for p in poses])
        moving = float(np.linalg.norm(centres - centres[0], axis=1).max()) >= STATIC_MOVE_M
        if not moving:  # đứng yên: một pose, có mặt quanh lúc xe ego đi ngang
            end = stamps[-1] + STATIC_AFTER_US
            if end >= int(rig_ts[-1]):
                end = int(rig_ts[-1]) + FOREVER_US
            poses = [poses[0], poses[0]]
            stamps = [max(int(rig_ts[0]), stamps[0] - STATIC_BEFORE_US), end]
        elif len(poses) == 1:
            poses, stamps = poses * 2, [stamps[0], stamps[0] + 1]
        tracks["tracks_id"].append(f"{item.get('class_name', 'obj')}_{key}")
        tracks["tracks_poses"].append(poses)
        tracks["tracks_timestamps_us"].append(stamps)
        tracks["tracks_label_class"].append(label)
        tracks["tracks_flags"].append(["DYNAMIC"] if moving else [])
        dims.append(np.median(np.asarray([s for _, _, s in frames], dtype=np.float64), axis=0).tolist())
        name = f"{label}{' (chạy)' if moving else ''}"
        counts[name] = counts.get(name, 0) + 1
    print("track: " + (", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "không có"))
    return {"dummy_chunk_id": data}


def empty_tracks() -> dict:
    return {
        "dummy_chunk_id": {
            "tracks_data": {
                "tracks_id": [],
                "tracks_poses": [],
                "tracks_timestamps_us": [],
                "tracks_label_class": [],
                "tracks_flags": [],
            },
            "cuboidtracks_data": {"cuboids_dims": []},
        }
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("json", help="file JSON của clip NCore (prep.ncore)")
    p.add_argument("--out", required=True, help="file .usdz cần ghi")
    p.add_argument("--scene-id", default="", help="sequence_id; mặc định tên file .usdz")
    p.add_argument("--cameras", default="", help="camera cần đưa vào, cách nhau dấu phẩy (mặc định: tất cả)")
    p.add_argument("--rig-height", type=float, default=1.65, help="độ cao gốc rig so với mặt đường (m)")
    p.add_argument("--lane-width", type=float, default=3.5)
    p.add_argument("--shoulder", type=float, default=1.5)
    p.add_argument("--step", type=float, default=2.0, help="khoảng cách điểm của đường trên bản đồ (m)")
    p.add_argument("--scene-root", default="", help="thư mục cảnh drive2sim: xe, người từ instances/")
    a = p.parse_args()

    out = Path(a.out)
    scene_id = a.scene_id or out.stem
    meta = json.loads(Path(a.json).read_text())
    base = UPath(a.json).parent
    reader = SequenceComponentGroupsReader([base / s["path"] for s in meta["component_stores"]])
    poses = reader.open_component_readers(PosesComponent.Reader)["default"]
    intrinsics = reader.open_component_readers(IntrinsicsComponent.Reader)["default"]
    rig_world, rig_ts = poses.get_dynamic_pose("rig", "world")
    rig_world = np.asarray(rig_world, dtype=np.float64)
    rig_ts = np.asarray(rig_ts, dtype=np.int64)

    # rig trên mặt đường: dời xuống theo trục z của rig
    down = translate(0.0, 0.0, -a.rig_height)
    up = translate(0.0, 0.0, a.rig_height)
    ground_rig_world = rig_world @ down

    wanted = [c for c in a.cameras.split(",") if c]
    calibrations, exact = {}, {}
    for index, (cam_id, cam) in enumerate(reader.open_component_readers(CameraSensorComponent.Reader).items()):
        if wanted and cam_id not in wanted:
            continue
        params = intrinsics.get_camera_model_parameters(cam_id)
        if not isinstance(params, (FThetaCameraModelParameters, OpenCVFisheyeCameraModelParameters,
                                   OpenCVPinholeCameraModelParameters)):
            print(f"bỏ {cam_id}: mô hình {type(params).__name__} chưa hỗ trợ")
            continue
        resolution = [int(v) for v in params.resolution]
        camera_model, lens = camera_entry(cam_id, params, resolution)
        t_cam_rig = up @ np.asarray(poses.get_static_pose(cam_id, "rig"), dtype=np.float64)
        calibrations[cam_id] = {
            "sequence_id": scene_id,
            "logical_sensor_name": cam_id,
            "unique_sensor_idx": index,
            "T_sensor_rig": t_cam_rig.tolist(),
            "camera_model": camera_model,
        }
        exact[cam_id] = {**lens, "T_sensor_rig": t_cam_rig.tolist()}
        print(f"{cam_id}: {lens['model']} {resolution[0]}x{resolution[1]}")
    if not calibrations:
        raise SystemExit("clip không có camera nào dùng được")

    rig_trajectories = {
        "T_world_base": np.eye(4).tolist(),
        "rig_trajectories": [{
            "sequence_id": scene_id,
            "T_rig_worlds": ground_rig_world.tolist(),
            "T_rig_world_timestamps_us": rig_ts.tolist(),
        }],
        "camera_calibrations": calibrations,
    }
    data_info = {
        "sequence_id": scene_id,
        "pose-range": {"start-timestamp_us": int(rig_ts[0]), "end-timestamp_us": int(rig_ts[-1])},
        "producer": "drive2sim ncore_carla_scene.py",
    }
    clip_info_path = Path(a.json).parent / CLIP_INFO
    if a.scene_root and clip_info_path.is_file():
        tracks = scene_tracks(Path(a.scene_root), json.loads(clip_info_path.read_text()), rig_world, rig_ts)
    else:
        tracks = empty_tracks()
    road = resample_path(ground_rig_world[:, :3, 3], a.step)
    xodr = opendrive(road, scene_id, a.lane_width, a.shoulder)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("map.xodr", xodr)
        z.writestr("rig_trajectories.json", json.dumps(rig_trajectories))
        z.writestr("sequence_tracks.json", json.dumps(tracks))
        z.writestr("data_info.json", json.dumps(data_info, indent=2))
        z.writestr(CAMERAS_FILE, json.dumps({"scene_id": scene_id, "cameras": exact}, indent=2))
    length = float(np.linalg.norm(np.diff(road[:, :2], axis=0), axis=1).sum())
    print(f"wrote {out}: {len(calibrations)} camera, {len(rig_ts)} pose, đường {length:.0f} m")


if __name__ == "__main__":
    main()
