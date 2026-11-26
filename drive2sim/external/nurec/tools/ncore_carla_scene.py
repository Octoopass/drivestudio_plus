"""(drive2sim) Clip NCore → file cảnh ``.usdz`` cho phần NuRec của CARLA 1.0.

    <python môi trường NuRec> ncore_carla_scene.py clip.json --out scene.usdz [--scene-id ID]
        [--cameras a,b] [--rig-height 1.65] [--lane-width 3.5]

CARLA 1.0 (``PythonAPI/examples/nvidia/nurec``) đọc cảnh từ một file zip đuôi ``.usdz`` gồm:

- ``map.xodr``: bản đồ OpenDRIVE, CARLA dựng thế giới vật lý từ đây
  (``client.generate_opendrive_world``);
- ``rig_trajectories.json``: quỹ đạo rig của xe ego và hiệu chỉnh từng camera
  (``T_sensor_rig``, mô hình F-theta);
- ``sequence_tracks.json``: quỹ đạo xe, người khác (ở đây để trống: cảnh dựng tĩnh);
- ``data_info.json``: ``sequence_id`` và khoảng thời gian của cảnh.

Ảnh camera do server gRPC render (``drive2sim.sim.server.sensorsim``); server đọc thêm
``drive2sim_cameras.json`` (thông số ống kính đúng như lúc dựng: pinhole hoặc Kannala-Brandt khớp từ
F-theta, giống ``ncore_eval_export.py``). CARLA bỏ qua file lạ trong zip.

Hệ toạ độ: world của file cảnh là world của clip NCore (cũng là hệ của PLY Instant NuRec), không có
``geoReference`` nên CARLA đặt bản đồ và quỹ đạo cùng một hệ. Rig của KITTI-360 nằm ở camera
``image_00`` (cao khoảng 1,65 m); CARLA đặt xe ego ở gốc rig nên file cảnh dời rig xuống mặt đường
(``--rig-height``), camera vẫn đúng chỗ vì ``T_sensor_rig`` dời ngược lại.

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
    road = resample_path(ground_rig_world[:, :3, 3], a.step)
    xodr = opendrive(road, scene_id, a.lane_width, a.shoulder)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("map.xodr", xodr)
        z.writestr("rig_trajectories.json", json.dumps(rig_trajectories))
        z.writestr("sequence_tracks.json", json.dumps(empty_tracks()))
        z.writestr("data_info.json", json.dumps(data_info, indent=2))
        z.writestr(CAMERAS_FILE, json.dumps({"scene_id": scene_id, "cameras": exact}, indent=2))
    length = float(np.linalg.norm(np.diff(road[:, :2], axis=0), axis=1).sum())
    print(f"wrote {out}: {len(calibrations)} camera, {len(rig_ts)} pose, đường {length:.0f} m")


if __name__ == "__main__":
    main()
