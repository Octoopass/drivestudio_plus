"""Server gRPC ``SensorsimService`` (giao thức renderer NuRec của NVIDIA) render cảnh drive2sim.

    python -m drive2sim.sim.server.sensorsim --usdz scene.usdz --ply scene.ply [--port 46435]
        [--host 127.0.0.1] [--carla-root ~/opt/carla/Carla-1.0-Linux-Shipping] [--gpu 0]

Thay cho container NRE (``nvcr.io/nvidia/nre/nre-ga``) mà ví dụ NuRec của CARLA 1.0 tự bật: CARLA
giữ map, xe và đồng hồ; mỗi tick nó gửi pose camera (``sensor_pose``: camera → world, trục OpenCV)
và thông số ống kính, server render PLY Gaussian bằng gsplat rồi trả ảnh.

- ``scene.usdz``: file cảnh do ``external/nurec/tools/ncore_carla_scene.py`` ghi; server đọc
  ``drive2sim_cameras.json`` (ống kính đúng như lúc dựng: pinhole hoặc Kannala-Brandt) và
  ``rig_trajectories.json`` (quỹ đạo, cho ``get_available_trajectories``).
- ``scene.ply``: PLY Instant NuRec ở hệ world của clip NCore (``train.nurec/scene_nurec.ply``),
  cùng hệ với file cảnh.

Làm: ``render_rgb``, ``batch_render_rgb`` (ảnh ``RGB_UINT8_PLANAR``, PNG hoặc JPEG),
``get_version``, ``get_available_scenes``, ``get_available_cameras``,
``get_available_trajectories``, ``get_available_ego_masks`` (rỗng), ``get_dynamic_objects`` và
``get_external_asset_objects`` (rỗng: cảnh tĩnh), ``shut_down``. Chưa làm (``UNIMPLEMENTED``):
LiDAR, sửa asset.

Stub Python của proto lấy từ gói CARLA (``PythonAPI/examples/nvidia/nurec/nre``), sinh cho
protobuf 5.29: chạy bằng ``~/venvs/carla`` (protobuf 5.x, có thêm torch, gsplat của môi trường
chính).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import zipfile
from concurrent import futures
from pathlib import Path

import numpy as np

DEFAULT_CARLA_ROOT = Path.home() / "opt" / "carla" / "Carla-1.0-Linux-Shipping"
NUREC_EXAMPLES = Path("PythonAPI") / "examples" / "nvidia" / "nurec"
CAMERAS_FILE = "drive2sim_cameras.json"
MAX_MESSAGE_LENGTH = 512 * 1024 * 1024
SERVER_VERSION = "drive2sim-sensorsim 0.1.0"

log = logging.getLogger("drive2sim.sensorsim")


def load_protos(carla_root: Path) -> tuple[object, object, object]:
    """Nạp stub ``nre.grpc.protos`` từ gói CARLA."""
    examples = Path(carla_root).expanduser() / NUREC_EXAMPLES
    if not (examples / "nre").is_dir():
        raise SystemExit(
            f"không thấy stub NuRec trong {examples}: đặt --carla-root tới thư mục CARLA 1.0"
        )
    sys.path.insert(0, str(examples))
    import nre.grpc.protos.common_pb2 as common_pb2
    import nre.grpc.protos.sensorsim_pb2 as sensorsim_pb2
    import nre.grpc.protos.sensorsim_pb2_grpc as sensorsim_pb2_grpc

    return common_pb2, sensorsim_pb2, sensorsim_pb2_grpc


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Ma trận quay của quaternion (w, x, y, z), tự chuẩn hoá."""
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) của ma trận quay."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return (r[k, j] - r[j, k]) / s, q[0], q[1], q[2]


def pose_to_matrix(pose: object) -> np.ndarray:
    """Ma trận 4x4 của ``common.Pose`` (vec, quat)."""
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z)
    m[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return m


def load_ply(path: Path, device: str) -> tuple[object, ...]:
    """Gaussian của PLY Instant NuRec (xyz, f_dc SH0, opacity, scale log, rot wxyz)."""
    import torch

    with path.open("rb") as f:
        props, count = [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts and parts[0] == "property":
                kinds = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}
                props.append((parts[2], kinds[parts[1]]))
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=count)

    def col(*names: str) -> object:
        return torch.from_numpy(np.stack([data[k].astype(np.float32) for k in names], 1)).to(
            device
        )

    means = col("x", "y", "z")
    quats = torch.nn.functional.normalize(col("rot_0", "rot_1", "rot_2", "rot_3"), dim=1)
    scales = torch.exp(col("scale_0", "scale_1", "scale_2"))
    opacity = torch.sigmoid(col("opacity")[:, 0])
    colors = (col("f_dc_0", "f_dc_1", "f_dc_2") * 0.28209479 + 0.5).clamp(0, 1)
    return means, quats, scales, opacity, colors


class Scene:
    """Cảnh đã nạp: Gaussian trên GPU, ống kính từng camera, quỹ đạo rig."""

    def __init__(self, usdz: Path, ply: Path, device: str) -> None:
        with zipfile.ZipFile(usdz) as z:
            cameras = json.loads(z.read(CAMERAS_FILE))
            self.trajectories = json.loads(z.read("rig_trajectories.json"))
            self.info = json.loads(z.read("data_info.json"))
        self.scene_id = self.info["sequence_id"]
        self.cameras: dict[str, dict] = cameras["cameras"]
        self.device = device
        began = time.monotonic()
        self.gaussians = load_ply(ply, device)
        log.info(
            "nạp %s: %d Gaussian trong %.1f s",
            ply.name,
            len(self.gaussians[0]),
            time.monotonic() - began,
        )
        self.lock = threading.Lock()

    def intrinsics(
        self, camera: dict, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """K, hệ số méo và mô hình của camera, co theo độ phân giải được hỏi."""
        native_w, native_h = camera["resolution"]
        k = np.array(camera["K"], dtype=np.float32)
        k[0] *= width / native_w
        k[1] *= height / native_h
        return k, np.array(camera["radial"], dtype=np.float32), camera["model"]

    def render(
        self,
        world_from_camera: np.ndarray,
        k: np.ndarray,
        radial: np.ndarray,
        model: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Ảnh RGB uint8 (H, W, 3) từ pose camera → world (trục OpenCV)."""
        import torch
        from gsplat import rasterization

        viewmat = np.linalg.inv(world_from_camera).astype(np.float32)
        extra = {}
        if model == "fisheye":
            extra = dict(
                camera_model="fisheye",
                radial_coeffs=torch.from_numpy(radial)[None].to(self.device),
                with_ut=True,
                with_eval3d=True,
            )
        means, quats, scales, opacity, colors = self.gaussians
        with self.lock, torch.no_grad():
            rgb, _, _ = rasterization(
                means,
                quats,
                scales,
                opacity,
                colors,
                torch.from_numpy(viewmat)[None].to(self.device),
                torch.from_numpy(k)[None].to(self.device),
                width,
                height,
                render_mode="RGB",
                packed=False,
                **extra,
            )
        return (rgb[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def lens_from_spec(spec: object) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    """Ống kính từ ``CameraSpec`` của camera không có trong cảnh (F-theta, OpenCV fisheye → KB)."""
    w, h = spec.resolution_w, spec.resolution_h
    which = spec.WhichOneof("camera_param")
    if which == "opencv_fisheye_param":
        p = spec.opencv_fisheye_param
        k = np.array(
            [
                [p.focal_length_x, 0, p.principal_point_x],
                [0, p.focal_length_y, p.principal_point_y],
                [0, 0, 1],
            ],
            np.float32,
        )
        radial = np.zeros(4, np.float32)
        radial[: min(4, len(p.radial_coeffs))] = list(p.radial_coeffs)[:4]
        return k, radial, "fisheye", (w, h)
    if which == "ftheta_param":
        p = spec.ftheta_param
        forward = np.array(list(p.angle_to_pixeldist_poly), dtype=np.float64)
        theta = np.linspace(1e-3, p.max_angle or np.pi / 2, 2000)
        if forward.size:
            radius = np.polynomial.polynomial.polyval(theta, forward)
        else:  # chỉ có đa thức ngược: đảo bằng nội suy
            backward = np.array(list(p.pixeldist_to_angle_poly), dtype=np.float64)
            r = np.linspace(0, max(w, h), 4000)
            radius = np.interp(theta, np.polynomial.polynomial.polyval(r, backward), r)
        a = np.stack([theta ** (2 * i) for i in range(5)], 1)
        sol, *_ = np.linalg.lstsq(a, radius / theta, rcond=None)
        c = p.linear_cde.linear_c or 1.0
        k = np.array(
            [
                [sol[0] * c, 0, p.principal_point_x + 0.5],
                [0, sol[0], p.principal_point_y + 0.5],
                [0, 0, 1],
            ],
            np.float32,
        )
        return k, (sol[1:] / sol[0]).astype(np.float32), "fisheye", (w, h)
    raise ValueError(f"mô hình camera {which!r} chưa hỗ trợ khi camera không có trong cảnh")


def make_servicer(
    scene: Scene, protos: tuple[object, object, object], stop_event: threading.Event
) -> object:
    """Servicer ``SensorsimService`` phục vụ ``scene``; ``shut_down`` đặt ``stop_event``."""
    common_pb2, pb2, pb2_grpc = protos
    import grpc

    def encode(image: np.ndarray, fmt: int, quality: float) -> bytes:
        if fmt in (pb2.RGB_UINT8_PLANAR, pb2.UNDEFINED):
            return np.ascontiguousarray(image.transpose(2, 0, 1)).tobytes()
        import cv2

        bgr = image[..., ::-1]
        if fmt == pb2.PNG:
            ok, buf = cv2.imencode(".png", bgr)
        elif fmt == pb2.JPEG:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality or 95)])
        else:
            raise ValueError(f"định dạng ảnh {pb2.ImageFormat.Name(fmt)} chưa hỗ trợ")
        if not ok:
            raise RuntimeError("không mã hoá được ảnh")
        return buf.tobytes()

    def camera_spec(cam_id: str, camera: dict) -> object:
        w, h = camera["resolution"]
        k, radial = np.array(camera["K"]), camera["radial"]
        return pb2.CameraSpec(
            opencv_fisheye_param=pb2.OpenCVFisheyeCameraParam(
                principal_point_x=float(k[0, 2]),
                principal_point_y=float(k[1, 2]),
                focal_length_x=float(k[0, 0]),
                focal_length_y=float(k[1, 1]),
                # pinhole viết dạng KB: tan θ = θ + θ³/3 + 2θ⁵/15 + 17θ⁷/315 + 62θ⁹/2835
                radial_coeffs=list(radial)
                if camera["model"] == "fisheye"
                else [1 / 3, 2 / 15, 17 / 315, 62 / 2835],
                max_angle=float(np.pi / 2),
            ),
            logical_id=cam_id,
            trajectory_idx=0,
            resolution_h=int(h),
            resolution_w=int(w),
            shutter_type=pb2.GLOBAL,
        )

    def pose_msg(m: np.ndarray) -> object:
        w, x, y, z = matrix_to_quat(m[:3, :3])
        return common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
            quat=common_pb2.Quat(w=float(w), x=float(x), y=float(y), z=float(z)),
        )

    def check_scene(scene_id: str, context: object) -> None:
        if scene_id and scene_id != scene.scene_id:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"cảnh {scene_id!r} không có; server đang phục vụ {scene.scene_id!r}",
            )

    def render_one(request: object) -> bytes:
        spec = request.camera_intrinsics
        w = request.resolution_w or spec.resolution_w
        h = request.resolution_h or spec.resolution_h
        camera = scene.cameras.get(spec.logical_id)
        if camera is not None:
            k, radial, model = scene.intrinsics(camera, w, h)
        else:
            k, radial, model, (native_w, native_h) = lens_from_spec(spec)
            k[0] *= w / (native_w or w)
            k[1] *= h / (native_h or h)
        pose = pose_to_matrix(request.sensor_pose.start_pose)
        image = scene.render(pose, k, radial, model, int(w), int(h))
        return encode(image, request.image_format, request.image_quality)

    class Servicer(pb2_grpc.SensorsimServiceServicer):
        def get_version(self, request: object, context: object) -> object:
            return common_pb2.VersionId(
                version_id=SERVER_VERSION,
                git_hash="",
                grpc_api_version=common_pb2.VersionId.APIVersion(major=26, minor=4, patch=0),
            )

        def get_available_scenes(self, request: object, context: object) -> object:
            return common_pb2.AvailableScenesReturn(scene_ids=[scene.scene_id])

        def get_available_cameras(self, request: object, context: object) -> object:
            check_scene(request.scene_id, context)
            items = []
            for cam_id, camera in scene.cameras.items():
                rig_to_camera = np.linalg.inv(np.array(camera["T_sensor_rig"]))
                items.append(
                    pb2.AvailableCamerasReturn.AvailableCamera(
                        intrinsics=camera_spec(cam_id, camera),
                        rig_to_camera=pose_msg(rig_to_camera),
                        logical_id=cam_id,
                        trajectory_idx=0,
                    )
                )
            return pb2.AvailableCamerasReturn(available_cameras=items)

        def get_available_trajectories(self, request: object, context: object) -> object:
            check_scene(request.scene_id, context)
            result = []
            for index, trajectory in enumerate(scene.trajectories["rig_trajectories"]):
                poses = [
                    common_pb2.PoseAtTime(pose=pose_msg(np.array(m)), timestamp_us=int(t))
                    for m, t in zip(
                        trajectory["T_rig_worlds"], trajectory["T_rig_world_timestamps_us"]
                    )
                ]
                result.append(
                    pb2.AvailableTrajectoriesReturn.AvailableTrajectory(
                        trajectory_idx=index, trajectory=common_pb2.Trajectory(poses=poses)
                    )
                )
            return pb2.AvailableTrajectoriesReturn(available_trajectories=result)

        def get_available_ego_masks(self, request: object, context: object) -> object:
            return pb2.AvailableEgoMasksReturn()

        def get_dynamic_objects(self, request: object, context: object) -> object:
            return pb2.AvailableDynamicObjectsReturn()

        def get_external_asset_objects(self, request: object, context: object) -> object:
            return pb2.ExternalAssetObjectsReturn()

        def render_rgb(self, request: object, context: object) -> object:
            check_scene(request.scene_id, context)
            try:
                return pb2.RGBRenderReturn(image_bytes=render_one(request))
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        def batch_render_rgb(self, request: object, context: object) -> object:
            items = []
            for item in request.items:
                try:
                    if item.request.scene_id and item.request.scene_id != scene.scene_id:
                        raise ValueError(f"cảnh {item.request.scene_id!r} không có")
                    result = pb2.RGBRenderReturn(image_bytes=render_one(item.request))
                    items.append(
                        pb2.BatchRGBRenderReturnItem(
                            camera_name=item.camera_name, result=result, success=True
                        )
                    )
                except Exception as error:  # noqa: BLE001  (mỗi camera báo lỗi riêng, như NRE)
                    log.exception("render %s lỗi", item.camera_name)
                    items.append(
                        pb2.BatchRGBRenderReturnItem(
                            camera_name=item.camera_name, success=False, error_message=str(error)
                        )
                    )
            return pb2.BatchRGBRenderReturn(items=items)

        def shut_down(self, request: object, context: object) -> object:
            stop_event.set()
            return common_pb2.Empty()

    return Servicer()


def main(argv: list[str] | None = None) -> int:
    """Nạp cảnh rồi phục vụ gRPC tới khi ``shut_down`` hoặc Ctrl+C."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--usdz", required=True, type=Path)
    p.add_argument("--ply", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=46435)
    p.add_argument(
        "--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", DEFAULT_CARLA_ROOT))
    )
    p.add_argument("--gpu", default=None, help="GPU dùng để render (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import grpc

    protos = load_protos(a.carla_root)
    scene = Scene(a.usdz, a.ply, "cuda")
    stop_event = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=a.workers),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    protos[2].add_SensorsimServiceServicer_to_server(
        make_servicer(scene, protos, stop_event), server
    )
    address = f"{a.host}:{a.port}"
    if not server.add_insecure_port(address):
        raise SystemExit(f"không mở được cổng {address}")
    server.start()
    # CARLA (nurec_render_service) dò dòng này trong log để biết server đã sẵn sàng
    print(
        f"Serving on {address} (scene {scene.scene_id}, {len(scene.cameras)} camera)", flush=True
    )
    try:
        while not stop_event.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    server.stop(grace=2).wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
