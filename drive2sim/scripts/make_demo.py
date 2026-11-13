"""Dựng một thư mục làm việc mẫu đầy đủ để xem Studio khi đã có kết quả (dữ liệu giả).

Dùng khi muốn xem giao diện mà chưa có dữ liệu thật hay GPU::

    python -m drive2sim.scripts.make_demo .demo        # hoặc: make demo
    d2s studio --workdir .demo

Thư mục mẫu có:
- 3 cảnh của 3 dataset (nuScenes, Waymo, KITTI), mỗi cảnh có ảnh mọi camera theo khung, split và
  báo cáo kiểm tra (có cảnh báo, có lỗi);
- 11 run với đủ trạng thái (xong, lỗi, đang chạy, đã huỷ, chạy thử nhanh), chạy trên máy hoặc
  Kaggle, trên các GPU khác nhau; có metric, thống kê train, máy đã chạy, log, video, file
  ``scene.ply`` và gói xem trước có xe chạy theo khung;
- cache dùng chung (mask, SMPL) và một mục cache không còn run nào dùng.

Mọi file được kiểm tra theo schema của tool trước khi ghi. Số liệu là giả; ảnh, video và splat là
hình tổng hợp, không phải dữ liệu thật. Chỉ dùng thư viện chuẩn (video cần lệnh ``ffmpeg``; không
có thì bỏ qua video).
"""

import argparse
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import zlib
from array import array
from pathlib import Path
from typing import Any, Optional

import yaml

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    ArtifactRecord,
    EnvRecord,
    Manifest,
    RunRecord,
    Summary,
)
from drive2sim.engine.scenes import create_split
from drive2sim.engine.workdir import DEMO_MARKER, Workdir

RNG = random.Random(7)
SH_C0 = 0.28209479177387814

# ---------------------------------------------------------------- ảnh


def _png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    """Ghi ảnh PNG RGB 8 bit từ các hàng điểm ảnh."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def street_image(path: Path, frame: int, camera: int, tint: tuple[int, int, int]) -> None:
    """Ảnh đường phố tổng hợp: trời, dãy nhà có cửa sổ, mặt đường có vạch chạy theo khung."""
    width, height = 240, 135
    rows = []
    shift = (frame * 6 + camera * 17) % 48
    for y in range(height):
        if y < 45:  # trời, sáng dần về chân trời
            k = y / 45
            px = bytes((int(96 + 70 * k), int(140 + 60 * k), int(200 + 30 * k)))
            rows.append(px * width)
        elif y < 85:  # nhà: các khối màu, cửa sổ sáng
            row = bytearray()
            for x in range(width):
                block = ((x + shift * 2) // 40) % 3
                base = [tint, (150, 140, 128), (110, 118, 132)][block]
                window = (x % 10 < 4) and (y % 9 < 4) and y > 50
                row += bytes((230, 220, 170) if window else base)
            rows.append(bytes(row))
        else:  # đường, vạch giữa đứt quãng chạy theo khung
            row = bytearray()
            for x in range(width):
                lane = abs(x - width // 2) < 2 and ((y + shift) % 24) < 12
                curb = y < 89
                row += bytes((235, 210, 70) if lane else (150, 150, 150) if curb else (64, 68, 74))
            rows.append(bytes(row))
    _png(path, width, height, rows)


# ---------------------------------------------------------------- splat


PLY_FIELDS = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip


def write_ply(
    path: Path, points: list[tuple[float, float, float, float, float, float]], scale: float
) -> None:
    """Ghi các điểm ``(x, y, z, r, g, b)`` thành PLY 3DGS (bố cục INRIA, SH bậc 0)."""
    data = array("f")
    log_scale = math.log(scale)
    for x, y, z, r, g, b in points:
        data.extend(
            (x, y, z, 0, 0, 0, (r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0, 4.0,
             log_scale, log_scale, log_scale, 1, 0, 0, 0)
        )  # fmt: skip
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
        + "".join(f"property float {name}\n" for name in PLY_FIELDS)
        + "end_header\n"
    )
    if sys.byteorder != "little":
        data.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode() + data.tobytes())


def _box(center: tuple[float, float, float], size: tuple[float, float, float],
         color: tuple[float, float, float], count: int) -> list[tuple[float, ...]]:  # fmt: skip
    """Điểm trên mặt một hộp, màu lệch nhẹ để trông có chất liệu."""
    points = []
    for _ in range(count):
        p = [RNG.uniform(-0.5, 0.5) * s for s in size]
        face = RNG.randrange(3)
        p[face] = RNG.choice((-0.5, 0.5)) * size[face]
        k = RNG.uniform(0.85, 1.1)
        points.append((p[0] + center[0], p[1] + center[1], p[2] + center[2],
                       min(1, color[0] * k), min(1, color[1] * k), min(1, color[2] * k)))  # fmt: skip
    return points


def street_background() -> list[tuple[float, ...]]:
    """Nền: mặt đường (trục z lên), vạch giữa, vỉa hè và dãy nhà hai bên."""
    points: list[tuple[float, ...]] = []
    for _ in range(40000):
        k = RNG.uniform(0.85, 1.15)
        points.append((RNG.uniform(-60, 60), RNG.uniform(-5, 5), 0, 0.24 * k, 0.25 * k, 0.27 * k))
    for _ in range(3000):
        x = RNG.uniform(-60, 60)
        if math.floor(x / 3) % 2 == 0:
            points.append((x, RNG.uniform(-0.08, 0.08), 0.01, 0.95, 0.75, 0.1))
    for side in (1, -1):
        for _ in range(8000):
            points.append((RNG.uniform(-60, 60), side * RNG.uniform(5, 9), 0.12, 0.62, 0.6, 0.56))
    palette = [(0.7, 0.55, 0.45), (0.55, 0.6, 0.7), (0.8, 0.78, 0.7), (0.5, 0.52, 0.5)]
    for x in range(-55, 60, 14):
        for side in (1, -1):
            h = RNG.uniform(6, 18)
            points += _box((x, side * 14, h / 2), (10, 8, h), RNG.choice(palette), 3500)
    return points


def car(color: tuple[float, float, float]) -> list[tuple[float, ...]]:
    """Một chiếc xe trong hệ toạ độ của chính nó: thân và cabin."""
    return _box((0, 0, 0.75), (4.4, 1.9, 1.1), color, 3000) + _box(
        (-0.3, 0, 1.6), (2.4, 1.7, 0.7), (0.15, 0.18, 0.22), 1500
    )


# ---------------------------------------------------------------- video


def write_video(path: Path, seconds: int = 3) -> bool:
    """Video MP4 H.264 tổng hợp bằng ``ffmpeg``; trả ``False`` nếu không có ``ffmpeg``."""
    if shutil.which("ffmpeg") is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", f"testsrc2=size=480x270:rate=12:duration={seconds}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    return subprocess.run(command, capture_output=True).returncode == 0


# ---------------------------------------------------------------- cảnh

SCENES: list[dict[str, Any]] = [
    {
        "scene_id": "nuscenes_scene-0061",
        "dataset": "nuscenes",
        "sequence": "scene-0061",
        "cameras": [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
        ],  # fmt: skip
        "resolution": [1600, 900],
        "frames": 20,
        "lidar": True,
        "tint": (140, 120, 90),
        "splits": ["every_nth_frame:4", "every_nth_frame:8"],
        "checks": [
            ("lidar.missing", "warning", "thiếu LiDAR ở 1 khung", {"count": 1, "frames": [17]}),
            (
                "projection.not_run",
                "info",
                "chưa chiếu LiDAR lên ảnh; calib và mốc thời gian chưa được kiểm tra hình học",
                {},
            ),
        ],  # fmt: skip
    },
    {
        "scene_id": "waymo_seg-1017",
        "dataset": "waymo",
        "sequence": "seg-1017",
        "cameras": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        "resolution": [1920, 1280],
        "frames": 30,
        "lidar": True,
        "tint": (110, 120, 150),
        "splits": ["every_nth_frame:10"],
        "checks": [],
    },
    {
        "scene_id": "kitti_0006",
        "dataset": "kitti",
        "sequence": "0006",
        "cameras": ["image_02", "image_03"],
        "resolution": [1242, 375],
        "frames": 24,
        "lidar": False,
        "tint": (120, 140, 100),
        "splits": ["every_nth_frame:4"],
        "checks": [
            (
                "images.missing",
                "error",
                "thiếu ảnh camera image_03 ở 2 khung",
                {"count": 2, "frames": [11, 12]},
            ),
            ("cuboids.sparse", "warning", "ít box 3D: 3 vật thể trong cả cảnh", {"count": 3}),
        ],  # fmt: skip
    },
]


def make_scene(work: Workdir, spec: dict[str, Any]) -> list[str]:
    """Ghi manifest, ảnh, báo cáo kiểm tra và split của một cảnh; trả về các ``split_id``."""
    scene_dir = work.scene_dir(spec["scene_id"])
    for frame in range(spec["frames"]):
        for index, _ in enumerate(spec["cameras"]):
            street_image(
                scene_dir / "images" / f"{frame:03d}_{index}.png", frame, index, spec["tint"]
            )
    manifest = {
        "schema_version": SCHEMA_VERSIONS["manifest"],
        "scene_id": spec["scene_id"],
        "source": {
            "dataset": spec["dataset"],
            "sequence_id": spec["sequence"],
            "converter": f"drivestudio.{spec['dataset']}",
            "converter_version": "0.1.0+demo",
        },
        "track": "pinhole",
        "cameras": [
            {
                "camera_id": c,
                "model": "opencv_pinhole",
                "resolution": spec["resolution"],
                "shutter": "GLOBAL",
            }
            for c in spec["cameras"]
        ],
        "lidars": ["LIDAR_TOP"] if spec["lidar"] else [],
        "timeline": {"reference_sensor": spec["cameras"][0], "num_frames": spec["frames"]},
        "available": {"lidar": spec["lidar"], "cuboids": "gt"},
    }
    files.dump(Manifest.model_validate(manifest), scene_dir / "manifest.yaml")
    checks = [
        {"check_id": c, "severity": s, "message": m, "details": d} for c, s, m, d in spec["checks"]
    ]
    report = {
        "schema_version": SCHEMA_VERSIONS["report"],
        "scene_id": spec["scene_id"],
        "created_utc": "2026-11-12T07:40:00Z",
        "summary": {
            "errors": sum(1 for c in checks if c["severity"] == "error"),
            "warnings": sum(1 for c in checks if c["severity"] == "warning"),
        },
        "checks": checks,
    }
    (scene_dir / "reports").mkdir(parents=True, exist_ok=True)
    (scene_dir / "reports" / "validation.json").write_text(json.dumps(report, indent=2), "utf-8")
    return [create_split(work, spec["scene_id"], strategy).split_id for strategy in spec["splits"]]


# ---------------------------------------------------------------- run

MACHINES = {
    "T4": {"cpu": "Intel(R) Xeon(R) CPU @ 2.00GHz", "cpu_cores": 4, "ram_gb": 31.4,
           "gpu": "Tesla T4", "gpu_count": 1, "gpu_memory_mb": 15360, "gpu_driver": "535.104.05",
           "os": "Linux 6.6.56+"},
    "P100": {"cpu": "Intel(R) Xeon(R) CPU @ 2.20GHz", "cpu_cores": 4, "ram_gb": 31.4,
             "gpu": "Tesla P100-PCIE-16GB", "gpu_count": 1, "gpu_memory_mb": 16384,
             "gpu_driver": "535.104.05", "os": "Linux 6.6.56+"},
    "4090": {"cpu": "AMD Ryzen 9 7950X 16-Core Processor", "cpu_cores": 32, "ram_gb": 125.6,
             "gpu": "NVIDIA GeForce RTX 4090", "gpu_count": 1, "gpu_memory_mb": 24564,
             "gpu_driver": "550.120", "os": "Linux 6.8.0-48-generic"},
}  # fmt: skip

CACHE_HASH = {"prep.masks_segformer": "4f1c9a0e", "prep.smpl": "7b2d10c3"}
PREP_TIME = {"prep.masks_segformer": 812.4, "prep.smpl": 1920.7}

TRAIN_LOG = """\
[train.py] config: {config}
[train.py] loading 6 cameras x {frames} frames, downscale_when_loading=[2, 2, 2]
[train.py] Background: 412,385 points initialised from LiDAR
[train.py] step 500  | loss 0.1842 | psnr 19.71 | gaussians 0.61M | 12.3 it/s
[train.py] step 5000 | loss 0.0731 | psnr 24.88 | gaussians 1.02M | 11.8 it/s
[train.py] step 15000 | saved checkpoint_15000.pth, checkpoint_resume.pth
"""


def stage_dir(run_dir: Path, stage_id: str) -> Path:
    """Thư mục kết quả của một stage trong run."""
    return run_dir / "d2s" / "stages" / stage_id


def _artifact(path: Path, stage_id: str, artifact_type: str, duration: Optional[float],
              cache_key: Optional[str] = None, params: Optional[dict] = None) -> None:  # fmt: skip
    record = ArtifactRecord(
        schema_version=SCHEMA_VERSIONS["artifact"],
        artifact_type=artifact_type,
        cache_key=cache_key,
        producer={"stage_id": stage_id, "version": "0.1.0", "code_commit": "e695d7d"},
        params=params or {},
        inputs=["scene:demo"],
        created_utc="2026-11-12T08:10:00Z",
        files=[],
        duration_s=duration,
    )
    files.dump(record, path / "artifact.json")


def make_cache(work: Workdir, scene_id: str) -> dict[str, str]:
    """Kết quả cache dùng chung của các stage chuẩn bị; trả ``stage_id`` → ``cache_key``."""
    keys = {}
    for stage_id, digest in CACHE_HASH.items():
        key = f"{scene_id}/{stage_id}@0.1.0-{digest}"
        folder = work.cache_dir / scene_id / key.split("/", 1)[1]
        kind = "masks" if "masks" in stage_id else "smpl"
        _artifact(folder, stage_id, kind, PREP_TIME[stage_id], key)
        (folder / "payload.bin").write_bytes(RNG.randbytes(RNG.randrange(3, 30) * 1024 * 1024))
        keys[stage_id] = key
    return keys


def make_run(
    work: Workdir, spec: dict[str, Any], cache: dict[str, str], scene_points: list
) -> None:
    """Ghi mọi file của một run theo trạng thái trong ``spec``."""
    run_dir = work.run_dir(spec["scene"], spec["approach"], spec["run_id"])
    stages = []
    order = (
        ["prep.masks_segformer", "prep.smpl"]
        if spec["approach"] == "omnire"
        else ["prep.masks_segformer"]
    ) + ["train.drivestudio", "eval.standard", "export.ply_world"]
    failed_at = spec.get("failed_at")
    running_at = spec.get("running_at")
    cancelled_at = spec.get("cancelled_at")
    stop = failed_at or running_at or cancelled_at
    reached_stop = False
    train_time = spec.get("train_time", 4380.0)
    for stage_id in order:
        record: dict[str, Any] = {"stage_id": stage_id, "params": {}}
        if reached_stop:
            record["status"] = "pending"
        elif stage_id == stop:
            reached_stop = True
            record["status"] = (
                "failed"
                if stage_id == failed_at
                else "running"
                if stage_id == running_at
                else "cancelled"
            )
            record["duration_s"] = spec.get("stop_duration", 905.0)
            if stage_id == failed_at:
                record["peak_vram_mb"] = 15800
                record["error"] = {
                    "code": "stage_failed",
                    "message": "stage train.drivestudio lỗi: RuntimeError: CUDA out of memory. "
                    "Tried to allocate 1.20 GiB (GPU 0; 15.89 GiB total capacity)",
                    "details": {"log": "d2s/log.txt"},
                }
        else:
            record["status"] = "succeeded"
            if stage_id in cache:
                record["cache_key"] = cache[stage_id]
                if spec.get("first", False):
                    record["duration_s"] = PREP_TIME[stage_id]
                else:
                    record["cache_hit"] = True
                    record["reused_duration_s"] = PREP_TIME[stage_id]
            elif stage_id == "train.drivestudio":
                record["duration_s"] = round(train_time * 1.07, 1)  # gồm khởi động, đánh giá cuối
                record["params"] = {"method": spec["approach"], **spec["options"]}
                if spec.get("quick"):
                    record["params"]["num_iters"] = 3000
            elif stage_id == "eval.standard":
                record["duration_s"] = 3.2
            else:
                record["duration_s"] = 41.5
        stages.append(record)
    status = (
        "failed"
        if failed_at
        else "running"
        if running_at
        else "cancelled"
        if cancelled_at
        else "succeeded"
    )
    if spec.get("preview"):
        # d2s export preview thêm stage này vào cuối run.json (engine/run.py, add_stage).
        stages.append(
            {"stage_id": "export.preview", "status": "succeeded", "duration_s": 28.4, "params": {}}
        )
    run = {
        "schema_version": SCHEMA_VERSIONS["run"],
        "run_id": spec["run_id"],
        "scene_id": spec["scene"],
        "split_id": spec["split"],
        "approach": {
            "approach_id": spec["approach"],
            "version": "0.1.0",
            "preset": spec.get("preset"),
            "options": spec["options"],
            "quick": spec.get("quick", False),
        },  # fmt: skip
        "level": "L1",
        "executor": spec.get("executor", "local"),
        "status": status,
        "created_utc": spec["created"],
        "lidar_usage": [],
        "stages": stages,
    }
    files.dump(RunRecord.model_validate(run), run_dir / "d2s" / "run.json")

    machines = [
        {**MACHINES[name], "first_seen_utc": spec["created"]}
        for name in spec.get("machines", ["T4"])
    ]
    env = {
        "schema_version": SCHEMA_VERSIONS["env"], "code_commit": "e695d7d", "python": "3.9.19",
        # Run trên máy cá nhân (4090) dùng môi trường mới, run trên Kaggle dùng bản cũ: để thấy
        # cảnh báo trộn phiên bản ở trang so sánh.
        **({"pytorch": "2.4.1+cu121", "cuda": "12.1", "gsplat": "1.5.3"} if "4090" in spec.get("machines", [])
           else {"pytorch": "2.0.0+cu117", "cuda": "11.7", "gsplat": "1.3.0"}),
        "gpu": machines[-1]["gpu"], "seed": 0, "executor": run["executor"], "machines": machines,
    }  # fmt: skip
    files.dump(EnvRecord.model_validate(env), run_dir / "d2s" / "env.json")

    log = [f"[d2s] run {spec['run_id']} bắt đầu trên {machines[-1]['gpu']}"]
    for record in stages:
        if record["status"] == "pending":
            break
        log.append(f"[d2s] stage {record['stage_id']}: {record['status']}")
        if record.get("error"):
            log += ["Traceback (most recent call last):", '  File "drive2sim/recon/tools/train.py", '
                    "line 268, in main", "    outputs = trainer(image_infos, cam_infos)",
                    "RuntimeError: CUDA out of memory."]  # fmt: skip
    (run_dir / "d2s" / "log.txt").write_text("\n".join(log) + "\n", "utf-8")

    for record in stages:
        stage_id = record["stage_id"]
        if record["status"] == "pending" or record.get("cache_hit"):
            continue
        folder = stage_dir(run_dir, stage_id)
        folder.mkdir(parents=True, exist_ok=True)
        lines = [f"[{stage_id}] bắt đầu"]
        if stage_id == "train.drivestudio":
            lines = TRAIN_LOG.format(config=spec["approach"], frames=20).splitlines()
            if record["status"] == "running":
                lines.append("[train.py] step 18250 | loss 0.0612 | psnr 25.94 | gaussians 1.21M")
        (folder / "log.txt").write_text("\n".join(lines) + "\n", "utf-8")
        if record["status"] == "succeeded":
            (folder / "done").write_text("2026-11-12T09:00:00Z\n")

    train_dir = stage_dir(run_dir, "train.drivestudio")
    if status in ("succeeded", "running"):
        write_train_curve(train_dir, spec, done=status == "succeeded")
    if status != "succeeded":
        return
    iters = 3000 if spec.get("quick") else 30000
    yaml_text = {"data": {"pixel_source": {"downscale_when_loading": [2, 2, 2], "downscale": 1}}}
    (train_dir / "config.yaml").write_text(yaml.safe_dump(yaml_text))
    gaussians = spec["gaussians"]
    per_node = {"Background": int(gaussians * 0.78), "RigidNodes": int(gaussians * 0.18)}
    if spec["approach"] == "omnire":
        per_node["SMPLNodes"] = int(gaussians * 0.03)
        per_node["DeformableNodes"] = gaussians - sum(per_node.values())
    else:
        per_node["RigidNodes"] = gaussians - per_node["Background"]
    stats = {"step": iters, "num_iters": iters, "train_time_s": train_time,
             "peak_vram_mb": spec["vram"], "num_gaussians": gaussians, "gaussians_per_node": per_node}  # fmt: skip
    (train_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    write_video(train_dir / "videos" / f"test_set_{iters}_rgbs.mp4")
    write_video(train_dir / "videos" / f"test_set_{iters}_error_maps.mp4")

    psnr = spec["psnr"]
    summary = {
        "schema_version": SCHEMA_VERSIONS["summary"],
        "protocol": {"version": "0.1", "image_dtype": "uint8",
                     "resolution": "downscale_when_loading=[2, 2, 2], downscale=1",
                     "lpips": {"net": "alex", "package": "torchmetrics", "package_version": "0.10.3"}},
        "split_id": spec["split"],
        "overall": {"psnr": psnr, "ssim": spec["ssim"], "lpips": spec["lpips"],
                    "human_psnr": round(psnr - 3.1, 2), "vehicle_psnr": round(psnr - 2.2, 2),
                    "dynamic_psnr": round(psnr - 2.6, 2), "occupied_psnr": round(psnr - 0.8, 2)},
        "train_time_s": train_time, "peak_vram_mb": spec["vram"], "num_gaussians": gaussians,
        "num_iters": iters, "render_fps": spec["fps"], "render_resolution": "800x450",
    }  # fmt: skip
    eval_dir = stage_dir(run_dir, "eval.standard")
    files.dump(Summary.model_validate(summary), eval_dir / "summary.json")

    export = stage_dir(run_dir, "export.ply_world")
    keep = max(1, int(len(scene_points) * spec.get("ply_share", 1.0)))
    write_ply(export / "scene.ply", scene_points[:keep], 0.18)
    (export / "scene.json").write_text(json.dumps({"frame": "world", "rigid_frame": -1}))

    if spec.get("preview"):
        preview = stage_dir(run_dir, "export.preview")
        write_ply(preview / "background.ply", scene_points[:keep], 0.18)
        colors = [(0.85, 0.15, 0.12), (0.15, 0.35, 0.85), (0.95, 0.95, 0.95), (0.2, 0.6, 0.3)]
        tracks = []
        for index, color in enumerate(colors):
            write_ply(preview / "actors" / f"rigid_{index:04d}.ply", car(color), 0.07)
            tracks.append(f"rigid_{index:04d}")
        poses = []
        for frame in range(60):
            row: list[Optional[list[float]]] = []
            for index in range(len(tracks)):
                lane_y = [-2.5, 2.5, -2.5, 2.5][index]
                speed = [0.9, -0.7, 0.6, -0.5][index]
                x = [-30, 30, -45, 45][index] + speed * frame
                if index == 2 and frame > 45:
                    row.append(None)  # xe ra khỏi cảnh
                    continue
                c, s = (1.0, 0.0) if speed > 0 else (-1.0, 0.0)
                row.append([c, -s, 0, x, s, c, 0, lane_y, 0, 0, 1, 0, 0, 0, 0, 1])
            poses.append(row)
        (preview / "actor_poses.json").write_text(
            json.dumps({"track_ids": tracks, "T_object_world": poses})
        )
        info = {"schema_version": SCHEMA_VERSIONS["preview"], "source_run": spec["run_id"],
                "sh_degree": 0, "rigid_actors": tracks, "box_only_actors": ["smpl_0000", "smpl_0001"],
                "omitted": ["sky", "per_camera_color", "SMPLNodes", "DeformableNodes"]}  # fmt: skip
        (preview / "preview.json").write_text(json.dumps(info, indent=2))
        (preview / "done").write_text("2026-11-12T09:00:00Z\n")

    if run["executor"] == "kaggle":
        state = {"username": "demo", "kernel": f"demo/d2s-{spec['run_id'][-4:]}0000-{len(machines)}",
                 "attempt": len(machines), "dataset": f"demo/d2s-{spec['scene'][-4:]}0000",
                 "commit": "e695d7d", "pushed_utc": spec["created"], "pulled_utc": spec["created"]}  # fmt: skip
        (run_dir / "d2s" / "kaggle.json").write_text(json.dumps(state, indent=2))


def write_train_curve(train_dir: Path, spec: dict[str, Any], done: bool) -> None:
    """``metrics.json`` như ``train.py`` ghi: mỗi 500 bước một dòng JSON."""
    iters = 3000 if spec.get("quick") else 30000
    last = iters if done else 18250
    final_psnr = spec.get("psnr", 25.0) + 3.0  # PSNR trên ảnh train cao hơn ảnh test
    gaussians = spec.get("gaussians", 1_200_000)
    lines = []
    for step in range(0, last + 1, 100 if spec.get("quick") else 500):
        k = 1 - math.exp(-step / (iters * 0.18))
        psnr = 14 + (final_psnr - 14) * k + RNG.uniform(-0.25, 0.25)
        loss = 0.32 * (1 - k) + 0.05 + RNG.uniform(-0.004, 0.004)
        grow = min(1.0, step / (iters * 0.5))
        count = int(400_000 + (gaussians - 400_000) * grow)
        lines.append(json.dumps({
            "iteration": step, "iter_time": 0.09, "data_time": 0.01,
            "train_metrics/psnr": round(psnr, 3), "losses/rgb_loss": round(loss * 0.7, 5),
            "losses/ssim_loss": round(loss * 0.3, 5),
            "train_stats/gaussian_num_Background": int(count * 0.8),
            "train_stats/gaussian_num_RigidNodes": count - int(count * 0.8),
        }))  # fmt: skip
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "metrics.json").write_text("\n".join(lines) + "\n")


RUNS: list[dict[str, Any]] = [
    {"run_id": "20261112-081000-omnire-a1b2", "scene": "nuscenes_scene-0061", "approach": "omnire",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": True}, "preset": "fastgs",
     "created": "2026-11-12T08:10:00Z", "executor": "kaggle", "first": True, "machines": ["T4"],
     "psnr": 27.64, "ssim": 0.861, "lpips": 0.198, "train_time": 4105.2, "vram": 7380,
     "gaussians": 1284000, "fps": 38.6, "preview": True},
    {"run_id": "20261112-093000-omnire-b7c1", "scene": "nuscenes_scene-0061", "approach": "omnire",
     "options": {"renderer": "3dgs", "campose": True, "fastgs": False}, "preset": "paper",
     "created": "2026-11-12T09:30:00Z", "executor": "kaggle", "machines": ["T4", "P100"],
     "psnr": 28.31, "ssim": 0.874, "lpips": 0.183, "train_time": 9822.0, "vram": 11240,
     "gaussians": 1912000, "fps": 24.9},
    {"run_id": "20261112-101500-streetgs-c3d4", "scene": "nuscenes_scene-0061", "approach": "streetgs",
     "options": {"renderer": "3dgeer", "campose": False, "fastgs": False}, "preset": "geer",
     "created": "2026-11-12T10:15:00Z", "machines": ["4090"],
     "psnr": 24.44, "ssim": 0.803, "lpips": 0.271, "train_time": 3020.4, "vram": 14900,
     "gaussians": 2150000, "fps": 61.3, "ply_share": 0.9, "preview": True},
    {"run_id": "20261112-120500-omnire-d9e0", "scene": "nuscenes_scene-0061", "approach": "omnire",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": False}, "quick": True,
     "created": "2026-11-12T12:05:00Z", "machines": ["4090"],
     "psnr": 21.05, "ssim": 0.702, "lpips": 0.388, "train_time": 182.6, "vram": 5100,
     "gaussians": 640000, "fps": 88.0, "ply_share": 0.5},
    {"run_id": "20261112-143000-streetgs-e5f6", "scene": "nuscenes_scene-0061", "approach": "streetgs",
     "options": {"renderer": "3dgs", "campose": True, "fastgs": False},
     "created": "2026-11-12T14:30:00Z", "executor": "kaggle", "machines": ["T4"],
     "failed_at": "train.drivestudio"},
    {"run_id": "20261113-083000-omnire-f2a8", "scene": "nuscenes_scene-0061", "approach": "omnire",
     "options": {"renderer": "3dgeer", "campose": False, "fastgs": False}, "preset": "geer",
     "created": "2026-11-13T08:30:00Z", "machines": ["4090"],
     "running_at": "train.drivestudio", "stop_duration": 1530.0},
    {"run_id": "20261112-160000-streetgs-a0b9", "scene": "nuscenes_scene-0061", "approach": "streetgs",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": True},
     "created": "2026-11-12T16:00:00Z", "machines": ["4090"],
     "cancelled_at": "train.drivestudio", "stop_duration": 640.0},
    {"run_id": "20261112-170000-omnire-c4d5", "scene": "waymo_seg-1017", "approach": "omnire",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": True}, "preset": "fastgs",
     "created": "2026-11-12T17:00:00Z", "executor": "kaggle", "first": True, "machines": ["T4"],
     "psnr": 29.12, "ssim": 0.889, "lpips": 0.171, "train_time": 5210.0, "vram": 9050,
     "gaussians": 1544000, "fps": 31.7, "preview": True},
    {"run_id": "20261112-190000-streetgs-e6f7", "scene": "waymo_seg-1017", "approach": "streetgs",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": False},
     "created": "2026-11-12T19:00:00Z", "executor": "kaggle", "machines": ["T4"],
     "psnr": 27.95, "ssim": 0.866, "lpips": 0.205, "train_time": 7480.0, "vram": 12310,
     "gaussians": 1998000, "fps": 27.4},
    # Hai run trên split thứ hai của nuScenes (every_nth_frame:8), để thử bảng trộn nhiều split.
    {"run_id": "20261112-210000-omnire-b3e1", "scene": "nuscenes_scene-0061", "approach": "omnire",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": True}, "preset": "fastgs",
     "split_index": 1, "created": "2026-11-12T21:00:00Z", "machines": ["4090"],
     "psnr": 26.12, "ssim": 0.842, "lpips": 0.221, "train_time": 2380.5, "vram": 7020,
     "gaussians": 1190000, "fps": 74.2},
    {"run_id": "20261112-223000-streetgs-d7a2", "scene": "nuscenes_scene-0061", "approach": "streetgs",
     "options": {"renderer": "3dgs", "campose": False, "fastgs": False},
     "split_index": 1, "created": "2026-11-12T22:30:00Z", "machines": ["4090"],
     "psnr": 25.37, "ssim": 0.829, "lpips": 0.236, "train_time": 3105.8, "vram": 10480,
     "gaussians": 1730000, "fps": 52.9},
]  # fmt: skip


def main(argv: Optional[list[str]] = None) -> int:
    """Dựng thư mục mẫu.

    Args:
        argv: Tham số dòng lệnh; mặc định lấy từ ``sys.argv``.

    Returns:
        Mã thoát ``0``.
    """
    parser = argparse.ArgumentParser(description="Dựng thư mục làm việc mẫu cho Studio")
    parser.add_argument("workdir", type=Path, help="thư mục sẽ tạo, ví dụ .demo")
    parser.add_argument("--force", action="store_true", help="xoá thư mục cũ nếu đã có")
    args = parser.parse_args(argv)
    root: Path = args.workdir
    if root.exists() and any(root.iterdir()):
        if not args.force:
            print(f"{root} đã có dữ liệu; thêm --force để tạo lại", file=sys.stderr)
            return 1
        for name in ("data", "runs", "cache"):
            shutil.rmtree(root / name, ignore_errors=True)
    work = Workdir(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / DEMO_MARKER).write_text(
        "Thư mục dữ liệu mẫu (giả) do drive2sim/scripts/make_demo.py tạo; không phải kết quả thật.\n",
        "utf-8",
    )
    splits: dict[str, list[str]] = {}
    for spec in SCENES:
        splits[spec["scene_id"]] = make_scene(work, spec)
        print(f"cảnh {spec['scene_id']}: {spec['frames']} khung, {len(spec['cameras'])} camera")
    background = street_background()
    RNG.shuffle(background)  # để cắt bớt (ply_share) vẫn đều khắp cảnh
    caches = {spec["scene_id"]: make_cache(work, spec["scene_id"]) for spec in SCENES[:2]}
    # Một mục cache cũ không còn run nào dùng, để thử nút dọn cache.
    stale = "nuscenes_scene-0061/prep.masks_segformer@0.1.0-0badf00d"
    _artifact(work.cache_dir / stale, "prep.masks_segformer", "masks", 790.0, stale)
    (work.cache_dir / stale / "payload.bin").write_bytes(RNG.randbytes(12 * 1024 * 1024))
    for spec in RUNS:
        spec["split"] = splits[spec["scene"]][spec.get("split_index", 0)]
        make_run(work, spec, caches[spec["scene"]], background)
        print(f"run {spec['run_id']}")
    print(f"xong: d2s studio --workdir {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
