r"""Nghiệm thu tầng dữ liệu: loader gốc và đường dữ liệu của tool đưa cho model cùng dữ liệu.

SPEC mục 13, ARCHITECTURE mục 7.3. Thay cho so PSNR của hai lần train (có yếu tố ngẫu nhiên, tốn
giờ GPU, cần ngưỡng), script cho hai đường cùng dựng ``DrivingDataset`` trên **cùng một thư mục
cảnh** rồi so từng tensor mà loader giữ: pose, intrinsics, ảnh, mask, box, SMPL, điểm LiDAR, bản
đồ độ sâu, chia khung train/test, khung bao.

- **Đường mới:** ``config.yaml`` mà stage train của một run đã dùng (phần ``data``), loader trong
  ``drive2sim/recon``.
- **Đường cũ:** mã DriveStudio ở một ref git (mặc định ``main``, trước khi tái cấu trúc), lấy ra
  một worktree tạm; config dataset gốc của ref đó, cùng lựa chọn camera, split, khung, độ phân
  giải và cờ nạp như run.
- Cảnh: ``data/<scene_id>/`` cộng mask, SMPL trong cache của run (thư mục cảnh ảo, như stage
  train).

Hai đường cùng ``torch.manual_seed(0)``. Mọi khác biệt được in ra; khác biệt đã biết, cố ý, được
giải thích ở ``KNOWN``. Thoát 0 khi không còn khác biệt nào ngoài ``KNOWN``.

Chạy trong môi trường ``drive2sim``, trên run có cảnh và cache trên máy::

    python -m drive2sim.scripts.data_acceptance <run_id> [--ref main]

Đã chạy 20/11/2026 với run ``28e5`` (nuScenes ``scene-0061``, ``front3``): 74 mảng, 71 giống từng
bit; khác ở 3 bản đồ độ sâu LiDAR (``KNOWN["lidar_depth_maps"]``): cùng tập pixel, khác 25–187
pixel trên khoảng 600 000, đường mới gần hơn ở mọi pixel đó (trung vị 12–16 m).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

# Khác biệt cố ý giữa đường cũ và đường mới: tên mảng chứa khoá → hàm kiểm (cũ, mới) → lý do.
KNOWN = {
    "lidar_depth_maps": (
        "nhiều điểm LiDAR rơi vào một pixel thì giữ điểm gần nhất (nearest_depth_map, commit "
        "91e23a7); DriveStudio giữ một điểm bất kỳ, thường là điểm bị vật phía trước che"
    ),
}
SMALL = 2_000_000
# Khoá config người dùng chọn (split, camera, cờ nạp); đường cũ lấy theo run, phần còn lại theo
# config dataset gốc của ref.
USER_KEYS = ("scene_idx", "start_timestep", "end_timestep")
USER_PIXEL_KEYS = (
    "cameras",
    "test_image_stride",
    "load_sky_mask",
    "load_dynamic_mask",
    "load_objects",
    "load_smpl",
    "downscale_when_loading",
    "downscale",
    "undistort",
)


def dump(which: str, config: Path, out: Path) -> None:
    """Dựng ``DrivingDataset`` rồi ghi tensor (``old``: mã ở thư mục hiện tại; ``new``: tool)."""
    import torch
    from omegaconf import OmegaConf

    if which == "old":
        sys.path.insert(0, str(Path.cwd()))
        from datasets.driving_dataset import DrivingDataset  # type: ignore[import-not-found]
    else:
        from drive2sim.recon.data.driving_dataset import DrivingDataset
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = DrivingDataset(data_cfg=OmegaConf.create(yaml.safe_load(config.read_text())))
    record: dict[str, np.ndarray] = {}

    def put(key: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu()
            array = (
                array.float().numpy()
                if array.dtype in (torch.float16, torch.bfloat16)
                else array.numpy()
            )
        elif isinstance(value, np.ndarray):
            array = value
        else:
            return
        if array.dtype == object:
            return
        record[key + "::shape"] = np.array(array.shape)
        if array.size <= SMALL:
            record[key] = array
        else:
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()
            record[key + "::sha"] = np.frombuffer(digest, np.uint8)
            record[key + "::mean"] = np.array(array.astype(np.float64).mean())

    def walk(prefix: str, obj: object, depth: int = 0) -> None:
        if depth > 2:
            return
        for name, value in vars(obj).items():
            key = f"{prefix}.{name}"
            if isinstance(value, (torch.Tensor, np.ndarray)):
                put(key, value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        put(f"{key}[{k}]", v)
                    elif hasattr(v, "__dict__") and depth < 2:
                        walk(f"{key}[{k}]", v, depth + 1)
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in value)
            ):
                for i, v in enumerate(value):
                    put(f"{key}[{i}]", v)

    pixel = dataset.pixel_source
    walk("pixel", pixel)
    # DriveStudio gốc khai camera_data là thuộc tính lớp, vars() không thấy.
    if "camera_data" not in vars(pixel):
        for k, v in pixel.camera_data.items():
            walk(f"pixel.camera_data[{k}]", v, 1)
    if dataset.lidar_source is not None:
        walk("lidar", dataset.lidar_source)
    for name in ("train_timesteps", "test_timesteps", "train_indices", "test_indices", "aabb"):
        value = getattr(dataset, name, None)
        if value is not None:
            put(f"dataset.{name}", value if isinstance(value, torch.Tensor) else np.asarray(value))
    # Bản đồ độ sâu LiDAR dạng thưa, để giải thích khác biệt từng pixel.
    for k, cam in pixel.camera_data.items():
        if cam.lidar_depth_maps is not None:
            depth = cam.lidar_depth_maps.detach().cpu().numpy().ravel()
            nonzero = np.flatnonzero(depth)
            record[f"sparse_depth[{k}]::idx"] = nonzero
            record[f"sparse_depth[{k}]::val"] = depth[nonzero]
    np.savez_compressed(out, **record)


def compare(old_path: Path, new_path: Path) -> list[str]:
    """In bảng so; trả các khác biệt không có trong ``KNOWN``."""
    a, b = np.load(old_path), np.load(new_path)
    keys = sorted(
        {k.split("::")[0] for k in a.files if not k.startswith("sparse_depth")}
        | {k.split("::")[0] for k in b.files if not k.startswith("sparse_depth")}
    )
    same, unexplained = 0, []
    for key in keys:
        has_a = any(f == key or f.startswith(key + "::") for f in a.files)
        has_b = any(f == key or f.startswith(key + "::") for f in b.files)
        if not (has_a and has_b):
            note = f"{key}: chỉ có ở đường {'cũ' if has_a else 'mới'}"
        elif tuple(a[key + "::shape"]) != tuple(b[key + "::shape"]):
            shapes = tuple(a[key + "::shape"]), tuple(b[key + "::shape"])
            note = f"{key}: khác kích thước {shapes[0]} và {shapes[1]}"
        elif key + "::sha" in a.files:
            if bytes(a[key + "::sha"]) == bytes(b[key + "::sha"]):
                same += 1
                continue
            means = float(a[key + "::mean"]), float(b[key + "::mean"])
            note = f"{key}: khác (trung bình {means[0]:.6g} và {means[1]:.6g})"
        else:
            x, y = a[key], b[key]
            if x.dtype.kind in "USbiu" or y.dtype.kind in "USbiu":
                equal = np.array_equal(x, y)
            else:
                equal = x.size == 0 or np.array_equal(
                    x.astype(np.float64), y.astype(np.float64), equal_nan=True
                )
            if equal:
                same += 1
                continue
            note = f"{key}: khác"
        reason = next((why for name, why in KNOWN.items() if name in key), None)
        print(f"  {note}" + (f"\n      đã biết: {reason}" if reason else "  <- CHƯA GIẢI THÍCH"))
        if reason is None:
            unexplained.append(note)
    print(
        f"{len(keys)} mảng: {same} giống từng bit, {len(keys) - same} khác, "
        f"{len(unexplained)} chưa giải thích"
    )
    for name in sorted(f for f in a.files if f.endswith("::idx")):
        cam = name[len("sparse_depth[") : name.index("]")]
        ia, va = a[name], a[name.replace("::idx", "::val")]
        ib, vb = b[name], b[name.replace("::idx", "::val")]
        if not np.array_equal(ia, ib):
            print(
                f"  độ sâu LiDAR camera {cam}: tập pixel có điểm khác nhau "
                f"({len(ia)} và {len(ib)})"
            )
            unexplained.append(f"sparse_depth[{cam}]: tập pixel")
            continue
        diff = va != vb
        nearer = bool(np.all(vb[diff] < va[diff]))
        print(
            f"  độ sâu LiDAR camera {cam}: cùng {len(ia)} pixel có điểm; "
            f"{int(diff.sum())} pixel khác "
            f"giá trị, đường mới gần hơn ở mọi pixel đó: {nearer}"
        )
        if not nearer:
            unexplained.append(f"sparse_depth[{cam}]: có pixel đường mới không gần hơn")
    return unexplained


def configs(run_dir: Path, worktree: Path, view: Path, out: Path) -> tuple[Path, Path]:
    """Ghi ``new.yaml`` (data của run) và ``old.yaml`` (config dataset gốc + lựa chọn của run)."""
    train = next(
        p for p in (run_dir / "d2s" / "stages").glob("train.*") if (p / "config.yaml").is_file()
    )
    full = yaml.safe_load((train / "config.yaml").read_text())
    new = full["data"]
    new["data_root"], new["preload_device"] = str(view), "cpu"
    record = json.loads((run_dir / "d2s" / "run.json").read_text())
    dataset_name = next(s for s in record["stages"] if s["stage_id"] == train.name)
    dataset = (dataset_name.get("params") or {}).get("dataset")
    if not dataset:
        num = len(new["pixel_source"]["cameras"])
        dataset = f"{new['dataset']}/{num}cams"
    old = yaml.safe_load((worktree / "configs" / "datasets" / f"{dataset}.yaml").read_text())[
        "data"
    ]
    old["data_root"], old["preload_device"] = str(view), "cpu"
    for key in USER_KEYS:
        old[key] = new[key]
    for key in USER_PIXEL_KEYS:
        if key in new["pixel_source"]:
            old["pixel_source"][key] = new["pixel_source"][key]
    paths = out / "old.yaml", out / "new.yaml"
    paths[0].write_text(yaml.safe_dump(old))
    paths[1].write_text(yaml.safe_dump(new))
    return paths


def main() -> None:
    """Dựng hai đường cho một run, so, thoát 1 nếu có khác biệt chưa giải thích."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--ref", default="main", help="ref git của mã DriveStudio gốc")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument(
        "--dump", nargs=3, metavar=("WHICH", "CONFIG", "OUT"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.dump:
        dump(args.dump[0], Path(args.dump[1]), Path(args.dump[2]))
        return
    from drive2sim.core import files
    from drive2sim.core.schema import RunRecord
    from drive2sim.engine.workdir import Workdir
    from drive2sim.stages.scene_view import build_view

    work = Workdir(args.workdir.resolve())
    run_dir = work.find_run(args.run_id)
    record = files.load(run_dir / "d2s" / "run.json", RunRecord)
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="d2s-accept-") as tmp:
        out = Path(tmp)
        worktree = out / "ref"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(worktree), args.ref],
            check=True,
        )
        try:
            extra = {}
            for stage in record.stages:
                if stage.cache_key:
                    folder = work.cache_dir / stage.cache_key
                    for name in ("sky_masks", "fine_dynamic_masks", "humanpose"):
                        if (folder / name).exists():
                            extra[name] = folder / name
            build_view(
                work.scene_dir(record.scene_id), out / "view", exclude=tuple(extra), extra=extra
            )
            old_cfg, new_cfg = configs(run_dir, worktree, out / "view", out)
            script = str(Path(__file__).resolve())
            subprocess.run(
                [sys.executable, script, "--dump", "old", str(old_cfg), str(out / "old.npz")],
                cwd=worktree,
                check=True,
            )
            subprocess.run(
                [sys.executable, script, "--dump", "new", str(new_cfg), str(out / "new.npz")],
                cwd=repo,
                check=True,
            )
            print(
                f"run {record.run_id}, cảnh {record.scene_id}, split {record.split_id}, "
                f"ref {args.ref}:"
            )
            unexplained = compare(out / "old.npz", out / "new.npz")
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                check=False,
            )
    sys.exit(1 if unexplained else 0)


if __name__ == "__main__":
    main()
