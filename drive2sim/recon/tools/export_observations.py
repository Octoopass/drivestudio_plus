"""drive2sim: xuất quan sát mà đường dữ liệu DriveStudio đưa cho trainer ra file thường.

Stage ``prep.observations`` chạy công cụ này ở môi trường chính với đúng tham số của
``drive2sim.recon.tools.train`` (config method, config dataset, đè trên dòng lệnh), nên
``DrivingDataset`` dựng ở đây là dữ liệu OmniRe, StreetGS... train và được chấm trên đó. Approach
chạy ở môi trường khác (XSIM) đọc các file này thay vì đọc lại cảnh gốc, nên đầu vào giống hệt
approach DriveStudio (CONTRACTS mục 5.4, loại ``observations``):

- ``images/<khung>_<camera>.png``: RGB đúng như loader giữ (PNG 8 bit; hard link tới file của
  cảnh khi pixel trùng);
- ``masks/train_<camera>.png``: pixel trainer được giám sát (ngược của ``egocar_masks``: vành mắt
  cá, viền tối, thân xe); ``masks/eval_<camera>.png``: pixel được chấm (``valid_masks``), không có
  khi loader chấm cả ảnh;
- ``sky/<khung>_<camera>.png``: mask bầu trời; ``cameras.npz``: camera → world từng khung;
  intrinsics, hệ số méo ở độ phân giải đã nạp nằm trong ``observations.json``;
- ``lidar/<khung>.npz``: điểm world, gốc tia, pose cảm biến của mỗi lượt quét loader giữ;
  ``instances.npz``: box trong world; ``smpl.npz``: phép quay khớp SMPL của người;
- ``observations.json``: chỉ mục (định dạng, camera, khung, timestamp, split, file nào có).

Khung là chỉ số khung của cảnh; pose ở hệ world của loader (pose xe ở khung đầu được nạp).
Cách dùng::

    python -m drive2sim.recon.tools.export_observations --config_file <method.yaml> \
        --out <thư mục> --scene_dir <cảnh> --index_json <file> [đè config...]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from drive2sim.core.config import load_config
from drive2sim.paths import CONFIGS_DIR
from drive2sim.recon.data.driving_dataset import DrivingDataset

FORMAT = "d2s-observations"
VERSION = 1


def build_config(config_file: str, opts: list) -> OmegaConf:
    """Merge method config, dataset config and overrides like ``recon.tools.train.setup``."""
    cfg = OmegaConf.create(load_config(config_file))
    cli = OmegaConf.from_cli(opts)
    if "dataset" in cli:
        cfg.dataset = cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create(load_config(os.path.join(CONFIGS_DIR, "datasets", f"{dataset_type}.yaml")))
        )
    return OmegaConf.merge(cfg, cli)


def _uint8(image: torch.Tensor) -> np.ndarray:
    """Float image in [0, 1] → uint8, refusing values that are not k/255 (lossy export)."""
    array = image.detach().cpu().float().numpy()
    scaled = np.round(array * 255.0)
    if np.abs(array * 255.0 - scaled).max() > 1e-3:
        raise ValueError("loader image is not 8-bit (runtime downscale != 1); export would be lossy")
    return scaled.clip(0, 255).astype(np.uint8)


def _write_image(pixels: np.ndarray, path: Path, source: str) -> str:
    """Write a PNG, or hard-link the scene's PNG when its decoded pixels are identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if source and source.lower().endswith(".png") and os.path.isfile(source):
        with Image.open(source) as img:
            original = np.asarray(img.convert("RGB"))
        if original.shape == pixels.shape and np.array_equal(original, pixels):
            try:
                # os.link trên Linux không đi theo symlink ở thành phần cuối: link file thật,
                # không thì cache giữ một symlink trỏ ra ngoài.
                os.link(os.path.realpath(source), path)
                return "link"
            except OSError:
                pass
    Image.fromarray(pixels).save(path, compress_level=1)
    return "png"


def _mask_png(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255).save(path)


def _static(masks: list, name: str) -> torch.Tensor:
    first = masks[0]
    for mask in masks[1:]:
        if not torch.equal(mask, first):
            raise ValueError(f"{name} changes between frames; observations format needs it static")
    return first


def export_cameras(dataset: DrivingDataset, camera_ids: list, frames: list, out: Path) -> list:
    """Images, masks and calibration of every loaded camera and frame."""
    pixel = dataset.pixel_source
    cameras, cam_to_world, links = [], {}, 0
    for index, cam in pixel.camera_data.items():
        model = getattr(cam, "camera_model", "pinhole")
        train_masks, eval_masks, intrinsics = [], [], None
        poses = []
        for t in range(len(frames)):
            image, info = cam.get_image(t)
            pixels = _uint8(image["pixels"])
            source = str(cam.img_filepaths[t]) if hasattr(cam, "img_filepaths") else ""
            name = f"{frames[t]:03d}_{index}.png"
            links += _write_image(pixels, out / "images" / name, source) == "link"
            if image.get("sky_masks") is not None:
                _mask_png(image["sky_masks"], out / "sky" / name)
            height, width = pixels.shape[:2]
            if "egocar_masks" in image and image["egocar_masks"] is not None:
                train_masks.append(image["egocar_masks"].detach().cpu() < 0.5)
            if "valid_masks" in image:
                eval_masks.append(image["valid_masks"].detach().cpu().bool())
            k = info["intrinsics"].detach().cpu().double().numpy()
            if intrinsics is not None and not np.allclose(k, intrinsics):
                raise ValueError(f"camera {index}: intrinsics change between frames")
            intrinsics = k
            poses.append(info["camera_to_world"].detach().cpu().double().numpy())
        if train_masks:
            _mask_png(_static(train_masks, "train mask"), out / "masks" / f"train_{index}.png")
        if eval_masks:
            _mask_png(_static(eval_masks, "eval mask"), out / "masks" / f"eval_{index}.png")
        if model == "fisheye":
            distortion = [float(v) for v in torch.as_tensor(cam.radial_coeffs).flatten()[:4]]
        elif getattr(cam, "undistort", False) or getattr(cam, "distortions", None) is None:
            distortion = [0.0] * 5
        else:
            distortion = [float(v) for v in cam.distortions[0].flatten()[:5]]
        cam_to_world[str(index)] = np.stack(poses)
        cameras.append(
            {
                "index": int(index),
                "camera_id": camera_ids[int(index)],
                "model": model,
                "width": int(width),
                "height": int(height),
                "intrinsics": [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
                "distortion": distortion,
                "train_mask": bool(train_masks),
                "eval_mask": bool(eval_masks),
            }
        )
    np.savez(out / "cameras.npz", **cam_to_world)
    print(f"images: {len(cameras)} cameras x {len(frames)} frames, {links} hard links")
    return cameras


def export_lidar(dataset: DrivingDataset, frames: list, out: Path) -> list:
    """World points, origin and sensor pose of each sweep the loader keeps."""
    lidar = dataset.lidar_source
    if lidar is None:
        return []
    (out / "lidar").mkdir(parents=True, exist_ok=True)
    timesteps = lidar.timesteps.cpu()
    points = (lidar.origins + lidar.directions * lidar.ranges).cpu()
    origins = lidar.origins.cpu()
    written = []
    for t, frame in enumerate(frames):
        select = timesteps == t
        if not select.any():
            continue
        np.savez(
            out / "lidar" / f"{frame:03d}.npz",
            points=points[select].numpy().astype(np.float32),
            origins=origins[select].numpy().astype(np.float32),
            sensor_to_world=lidar.lidar_to_worlds[t].cpu().double().numpy(),
        )
        written.append(frame)
    print(f"lidar: {len(written)} sweeps, {int(len(timesteps))} points")
    return written


def export_instances(dataset: DrivingDataset, out: Path) -> bool:
    """Boxes as the loader holds them: pose per frame, mean size, visibility, node type."""
    pixel = dataset.pixel_source
    if getattr(pixel, "instances_pose", None) is None:
        return False
    np.savez(
        out / "instances.npz",
        poses=pixel.instances_pose.cpu().double().numpy(),
        sizes=pixel.instances_size.cpu().double().numpy(),
        visible=pixel.per_frame_instance_mask.cpu().numpy().astype(bool),
        model_types=pixel.instances_model_types.cpu().numpy().astype(np.int64),
        true_ids=pixel.instances_true_id.cpu().numpy().astype(np.int64),
    )
    return True


def export_smpl(dataset: DrivingDataset, out: Path) -> bool:
    """SMPL joint rotations (world root, local body) as matrices, per instance column."""
    from pytorch3d.transforms import quaternion_to_matrix

    pixel = dataset.pixel_source
    humans = getattr(pixel, "smpl_human_all", None)
    if not humans:
        return False
    true_ids = pixel.instances_true_id.cpu().tolist()
    columns, rotations, betas, valid = [], [], [], []
    for instance_id, data in sorted(humans.items(), key=lambda item: int(item[0])):
        if int(instance_id) not in true_ids:
            continue
        columns.append(true_ids.index(int(instance_id)))
        rotations.append(quaternion_to_matrix(data["smpl_quats"]).numpy())
        betas.append(data["smpl_betas"].numpy())
        valid.append(data["frame_valid"].numpy())
    if not columns:
        return False
    np.savez(
        out / "smpl.npz",
        columns=np.array(columns, dtype=np.int64),
        rotations=np.stack(rotations).astype(np.float32),
        betas=np.stack(betas).astype(np.float32),
        valid=np.stack(valid).astype(bool),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scene_dir", required=True, type=Path)
    parser.add_argument("--index_json", required=True, type=Path)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = build_config(args.config_file, args.opts)
    pixel_cfg = cfg.data.pixel_source
    if float(pixel_cfg.get("downscale", 1)) != 1:
        raise ValueError("observations need data.pixel_source.downscale = 1 (use downscale_when_loading)")
    torch.manual_seed(0)
    np.random.seed(0)
    cfg.data.preload_device = "cpu"
    dataset = DrivingDataset(data_cfg=cfg.data)
    index = json.loads(args.index_json.read_text())
    first = int(cfg.data.start_timestep)
    frames = list(range(first, first + dataset.pixel_source.num_timesteps))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cameras = export_cameras(dataset, index["camera_ids"], frames, out)
    lidar_frames = export_lidar(dataset, frames, out)
    has_instances = export_instances(dataset, out)
    has_smpl = export_smpl(dataset, out)
    stamps = [int(v) for v in (args.scene_dir / "timestamps.txt").read_text().split()]
    source_ids = [int(v) for v in (args.scene_dir / "frame_ids.txt").read_text().split()]
    test = sorted(int(t) + first for t in np.asarray(dataset.test_timesteps).tolist())
    payload = {
        "format": FORMAT,
        "version": VERSION,
        **{k: v for k, v in index.items() if k != "camera_ids"},
        "frames": frames,
        "source_frames": [source_ids[f] for f in frames],
        "timestamps_us": [stamps[f] for f in frames],
        "loader_test_frames": test,
        "cameras": cameras,
        "lidar_frames": lidar_frames,
        "instances": has_instances,
        "smpl": has_smpl,
        "data_config": OmegaConf.to_container(cfg.data, resolve=True),
    }
    (out / "observations.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
