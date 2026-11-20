r"""Đo chất lượng ảnh tái dựng của camera mắt cá KITTI-360 theo dải góc lệch trục.

Render ảnh mắt cá đầy đủ (không che vành ngoài) bằng renderer của run, ở pose camera gốc của các
khung test, rồi so với ảnh gốc trong file tar KITTI-360 trên máy, tính PSNR riêng cho từng dải góc
(bỏ pixel thân xe). Dùng để xem renderer có vẽ hỏng gần 90 độ không, và so các run train với ngưỡng
che khác nhau (``train.drivestudio.fisheye_max_angle``; ``docs/datasets/kitti360.md`` mục 5.1).

Chạy trong môi trường GPU (cần PyTorch, gsplat, checkpoint của run, pose và calib của cảnh
trên máy):

    python -m drive2sim.scripts.fisheye_angle_bands <run_id> [<run_id> ...] \
        --raw data/kitti360/raw --camera image_02 [--save <thư mục ảnh>]

Khung test không có trong file tar trên máy thì bỏ qua.
"""

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image

from drive2sim.engine.workdir import Workdir
from drive2sim.recon.tools.frame_server import load
from drive2sim.recon.tools.viewer import (
    GsplatRenderTabState,
    _dataset_camera_tools,
    _initial_render_settings_from_trainer,
    _set_frame_on_trainer,
    make_viewer_render_fn,
)
from drive2sim.studio.api.gpu_viewer import prepare

BANDS = [(0, 60), (60, 75), (75, 80), (80, 83), (83, 86), (86, 88), (88, 90), (90, 93), (93, 96)]


class _Cam:
    """Camera cho ``make_viewer_render_fn``: pose ``c2w`` và K cố định."""

    def __init__(self, c2w: np.ndarray, K: np.ndarray) -> None:
        self.c2w, self._K = c2w, K

    def get_K(self, _size: tuple) -> np.ndarray:
        return self._K


def angle_map(K: np.ndarray, coeffs: list[float], width: int, height: int) -> np.ndarray:
    """Góc lệch trục (độ) của từng pixel theo OpenCV fisheye, giải ngược bằng bảng tra."""
    v, u = np.mgrid[0:height, 0:width]
    rd = np.hypot((u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1])
    theta = np.linspace(0, 1.98, 20000)
    poly = 1 + sum(k * theta ** (2 * (i + 1)) for i, k in enumerate(coeffs))
    return np.degrees(np.interp(rd, theta * poly, theta))


def measure(run_id: str, raw: Path, camera: str, save: Optional[Path]) -> dict:
    """PSNR theo dải góc của một run."""
    work = Workdir(Path(".").resolve())
    run = work.find_run(run_id)
    record = json.loads((run / "d2s/run.json").read_text())
    scene_dir = work.scene_dir(record["scene_id"])
    manifest = yaml.safe_load((scene_dir / "manifest.yaml").read_text())
    index = [c["camera_id"] for c in manifest["cameras"]].index(camera)
    coeffs = list(np.loadtxt(scene_dir / "intrinsics" / f"{index}.txt")[4:8])
    raw_ids = [int(x) for x in (scene_dir / "frame_ids.txt").read_text().split()]
    split = json.loads((scene_dir / "splits" / f"{record['split_id']}.json").read_text())
    stride = int(str(split["strategy"]).split(":")[-1])
    train = next(
        p for p in (run / "d2s/stages").glob("train.*") if (p / "checkpoint_final.pth").is_file()
    )
    params = next(s for s in record["stages"] if s["stage_id"] == train.name)["params"]

    ckpt, config, _ = prepare(work, run)
    trainer, dataset, frames, device = load(str(ckpt), str(config))
    tools = _dataset_camera_tools(dataset)
    option = next(o for o in tools["camera_snap_options"] if o.endswith(camera))
    frame = {"value": 0}
    render_fn = make_viewer_render_fn(
        trainer,
        device,
        frames,
        frame_getter=lambda: frame["value"],
        image_context_getter=tools["image_context_getter"],
    )
    state = GsplatRenderTabState()
    for name, value in _initial_render_settings_from_trainer(trainer).items():
        setattr(state, name, value)
    state.selected_gaussian_types = tuple(trainer.gaussian_classes.keys())
    state.background_mode, state.render_mode, state.preview_render = "sky", "rgb", False
    state.camera_model = "fisheye"
    for i, value in enumerate(coeffs):
        setattr(state, f"radial_k{i + 1}", float(value))

    cam = next(
        c
        for c in json.loads((train / "cameras.json").read_text())["cameras"]
        if c["name"] == camera
    )
    width, height = cam["width"], cam["height"]
    state.viewer_width, state.viewer_height = width, height
    theta = angle_map(np.array(cam["K"][0]).reshape(3, 3), coeffs, width, height)
    ego_path = Path(f"drive2sim/assets/ego_masks/kitti360/{camera}.png")
    ego = np.asarray(Image.open(ego_path).convert("L").resize((width, height), Image.NEAREST)) > 0

    tar = tarfile.open(raw / f"data_2d_raw/{manifest['source']['sequence_id']}_{camera}.tar")
    members = {Path(m.name).name: m for m in tar.getmembers() if m.name.endswith(".png")}
    errors = {band: [] for band in BANDS}
    for f in range(0, frames, stride):
        name = f"{raw_ids[f]:010d}.png"
        if name not in members:
            continue
        image = Image.open(io.BytesIO(tar.extractfile(members[name]).read())).convert("RGB")
        gt = np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255
        frame["value"] = f
        tools["camera_snap_setter"](option)
        _set_frame_on_trainer(trainer, f)
        c2w = np.array(cam["c2w"][f], dtype=np.float32).reshape(4, 4)
        K = np.array(cam["K"][f], dtype=np.float32).reshape(3, 3)
        with torch.no_grad():
            render = np.clip(render_fn(_Cam(c2w, K), state), 0, 1)
        if save is not None and f == stride * 2:
            save.mkdir(parents=True, exist_ok=True)
            Image.fromarray((render * 255).astype(np.uint8)).save(
                save / f"{run_id}_{camera}_{f:03d}.png"
            )
            Image.fromarray((gt * 255).astype(np.uint8)).save(save / f"gt_{camera}_{f:03d}.png")
        squared = ((render - gt) ** 2).mean(-1)
        for low, high in BANDS:
            mask = (theta > low) & (theta <= high) & ~ego
            if mask.any():
                errors[(low, high)].append(float(squared[mask].mean()))
    del trainer
    torch.cuda.empty_cache()
    return {
        "renderer": record["approach"]["options"].get("renderer"),
        "max_angle": params.get("fisheye_max_angle"),
        "frames": len(errors[BANDS[0]]),
        "psnr": {
            f"{low}-{high}": 10 * np.log10(1 / np.mean(values)) if values else None
            for (low, high), values in errors.items()
        },
    }


def main() -> None:
    """Đo các run trên dòng lệnh rồi in bảng PSNR theo dải góc."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--raw", type=Path, default=Path("data/kitti360/raw"))
    parser.add_argument("--camera", default="image_02")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    results = {
        run_id: measure(run_id, args.raw, args.camera, args.save) for run_id in args.run_ids
    }
    print("dải góc   " + "".join(f"{r[-4:]:>10}" for r in results))
    for low, high in BANDS:
        key = f"{low}-{high}"
        cells = "".join(
            f"{res['psnr'][key]:10.2f}" if res["psnr"][key] is not None else f"{'-':>10}"
            for res in results.values()
        )
        print(f"{key:>7}°  {cells}")
    print(
        json.dumps(
            {r: {k: v for k, v in res.items() if k != "psnr"} for r, res in results.items()}
        )
    )


if __name__ == "__main__":
    main()
