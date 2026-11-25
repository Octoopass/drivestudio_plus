"""Measure the export gap of a static-background .ply against the model's own render (R3).

Renders the .ply written by ``export_ply.py --rigid_frame -1`` at every test image of the cameras
that were trained on, with the common 3DGS rasterizer (gsplat; 3DGUT for fisheye cameras), and
scores it on the static-background mask (valid & ~sky & ~dynamic). The model's own PSNR on the same
mask comes from ``frames.jsonl`` (key ``psnr_static``, protocol 0.5 of eval.standard), so the gap
``native - ply`` compares the same pixels. Writes ``ply_gap.json``.

    python -m drive2sim.recon.tools.eval_ply --model_dir <log_dir> --ply <scene.ply> \\
        --frames <frames.jsonl> --out <dir> data.data_root=<scene view>

What the gap contains: everything lost on export, i.e. pruned near-transparent Gaussians, no sky
(pixels under the sky mask are excluded anyway), no per-image colour affine, no rigid/SMPL/deformable
nodes (their pixels are under the dynamic mask), and any difference between the training renderer
and the common one. Camera pose refinement (CamPose) is applied to both renders.

Not run on a GPU in this repository's tests; ``load_ply_gaussians`` and ``camera_group`` are.
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
from plyfile import PlyData

logger = logging.getLogger()

GAP_FILE = "ply_gap.json"
MASK_DESCRIPTION = "valid & ~sky & ~dynamic"
FINAL_CHECKPOINT = "checkpoint_final.pth"


def load_ply_gaussians(path, device="cpu"):
    """Read an INRIA-layout 3DGS .ply (as written by export_ply.py) into activated tensors.

    Returns:
        dict with ``means`` (N,3), ``quats`` (N,4, normalised wxyz), ``scales`` (N,3, exp applied),
        ``opacities`` (N, sigmoid applied) and ``sh`` (N, K, 3) with K = (degree + 1) ** 2 SH
        coefficients (DC first), the layout ``gsplat.spherical_harmonics`` expects.
    """
    vertex = PlyData.read(path)["vertex"]

    def columns(prefix, count):
        return np.stack([np.asarray(vertex[f"{prefix}{i}"], dtype=np.float32) for i in range(count)], 1)

    names = vertex.data.dtype.names
    num_rest = sum(name.startswith("f_rest_") for name in names)
    f_dc = columns("f_dc_", 3)
    # f_rest is channel-major (all R coefficients, then G, then B): (N, 3 * K) -> (N, K, 3)
    f_rest = columns("f_rest_", num_rest).reshape(len(f_dc), 3, num_rest // 3).transpose(0, 2, 1)
    sh = np.concatenate([f_dc[:, None, :], f_rest], axis=1)
    means = np.stack([np.asarray(vertex[axis], dtype=np.float32) for axis in "xyz"], 1)
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float32)

    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    return {
        "means": tensor(means),
        "quats": torch.nn.functional.normalize(tensor(columns("rot_", 4)), dim=-1),
        "scales": torch.exp(tensor(columns("scale_", 3))),
        "opacities": torch.sigmoid(tensor(opacity_logit)),
        "sh": tensor(sh),
    }


def camera_group(camera_model):
    """Camera group used in reports and for the renderer: ``pinhole`` or ``fisheye`` (the rest)."""
    return "pinhole" if camera_model == "pinhole" else "fisheye"


def read_native_psnr(frames_path):
    """``{(frame_index, camera_index): psnr_static}`` of the model's own renders, ``None`` skipped."""
    native = {}
    with open(frames_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("psnr_static") is not None:
                native[(int(record["frame_index"]), int(record["camera_index"]))] = float(
                    record["psnr_static"]
                )
    return native


def renderer_mode(camera_model, has_distortion):
    """Rasterizer mode for the common renderer: ``default`` (3DGS), ``ut`` (3DGUT) or ``None``.

    Pinhole cameras use plain 3DGS and every other camera 3DGUT, whatever mode the model was
    trained with. A pinhole camera that carries lens-distortion coefficients (Waymo) cannot be
    drawn in gsplat's default mode, so ``None`` keeps the mode the model was trained with.
    """
    if camera_group(camera_model) == "fisheye":
        return "ut"
    return None if has_distortion else "default"


def render_ply_image(trainer, gaussians, image_infos, cam_infos):
    """Render the .ply Gaussians through the trainer's rasterizer settings for one test image.

    The mode comes from ``renderer_mode``. The .ply render touches no node model (no rigid, SMPL
    or deformable nodes), so there is no per-frame node state to set up.
    """
    import torch.nn.functional as F
    from gsplat import spherical_harmonics

    from drive2sim.recon.models.gaussians.basics import dataclass_gs

    cam_infos = dict(cam_infos)
    has_distortion = (
        cam_infos.get("radial_coeffs") is not None or cam_infos.get("tangential_coeffs") is not None
    )
    mode = renderer_mode(cam_infos.get("camera_model", "pinhole"), has_distortion)
    if mode is not None:
        cam_infos["render_mode"] = mode
    # same camera as the model's own render: CamPose, when trained, refines both
    cam = trainer.process_camera(cam_infos, image_infos["img_idx"].flatten()[0])
    viewdirs = F.normalize(gaussians["means"] - cam.camtoworlds[..., :3, 3], dim=-1)
    degree = int(round(gaussians["sh"].shape[1] ** 0.5)) - 1
    rgbs = torch.clamp(spherical_harmonics(degree, viewdirs, gaussians["sh"]) + 0.5, 0.0, 1.0)
    gs = dataclass_gs(
        _means=gaussians["means"],
        _scales=gaussians["scales"],
        _quats=gaussians["quats"],
        _rgbs=rgbs,
        _opacities=gaussians["opacities"][:, None],
        detach_keys=[],
    )
    results, _ = trainer.render_gaussians(
        gs,
        cam,
        near_plane=trainer.render_cfg.near_plane,
        far_plane=trainer.render_cfg.far_plane,
        render_mode="RGB",
        radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
    )
    return results["rgb_gaussians"]


def main(args):
    from omegaconf import OmegaConf

    from drive2sim.recon.data.driving_dataset import DrivingDataset
    from drive2sim.recon.rendering.color_match import export_gap
    from drive2sim.recon.rendering.videos import static_psnr
    from drive2sim.recon.utils.misc import import_str

    checkpoint = os.path.join(args.model_dir, FINAL_CHECKPOINT)
    cfg = OmegaConf.load(os.path.join(args.model_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)
    if dataset.test_image_set is None:
        raise SystemExit("run has no test image: nothing to measure")
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=checkpoint, load_only_model=True)
    trainer.set_eval()

    # camera_index in frames.jsonl is the manifest position, i.e. the entries of
    # data.pixel_source.cameras; test_only_cameras holds positions in that list.
    cameras = [int(c) for c in cfg.data.pixel_source.cameras]
    test_only = {cameras[p] for p in cfg.data.pixel_source.get("test_only_cameras", [])}
    native = read_native_psnr(args.frames)

    gaussians = load_ply_gaussians(args.ply, device)
    ply_psnr, groups = {}, {}
    downscale = trainer._get_downscale_factor()
    with torch.no_grad():
        for i in range(len(dataset.test_image_set)):
            image_infos, cam_infos = dataset.test_image_set.get_image(i, downscale)
            image_infos = {k: v.to(device) if torch.is_tensor(v) else v for k, v in image_infos.items()}
            cam_infos = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cam_infos.items()}
            camera_index = int(cam_infos["cam_id"].flatten()[0])
            if camera_index in test_only:
                continue  # the gap is measured on cameras the model was trained on
            rgb = render_ply_image(trainer, gaussians, image_infos, cam_infos)
            rgb = torch.round(rgb * 255.0) / 255.0  # 8-bit, like every other protocol metric
            psnr = static_psnr(
                rgb,
                image_infos["pixels"],
                image_infos.get("valid_masks"),
                image_infos.get("sky_masks"),
                image_infos.get("dynamic_masks"),
            )
            key = (int(image_infos["frame_idx"].flatten()[0]), camera_index)
            groups[key] = camera_group(cam_infos.get("camera_model", "pinhole"))
            if psnr is not None:
                ply_psnr[key] = psnr
            logger.info("frame %d camera %d: static PSNR %s", *key, psnr)

    result = export_gap(native, ply_psnr)
    if result is None:
        raise SystemExit("no test image has both a native and a .ply static PSNR")
    result["mask"] = MASK_DESCRIPTION
    result["per_group"] = {}
    for group in sorted(set(groups.values())):
        members = {key for key, name in groups.items() if name == group}
        part = export_gap(
            {k: v for k, v in native.items() if k in members},
            {k: v for k, v in ply_psnr.items() if k in members},
        )
        if part is not None:
            result["per_group"][group] = part
    result["ply_gaussians"] = int(len(gaussians["means"]))
    result["ply_size_mb"] = round(os.path.getsize(args.ply) / 2**20, 2)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, GAP_FILE), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export gap of a static-background .ply (R3)")
    parser.add_argument("--model_dir", required=True, help="train log dir with config.yaml and checkpoint_final.pth")
    parser.add_argument("--ply", required=True, help="static-background .ply from export_ply.py")
    parser.add_argument("--frames", required=True, help="frames.jsonl of eval.standard (protocol 0.5)")
    parser.add_argument("--out", required=True, help="directory for ply_gap.json")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. data.data_root=...")
    main(parser.parse_args())
