"""Xuất gói xem trước cho Spark từ checkpoint DriveStudio (CONTRACTS mục 7).

Khác ``export_ply.py`` (gộp mọi thứ vào một ``scene.ply`` ở một khung), gói này tách phần tĩnh
và phần động, để trình duyệt tự đặt phần động theo khung (CONTRACTS mục 7):
- ``background.spz``: Gaussian của nền tĩnh, hệ world (SPZ: nhỏ hơn PLY khoảng 11 lần);
- ``actors/<track_id>.spz``: Gaussian của từng xe (``RigidNodes``) trong hệ của xe;
- ``actor_poses.json``: pose của từng xe theo khung;
- ``humans/<track_id>.spz`` và ``.skin``: Gaussian của từng người (``SMPLNodes``) ở dáng gốc, và
  4 xương ảnh hưởng nhất của mỗi Gaussian kèm trọng số;
- ``human_bones.bin``, ``human_poses.json``: ma trận 24 xương và vị trí của từng người theo khung;
  trình duyệt uốn người bằng ``SplatSkinning`` của Spark (cộng có trọng số chuyển động các xương,
  như ``SMPLNodes.transform_means_and_quats``);
- ``sky_cube.jpg``: cubemap trời (``EnvLight``), 6 mặt xếp 3 cột × 2 hàng;
- ``preview.json``: gói có gì và thiếu gì.

Vật biến dạng (``DeformableNodes``) chưa có Gaussian trong gói, chỉ được liệt kê trong
``box_only_actors``: chuyển động của chúng do một mạng nơ-ron tính, trình duyệt không chạy lại.
Màu theo góc nhìn (SH bậc cao) của xe giữ trong hệ của xe; viewer xoay cả đối tượng nên không cần
xoay SH.

    python -m drive2sim.recon.tools.export_preview --ckpt <log_dir>/checkpoint_final.pth \\
        --out <dir> --source_run runs/<scene>/<approach>/<run_id>

Chưa chạy thật: cần checkpoint thật và PyTorch.
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from drive2sim.recon.tools.export_ply import (
    CAMERAS_FILE,
    gaussians_from_state,
    quat_to_mat,
    up_axis_from_cameras,
)
from drive2sim.recon.tools.spz import write_spz

SCHEMA_VERSION = "d2s-preview/0.2"
SKY_FILE = "sky_cube.jpg"
NUM_BONES = 24  # khớp của SMPL
SKIN_BONES = 4  # Spark giữ tối đa 4 xương mỗi Gaussian


def _filter(gaussians: dict, keep: torch.Tensor) -> dict:
    """Giữ các Gaussian có ``keep`` đúng."""
    return {key: value[keep] for key, value in gaussians.items()}


def _visible(gaussians: dict, min_opacity: float) -> torch.Tensor:
    """Gaussian có độ mờ đục lớn hơn ``min_opacity`` (opacity lưu dạng logit)."""
    return torch.sigmoid(gaussians["opac"][:, 0]) > min_opacity


def _poses(state: dict) -> tuple[list[str], list[list]]:
    """Pose của từng xe theo khung.

    Returns:
        ``(track_ids, poses)`` với ``poses[f][k]`` là ma trận 4x4 (16 số, theo hàng) đưa điểm từ
        hệ của xe ``k`` sang world ở khung ``f``, hoặc ``None`` khi xe không có mặt.
    """
    quats = torch.nn.functional.normalize(state["instances_quats"].float(), dim=-1)  # (F,K,4)
    trans = state["instances_trans"].float()  # (F,K,3)
    present = state["instances_fv"].bool()  # (F,K)
    num_frames, num_actors = present.shape
    matrices = torch.eye(4).repeat(num_frames, num_actors, 1, 1)
    matrices[..., :3, :3] = quat_to_mat(quats)
    matrices[..., :3, 3] = trans
    poses = [
        [matrices[f, k].reshape(-1).tolist() if present[f, k] else None for k in range(num_actors)]
        for f in range(num_frames)
    ]
    return [f"rigid_{k:04d}" for k in range(num_actors)], poses


def _human_weights(state: dict) -> torch.Tensor:
    """Trọng số bám xương (H, 6890, 24) của từng Gaussian người ở dáng gốc.

    Như ``SMPLTemplate.forward``: có ``voxel_deformer`` thì lấy mẫu lưới trọng số tại vị trí gốc
    (``VoxelDeformer.forward``), không thì dùng trọng số SMPL ``template.W``. Vị trí gốc không đổi
    sau khi train nên trọng số tính một lần.
    """
    num_humans = state["instances_fv"].shape[1]
    means = state["_means"].float().reshape(num_humans, -1, 3)
    prefix = "template.voxel_deformer."
    if prefix + "lbs_voxel_base" not in state:
        return state["template.W"].float()
    grid = state[prefix + "lbs_voxel_base"].float()
    if prefix + "voxel_w_correction" in state:
        grid = grid + state[prefix + "voxel_w_correction"].float()
    x = (means - state[prefix + "offset"].float()) / state[prefix + "scale"].float()
    x[..., -1] = x[..., -1] * state[prefix + "ratio"].float()  # short_dim_dhw = 0 (trục d là z)
    weights = torch.nn.functional.grid_sample(
        grid, x[:, :, None, None], align_corners=True, mode="bilinear", padding_mode="border"
    )
    return weights.squeeze(-1).squeeze(-1).permute(0, 2, 1)[..., :NUM_BONES]


def _skin(weights: torch.Tensor) -> np.ndarray:
    """Gói trọng số (N, 24) thành dữ liệu skin của Spark: 4 số uint16 mỗi Gaussian.

    Mỗi số là ``trọng số 8 bit | (chỉ số xương << 8)``, đúng như ``SplatSkinning.setSplatBones``.
    Giữ 4 xương có trọng số lớn nhất rồi nhân để tổng bằng tổng gốc: trọng số của OmniRe không
    chuẩn hoá (tổng 0,83–1,21 ở run ``f8a6``), nên giữ tổng thì vị trí lệch ít nhất.
    """
    values, bones = weights.topk(SKIN_BONES, dim=1)
    values = values.clamp(min=0)
    kept = values.sum(1, keepdim=True).clamp(min=1e-8)
    values = (values * weights.sum(1, keepdim=True) / kept).clamp(0, 1)
    packed = torch.round(values * 255).to(torch.int64) | (bones.to(torch.int64) << 8)
    return packed.numpy().astype("<u2")


def _bones(state: dict) -> torch.Tensor:
    """Ma trận 24 xương (F, H, 24, 4, 4) của từng người theo khung, hệ gốc của người.

    Như ``SMPLTemplate.forward``: góc khớp (quaternion của cả người rồi 23 khớp) qua
    ``batch_rigid_transform`` của smplx, nhân ``A0_inv`` để dáng gốc là ma trận đơn vị. Chỉ cần
    các tensor trong checkpoint, không cần file mô hình SMPL.
    """
    from smplx.lbs import batch_rigid_transform

    theta = torch.cat([state["instances_quats"], state["smpl_qauts"]], dim=2).float()
    frames, humans = theta.shape[:2]
    rot = quat_to_mat(torch.nn.functional.normalize(theta, dim=-1)).reshape(
        frames * humans, NUM_BONES, 3, 3
    )
    joints = state["template.J_canonical"].float()[None].expand(frames, -1, -1, -1)
    parents = state["template._template_layer.parents"].long()
    _, transforms = batch_rigid_transform(rot, joints.reshape(-1, NUM_BONES, 3), parents)
    transforms = transforms.reshape(frames, humans, NUM_BONES, 4, 4)
    return transforms @ state["template.A0_inv"].float()[None]


def _export_humans(state: dict, out: str, min_opacity: float) -> tuple[list[str], list[str]]:
    """Ghi Gaussian, skin, xương và vị trí của từng người; trả (mọi track_id, track_id có file).

    ``human_bones.bin``: float32 little-endian (F, H, 24, 12), mỗi xương là 3 cột của phần xoay
    rồi phần dời, đúng thứ tự ``SplatSkinning.boneData`` của Spark. ``human_poses.json``:
    ``{"track_ids", "T_object_world": [F][H] ma trận 4x4 dời (16 số theo hàng) | null}``; vị trí
    cộng sau khi uốn, như ``instances_trans`` của OmniRe.
    """
    os.makedirs(os.path.join(out, "humans"), exist_ok=True)
    gaussians = gaussians_from_state(state)
    num_humans = state["instances_fv"].shape[1]
    owner = state["points_ids"][:, 0].long()
    weights = _human_weights(state).reshape(-1, NUM_BONES)
    track_ids = [f"smpl_{k:04d}" for k in range(num_humans)]
    written = []
    for index, track_id in enumerate(track_ids):
        keep = (owner == index) & _visible(gaussians, min_opacity)
        if not keep.any():
            continue
        write_spz(_filter(gaussians, keep), os.path.join(out, "humans", f"{track_id}.spz"))
        _skin(weights[keep]).tofile(os.path.join(out, "humans", f"{track_id}.skin"))
        written.append(track_id)
    bones = _bones(state)
    columns = torch.cat(
        [bones[..., :3, :3].transpose(-1, -2).reshape(*bones.shape[:3], 9), bones[..., :3, 3]],
        dim=-1,
    )
    columns.numpy().astype("<f4").tofile(os.path.join(out, "human_bones.bin"))
    present = state["instances_fv"].bool()
    trans = state["instances_trans"].float()
    poses = [
        [
            [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1] if present[f, k] else None
            for k, t in enumerate(trans[f].tolist())
        ]
        for f in range(present.shape[0])
    ]
    with open(os.path.join(out, "human_poses.json"), "w") as file:
        json.dump({"track_ids": track_ids, "T_object_world": poses}, file)
    return track_ids, written


def _export_sky(state: dict, out: str) -> str:
    """Ghi cubemap trời (``EnvLight.base``, 6 mặt) thành một ảnh 3 cột × 2 hàng.

    Mặt ``i`` (thứ tự OpenGL +x, -x, +y, -y, +z, -z) ở cột ``i % 3``, hàng ``i // 3``; trong mỗi
    mặt, hàng ảnh là toạ độ ``t``, cột là ``s`` của bảng chọn mặt OpenGL, đúng như
    ``nvdiffrast.texture(boundary_mode="cube")`` (đã so khớp 100% điểm với nvdiffrast). Hướng tra
    là ``EnvLight.to_opengl @ hướng nhìn world``.
    """
    from PIL import Image

    faces = state["base"].float().clamp(0, 1)  # (6, R, R, 3), màu như ảnh train
    rows = [torch.cat(list(faces[r * 3 : r * 3 + 3]), dim=1) for r in range(2)]
    atlas = (torch.cat(rows, dim=0) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(atlas).save(os.path.join(out, SKY_FILE), quality=92)
    return SKY_FILE


def _antialiased(ckpt: str) -> bool:
    """Model có train với chế độ chống răng cưa của gsplat không (``render.antialiased``)."""
    path = os.path.join(os.path.dirname(ckpt), "config.yaml")
    if not os.path.isfile(path):
        return False
    with open(path) as file:
        config = yaml.safe_load(file) or {}
    return bool((config.get("render") or {}).get("antialiased", False))


def main() -> None:
    """Đọc checkpoint, ghi gói xem trước vào ``--out``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--min_opacity", type=float, default=0.005)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "actors"), exist_ok=True)
    # Checkpoint do chính stage train của tool ghi ra, không phải file của bên thứ ba.
    models = torch.load(args.ckpt, map_location="cpu", weights_only=False)["models"]
    print("các node trong checkpoint:", list(models))

    up = up_axis_from_cameras(os.path.join(os.path.dirname(args.ckpt), CAMERAS_FILE))
    background = gaussians_from_state(models["Background"])
    write_spz(
        _filter(background, _visible(background, args.min_opacity)),
        os.path.join(args.out, "background.spz"),
    )
    sh_degree = int(round((background["f_rest"].shape[1] // 3 + 1) ** 0.5)) - 1

    rigid_actors: list[str] = []
    track_ids: list[str] = []
    poses: list[list] = []
    if "RigidNodes" in models:
        state = models["RigidNodes"]
        gaussians = gaussians_from_state(state)
        owner = state["points_ids"][:, 0].long()
        track_ids, poses = _poses(state)
        for index, track_id in enumerate(track_ids):
            keep = (owner == index) & _visible(gaussians, args.min_opacity)
            if keep.any():
                write_spz(
                    _filter(gaussians, keep),
                    os.path.join(args.out, "actors", f"{track_id}.spz"),
                )
                rigid_actors.append(track_id)

    humans: list[str] = []
    if "SMPLNodes" in models:
        _, humans = _export_humans(models["SMPLNodes"], args.out, args.min_opacity)
    sky = _export_sky(models["Sky"], args.out) if "base" in models.get("Sky", {}) else None

    box_only: list[str] = []
    omitted = ["per_camera_color", "fisheye"] + ([] if sky else ["sky"])
    if "DeformableNodes" in models:
        count = models["DeformableNodes"]["instances_fv"].shape[1]
        box_only += [f"deformable_{k:04d}" for k in range(count)]
        omitted.append("DeformableNodes")

    with open(os.path.join(args.out, "actor_poses.json"), "w") as file:
        json.dump({"track_ids": track_ids, "T_object_world": poses}, file)
    preview = {
        "schema_version": SCHEMA_VERSION,
        "source_run": args.source_run,
        "sh_degree": sh_degree,
        "splat_format": "spz",
        "up_axis": up,
        "antialiased": _antialiased(args.ckpt),
        "rigid_actors": rigid_actors,
        "humans": humans,
        "sky": sky,
        "box_only_actors": box_only,
        "omitted": omitted,
    }
    with open(os.path.join(args.out, "preview.json"), "w") as file:
        json.dump(preview, file, indent=2)
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
