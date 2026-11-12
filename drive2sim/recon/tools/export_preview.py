"""Xuất gói xem trước cho Spark từ checkpoint DriveStudio (CONTRACTS mục 7).

Khác ``export_ply.py`` (gộp mọi thứ vào một ``scene.ply`` ở một khung), gói này tách:
- ``background.ply``: Gaussian của nền tĩnh, hệ world;
- ``actors/<track_id>.ply``: Gaussian của từng xe (``RigidNodes``) trong hệ của xe;
- ``actor_poses.json``: pose của từng xe theo khung, để trình duyệt đặt xe đúng chỗ ở mọi khung;
- ``preview.json``: gói có gì và thiếu gì.

Người (``SMPLNodes``) và vật biến dạng (``DeformableNodes``) không có Gaussian trong gói, chỉ được
liệt kê trong ``box_only_actors``. Màu theo góc nhìn (SH bậc cao) của xe giữ trong hệ của xe;
viewer xoay cả đối tượng nên không cần xoay SH.

    python -m drive2sim.recon.tools.export_preview --ckpt <log_dir>/checkpoint_final.pth \\
        --out <dir> --source_run runs/<scene>/<approach>/<run_id>

Chưa chạy thật: cần checkpoint thật và PyTorch.
"""

import argparse
import json
import os

import torch

from drive2sim.recon.tools.export_ply import gaussians_from_state, quat_to_mat, write_ply

SCHEMA_VERSION = "d2s-preview/0.1"


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
        [
            matrices[f, k].reshape(-1).tolist() if present[f, k] else None
            for k in range(num_actors)
        ]
        for f in range(num_frames)
    ]
    return [f"rigid_{k:04d}" for k in range(num_actors)], poses


def main() -> None:
    """Đọc checkpoint, ghi gói xem trước vào ``--out``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--min_opacity", type=float, default=0.005)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "actors"), exist_ok=True)
    models = torch.load(args.ckpt, map_location="cpu")["models"]
    print("các node trong checkpoint:", list(models))

    background = gaussians_from_state(models["Background"])
    write_ply(
        _filter(background, _visible(background, args.min_opacity)),
        os.path.join(args.out, "background.ply"),
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
                write_ply(_filter(gaussians, keep), os.path.join(args.out, "actors", f"{track_id}.ply"))
                rigid_actors.append(track_id)

    box_only: list[str] = []
    omitted = ["sky", "per_camera_color", "fisheye"]
    for node, prefix in (("SMPLNodes", "smpl"), ("DeformableNodes", "deformable")):
        if node in models:
            count = models[node]["instances_fv"].shape[1]
            box_only += [f"{prefix}_{k:04d}" for k in range(count)]
            omitted.append(node)

    with open(os.path.join(args.out, "actor_poses.json"), "w") as file:
        json.dump({"track_ids": track_ids, "T_object_world": poses}, file)
    preview = {
        "schema_version": SCHEMA_VERSION,
        "source_run": args.source_run,
        "sh_degree": sh_degree,
        "rigid_actors": rigid_actors,
        "box_only_actors": box_only,
        "omitted": omitted,
    }
    with open(os.path.join(args.out, "preview.json"), "w") as file:
        json.dump(preview, file, indent=2)
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
