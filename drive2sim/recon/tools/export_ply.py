"""Export a drivestudio checkpoint to a standard 3DGS .ply (INRIA layout) + metadata sidecar.

Exports the static Background node in drivestudio's world frame (no normalisation, unlike
utils/misc.export_gaussians_to_ply). Optionally also bakes RigidNodes (vehicles) at one frame.

    python export_ply.py --ckpt <log_dir>/checkpoint_final.pth --out <dir> [--rigid_frame 0]
"""
import argparse, json, os
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def quat_to_mat(q):  # wxyz -> 3x3
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def quat_mul(a, b):  # wxyz
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def gaussians_from_state(sd):
    means = sd["_means"].float()
    scales = sd["_scales"].float()
    if scales.shape[1] == 1:          # ball gaussians
        scales = scales.repeat(1, 3)
    quats = torch.nn.functional.normalize(sd["_quats"].float(), dim=-1)
    f_dc = sd["_features_dc"].float().reshape(len(means), -1)                 # (N,3)
    f_rest = sd["_features_rest"].float()                                     # (N,K,3)
    f_rest = f_rest.transpose(1, 2).reshape(len(means), -1)                   # INRIA order: (N,3*K)
    opac = sd["_opacities"].float().reshape(-1, 1)                            # logit
    return dict(means=means, scales=scales, quats=quats, f_dc=f_dc, f_rest=f_rest, opac=opac)


PLY_COMMENT = "drive2sim up=+z frame=drivestudio_world units=m"


def write_ply(g, path):
    n = len(g["means"])
    cols = [g["means"], torch.zeros(n, 3), g["f_dc"], g["f_rest"], g["opac"], g["scales"], g["quats"]]
    data = torch.cat(cols, 1).cpu().numpy().astype(np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(g["f_dc"].shape[1])]
             + [f"f_rest_{i}" for i in range(g["f_rest"].shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    arr = np.empty(n, dtype=[(k, "f4") for k in names])
    for i, k in enumerate(names):
        arr[k] = data[:, i]
    # drive2sim: tag the frame so viewers (Studio's 3D page) pick the right up axis on import.
    PlyData([PlyElement.describe(arr, "vertex")], comments=[PLY_COMMENT]).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rigid_frame", type=int, default=-1, help="bake RigidNodes at this frame (-1 = skip)")
    ap.add_argument("--min_opacity", type=float, default=0.005)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    models = ck["models"]
    print("nodes in checkpoint:", list(models.keys()))

    parts, stats = [], {}
    bg = gaussians_from_state(models["Background"])
    parts.append(bg); stats["Background"] = len(bg["means"])

    if a.rigid_frame >= 0 and "RigidNodes" in models:
        sd = models["RigidNodes"]
        g = gaussians_from_state(sd)
        ids = sd["points_ids"][:, 0].long()                       # drivestudio saves it as "points_ids"
        q_ins = torch.nn.functional.normalize(sd["instances_quats"][a.rigid_frame].float(), dim=-1)[ids]
        t_ins = sd["instances_trans"][a.rigid_frame].float()[ids]
        g["means"] = (quat_to_mat(q_ins) @ g["means"][..., None])[..., 0] + t_ins
        g["quats"] = torch.nn.functional.normalize(quat_mul(q_ins, g["quats"]), dim=-1)
        visible = sd["instances_fv"][a.rigid_frame].bool()[ids]   # only actors present at that frame
        g = {k: v[visible] for k, v in g.items()}
        # NOTE: f_rest (view-dependent SH) is not rotated; fine for a quick look, slightly wrong colours off-axis.
        parts.append(g); stats[f"RigidNodes@frame{a.rigid_frame}"] = len(g["means"])

    allg = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    keep = torch.sigmoid(allg["opac"][:, 0]) > a.min_opacity
    allg = {k: v[keep] for k, v in allg.items()}
    ply_path = os.path.join(a.out, "scene.ply")
    write_ply(allg, ply_path)

    s = torch.exp(allg["scales"]).max(1).values
    sidecar = {
        "source_checkpoint": os.path.abspath(a.ckpt),
        "step": int(ck.get("step", -1)),
        "frame": "drivestudio world frame (as stored in the checkpoint; no normalisation)",
        "units": "metres",
        "layout": "INRIA 3DGS: xyz, f_dc(SH0), f_rest(SH1-3, channel-major), opacity(logit), scale(log), rot(wxyz)",
        "dynamic_actors": "excluded" if a.rigid_frame < 0 else f"RigidNodes baked at frame {a.rigid_frame}",
        "num_gaussians_per_node": stats,
        "num_gaussians_written": int(keep.sum()),
        "file_size_mb": round(os.path.getsize(ply_path) / 2**20, 1),
        "asset_health": {
            "near_transparent_lt_0.05": int((torch.sigmoid(allg["opac"][:, 0]) < 0.05).sum()),
            "oversized_gt_5m": int((s > 5).sum()),
            "max_scale_m": float(s.max()),
        },
    }
    json.dump(sidecar, open(os.path.join(a.out, "scene.json"), "w"), indent=2)
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
