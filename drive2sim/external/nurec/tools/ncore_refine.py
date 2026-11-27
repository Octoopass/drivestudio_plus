"""(drive2sim) Tối ưu tiếp PLY của Instant NuRec trên ảnh của clip NCore bằng gsplat.

    <python có gsplat> tools/ncore_refine.py scene_nurec.ply train_views.npz --out refined.ply
        [--iters 7000] [--strategy mcmc|none] [--cap-factor 1.5] [--max-side 1408]
        [--means-lr 1e-3] [--ssim-weight 0.2] [--max-gaussians 0] [--gpus 1]
        [--overlap-m 50] [--seed 0]

``train_views.npz`` là file của ``ncore_eval_export.py`` (pose, thông số, ảnh, mask ego của các
khung train). Gaussian của NuRec là điểm xuất phát: vị trí, hướng, kích thước, độ đục và màu (SH
bậc 0) được tối ưu bằng L1 + D-SSIM. Camera mắt cá vẽ bằng 3DGUT (``with_ut``, ``with_eval3d``) như
lúc chấm. Lỗi chỉ tính ở pixel ngoài mask ego mà Gaussian đã phủ (alpha > ``--alpha-min``): bầu
trời của NuRec nằm ngoài PLY, nên không bắt Gaussian lấp trời.

``--strategy mcmc`` (thử nghiệm): ``MCMCStrategy`` của gsplat dời Gaussian gần trong suốt tới chỗ
cần và thêm Gaussian tới ``--cap-factor`` lần số ban đầu; không cần gradient 2D nên dùng được với
3DGUT. ``none`` (mặc định): giữ nguyên số Gaussian.

Tỉa trước khi tối ưu: tối ưu cần chừng ``BYTES_PER_GAUSSIAN`` byte VRAM mỗi Gaussian (tham số,
gradient, hai trạng thái Adam, bộ đệm rasterize); PLY "không giới hạn" của cả clip dài (153 triệu
Gaussian ở drive_0006) hết 48 GB ngay bước đầu. PLY nhiều hơn ``--max-gaussians`` (0: tự tính theo
VRAM trống, ``VRAM_SHARE`` của phần trống) thì giữ các Gaussian đục nhất (Gaussian gần trong suốt
góp ít nhất vào ảnh) rồi mới tối ưu; ``refine.json`` ghi số trước và sau khi tỉa.

Nhiều GPU (``--gpus`` > 1, hoặc 0 là mọi GPU còn trống): chia clip theo quãng đường đi. Mỗi
Gaussian thuộc khung có tâm camera gần nó nhất; các khung được cắt thành ``--gpus`` đoạn liền
nhau sao cho số Gaussian mỗi đoạn gần bằng nhau. Mỗi GPU (một tiến trình con ``--worker``) tối ưu
Gaussian của đoạn mình trên các view của đoạn đó, kèm Gaussian của đoạn khác nằm trong
``--overlap-m`` mét quanh camera của đoạn làm ngữ cảnh (cũng được tối ưu nhưng bỏ khi ghép). Các
GPU không trao đổi gì với nhau (gsplat ``distributed`` không dùng được với 3DGUT ``with_eval3d``,
và cộng gradient của cả trăm triệu Gaussian qua PCIe chậm hơn bước tính). ``--iters`` là tổng số
bước, chia cho các GPU theo số view, nên mỗi view vẫn được thấy chừng ấy lần mà xong nhanh gần
``--gpus`` lần. Gaussian bị tỉa trong một đoạn giữ nguyên giá trị gốc (không mất). Chỉ dùng với
``--strategy none``.

PLY ghi ra cùng cột với PLY gốc (``x y z``, ``rot_*``, ``scale_*``, ``opacity``, ``f_dc_*``...):
các stage đọc PLY NuRec dùng được ngay. Cột phụ (``nx``, ``road_mask``, ``sky_mask``) giữ của
Gaussian gốc khi số Gaussian không đổi, không thì đặt 0. In tiến độ dạng thanh tqdm và
``refine.json`` (thời gian, PSNR train đầu/cuối, VRAM) cạnh file ra.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

C0 = 0.28209479177387814
BYTES_PER_GAUSSIAN = 320  # đo: 153,6 triệu Gaussian dùng 45,9 GB ở bước đầu
VRAM_SHARE = 0.55  # phần VRAM trống dành cho Gaussian; phần còn lại cho ảnh và giao tile
FREE_GPU_SHARE = 0.9  # GPU "trống": VRAM trống ít nhất chừng này phần (--gpus 0)
PROBE_VIEWS = 40
PROGRESS = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
TYPES = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}


def gaussian_budget(requested: int) -> int:
    """Số Gaussian tối đa được tối ưu: ``requested`` nếu > 0, không thì theo VRAM trống."""
    if requested > 0:
        return requested
    free, _ = torch.cuda.mem_get_info()
    return int(free * VRAM_SHARE / BYTES_PER_GAUSSIAN)


def top_opacity(opacity: np.ndarray, keep: int) -> np.ndarray:
    """Chỉ số (tăng dần) của ``keep`` Gaussian có logit độ đục cao nhất."""
    if len(opacity) <= keep:
        return np.arange(len(opacity))
    return np.sort(np.argpartition(-opacity.astype(np.float32), keep - 1)[:keep])


def prune_by_opacity(data: np.ndarray, keep: int) -> np.ndarray:
    """Giữ ``keep`` Gaussian có độ đục (logit ``opacity``) cao nhất, giữ thứ tự ban đầu."""
    if len(data) <= keep:
        return data
    return data[top_opacity(data["opacity"], keep)]


def ply_layout(path: Path) -> tuple[list[str], np.dtype, int, int]:
    """Header, kiểu một dòng, số Gaussian và vị trí bắt đầu dữ liệu của PLY nhị phân."""
    with path.open("rb") as f:
        header, props, n = [], [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            header.append(line)
            parts = line.split()
            if parts[0] == "element" and parts[1] == "vertex":
                n = int(parts[2])
            elif parts[0] == "property":
                props.append((parts[2], TYPES[parts[1]]))
        return header, np.dtype(props), n, f.tell()


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    header, dtype, n, offset = ply_layout(path)
    with path.open("rb") as f:
        f.seek(offset)
        data = np.frombuffer(f.read(), dtype=dtype, count=n).copy()
    return header, data


def map_ply(path: Path) -> tuple[list[str], np.ndarray]:
    """PLY dạng memmap (chỉ đọc): đọc từng phần mà không nạp cả file."""
    header, dtype, n, offset = ply_layout(path)
    return header, np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n,))


def write_ply(path: Path, header: list[str], data: np.ndarray) -> None:
    lines = [
        f"element vertex {len(data)}" if l.startswith("element vertex") else l for l in header
    ]
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\nend_header\n").encode("ascii"))
        f.write(data.tobytes())


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM trung bình, ảnh [1, 3, H, W] trong [0, 1]."""
    try:
        from fused_ssim import fused_ssim

        return fused_ssim(a, b, padding="valid")
    except ImportError:
        pass
    k = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=a.dtype, device=a.device)
    w = (k @ k.T)[None, None].repeat(3, 1, 1, 1)
    blur = lambda x: F.conv2d(x, w, groups=3)  # noqa: E731
    mu1, mu2 = blur(a), blur(b)
    s1, s2, s12 = blur(a * a) - mu1**2, blur(b * b) - mu2**2, blur(a * b) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))
    ).mean()


def load_views(path: Path, max_side: int, subset: np.ndarray | None = None) -> list[dict]:
    with np.load(path, allow_pickle=True) as npz:
        v = {k: npz[k] for k in npz.files}
    views = []
    for i in range(len(v["camera"])) if subset is None else subset.tolist():
        img = v["image"][i]
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        K = v["K"][i].astype(np.float32).copy()
        mask = v["mask"][i].astype(np.uint8)
        if s < 1:
            w, h = int(w * s), int(h * s)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            K[:2] *= s
        views.append({
            "camera": str(v["camera"][i]), "fisheye": str(v["model"][i]) == "fisheye",
            "viewmat": v["viewmat"][i].astype(np.float32), "K": K,
            "radial": v["radial"][i].astype(np.float32), "image": np.ascontiguousarray(img),
            "valid": torch.from_numpy(mask == 0), "w": w, "h": h,
        })  # fmt: skip
    return views


def view_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Tâm camera (thế giới) và chỉ số khung của mọi view, không nạp ảnh."""
    with np.load(path, allow_pickle=True) as npz:
        viewmat = npz["viewmat"].astype(np.float64)
        frame = npz["frame"] if "frame" in npz.files else np.arange(len(viewmat))
    R, t = viewmat[:, :3, :3], viewmat[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t), np.asarray(frame)


def optimize(data: np.ndarray, views: list[dict], a: argparse.Namespace, iters: int,
             rng: np.random.Generator) -> dict:  # fmt: skip
    """Tối ưu ``data`` (Gaussian của PLY) trên ``views``; trả giá trị mới và thống kê."""
    dev = "cuda"
    col = lambda *names: torch.from_numpy(  # noqa: E731
        np.stack([data[k].astype(np.float32) for k in names], 1)
    ).to(dev)
    params = torch.nn.ParameterDict({
        "means": col("x", "y", "z"),
        "quats": col("rot_0", "rot_1", "rot_2", "rot_3"),
        "scales": col("scale_0", "scale_1", "scale_2"),
        "opacities": col("opacity")[:, 0],
        "sh0": col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :],
    })  # fmt: skip
    n0 = len(data)
    lrs = {"means": a.means_lr, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    gamma = 0.01 ** (1 / max(1, iters))  # lr vị trí giảm tới 1% ở bước cuối
    strategy = state = None
    if a.strategy == "mcmc":
        from gsplat.strategy import MCMCStrategy

        strategy = MCMCStrategy(
            cap_max=int(n0 * a.cap_factor), refine_start_iter=500,
            refine_stop_iter=int(iters * 0.8), refine_every=100, min_opacity=0.005,
        )  # fmt: skip
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()
    print(f"{n0:,} Gaussian, {len(views)} view train, {iters} bước, strategy {a.strategy}",
          flush=True)  # fmt: skip

    def render(view: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        kw = dict(camera_model="fisheye", radial_coeffs=torch.from_numpy(view["radial"])[None].to(dev),
                  with_ut=True, with_eval3d=True) if view["fisheye"] else {}  # fmt: skip
        colors = params["sh0"][:, 0] * C0 + 0.5
        rgb, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors,
            torch.from_numpy(view["viewmat"])[None].to(dev), torch.from_numpy(view["K"])[None].to(dev),
            view["w"], view["h"], render_mode="RGB", packed=False, **kw,
        )  # fmt: skip
        return rgb[0], alpha[0, ..., 0], info

    def train_psnr(sample: list[int]) -> float:
        values = []
        with torch.no_grad():
            for i in sample:
                rgb, alpha, _ = render(views[i])
                gt = torch.from_numpy(views[i]["image"]).to(dev).float() / 255
                valid = views[i]["valid"].to(dev) & (alpha > 0.9)
                if valid.any():
                    values.append(
                        float(-10 * torch.log10(((rgb.clamp(0, 1) - gt) ** 2)[valid].mean()))
                    )
        return float(np.mean(values)) if values else float("nan")

    probe = rng.choice(len(views), size=min(a.probe_views, len(views)), replace=False).tolist()
    psnr_before = train_psnr(probe)
    print(f"PSNR train (mẫu {len(probe)} view) trước: {psnr_before:.2f} dB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    order = rng.permutation(len(views))
    loss = torch.zeros(())
    for step in range(iters):
        view = views[order[step % len(order)]]
        if step % len(order) == len(order) - 1:
            order = rng.permutation(len(views))
        rgb, alpha, info = render(view)
        gt = torch.from_numpy(view["image"]).to(dev).float() / 255
        keep = (view["valid"].to(dev) & (alpha.detach() > a.alpha_min))[..., None]
        pred = torch.where(keep, rgb, gt)  # pixel không tính lỗi lấy theo ảnh thật: gradient 0
        l1 = (pred - gt).abs().mean()
        loss = (1 - a.ssim_weight) * l1
        if a.ssim_weight > 0:
            loss = loss + a.ssim_weight * (
                1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            )
        if strategy is not None:
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean()
            loss = loss + 0.01 * torch.exp(params["scales"]).mean()
            strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= gamma
        if strategy is not None:
            strategy.step_post_backward(
                params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"]
            )
        if step % 100 == 0 or step == iters - 1:
            done = step + 1
            bar = "#" * int(20 * done / iters)
            # dạng thanh tqdm để Studio (log_progress) đọc được tiến độ
            print(f"refine: {int(100 * done / iters):3d}%|{bar:<20}| {done}/{iters} "
                  f"[loss {float(loss):.4f}, {len(params['means']):,} Gaussian]", flush=True)  # fmt: skip

    psnr_after = train_psnr(probe)
    print(f"PSNR train (mẫu {len(probe)} view) sau: {psnr_after:.2f} dB", flush=True)
    get = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    with torch.no_grad():
        fields = {
            "means": get(params["means"]), "quats": get(F.normalize(params["quats"], dim=-1)),
            "scales": get(params["scales"]), "opacity": get(params["opacities"]),
            "sh0": get(params["sh0"][:, 0]),
        }  # fmt: skip
    return {
        "fields": fields, "n0": n0, "n": len(fields["means"]), "probe": len(probe),
        "psnr_before": psnr_before, "psnr_after": psnr_after,
        "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20),
    }  # fmt: skip


def put_fields(out: np.ndarray, fields: dict, index: np.ndarray | slice = slice(None)) -> None:
    """Ghi giá trị đã tối ưu vào các dòng ``index`` của mảng PLY ``out``."""
    for i, k in enumerate("xyz"):
        out[k][index] = fields["means"][:, i]
    for i in range(4):
        out[f"rot_{i}"][index] = fields["quats"][:, i]
    for i in range(3):
        out[f"scale_{i}"][index] = fields["scales"][:, i]
        out[f"f_dc_{i}"][index] = fields["sh0"][:, i]
    out["opacity"][index] = fields["opacity"]
    rgb8 = np.clip((fields["sh0"] * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    for i, k in enumerate(("red", "green", "blue")):
        if k in out.dtype.names:
            out[k][index] = rgb8[:, i]
    if "alpha" in out.dtype.names:
        out["alpha"][index] = np.clip(255 / (1 + np.exp(-fields["opacity"])), 0, 255).astype(
            np.uint8
        )


def free_gpus(wanted: int) -> list[int]:
    """GPU dùng cho ``--gpus``: ``wanted`` GPU trống nhiều VRAM nhất (0: mọi GPU còn trống)."""
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    rows = [tuple(int(x) for x in line.split(",")) for line in done.stdout.strip().splitlines()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        allowed = {int(x) for x in visible.split(",") if x.strip().isdigit()}
        rows = [r for r in rows if r[0] in allowed]
    rows.sort(key=lambda r: -r[1])
    if wanted == 0:
        return sorted(r[0] for r in rows if r[1] >= FREE_GPU_SHARE * r[2]) or [rows[0][0]]
    return sorted(r[0] for r in rows[:wanted])


def plan_shards(means: np.ndarray, centers: np.ndarray, frames: np.ndarray, parts: int,
                overlap_m: float) -> list[dict]:  # fmt: skip
    """Chia Gaussian và view thành ``parts`` đoạn liền nhau theo khung (xem docstring đầu file)."""
    from scipy.spatial import cKDTree

    keys, view_frame = np.unique(frames, return_inverse=True)
    frame_center = np.stack([centers[view_frame == k].mean(0) for k in range(len(keys))])
    _, nearest = cKDTree(frame_center).query(means, workers=-1)
    per_frame = np.bincount(nearest, minlength=len(keys))
    cum = np.cumsum(per_frame)
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, parts) / parts, side="right")
    frame_shard = np.searchsorted(cuts, np.arange(len(keys)), side="right")
    owner = frame_shard[nearest]
    del nearest
    shards = []
    for k in range(parts):
        views = np.flatnonzero(frame_shard[view_frame] == k)
        mine = frame_shard == k
        near, _ = cKDTree(frame_center[mine]).query(
            means, distance_upper_bound=overlap_m, workers=-1
        )
        members = np.flatnonzero((owner == k) | np.isfinite(near))
        del near
        shards.append({"members": members, "owned": owner[members] == k, "views": views,
                       "frames": int(mine.sum())})  # fmt: skip
    return shards


def run_worker(a: argparse.Namespace) -> None:
    """Tiến trình con của một GPU: tối ưu một đoạn, ghi ``--out`` (npz các Gaussian sở hữu)."""
    shard = np.load(a.worker)
    members, owned, view_ids = shard["members"], shard["owned"], shard["views"]
    _, ply = map_ply(Path(a.ply))
    data = np.asarray(ply[members])
    budget = gaussian_budget(a.max_gaussians)
    chosen = top_opacity(data["opacity"], budget)
    if len(chosen) < len(data):
        print(f"tỉa {len(data):,} → {len(chosen):,} Gaussian (Gaussian bị tỉa giữ giá trị gốc)",
              flush=True)  # fmt: skip
        data, members, owned = data[chosen], members[chosen], owned[chosen]
    views = load_views(Path(a.views), a.max_side, view_ids)
    result = optimize(data, views, a, int(shard["iters"]), np.random.default_rng(a.seed))
    fields = {k: v[owned] for k, v in result.pop("fields").items()}
    np.savez(a.out, index=members[owned], **fields)
    Path(a.out + ".json").write_text(json.dumps({**result, "optimized": int(len(chosen))}))


def _kill_with_parent() -> None:
    """Tiến trình con tự nhận SIGTERM khi tiến trình cha chết (Linux)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def run_sharded(a: argparse.Namespace, gpus: list[int], began: float) -> None:
    """Chia clip cho các GPU, chạy mỗi đoạn một tiến trình con rồi ghép PLY."""
    out_path = Path(a.out)
    work = out_path.parent / "refine_shards"
    work.mkdir(parents=True, exist_ok=True)
    header, ply = map_ply(Path(a.ply))
    loaded = len(ply)
    centers, frames = view_poses(Path(a.views))
    means = np.stack([np.asarray(ply[k], dtype=np.float32) for k in "xyz"], 1)
    shards = plan_shards(means, centers, frames, len(gpus), a.overlap_m)
    del means
    total_views = sum(len(s["views"]) for s in shards)
    print(f"{loaded:,} Gaussian, {total_views} view train, {a.iters} bước, chia {len(gpus)} GPU "
          f"{gpus} theo quãng đường (ngữ cảnh {a.overlap_m:g} m)", flush=True)  # fmt: skip
    procs, progress = [], {}
    for k, (gpu, s) in enumerate(zip(gpus, shards)):
        iters = max(1, round(a.iters * len(s["views"]) / max(1, total_views)))
        progress[k] = (0, iters)
        np.savez(work / f"shard{k}.npz", members=s["members"], owned=s["owned"],
                 views=s["views"], iters=iters)  # fmt: skip
        print(f"  GPU {gpu}: {s['frames']} khung, {len(s['views'])} view, "
              f"{int(s['owned'].sum()):,} Gaussian sở hữu + {int((~s['owned']).sum()):,} ngữ cảnh, "
              f"{iters} bước", flush=True)  # fmt: skip
        command = [sys.executable, "-u", __file__, a.ply, a.views, "--out", str(work / f"out{k}.npz"),
                   "--worker", str(work / f"shard{k}.npz"), "--max-side", str(a.max_side),
                   "--means-lr", str(a.means_lr), "--ssim-weight", str(a.ssim_weight),
                   "--alpha-min", str(a.alpha_min), "--max-gaussians", str(a.max_gaussians),
                   "--probe-views", str(max(5, PROBE_VIEWS // len(gpus))), "--seed", str(a.seed + k)]  # fmt: skip
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        procs.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, env=env, preexec_fn=_kill_with_parent))  # fmt: skip
    del shards
    lock = threading.Lock()

    def pump(k: int, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            match = PROGRESS.search(line)
            with lock:
                if line.startswith("refine:") and match:
                    progress[k] = (int(match.group(1)), int(match.group(2)))
                else:
                    print(f"[GPU {gpus[k]}] {line.rstrip()}", flush=True)

    threads = [
        threading.Thread(target=pump, args=(k, p), daemon=True) for k, p in enumerate(procs)
    ]
    for t in threads:
        t.start()
    shown, healthy = -1, False
    try:
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None, 0) for p in procs):
                break  # một GPU lỗi: dừng các GPU còn lại
            with lock:
                done = sum(d for d, _ in progress.values())
                total = sum(t for _, t in progress.values())
            if done // 100 != shown:
                shown = done // 100
                bar = "#" * int(20 * done / total)
                print(f"refine: {int(100 * done / total):3d}%|{bar:<20}| {done}/{total} "
                      f"[{len(gpus)} GPU]", flush=True)  # fmt: skip
            time.sleep(2)
        healthy = all(p.poll() == 0 for p in procs)
    finally:
        if not healthy:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
        for p in procs:
            p.wait()
        for t in threads:
            t.join(timeout=5)
    bad = [gpus[k] for k, p in enumerate(procs) if p.returncode != 0]
    if bad:
        raise SystemExit(f"tối ưu trên GPU {bad} lỗi (xem dòng [GPU ...] phía trên)")
    total = sum(t for _, t in progress.values())
    print(f"refine: 100%|{'#' * 20}| {total}/{total} [{len(gpus)} GPU]", flush=True)

    out = np.array(ply)  # bản sao đầy đủ; Gaussian không thuộc đoạn nào được tối ưu giữ nguyên
    stats_k, updated = [], 0
    for k in range(len(gpus)):
        with np.load(work / f"out{k}.npz") as res:
            index = res["index"]
            put_fields(
                out, {f: res[f] for f in ("means", "quats", "scales", "opacity", "sh0")}, index
            )
        updated += len(index)
        stats_k.append({**json.loads((work / f"out{k}.npz.json").read_text()), "gpu": gpus[k],
                        "owned_optimized": int(len(index))})  # fmt: skip
    write_ply(out_path, header, out)
    weight = [s["probe"] for s in stats_k]
    mean = lambda key: float(np.average([s[key] for s in stats_k], weights=weight))  # noqa: E731
    stats = {
        "iters": a.iters, "strategy": a.strategy, "views": total_views, "gpus": gpus,
        "gaussians_loaded": loaded, "gaussians_before": loaded, "gaussians_after": loaded,
        "gaussians_optimized": updated,
        "train_psnr_before": round(mean("psnr_before"), 3),
        "train_psnr_after": round(mean("psnr_after"), 3),
        "refine_time_s": round(time.perf_counter() - began, 1),
        "peak_vram_mb": max(s["peak_vram_mb"] for s in stats_k),
        "shards": [{k: v for k, v in s.items() if k != "fields"} for s in stats_k],
    }  # fmt: skip
    (out_path.parent / "refine.json").write_text(json.dumps(stats, indent=2) + "\n")
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    print(f"PSNR train sau: {stats['train_psnr_after']:.2f} dB (trước {stats['train_psnr_before']:.2f})",
          flush=True)  # fmt: skip
    print(f"wrote {out_path} ({loaded:,} Gaussian, tối ưu {updated:,}) {json.dumps(stats)}",
          flush=True)  # fmt: skip


def run_single(a: argparse.Namespace, began: float) -> None:
    """Một GPU: tỉa nếu không vừa VRAM rồi tối ưu cả clip."""
    header, data = read_ply(Path(a.ply))
    loaded = len(data)
    budget = gaussian_budget(a.max_gaussians)
    if loaded > budget:
        data = prune_by_opacity(data, budget)
        print(f"tỉa {loaded:,} → {len(data):,} Gaussian (giữ Gaussian đục nhất) cho vừa VRAM",
              flush=True)  # fmt: skip
    views = load_views(Path(a.views), a.max_side)
    result = optimize(data, views, a, a.iters, np.random.default_rng(a.seed))
    n0, n = result["n0"], result["n"]
    out = data.copy() if n == n0 else np.zeros(n, dtype=data.dtype)
    put_fields(out, result["fields"])
    target = Path(a.out)
    write_ply(target, header, out)
    stats = {
        "iters": a.iters, "strategy": a.strategy, "views": len(views),
        "gaussians_loaded": loaded, "gaussians_before": n0, "gaussians_after": int(n),
        "train_psnr_before": round(result["psnr_before"], 3),
        "train_psnr_after": round(result["psnr_after"], 3),
        "refine_time_s": round(time.perf_counter() - began, 1),
        "peak_vram_mb": result["peak_vram_mb"],
    }  # fmt: skip
    (target.parent / "refine.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {target} ({n:,} Gaussian) {json.dumps(stats)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("views")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=7000)
    p.add_argument("--strategy", choices=["mcmc", "none"], default="none")
    p.add_argument("--cap-factor", type=float, default=1.5)
    p.add_argument("--max-side", type=int, default=1408)
    p.add_argument("--means-lr", type=float, default=1e-3, help="lr vị trí (mét), giảm dần x0.01")
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--alpha-min", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gaussians", type=int, default=0, help="tỉa nếu nhiều hơn; 0: theo VRAM")
    p.add_argument("--gpus", type=int, default=1, help="số GPU; 0: mọi GPU còn trống")
    p.add_argument("--overlap-m", type=float, default=50.0, help="ngữ cảnh quanh camera mỗi đoạn")
    p.add_argument("--probe-views", type=int, default=PROBE_VIEWS, help=argparse.SUPPRESS)
    p.add_argument("--worker", help=argparse.SUPPRESS)  # tiến trình con của --gpus
    a = p.parse_args()
    torch.manual_seed(a.seed)
    began = time.perf_counter()
    if a.worker:
        run_worker(a)
        return
    gpus = free_gpus(a.gpus) if a.gpus != 1 else []
    if len(gpus) > 1:
        if a.strategy != "none":
            raise SystemExit("--gpus > 1 chỉ dùng với --strategy none")
        run_sharded(a, gpus, began)
    else:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        run_single(a, began)


if __name__ == "__main__":
    main()
