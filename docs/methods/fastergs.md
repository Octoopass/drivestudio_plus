# Faster-GS speed-ups

## Where the code is

- Python: `drive2sim/recon/speedups/fastergs.py` (`FasterGS`), `drive2sim/recon/trainers/base.py`. Flags in `trainer.fastergs`: `fused_adam`, `fused_ssim`, `fewer_syncs`, `morton_order`.
- Kernels: the project's gsplat-geer fork, `third_party/gsplat-geer`. Each technique is a new `rasterization()` argument, off by default: `exact_tiles`, `two_stage_sort` and `per_gaussian_backward`. The trainer enables only what `gsplat.fastergs_support()` reports for the renderer.
- GPU equivalence tests: `third_party/gsplat-geer/tests/test_drive2sim_*.py`. `summary.json` records active techniques under `speedups`.

## What it does

It makes each step faster without changing the algorithm, densification or Gaussian count. Images match bit for bit, and gradients differ only by rounding.

- Fused Adam: one kernel updates a whole parameter group.
- Fused SSIM: one kernel computes SSIM and its gradient.
- Fewer host synchronisations: the CPU waits less for the GPU.
- Exact tile selection: a Gaussian goes only to tiles its ellipse overlaps.
- Two-stage sorting: depth and tile sorting run in two cheaper passes.
- Per-Gaussian backward: the rasterizer computes gradients per Gaussian, not per pixel.

## Why this method

OmniRe runs must fit a 12-hour Kaggle T4 session, and a speed-up must not change results. Alternatives were:

- Taming 3DGS provides per-Gaussian backward and fused SSIM kernels, which Faster-GS includes. Taming's code modifies the Inria rasterizer, not gsplat.
- gsplat's packed mode keeps only visible Gaussian-camera pairs. gsplat calls it more memory efficient, not necessarily faster, and the trainer requires `packed=false` for 3DGUT and 3DGEER.
- Speedy-Splat adds exact tile intersection and prunes over 90 % of Gaussians. Pruning changes the model (FastGS covers that here), and the Faster-GS paper found Radl et al.'s tile culling faster.

Faster-GS was chosen because it collects these kernels and a fused optimizer and makes each technique a flag. On nuScenes (OmniRe, 3DGS, single runs), training fell from 219 to 198 min, PSNR rose from 25.15 to 25.20 and peak memory from 6.6 to 6.9 GB.

## First-principles explanation

A step has four parts: rendering, loss, parameter update, and method-specific work (for OmniRe, gathering and transforming node Gaussians). Faster-GS speeds up rendering, SSIM and Adam, so the untouched part limits the gain (Amdahl's law). A profile at step 20 000 showed about 9 % of GPU time in the rasterizer, 14 % in the optimizer and 77 % in OmniRe's own work. Doubling the speed of that 23 % share gives at most about 13 %, and the measured gain was about 10 %. In static scenes, rendering and the update dominate, which explains the paper's four- to fivefold speed-up.

## Difference from the original code, method or paper

Reference: "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm et al., CVPR 2026), code `nerficg-project/faster-gaussian-splatting`.

- The original NeRFICG rasterizer supports only pinhole cameras and returns only colour. OmniRe needs depth, alpha and fisheye, so the techniques were re-implemented in gsplat-geer.
- The paper's fused activations and separate spherical-harmonics buffer are not implemented.
- The update does not skip invisible Gaussians, because gsplat-geer's `SelectiveAdam` also drops Adam's bias correction.
- Kernel techniques exist only for the standard renderer. For 3DGUT, exact tile selection is unsafe because 3DGUT evaluates Gaussians along 3D rays, and the other two have no 3DGUT version. 3DGEER's tile selection is already exact.

## Limits, risks and possible gains

- Per-Gaussian backward uses about 300 MB more GPU memory.
- `morton_order` (Z-order sorting of background Gaussians) gives only statistically equivalent results, so it is off by default.
- Larger gains lie in OmniRe's own work: fewer Gaussians (FastGS) or faster node gathering (`omnire_node_gather.md`).
- For 3DGUT, the rasterizer takes about 14 % of a step, so porting the kernels would gain at most about 5 %.
