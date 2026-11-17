# Faster-GS speed-ups

## Where the code is

- Python: `drive2sim/recon/speedups/fastergs.py` (`FasterGS`), `drive2sim/recon/trainers/base.py`. Flags in `trainer.fastergs`: `fused_adam`, `fused_ssim`, `fewer_syncs`, `morton_order`.
- Kernels: the project's gsplat-geer fork, `third_party/gsplat-geer`. Each technique is a new `rasterization()` argument defaulting to the old behaviour: `exact_tiles`, `two_stage_sort`, `per_gaussian_backward`. The trainer enables only what `gsplat.fastergs_support()` reports for the renderer.
- GPU equivalence tests: `third_party/gsplat-geer/tests/test_drive2sim_*.py`. `summary.json` records active techniques under `speedups`.

## What it does

It makes each step faster without changing the algorithm: densification and Gaussian count are unchanged, images match bit for bit, gradients differ only by rounding.

- Fused Adam: one kernel updates a whole parameter group.
- Fused SSIM: one kernel computes the SSIM loss and its gradient.
- Fewer host synchronisations: the CPU waits less for the GPU.
- Exact tile selection: a Gaussian goes only to tiles its ellipse overlaps.
- Two-stage sorting: depth and tile sorting in two cheaper passes.
- Per-Gaussian backward: rasterizer gradients per Gaussian, not per pixel.

## Why this method

OmniRe runs must fit a 12-hour Kaggle T4 session, and a speed-up must not change results. Options:

- Taming 3DGS kernels: per-Gaussian backward and fused SSIM. Faster-GS includes both; Taming's code modifies the Inria rasterizer, not gsplat.
- gsplat packed mode: keeps only visible Gaussian-camera pairs. gsplat calls it more memory efficient, not necessarily faster; the trainer requires `packed=false` for 3DGUT and 3DGEER.
- Speedy-Splat: exact tile intersection plus pruning of over 90 % of Gaussians. Pruning changes the model (FastGS covers that here), and for tile culling the Faster-GS paper found Radl et al.'s method faster.

Faster-GS was chosen because it collects these kernels and a fused optimizer, keeps results identical, and makes each technique a flag. On nuScenes (OmniRe, 3DGS, single runs) training fell from 219 to 198 min; PSNR went from 25.15 to 25.20 and peak memory from 6.6 to 6.9 GB.

## First-principles explanation

A step has four parts: rendering, loss, parameter update, and method-specific work (for OmniRe, gathering and transforming node Gaussians). Faster-GS speeds up rendering, SSIM and Adam; the untouched part limits the gain (Amdahl's law). Profiled at step 20 000, OmniRe spent about 9 % of GPU time in the rasterizer, 14 % in the optimizer and 77 % in its own work. Doubling the speed of that 23 % gives at most about 13 %; the measured gain was about 10 %. In static scenes rendering and the update dominate, hence the paper's four- to fivefold speed-up.

## Difference from the original code, method or paper

Reference: "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm et al., CVPR 2026), code `nerficg-project/faster-gaussian-splatting`.

- The original NeRFICG rasterizer supports only pinhole cameras and returns only colour. OmniRe needs depth, alpha and fisheye, so the techniques were re-implemented in gsplat-geer.
- The paper's fused activations and separate spherical-harmonics buffer are not implemented.
- Skipping invisible Gaussians in the update is unused: gsplat-geer's `SelectiveAdam` also drops Adam's bias correction, changing optimisation.
- Kernel techniques exist only for the standard renderer. For 3DGUT, exact tile selection is unsafe because it evaluates Gaussians along 3D rays; the other two lack a 3DGUT version. 3DGEER's tile selection is already exact.

## Limits, risks and possible gains

- Per-Gaussian backward uses about 300 MB more GPU memory.
- `morton_order` (Z-order sorting of background Gaussians) gives only statistically equivalent results; off by default.
- Larger gains lie in OmniRe's own work: fewer Gaussians (FastGS) or faster node gathering (`omnire_node_gather.md`).
- For 3DGUT the rasterizer is about 14 % of a step; porting the kernels would gain at most about 5 %.
