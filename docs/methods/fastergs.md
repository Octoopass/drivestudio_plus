# Faster-GS speed-ups

## Where the code is

- Python part: `drive2sim/recon/speedups/fastergs.py` (class `FasterGS`) and `drive2sim/recon/trainers/base.py`. Each technique has a flag in `trainer.fastergs`: `fused_adam`, `fused_ssim`, `fewer_syncs`, and `morton_order` (off by default).
- Kernel part: the project's fork of gsplat-geer in `third_party/gsplat-geer`. Each technique is a new argument of `rasterization()` that keeps the old behaviour by default: `exact_tiles`, `two_stage_sort`, `per_gaussian_backward`. The fork reports which renderer supports which technique through `gsplat.fastergs_support()`, and the trainer only enables supported ones.
- GPU equivalence tests: `third_party/gsplat-geer/tests/test_drive2sim_*.py`. Active techniques are recorded under `speedups` in `summary.json`.

## What it does

It makes each training step faster without changing the algorithm. Densification, the number of Gaussians and the results stay the same. Images match bit for bit, and gradients differ only at the level of floating-point rounding.

- Fused Adam updates all parameters of a group in one kernel.
- Fused SSIM computes the SSIM loss and its gradient in one kernel.
- Fewer host synchronisations remove points where the CPU waits for the GPU.
- Exact tile selection assigns a Gaussian only to the image tiles that its ellipse really overlaps.
- Two-stage sorting sorts Gaussians by depth and tiles in two cheaper passes.
- Per-Gaussian backward computes the rasterizer gradient per Gaussian instead of per pixel.

## First-principles explanation

A training step has four parts: rendering (forward and backward), the loss, the parameter update, and method-specific work. For OmniRe the method-specific work is gathering and transforming the Gaussians of the background, vehicles and humans. Faster-GS speeds up rendering, the SSIM loss and the Adam update. The total gain is limited by the part that it does not touch (Amdahl's law). In a profile of OmniRe at step 20 000, the rasterizer took about 9 % of GPU time per step, the optimizer about 14 %, and the remaining 77 % was OmniRe's own work. Making 23 % of the step twice as fast gives at most about 13 % on the whole step. The measured training time fell by about 10 %, which matches this estimate. For static scenes, rendering and the update dominate the step, which is why the paper reports four to five times faster training.

## Difference from the original code, method or paper

The reference is "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" (Hahlbohm et al., CVPR 2026) and its code `nerficg-project/faster-gaussian-splatting`.

- The original runs in the NeRFICG framework with its own rasterizer, which supports only pinhole cameras and returns only colour. OmniRe needs depth, alpha and fisheye cameras, so the techniques were re-implemented inside gsplat-geer instead of using the original rasterizer.
- Fused activations and the separate spherical-harmonics buffer of the paper are not implemented.
- Skipping invisible Gaussians in the update is not used. gsplat-geer's `SelectiveAdam` also removes Adam's bias correction, which changes the optimisation.
- The kernel techniques are implemented only for the standard renderer. For 3DGUT, exact tile selection is unsafe because it evaluates Gaussians along 3D rays, and the other two techniques do not yet have a 3DGUT version. For 3DGEER, tile selection is already exact.

## Limits, risks and possible gains

- Per-Gaussian backward uses about 300 MB more GPU memory.
- `morton_order` (Z-order sorting of background Gaussians) gives only statistically equivalent results; it is off by default.
- A larger gain for OmniRe would come from the method-specific part: fewer Gaussians (FastGS) or faster node gathering (`omnire_node_gather.md`).
- For 3DGUT, the rasterizer is only about 14 % of a step, so porting the kernel techniques to 3DGUT would make a step at most about 5 % faster.
