# MCMC densification for OmniRe

## Where the code is

- `drive2sim/recon/densify/mcmc.py`: `MCMC` and `relocation`.
- `drive2sim/recon/trainers/base.py`: calls `MCMC.step` in `postprocess_per_train_step`, adds `MCMC.regularization` to the losses.
- `drive2sim/stages/train/drivestudio.py`, `drive2sim/approaches/omnire.yaml`: the option `densify: default | mcmc`, not combinable with FastGS.
- Tests: `tests/drive2sim/recon/densify/test_mcmc.py`.

## What it does

Background, vehicle and deformable nodes replace the gradient rule, pruning and opacity reset with:

- every 100 steps from 500 to 25 000, Gaussians with opacity ≤ 0.005 move to living Gaussians sampled by opacity;
- at those steps, the node grows 5 % with copies sampled by opacity, up to `cap_max` (1 000 000 background, 50 000 vehicles, 10 000 deformable);
- positions get covariance-shaped noise scaled by `noise_lr` (5·10⁵) times the position learning rate;
- the loss gains `0.01·mean(opacity)` and `0.01·mean(scale)`.

New Gaussians get a zero Adam state and copy `point_ids` (their vehicle or object). Human nodes are unchanged: their Gaussians sit on SMPL vertices and are never densified.

## Why this method

DriveStudio's gradient threshold has a different scale per renderer and the project needs one rule for all three. Besides the default rule (tuned for 3DGS), options are:

- AbsGS: sums absolute per-pixel gradients so large Gaussians in detailed regions split. It is on for 3DGS (`absgrad: true`), unsupported by the 3DGUT kernel, and renderer-scaled.
- "Revising Densification in Gaussian Splatting" (Rota Bulò et al., ECCV 2024): per-pixel error densification, opacity correction for cloning, and a primitive limit. gsplat-geer has only the opacity correction; not compared.
- FastGS (`fastgs.md`): filters the gradient rule by image error; the scale problem stays.

MCMC was chosen because its only signal, opacity, means the same in every renderer, and gsplat-geer has a reference implementation. It was clearly worse (nuScenes, single runs, Faster-GS). With 3DGS it gave PSNR 23.56, SSIM 0.716, LPIPS 0.403 in 151 min and 3.8 GB; the default rule gave 25.20, 0.785, 0.251 in 198 min and 6.9 GB. With 3DGUT: 23.19 dB, against 23.99 with the raw threshold and FastGS. The cause is unconfirmed; the budget may be too low: the default run ended with 1.52 million Gaussians; the budgets sum to 1.06 million.

## First-principles explanation

MCMC asks where a fixed number of Gaussians should go. An almost transparent Gaussian does not affect the images, so it moves to opaque regions, which likely need detail. A split reduces the copies' opacity and scale so the image stays the same (`relocation`, equation 9 of the paper). The noise weight is `sigmoid(100·((1 − opacity) − 0.995))`, so only Gaussians with opacity below about 0.005 move noticeably. Regularisation fades unneeded Gaussians until relocated.

## Difference from the original code, method or paper

Method: Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024, arXiv 2404.09591). It follows `gsplat/strategy/mcmc.py` and `ops.py` in gsplat-geer 1.5.3, except:

- gsplat needs an optimizer per parameter; OmniRe has one, with named groups.
- Relocation runs in PyTorch (float64) instead of the CUDA kernel `compute_relocation`; a GPU test agrees within 1e-5.
- gsplat resets Adam state of source Gaussians only; this code also resets relocated ones.
- The budget is per node, not per scene.

## Limits, risks and possible gains

- MCMC never removes Gaussians; a node above budget only relocates. nuScenes starts with about 660 000 background Gaussians, so a 500 000 budget would block growth.
- Sampling is over the whole node, so a small vehicle could lose all its Gaussians; per-vehicle sampling would prevent this.
- Noise and regularisation use the paper's defaults, untuned for driving.
- gsplat's reference +1.5 dB came from a short static clip.
