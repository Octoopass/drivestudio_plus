# MCMC densification for OmniRe

## Where the code is

- `drive2sim/recon/densify/mcmc.py`: class `MCMC` and the function `relocation`.
- `drive2sim/recon/trainers/base.py`: calls `MCMC.step` in `postprocess_per_train_step` and adds `MCMC.regularization` to the losses.
- `drive2sim/stages/train/drivestudio.py` and `drive2sim/approaches/omnire.yaml`: the option `densify: default | mcmc`, which cannot be combined with FastGS.
- Tests: `tests/drive2sim/recon/densify/test_mcmc.py`.

## What it does

When enabled, the background, vehicle and deformable nodes replace DriveStudio's gradient rule, pruning and opacity reset with these steps after each optimizer step:

- every 100 steps from step 500 to 25 000, Gaussians with opacity at or below 0.005 move to living Gaussians sampled by opacity;
- at the same steps, the node grows by 5 % with copies of Gaussians sampled by opacity, up to `cap_max` (1 000 000 background, 50 000 vehicles, 10 000 deformable);
- positions receive noise shaped by each Gaussian's covariance, scaled by `noise_lr` (5·10⁵) times the position learning rate;
- the loss gains `0.01 × mean opacity` and `0.01 × mean scale`.

Relocated and copied Gaussians get a zero Adam state and copy `point_ids` (the vehicle or object they belong to) from the source. Human nodes are not changed, because their Gaussians are attached to SMPL body vertices and are never densified.

## First-principles explanation

DriveStudio adds Gaussians where the position gradient exceeds a threshold. The gradient is in pixels for the standard renderer and in metres for 3DGUT, so one threshold does not fit both. On a nuScenes scene, 3DGUT ended with about 40 % fewer Gaussians and about 1 dB lower PSNR.

MCMC densification asks where a fixed number of Gaussians should be placed. Its only signal is opacity, which means the same for every renderer. An almost transparent Gaussian does not contribute to the images, so it is moved to a region with opaque Gaussians, which is likely to need more detail. When a Gaussian is split into several copies, the opacity and scale of the copies are reduced so that the rendered image stays the same (`relocation` implements equation 9 of the paper). The noise weight is `sigmoid(100·((1 − opacity) − 0.995))`, so only Gaussians with opacity below about 0.005 move noticeably. The regularisation makes unneeded Gaussians fade until they are relocated.

## Difference from the original code, method or paper

The method is Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024, arXiv 2404.09591). The implementation follows `gsplat/strategy/mcmc.py` and `gsplat/strategy/ops.py` in gsplat-geer 1.5.3, with these differences:

- gsplat requires one optimizer per parameter; OmniRe uses one optimizer with named groups, so the updates are written for that layout.
- The relocation formula runs in PyTorch (float64) instead of the CUDA kernel `compute_relocation`; a GPU test shows agreement within 1e-5.
- gsplat resets the Adam state of the source Gaussians only. This code also resets the state of the relocated Gaussians.
- The budget is set per node instead of once for the whole scene.

## Limits, risks and possible gains

- MCMC never removes Gaussians. If a node starts with more Gaussians than its budget, it only relocates and never grows. The nuScenes test scene starts with about 660 000 background Gaussians, so a budget of 500 000 would have blocked all growth.
- Sampling is over the whole node, so a small vehicle could lose all its Gaussians to other vehicles. Sampling within each vehicle would prevent this.
- Noise and regularisation strengths are the paper's defaults, not tuned for driving scenes.
- The gain on OmniRe is not measured yet. The +1.5 dB in gsplat's reference experiment came from a short static clip.
