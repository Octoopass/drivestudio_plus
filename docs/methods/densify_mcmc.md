# MCMC densification for OmniRe

## Status

Removed from the OmniRe options on 2026-11-19 after a negative result. Both 30 000-step runs on nuScenes `scene-0061` (`front3`, Faster-GS) scored below their baselines: 3DGS 23.56 dB against 25.20 (`a022` against `8d9c`), 3DGUT 23.19 against 23.99 with FastGS (`44d8` against `60e5`). The background lost detail visibly, and vehicles lost the most PSNR.

3DGEER with MCMC was not run, for these reasons:

- With FastGS, 3DGEER keeps as many background Gaussians as 3DGS (0.39 million at the end of `a8a9` and `f8a6`) yet scores 3.2 dB lower, so the number and placement of Gaussians, which is all MCMC changes, is not what limits it.
- The 3DGEER gap lies mainly in the static background, where MCMC also lost quality with 3DGS.
- Bosch recommends the default strategy for 3DGEER (`third_party/gsplat-geer/README.md`). Its MCMC benchmark scripts come from upstream gsplat and use 3DGUT (`--with_ut`), and the 3DGEER paper (arXiv 2505.24053) does not use MCMC.

The code and tests stay so that `a022` and `44d8` remain readable and resumable.

## Where the code is

- `drive2sim/recon/densify/mcmc.py`: `MCMC` and `relocation`, tested in `tests/drive2sim/recon/densify/test_mcmc.py`.
- `drive2sim/recon/trainers/base.py`: `postprocess_per_train_step` calls `MCMC.step` and adds `MCMC.regularization` to the losses.
- `drive2sim/stages/train/drivestudio.py`: the stage parameter `densify: default | mcmc` (incompatible with FastGS). The OmniRe approach no longer offers it (see Status); `--set train.drivestudio.densify=mcmc` still enables it.

## What it does

In background, vehicle and deformable nodes (SMPL humans are never densified), MCMC replaces the gradient rule, pruning and opacity reset:

- Every 100 steps from 500 to 25 000, Gaussians with opacity ≤ 0.005 move to living Gaussians sampled by opacity. Each node also adds 5 % copies, sampled the same way, up to `cap_max` (1 000 000 background, 50 000 vehicle, 10 000 deformable).
- Positions get covariance-shaped noise scaled by `noise_lr` (5·10⁵) times the position learning rate.
- The loss gains `0.01·mean(opacity)` and `0.01·mean(scale)`.

New Gaussians get a zero Adam state and copy `point_ids` (their vehicle or object).

## Why this method

Alternatives to the default rule (tuned for 3DGS) were:

- AbsGS is enabled for 3DGS (`absgrad: true`), but the 3DGUT kernel does not support it.
- Rota Bulò et al. (ECCV 2024) revise densification. It was not compared, and gsplat-geer has only its cloning opacity correction.
- FastGS (`fastgs.md`) only filters the gradient rule, keeping its renderer-dependent scale.

MCMC was chosen because its only signal, opacity, means the same in every renderer, and gsplat-geer has a reference implementation. In single nuScenes runs with Faster-GS, it was clearly worse. With 3DGS, its PSNR/SSIM/LPIPS were 23.56/0.716/0.403 (151 min, 3.8 GB), against 25.20/0.785/0.251 (198 min, 6.9 GB) for the default rule. With 3DGUT it reached 23.19 dB, against 23.99 with the raw threshold and FastGS.

Both MCMC runs filled every cap, but the cap is unlikely to be the main cause: the FastGS run reached 25.01 dB with only 0.39 million background and 29 000 vehicle Gaussians. Against the default rule, MCMC with 3DGS lost most PSNR on vehicles (20.82 against 25.04) and dynamic regions (21.76 against 24.81), and less on non-sky regions (23.48 against 25.21). The first two limits below are an unverified explanation, which a background-only MCMC run will test.

## First-principles explanation

An almost transparent Gaussian does not affect the images, so MCMC moves it to opaque regions, which likely need detail. A split reduces the copies' opacity and scale so the image stays the same (`relocation`, equation 9 of the paper). The noise weight is `sigmoid(100·((1−opacity)−0.995))`, so only Gaussians with opacity below about 0.005 move noticeably.

## Difference from the original code, method or paper

The method is Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024, arXiv 2404.09591). The code follows `gsplat/strategy/mcmc.py` and `ops.py` in gsplat-geer 1.5.3, except:

- gsplat needs an optimizer per parameter, while OmniRe has one with named groups.
- Relocation runs in PyTorch (float64), not the CUDA kernel `compute_relocation`, and agrees with it within 1e-5 on GPU.
- Unlike gsplat, which resets Adam state only for source Gaussians, this code also resets relocated ones.
- The budget is per node, not per scene.

## Limits, risks and possible gains

- Sampling covers the whole vehicle node, so relocation can move Gaussians between vehicles and empty a small one. Per-vehicle sampling would prevent this.
- Noise and regularisation use the paper's defaults, untuned for driving, and in each vehicle's local frame noise and relocation use the background's factors.
- MCMC never removes Gaussians, so a node above budget only relocates. With about 660 000 initial background Gaussians on nuScenes, a 500 000 budget would block growth.
- gsplat's reference gain of 1.5 dB came from a short static clip.
