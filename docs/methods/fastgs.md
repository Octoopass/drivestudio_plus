# FastGS-style densification and pruning

## Where the code is

- `drive2sim/recon/speedups/fastgs.py`: `FastGS`, which scores Gaussians before refinement.
- `drive2sim/recon/speedups/fastgs_rules.py`: defaults, step schedule and view choice. CPU tests: `tests/drive2sim/recon/speedups/`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `fastgs_filter_densify` and `fastgs_extra_culls` apply the scores in `refinement_after()`.
- `drive2sim/recon/trainers/base.py`: calls `FastGS.before_step` before a step and `FastGS.clear_masks` after refinement. `trainer.fastgs.enabled: true` enables it.

## What it does

Before each refinement that densifies or prunes, the trainer renders `num_views` (10) training views spread over the clip and cameras. It marks pixels whose min-max normalised L1 error exceeds `loss_thresh` (0.1) and measures each Gaussian's colour contribution to them.

- Densification: a Gaussian is split or cloned only if it passes DriveStudio's gradient test and its error score is at least `score_thresh` (1.0).
- Pruning: from step `prune_from` (15 000), at the last refinement before each opacity reset, FastGS removes Gaussians with opacity below `prune_min_opacity` (0.1) or normalised error score above `prune_score_thresh` (0.9), besides DriveStudio's culling.

It covers background, vehicle and deformable nodes, not SMPL humans. Outside OmniRe and Street GS it covers DeformableGS, whose `DeformableGaussians` keeps the refinement of `VanillaGaussians` (configs `deformablegs*_fastgs*.yaml`, untested on GPU), but not PVG: `PeriodicVibrationGaussians.refinement_after` has its own densify rule (gradient in space or in time) and does not apply the FastGS masks (`fastgs_supported = False`), so PVG has no FastGS config.

## Why this method

The cost of an OmniRe step (node transforms, rasterization, optimizer) grows with the Gaussian count, and runs must fit a 12-hour Kaggle T4 session. Alternatives were:

- Taming 3DGS grows the model towards an exact budget with score-based densification. It would need a per-node, per-scene budget and was not compared.
- Mini-Splatting uses blur split, depth reinitialisation and simplification by sampling for fewer Gaussians. It replaces more of the pipeline and was not compared.
- LightGaussian prunes a trained model by global significance and fine-tunes it. It compresses after training, so it does not shorten training.

FastGS was chosen because it targets training time, only filters DriveStudio's densification and adds pruning, and is MIT-licensed. With Faster-GS on nuScenes (single runs), Gaussians fell from 1.52 to 0.55 million, training from 198 to 149 min, peak memory from 6.9 to 4.7 GB and PSNR from 25.20 to 25.01, while LPIPS rose from 0.251 to 0.269.

## First-principles explanation

Densification should add Gaussians where the image is wrong and more would fix it. 3DGS uses the position gradient as a proxy. A large gradient means the optimiser wants to move the Gaussian, not that the image is wrong, and a Gaussian near the border can have one while the image is correct.

FastGS instead finds clearly wrong pixels in several views and densifies only Gaussians that contribute to them. Pruning uses the scores to remove faint Gaussians and those mostly feeding wrong pixels.

## Difference from the original code, method or paper

Reference: the FastGS repository (`github.com/fastgs/FastGS`).

1. FastGS counts each high-error pixel a Gaussian touches as 1, using its own rasterizer. This code weights each pixel by the blending weight (alpha times transmittance), obtained as a gradient through gsplat, so FastGS's threshold of 5 does not transfer.
2. FastGS averages the score over all sampled views. This code averages only over the views that see the Gaussian, because in a forward-driving clip most do not.
3. This code omits FastGS's pruning budget during densification and its rasterizer speed-ups.
4. Sampled views are split evenly between cameras and spread over each camera's frames.

## Limits, risks and possible gains

- `score_thresh` is not tuned per scene. Each scoring pass logs percentiles for tuning.
- Scoring costs about two extra renders per sampled view at each refinement.
- Fewer Gaussians can lose fine detail. Detail-sensitive LPIPS worsens more than PSNR.
- FastGS keeps the gradient test. With 3DGUT, whose threshold is uncalibrated, filtering relies almost entirely on the error score.
- FastGS cannot be combined with MCMC, which replaces the gradient rule FastGS modifies. MCMC is no longer an OmniRe option (`densify_mcmc.md`).
