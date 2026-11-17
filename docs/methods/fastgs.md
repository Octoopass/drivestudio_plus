# FastGS-style densification and pruning

## Where the code is

- `drive2sim/recon/speedups/fastgs.py`: `FastGS`, which scores Gaussians before refinement.
- `drive2sim/recon/speedups/fastgs_rules.py`: defaults, step schedule and view choice. CPU tests: `tests/drive2sim/recon/speedups/`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `fastgs_filter_densify` and `fastgs_extra_culls` apply the scores in `refinement_after()`.
- `drive2sim/recon/trainers/base.py`: calls `FastGS.before_step` before a step, `FastGS.clear_masks` after refinement. Enabled by `trainer.fastgs.enabled: true`.

## What it does

Before each refinement that densifies or prunes, the trainer renders `num_views` (10) training views spread over the clip and cameras. It marks pixels whose min-max normalised L1 error exceeds `loss_thresh` (0.1) and measures each Gaussian's colour contribution to them.

- Densification: a Gaussian is split or cloned only if it passes DriveStudio's gradient test and its error score is at least `score_thresh` (1.0).
- Pruning: from step `prune_from` (15 000), at the last refinement before each opacity reset, Gaussians with opacity below `prune_min_opacity` (0.1) or normalised error score above `prune_score_thresh` (0.9) are removed, in addition to DriveStudio's culling.

It applies to background, vehicle and deformable nodes, not SMPL humans.

## Why this method

The cost of an OmniRe step grows with the Gaussian count, and runs must fit a 12-hour Kaggle T4 session. Options:

- Taming 3DGS: grows the model towards an exact budget with score-based densification. Each node and scene would need a budget; not compared.
- Mini-Splatting: blur split, depth reinitialisation and simplification by sampling, for fewer Gaussians. It replaces more of the pipeline; not compared.
- LightGaussian: prunes a trained model by global significance, then fine-tunes. It compresses after training, so training is not shorter.

FastGS was chosen because it targets training time, only filters DriveStudio's densification and adds pruning, and is MIT-licensed. With Faster-GS on nuScenes (single runs), Gaussians fell from 1.52 to 0.55 million, training from 198 to 149 min, peak memory from 6.9 to 4.7 GB, PSNR from 25.20 to 25.01; LPIPS rose from 0.251 to 0.269.

## First-principles explanation

Densification should add Gaussians where the image is wrong and more would fix it. 3DGS uses the position gradient as a proxy. A large gradient means the optimiser wants to move the Gaussian, not that the image is wrong: a Gaussian near the border can have one while the image is correct.

FastGS asks directly: it finds clearly wrong pixels in several views and densifies only Gaussians that contribute to them. Pruning uses the scores to remove faint Gaussians and those mostly feeding wrong pixels. Fewer Gaussians make node transforms, rasterization and the optimizer cheaper.

## Difference from the original code, method or paper

Reference: the FastGS repository (`github.com/fastgs/FastGS`).

1. FastGS counts each high-error pixel a Gaussian touches as 1, with its own rasterizer. This code weights each pixel by the blending weight (alpha times transmittance), obtained as a gradient through gsplat, so FastGS's threshold of 5 does not transfer.
2. FastGS averages the score over all sampled views; this code over the views that see the Gaussian, because in a forward-driving clip most do not.
3. No pruning budget during densification, and none of FastGS's rasterizer speed-ups.
4. Sampled views are split evenly between cameras, then spread over each camera's frames.

## Limits, risks and possible gains

- `score_thresh` is not tuned per scene; each scoring pass logs score percentiles for tuning.
- Scoring costs about two extra renders per sampled view per refinement.
- Fewer Gaussians can lose fine detail; LPIPS, sensitive to detail, worsens more than PSNR.
- FastGS keeps the gradient gate. With 3DGUT, whose threshold is uncalibrated, filtering relies almost entirely on the error score.
- FastGS cannot be combined with MCMC, which replaces the gradient rule FastGS modifies.
