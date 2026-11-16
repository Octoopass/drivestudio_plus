# FastGS-style densification and pruning

## Where the code is

- `drive2sim/recon/speedups/fastgs.py`: class `FastGS`, which scores Gaussians before refinement.
- `drive2sim/recon/speedups/fastgs_rules.py`: default values, the step schedule and the choice of views. CPU tests are in `tests/drive2sim/recon/speedups/`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `fastgs_filter_densify` and `fastgs_extra_culls` apply the scores inside `refinement_after()`.
- `drive2sim/recon/trainers/base.py`: calls `FastGS.before_step` before a step and `FastGS.clear_masks` after refinement. It is enabled with `trainer.fastgs.enabled: true`.

## What it does

Before every refinement step that densifies or prunes, the trainer renders `num_views` (10) training views spread over the clip and over the cameras. In each view it marks pixels whose min-max normalised L1 error is above `loss_thresh` (0.1). It then measures how much colour each Gaussian contributes to the marked pixels.

- Densification: a Gaussian is split or cloned only if it passes DriveStudio's gradient test and its error score is at least `score_thresh` (1.0).
- Pruning: from step `prune_from` (15 000), at the last refinement before each opacity reset, Gaussians with opacity below `prune_min_opacity` (0.1) or a normalised error score above `prune_score_thresh` (0.9) are removed in addition to DriveStudio's own culling.

It applies to the background, vehicle and deformable nodes, not to SMPL human nodes.

## First-principles explanation

Densification should add Gaussians where the rendered image is wrong and more Gaussians would fix it. The standard rule in 3D Gaussian Splatting uses the size of the position gradient as a proxy for this. A large gradient means that the optimiser wants to move the Gaussian, but it does not prove that the image is wrong at that place. A Gaussian near the image border can have a large gradient while the image is already correct.

FastGS asks the question directly. It renders several views, finds the pixels that are clearly wrong in those views, and keeps densification only for Gaussians that contribute to those pixels. Pruning uses the same scores to remove faint Gaussians and Gaussians that mostly contribute to wrong pixels. Fewer Gaussians make every part of a step cheaper, because node transforms, rasterization and the optimizer all scale with their number. On a nuScenes scene with OmniRe, the final count fell from about 1.5 to 0.55 million Gaussians and training time by about a quarter, with a small loss in quality.

## Difference from the original code, method or paper

The reference is the FastGS repository (`github.com/fastgs/FastGS`).

1. FastGS counts each high-error pixel that a Gaussian touches as 1, using its own rasterizer. This code weights each pixel by the Gaussian's blending weight (alpha times transmittance), obtained as a gradient through the existing gsplat rasterizer. FastGS's threshold of 5 therefore does not transfer.
2. FastGS averages the score over all sampled views. This code averages over the views that actually see the Gaussian, because in a forward-driving clip most views do not see a given Gaussian.
3. There is no pruning budget during densification, and none of FastGS's rasterizer speed-ups are included.
4. The sampled views are split evenly between cameras and then spread over each camera's frames.

## Limits, risks and possible gains

- `score_thresh` is not tuned per scene. Each scoring pass logs score percentiles, which can be used to tune it.
- Scoring costs about two extra renders per sampled view per refinement.
- With fewer Gaussians, fine detail can be lost. LPIPS, which is sensitive to detail, worsens more than PSNR.
- FastGS keeps DriveStudio's gradient gate. With 3DGUT, whose gradient threshold is not calibrated, the filtering relies almost entirely on the error score.
- FastGS cannot be combined with MCMC densification, because MCMC replaces the gradient rule that FastGS modifies.
