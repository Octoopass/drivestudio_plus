# 3DGUT densification threshold

## Where the code is

- `drive2sim/recon/trainers/base.py`, `postprocess_per_train_step`: the `ut` branch passes `info["means3d"].grad` to densification.
- `drive2sim/recon/trainers/screen_grad.py`: `screen_equivalent_grad` and `ScreenGradMultiTrainer` convert the 3DGUT gradient to pixels; enabled by `ut_densify=screen`, which sets `trainer.type`.
- `drive2sim/recon/utils/densify_stats.py`: optional statistics (`densify_stats=true`) in `densify_stats.jsonl`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `after_train`, `refinement_after` (the rule).

## What it does

DriveStudio densifies a Gaussian when its average position gradient exceeds `densify_grad_thresh` (0.0005). With 3DGUT this gradient is of the 3D centre, in metres. `ut_densify=screen` converts it to the gradient of the projected 2D centre, in pixels, so the 3DGS threshold keeps its meaning. The statistics record, per refinement, percentiles of the quantity and the fraction above the threshold.

## Why this method

3DGUT is one of the two fisheye renderers. The options are:

- Keep the raw threshold (Bosch's branch): on nuScenes it densified 60 to 380 times less than 3DGS over 2000 steps, and after 30 000 steps had 40 % fewer Gaussians and 23.99 dB PSNR, against 25.01.
- Hand-tune a smaller threshold: one number cannot fit, because the scale depends on depth and focal length, and the ratio between the quantities fell from about 0.15 to 0.04 during training.
- MCMC (`densify_mcmc.md`): places a fixed budget of Gaussians by opacity, without gradients. With 3DGUT it reached 23.19 dB, below the raw threshold (23.99, with FastGS); its budget may be too low.
- Per-pixel absolute gradients (AbsGS): used for 3DGS (`absgrad: true`), but the 3DGUT kernel in `gsplat-geer` does not accept `absgrad`; it needs CUDA changes.

The conversion was chosen because it is pure Python, cancels depth and focal length by construction, and reuses the tuned 3DGS threshold. Its 30 000-step run is planned, not run.

## First-principles explanation

The rule compares a number with a fixed threshold, so the number needs the same unit and scale for every renderer. For 3DGS it is the 2D-centre gradient times `W/2` and `H/2`; for 3DGUT, the 3D-centre gradient. Moving a Gaussian at depth `z` by 1 m moves its projection about `f/z` pixels (`f`: focal length in pixels). So the 3DGUT quantity is about `2f/(W·z)` times the standard one: `1.58/z` for nuScenes at 800 pixels width, three times smaller at 5 m and thirteen times at 20 m.

The conversion uses the projection Jacobian `J = ∂u/∂X` (2×3) of the rendering camera. If the loss depends on `X` only through `u`, then `g3d = Jᵀ g2d`, with least-squares solution `g2d = (J Jᵀ)⁻¹ J g3d`. `J` contains focal length, depth and lens model, so these factors cancel. The code computes `J` by central differences through `project_camera_model`, inverts 2×2 in closed form, and returns zero where the determinant is below `1e-6` times the squared trace.

## Difference from the original code, method or paper

- Bosch's DriveStudio patch (the `ut` branch) reuses the 3DGS threshold 0.0005 for the 3D gradient without conversion. Bosch tuned a separate threshold for 3DGEER, not for 3DGUT.
- This project keeps that branch and adds the conversion as a separate trainer class, so both can be compared. Statistics do not change training.

## Limits, risks and possible gains

- The conversion is approximate: 3DGUT projects sigma points, and part of the 3D gradient comes from the change of projected shape.
- It cannot recover `absgrad`. The standard renderer sums each pixel's absolute gradient; the 3DGUT branch takes the absolute value after summing, so opposite contributions cancel. Over 2000 steps on nuScenes, the conversion raised the fraction above the threshold 5 to 30 times, still 6 to 12 times below the standard renderer. This cause is an unverified hypothesis.
