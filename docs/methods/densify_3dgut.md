# 3DGUT densification threshold

## Where the code is

- `drive2sim/recon/trainers/base.py`, `postprocess_per_train_step`: the `ut` branch passes `info["means3d"].grad` to densification.
- `drive2sim/recon/trainers/screen_grad.py`: `screen_equivalent_grad` and `ScreenGradMultiTrainer` convert the 3DGUT gradient to pixels. `ut_densify=screen` enables them by setting `trainer.type`.
- `drive2sim/recon/utils/densify_stats.py`: optional statistics (`densify_stats=true`) in `densify_stats.jsonl`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `after_train` and `refinement_after` apply the rule.

## What it does

DriveStudio densifies a Gaussian when its average position gradient exceeds `densify_grad_thresh` (0.0005). With 3DGUT, this is the 3D-centre gradient, in metres. `ut_densify=screen` converts it to the gradient of the projected 2D centre, in pixels, so the 3DGS threshold keeps its meaning. At each refinement, the statistics record percentiles of this quantity and the fraction above the threshold.

## Why this method

3DGUT is one of the two fisheye renderers. The alternatives were:

- The raw threshold densified 60 to 380 times less than 3DGS on nuScenes over 2000 steps. After 30 000 steps it had 40 % fewer Gaussians and 23.99 dB PSNR, against 25.01.
- A hand-tuned smaller threshold cannot fit, because the scale depends on depth and focal length. The ratio between the two quantities also fell from about 0.15 to 0.04 during training.
- MCMC (`densify_mcmc.md`) places a fixed budget of Gaussians by opacity, without gradients. With 3DGUT it reached 23.19 dB, below the raw threshold with FastGS (23.99).
- AbsGS (per-pixel absolute gradients) is enabled for 3DGS (`absgrad: true`), but the 3DGUT kernel in `gsplat-geer` does not accept `absgrad` without CUDA changes.

The conversion was chosen because it is pure Python, cancels depth and focal length by construction, and reuses the tuned 3DGS threshold. Its 30 000-step run is still planned.

## First-principles explanation

The rule compares a number with a fixed threshold, so it must have the same unit and scale for every renderer. For 3DGS, it is the 2D-centre gradient times `W/2` and `H/2`. Moving a Gaussian at depth `z` by 1 m moves its projection about `f/z` pixels, where `f` is the focal length in pixels. The 3DGUT quantity is therefore about `2f/(W·z)` times the standard one. For nuScenes at 800 pixels width this is `1.58/z`, which makes it three times smaller at 5 m and thirteen times at 20 m.

The conversion uses the 2×3 projection Jacobian `J = ∂u/∂X` of the rendering camera. If the loss depends on `X` only through `u`, then `g3d = Jᵀ g2d`, and the least-squares solution is `g2d = (J Jᵀ)⁻¹ J g3d`. `J` contains focal length, depth and lens model, so these cancel. The code computes `J` by central differences through `project_camera_model` and inverts the 2×2 matrix in closed form. It returns zero where the determinant is below `1e-6` times the squared trace.

## Difference from the original code, method or paper

- Bosch's DriveStudio patch (the `ut` branch) applies the 3DGS threshold 0.0005 to the 3D gradient without conversion. Bosch tuned a separate threshold only for 3DGEER.
- This project keeps that branch and adds the conversion as a separate trainer class, so the two can be compared. The statistics do not change training.

## Limits, risks and possible gains

- The conversion is approximate, because 3DGUT projects sigma points and part of the 3D gradient comes from the change of projected shape.
- It cannot recover `absgrad`. The standard renderer sums absolute per-pixel gradients, while the 3DGUT branch takes the absolute value after summing, so opposite contributions cancel. Over 2000 steps on nuScenes, the conversion raised the fraction above the threshold 5 to 30 times, but stayed 6 to 12 times below the standard renderer. This explanation is an unverified hypothesis.
