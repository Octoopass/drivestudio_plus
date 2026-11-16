# 3DGUT densification threshold

## Where the code is

- `drive2sim/recon/trainers/base.py`, `postprocess_per_train_step`: the `ut` branch passes `info["means3d"].grad` to the densification rule.
- `drive2sim/recon/trainers/screen_grad.py`: `screen_equivalent_grad` and `ScreenGradMultiTrainer`, which convert the 3DGUT gradient to a pixel-equivalent gradient. Enabled with the stage parameter `ut_densify=screen`, which sets `trainer.type`.
- `drive2sim/recon/utils/densify_stats.py`: optional statistics (`densify_stats=true`), written to `densify_stats.jsonl`.
- `drive2sim/recon/models/gaussians/vanilla.py`: `after_train` and `refinement_after`, the densification rule itself.

## What it does

DriveStudio densifies a Gaussian when its average position gradient is above `densify_grad_thresh` (0.0005). With 3DGUT, this is the gradient of the 3D centre, in metres. The option `ut_densify=screen` converts this 3D gradient into the gradient with respect to the projected 2D centre, in pixels, so that the 3D Gaussian Splatting threshold applies with the same meaning. The statistics module records, at each refinement, percentiles of the quantity and the fraction above the threshold.

## First-principles explanation

The densification rule compares a number with a fixed threshold, so the number must have the same unit and scale for every renderer. For the standard renderer, the number is the gradient with respect to the 2D centre, converted to normalised image coordinates by multiplying with `W/2` and `H/2`. For 3DGUT, the number is the gradient with respect to the 3D centre. Moving a Gaussian at depth `z` by 1 m moves its projection by about `f/z` pixels, where `f` is the focal length in pixels. By the chain rule, the 3DGUT quantity is about `2f/(W·z)` times the standard quantity. For nuScenes at 800 pixels width this is about `1.58/z`, so the quantity is three times smaller at 5 m and thirteen times smaller at 20 m. With the same threshold, 3DGUT densifies far less.

The conversion uses the projection Jacobian `J = ∂u/∂X` (2×3) of the camera that rendered the step. If the loss depends on the 3D centre `X` only through its projection `u`, then `g3d = Jᵀ g2d`. The least-squares solution is `g2d = (J Jᵀ)⁻¹ J g3d`. The Jacobian contains the focal length, the depth and the lens model, so these dataset-dependent factors cancel. The code computes `J` by central differences through `project_camera_model`, uses a closed-form 2×2 inverse, and returns zero where the determinant is below `1e-6` relative to the squared trace.

## Difference from the original code, method or paper

- In Bosch's DriveStudio patch (the `ut` branch of `trainers/base.py`), 3DGUT reuses the 3D Gaussian Splatting threshold of 0.0005 for the 3D gradient, without a scale conversion. Bosch tuned a separate threshold for 3DGEER (`omnire_geer.yaml`) but not for 3DGUT.
- This project keeps that branch unchanged and adds the conversion as a separate trainer class, so both versions can be run and compared. The statistics module only reads values and does not change training.

## Limits, risks and possible gains

- The conversion is approximate. 3DGUT projects sigma points, not a single centre, and part of the 3D gradient comes from the change of the projected shape.
- The conversion cannot recover the effect of `absgrad`. The standard renderer sums the absolute value of each pixel's gradient, while the 3DGUT branch takes the absolute value after summing signed gradients, so opposite contributions cancel. On a nuScenes test over 2000 steps, the conversion raised the fraction of Gaussians above the threshold by 5 to 30 times, but it stayed 6 to 12 times below the standard renderer. This cause is a hypothesis that has not been verified.
- A fixed scale factor is not enough: the ratio between the quantities fell from about 0.15 to 0.04 during training.
- A possible gain is MCMC densification (`densify_mcmc.md`), which uses no gradient threshold.
