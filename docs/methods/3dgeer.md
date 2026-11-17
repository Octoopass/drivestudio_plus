# 3DGEER renderer

## Where the code is

- The rasterizer is Bosch Research's `gsplat-geer` (`boschresearch/3dgeer`, branch `gsplat-geer`), the submodule `third_party/gsplat-geer`. It adds the argument `with_geer` to `gsplat.rasterization()`.
- `drive2sim/recon/trainers/base.py`: `render_mode: geer` passes `with_geer=True`; `postprocess_per_train_step` reads `info["geer_gradient"]` for densification.
- `drive2sim/configs/methods/omnire_geer.yaml` inherits `omnire_nocampose.yaml` and sets `render_mode: geer`, `densify_grad_thresh: 0.01`, `refine_interval: 50`.

## What it does

3DGEER integrates each Gaussian's density exactly along every pixel's camera ray, without projecting the Gaussian onto the image first. Working on rays, the same code handles pinhole, OpenCV fisheye and F-theta cameras, including fields of view near or above 180 degrees.

## Why this method

OmniRe must train on fisheye cameras such as KITTI-360's, which DriveStudio's 3DGS renderer rejects. The options are:

- Undistort, then 3DGS: resample each fisheye image to a pinhole image. A wide fisheye image cannot become one pinhole image without clipping and stretching its border; the Fisheye-GS paper reports this loss. The project's `prep.undistort` stage is also not written.
- 3DGUT: projects sigma points of each Gaussian through the exact camera function, then rasterizes. It is in the same package and kept as the second fisheye renderer. Here it does not render rays at π/2 or more from the axis, and the 3DGEER paper reports grid-line artifacts at extreme fields of view and lower ScanNet++ PSNR (30.64 against 31.50).
- Fisheye-GS: changes only the 3DGS projection, for the equidistant model. KITTI-360 needs OpenCV fisheye with four distortion coefficients, and the 3DGEER paper reports lower peripheral PSNR for Fisheye-GS (23.28 against 28.94).

3DGEER is exact for every camera model the project uses and is selected with one config value. It has not won yet: on pinhole nuScenes (FastGS + Faster-GS) it reached 21.77 dB, against 23.99 for 3DGUT and 25.01 for 3DGS. The KITTI-360 fisheye comparison with 3DGUT is planned, not run. Rays beyond 1.45 rad are currently masked because of the calibration fit, so the wider view is not yet used.

## First-principles explanation

A pixel value is the colour accumulated along one ray. Standard 3DGS projects each 3D Gaussian to a 2D Gaussian with a local linear approximation of the camera projection. This is accurate for a pinhole camera near the image centre, not for strongly non-linear projections: fisheye lenses, or large Gaussians near the border. 3DGEER evaluates each Gaussian along the pixel's 3D ray in closed form, so it is correct for any camera that gives a ray per pixel.

3DGEER does not produce the screen-space gradient that 3DGS densification uses. It returns `geer_gradient`, which accumulates absolute per-pixel contributions scaled with distance.

## Difference from the original code, method or paper

- The renderer is Bosch's 3DGEER release. The integration follows Bosch's DriveStudio patch, which adds `render_mode` (`default`, `ut`, `geer`) to `trainers/base.py`.
- Bosch densifies every 300 steps; this project uses 50. On nuScenes (three cameras, 30 000 steps) this gave slightly better metrics, with about 6 % longer training, 0.7 GB more memory and 23 % more Gaussians. Street Gaussians with 3DGEER keeps 300 (50 untested).
- 3DGEER cannot optimise camera poses, so every 3DGEER configuration removes `CamPose`.

## Limits, risks and possible gains

- The 3DGEER threshold (`0.01`) has a different scale from 3DGS (`0.0005`) and 3DGUT; thresholds are not interchangeable.
- On pinhole nuScenes, OmniRe with 3DGEER was about 3 dB below 3DGS, while human and vehicle regions were close. The gap is mainly static background; the cause is unknown.
- Only cameras supported by `gsplat-geer` can be trained. Rolling shutter needs exposure start and end poses.
- MCMC densification (`densify_mcmc.md`) does not depend on the gradient scale and could replace the threshold.
