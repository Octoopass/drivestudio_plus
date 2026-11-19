# 3DGEER renderer

## Where the code is

- The rasterizer is Bosch Research's `gsplat-geer` (`boschresearch/3dgeer`, branch `gsplat-geer`), the submodule `third_party/gsplat-geer`. It adds the argument `with_geer` to `gsplat.rasterization()`.
- `drive2sim/recon/trainers/base.py`: `render_mode: geer` passes `with_geer=True`, and `postprocess_per_train_step` reads `info["geer_gradient"]` for densification.
- `drive2sim/configs/methods/omnire_geer.yaml` inherits `omnire_nocampose.yaml` and sets `render_mode: geer`, `densify_grad_thresh: 0.01` and `refine_interval: 50`.

## What it does

3DGEER integrates each Gaussian's density exactly along every pixel's camera ray, without first projecting it onto the image. Because it works on rays, the same code handles pinhole, OpenCV fisheye and F-theta cameras, including fields of view near or above 180 degrees.

## Why this method

OmniRe must train on fisheye cameras such as KITTI-360's, which DriveStudio's 3DGS renderer rejects. The alternatives were:

- Undistort, then use 3DGS. A wide fisheye image cannot be resampled to one pinhole image without clipping and stretching its border, and the Fisheye-GS paper reports this loss. The project's `prep.undistort` stage is also unwritten.
- 3DGUT projects sigma points of each Gaussian through the exact camera function and then rasterizes. It is in the same package and remains the second fisheye renderer. Here it does not render rays at π/2 or more from the axis. The 3DGEER paper reports grid-line artifacts for it at extreme fields of view and a lower ScanNet++ PSNR (30.64 against 31.50).
- Fisheye-GS changes only the 3DGS projection, for the equidistant model. KITTI-360 needs OpenCV fisheye with four distortion coefficients. The 3DGEER paper also reports lower peripheral PSNR for Fisheye-GS (23.28 against 28.94).

3DGEER is exact for every camera model the project uses and is selected with one config value. On pinhole nuScenes (FastGS and Faster-GS), it scored only 21.77 dB, against 23.99 for 3DGUT and 25.01 for 3DGS. The KITTI-360 fisheye comparison with 3DGUT is planned but not run. Since 19/11/2026 the KITTI-360 loader masks, for 3DGEER, only the car body, the dark region outside the image circle (detected from the images) and rays where the OpenCV fisheye fit drifts more than 0.5 px from Mei (computed per camera from its calibration: 96.1 degrees for `image_02`, 92.5 for `image_03`); 3DGUT is also limited to 88 degrees because it cannot draw rays at 90 degrees or more. LiDAR depth still stops at 90 degrees. Earlier runs, such as `e3e3` and `0b27`, were masked at 1.45 rad (83 degrees). Rim quality with 3DGEER has not been measured on a full run (`docs/datasets/kitti360.md` section 5.1).

## First-principles explanation

A pixel value is the colour accumulated along one ray. Standard 3DGS projects each 3D Gaussian to a 2D Gaussian with a local linear approximation of the projection. This is accurate for a pinhole camera near the image centre, but not for strongly non-linear projections, such as fisheye lenses, or for large Gaussians near the border. 3DGEER evaluates each Gaussian along the pixel's 3D ray in closed form, so it is correct for any camera that gives one ray per pixel.

Instead of the screen-space gradient that 3DGS densification uses, 3DGEER returns `geer_gradient`, which accumulates absolute per-pixel contributions scaled with distance.

## Difference from the original code, method or paper

- The renderer is Bosch's 3DGEER release. The integration follows Bosch's DriveStudio patch, which adds `render_mode` (`default`, `ut`, `geer`) to `trainers/base.py`.
- Bosch densifies every 300 steps, and this project every 50. On nuScenes (three cameras, 30 000 steps), this gave slightly better metrics, with about 6 % longer training, 0.7 GB more memory and 23 % more Gaussians. Street Gaussians with 3DGEER keeps 300, because 50 is untested.
- 3DGEER cannot optimise camera poses, so every 3DGEER configuration removes `CamPose`.

## Limits, risks and possible gains

- The 3DGEER threshold (`0.01`) is on a different scale from the 3DGS (`0.0005`) and 3DGUT thresholds, so they are not interchangeable.
- The roughly 3 dB gap to 3DGS on pinhole nuScenes lies mainly in the static background, while human and vehicle regions were close. Its cause is unknown.
- Only cameras supported by `gsplat-geer` can be trained. Rolling shutter needs exposure start and end poses.
- MCMC densification (`densify_mcmc.md`) ignores the gradient scale, but it was not run with 3DGEER: with FastGS, 3DGEER already keeps as many Gaussians as 3DGS, and MCMC was removed after scoring lower with 3DGS and 3DGUT.
