# Static scene with gsplat 3DGUT (approach `gsplat_3dgut`)

## Where the code is

- Approach: `drive2sim/approaches/gsplat_3dgut.yaml`; stage: `drive2sim/stages/train/gsplat_3dgut.py`.
- Trainer: `drive2sim/recon/tools/train_gsplat_3dgut.py`, run as a subprocess.
- Renderer: `gsplat.rasterization(..., with_ut=True, with_eval3d=True)` from gsplat-geer 1.5.3.

## What it does

It reconstructs a static scene with Gaussians and renders pinhole and OpenCV fisheye cameras in the same model.

1. Data: the tool's scene folder (KITTI layout) and the run's split; camera-to-world pose is `ego_pose[frame] · extrinsics[camera]`; images are downscaled by 2.
2. Initialisation: up to 120 000 LiDAR points between 2.5 and 80 m, coloured by the first training camera that sees them. The initial size is the mean distance to the three nearest points, clamped to 3 cm to 1 m; the initial opacity is 0.3.
3. Sky dome (`sky`, on by default): about 400 Gaussians per frame on the upper half of a 200 m sphere around the vehicle, 12 m in size, coloured from the images.
4. Rendering: fisheye cameras use `camera_model="fisheye"` and the radial coefficients k1 to k4. Colour is RGB without spherical harmonics.
5. Loss: L1, or `0.8·L1 + 0.2·(1 − SSIM)` when `ssim` is `fused` or `torch`, plus `0.01·mean(opacity) + 0.01·mean(scale)`.
6. Masks: fisheye pixels whose ray is more than `fisheye_max_angle` (1.45 rad) from the optical axis are excluded. `hood_mask` also excludes the bonnet and the black area outside the fisheye circle, found as image edges that stay fixed across frames.
7. Densification: `gsplat.strategy.MCMCStrategy` from step 500, every 100 steps, up to `cap_max` (1 000 000), stopping at 80 % of the steps.
8. Optimiser: `selective_adam` (default) or `adam`.

## First-principles explanation

A linear approximation of the projection fails for fisheye images, because the fisheye projection is strongly non-linear. 3DGUT (Wu et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting", CVPR 2025) projects sigma points through the exact camera model and evaluates the Gaussian along each pixel's 3D ray. One model can therefore be supervised by pinhole and fisheye images together.

3DGUT does not produce the screen-space gradient that standard densification needs, so MCMC densification is used (`densify_mcmc.md`). LiDAR gives metric 3D starting points. The sky is beyond LiDAR range, so without the dome no Gaussian would cover it. The masks remove pixels that the renderer cannot draw or that never show the scene.

## Difference from the original code, method or paper

The reference trainer is gsplat's `examples/simple_trainer.py` with `with_ut` and `with_eval3d`.

- `simple_trainer.py` initialises from COLMAP structure-from-motion points. This trainer initialises from LiDAR and adds the sky dome.
- `simple_trainer.py` uses spherical harmonics of degree 3. This trainer uses plain RGB.
- The fisheye angle mask and the bonnet mask are not in `simple_trainer.py`.
- `SelectiveAdam` in gsplat-geer (`gsplat/cuda/csrc/AdamCUDA.cu`) updates only visible Gaussians, but it does not apply Adam's bias correction. For the first few hundred steps its update is about 3.2 times larger than standard Adam.
- The position learning rate uses a scene scale of 1.1 times the trajectory radius, clamped to 5–20 m.

## Limits, risks and possible gains

- The scene is static. Moving vehicles become blurred, and there are no actors.
- The bonnet mask can be wrong. On one KITTI-360 clip, a fixed horizontal sun streak was taken as a bonnet edge, and about 40 % of the fisheye image was excluded.
- Sky dome Gaussians remain in the exported `scene.ply` as large Gaussians at 200 m.
- F-theta cameras are not supported by this trainer.
- Possible gains: spherical harmonics of degree 3 and a LiDAR depth loss, which gsplat's newer `av_trainer.py` reports as useful, and comparing `selective_adam` against `adam`.
