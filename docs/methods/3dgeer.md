# 3DGEER renderer

## Where the code is

- The rasterizer is the `gsplat-geer` package from Bosch Research (`boschresearch/3dgeer`, branch `gsplat-geer`), kept as the submodule `third_party/gsplat-geer`. It adds the argument `with_geer` to `gsplat.rasterization()`.
- The trainer selects it in `drive2sim/recon/trainers/base.py`: `trainer.render.render_mode: geer` passes `with_geer=True`, and `postprocess_per_train_step` reads `info["geer_gradient"]` for densification.
- The OmniRe configuration is `drive2sim/configs/methods/omnire_geer.yaml`. It inherits `omnire_nocampose.yaml` and sets `render_mode: geer`, `densify_grad_thresh: 0.01` and `refine_interval: 50`.

## What it does

3DGEER renders each Gaussian by integrating its density exactly along the camera ray of every pixel. It does not project the Gaussian onto the image plane first. Because the computation works on rays, the same code handles pinhole, OpenCV fisheye and F-theta cameras, including fields of view close to or above 180 degrees.

## First-principles explanation

A pixel value is the colour accumulated along one ray through the scene. Standard 3D Gaussian Splatting approximates this by projecting each 3D Gaussian to a 2D Gaussian on the image with a local linear approximation of the camera projection. This approximation is accurate for a pinhole camera near the image centre. It becomes inaccurate when the projection is strongly non-linear, which is the case for wide-angle and fisheye lenses and for large Gaussians near the image border. 3DGEER avoids the approximation. It evaluates each 3D Gaussian directly along the 3D ray of the pixel in closed form, so the contribution does not depend on a linearised projection. The result stays correct for any camera model that can produce a ray per pixel.

Densification needs a signal that says which Gaussians should be split or cloned. 3DGEER does not produce the screen-space gradient that 3D Gaussian Splatting uses. It returns its own quantity, `geer_gradient`, which accumulates absolute per-pixel contributions and scales them with distance.

## Difference from the original code, method or paper

- The renderer is the method of Bosch's 3DGEER release. The renderer integration into DriveStudio follows Bosch's DriveStudio patch, which adds `render_mode` (`default`, `ut`, `geer`) to `trainers/base.py`.
- Bosch's `omnire_geer.yaml` densifies every 300 steps. This project densifies every 50 steps (`refine_interval: 50`). On a nuScenes scene with three cameras and 30 000 steps, this setting gave slightly better image metrics than 300 steps, at the cost of about 6 % longer training, 0.7 GB more GPU memory and 23 % more Gaussians. Street Gaussians with 3DGEER still uses 300 because 50 was not tested there.
- 3DGEER cannot optimise camera poses, so `CamPose` is removed from the model in every 3DGEER configuration.

## Limits, risks and possible gains

- The densification threshold of 3DGEER (`0.01`) is on a different scale from the threshold of 3D Gaussian Splatting (`0.0005`) and of 3DGUT. The thresholds cannot be exchanged between renderers.
- On the pinhole nuScenes scene, OmniRe with 3DGEER scored about 3 dB PSNR lower than OmniRe with the standard renderer, while the human and vehicle regions were close. The gap is therefore mainly in the static background. The cause is not yet known.
- Training is limited to the cameras that `gsplat-geer` supports. Rolling shutter is modelled only for datasets that provide exposure start and end poses.
- A possible gain is to replace the gradient threshold with MCMC densification (`densify_mcmc.md`), which does not depend on the renderer's gradient scale.
