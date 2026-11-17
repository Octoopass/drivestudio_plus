# KITTI-360 vehicle masks

`image_02.png` and `image_03.png` are 1400 × 1400 grayscale binary masks.
255 excludes vehicle hardware; 0 retains the scene. They do not include the
fisheye angle mask, which the loader calculates from calibration.

The outlines were traced from the original KITTI-360 sequence 0003 images and
checked on raw frames 2, 257, 507, 759 and 1009. The earlier inspection also used
197 and 590. The vehicle rim, emergency button, mirror and GPS receiver are
excluded. Shadows cast onto the road and sun streaks are retained. The outline
has a small conservative margin and has not been validated on other sequences.
Cropped, rotated or modified rigs need new masks.

`outlines.json` records pixel coordinates at the reference resolution. To
reproduce the PNGs with Pillow, fill `body_polygon`, `gps_stand_polygon` and
`hardware_ellipses` with 255 on a black L image. This creates `image_02.png`.
Mirror it horizontally to create `image_03.png`; both sides were checked against
source images. No images of the dataset are distributed with these assets.
The loader uses camera names from the manifest, then resizes with nearest
neighbour interpolation. It never estimates the mask from a run's train/test images.
