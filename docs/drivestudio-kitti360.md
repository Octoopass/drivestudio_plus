# KITTI-360 in DriveStudio+

DriveStudio+ has no KITTI-360 loader. KITTI-360 reaches it through NCore and the PhysicalAI loader:

```
KITTI-360 archive --kitti360_to_ncore.py --target drivestudio--> NCore V4 clip
                  --datasets/preprocess.py --dataset physicalai --frame_stride 1--> processed scene
                  --datasets/tools/extract_masks_hf.py--> sky masks
                  --fflab train --backend drivestudio--> Gaussian scene + test metrics
```

The Web UI runs the same chain: step 1 "Chuẩn bị dữ liệu → Cho DriveStudio+", step 2 "Dựng cảnh → DriveStudio+",
step 3 compares runs (metrics are read from `fflab-runs/<run>/fflab-train.log`, so CLI runs appear too).

## What `--target drivestudio` writes

| KITTI-360 | NCore sensor id (PhysicalAI name) | Camera model |
|---|---|---|
| image_00 (front, rectified perspective) | `camera_front_wide_120fov` | F-theta fit of `r = f tan(theta)` up to the image corner (55 deg); < 0.23 px |
| image_02 (left fisheye) | `camera_cross_left_120fov` | F-theta fit of the MEI model up to 92 deg; < 0.6 px |
| image_03 (right fisheye) | `camera_cross_right_120fov` | same |
| Velodyne | `lidar_velodyne` | |
| ROSA 6-class 3D boxes | cuboid tracks | `instance_token` = track id |

- The names are labels the PhysicalAI preprocessor requires (index 0 is its reference clock); the geometry is KITTI-360's.
  image_01 is not written: the preprocessor has no second camera of that kind.
- The ROSA boxes (`ImageSets-rosa-6class/*.pkl`) are in the Velodyne frame of their scan, `[x, y, z, l, w, h, yaw]`
  with z at the box centre. This was checked by projecting them onto image_00 (bottom-centre boxes float half a
  box above the cars). The 6 classes map to Vehicle (car, truck, trailer), Cyclist (motorcycle, bicycle) and Pedestrian.
  `--no-objects` leaves the cuboids empty.
- Sequence 0003, 100 frames from frame 167: 738 m, 140 boxes in 29 tracks (28 vehicles, 1 pedestrian).

## Commands

```bash
export PYTHONPATH=$PWD
P=~/venvs/ds/bin/python   # DriveStudio+ venv: torch 2.6 cu124, gsplat-geer, pytorch3d, nvdiffrast, nvidia-ncore
$P tools/kitti360_to_ncore.py --data ./kitti360 --out data/kitti360/raw_obj/k360_0003_100f \
   --frames 100 --target drivestudio --cameras image_00,image_02,image_03
$P datasets/preprocess.py --data_root data/kitti360/raw_obj --target_dir data/kitti360/processed_obj \
   --dataset physicalai --frame_stride 1 --scene_ids 0 --workers 1 \
   --process_keys images lidar calib pose objects dynamic_masks
$P datasets/tools/extract_masks_hf.py --data_root data/kitti360/processed_obj --scene_ids 0
node fflab/bin/fflab.js train --backend drivestudio --venv ~/venvs/ds --method configs/streetgs_ut.yaml \
   --dataset kitti360/ncore_3cams_objects --run-name k360_ut_obj
```

Dataset configs: `kitti360/ncore_3cams` (no objects, `data/kitti360/processed`) and `kitti360/ncore_3cams_objects`
(`data/kitti360/processed_obj`). `fflab train --overrides "data.data_root=..."` points either at another prepared scene.

## Renderers

Fisheye cameras need `render_mode: ut` or `geer`; `streetgs_fastgs.yaml` and `omnire_fastgs.yaml` set no
render mode (3DGS, pinhole only), so pass `--render-mode ut` with them.

## Fixes made on the way

- `DrivingDataset.instance_num` returned an error instead of 0 for scenes without objects.
- `tools/eval.py` crashed writing numpy float32 metrics to JSON (current numpy).
- The PhysicalAI preprocessor sampled every 3rd frame (30 Hz sources); `--frame_stride` makes it configurable.
