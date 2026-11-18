# Asset Harvester (research note, not integrated)

## Status

No code in this repo uses Asset Harvester. This note records what was read on 18/11/2026 for the planned L3 feature (edit actor trajectories, `sim/actors.py`, ARCHITECTURE section 4.8). Sources read: the project page, the Hugging Face model card and the arXiv abstract. The full paper and the GitHub code were not read.

## Problem it targets

Moving a car in an OmniRe scene, for example driving it in the opposite lane, exposes four defects:

- Sides no camera saw are blurred or missing, because the car's Gaussians were fitted only to the observed views.
- Its shadow is baked into the background Gaussians, so it stays at the old place.
- Shading on the body follows the original light direction, so a car turned 180° looks wrong.
- The road behind its old position may be poorly reconstructed.

NVIDIA states that neural scene reconstruction does not produce complete object assets for agent manipulation and large-viewpoint novel views. Asset Harvester (arXiv 2604.18468) builds such assets from driving logs.

## Pipeline

| Step | Model | Role |
|---|---|---|
| 1 | AV Object Mask2former | Instance masks on camera views from NCore logs |
| 2 | C-RADIO | Encodes 1–4 object crops (512×512) plus camera orientation, distance, field of view and box size |
| 3 | SparseViewDiT | Multiview diffusion: generates 16 consistent views, including unseen sides; estimates poses when none are given |
| 4 | Object TokenGS | Feed-forward lift of the 16 views to 3D Gaussians in seconds |

The output is one 3DGS `.ply` per object: position, DC colour, opacity, scale and rotation. Supported classes are cars, trucks, trailers, buses, pedestrians, cyclists, strollers and bins. NVIDIA says the assets plug into NuRec scenes for object-level manipulation.

## What it solves

| Defect | Asset Harvester |
|---|---|
| Unseen sides | Solved: diffusion completes them |
| Shadow, relighting | Not described in the sources read |
| Removing the original car, filling the road | Not described in the sources read |

Limits:

- Colour is DC only, so it does not change with view direction.
- Heavily occluded or out-of-distribution objects can give poor or hallucinated assets (model card).

## Fit with this repo

- **Hardware:** one GPU with compute capability ≥ 8.0 and ≥ 30 GB memory, tested on A100 and H100. Kaggle T4 (16 GB, 7.5) and the local Quadro T2000 cannot run it; a Colab A100 may.
- **Licence:** the weights are under the NVIDIA Open Model License, which allows commercial use. The code licence was not checked.
- **Inputs:** crops, masks, camera poses and boxes are available from the manifest and `prep.masks_segformer`; they need converting from this repo's layout instead of NCore.
- **Insertion:** replace one instance's Gaussians in `RigidNodes` with the asset, transformed into that instance's local frame, then set a new path through `instances_trans` and `instances_quats`.

## Proposed experiment

1. Edit one car's trajectory with its original OmniRe Gaussians and render a video with the approach renderer. This runs locally and shows the four defects.
2. Generate the same car with Asset Harvester on a Colab A100, swap it in, and render the same path. Compare the two videos, and compare both against held-out frames where the car is visible.

## Sources

- [Project page](https://research.nvidia.com/labs/sil/projects/asset-harvester/)
- [arXiv 2604.18468](https://arxiv.org/abs/2604.18468)
- [Model card](https://huggingface.co/nvidia/asset-harvester)
- [Code](https://github.com/NVIDIA/asset-harvester)
