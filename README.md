<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/media/logo.png">
    <img alt="Logo" src="docs/media/logo_clipped.png" width="700">
  </picture>
</p>
<p align="center">
A 3DGS framework for omni urban scene reconstruction and simulation!
</p>

<p align="center">
    <!-- project -->
    <a href="https://ziyc.github.io/omnire/"><img src="https://img.shields.io/badge/Project-Page-FFFACD" height="28"/></a>
    <!-- paper -->
    <a href="https://arxiv.org/abs/2408.16760">
        <img src='https://img.shields.io/badge/arXiv-Paper-E6E6FA' height="28"/>
    </a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08e6c613-f61a-4d0d-a2a9-1538fcd4f5ff" width="49%" style="max-width: 100%; height: auto;" />
  <img src="https://github.com/user-attachments/assets/d2a47e7d-2934-46de-94d6-85ea8a52aba6" width="49%" style="max-width: 100%; height: auto;" />
</p>

## About this fork (drivestudio_plus)

This repository is a fork of [ziyc/drivestudio](https://github.com/ziyc/drivestudio) at commit `e59bda4`, extended for the project *Reconstructing Real-world Driving Scenes*. The upstream README continues below, with the installation steps adjusted for this fork.

### Layout on branch `drive2sim`

On this branch all code lives in one Python package, `drive2sim/`. The DriveStudio code moved as follows, with imports and config import strings updated and no change in behaviour:

| Before | Now |
|---|---|
| `datasets/<dataset>/<dataset>_sourceloader.py` | `drive2sim/recon/data/loaders/<dataset>.py` |
| `datasets/base/`, `datasets/driving_dataset.py`, `datasets/dataset_meta.py` | `drive2sim/recon/data/` (same names) |
| `datasets/preprocess.py` | `drive2sim/recon/preprocess/run.py` |
| `datasets/<dataset>/<dataset>_preprocess.py`, `_human_utils.py`, `_preview.py` | `drive2sim/recon/preprocess/<dataset>/process.py`, `human.py`, `preview.py` |
| `datasets/tools/extract_masks.py`, `extract_masks_hf.py` | `drive2sim/recon/preprocess/masks/segformer_mmseg.py`, `segformer_hf.py` |
| `datasets/tools/extract_smpl.py`, `humanpose_process.py`, `postprocess.py` | `drive2sim/recon/preprocess/smpl/extract.py`, `run.py`, `postprocess.py` |
| `datasets/tools/multiprocess_utils.py` | `drive2sim/recon/preprocess/multiprocess.py` |
| `models/` | `drive2sim/recon/models/` |
| `models/trainers/` | `drive2sim/recon/trainers/` |
| `models/video_utils.py` | `drive2sim/recon/rendering/videos.py` |
| `utils/` | `drive2sim/recon/utils/` |
| `tools/`, `utils/gather_results.py` | `drive2sim/recon/tools/` |
| `data/ego_masks/`, `data/*.txt` | `drive2sim/assets/ego_masks/`, `drive2sim/assets/scene_lists/` |
| `configs/<method>.yaml`, `configs/paper_legacy/` | `drive2sim/configs/methods/` |
| `configs/datasets/` | `drive2sim/configs/datasets/` |
| `configs/3dgeer.md`, `configs/fastgs.md` | `docs/methods/` |
| `docs/<Dataset>.md` | `docs/datasets/<dataset>.md` |

Install with `pip install -e .` from the repository root (or keep `export PYTHONPATH=$(pwd)`), and run scripts as modules, for example `python -m drive2sim.recon.tools.train`. `smplx` and `hmr2` are imported as installed packages (`pip install -e third_party/smplx`, and `third_party/Humans4D` as described in `docs/HumanPose.md`). The design of the drive2sim tool is in `docs/drive2sim/`.

### What this fork adds

| Change | Where | Guide |
|---|---|---|
| **Choice of renderer**: standard 3DGS, 3DGUT or 3DGEER, selected with `trainer.render.render_mode: default \| ut \| geer`. Also adds native fisheye / FTheta and rolling-shutter training, the PhysicalAI dataset and an interactive viewer. This is Bosch's DriveStudio patch from [boschresearch/3dgeer](https://github.com/boschresearch/3dgeer) (branch `gsplat-geer`, commit `f15cee2`), applied unchanged. | `drive2sim/recon/trainers/base.py`, `drive2sim/recon/tools/viewer.py`, `drive2sim/recon/data/loaders/physicalai.py`, `drive2sim/recon/preprocess/physicalai/`, and others | [docs/methods/3dgeer.md](docs/methods/3dgeer.md) |
| **FastGS-style multi-view densification and pruning**: Gaussians are added only where several training views agree the reconstruction is wrong, and low-opacity Gaussians are pruned late in training. Off by default; enabled with `trainer.fastgs.enabled: true`. Works with all three renderers. | `drive2sim/recon/trainers/base.py`, `drive2sim/recon/models/gaussians/vanilla.py`, `drive2sim/recon/models/nodes/rigid.py` | [docs/methods/fastgs.md](docs/methods/fastgs.md) |
| **Configs without camera-pose refinement**, so that training and held-out frames use the same dataset poses. | `drive2sim/configs/methods/*_nocampose.yaml` | |
| **Sky and dynamic masks without mmcv**: a drop-in replacement for `extract_masks.py` that uses the Hugging Face SegFormer-B5 port and runs in the main environment. | `drive2sim/recon/preprocess/masks/segformer_hf.py` | |
| **World-frame `.ply` export**: writes the static background (optionally with vehicles baked at one frame) in the standard INRIA 3DGS layout, plus a `scene.json` sidecar. DriveStudio's own export normalises positions into a unit box. | `drive2sim/recon/tools/export_ply.py` | |

### New configs

| Config | Method | Renderer | CamPose | FastGS |
|---|---|---|---|---|
| `streetgs_nocampose.yaml` | Street Gaussians | 3DGS | off | off |
| `omnire_nocampose.yaml` | OmniRe | 3DGS | off | off |
| `streetgs_ut.yaml` | Street Gaussians | 3DGUT | off | off |
| `streetgs_geer.yaml` | Street Gaussians | 3DGEER | off | off |
| `omnire_geer.yaml` (from the Bosch patch) | OmniRe | 3DGEER | off | off |
| `streetgs_fastgs.yaml` | Street Gaussians | 3DGS | off | on |
| `streetgs_geer_fastgs.yaml` | Street Gaussians | 3DGEER | off | on |
| `omnire_fastgs.yaml` | OmniRe | 3DGS | off | on |

3DGEER cannot optimise camera poses, so every 3DGUT / 3DGEER config removes `CamPose`; the other new configs remove it too, so that runs differ only in the renderer or the densification rule. The 3DGEER configs use `densify_grad_thresh: 0.01` and `refine_interval: 300`, because 3DGEER's densification gradients are much larger than those of 3DGS.

### How the FastGS-style densification works

Before each refinement step that densifies or prunes, the trainer renders a few training views spread evenly over the clip, marks the high-error pixels in each, and measures how much every Gaussian contributes to those pixels. A Gaussian is split or duplicated only if it passes DriveStudio's usual gradient test *and* this multi-view score. From step 15,000 on, Gaussians with low opacity or a very high error score are removed just before each opacity reset.

It follows [FastGS](https://github.com/fastgs/FastGS) in spirit but is not a port:

- Each high-error pixel counts with the Gaussian's blending weight instead of as 1. The weight is obtained as a gradient through the existing rasterizer, so no CUDA code changes and it works with 3DGEER.
- Scores are averaged over the views that actually see a Gaussian. In a forward-driving clip most views do not see a given Gaussian, so averaging over all sampled views would block nearly all densification.
- FastGS's pruning budget and its rasterizer speed-ups are not included.

It applies to `Background`, `RigidNodes` and `DeformableNodes`; `SMPLNodes` and PVG are unchanged. Settings and tuning are described in [docs/methods/fastgs.md](docs/methods/fastgs.md).

### Results so far

nuScenes v1.0-mini, scene 000, front camera, 800x450, 30k iterations, every 10th frame held out, one Kaggle T4 (16 GB).

| Run | Test PSNR | Gaussians | Peak VRAM | Training time |
|---|---|---|---|---|
| `streetgs.yaml`, upstream DriveStudio with gsplat 1.3.0 | 23.15 | 1.23M | 14.9 GB | 2h03m |
| `streetgs_geer.yaml` | 24.44 | 4.23M | 14.9 GB | 4h47m |
| `streetgs_fastgs.yaml` | **27.64** | 0.36M | 7.4 GB | 1h13m (including evaluation) |

These runs differ in more than one setting (densification rule, CamPose on or off, gsplat version), and the control runs that would separate them have not been done. The table shows what each configuration reached, not how much of the difference is due to the FastGS-style rule. All numbers come from a single scene and a single seed. The first two runs were made before this fork existed, from upstream DriveStudio with the same settings applied at run time.

### Using the added tools

```shell
export PYTHONPATH=$(pwd)

# sky and dynamic masks for processed scenes (replaces drive2sim/recon/preprocess/masks/segformer_mmseg.py)
python -m drive2sim.recon.preprocess.masks.segformer_hf \
    --data_root data/nuscenes/processed_10Hz/mini \
    --start_idx 0 --num_scenes 1 --process_dynamic_mask

# train with the FastGS-style densification
python -m drive2sim.recon.tools.train \
    --config_file drive2sim/configs/methods/streetgs_fastgs.yaml \
    --output_root $output_root --project $project --run_name $expname \
    dataset=nuscenes/1cams data.scene_idx=0 \
    data.pixel_source.load_smpl=False data.pixel_source.test_image_stride=10

# export a standard 3DGS .ply in the world frame (add --rigid_frame 0 to bake vehicles at frame 0)
python -m drive2sim.recon.tools.export_ply --ckpt $output_root/$project/$expname/checkpoint_final.pth --out $export_dir
```

`load_smpl=False` is needed for Street Gaussians configs, which have no human nodes. The `1cams` dataset configs hold out no test frames unless `test_image_stride` is set.

### Licence of this fork

Upstream DriveStudio is MIT-licensed. The Bosch 3DGEER patch included here is released under **AGPL-3.0**, so this repository contains AGPL-3.0 code; check those terms before redistributing it or building a service on it. The branch `phase3` holds only the MIT-compatible additions (the two tools and the `*_nocampose` configs).

## About
DriveStudio is a 3DGS codebase for urban scene reconstruction/simulation. It offers a system with multiple Gaussian representations to jointly reconstruct backgrounds, vehicles, and non-rigid categories (pedestrians, cyclists, etc.) from driving logs. DriveStudio also provides a unified data system supporting various popular driving datasets, including [Waymo](https://waymo.com/open/), [PandaSet](https://pandaset.org/), [Argoverse2](https://www.argoverse.org/av2.html), [KITTI](http://www.cvlibs.net/datasets/kitti/), [NuScenes](https://www.nuscenes.org/), [NuPlan](https://www.nuscenes.org/nuplan), and [PhysicalAI NCore](docs/datasets/physicalai.md).

This codebase also contains the **official implementation** of:
  > **OmniRe: Omni Urban Scene Reconstruction** <br> ICLR 2025 (Spotlight) <br> [Project Page](https://ziyc.github.io/omnire/) | [Paper](https://arxiv.org/abs/2408.16760) <br> [Ziyu Chen](https://ziyc.github.io/), [Jiawei Yang](https://jiawei-yang.github.io/), [Jiahui Huang](https://huangjh-pub.github.io/), [Riccardo de Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://www.jme.pub/), [Boris Ivanovic](https://www.borisivanovic.com/), [Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Marco Pavone](https://stanford.edu/~pavone/), [Li Song](https://medialab.sjtu.edu.cn/author/li-song/), [Yue Wang](https://yuewang.xyz/)

## 🎉 Try your own projects/research on DriveStudio!
### 🔥 Highlighted implementations

Our codebase supports two types of Gaussian trainers:

1. Single-Representation trainer (single Gaussian representation for the entire scene):
   - Deformable Gaussians
   - Periodic Vibration Gaussians

2. Multi-Representation trainer (Gaussian scene graphs trainer):
   - Background: Static Gaussians (Vanilla Gaussians)
   - Vehicles: Static Gaussians
   - Humans: SMPL-Gaussians, Deformable Gaussians
   - Other non-rigid categories: Deformable Gaussians

**Implemented methods:**

| Method Name | Implementation | Trainer Type | Gaussian Representations |
|-------------|----------------|--------------|--------------------------|
| [OmniRe](https://ziyc.github.io/omnire/) | Official | Multi | • Static Gaussians: Background, Vehicles<br>• SMPL Gaussians: Pedestrians (majority)<br>• Deformable Gaussians: Cyclists, far-range pedestrians, other non-rigid categories |
| [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians) | Unofficial | Single | • Deformable Gaussians: Entire scene |
| [PVG](https://github.com/fudan-zvg/PVG) | Unofficial | Single | • Periodic Vibration Gaussians: Entire scene |
| [Street Gaussians](https://github.com/zju3dv/street_gaussians) | Unofficial | Multi | • Static Gaussians: Background, Vehicles |
| [3DGEER](https://github.com/boschresearch/3dgeer) | Unofficial | Single/Multi | • Ray-Based Gaussians: Can be integrated with any other representation (see [here](docs/methods/3dgeer.md) for details) |

We extend our gratitude to the authors for their remarkable contributions. If you find these works useful, please consider citing them.

### 🚗 Dataset Support
This codebase provides support for popular driving datasets. We offer instructions and scripts on how to download and process these datasets:

| Dataset | Instruction | Cameras | Sync Frequency | Object Annotation |
|---------|-------------|---------|----------------|-------------------|
| Waymo | [Data Process Instruction](docs/datasets/waymo.md) | 5 cameras | 10Hz | ✅ |
| NuScenes | [Data Process Instruction](docs/datasets/nuscenes.md) | 6 cameras | 2Hz (up to 10Hz*) | ✅ |
| NuPlan | [Data Process Instruction](docs/datasets/nuplan.md) | 8 cameras | 10Hz | ✅ |
| ArgoVerse | [Data Process Instruction](docs/datasets/argoverse.md) | 7 cameras | 10Hz | ✅ |
| PandaSet | [Data Process Instruction](docs/datasets/pandaset.md) | 6 cameras | 10Hz | ✅ |
| KITTI | [Data Process Instruction](docs/datasets/kitti.md) | 2 cameras | 10Hz | ✅ |
| PhysicalAI NCore | [Data Process Instruction](docs/datasets/physicalai.md) | 7 cameras | ~10Hz (processed) | ✅ |

*NOTE: For NuScenes data, LiDAR operates at 20Hz and cameras at 12Hz, but keyframes (with object annotations) are only at 2Hz. We provide a method to interpolate annotations up to 10Hz.

### ✨ Functionality

<details>
<summary>Click to expand functionality details</summary>

We have implemented interesting and useful functionalities:

1. **Flexible multi-camera training:** Choose any combination of cameras for training - single, multiple, or all. You can set these up by **SIMPLY** configuring your selection in the config file.

2. **Powered by gsplat** Integrated [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization kernel with its advanced functions, e.g. absolute gradients, anti-aliasing, etc.

3. **Camera Pose Refinement:** Recognizing that camera poses may not always be sufficiently accurate, we provide a method to refine and optimize these poses.

4. **Objects' GT Bounding Box Refinement:** To address noise in ground truth boxes, we've added this feature to further improve accuracy and robustness.

5. **Affine Transformation:** This feature handles camera exposure and other related issues, enhancing the quality of scene reconstruction. 

6. ...

These functionalities are designed to enhance the overall performance and flexibility of our system, allowing for more accurate and adaptable scene reconstruction across various datasets and conditions.
</details>

## 📢 Updates

**[Nov 2026]**  (this fork) Added the 3DGUT / 3DGEER renderers via Bosch's patch, FastGS-style multi-view densification and pruning, no-CamPose configs, Hugging Face mask extraction and world-frame `.ply` export. See [About this fork](#about-this-fork-drivestudio_plus).

**[Jan 2025]**  OmniRe is accepted by ICLR 2025 as a spotlight paper!

**[Aug 2024]**  Release code of DriveStudio!

## 🔨 Installation

Run the following commands to set up the environment:

```shell
# Clone the repository with submodules
git clone --recursive https://github.com/Octoopass/drivestudio_plus.git
cd drivestudio_plus

# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt

# gsplat: this fork is developed and run with Bosch's gsplat-geer build (gsplat 1.5.3),
# which contains the standard 3DGS, 3DGUT and 3DGEER renderers.
git clone --recursive --branch gsplat-geer https://github.com/boschresearch/3dgeer.git third_party/3dgeer
git -C third_party/3dgeer checkout f15cee2 && git -C third_party/3dgeer submodule update --init --recursive
pip install ./third_party/3dgeer --no-build-isolation
# Stock gsplat (pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0) only supports
# render_mode: default and has not been tested with this fork.

pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..
```

Notes for this fork:

- `requirements.txt` now also lists `plyfile` (for `drive2sim/recon/tools/export_ply.py`) and `transformers` (for `drive2sim/recon/preprocess/masks/segformer_hf.py`).
- If the gsplat-geer build stops with `glm/gtc/type_ptr.hpp: No such file`, the `glm` submodule is empty: clone [glm](https://github.com/g-truc/glm) 1.0.1 into `third_party/3dgeer/gsplat/cuda/csrc/third_party/glm` and install again.
- Dataset-specific packages are still installed separately, as described in each dataset guide (for example `nuscenes-devkit` for nuScenes, the NCore SDK for PhysicalAI).

## 📊 Prepare Data
We support most popular public driving datasets. Detailed instructions for downloading and processing each dataset are available in the following documents:

- Waymo: [Data Process Instruction](docs/datasets/waymo.md)
- NuScenes: [Data Process Instruction](docs/datasets/nuscenes.md)
- NuPlan: [Data Process Instruction](docs/datasets/nuplan.md)
- ArgoVerse: [Data Process Instruction](docs/datasets/argoverse.md)
- PandaSet: [Data Process Instruction](docs/datasets/pandaset.md)
- KITTI: [Data Process Instruction](docs/datasets/kitti.md)
- PhysicalAI NCore: [Data Process Instruction](docs/datasets/physicalai.md)

## 🚀 Running
### Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python -m drive2sim.recon.tools.train \
    --config_file drive2sim/configs/methods/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

- To run other methods, change `--config_file`. See `drive2sim/configs/` for more options.
- Specify dataset and number of cameras by setting `dataset`. Examples: `waymo/1cams`, `waymo/5cams`, `pandaset/6cams`, `argoverse/7cams`, etc.
  You can set up arbitrary camera combinations for each dataset. See `drive2sim/configs/datasets/` for custom configuration details.
- For over 3 cameras or 450+ images, we recommend using `omnire_extended_cam.yaml`. It works better in practice.
- For PhysicalAI, use `dataset=physicalai/7cams`. Native FTheta training uses GEER or UT with `data.pixel_source.undistort: False` in the dataset config; set it to `True` for pinhole rectification. See the [3DGEER guide](docs/methods/3dgeer.md) for setup.
### Evaluation
```shell
python -m drive2sim.recon.tools.eval --resume_from $ckpt_path
```

<details>
<summary>Compare with OmniRe</summary>
To compare with OmniRe, run the following commands to reproduce the paper’s checkpoints.

```
# Scene Reconstruction
for scene_idx in {0..7}; do
    IFS=',' read scene_idx seg_name start_timestep end_timestep <<< $(awk -v idx="$(($scene_idx + 2))" \
        'NR==idx {print $1, $2, $3, $4}' drive2sim/assets/scene_lists/waymo_example_scenes.txt)
    
    python -m drive2sim.recon.tools.train \
        --config_file drive2sim/configs/methods/paper_legacy/omnire.yaml \
        --output_root logs/omnire_waymo/ \
        --project recon \
        --run_name ${scene_idx} \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.test_image_stride=0
done
```

```
# Novel View Synthesis
for scene_idx in {0..7}; do
    IFS=',' read scene_idx seg_name start_timestep end_timestep <<< $(awk -v idx="$(($scene_idx + 2))" \
        'NR==idx {print $1, $2, $3, $4}' drive2sim/assets/scene_lists/waymo_example_scenes.txt)
    
    python -m drive2sim.recon.tools.train \
        --config_file drive2sim/configs/methods/paper_legacy/omnire.yaml \
        --output_root logs/omnire_waymo/ \
        --project nvs \
        --run_name ${scene_idx} \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.test_image_stride=10
done
```

and run `drive2sim/recon/tools/gather_results.py` to get the results.
</details>

## 👏 Contributions
We're improving our project to develop a robust driving recom/sim system. Some areas we're focusing on:

- A real-time viewer for background and foreground visualization
- Scene editing and simulation tools
- Other Gaussian representations (e.g., 2DGS, surfels)

We welcome pull requests and collaborations. If you'd like to contribute or have questions, feel free to open an issue or contact [Ziyu Chen](https://github.com/ziyc) (ziyu.sjtu@gmail.com).

## 🙏 Acknowledgments
We utilize the rasterization kernel from [gsplat](https://github.com/nerfstudio-project/gsplat). Parts of our implementation are based on work from [EmerNeRF](https://github.com/NVlabs/EmerNeRF), [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), [GART](https://github.com/JiahuiLei/GART), and [Neuralsim](https://github.com/PJLab-ADG/neuralsim). We've also implemented unofficial versions of [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians), [PVG](https://github.com/fudan-zvg/PVG), and [Street Gaussians](https://github.com/zju3dv/street_gaussians), with reference to their original codebases.

We extend our deepest gratitude to the authors for their contributions to the community, which have greatly supported our research.

## Citation
If you find this codebase helpful, please kindly cite:
```
@inproceedings{
  chen2025omnire,
  title={OmniRe: Omni Urban Scene Reconstruction},
  author={Ziyu Chen and Jiawei Yang and Jiahui Huang and Riccardo de Lutio and Janick Martinez Esturo and Boris Ivanovic and Or Litany and Zan Gojcic and Sanja Fidler and Marco Pavone and Li Song and Yue Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
