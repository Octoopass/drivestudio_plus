<a id="top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/media/logo.png">
    <img alt="Logo" src="docs/media/logo.png" width="560">
  </picture>
</p>

# drive2sim

[Tiếng Việt: README_VN.md](README_VN.md)

drive2sim reconstructs a recorded road segment as a 3D scene from camera images, LiDAR measurements and the positions and orientations of the sensors. You can inspect the scene from another viewpoint, play vehicle and human motion over time, and compare reconstruction quality and training time across configurations.

The scene uses **3D Gaussian Splatting**: many small 3D elements, each with a position, size, color and transparency. A renderer projects and combines these elements to produce an image from a camera viewpoint. **Training** adjusts the scene parameters so that these images match the recorded images.

Built on [DriveStudio](https://github.com/ziyc/drivestudio), drive2sim adds the `d2s` command line and **Studio**, a web interface served by your own machine. It connects data preparation, training, evaluation and result storage so that comparing configurations does not require assembling separate scripts.

## Contents

- [What a run contains](#what-a-run-contains)
- [Choosing a method and renderer](#choosing-a-method-and-renderer)
- [Using pinhole and fisheye cameras together](#using-pinhole-and-fisheye-cameras-together)
- [Installation and first use](#installation-and-first-use)
- [Existing results and validation status](#existing-results-and-validation-status)
- [Moving results to another machine](#moving-results-to-another-machine)
- [Documentation and source layout](#documentation-and-source-layout)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## What a run contains

A **dataset** is a collection of recordings, such as nuScenes or KITTI-360. A **scene** is the recording segment you ingest from that collection, including images, LiDAR points, camera calibration and any available annotations of vehicles or people.

A **split** selects the cameras and frame range for an experiment, and separates images used for training from images held out for testing. For example, you can use the three front cameras and hold out every tenth frame. Test images are not used to update the model.

A **run** is one experiment with a particular scene, split and configuration. It normally prepares the inputs the method needs, trains the scene model, evaluates rendered images against the held-out images, and saves checkpoints, images, videos and 3D exports. A **checkpoint** stores learned model parameters so the model can be loaded without training it from scratch.

You choose a **working directory**, or *workdir*, to hold the data. Its `data/` folder contains ingested scenes, `runs/` contains experiment results, and `cache/` stores reusable preparation results. For example, two runs can reuse masks when the scene and mask-generation settings match. A mask marks image regions that need separate treatment, such as the sky or moving objects.

## Choosing a method and renderer

The **reconstruction method** decides how the scene is represented and divided into components. The **renderer** produces an image of that scene using a camera's parameters. Rendering is needed both during training and when viewing the model, so choosing OmniRe and choosing 3DGUT are decisions at different parts of the process. 3D Gaussian Splatting names the scene representation; `renderer=3dgs` names the baseline renderer. Selecting 3DGUT or 3DGEER still uses a scene represented by 3D Gaussians.

| Choice | Role in this tool |
|---|---|
| **OmniRe** | Models the background, vehicles, people and deformable objects as separate components of one scene. People use SMPL, an articulated body model for representing human poses. |
| **Street Gaussians** (`streetgs`) | Models the background and vehicles, without OmniRe's SMPL human component. |
| **3DGS** (`3dgs`) | The renderer for pinhole cameras, which use the usual perspective projection model. |
| **3DGUT** (`3dgut`) and **3DGEER** (`3dgeer`) | Renderers that support pinhole and the wide-angle camera models integrated in the tool. |

Optional settings change training speed or the way Gaussians are added and removed. **FastGS** changes the addition and removal rules. **Faster-GS** optimizes some training computations; the optimizations used depend on the renderer. MCMC densification was tried and removed from the options because it scored lower than the default rule ([docs/methods/densify_mcmc.md](docs/methods/densify_mcmc.md)). `d2s approach show omnire` lists the settings, and `d2s approach check` checks a configuration before creating a run. See [docs/methods/](docs/methods/) for the mechanisms and limits.

## Using pinhole and fisheye cameras together

For KITTI-360, **one run can select both pinhole cameras and both fisheye cameras**. Images from all four cameras train **one shared scene model**. During training, the tool takes an image from the training set, renders the scene using that image's camera, and updates the scene parameters from the image error. Each camera retains its own lens calibration and pose.

When rendering a fisheye view, the predicted image has fisheye distortion so it can be compared with the image captured through that lens. Rendering the same 3D scene with a pinhole camera produces a perspective image. For example, a straight lamp post in space can appear curved near a fisheye image edge, while a pinhole projection preserves the straight line. Lens distortion belongs to the projection from the scene to an image; the aim is not to bend the scene geometry to fit a fisheye image.

Opening the result shows a viewport through a virtual camera. Having two input camera types does not create two screens or two scenes. The browser preview currently uses a pinhole camera; other choices depend on the viewer and renderer. The camera used for viewing need not have the same projection as the cameras that supplied the training images.

Fisheye images retain their original projection and distortion, although they may be downsampled according to the configuration. KITTI-360 describes its fisheye lens calibration using the Mei model; ingestion fits those parameters to the **OpenCV fisheye** representation supported by the renderers. This converts the **camera calibration**, rather than straightening the images into pinhole images. The renderer uses the appropriate camera model when projecting the scene. The conversion keeps the full field of view: the fitted model stays within 0.5 px of the Mei model up to the left and right image edges (about 96° from the optical axis, about 191° across). Training masks the dark region outside the image circle (beyond 96° from the axis, declared by the converter) and the recording vehicle's body; with 3DGUT it also masks rays beyond 88°, because 3DGUT cannot draw rays at 90° or more. A 3DGEER reconstruction therefore covers about 191°, a 3DGUT one 176° ([docs/datasets/kitti360_fisheye_workflow.md](docs/datasets/kitti360_fisheye_workflow.md)).

The split determines which images enter a run:

| KITTI-360 cameras selected in the split | Training images in that run |
|---|---|
| `image_00`, `image_01` | Only the two pinhole cameras. |
| `image_02`, `image_03` | Only the two fisheye cameras. |
| All four cameras | Pinhole and fisheye images update the same scene model. |

A run containing fisheye images must select `renderer=3dgut` or `renderer=3dgeer`; `3dgs` currently accepts only pinhole cameras. To compare these three camera selections, create three splits and train three separate runs, each with its own scene model.

**Validation status:** KITTI-360 ingestion, camera calibration and projection have been checked on real data. The three OmniRe experiments using pinhole only, fisheye only and both together have not been completed. Joint training is implemented, but its quality benefit has not yet been established by those experiments. See the [KITTI-360 guide](docs/datasets/kitti360.md) and [experiment plan](docs/experiments/run_plan.md).

## Installation and first use

Clone the source first. Run the commands below from the resulting `drivestudio_plus` folder:

```bash
git clone --recursive --branch drive2sim https://github.com/Octoopass/drivestudio_plus.git
cd drivestudio_plus
```

Then choose whether to open existing results or reconstruct a new scene. The example uses `../drive2sim-workdir`, a data folder beside the source repository; you can choose another path.

### Open an existing results package

If you have a `.tar` created by `d2s pack`, the light environment can show metrics, images, videos and the browser 3D preview. Installing it from source needs Python 3.9 or newer, [uv](https://docs.astral.sh/uv/) and Node.js:

```bash
make setup
mkdir ../drive2sim-workdir
tar -xf /path/to/results.tar -C ../drive2sim-workdir
make studio WORKDIR=../drive2sim-workdir
```

Replace `/path/to/results.tar` with the archive you received and extract into an empty directory. Open the address printed by Studio. This viewing path does not require an NVIDIA GPU. Rendering with the model's renderer, re-exporting or training locally needs the GPU environment below.

### Reconstruct a scene from raw data

The local reconstruction path uses Linux, an NVIDIA GPU and a CUDA toolkit with `nvcc`. Install [Conda](https://docs.conda.io/) to create the project's Python environment:

```bash
conda env create -f envs/main.yml
conda activate drive2sim
bash envs/install_main.sh
```

The script installs reconstruction dependencies and builds the GPU components. If a step fails, resolve its cause and continue with `bash envs/install_main.sh --from <step>`; the step names are listed in the script. The light `make setup` environment does not install these dependencies.

This example uses **nuScenes v1.0-mini**. Download it from [nuscenes.org](https://www.nuscenes.org/download) after registering, and extract the data so the source folder contains `v1.0-mini/`, `samples/` and `sweeps/`. The sweeps contain images and LiDAR measurements between annotated keyframes; the default ingestion uses them to build a denser time sequence.

Create the workdir, inspect the raw source and ingest just `scene-0061`:

```bash
mkdir -p ../drive2sim-workdir
d2s source inspect /path/to/nuscenes
d2s scene ingest /path/to/nuscenes --version v1.0-mini --sequence scene-0061 \
  --workdir ../drive2sim-workdir
```

Replace `/path/to/nuscenes` with your extracted raw data folder. Ingestion creates a scene named `nuscenes_scene-0061`. Next create a split named `front3`, selecting the three front cameras and holding out every tenth frame:

```bash
d2s split create nuscenes_scene-0061 --split front3 \
  --cameras CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT --strategy every_nth_frame:10 \
  --workdir ../drive2sim-workdir
```

OmniRe needs the body-model file `SMPL_NEUTRAL.pkl` to reconstruct people. If your workdir does not contain `smpl_models/SMPL_NEUTRAL.pkl`, obtain the file from [SMPL](https://smpl.is.tue.mpg.de) or the supplied results package and place it there. The file is not in the Git source repository; handoff archives made by `d2s pack` include it when a selected run needs SMPL.

Start an OmniRe run with the default configuration:

```bash
d2s run start omnire --scene nuscenes_scene-0061 --split front3 \
  --workdir ../drive2sim-workdir
```

The command prints a `run_id` identifying this experiment. The first run may download weights and compute preparation results. Later runs reuse preparation only when the relevant inputs and settings match. To inspect the results or open Studio in the GPU environment you just installed:

```bash
d2s report --workdir ../drive2sim-workdir
make studio CONDA_ENV=drive2sim WORKDIR=../drive2sim-workdir
```

Studio also provides scene ingestion, split creation and run creation. To create a **new run on Kaggle**, add `--executor kaggle` to `run start`; Colab uses `--executor colab`. Both require their own login and data setup first. The [Kaggle](docs/drive2sim/CONTRACTS.md#92-gói-job-cho-kaggle) and [Colab](docs/drive2sim/CONTRACTS.md#93-chạy-trên-colab) instructions explain those steps, including how to supply SMPL for runs with people.

## Existing results and validation status

These runs all use nuScenes `scene-0061`, the three front cameras, 30,000 training steps, 800×450 images and one Tesla T4. The extra settings are part of each experiment, so this is not a controlled comparison changing only the renderer.

| Run | Method | Renderer | Extra settings | Test PSNR | Training time | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|
| `f8a6` | OmniRe | 3DGS | FastGS, Faster-GS | 25.01 | 2 h 29 min | 4727 |
| `ccec` | Street Gaussians | 3DGS | FastGS | 25.02 | 1 h 11 min | 3381 |
| `60e5` | OmniRe | 3DGUT | FastGS, Faster-GS | 23.99 | 2 h 26 min | 3463 |
| `a8a9` | OmniRe | 3DGEER | FastGS, Faster-GS, refinement every 50 steps | 21.77 | 2 h 43 min | 4257 |

PSNR measures how closely a rendered image matches its held-out reference; higher is better when comparing the same image set. VRAM is GPU memory, reported here in MiB (1024 MiB = 1 GiB). Other scenes may require more memory depending on image count, resolution and settings. Each row is one run. The complete results and comparison limits are in [run_plan.md](docs/experiments/run_plan.md).

nuScenes has completed the full path from raw data to results. KITTI-360 ingestion and camera geometry have been checked, while the OmniRe comparisons above remain pending. Waymo, Argoverse 2, PandaSet, nuPlan, KITTI and PhysicalAI have ingestion/loading paths based on DriveStudio, but this project has not validated them with end-to-end training on real data. Kaggle has completed real runs; the Colab integration has not yet been validated with a real GPU run.

The viewing outputs serve different purposes. The GPU viewer uses the checkpoint and the model's renderer. The browser preview uses exported data and can run without a CUDA environment. The `.ply` export is a static scene containing the background and, optionally, vehicles at one frame; it does not contain the full motion and SMPL people available in the other viewing paths.

## Moving results to another machine

`d2s pack` collects selected runs, their ingested scenes, referenced preparation results and every checkpoint. When a run needs SMPL, it also copies the body model to `smpl_models/SMPL_NEUTRAL.pkl` inside the archive. The recipient gets the data needed to reopen the runs; they still need the software environment described above.

```bash
d2s pack RUN_ID_1 RUN_ID_2 --output results.tar \
  --workdir ../drive2sim-workdir --dry-run
d2s pack RUN_ID_1 RUN_ID_2 --output results.tar \
  --workdir ../drive2sim-workdir
```

Replace `RUN_ID_1` and `RUN_ID_2` with the full IDs shown in Studio or by `d2s run list`. The first command only lists dependencies and estimates size; the second writes the archive. It includes a bilingual README and `pack.json`, which lists files and their SHA-256 hashes for checking transferred data. Login tokens and raw datasets are not included. Raw data is needed only to re-ingest or add other scenes.

## Documentation and source layout

| Document | Use it to |
|---|---|
| [SPEC](docs/drive2sim/SPEC.md) | Understand requirements and the user workflow. |
| [ARCHITECTURE](docs/drive2sim/ARCHITECTURE.md) | Understand component boundaries and design decisions. |
| [CONTRACTS](docs/drive2sim/CONTRACTS.md) | Look up CLI commands, APIs and file formats. |
| [Dataset guides](docs/datasets/overview.md) | Download, arrange and ingest each dataset. |
| [Method notes](docs/methods/) | Read mechanisms, limits and differences from the papers. |
| [DriveStudio script guide](docs/drivestudio.md) | Call reconstruction scripts directly, without the tool's CLI. |

The tool lives in `drive2sim/`: `cli/` handles commands, `engine/` manages runs, `ingest/` imports data, `stages/` implements individual steps, and `studio/` serves the interface. `drive2sim/recon/` contains the DriveStudio reconstruction code. `envs/` holds installation scripts, `third_party/` holds external code, and `tests/` contains automated checks.

## Contributing

Read [AGENTS.md](AGENTS.md) first. It holds the coding rules, the test commands and the rule that docs change together with the code. Run `make check` (format, lint, tests) before each commit. Report problems in the repository's Issues.

## Acknowledgement

The reconstruction code comes from [DriveStudio](https://github.com/ziyc/drivestudio) and its OmniRe paper. The 3DGUT and 3DGEER renderers come from Bosch Research's [3DGEER patch](https://github.com/boschresearch/3dgeer) of [gsplat](https://github.com/nerfstudio-project/gsplat). The speed-ups follow [FastGS](https://github.com/fastgs/FastGS), [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) and [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc); every difference from the original is listed in [docs/methods/](docs/methods/). If you use this code, please cite OmniRe:

```bibtex
@inproceedings{chen2025omnire,
  title={OmniRe: Omni Urban Scene Reconstruction},
  author={Ziyu Chen and Jiawei Yang and Jiahui Huang and Riccardo de Lutio and Janick Martinez Esturo and Boris Ivanovic and Or Litany and Zan Gojcic and Sanja Fidler and Marco Pavone and Li Song and Yue Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## License

DriveStudio is under the MIT licence ([LICENSE](LICENSE)). The Bosch 3DGEER patch included here is under AGPL-3.0, so the repository as a whole has to follow the AGPL-3.0 terms. Check them before you redistribute it or offer it as a service. Datasets and the SMPL model keep their own licences; KITTI-360, for example, is for non-commercial use only.

[Back to Top](#top)
