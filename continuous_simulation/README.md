# EgoSim Incremental Inference

Incremental egocentric world-model inference for multi-clip hand-object interaction video generation.

This repo is the continuous generation subfunction of EgoSim. It uses the same single-clip EgoSim inference code path, metadata format, and **EgoSim-14B** model directory as the main project, then reconstructs and carries an updatable 3D scene state between clips.

## What this repo contains

- **Incremental inference** - generate a sequence of clips from one source video while updating an updatable 3D scene state after each generated clip.
- **Per-clip EgoSim generation** - calls the same standard inference stack used by `egowm/inference/runner.py`.
- **Updatable 3D scene state backend** - runs the bundled modified `egosim_state` backend to reconstruct, align, fuse, and render scene state for the next clip.

## Installation

Requires the main EgoSim inference environment plus a separate scene environment for reconstruction.

```bash
cd continuous_simulation
```

Install the main EgoSim project first by following its `Installation` section. This provides the Python environment used for per-clip EgoSim generation.

**SAM3** and **Depth-Anything-3** are third-party dependencies and are **not** shipped in this repository. Clone them into the paths below before running the scene setup script (from `continuous_simulation/`):

```bash
git clone https://github.com/facebookresearch/sam3.git \
  scene_backend/egosim_state/sam3

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git \
  scene_backend/egosim_state/Depth-Anything-3
```

Then create the scene reconstruction environment for incremental inference:

```bash
cp configs/project.yaml.example configs/project.yaml
bash scripts/setup_envs.sh
```

The setup script builds wheels for SAM3 and Depth-Anything-3, installs them into `egosim-scene`, and verifies imports. It creates `egosim-scene`, which runs EgoSim-state reconstruction, phrase prediction, and updatable 3D scene state rendering.

Expected layout after cloning:

```text
continuous_simulation/scene_backend/egosim_state/
├── sam3/                  # git clone facebookresearch/sam3
├── Depth-Anything-3/      # git clone ByteDance-Seed/Depth-Anything-3
└── ...                    # bundled egosim_state backend (in repo)
```

This subproject lives inside the main **EgoSim** repository. The launcher imports standard EgoSim inference code from the repo root (`../` relative to `continuous_simulation/`).

## Model weights

Download weights from the **EgoSim** repo root:

```bash
cd ..   # EgoSim repository root (parent of continuous_simulation/)

# Scene reconstruction weights (required for recon_visualize / full)
bash continuous_simulation/scripts/download_scene_weights.sh

# EgoSim-14B (required for generation; same as single-clip inference in ../README.md)
huggingface-cli download wuzhi-hao/EgoSim --local-dir ./EgoSim-14B
```

Expected `EgoSim-14B/` layout (at `EgoSim/EgoSim-14B/`):

```text
EgoSim-14B/
├── diffusion_pytorch_model.safetensors
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
└── google/umt5-xxl/
```

Copy `configs/project.yaml.example` to `configs/project.yaml`; it points at this directory via `models.model_root: ../EgoSim-14B` (relative to `continuous_simulation/`).

Phrase prediction with **Qwen3-VL-4B-Instruct** is enabled by default. After copying `configs/project.yaml.example` to `configs/project.yaml`, set `models.qwen_vl_root` if needed (default: `../artifacts/models/qwen-vl/Qwen3-VL-4B-Instruct`) and download the weights when ready:

```bash
bash continuous_simulation/scripts/download_scene_weights.sh --with-qwen
```

Optional flags: `--with-alternate`, `--with-aot`, `--all`. Run `bash continuous_simulation/scripts/download_scene_weights.sh --help` for details.

Expected layout:

```text
EgoSim/                          # repository root
├── continuous_simulation/
├── EgoSim-14B/
└── artifacts/models/
    ├── priorda/
    ├── sam3/
    ├── depth-anything-3/
    ├── groundingdino/
    ├── bert-base-uncased/
    ├── aot/
    ├── qwen-vl/                 # Qwen3-VL phrase prediction (enabled by default)
    ├── unidepth-v2-vitl14/      # alternate scene pipelines only
    ├── video-depth-anything/    # alternate scene pipelines only
    └── moge-2-vitl-normal/      # alternate scene pipelines only
```

After `cp configs/project.yaml.example configs/project.yaml`, paths already point at the defaults above. See the main [`README.md`](../README.md#model-weights) for **EgoSim-14B** download details.

## Mini sample (quicktest data)

Git ships only `metadata.csv`. Download the 2-clip test assets ([Google Drive](https://drive.google.com/drive/folders/1a0ssi752vgqiPCovNknLoqbdBqcz1emY?usp=drive_link)), extract the `14/` folder into `tests/samples/mini_sample/continuous_generation/`, then run:

```bash
cd tests/samples/mini_sample/continuous_generation
mkdir -p process_result/test_3dmem/add_remove_lid/14
cd process_result/test_3dmem/add_remove_lid/14
ln -sf ../../../../14/0_60 0_60
ln -sf ../../../../14/60_120 60_120
ln -sf ../../../../14/14_0_120.hdf5 14_0_120.hdf5
```

Paths above are relative to the **EgoSim** repository root. Copy `configs/project.yaml.example` to `configs/project.yaml` before running quicktest.

## Data preparation

The incremental metadata extends the main EgoSim CSV format.

Required columns are the same as single-clip inference:

```text
video,ego_prior_video,hand_keypoint_video,first_frame,prompt
```

Incremental reconstruction uses additional columns when available:

```text
task_name,part_idx,process_result_dir,hdf5_path,gt_process_result_dir
```

`video` should encode clip ranges in its filename, for example:

```text
test_3dmem/add_remove_lid/GT_14_0_60.mp4
test_3dmem/add_remove_lid/GT_14_60_120.mp4
```

Clips with the same task and source video id are grouped and processed in temporal order. See [`../tests/samples/mini_sample/continuous_generation/metadata.csv`](../tests/samples/mini_sample/continuous_generation/metadata.csv) for the bundled example (after downloading the mini sample above).

The default config sets:

```text
data.dataset_root: ../../tests/samples/mini_sample/continuous_generation
data.metadata_path: ../../tests/samples/mini_sample/continuous_generation/metadata.csv
```

Paths in the CSV may include a `process_result/` prefix; the loader resolves them against `dataset_root` automatically.

## Inference

All commands below assume you are inside this repo:

```bash
cd continuous_simulation
```

This repository ships `configs/project.yaml.example`. Copy it to `configs/project.yaml` before running (paths are relative to `continuous_simulation/`):

```bash
cp configs/project.yaml.example configs/project.yaml
```

With the standard layout, no further config edit is needed.

Run a reconstruction visualization smoke test:

```bash
PYTHONPATH=src python -m egowm_incremental.cli run-incremental \
  --config configs/project.yaml \
  --mode recon_visualize
```

Run full incremental generation:

```bash
PYTHONPATH=src python -m egowm_incremental.cli run-incremental \
  --config configs/project.yaml \
  --mode full
```

Available modes:

- `recon_visualize` - generate the first clip, run scene reconstruction, and optionally open/write visualization outputs.
- `full` - run the multi-clip incremental pipeline with updatable 3D scene state updates.

## Acknowledgements

The updatable scene-state backend is adapted from NVIDIA's [ViPE](https://github.com/nv-tlabs/vipe) (Video Pose Engine). It also relies on:

- [SAM 3](https://github.com/facebookresearch/sam3) — open-vocabulary segmentation
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — multi-view depth and pose
- [DROID-SLAM](https://github.com/cvg/DROID-SLAM) — visual SLAM
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — text-guided detection
- [Prior Depth Anything](https://github.com/SpatialVision/Prior-Depth-Anything) — depth prior fusion
- [GeoCalib](https://github.com/cvg/GeoCalib) — camera intrinsics estimation
- [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything) — instance tracking (DeAOT)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — phrase prediction for scene segmentation (optional)

## License

Apache 2.0
