<!-- Copyright (c) jiamingda (https://github.com/Luyitas) -->

# EgoVid Dataset Annotation Pipeline

Organized in the order: `00_first_frame_mask_inpaint` → `01_depth_pose_da3` → `02_mano_predict`.

---

## 1. System Requirements

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA GPU ≥ 24 GB VRAM (tested: RTX 4090 49 GB) |
| Driver | NVIDIA Driver ≥ 570.x (CUDA 12.8 runtime) |
| OS | Ubuntu 22.04 (glibc ≥ 2.35) |
| GCC | ≥ 11.x (required for mmcv source build) |
| Conda | Miniconda / Anaconda 3 |
| ffmpeg | Available in system PATH |

> Inpainting (Step 00b) uses `cpu_offload`, peaking at ~30 GB VRAM + significant CPU memory.
> HaMeR MANO prediction (Step 02a) peaks at ~21 GB VRAM.

---

## 2. Environments Overview

3 separate conda environments are required because PyTorch / CUDA / mmcv versions conflict and cannot be merged. A 4th lightweight environment is used for captioning.

| Env Name | Python | PyTorch | CUDA | Steps | Working Directory |
|----------|--------|---------|------|-------|-------------------|
| `sam3` | 3.12 | 2.7.0 | cu126 | 1.2 SAM3 segmentation | this directory |
| `da3` | 3.10 | 2.10.0 | cu128 | 1.3 Inpaint, 2.1–2.5 DA3/rendering | DA3 source directory |
| `hamer` | 3.10 | 2.9.1 | cu128 | 3.1–3.4 MANO full pipeline | hamer source directory |
| `caption` | 3.10 | 2.x | cu12x | 4.1 Qwen2.5-VL captioning | this directory |

---

## 3. Clone Source Repos

```bash
REPOS_DIR="<your_repos_dir>"   # e.g. ~/repos
mkdir -p "${REPOS_DIR}" && cd "${REPOS_DIR}"

# Depth-Anything-3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git

# SAM3 (Segment Anything Model 3)
git clone https://github.com/facebookresearch/sam3.git

# HaMeR (with ViTPose submodule)
git clone --recursive https://github.com/geopavlakos/hamer.git
```

---

## 4. Environment Setup

### 4.1 Environment: `sam3`

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

cd "${REPOS_DIR}/sam3"
pip install -e .
pip install opencv-python tqdm scipy
```

### 4.2 Environment: `da3`

```bash
conda create -n da3 python=3.10 -y
conda activate da3

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install xformers

cd "${REPOS_DIR}/Depth-Anything-3"
pip install -e .
pip install 'moviepy<2'           # DA3's gs.py depends on the old API, must be <2

pip install diffusers transformers accelerate safetensors   # Qwen Inpaint
pip install pyrender trimesh                                # rendering
pip install opencv-python numpy scipy pillow tqdm
```

### 4.3 Environment: `hamer`

The most complex environment: mmcv 1.3.9 must be compiled from source; detectron2 and chumpy must be installed from git.

```bash
conda create -n hamer python=3.10 -y
conda activate hamer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# mmcv 1.3.9 source build (hard dependency for HaMeR + ViTPose, takes 10–30 min)
git clone https://github.com/open-mmlab/mmcv.git /tmp/mmcv_build
cd /tmp/mmcv_build && git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd -

# detectron2 / chumpy (from source)
pip install 'git+https://github.com/facebookresearch/detectron2'
pip install 'git+https://github.com/mattloper/chumpy'

# HaMeR
cd "${REPOS_DIR}/hamer"
pip install -e ".[all]"

# ViTPose (editable install as mmpose)
pip install -v -e third-party/ViTPose

pip install timm einops smplx==0.1.28 pyrender yacs iopath
pip install opencv-python numpy scipy pillow tqdm
```

> mmcv 1.3.9 compilation requires matching PyTorch + CUDA + GCC versions. Ensure `nvcc` is in PATH and its version matches the PyTorch CUDA version before building.

### 4.4 Environment: `caption`

```bash
conda create -n caption python=3.10 -y
conda activate caption

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
pip install qwen-vl-utils
```

---

## 5. Model Checkpoints

| Model | Download | Notes |
|-------|----------|-------|
| SAM3 | `huggingface-cli download facebook/sam3 sam3.pt --local-dir ${REPOS_DIR}/sam3/checkpoints` | ~3.3 GB |
| DA3 | `huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ${REPOS_DIR}/Depth-Anything-3/checkpoints/DA3NESTED-GIANT-LARGE-1.1` | |
| Qwen Inpaint | `huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir ${REPOS_DIR}/models/Qwen-Image-Edit-2511` | ~30 GB |
| HaMeR | `cd ${REPOS_DIR}/hamer && bash fetch_demo_data.sh` | auto-downloads weights |
| ViTDet | `wget -P ${REPOS_DIR}/hamer/_DATA/ https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl` | or auto-downloaded on first run |
| ViTPose | Download `wholebody.pth` from [ViTPose releases](https://github.com/ViTAE-Transformer/ViTPose) and place it in `${REPOS_DIR}/hamer/_DATA/vitpose_ckpts/vitpose+_huge/` | |
| MANO | Register at https://mano.is.tue.mpg.de/, download `MANO_RIGHT.pkl`, and place it in `${REPOS_DIR}/hamer/_DATA/data/mano/` | license required |
| Qwen2.5-VL | `huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct --local-dir ${REPOS_DIR}/models/Qwen2.5-VL-32B-Instruct` | ~65 GB; smaller variants (7B/72B) also work |

---

## 6. Pipeline Configuration

Edit `data_process/run_pipeline.env.sh` to set your paths, then source it before running any step:

```bash
source data_process/run_pipeline.env.sh
```

| Variable | Description |
|----------|-------------|
| `VIDEO_PATH` | Path to your source video (any filename) |
| `DA3_ROOT` | Depth-Anything-3 repo root |
| `DA3_MODEL` | DA3 pretrained checkpoint directory |
| `HAMER_ROOT` | HaMeR repo root |
| `MANO_PATH` | MANO model directory (`hamer/_DATA/data/mano`) |
| `SAM3_ROOT` | SAM3 repo root |
| `SAM3_CHECKPOINT` | `sam3.pt` checkpoint path |
| `INPAINT_MODEL` | Qwen-Image-Edit-2511 local snapshot directory |
| `VITDET_INIT_CHECKPOINT` | ViTDet `model_final_f05665.pkl` path |

---

## 7. Run Pipeline — Step by Step

```bash
cd EgoSim
```

Edit `data_process/run_pipeline.env.sh` once to set your paths, then source it:

```bash
# open and fill in VIDEO_PATH, DA3_ROOT, HAMER_ROOT, SAM3_ROOT, model paths
nano data_process/run_pipeline.env.sh

source data_process/run_pipeline.env.sh
```

The clip name is derived automatically from the video filename (e.g. `my_clip.mp4` → `my_clip`). Final outputs are written to `tests/samples/<clip_name>/`; intermediate files go under `tests/samples/<clip_name>/_proc/`.

The three inference inputs produced by this pipeline are:

| File | Description |
|------|-------------|
| `hand_inpaint.png` | First frame with hands inpainted (clean background) |
| `rendered_scene.mp4` + `pc_mask_video.mp4` | Ego prior: colored point cloud video + binary mask video |
| `skeleton_3d.mp4` | Hand skeleton keypoint video |

---

### Goal 1: Inpainted First Frame → `hand_inpaint.png`

#### Step 1.1 — Convert video to 16fps 720p 61-frame

```bash
bash data_process/run_step00_convert.sh
```

**Output:** `tests/samples/<clip_name>/video_16fps.mp4`

---

#### Step 1.2 — SAM3 hand segmentation on first frame

```bash
conda activate sam3
bash data_process/00_first_frame_mask_inpaint/run_step00a_sam_seg.sh
```

**Output:** `_proc/sam_results/hand_seg.png`, `hand_seg_vis.jpg`

---

#### Step 1.3 — Inpaint hands out of first frame

```bash
conda activate da3
bash data_process/00_first_frame_mask_inpaint/run_step00b_inpaint.sh
```

**Output:** `_proc/inpainted/<clip_name>/hand_inpaint.png`

---

### Goal 2: Point Cloud Videos → `rendered_scene.mp4` + `pc_mask_video.mp4`

Steps 2.1–2.5 annotate **one clip at a time** — the clip configured in `VIDEO_PATH`.
Requires Step 1.3 (`video_16fps.mp4` + `hand_inpaint.png`) and `conda activate da3`.

Scripts live under `data_process/01_depth_pose_da3/`:

| Step | Script | Output |
|------|--------|--------|
| 2.1 | `run_step01a_da3_predict.sh` | `_proc/poses_da3/<clip_name>/` |
| 2.2 | `run_step01b_smooth.sh` | `_proc/poses_da3_smoothed/<clip_name>/` |
| 2.3 | `run_step01c_depth_inpainted.sh` | `_proc/inpainted/<clip_name>/depth_first_frame.npy` |
| 2.4 | `run_step01d_render.sh` | `rendered_scene.mp4`, `overlay.mp4` |
| 2.5 | `run_step01d_render_mask.sh` | `pc_mask_video.mp4` |

#### Step 2.1 — DA3 depth + camera parameter prediction

```bash
conda activate da3
bash data_process/01_depth_pose_da3/run_step01a_da3_predict.sh
```

**Output:** `_proc/poses_da3/<clip_name>/` — `depth_000000.npy`, `intrinsics_*.npy`, `extrinsics_*.npy`

---

#### Step 2.2 — Kalman smoothing of camera parameters

```bash
conda activate da3
bash data_process/01_depth_pose_da3/run_step01b_smooth.sh
```

**Output:** `_proc/poses_da3_smoothed/<clip_name>/extrinsics_*.npy`

---

#### Step 2.3 — DA3 depth prediction on inpainted first frame

```bash
conda activate da3
bash data_process/01_depth_pose_da3/run_step01c_depth_inpainted.sh
```

**Output:** `_proc/inpainted/<clip_name>/depth_first_frame.npy`, `intrinsics_first_frame.npy`, `cloud_first_frame.npy`

---

#### Step 2.4 — Render colored point cloud video

```bash
conda activate da3
bash data_process/01_depth_pose_da3/run_step01d_render.sh
```

**Output:** `rendered_scene.mp4` (inference input), `overlay.mp4` (visualization)

---

#### Step 2.5 — Render point cloud mask video

```bash
conda activate da3
bash data_process/01_depth_pose_da3/run_step01d_render_mask.sh
```

**Output:** `pc_mask_video.mp4` (black points on white background)

Or run Steps 2.4 and 2.5 together:

```bash
bash data_process/01_depth_pose_da3/run_render_16fps_example.sh
```

**Python entry points** (called by the scripts above):

| File | Role |
|------|------|
| `pred_multi_gpu_2.py` | DA3 depth + camera on `video_16fps.mp4` |
| `smooth_camera_kalman_egovid.py` | Kalman smoothing (`--clip_name`) |
| `process_depth_inpainted.py` | First-frame depth on `hand_inpaint.png` (`--clip_dir`) |
| `render_16fps_aligned.py` | Point-cloud render for one clip |
| `camera_trajectory_render_pyrender.py` | PyRender backend used by `render_16fps_aligned.py` |

> `split_data.py` is a legacy batch helper and is not used by the single-clip pipeline.

---

### Goal 3: Hand Skeleton Video → `skeleton_3d.mp4`

#### Step 3.1 — HaMeR MANO hand pose prediction

```bash
conda activate hamer
bash data_process/02_mano_predict/run_step02a_mano.sh
```

**Output:** `_proc/mano_annotations/<clip_name>.json`

---

#### Step 3.2 — MANO filtering and deduplication

```bash
conda activate hamer
bash data_process/02_mano_predict/run_step02b_filter.sh
```

**Output:** `_proc/mano_filtered/<clip_name>.json`

---

#### Step 3.3 — Reconstruct full MANO (vertices + keypoints_3d)

```bash
conda activate hamer
bash data_process/02_mano_predict/run_step02c_reconstruct_full.sh
```

**Output:** `_proc/mano_full/<clip_name>.json` (vertices 778×3 + keypoints_3d 21×3)

---

#### Step 3.4 — Hand skeleton visualization

```bash
conda activate hamer
bash data_process/02_mano_predict/run_step02d_visualize_skeleton.sh
```

**Output:** `skeleton_3d.mp4`

---

### Goal 4: Video Caption → `caption.txt`

#### Step 4.1 — Qwen2.5-VL video captioning

Generates a natural-language description of the clip, used as the `prompt` field in the metadata CSV.

```bash
conda activate caption
bash data_process/03_caption/run_step03_caption.sh
```

**Output:** `tests/samples/<clip_name>/caption.txt`

To skip clips that already have a caption:

```bash
SKIP_EXISTING=1 bash data_process/03_caption/run_step03_caption.sh
```

---

### Goal 5: Metadata CSV → `<clip_name>_metadata.csv`

#### Step 5.1 — Generate metadata CSV

Assembles all pipeline outputs and the caption into the standard CSV format required by `runner.py`.

```bash
bash data_process/03_caption/run_step04_metadata.sh
```

**Output:** `tests/samples/<clip_name>_metadata.csv`

Run inference directly after:

```bash
PYTHONPATH=. python egowm/inference/runner.py \
  --dataset egovid \
  --model_root ../EgoSim-14B \
  --dataset_root tests/samples \
  --metadata_path tests/samples/<clip_name>_metadata.csv \
  --output_dir output_<clip_name> \
  --gpu_id 0
```

---

## 8. Output Structure

```
tests/samples/<clip_name>/
├── video.mp4                 # source video (copied from VIDEO_PATH by Step 0)
├── video_16fps.mp4           # step 1.1 — 16fps 720p 61-frame
├── hand_inpaint.png          # step 1.3 — inpainted first frame (inference input)
├── rendered_scene.mp4        # step 2.4 — ego prior point cloud (inference input)
├── overlay.mp4               # step 2.4 — overlay visualization (optional)
├── pc_mask_video.mp4         # step 2.5 — ego prior mask (inference input)
├── skeleton_3d.mp4           # step 3.4 — hand skeleton video (inference input)
├── caption.txt               # step 4.1 — Qwen2.5-VL generated caption
└── _proc/                    # intermediate files, not used by inference
    ├── sam_results/
    │   ├── hand_seg.png
    │   └── hand_seg_vis.jpg
    ├── inpainted/<clip_name>/
    │   ├── hand_inpaint.png
    │   ├── depth_first_frame.npy
    │   ├── intrinsics_first_frame.npy
    │   └── cloud_first_frame.npy
    ├── poses_da3/<clip_name>/
    │   ├── depth_000000.npy
    │   ├── intrinsics_*.npy
    │   ├── extrinsics_*.npy
    │   └── summary.txt
    ├── poses_da3_smoothed/<clip_name>/
    │   ├── extrinsics_*.npy
    │   └── intrinsics_*.npy
    ├── mano_annotations/<clip_name>.json
    ├── mano_filtered/<clip_name>.json
    └── mano_full/<clip_name>.json
```