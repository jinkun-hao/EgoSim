"""
Incremental video generation with accumulated 3D memory.

For each source video in the dataset, clips are grouped and processed
sequentially.  The first clip uses GT scene/skeleton inputs; subsequent
clips render the accumulated 3D point-cloud memory along the new clip's
GT camera trajectory to produce the cloud_latent input.

After each clip is generated, the EgoSim state scene pipeline runs on the
generated video to reconstruct a point cloud, which is aligned to the GT
global frame via Sim3 and fused into the accumulated memory.

Usage:
    PYTHONPATH=src python -m egowm_incremental.backends.incremental_inference \\
        --dataset egodex \\
        --model_root ../EgoSim-14B \\
        --dataset_root /path/to/Egodex \\
        --metadata_path /path/to/test_metadata.csv \\
        --output_dir /path/to/output \\
        --gpu_id 0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from egowm_incremental.paths import resolve_egosim_opensource_root
from egowm_incremental.phrases import (
    DEFAULT_SCENE_PHRASES,
    FALLBACK_QWEN_PHRASES,
    has_custom_scene_phrases,
    normalize_scene_phrases,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("incremental_inference")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 61
LATENT_CHANNELS = 16


def configure_standard_egowm_imports(egowm_root: str | None) -> None:
    candidate = resolve_egosim_opensource_root(egowm_root)
    if candidate is None:
        return
    root = str(candidate)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_standard_pipeline(model_root: str, device: torch.device):
    from egowm.inference.pipeline import load_pipeline as _load_pipeline

    return _load_pipeline(model_root, device=str(device))


def release_standard_pipeline_gpu(pipe, device: torch.device):
    """Free GPU memory held by the EgoWM pipeline before scene reconstruction."""
    if pipe is None:
        return None
    logger.info("  Releasing EgoWM pipeline GPU memory for scene reconstruction ...")
    for attr in ("dit", "vae", "text_encoder", "image_encoder"):
        module = getattr(pipe, attr, None)
        if module is not None:
            try:
                module.to("cpu")
            except Exception as exc:
                logger.warning(f"  Failed to move pipeline.{attr} to CPU: {exc}")
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None


def _standard_inference_api():
    from egowm.inference.encoders import encode_ego_prior, encode_first_frame, encode_prompt
    from egowm.inference.runner import load_mask_video as standard_load_mask_video
    from egowm.inference.runner import run_inference_single as standard_run_inference_single

    return encode_ego_prior, encode_first_frame, encode_prompt, standard_load_mask_video, standard_run_inference_single


def resolve_dataset_relative_path(dataset_root: Path, relative: str | Path | None) -> Path:
    """Resolve a metadata-relative path for both Egodex layouts and bundled mini-samples.

    Callers may set ``--dataset_root`` to either:

    - ``…/continuous_generation`` — contains ``process_result/``, optional ``test_*/`` video trees; CSV paths may
      include a ``process_result/`` prefix.
    - ``…/continuous_generation/process_result`` — all artifacts live under this root; CSV paths should omit the
      extra ``process_result/`` prefix, but rows that still include it are accepted.

    The first candidate that exists on disk is returned; otherwise the first candidate is returned for error messages.
    """
    if relative is None:
        return Path()
    rel = str(relative).strip()
    if not rel or rel.lower() == "nan":
        return Path()
    rel = rel.replace("\\", "/").lstrip("/")
    root = dataset_root.expanduser().resolve()

    candidates: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    add(root / rel)
    if rel.startswith("process_result/"):
        add(root / rel.split("process_result/", 1)[1])
    else:
        add(root / "process_result" / rel)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_relative_path(root: Path, value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    resolved = resolve_dataset_relative_path(root, text)
    return resolved if resolved != Path() else None


# ======================================================================
# Local video helpers
# ======================================================================

def _crop_and_resize(pil_image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Scale to cover target area then center-crop (matches standard ImageCropAndResize)."""
    import torchvision.transforms.functional as TF
    w, h = pil_image.size
    scale = max(target_width / w, target_height / h)
    pil_image = TF.resize(
        pil_image,
        (round(h * scale), round(w * scale)),
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    pil_image = TF.center_crop(pil_image, (target_height, target_width))
    return pil_image


def load_video_as_tensor(video_path: str, height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Read video -> [3, F, H, W] float tensor in [-1, 1].

    Uses crop_and_resize + BILINEAR to match standard preprocess pipeline.
    """
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()

    if len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))
    else:
        frames = frames[:NUM_FRAMES]

    processed = []
    for f in frames:
        pil_img = Image.fromarray(f)
        pil_img = _crop_and_resize(pil_img, height, width)
        processed.append(np.array(pil_img))

    arr = np.stack(processed, axis=0)
    t = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    t = t / 255.0 * 2.0 - 1.0
    return t


def frames_to_tensor(frames: list[np.ndarray], height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Convert list of uint8 [H, W, 3] frames -> [3, F, H, W] float tensor in [-1, 1].

    Each frame is crop-and-resized to (height, width) before conversion.
    """
    if len(frames) < NUM_FRAMES:
        frames = frames + [frames[-1]] * (NUM_FRAMES - len(frames))
    else:
        frames = frames[:NUM_FRAMES]

    processed = []
    for f in frames:
        pil_img = Image.fromarray(f)
        pil_img = _crop_and_resize(pil_img, height, width)
        processed.append(np.array(pil_img))

    arr = np.stack(processed, axis=0)
    t = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    t = t / 255.0 * 2.0 - 1.0
    return t


def _save_tensor_as_video(tensor: torch.Tensor, save_path: str, fps: int = 16):
    """Save [3, F, H, W] float tensor in [-1, 1] back to mp4 (for debug / visualization)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = tensor.permute(1, 2, 3, 0)  # [F, H, W, 3]
    out = ((out + 1.0) / 2.0).clamp(0, 1)
    frames = (out * 255).numpy().astype(np.uint8)
    imageio.mimwrite(save_path, frames, fps=fps, quality=8)


def _save_mask_as_video(mask: torch.Tensor, save_path: str, fps: int = 16):
    """Save [1, F, H, W] float tensor in [0, 1] as grayscale mp4 (for debug)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = mask.squeeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)  # [F, H, W, 3]
    frames = (out.clamp(0, 1) * 255).numpy().astype(np.uint8)
    imageio.mimwrite(save_path, frames, fps=fps, quality=8)


@torch.no_grad()
def encode_video_online(pipe, video_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """VAE encode: [3, F, H, W] -> [16, f_lat, h_lat, w_lat]."""
    vae = pipe.vae
    vae.to(device)
    dtype = next(vae.parameters()).dtype
    x = video_tensor.unsqueeze(0).to(device, dtype=dtype)
    latent = vae.encode(x, device=device)
    return latent.squeeze(0).to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_text_online(pipe, text: str, device: torch.device) -> torch.Tensor:
    """T5 encode: text -> [512, 4096]."""
    pipe.text_encoder.to(device)
    if pipe.prompter.text_encoder is None:
        pipe.prompter.fetch_models(pipe.text_encoder)
    emb = pipe.prompter.encode_prompt(text, positive=True, device=device)
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_image_online(pipe, pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """CLIP encode: PIL image -> [257, 1280].

    Matches standard preprocess: image in [0, 1] fed to encode_image
    (encode_image internally does x*0.5+0.5 then CLIP norm).
    """
    pil_image = _crop_and_resize(pil_image, HEIGHT, WIDTH)
    pipe.image_encoder.to(device)
    img_arr = np.array(pil_image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    emb = pipe.image_encoder.encode_image(img_tensor)
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.to(dtype=torch.bfloat16)


# ======================================================================
# Denoising (from inference_realcap_wan21.py)
# ======================================================================

def load_mask_video(mask_path: str, target_frames: int = NUM_FRAMES, height: int = HEIGHT, width: int = WIDTH):
    """Load mask video -> [1, F, H, W] tensor in [0, 1]."""
    from PIL import Image as _PILImage

    reader = imageio.get_reader(mask_path)
    frames = []
    for frame_data in reader:
        frame = _crop_and_resize(_PILImage.fromarray(frame_data), height, width)
        frames.append(frame)
    reader.close()

    if len(frames) < target_frames:
        last = frames[-1] if frames else _PILImage.new("RGB", (width, height))
        frames += [last] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]

    arr = np.stack([np.array(f) for f in frames], axis=0)
    tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    tensor = tensor / 255.0 * 2.0 - 1.0
    tensor = tensor.unsqueeze(0)

    mask_video_raw = (tensor + 1.0) / 2.0
    mask_video_raw = mask_video_raw.clamp(0, 1)
    mask_video_raw = mask_video_raw[:, :1].squeeze(0)
    mask_video_raw[:, 0, :, :] = 0.0
    return mask_video_raw


def encode_mask_to_latent(mask_video_raw: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """mask_video_raw: [1, F, H, W] -> latent mask [4, f, h, w]."""
    import torch.nn.functional as F_func
    _, target_f, target_h, target_w = target_shape
    mask = torch.where(mask_video_raw > 0.5, 1.0, 0.0)
    mask_ds = F_func.interpolate(
        mask.unsqueeze(0),
        size=(target_f, target_h * 2, target_w * 2),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    mask_p = mask_ds.view(target_f, target_h, 2, target_w, 2)
    mask_p = mask_p.permute(2, 4, 0, 1, 3).reshape(4, target_f, target_h, target_w)
    return mask_p


def model_fn_ego_hand(dit, latents, timestep, context, clip_feature):
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
    from einops import rearrange

    dev = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype
    timestep = timestep.to(device=dev)
    latents = latents.to(device=dev, dtype=dtype)
    context = context.to(device=dev, dtype=dtype)
    if clip_feature is not None:
        clip_feature = clip_feature.to(device=dev, dtype=dtype)

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if clip_feature is not None and hasattr(dit, "img_emb"):
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)

    x = dit.patch_embedding(latents)
    b, c, f, h, w = x.shape
    x = rearrange(x, "b c f h w -> b (f h w) c")

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)

    t_mod_head = t.unsqueeze(1).expand(-1, 2, -1)
    x = dit.head(x, t_mod_head)
    return dit.unpatchify(x, (f, h, w))


@torch.no_grad()
def run_inference_single(
    pipe,
    cloud_latent: torch.Tensor,
    hand_latent: torch.Tensor,
    mask_video_raw: torch.Tensor,
    prompt_embedding: torch.Tensor,
    image_embedding: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = 50,
    cfg_scale: float = 1.0,
) -> np.ndarray:
    """Denoise and VAE-decode. Returns [T, H, W, C] uint8."""
    dtype = pipe.torch_dtype

    cloud = cloud_latent.unsqueeze(0).to(device, dtype=dtype)
    hand = hand_latent.unsqueeze(0).to(device, dtype=dtype)
    mask_r = mask_video_raw.to(device, dtype=dtype)

    mask_latent = encode_mask_to_latent(mask_r, cloud.shape[1:])
    mask_latent = mask_latent.unsqueeze(0).to(device, dtype=dtype)

    ctx = prompt_embedding.unsqueeze(0).to(device, dtype=dtype)
    clip = image_embedding.unsqueeze(0).to(device, dtype=dtype)

    latents = torch.randn_like(cloud)

    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps.to(device)

    pipe.dit.to(device)

    for t_step in tqdm(timesteps, desc="Denoising", leave=False):
        mask_w = mask_latent[:, :1].expand(-1, LATENT_CHANNELS, -1, -1, -1)
        masked_ego = cloud * (1.0 - mask_w)
        model_in = torch.cat([latents, mask_latent, masked_ego, hand], dim=1)
        ts = t_step.unsqueeze(0).to(dtype=dtype, device=device)
        noise_pred = model_fn_ego_hand(pipe.dit, model_in, ts, ctx, clip)
        latents = pipe.scheduler.step(noise_pred, t_step, latents)

    pipe.vae = pipe.vae.to(device)

    generated_video = pipe.vae.decode(
        [latents.squeeze(0)], device=device,
    )
    out = generated_video.squeeze(0).permute(1, 2, 3, 0)
    out = ((out + 1.0) / 2.0).clamp(0, 1)
    return (out * 255).float().cpu().numpy().astype(np.uint8)


# ======================================================================
# Subprocess helpers for env-switching
# ======================================================================

def _run_subprocess(python_bin: str, script: str, args_list: list[str],
                    env_extra: dict | None = None, label: str = "subprocess"):
    """Run an external Python script in a different conda env, streaming output."""
    cmd = [python_bin, "-u", script] + args_list
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
    logger.info(f"[{label}] Running: {' '.join(cmd[:5])}...")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    last_lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        logger.info(f"[{label}] {line}")
        last_lines.append(line)
        if len(last_lines) > 200:
            last_lines.pop(0)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")
    return proc


def predict_phrases_subprocess(
    video_path: str,
    output_json: str,
    da3_python: str,
    predict_phrases_script: str,
    qwen_model_path: str,
) -> list[str]:
    """Call predict_phrases.py in the depthanything3 env via subprocess."""
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    _run_subprocess(
        da3_python, predict_phrases_script,
        ["--video", str(video_path),
         "--model_path", qwen_model_path,
         "--output_json", output_json],
        env_extra={"PYOPENGL_PLATFORM": "egl"},
        label="predict_phrases",
    )
    with open(output_json) as f:
        return json.load(f)


def resolve_scene_phrases_for_clip(
    *,
    clip_video_path: Path,
    phrases_json_path: Path,
    predict_phrases_script: str,
    qwen_model_path: str,
    da3_python: str,
) -> list[str]:
    if phrases_json_path.exists():
        with open(phrases_json_path) as f:
            existing = json.load(f)
        merged_existing = normalize_scene_phrases(existing)
        with open(phrases_json_path, "w") as f:
            json.dump(merged_existing, f, indent=2, ensure_ascii=False)
        return merged_existing

    try:
        predicted = predict_phrases_subprocess(
            video_path=str(clip_video_path),
            output_json=str(phrases_json_path),
            da3_python=da3_python,
            predict_phrases_script=predict_phrases_script,
            qwen_model_path=qwen_model_path,
        )
        merged = normalize_scene_phrases(predicted)
    except Exception as e:
        logger.warning(f"  Phrase prediction subprocess failed: {e}")
        traceback.print_exc()
        merged = normalize_scene_phrases(FALLBACK_QWEN_PHRASES)

    with open(phrases_json_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return merged


def should_reuse_existing_generated_video(
    *,
    out_video_exists: bool,
    skip_existing: bool,
    egosim_state_only: bool,
    run_egosim_state: bool,
) -> bool:
    return skip_existing and out_video_exists and not egosim_state_only and run_egosim_state


def run_egosim_state_subprocess(
    generated_video: str,
    hdf5_path: str,
    process_result_dir: str,
    egosim_state_output_dir: str,
    output_memory: str,
    egosim_state_python: str,
    egosim_state_script: str,
    cumulative_memory: str = "",
    phrases: list[str] | None = None,
    phrases_json: str = "",
    next_hdf5_path: str = "",
    next_gt_process_result_dir: str = "",
    clip_start_frame: int = -1,
    clip_end_frame: int = -1,
    next_clip_start_frame: int = -1,
    next_clip_end_frame: int = -1,
    rendered_video_out: str = "",
    mask_video_out: str = "",
    egosim_state_pipeline: str = "dav3",
    scene_save_viz: bool = False,
    scene_slam_visualize: bool = False,
    gpu_id: int = 0,
    spatial_subsample: int = 2,
    temporal_subsample: int = 1,
    voxel_size: float = 0.0,
    fuse_overlap_radius: float = 0.005,
    render_point_size: float = 3.0,
    fps: int = 16,
    filter_interactive: bool = True,
    mask_dilation: int = 5,
    use_tsdf: bool = True,
    tsdf_voxel_size: float = 0.0025,
    tsdf_trunc_multiplier: float = 20.0,
    use_color_depth_overlap: bool = False,
    overlap_depth_thresh: float = 0.05,
    overlap_color_thresh: float = 40.0,
    use_last_frame_objects: bool = True,
    statistical_outlier_removal: bool = True,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    render_gt_pointcloud: bool = False,
    opencv_to_opengl_points: bool = False,
    prefer_icp_alignment: bool = False,
    align_method: str = "pose_sim3",
    pose_center_sample_step: int = 1,
    icp_refine: bool = False,
    icp_refine_iters: int = 10,
    icp_refine_corr_dist: float = 0.05,
    pointcloud_video_out: str = "",
    viser: bool = False,
    viser_backend: str = "egosim_state",
    viser_host: str = "auto",
    viser_port: int = 20540,
    viser_point_size: float = 0.002,
    viser_max_points: int = 800_000,
) -> dict:
    """Call egosim_state_subprocess.py in the scene-backend env via subprocess.

    Returns the result dict from the subprocess (alignment, fusion stats, etc).
    """
    Path(output_memory).parent.mkdir(parents=True, exist_ok=True)
    cmd_args = [
        "--generated_video", str(generated_video),
        "--hdf5_path", str(hdf5_path),
        "--process_result_dir", str(process_result_dir),
        "--output_dir", str(egosim_state_output_dir),
        "--output_memory", str(output_memory),
        "--egosim_state_pipeline", egosim_state_pipeline,
        "--gpu_id", str(gpu_id),
        "--spatial_subsample", str(spatial_subsample),
        "--temporal_subsample", str(temporal_subsample),
        "--voxel_size", str(voxel_size),
        "--fuse_overlap_radius", str(fuse_overlap_radius),
        "--render_point_size", str(render_point_size),
        "--fps", str(fps),
        "--mask_dilation", str(mask_dilation),
        "--tsdf_voxel_size", str(tsdf_voxel_size),
        "--tsdf_trunc_multiplier", str(tsdf_trunc_multiplier),
        "--overlap_depth_thresh", str(overlap_depth_thresh),
        "--overlap_color_thresh", str(overlap_color_thresh),
        "--outlier_nb_neighbors", str(outlier_nb_neighbors),
        "--outlier_std_ratio", str(outlier_std_ratio),
    ]
    cmd_args.append("--scene_save_viz" if scene_save_viz else "--no_scene_save_viz")
    cmd_args.append("--scene_slam_visualize" if scene_slam_visualize else "--no_scene_slam_visualize")
    if cumulative_memory:
        cmd_args += ["--cumulative_memory", str(cumulative_memory)]
    if phrases:
        cmd_args += ["--phrases", *[str(phrase) for phrase in phrases]]
    elif phrases_json:
        cmd_args += ["--phrases_json", str(phrases_json)]
    if next_hdf5_path:
        cmd_args += ["--next_hdf5_path", str(next_hdf5_path)]
    if next_gt_process_result_dir:
        cmd_args += ["--next_gt_process_result_dir", str(next_gt_process_result_dir)]
    if clip_start_frame >= 0 and clip_end_frame >= 0:
        cmd_args += ["--clip_start_frame", str(clip_start_frame)]
        cmd_args += ["--clip_end_frame", str(clip_end_frame)]
    if next_clip_start_frame >= 0 and next_clip_end_frame >= 0:
        cmd_args += ["--next_clip_start_frame", str(next_clip_start_frame)]
        cmd_args += ["--next_clip_end_frame", str(next_clip_end_frame)]
    if rendered_video_out:
        cmd_args += ["--rendered_video_out", str(rendered_video_out)]
    if mask_video_out:
        cmd_args += ["--mask_video_out", str(mask_video_out)]
    cmd_args.append("--filter_interactive" if filter_interactive else "--no_filter_interactive")
    cmd_args.append("--use_tsdf" if use_tsdf else "--no_use_tsdf")
    cmd_args.append("--use_color_depth_overlap" if use_color_depth_overlap else "--no_use_color_depth_overlap")
    cmd_args.append("--use_last_frame_objects" if use_last_frame_objects else "--no_use_last_frame_objects")
    cmd_args.append("--statistical_outlier_removal" if statistical_outlier_removal else "--no_statistical_outlier_removal")
    if render_gt_pointcloud:
        cmd_args.append("--render_gt_pointcloud")
    if opencv_to_opengl_points:
        cmd_args.append("--opencv_to_opengl_points")
    if prefer_icp_alignment:
        cmd_args.append("--prefer_icp_alignment")
    cmd_args += ["--align_method", align_method]
    cmd_args += ["--pose_center_sample_step", str(pose_center_sample_step)]
    if pointcloud_video_out:
        cmd_args += ["--pointcloud_video_out", str(pointcloud_video_out)]
    if viser:
        cmd_args.append("--viser")
        cmd_args += ["--viser_backend", str(viser_backend)]
        cmd_args += ["--viser_host", str(viser_host)]
        cmd_args += ["--viser_port", str(viser_port)]
        cmd_args += ["--viser_point_size", str(viser_point_size)]
        cmd_args += ["--viser_max_points", str(viser_max_points)]
    cmd_args.append("--icp_refine" if icp_refine else "--no_icp_refine")
    cmd_args += ["--icp_refine_iters", str(icp_refine_iters)]
    cmd_args += ["--icp_refine_corr_dist", str(icp_refine_corr_dist)]

    _env_extra = {
        "PYOPENGL_PLATFORM": "egl",
        "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [
            "/usr/local/nvidia/lib64",
            "/usr/lib/x86_64-linux-gnu",
            os.environ.get("LD_LIBRARY_PATH", ""),
        ])),
        "TORCH_HOME": os.environ.get(
            "TORCH_HOME",
            str(Path(__file__).resolve().parent / "checkpoints"),
        ),
        "PRIORDA_WEIGHTS_DIR": os.environ.get(
            "PRIORDA_WEIGHTS_DIR",
            str(Path(__file__).resolve().parents[3] / "artifacts" / "models" / "priorda"),
        ),
    }
    _egosim_state_root = (
        os.environ.get("EGOWM_EGOSIM_STATE_ROOT")
        or os.environ.get("EGOSIM_STATE_ROOT")
    )
    if viser and viser_backend == "egosim_state" and _egosim_state_root:
        _pp = os.environ.get("PYTHONPATH", "")
        _env_extra["PYTHONPATH"] = (
            os.pathsep.join([_egosim_state_root, _pp]) if _pp else _egosim_state_root
        )

    _run_subprocess(
        egosim_state_python, egosim_state_script, cmd_args,
        env_extra=_env_extra,
        label="egosim_state_subprocess",
    )

    result_json = Path(output_memory).with_suffix(".json")
    if result_json.exists():
        with open(result_json) as f:
            return json.load(f)
    return {"status": "unknown"}






# ======================================================================
# Clip grouping
# ======================================================================

def group_clips_by_source_video(metadata_path: str, part_prefix: str = "test_16fps_720p"):
    """Parse metadata CSV and group clips by (task_name, video_id).

    Returns dict: (task_name, video_id) -> list of clip dicts sorted by start_frame.
    Each clip dict has keys: video, ego_prior_video, prompt, task_name, clip_stem,
    start_frame, video_id.
    """
    import pandas as pd

    df = pd.read_csv(metadata_path)
    groups = defaultdict(list)

    for _, row in df.iterrows():
        video_path = row["video"]
        parts = video_path.split("/")
        if "task_name" in row and pd.notna(row["task_name"]) and str(row["task_name"]).strip():
            task_name = str(row["task_name"]).strip()
        else:
            task_name = parts[1] if len(parts) > 2 else parts[0]
        stem = Path(video_path).stem
        stem_for_tokens = stem.removeprefix("GT_") if stem.startswith("GT_") else stem
        tokens = stem_for_tokens.split("_")
        video_id = tokens[0]
        start_frame = int(tokens[1])
        end_frame = int(tokens[2]) if len(tokens) >= 3 else start_frame + NUM_FRAMES

        clip_info = {
            "video": video_path,
            "ego_prior_video": row.get("ego_prior_video", ""),
            "hand_keypoint_video": row.get("hand_keypoint_video", ""),
            "first_frame": row.get("first_frame", ""),
            "prompt": row.get("prompt", ""),
            "task_name": task_name,
            "clip_stem": stem,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "video_id": video_id,
            "part_prefix": part_prefix,
        }
        if "process_result_dir" in row and pd.notna(row["process_result_dir"]):
            clip_info["process_result_dir"] = row["process_result_dir"]
        if "hdf5_path" in row and pd.notna(row["hdf5_path"]):
            clip_info["hdf5_path"] = row["hdf5_path"]
        if "gt_process_result_dir" in row and pd.notna(row["gt_process_result_dir"]):
            clip_info["gt_process_result_dir"] = row["gt_process_result_dir"]
        groups[(task_name, video_id)].append(clip_info)

    for key in groups:
        groups[key].sort(key=lambda x: x["start_frame"])

    return dict(groups)


# ======================================================================
# Main pipeline
# ======================================================================

def process_clip_group(
    group_key: tuple[str, str],
    clips: list[dict],
    pipe,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """Process a group of clips from the same source video incrementally."""
    task_name, video_id = group_key
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    group_output = output_dir / task_name / video_id
    group_output.mkdir(parents=True, exist_ok=True)

    summary = {"task_name": task_name, "video_id": video_id, "clips": []}

    memory_path = None  # path to current cumulative memory .npz
    has_memory = False

    for clip_idx, clip_info in enumerate(clips):
        clip_stem = clip_info["clip_stem"]
        part_prefix = clip_info["part_prefix"]
        logger.info(f"[{task_name}/{video_id}] Processing clip {clip_idx}/{len(clips)}: {clip_stem}")

        out_video_path = group_output / f"{clip_stem}.mp4"
        clip_phrases_json_path = group_output / f"{clip_stem}_phrases.json"
        clip_result = {"clip_stem": clip_stem, "clip_idx": clip_idx}

        if args.skip_existing and out_video_path.exists() and not args.egosim_state_only and not args.run_egosim_state:
            logger.info(f"  Skipping (exists): {out_video_path}")
            clip_result["status"] = "skipped"
            summary["clips"].append(clip_result)
            # If EgoSim state output exists from a previous run, track its memory
            prev_mem = group_output / f"{clip_stem}_memory.npz"
            if prev_mem.exists():
                memory_path = str(prev_mem)
                has_memory = True
            continue

        try:
            if "process_result_dir" in clip_info:
                process_result_dir = resolve_dataset_relative_path(
                    dataset_root, str(clip_info["process_result_dir"])
                )
            else:
                process_result_dir = resolve_dataset_relative_path(
                    dataset_root, f"process_result/{part_prefix}/{task_name}/{clip_stem}"
                )

            if "hdf5_path" in clip_info:
                hdf5_path = resolve_dataset_relative_path(
                    dataset_root, str(clip_info["hdf5_path"])
                )
            else:
                hdf5_path = resolve_dataset_relative_path(
                    dataset_root, f"{part_prefix}/{task_name}/{clip_stem}.hdf5"
                )

            if "gt_process_result_dir" in clip_info:
                gt_process_result_dir = resolve_dataset_relative_path(
                    dataset_root, str(clip_info["gt_process_result_dir"])
                )
            else:
                gt_process_result_dir = process_result_dir

            if not gt_process_result_dir.exists():
                gt_process_result_dir = process_result_dir

            if not process_result_dir.exists():
                logger.warning(f"  Missing process_result dir: {process_result_dir}")
                if args.run_egosim_state:
                    clip_result["status"] = "missing_data"
                    summary["clips"].append(clip_result)
                    continue

            has_hdf5 = hdf5_path.exists()
            if not has_hdf5:
                logger.info(f"  No HDF5 found: {hdf5_path} (Sim3 alignment & rendering will be skipped)")

            # ---- EgoSim-state-only: use existing video if available, else fall back to generation ----
            _skip_generation = False
            if should_reuse_existing_generated_video(
                out_video_exists=out_video_path.exists(),
                skip_existing=args.skip_existing,
                egosim_state_only=args.egosim_state_only,
                run_egosim_state=args.run_egosim_state,
            ):
                logger.info(f"  Reusing existing generated video for reconstruction: {out_video_path}")
                _skip_generation = True
                clip_result["reused_existing_video"] = True
            elif args.egosim_state_only:
                if out_video_path.exists():
                    logger.info(f"  egosim_state_only: using existing {out_video_path}")
                    _skip_generation = True
                else:
                    logger.info(f"  egosim_state_only: video not found {out_video_path}, falling back to generation")
                    if pipe is None:
                        logger.info("  Lazy-loading standard EgoWM pipeline for generation ...")
                        pipe = load_standard_pipeline(args.model_root, device)
                        logger.info("  Pipeline loaded.")

            # ---- Prepare inputs + inference + phrases (skip when video already exists) ----
            if not _skip_generation:
                if pipe is None:
                    logger.info("  Lazy-loading standard EgoWM pipeline for generation ...")
                    pipe = load_standard_pipeline(args.model_root, device)
                    logger.info("  Pipeline loaded.")

                (
                    standard_encode_ego_prior,
                    standard_encode_first_frame,
                    standard_encode_prompt,
                    standard_load_mask_video,
                    standard_run_inference_single,
                ) = _standard_inference_api()

                if clip_idx == 0:
                    scene_path = _resolve_relative_path(dataset_root, clip_info.get("ego_prior_video"))
                    skel_path = _resolve_relative_path(dataset_root, clip_info.get("hand_keypoint_video"))
                    inpaint_path = _resolve_relative_path(dataset_root, clip_info.get("first_frame"))

                    if scene_path is None:
                        scene_path = process_result_dir / "rendered_scene.mp4"
                    if skel_path is None:
                        skel_path = process_result_dir / "skeleton_3d.mp4"
                    if inpaint_path is None:
                        inpaint_path = process_result_dir / "hand_inpaint.png"

                    if not scene_path.exists() or not skel_path.exists() or not inpaint_path.exists():
                        logger.warning(f"  Missing GT inputs for first clip: {clip_stem}")
                        clip_result["status"] = "missing_gt_inputs"
                        summary["clips"].append(clip_result)
                        continue

                    logger.info(f"  Standard input ego_prior_video: {scene_path}")
                    logger.info(f"  Standard input hand_keypoint_video: {skel_path}")
                    logger.info(f"  Standard input first_frame: {inpaint_path}")
                    cloud_latent = standard_encode_ego_prior(
                        pipe, str(scene_path), device,
                        target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                    )
                    hand_latent = standard_encode_ego_prior(
                        pipe, str(skel_path), device,
                        target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                    )
                    image_embedding = standard_encode_first_frame(
                        pipe, str(inpaint_path), device, height=HEIGHT, width=WIDTH,
                    )

                    mask_path = process_result_dir / "pc_mask_video.mp4"
                    metadata_ego_prior = _resolve_relative_path(dataset_root, clip_info.get("ego_prior_video"))
                    if metadata_ego_prior is not None:
                        mask_path = metadata_ego_prior.parent / "pc_mask_video.mp4"
                    if mask_path.exists():
                        mask_video_raw = standard_load_mask_video(
                            str(mask_path), target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                        )
                        non_zero_ratio = (mask_video_raw.abs() > 0.01).float().mean().item()
                        logger.info(f"  [DEBUG] Loaded GT pc_mask from {mask_path}, "
                                    f"shape={list(mask_video_raw.shape)}, non_zero_ratio={non_zero_ratio:.4f}")
                    else:
                        mask_video_raw = torch.zeros(1, NUM_FRAMES, HEIGHT, WIDTH)
                        logger.info(f"  [DEBUG] pc_mask NOT found at {mask_path}, using all-zeros mask")

                    # Save resized inputs for debugging
                    _save_mask_as_video(mask_video_raw, str(group_output / f"{clip_stem}_input_mask.mp4"), fps=args.fps)
                    logger.info(f"  Saved resized mask input to {group_output}")

                else:
                    # Subsequent clips: keep standard single-clip fields, but substitute ego_prior_video
                    # with the rendered cumulative memory when it is available.
                    rendered_memory_path = group_output / f"{clip_stem}_rendered_memory.mp4"
                    if has_memory and rendered_memory_path.exists():
                        logger.info(f"  Using rendered memory: {rendered_memory_path}")
                        scene_path = rendered_memory_path
                    else:
                        scene_path = _resolve_relative_path(dataset_root, clip_info.get("ego_prior_video"))
                        if scene_path is None:
                            scene_path = process_result_dir / "rendered_scene.mp4"
                        logger.warning(f"  No rendered memory for {clip_stem}, using standard ego_prior_video fallback")

                    skel_path = _resolve_relative_path(dataset_root, clip_info.get("hand_keypoint_video"))
                    inpaint_path = _resolve_relative_path(dataset_root, clip_info.get("first_frame"))
                    if skel_path is None:
                        skel_path = process_result_dir / "skeleton_3d.mp4"
                    if inpaint_path is None:
                        inpaint_path = process_result_dir / "hand_inpaint.png"

                    if not scene_path.exists() or not skel_path.exists() or not inpaint_path.exists():
                        logger.warning(f"  Missing standard inputs for clip: {clip_stem}")
                        clip_result["status"] = "missing_standard_inputs"
                        summary["clips"].append(clip_result)
                        continue

                    logger.info(f"  Standard input ego_prior_video: {scene_path}")
                    logger.info(f"  Standard input hand_keypoint_video: {skel_path}")
                    logger.info(f"  Standard input first_frame: {inpaint_path}")
                    cloud_latent = standard_encode_ego_prior(
                        pipe, str(scene_path), device,
                        target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                    )
                    hand_latent = standard_encode_ego_prior(
                        pipe, str(skel_path), device,
                        target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                    )
                    image_embedding = standard_encode_first_frame(
                        pipe, str(inpaint_path), device, height=HEIGHT, width=WIDTH,
                    )

                    mask_path = group_output / f"{clip_stem}_mask.mp4"
                    if mask_path.exists():
                        logger.info(f"  Using rendered mask: {mask_path}")
                        mask_video_raw = standard_load_mask_video(
                            str(mask_path), target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                        )
                    else:
                        metadata_ego_prior = _resolve_relative_path(dataset_root, clip_info.get("ego_prior_video"))
                        gt_mask_path = (
                            metadata_ego_prior.parent / "pc_mask_video.mp4"
                            if metadata_ego_prior is not None
                            else process_result_dir / "pc_mask_video.mp4"
                        )
                        if gt_mask_path.exists():
                            mask_video_raw = standard_load_mask_video(
                                str(gt_mask_path), target_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                            )
                        else:
                            logger.warning(f"  No mask video for {clip_stem}, using all-zeros (no inpainting mask)")
                            mask_video_raw = torch.zeros(1, NUM_FRAMES, HEIGHT, WIDTH)

                    # Save resized inputs for debugging
                    _save_mask_as_video(mask_video_raw, str(group_output / f"{clip_stem}_input_mask.mp4"), fps=args.fps)
                    logger.info(f"  Saved resized mask input to {group_output}")

                # Prompt embedding
                prompt_text = clip_info.get("prompt", "")
                if not isinstance(prompt_text, str):
                    prompt_text = ""
                if args.use_long_prompt:
                    long_prompt_path = process_result_dir / "long_prompt.txt"
                    if long_prompt_path.exists():
                        prompt_text = long_prompt_path.read_text().strip()
                        logger.info(f"  Using long prompt from {long_prompt_path}: {prompt_text[:80]}...")
                    else:
                        logger.warning(f"  long_prompt.txt not found at {long_prompt_path}, falling back to CSV prompt")
                if prompt_text:
                    prompt_embedding = standard_encode_prompt(pipe, prompt_text, device)
                else:
                    prompt_embedding = torch.zeros(512, 4096, dtype=torch.bfloat16, device=device)

                # ---- Run denoising ----
                logger.info(f"  Running inference for {clip_stem}")
                generated_video = standard_run_inference_single(
                    pipe=pipe,
                    cloud_latent=cloud_latent,
                    hand_latent=hand_latent,
                    mask_video_raw=mask_video_raw,
                    prompt_embedding=prompt_embedding,
                    image_embedding=image_embedding,
                    device=device,
                    num_inference_steps=args.num_inference_steps,
                )

                imageio.mimwrite(str(out_video_path), generated_video, fps=args.fps, quality=8)
                logger.info(f"  Saved generated video: {out_video_path}")

            clip_phrases = None
            if args.predict_phrases and out_video_path.exists():
                clip_phrases = resolve_scene_phrases_for_clip(
                    clip_video_path=out_video_path,
                    phrases_json_path=clip_phrases_json_path,
                    predict_phrases_script=args.predict_phrases_script,
                    qwen_model_path=args.qwen_model_path,
                    da3_python=args.da3_python,
                )
                logger.info(f"  Scene phrases for {clip_stem}: {clip_phrases}")
                clip_result["phrases_json"] = str(clip_phrases_json_path)
                clip_result["phrases_count"] = len(clip_phrases)
            elif args.run_egosim_state:
                clip_phrases = normalize_scene_phrases(args.scene_phrases)
                logger.info(f"  Using configured scene phrases for {clip_stem}: {clip_phrases}")

            # ---- EgoSim state reconstruction via subprocess ----
            if args.run_egosim_state:
                try:
                    egosim_state_workdir = group_output / f"{clip_stem}_egosim_state"
                    clip_memory_path = str(group_output / f"{clip_stem}_memory.npz")

                    # Determine next clip's HDF5 and frame range for pre-rendering
                    next_hdf5 = ""
                    next_clip_start_frame = -1
                    next_clip_end_frame = -1
                    rendered_for_next = ""
                    mask_for_next = ""
                    next_gt_process_result_dir = ""
                    clip_start_frame = clip_info.get("start_frame", -1)
                    clip_end_frame = clip_info.get("end_frame", -1)
                    next_clip_start_frame = -1
                    next_clip_end_frame = -1
                    if (
                        has_hdf5
                        and clip_idx + 1 < len(clips)
                        and getattr(args, "incremental_mode", "continue_generate") != "recon_visualize"
                    ):
                        next_clip = clips[clip_idx + 1]
                        next_stem = next_clip["clip_stem"]
                        next_clip_start_frame = next_clip.get("start_frame", -1)
                        next_clip_end_frame = next_clip.get("end_frame", -1)
                        if "hdf5_path" in next_clip:
                            next_hdf5_candidate = resolve_dataset_relative_path(
                                dataset_root, str(next_clip["hdf5_path"])
                            )
                        else:
                            next_hdf5_candidate = resolve_dataset_relative_path(
                                dataset_root,
                                f"{next_clip['part_prefix']}/{task_name}/{next_stem}.hdf5",
                            )
                        if next_hdf5_candidate.exists():
                            next_hdf5 = str(next_hdf5_candidate)
                            rendered_for_next = str(group_output / f"{next_stem}_rendered_memory.mp4")
                            mask_for_next = str(group_output / f"{next_stem}_mask.mp4")
                            if "gt_process_result_dir" in next_clip:
                                next_gt = resolve_dataset_relative_path(
                                    dataset_root, str(next_clip["gt_process_result_dir"])
                                )
                            elif "process_result_dir" in next_clip:
                                next_gt = resolve_dataset_relative_path(
                                    dataset_root, str(next_clip["process_result_dir"])
                                )
                            else:
                                next_gt = Path()
                            if next_gt != Path() and next_gt.exists():
                                next_gt_process_result_dir = str(next_gt)
                            elif "process_result_dir" in next_clip:
                                next_gt_process_result_dir = str(
                                    resolve_dataset_relative_path(
                                        dataset_root, str(next_clip["process_result_dir"])
                                    )
                                )

                    _pc_turntable = ""
                    _use_viser = False
                    if getattr(args, "incremental_mode", "continue_generate") == "recon_visualize":
                        _use_viser = not getattr(args, "no_recon_viser", False)

                    if not args.egosim_state_only:
                        pipe = release_standard_pipeline_gpu(pipe, device)

                    egosim_state_result = run_egosim_state_subprocess(
                        generated_video=str(out_video_path),
                        hdf5_path=str(hdf5_path) if has_hdf5 else "",
                        process_result_dir=str(gt_process_result_dir),
                        egosim_state_output_dir=str(egosim_state_workdir),
                        output_memory=clip_memory_path,
                        egosim_state_python=args.egosim_state_python,
                        egosim_state_script=args.egosim_state_script,
                        cumulative_memory=memory_path or "",
                        phrases=clip_phrases,
                        phrases_json=(
                            str(clip_phrases_json_path)
                            if args.predict_phrases and clip_phrases_json_path.exists()
                            else ""
                        ),
                        next_hdf5_path=next_hdf5,
                        next_gt_process_result_dir=next_gt_process_result_dir or "",
                        clip_start_frame=clip_start_frame,
                        clip_end_frame=clip_end_frame,
                        next_clip_start_frame=next_clip_start_frame,
                        next_clip_end_frame=next_clip_end_frame,
                        rendered_video_out=rendered_for_next,
                        mask_video_out=mask_for_next,
                        egosim_state_pipeline=args.egosim_state_pipeline,
                        scene_save_viz=(
                            getattr(args, "incremental_mode", "continue_generate") == "recon_visualize"
                        ),
                        scene_slam_visualize=(
                            getattr(args, "incremental_mode", "continue_generate") == "recon_visualize"
                        ),
                        gpu_id=args.gpu_id,
                        spatial_subsample=args.spatial_subsample,
                        temporal_subsample=args.temporal_subsample,
                        voxel_size=args.voxel_size,
                        fuse_overlap_radius=args.fuse_overlap_radius,
                        render_point_size=args.render_point_size,
                        fps=args.fps,
                        filter_interactive=args.filter_interactive,
                        mask_dilation=args.mask_dilation,
                        use_tsdf=args.use_tsdf,
                        tsdf_voxel_size=args.tsdf_voxel_size,
                        tsdf_trunc_multiplier=args.tsdf_trunc_multiplier,
                        use_color_depth_overlap=args.use_color_depth_overlap,
                        overlap_depth_thresh=args.overlap_depth_thresh,
                        overlap_color_thresh=args.overlap_color_thresh,
                        use_last_frame_objects=args.use_last_frame_objects,
                        statistical_outlier_removal=args.statistical_outlier_removal,
                        outlier_nb_neighbors=args.outlier_nb_neighbors,
                        outlier_std_ratio=args.outlier_std_ratio,
                        render_gt_pointcloud=args.render_gt_pointcloud,
                        opencv_to_opengl_points=args.opencv_to_opengl_points,
                        prefer_icp_alignment=args.prefer_icp_alignment,
                        align_method=args.align_method,
                        pose_center_sample_step=args.pose_center_sample_step,
                        icp_refine=args.icp_refine,
                        icp_refine_iters=args.icp_refine_iters,
                        icp_refine_corr_dist=args.icp_refine_corr_dist,
                        pointcloud_video_out=_pc_turntable,
                        viser=_use_viser,
                        viser_backend=args.recon_viser_backend,
                        viser_host=args.recon_viser_host,
                        viser_port=args.recon_viser_port,
                        viser_point_size=args.recon_viser_point_size,
                        viser_max_points=args.recon_viser_max_points,
                    )

                    memory_path = clip_memory_path
                    has_memory = True
                    if "alignment" in egosim_state_result:
                        clip_result["alignment"] = egosim_state_result["alignment"]
                    if "fusion" in egosim_state_result:
                        clip_result["fusion"] = egosim_state_result["fusion"]
                    if "cumulative_points" in egosim_state_result:
                        clip_result["cumulative_points"] = egosim_state_result["cumulative_points"]

                    logger.info(f"  egosim_state_subprocess completed: {egosim_state_result.get('status', 'unknown')}")

                except Exception as e:
                    logger.warning(f"  egosim_state_subprocess failed for {clip_stem}: {e}")
                    traceback.print_exc()

            clip_result["status"] = "success"

        except Exception as e:
            logger.error(f"  Failed processing {clip_stem}: {e}")
            traceback.print_exc()
            clip_result["status"] = "error"
            clip_result["error"] = str(e)

        summary["clips"].append(clip_result)

        if (
            getattr(args, "incremental_mode", "continue_generate") == "recon_visualize"
            and clip_idx == 0
            and clip_result.get("status") == "success"
        ):
            clip_result["incremental_mode"] = "recon_visualize"
            logger.info(
                "  incremental_mode=recon_visualize: stopping after clip 0 "
                "(generated video + EgoSim state + optional Viser). "
                "If Viser was enabled, the subprocess blocked until you stopped it. "
                "Turntable video generation is disabled in this mode."
            )
            break

    # Copy last memory as final_memory
    if memory_path and Path(memory_path).exists():
        import shutil
        final_path = group_output / "final_memory.npz"
        shutil.copy2(memory_path, final_path)
        logger.info(f"  Saved final memory: {final_path}")

    return summary, pipe


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental video generation with accumulated 3D memory"
    )
    parser.add_argument("--dataset", type=str, default="egodex",
                        choices=["egodex", "egovid", "agibot"],
                        help="Dataset name, matching egowm/inference/runner.py")
    parser.add_argument("--model_root", type=str, required=True,
                        help="Path to the standard EgoSim-14B model directory")
    parser.add_argument("--dataset_root", type=str,
                        default=str(Path(__file__).resolve().parents[4] / "tests" / "samples" /
                                    "mini_sample" / "continuous_generation"))
    parser.add_argument("--metadata_path", type=str,
                        default=str(Path(__file__).resolve().parents[4] / "tests" / "samples" /
                                    "mini_sample" / "continuous_generation" / "metadata.csv"))
    parser.add_argument("--eval_set_path", type=str, default=None,
                        help="Path to eval_set.txt for egovid, matching the standard runner")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--egowm_root", type=str, default=None,
                        help="Path to EgoSim repo root (default: ../ from continuous_simulation/)")
    parser.add_argument("--part_prefix", type=str, default="test_16fps_720p",
                        help="Subdirectory prefix for videos/HDF5 in dataset_root")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--use_long_prompt", action="store_true", default=True,
                        help="Use long_prompt.txt from process_result_dir instead of CSV prompt")
    parser.add_argument("--no_use_long_prompt", action="store_false", dest="use_long_prompt",
                        help="Disable long_prompt.txt, use CSV prompt instead")

    parser.add_argument("--max_groups", type=int, default=-1,
                        help="Limit number of video groups to process (-1 = all)")
    parser.add_argument("--only_multi_clip", action="store_true", default=False,
                        help="Only process groups with more than one clip")
    parser.add_argument(
        "--incremental_mode",
        type=str,
        default="continue_generate",
        choices=["continue_generate", "recon_visualize"],
        help="continue_generate: multi-clip incremental video generation (default). "
             "recon_visualize: clip 0 only — generate + EgoSim reconstruction + "
             "interactive Viser, then stop (no further clip generation).",
    )
    parser.add_argument(
        "--no_recon_viser",
        action="store_true",
        default=False,
        help="With recon_visualize: skip Viser.",
    )
    parser.add_argument(
        "--recon_viser_backend",
        type=str,
        choices=["egosim_state", "points"],
        default="egosim_state",
        help="egosim_state: full Viser UI via scene stack run_viser (default). points: minimal fused point cloud.",
    )
    parser.add_argument(
        "--recon_viser_host",
        type=str,
        default="auto",
        help="Viser bind (points backend only): auto = get_host_ip(); 0.0.0.0 = all interfaces",
    )
    parser.add_argument(
        "--recon_viser_port",
        type=int,
        default=20540,
        help="Viser port (run_viser default 20540)",
    )
    parser.add_argument(
        "--recon_viser_point_size",
        type=float,
        default=0.002,
        help="Point size (points backend only)",
    )
    parser.add_argument(
        "--recon_viser_max_points",
        type=int,
        default=800_000,
        help="Max points in Viser (random subsample; 0 = no limit)",
    )

    # EgoSim state reconstruction
    parser.add_argument("--run_egosim_state", action="store_true", default=True,
                        help="Run EgoSim state scene reconstruction on generated videos for memory")
    parser.add_argument("--no_egosim_state", action="store_true", default=False,
                        help="Disable EgoSim state reconstruction")
    parser.add_argument("--egosim_state_pipeline", type=str, default="dav3")
    parser.add_argument("--spatial_subsample", type=int, default=2)
    parser.add_argument("--temporal_subsample", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--fuse_overlap_radius", type=float, default=0.005)

    # EgoSim state / scene reconstruction options (viser-global naming in pipeline)
    parser.add_argument("--filter_interactive", action="store_true", default=True)
    parser.add_argument("--no_filter_interactive", action="store_false", dest="filter_interactive")
    parser.add_argument("--mask_dilation", type=int, default=5)
    parser.add_argument("--use_tsdf", action="store_true", default=True)
    parser.add_argument("--no_use_tsdf", action="store_false", dest="use_tsdf")
    parser.add_argument("--tsdf_voxel_size", type=float, default=0.0025)
    parser.add_argument("--tsdf_trunc_multiplier", type=float, default=20.0)
    parser.add_argument("--use_color_depth_overlap", action="store_true", default=False)
    parser.add_argument("--no_use_color_depth_overlap", action="store_false", dest="use_color_depth_overlap")
    parser.add_argument("--overlap_depth_thresh", type=float, default=0.05)
    parser.add_argument("--overlap_color_thresh", type=float, default=40.0)
    parser.add_argument("--use_last_frame_objects", action="store_true", default=True)
    parser.add_argument("--no_use_last_frame_objects", action="store_false", dest="use_last_frame_objects")
    parser.add_argument("--statistical_outlier_removal", action="store_true", default=True)
    parser.add_argument("--no_statistical_outlier_removal", action="store_false", dest="statistical_outlier_removal")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=20)
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0)


    # Phrase prediction
    parser.add_argument("--predict_phrases", action="store_true", default=False,
                        help="Use QwenVL to predict phrases for instance segmentation")
    parser.add_argument("--no_predict_phrases", action="store_true", default=False)
    parser.add_argument("--scene_phrases", nargs="*", default=list(DEFAULT_SCENE_PHRASES),
                        help="Manual scene phrases for instance segmentation when Qwen prediction is disabled")
    parser.add_argument("--qwen_model_path", type=str,
                        default="")

    # Rendering
    parser.add_argument("--render_point_size", type=float, default=3.0)

    # EgoSim-state-only: skip video generation & phrase prediction, use existing clips, only run reconstruction
    parser.add_argument("--egosim_state_only", action="store_true",
                        help="Skip generation & phrases; use existing clip videos, only run EgoSim state reconstruction")

    # Debug: rendering / alignment
    parser.add_argument("--render_gt_pointcloud", action="store_true",
                        help="Debug: render GT pointcloud instead of reconstruction")
    parser.add_argument("--opencv_to_opengl_points", action="store_true",
                        help="Debug: transform points [x,-y,-z] before render")
    parser.add_argument("--prefer_icp_alignment", action="store_true",
                        help="Use ICP instead of pose-based Sim3 when GT pointcloud available")
    parser.add_argument("--align_method", type=str, default="pose_full_sim3",
                        choices=["pose_sim3", "pose_full_sim3", "pose_full_sim3_scale",
                                 "pose_then_icp", "icp"],
                        help="Alignment: pose_then_icp (SO3 avg R + coarse-to-fine ICP for s+t, recommended), "
                             "pose_full_sim3_scale (SO3 avg R + center-based scale), "
                             "pose_full_sim3 (R from orientations, s=1.0), "
                             "pose_sim3 (Umeyama on centers only), icp (point cloud ICP)")
    parser.add_argument("--icp_refine", action="store_true", default=False,
                        help="After pose-based alignment, refine with ICP against GT point cloud")
    parser.add_argument("--no_icp_refine", action="store_false", dest="icp_refine")
    parser.add_argument("--icp_refine_iters", type=int, default=10)
    parser.add_argument("--icp_refine_corr_dist", type=float, default=0.05)
    parser.add_argument("--pose_center_sample_step", type=int, default=1,
                        help="Subsample step for pose-center correspondences in Sim3 alignment")
    # Env-switch: external Python interpreters for subprocess calls
    parser.add_argument("--egosim_state_python", type=str,
                        default=sys.executable,
                        help="Python interpreter for the scene-backend EgoSim state env")
    parser.add_argument("--da3_python", type=str,
                        default=sys.executable,
                        help="Python interpreter for the scene env used for QwenVL phrase prediction")
    parser.add_argument("--egosim_state_script", type=str,
                        default=str(Path(__file__).resolve().parent / "egosim_state_subprocess.py"),
                        help="Path to egosim_state_subprocess.py")
    parser.add_argument("--predict_phrases_script", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "vendor" / "predict_phrases.py"),
                        help="Path to predict_phrases.py")

    return parser.parse_args()


def main():
    args = parse_args()
    configure_standard_egowm_imports(args.egowm_root)

    if args.no_egosim_state:
        args.run_egosim_state = False
    if args.no_predict_phrases:
        args.predict_phrases = False
    elif args.qwen_model_path and not args.predict_phrases:
        args.predict_phrases = True
    if args.predict_phrases and has_custom_scene_phrases(args.scene_phrases):
        logger.info("Custom scene phrases detected; disabling predict_phrases and using manual phrases instead.")
        args.predict_phrases = False

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    logger.info(f"Device: {device}")

    # ---- Group clips ----
    logger.info(f"Loading metadata from {args.metadata_path}")
    groups = group_clips_by_source_video(args.metadata_path, args.part_prefix)
    logger.info(f"Total video groups: {len(groups)}")

    if args.only_multi_clip:
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        logger.info(f"Multi-clip groups only: {len(groups)}")

    if args.max_groups > 0:
        keys = sorted(groups.keys())[:args.max_groups]
        groups = {k: groups[k] for k in keys}
        logger.info(f"Limited to {len(groups)} groups")

    # ---- Load standard EgoWM pipeline (skip when egosim_state_only) ----
    pipe = None
    if not args.egosim_state_only:
        logger.info("Loading standard EgoWM pipeline ...")
        pipe = load_standard_pipeline(args.model_root, device)
    else:
        logger.info("egosim_state_only mode: skipping pipeline load")

    # ---- Validate subprocess scripts ----
    if args.run_egosim_state:
        if not Path(args.egosim_state_python).exists():
            logger.error(f"egosim_state_python not found: {args.egosim_state_python}")
            sys.exit(1)
        if not Path(args.egosim_state_script).exists():
            logger.error(f"egosim_state_script not found: {args.egosim_state_script}")
            sys.exit(1)
        logger.info(f"EgoSim state env: {args.egosim_state_python}")
        logger.info(f"EgoSim state script: {args.egosim_state_script}")

    if args.predict_phrases and args.run_egosim_state and not args.egosim_state_only:
        if not Path(args.da3_python).exists():
            logger.warning(f"da3_python not found: {args.da3_python}. Phrase prediction disabled.")
            args.predict_phrases = False
        elif not Path(args.predict_phrases_script).exists():
            logger.warning(f"predict_phrases_script not found: {args.predict_phrases_script}. Phrase prediction disabled.")
            args.predict_phrases = False
        else:
            logger.info(f"QwenVL env: {args.da3_python}")
            logger.info(f"Phrases script: {args.predict_phrases_script}")

    # ---- Process groups ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    sorted_groups = sorted(groups.items())

    for group_idx, (group_key, clips) in enumerate(tqdm(sorted_groups, desc="Video groups")):
        task_name, video_id = group_key
        logger.info(
            f"\n{'='*60}\n"
            f"[Group {group_idx+1}/{len(sorted_groups)}] {task_name}/{video_id} "
            f"({len(clips)} clips)\n"
            f"{'='*60}"
        )

        summary, pipe = process_clip_group(
            group_key=group_key,
            clips=clips,
            pipe=pipe,
            device=device,
            args=args,
        )
        all_summaries.append(summary)

        summary_path = output_dir / "incremental_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone. Output: {output_dir}")
    logger.info(f"Summary: {output_dir / 'incremental_summary.json'}")


if __name__ == "__main__":
    main()
