"""
EgoSim state subprocess for the incremental inference pipeline.

Runs in a dedicated conda env (scene / backend stack). Called by the
bundled incremental inference backend (wan env) via subprocess.

Operations:
  1. Run scene-pipeline inference on a generated video
  2. Reconstruct point cloud from pipeline artifacts
  3. Sim3-align to GT global frame
  4. Fuse with cumulative memory
  5. (Optional) Render updated memory for the next clip's camera trajectory

Input/output is via files (npz, json, mp4) and CLI args.

Usage:
    PYTHONPATH=src python -m egowm_incremental.backends.egosim_state_subprocess \
        --generated_video /path/to/generated.mp4 \
        --hdf5_path /path/to/clip.hdf5 \
        --process_result_dir /path/to/process_result/clip/ \
        --output_dir /path/to/egosim_state_artifacts/ \
        --output_memory /path/to/updated_memory.npz \
        [--cumulative_memory /path/to/current_memory.npz] \
        [--phrases_json /path/to/phrases.json] \
        [--next_hdf5_path /path/to/next_clip.hdf5] \
        [--rendered_video_out /path/to/rendered_for_next.mp4] \
        [--egosim_state_pipeline dav3] \
        [--gpu_id 0]
"""

import argparse
import json
import logging
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
import torch
from scipy.spatial import cKDTree

from egowm_incremental.phrases import DEFAULT_SCENE_PHRASES, normalize_scene_phrases

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("egosim_state_subprocess")

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 61

BODY_PART_WORDS = frozenset({
    "hand", "hands", "arm", "arms", "person", "finger", "fingers",
    "wrist", "thumb", "palm", "forearm", "elbow", "fist", "body",
})


def _is_body_part_phrase(phrase):
    """True when the phrase primarily describes a body part.

    Checks the first two words so that "left hand" or "right arm" match,
    but "silver pot in hand" does not (the leading noun is an object).
    """
    words = phrase.lower().replace("'s", "").split()
    if not words:
        return False
    return any(w in BODY_PART_WORDS for w in words[:2])


# ======================================================================
# Scene pipeline inference (backend)
# ======================================================================

def run_egosim_state_infer(
    video_path,
    output_dir,
    pipeline_name,
    phrases=None,
    *,
    save_viz=False,
    slam_visualize=False,
    force_reinfer=False,
):
    """Run scene reconstruction pipeline on a video, return ArtifactPath.
    When force_reinfer=True, bypass artifact existence check and re-run (e.g. to refresh SAM segmentation).
    """
    import time as _time

    logger.info("  Importing scene backend modules...")
    _t_import = _time.time()
    import hydra
    logger.info(f"  hydra imported in {_time.time() - _t_import:.1f}s")
    _t_import = _time.time()
    from egosim_state import get_config_path, make_pipeline
    logger.info(f"  scene backend core imported in {_time.time() - _t_import:.1f}s")
    _t_import = _time.time()
    from egosim_state.streams.base import ProcessedVideoStream
    from egosim_state.streams.raw_mp4_stream import RawMp4Stream
    from egosim_state.utils.io import ArtifactPath
    logger.info(f"  scene backend streams/io imported in {_time.time() - _t_import:.1f}s")

    artifact = ArtifactPath(output_dir, Path(video_path).stem)
    has_required_artifacts = (
        artifact.rgb_path.exists()
        and artifact.pose_path.exists()
        and artifact.depth_path.exists()
        and artifact.intrinsics_path.exists()
    )
    has_required_viz = (not save_viz) or artifact.meta_vis_path.exists()
    if has_required_artifacts and has_required_viz:
        logger.info(
            "Scene artifacts already exist but cache reuse is disabled; "
            f"re-running scene infer for {output_dir}"
        )
        shutil.rmtree(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={output_dir}",
        "pipeline.output.save_artifacts=true",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
        f"pipeline.slam.visualize={'true' if slam_visualize else 'false'}",
    ]
    if phrases:
        # Use JSON to safely escape any quotes/brackets in phrases.
        # Hydra override grammar accepts JSON-like list literals for simple types.
        try:
            phrases_list = normalize_scene_phrases(
                [str(p) for p in phrases if p is not None and str(p).strip()]
            )
        except Exception:
            phrases_list = list(DEFAULT_SCENE_PHRASES)
        if phrases_list:
            phrases_str = json.dumps(phrases_list, ensure_ascii=False)
            overrides.append(f"pipeline.init.instance.phrases={phrases_str}")

    logger.info("  Running hydra.initialize_config_dir + compose...")
    _t_hydra = _time.time()
    with hydra.initialize_config_dir(
        config_dir=str(get_config_path()), version_base=None
    ):
        args = hydra.compose("default", overrides=overrides)
    logger.info(f"  Hydra config ready in {_time.time() - _t_hydra:.1f}s")

    logger.info(f"Running EgoSim state scene infer for {video_path}")
    logger.info(f"  pipeline={pipeline_name}, overrides={overrides}")
    t0 = _time.time()
    logger.info("  Creating pipeline (make_pipeline)...")
    pipeline = make_pipeline(args.pipeline)
    logger.info(f"  Pipeline created in {_time.time() - t0:.1f}s")
    t1 = _time.time()
    logger.info("  Loading video stream...")
    video_stream = ProcessedVideoStream(
        RawMp4Stream(video_path), []
    ).cache(desc="Reading video stream")
    logger.info(f"  Video stream loaded in {_time.time() - t1:.1f}s")
    t2 = _time.time()
    logger.info("  Running pipeline.run()...")
    pipeline.run(video_stream)
    logger.info(f"  pipeline.run() finished in {_time.time() - t2:.1f}s")
    logger.info(f"  Total scene infer time: {_time.time() - t0:.1f}s")
    return artifact


# ======================================================================
# Point cloud reconstruction helpers (from incremental_reconstruction_new.py)
# ======================================================================

def _dilate_mask(mask, dilation):
    if dilation <= 0:
        return mask
    kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _safe_load_interactive_ids(artifact):
    interactive_path = artifact.mask_path.parent / "interactive.npz"
    if not interactive_path.exists():
        return {}
    try:
        data = dict(np.load(interactive_path, allow_pickle=True))
        return {int(k): {int(x) for x in v} - {0} for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to read interactive ids from {interactive_path}: {e}")
        return {}


def _safe_load_instance_phrases(artifact):
    from egosim_state.utils.io import read_instance_phrases
    if not artifact.mask_phrase_path.exists():
        return {}
    try:
        return read_instance_phrases(artifact.mask_phrase_path)
    except Exception as e:
        logger.warning(f"Failed to read instance phrases from {artifact.mask_phrase_path}: {e}")
        return {}


def _safe_load_depth_scales(artifact):
    from egosim_state.utils.io import read_depth_scale_artifacts
    if not artifact.depth_scale_path.exists():
        return {}
    try:
        return read_depth_scale_artifacts(artifact.depth_scale_path)
    except Exception as e:
        logger.warning(f"Failed to read depth scales from {artifact.depth_scale_path}: {e}")
        return {}


def _safe_load_interaction_states(artifact):
    from egosim_state.utils.io import read_interaction_states_artifacts
    if not artifact.interaction_states_path.exists():
        return {}
    try:
        return read_interaction_states_artifacts(artifact.interaction_states_path)
    except Exception as e:
        logger.warning(f"Failed to read interaction states from {artifact.interaction_states_path}: {e}")
        return {}


def _safe_load_keyframe_indices(artifact):
    if not artifact.slam_map_path.exists():
        return []
    try:
        slam_map = torch.load(artifact.slam_map_path, map_location="cpu")
        dense_inds = getattr(slam_map, "dense_disp_frame_inds", None)
        if dense_inds is None:
            return []
        return dense_inds.tolist()
    except Exception as e:
        logger.warning(f"Failed to read SLAM map from {artifact.slam_map_path}: {e}")
        return []


def _get_prev_keyframe(frame_idx, keyframes):
    if not keyframes:
        return 0
    prev_kfs = [kf for kf in keyframes if kf < frame_idx]
    return prev_kfs[-1] if prev_kfs else 0


def _check_hand_object_interaction(obj_mask, hand_mask, obj_depth, hand_depth,
                                   iou_threshold=0.1, depth_threshold=0.15):
    if obj_mask is None or hand_mask is None:
        return False

    intersection = np.logical_and(obj_mask, hand_mask).sum()
    union = np.logical_or(obj_mask, hand_mask).sum()
    if union == 0:
        return False
    iou = intersection / union
    if iou < iou_threshold:
        return False

    overlap_region = np.logical_and(obj_mask, hand_mask)
    if overlap_region.sum() == 0:
        return False

    obj_depths_overlap = obj_depth[overlap_region]
    hand_depths_overlap = hand_depth[overlap_region]
    valid_mask = (obj_depths_overlap > 0) & (hand_depths_overlap > 0)
    if valid_mask.sum() <= 10:
        return False

    depth_diff = np.abs(obj_depths_overlap[valid_mask] - hand_depths_overlap[valid_mask])
    return np.median(depth_diff) < depth_threshold


def _check_overlap_depth_color(
    points_world, prev_c2w, prev_camera_model, prev_depth_map,
    prev_rgb_map, curr_rgb_points, depth_threshold, rgb_threshold,
):
    R = prev_c2w[:3, :3].T
    t = -R @ prev_c2w[:3, 3]
    points_cam = (points_world @ R.T) + t[None]

    points_cam_homo = np.concatenate(
        [points_cam, np.ones((len(points_cam), 1), dtype=np.float32)], axis=-1
    )
    coords, _, _ = prev_camera_model.proj_points(
        torch.from_numpy(points_cam_homo).float(),
        compute_jp=False, compute_jf=False, limit_min_depth=False,
    )
    coords = coords.numpy()

    u, v = coords[:, 0], coords[:, 1]
    z_curr = points_cam[:, 2]
    H, W = prev_depth_map.shape
    valid_proj_mask = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z_curr > 0)

    is_overlapped = np.zeros(len(points_world), dtype=bool)
    if not valid_proj_mask.any():
        return is_overlapped

    u_int = np.clip(np.round(u[valid_proj_mask]).astype(int), 0, W - 1)
    v_int = np.clip(np.round(v[valid_proj_mask]).astype(int), 0, H - 1)

    d_prev = prev_depth_map[v_int, u_int]
    z_curr_valid = z_curr[valid_proj_mask]
    valid_depth_mask = d_prev > 1e-3
    abs_diff = np.abs(z_curr_valid - d_prev)

    prev_colors = prev_rgb_map[v_int, u_int].astype(np.float32)
    curr_colors_valid = curr_rgb_points[valid_proj_mask].astype(np.float32)
    color_diff = np.linalg.norm(prev_colors - curr_colors_valid, axis=1)

    overlap_indices = valid_depth_mask & (abs_diff < depth_threshold) & (color_diff <= rgb_threshold)
    temp_mask = np.zeros(np.sum(valid_proj_mask), dtype=bool)
    temp_mask[overlap_indices] = True
    is_overlapped[valid_proj_mask] = temp_mask
    return is_overlapped


def _remove_statistical_outliers(points, colors, nb_neighbors, std_ratio):
    if len(points) == 0:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    avg_dists = np.mean(dists[:, 1:], axis=1)
    mean_distance = float(np.mean(avg_dists))
    std_distance = float(np.std(avg_dists))
    threshold = mean_distance + std_ratio * std_distance
    inlier_mask = avg_dists < threshold
    return points[inlier_mask], colors[inlier_mask]


def _save_point_cloud_ply(points, colors, out_path):
    if len(points) == 0:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _to_uint8_colors(colors)
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]).encode("ascii")
    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    with open(out_path, "wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())


def _remove_small_clusters(points, colors, distance_thresh=0.05,
                           min_cluster_fraction=0.02, min_cluster_size=500):
    """Remove small disconnected point clusters (e.g. floating hand/arm fragments).

    Uses a voxel grid with 26-connected flood-fill, which is O(N) memory
    (no quadratic pair storage) and works on arbitrarily large point clouds.
    """
    from collections import deque

    if len(points) < min_cluster_size * 2:
        return points, colors

    voxel_size = distance_thresh
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)

    voxel_map = {}
    for i, vc in enumerate(voxel_coords):
        key = (int(vc[0]), int(vc[1]), int(vc[2]))
        voxel_map.setdefault(key, []).append(i)

    _OFFSETS = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    visited = set()
    keep_mask = np.zeros(len(points), dtype=bool)
    threshold = max(min_cluster_size, int(len(points) * min_cluster_fraction))
    n_clusters = 0
    n_kept = 0

    for seed in voxel_map:
        if seed in visited:
            continue
        n_clusters += 1
        component_indices = []
        queue = deque([seed])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            component_indices.extend(voxel_map[cur])
            cx, cy, cz = cur
            for dx, dy, dz in _OFFSETS:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in voxel_map and nb not in visited:
                    queue.append(nb)

        if len(component_indices) >= threshold:
            keep_mask[component_indices] = True
            n_kept += 1

    removed = int((~keep_mask).sum())
    if removed > 0:
        logger.info(
            f"  Small cluster removal: removed {removed}/{len(points)} points "
            f"({n_clusters} clusters found, {n_kept} kept, "
            f"threshold={threshold}, voxel_size={voxel_size})"
        )
    return points[keep_mask], colors[keep_mask]


def _integrate_frames_tsdf(frames_data, voxel_size, trunc_multiplier):
    if len(frames_data) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    try:
        import open3d as o3d
    except Exception as e:
        logger.warning(f"open3d not available, fallback to direct concatenation: {e}")
        pts = [f["points_world"] for f in frames_data if len(f["points_world"]) > 0]
        cols = [f["colors"] for f in frames_data if len(f["colors"]) > 0]
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(pts, axis=0).astype(np.float32), np.concatenate(cols, axis=0).astype(np.uint8)

    sdf_trunc = voxel_size * trunc_multiplier
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame_idx, frame in enumerate(frames_data):
        depth_array = frame["depth"].astype(np.float32).copy()
        depth_array[~frame["mask"].astype(bool)] = 0.0

        color_array = frame["color"]
        if color_array.dtype != np.uint8:
            color_array = color_array.astype(np.uint8)
        if not color_array.flags['C_CONTIGUOUS']:
            color_array = np.ascontiguousarray(color_array)

        color_o3d = o3d.geometry.Image(color_array)
        depth_o3d = o3d.geometry.Image(depth_array)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False,
        )

        fx, fy, cx, cy = frame["intrinsic"]
        h, w = frame["frame_height"], frame["frame_width"]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        c2w = frame["pose"]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = c2w[:3, :3].T
        w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]

        volume.integrate(rgbd, intrinsic_o3d, w2c)

    pcd_o3d = volume.extract_point_cloud()
    points = np.asarray(pcd_o3d.points).astype(np.float32)
    colors_normalized = np.asarray(pcd_o3d.colors)
    colors = (colors_normalized * 255).astype(np.uint8)

    if len(colors) > 0:
        logger.info(f"TSDF: mean_color={colors.mean():.2f}, points={len(points)}")

    return points, colors


# ======================================================================
# Point cloud reconstruction from pipeline artifacts (egosim_state scene options)
# ======================================================================
#
# Pipeline (matches incremental_reconstruction_new.py):
# 1. Reconstruct: pcd_cam = rays * depth, pcd_world = (c2w @ pcd_cam.T).T + c2w[:3,3]
#    - c2w from predicted poses -> pcd_world in local world frame
# 2. Sim3 align (pose_sim3): extract camera centers from pred & GT poses,
#    Umeyama -> (s, R, t), apply: points_gt = s * points_local @ R.T + t
# 3. Render: GT poses (OpenCV) -> c2w @ diag(1,-1,-1,1) for PyRender (OpenGL)
#


def reconstruct_from_artifacts(
    artifact,
    spatial_subsample=4,
    temporal_subsample=1,
    max_frames=-1,
    filter_interactive=False,
    mask_dilation=5,
    filter_body_parts=True,
    body_part_dilation=10,
    use_tsdf=False,
    tsdf_voxel_size=0.005,
    tsdf_trunc_multiplier=4.0,
    use_color_depth_overlap=False,
    overlap_depth_thresh=0.05,
    overlap_color_thresh=40.0,
    use_last_frame_objects=False,
    statistical_outlier_removal=False,
    outlier_nb_neighbors=20,
    outlier_std_ratio=2.0,
    remove_small_clusters=True,
    cluster_distance_thresh=0.05,
    min_cluster_fraction=0.02,
    min_cluster_size=500,
):
    """Reconstruct the canonical EgoSim State `visualize(All)` point cloud.

    This function intentionally follows the reference viewer's default scene
    reconstruction path before the incremental pipeline reorganizes the result
    into memory artifacts for later clips.
    """
    from egosim_state.utils.cameras import CameraType
    from egosim_state.utils.depth import reliable_depth_mask_range
    from egosim_state.utils.io import (
        read_depth_artifacts,
        read_instance_artifacts,
        read_intrinsics_artifacts,
        read_pose_artifacts,
        read_rgb_artifacts,
    )

    pose_seq = read_pose_artifacts(artifact.pose_path)[1].matrix().numpy()
    rgb_seq = read_rgb_artifacts(artifact.rgb_path)
    _, intr_seq, camera_types = read_intrinsics_artifacts(
        artifact.intrinsics_path, artifact.camera_type_path
    )
    depth_seq = read_depth_artifacts(artifact.depth_path)

    # Mirror the reference viewer: background is instance id 0 only, while the
    # `use_last_frame_objects` path tracks non-interactive objects and appends
    # their latest visible state into the global point cloud.
    interactive_ids_per_frame = _safe_load_interactive_ids(artifact) if (
        filter_interactive or use_last_frame_objects
    ) else {}
    instance_phrases = _safe_load_instance_phrases(artifact)
    depth_scales = _safe_load_depth_scales(artifact)
    interaction_states = _safe_load_interaction_states(artifact) if use_last_frame_objects else {}
    keyframe_indices = _safe_load_keyframe_indices(artifact) if use_last_frame_objects else []

    instance_iter = None
    if artifact.mask_path.exists():
        try:
            instance_iter = read_instance_artifacts(artifact.mask_path)
        except Exception:
            instance_iter = None

    global_pcd_points = []
    global_pcd_colors = []
    tsdf_frames = []
    object_tracking_data = {}
    all_instance_points_list = []

    prev_frame_info = None
    rays = None
    hand_related_phrases = ("hand", "arm")

    stats = {
        "interactive_filtered_points": 0,
        "body_part_filtered_points": 0,
        "raw_valid_points": 0,
        "overlap_dropped_points": 0,
        "tsdf_frames": 0,
        "cluster_removed_points": 0,
        "objects_added": 0,
    }

    for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth)) in enumerate(
        zip(pose_seq, rgb_seq, intr_seq, camera_types, depth_seq)
    ):
        if max_frames > 0 and frame_idx >= max_frames:
            break
        if frame_idx % temporal_subsample != 0:
            if instance_iter is not None:
                try:
                    next(instance_iter)
                except StopIteration:
                    pass
            continue

        instance_mask = None
        if instance_iter is not None:
            try:
                _, instance_mask = next(instance_iter)
                if instance_mask is not None:
                    instance_mask = instance_mask.numpy()
            except StopIteration:
                instance_mask = None

        frame_h, frame_w = rgb.shape[:2]
        camera_model = camera_type.build_camera_model(intr)

        if rays is None:
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_h).float()[::spatial_subsample],
                torch.arange(frame_w).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_h - 1)
                disp_u = disp_u / (frame_w - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        depth_np = depth.numpy()[::spatial_subsample, ::spatial_subsample]
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)[::spatial_subsample, ::spatial_subsample]
        depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
        depth_mask_original = depth_mask.copy()
        pcd_cam = rays * depth_np[..., None]

        if filter_interactive and instance_mask is not None:
            inst_sub = instance_mask[::spatial_subsample, ::spatial_subsample]
            interactive_ids = {int(x) for x in interactive_ids_per_frame.get(frame_idx, set())} - {0}
            for uid in np.unique(inst_sub):
                uid = int(uid)
                if uid == 0 or uid in interactive_ids:
                    continue
                phrase = instance_phrases.get(uid, "").lower()
                is_human = any(kw in phrase for kw in hand_related_phrases)
                is_in_hand = "in hand" in phrase
                if is_human and not is_in_hand:
                    interactive_ids.add(uid)
            if interactive_ids:
                interactive_mask = np.isin(inst_sub, list(interactive_ids))
                interactive_mask = _dilate_mask(interactive_mask, mask_dilation)
                before = int(depth_mask.sum())
                depth_mask = depth_mask & (~interactive_mask)
                stats["interactive_filtered_points"] += before - int(depth_mask.sum())

        stats["raw_valid_points"] += int(depth_mask.sum())

        if instance_mask is not None:
            inst_sub = instance_mask[::spatial_subsample, ::spatial_subsample]
            unique_obj_ids = np.unique(inst_sub)
            frame_interactive_ids = {int(x) for x in interactive_ids_per_frame.get(frame_idx, set())}

            bg_mask = inst_sub == 0
            valid_bg_mask = depth_mask & bg_mask
            if int(valid_bg_mask.sum()) > 0:
                if use_tsdf:
                    scaled_camera_model = camera_model.scaled(1.0 / spatial_subsample)
                    tsdf_frames.append({
                        "depth": depth_np,
                        "color": rgb_np,
                        "mask": valid_bg_mask.astype(np.uint8),
                        "intrinsic": scaled_camera_model.pinhole().intrinsics.cpu().numpy(),
                        "pose": c2w,
                        "frame_height": rgb_np.shape[0],
                        "frame_width": rgb_np.shape[1],
                    })
                else:
                    bg_pcd_flat = pcd_cam[valid_bg_mask]
                    bg_rgb_flat = rgb_np[valid_bg_mask]
                    bg_pcd_world = (c2w[:3, :3] @ bg_pcd_flat.T).T + c2w[:3, 3]
                    if use_color_depth_overlap and prev_frame_info is not None:
                        overlap_mask = _check_overlap_depth_color(
                            bg_pcd_world,
                            prev_frame_info["c2w"],
                            prev_frame_info["camera_model"],
                            prev_frame_info["depth"],
                            prev_frame_info["rgb"],
                            bg_rgb_flat,
                            overlap_depth_thresh,
                            overlap_color_thresh,
                        )
                        stats["overlap_dropped_points"] += int(overlap_mask.sum())
                        keep_mask = ~overlap_mask
                        bg_pcd_world = bg_pcd_world[keep_mask]
                        bg_rgb_flat = bg_rgb_flat[keep_mask]
                    global_pcd_points.append(bg_pcd_world.astype(np.float32))
                    global_pcd_colors.append(bg_rgb_flat.astype(np.uint8))

            if use_last_frame_objects:
                hand_masks = {}
                hand_depths = {}
                for obj_id in unique_obj_ids:
                    obj_id_int = int(obj_id)
                    if obj_id_int in frame_interactive_ids:
                        obj_mask = inst_sub == obj_id_int
                        hand_masks[obj_id_int] = obj_mask
                        hand_depths[obj_id_int] = depth_np * obj_mask

                for obj_id in unique_obj_ids:
                    obj_id_int = int(obj_id)
                    if obj_id_int == 0 or obj_id_int in frame_interactive_ids:
                        continue

                    phrase = instance_phrases.get(obj_id_int, "").lower()
                    is_human_body = any(kw in phrase for kw in hand_related_phrases)
                    is_in_hand = "in hand" in phrase
                    if is_human_body and not is_in_hand:
                        continue

                    obj_mask = inst_sub == obj_id_int
                    valid_obj_mask = depth_mask & obj_mask
                    if int(valid_obj_mask.sum()) == 0:
                        continue

                    if obj_id_int not in object_tracking_data:
                        object_tracking_data[obj_id_int] = {
                            "frames": [],
                            "frame_data": [],
                            "last_frame": frame_idx,
                            "prev_keyframe": _get_prev_keyframe(frame_idx, keyframe_indices),
                            "interacted": False,
                            "disappeared_after_interaction": False,
                        }

                    object_tracking_data[obj_id_int]["last_frame"] = frame_idx

                    is_interacting = False
                    if interaction_states and frame_idx in interaction_states:
                        obj_state = interaction_states[frame_idx].get(obj_id_int, {})
                        is_interacting = bool(obj_state.get("is_interacting", False))
                        if is_interacting:
                            object_tracking_data[obj_id_int]["interacted"] = True
                    else:
                        for hand_id, hand_mask in hand_masks.items():
                            hand_depth = hand_depths[hand_id]
                            obj_depth_full = depth_np
                            if _check_hand_object_interaction(obj_mask, hand_mask, obj_depth_full, hand_depth):
                                is_interacting = True
                                object_tracking_data[obj_id_int]["interacted"] = True
                                break

                    obj_pcd_flat = pcd_cam[valid_obj_mask]
                    obj_rgb_flat = rgb_np[valid_obj_mask]
                    obj_depth_flat = depth_np[valid_obj_mask]
                    frame_scale = depth_scales.get(frame_idx, 1.0)
                    scaled_camera_model = camera_model.scaled(1.0 / spatial_subsample)

                    object_tracking_data[obj_id_int]["frames"].append(frame_idx)
                    object_tracking_data[obj_id_int]["frame_data"].append({
                        "pcd_camera": (obj_pcd_flat * frame_scale).astype(np.float32),
                        "colors": obj_rgb_flat.astype(np.uint8),
                        "depth": (obj_depth_flat * frame_scale).astype(np.float32),
                        "mask": valid_obj_mask.astype(np.uint8),
                        "c2w": c2w,
                        "intrinsic": scaled_camera_model.pinhole().intrinsics.cpu().numpy(),
                        "frame_idx": frame_idx,
                        "is_interacting": is_interacting,
                        "depth_scale": frame_scale,
                    })

            for obj_id in unique_obj_ids:
                obj_id_int = int(obj_id)
                if obj_id_int == 0:
                    continue
                phrase = instance_phrases.get(obj_id_int, "").lower()
                if filter_body_parts and _is_body_part_phrase(phrase):
                    continue
                obj_mask_all = depth_mask_original & (inst_sub == obj_id_int)
                if obj_mask_all.sum() == 0:
                    continue
                obj_cam_all = pcd_cam[obj_mask_all]
                obj_world_all = (c2w[:3, :3] @ obj_cam_all.T).T + c2w[:3, 3]
                all_instance_points_list.append(obj_world_all.astype(np.float32))
        else:
            pcd_flat = pcd_cam[depth_mask]
            rgb_flat = rgb_np[depth_mask]
            if len(pcd_flat) > 0:
                pcd_world = (c2w[:3, :3] @ pcd_flat.T).T + c2w[:3, 3]
                global_pcd_points.append(pcd_world.astype(np.float32))
                global_pcd_colors.append(rgb_flat.astype(np.uint8))

        prev_frame_info = {
            "c2w": c2w,
            "camera_model": camera_model.scaled(1.0 / spatial_subsample),
            "depth": depth_np,
            "rgb": rgb_np,
        }

    if use_tsdf and tsdf_frames:
        points, colors = _integrate_frames_tsdf(
            tsdf_frames, voxel_size=tsdf_voxel_size, trunc_multiplier=tsdf_trunc_multiplier,
        )
        stats["tsdf_frames"] = len(tsdf_frames)
    else:
        if global_pcd_points:
            points = np.concatenate(global_pcd_points, axis=0)
            colors = np.concatenate(global_pcd_colors, axis=0)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

    background_points = points.astype(np.float32)
    background_colors = colors.astype(np.uint8)

    last_frame_points = np.zeros((0, 3), dtype=np.float32)
    last_frame_colors = np.zeros((0, 3), dtype=np.uint8)

    if use_last_frame_objects and object_tracking_data:
        obj_points_list = []
        obj_colors_list = []

        for obj_id, obj_data in object_tracking_data.items():
            frames = obj_data["frames"]
            frame_data_list = obj_data["frame_data"]
            last_frame = obj_data["last_frame"]
            if not frame_data_list:
                continue

            if obj_data["interacted"]:
                last_interaction_frame = -1
                for fd in reversed(frame_data_list):
                    if fd["is_interacting"]:
                        last_interaction_frame = fd["frame_idx"]
                        break
                frames_after_interaction = [f for f in frames if f > last_interaction_frame]
                if len(frames_after_interaction) < 2:
                    object_tracking_data[obj_id]["disappeared_after_interaction"] = True
                    continue

            max_frames_for_fusion = 1
            start_frame = max(0, last_frame - max_frames_for_fusion + 1)
            selected_frame_data = [
                fd for fd in frame_data_list
                if start_frame <= fd["frame_idx"] <= last_frame
            ]
            if not selected_frame_data:
                continue

            if len(selected_frame_data) > 1 and use_tsdf:
                obj_tsdf_data = []
                for fd in selected_frame_data:
                    h, w = fd["mask"].shape
                    obj_depth_map = np.zeros((h, w), dtype=np.float32)
                    obj_depth_map[fd["mask"].astype(bool)] = fd["depth"]
                    color_map = np.zeros((h, w, 3), dtype=np.uint8)
                    valid_mask = fd["mask"].astype(bool)
                    if valid_mask.sum() == len(fd["colors"]):
                        color_map[valid_mask] = fd["colors"]
                    obj_tsdf_data.append({
                        "depth": obj_depth_map,
                        "color": color_map,
                        "mask": fd["mask"],
                        "intrinsic": fd["intrinsic"],
                        "pose": fd["c2w"],
                        "frame_height": h,
                        "frame_width": w,
                    })
                try:
                    obj_fused_points, obj_fused_colors = _integrate_frames_tsdf(
                        obj_tsdf_data,
                        voxel_size=tsdf_voxel_size,
                        trunc_multiplier=tsdf_trunc_multiplier,
                    )
                    if len(obj_fused_points) > 0:
                        obj_points_list.append(obj_fused_points.astype(np.float32))
                        obj_colors_list.append(obj_fused_colors.astype(np.uint8))
                        continue
                except Exception as e:
                    logger.warning(f"Object TSDF fusion failed for {obj_id}: {e}")

            last_fd = selected_frame_data[-1]
            last_pcd_world = (last_fd["c2w"][:3, :3] @ last_fd["pcd_camera"].T).T + last_fd["c2w"][:3, 3]
            obj_points_list.append(last_pcd_world.astype(np.float32))
            obj_colors_list.append(last_fd["colors"].astype(np.uint8))

        if obj_points_list:
            obj_points_concat = np.concatenate(obj_points_list, axis=0)
            obj_colors_concat = np.concatenate(obj_colors_list, axis=0)
            if len(background_points) > 0:
                background_points = np.concatenate([background_points, obj_points_concat], axis=0)
                background_colors = np.concatenate([background_colors, obj_colors_concat], axis=0)
            else:
                background_points = obj_points_concat
                background_colors = obj_colors_concat
            stats["objects_added"] = len(obj_points_list)

    if statistical_outlier_removal and len(background_points) > 0:
        background_points, background_colors = _remove_statistical_outliers(
            background_points, background_colors, outlier_nb_neighbors, outlier_std_ratio
        )

    if remove_small_clusters and len(background_points) > 0:
        before_cluster = len(background_points)
        background_points, background_colors = _remove_small_clusters(
            background_points, background_colors,
            distance_thresh=cluster_distance_thresh,
            min_cluster_fraction=min_cluster_fraction,
            min_cluster_size=min_cluster_size,
        )
        stats["cluster_removed_points"] = before_cluster - len(background_points)

    all_instance_points = np.zeros((0, 3), dtype=np.float32)
    if all_instance_points_list:
        all_instance_points = np.concatenate(all_instance_points_list, axis=0).astype(np.float32)
    logger.info(f"  All-instance points collected: {len(all_instance_points)}")

    return (
        background_points,
        background_colors,
        last_frame_points,
        last_frame_colors,
        all_instance_points,
        pose_seq.astype(np.float32),
        stats,
    )


# ======================================================================
# GT data loading
# ======================================================================

def _to_uint8_colors(colors):
    if colors.size == 0:
        return colors.astype(np.uint8)
    if colors.dtype == np.uint8:
        return colors
    c = colors.astype(np.float32)
    if c.max() <= 1.0:
        c = c * 255.0
    return np.clip(c, 0, 255).astype(np.uint8)


def load_gt_pointcloud(process_result_dir, hdf5_path):
    gt_path = Path(process_result_dir) / "pointcloud.npz"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT point cloud: {gt_path}")
    data = np.load(gt_path)
    points_cam = data["points"].astype(np.float32)
    colors = _to_uint8_colors(data["colors"])
    intrinsics = data["intrinsics"]
    original_size = data["original_size"]

    with h5py.File(hdf5_path, "r") as f:
        c2w_first = np.asarray(f["transforms"]["camera"][0]).astype(np.float32)

    R = c2w_first[:3, :3]
    t = c2w_first[:3, 3]
    points_world = (R @ points_cam.T).T + t[None]

    return points_world.astype(np.float32), colors, intrinsics, original_size


def load_intrinsics_from_hdf5(hdf5_path):
    """Load camera intrinsics from EgoDex-style HDF5 when pointcloud.npz is absent."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        return None, None
    with h5py.File(hdf5_path, "r") as f:
        if "camera" not in f or "intrinsic" not in f["camera"]:
            return None, None
        intrinsics = np.asarray(f["camera"]["intrinsic"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        return None, None
    original_size = np.array([1280, 720], dtype=np.int32)
    return intrinsics, original_size


def load_intrinsics_from_scene_artifact(artifact):
    """Load pinhole intrinsics from EgoSim state scene artifacts."""
    if artifact is None or not artifact.intrinsics_path.exists():
        return None, None
    from egosim_state.utils.io import read_intrinsics_artifacts

    _, intr_seq, _ = read_intrinsics_artifacts(
        artifact.intrinsics_path, artifact.camera_type_path
    )
    fx, fy, cx, cy = intr_seq[0].numpy()
    intrinsics = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    if not artifact.rgb_path.exists():
        original_size = np.array([WIDTH, HEIGHT], dtype=np.int32)
        return intrinsics, original_size
    frame = imageio.imread(str(artifact.rgb_path), index=0)
    original_size = np.array([frame.shape[1], frame.shape[0]], dtype=np.int32)
    return intrinsics, original_size


def resolve_render_intrinsics(
    *,
    gt_intrinsics,
    gt_original_size,
    hdf5_path,
    process_result_dir,
    scene_artifact=None,
):
    if gt_intrinsics is not None:
        return gt_intrinsics, gt_original_size

    if process_result_dir and hdf5_path:
        try:
            _, _, intrinsics, original_size = load_gt_pointcloud(process_result_dir, hdf5_path)
            return intrinsics, original_size
        except (FileNotFoundError, Exception):
            pass

    intrinsics, original_size = load_intrinsics_from_hdf5(hdf5_path)
    if intrinsics is not None:
        return intrinsics, original_size

    return load_intrinsics_from_scene_artifact(scene_artifact)


def load_gt_camera_transforms(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        poses = np.asarray(f["transforms"]["camera"])
    return poses.astype(np.float32)


def _hdf5_start_frame(hdf5_path):
    """Extract the start-frame index from an HDF5 filename like ``2_121_241.hdf5``."""
    stem = Path(hdf5_path).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def _load_gt_cameras_range(hdf5_path, clip_start, clip_end):
    """Load GT camera transforms for global frames [clip_start, clip_end] inclusive,
    stitching from multiple HDF5 files in the same directory if needed.

    HDF5 files are named ``<video_id>_<start>_<end>.hdf5`` with 121 poses each.
    Clips at HDF5 boundaries (e.g. clip 120-180 with HDF5s 0_120 and 121_241)
    need poses from two files.
    """
    hdf5_path = Path(hdf5_path)
    hdf5_dir = hdf5_path.parent
    video_id = hdf5_path.stem.split("_")[0]
    seg_len = clip_end - clip_start + 1

    hdf5_files = []
    for f in hdf5_dir.iterdir():
        if not f.name.endswith(".hdf5"):
            continue
        parts = f.stem.split("_")
        if len(parts) < 3 or parts[0] != video_id:
            continue
        try:
            h_start, h_end = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if h_end < clip_start or h_start > clip_end:
            continue
        hdf5_files.append((h_start, h_end, f))
    hdf5_files.sort()

    if not hdf5_files:
        return load_gt_camera_transforms(hdf5_path)

    result = []
    frame = clip_start
    for h_start, h_end, h_path in hdf5_files:
        if frame > clip_end:
            break
        if h_end < frame:
            continue
        h_poses = load_gt_camera_transforms(h_path)
        n_poses = len(h_poses)
        while frame <= min(clip_end, h_end):
            idx = frame - h_start
            if 0 <= idx < n_poses:
                result.append(h_poses[idx])
            frame += 1

    if len(result) == 0:
        return load_gt_camera_transforms(hdf5_path)

    return np.array(result, dtype=np.float32)


# ======================================================================
# Sim3 alignment
# ======================================================================

def _apply_sim3(points, scale, R, t):
    return (scale * (points @ R.T) + t[None]).astype(np.float32)


def _scaled_transform_to_components(transform):
    rot = transform.rotation.matrix()
    if rot.ndim == 3:
        rot = rot[0]
    R = rot.detach().cpu().numpy()
    if R.shape == (4, 4):
        R = R[:3, :3]
    t_vec = transform.translation
    if t_vec.ndim == 2:
        t_vec = t_vec[0]
    return float(transform.scale), R, t_vec.detach().cpu().numpy()


def estimate_sim3_from_pose_centers(pred_poses, gt_poses, sample_step=1):
    """Estimate Sim3 (DA3-local -> GT-global) from corresponding camera centers.
    Both pred and GT poses are assumed to be c2w. Returns (s, R, t) such that
    points_gt = s * points_local @ R.T + t."""
    from egosim_state.utils.geometry import align_points

    n = min(len(pred_poses), len(gt_poses))
    if n < 3:
        raise ValueError(f"Not enough frames for alignment: {n}")

    pred_centers = pred_poses[:n, :3, 3].astype(np.float32)
    gt_centers = gt_poses[:n, :3, 3].astype(np.float32)
    if sample_step > 1:
        pred_centers = pred_centers[::sample_step]
        gt_centers = gt_centers[::sample_step]

    m = len(pred_centers)
    sim3 = align_points(
        torch.from_numpy(pred_centers), torch.from_numpy(gt_centers), scale=True
    )
    s, R, t = _scaled_transform_to_components(sim3)
    aligned = _apply_sim3(pred_centers, s, R, t)
    rmse = float(np.sqrt(np.mean((aligned - gt_centers) ** 2)))
    return s, R, t, {
        "pose_center_rmse": rmse,
        "pose_center_pairs": int(m),
        "pose_center_sample_step": int(sample_step),
    }



def estimate_sim3_from_full_poses(pred_poses, gt_poses, sample_step=1):
    """Estimate rigid transform (R, t) from camera poses. Scale fixed to 1.0.

    Step 1: R from camera orientations via SO3 averaging.
    Step 2: t from camera center means (s=1.0).
    """
    n = min(len(pred_poses), len(gt_poses))
    if n < 2:
        raise ValueError(f"Not enough frames for alignment: {n}")

    pred = pred_poses[:n:sample_step].astype(np.float64)
    gt = gt_poses[:n:sample_step].astype(np.float64)
    m = len(pred)

    # Step 1: R from SO3 averaging
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for i in range(m):
        R_sum += gt[i, :3, :3] @ pred[i, :3, :3].T
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = (U @ D @ Vt).astype(np.float32)

    # Step 2: translation (scale = 1.0)
    s = 1.0
    pred_centers = pred[:, :3, 3].astype(np.float32)
    gt_centers = gt[:, :3, 3].astype(np.float32)
    rotated_pred = pred_centers @ R.T

    rp_mean = rotated_pred.mean(axis=0)
    gc_mean = gt_centers.mean(axis=0)
    t = (gc_mean - rp_mean).astype(np.float32)

    aligned_centers = _apply_sim3(pred_centers, s, R, t)
    rmse = float(np.sqrt(np.mean((aligned_centers - gt_centers) ** 2)))

    orient_errors = []
    for i in range(m):
        R_est = R @ pred[i, :3, :3].astype(np.float32)
        R_gt = gt[i, :3, :3].astype(np.float32)
        cos_angle = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
        orient_errors.append(float(np.degrees(np.arccos(cos_angle))))
    mean_orient_err = float(np.mean(orient_errors))

    logger.info(f"  Rigid transform: scale=1.0 (fixed), center_rmse={rmse:.4f}, "
                f"mean_orient_err={mean_orient_err:.2f}deg, pairs={m}")

    return s, R, t, {
        "pose_full_rmse": rmse,
        "pose_full_mean_orient_err_deg": mean_orient_err,
        "pose_full_pairs": int(m),
    }


def estimate_sim3_from_full_poses_with_scale(pred_poses, gt_poses, sample_step=1):
    """Estimate full Sim3 (s, R, t) by combining SO3 averaging with scale estimation.

    Better than pose_sim3 (Umeyama on centers: R inaccurate for near-collinear
    ego trajectories) and pose_full_sim3 (SO3 avg but s fixed to 1.0: wrong when
    backend SLAM has its own metric scale).

    Step 1: R from SO3 averaging over all frame rotation pairs — robust because
            it uses full 3x3 rotation matrices, not just center positions.
    Step 2: s from camera centers after applying R — Umeyama-style optimal scale.
    Step 3: t from mean centers.
    """
    n = min(len(pred_poses), len(gt_poses))
    if n < 3:
        raise ValueError(f"Not enough frames for alignment: {n}")

    pred = pred_poses[:n:sample_step].astype(np.float64)
    gt = gt_poses[:n:sample_step].astype(np.float64)
    m = len(pred)

    # Step 1: R from SO3 averaging  —  R_align = argmin Σ ||R_align @ R_pred_i − R_gt_i||²
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for i in range(m):
        R_sum += gt[i, :3, :3] @ pred[i, :3, :3].T
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = (U @ D @ Vt).astype(np.float32)

    # Step 2: s from camera centers (optimal scale given R)
    pred_centers = pred[:, :3, 3].astype(np.float32)
    gt_centers = gt[:, :3, 3].astype(np.float32)
    rotated_pred = pred_centers @ R.T

    rp_mean = rotated_pred.mean(axis=0)
    gc_mean = gt_centers.mean(axis=0)
    rp_centered = rotated_pred - rp_mean
    gc_centered = gt_centers - gc_mean

    denom = float(np.sum(rp_centered * rp_centered))
    if denom < 1e-12:
        s = 1.0
        logger.warning("  Camera centers nearly stationary, defaulting scale to 1.0")
    else:
        s = float(np.sum(gc_centered * rp_centered) / denom)
    if s <= 0 or s > 100:
        logger.warning(f"  Computed scale {s:.4f} out of range, clamping to 1.0")
        s = 1.0

    # Step 3: translation
    t = (gc_mean - s * rp_mean).astype(np.float32)

    # Quality metrics
    aligned_centers = _apply_sim3(pred_centers, s, R, t)
    rmse = float(np.sqrt(np.mean((aligned_centers - gt_centers) ** 2)))

    orient_errors = []
    for i in range(m):
        R_est = R @ pred[i, :3, :3].astype(np.float32)
        R_gt = gt[i, :3, :3].astype(np.float32)
        cos_angle = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
        orient_errors.append(float(np.degrees(np.arccos(cos_angle))))
    mean_orient_err = float(np.mean(orient_errors))

    logger.info(f"  Full Sim3 (SO3 avg + scale): scale={s:.6f}, center_rmse={rmse:.4f}, "
                f"mean_orient_err={mean_orient_err:.2f}deg, pairs={m}")

    return s, R, t, {
        "pose_full_scale_rmse": rmse,
        "pose_full_scale_mean_orient_err_deg": mean_orient_err,
        "pose_full_scale_pairs": int(m),
        "computed_scale": float(s),
    }


def estimate_sim3_pose_then_icp(
    pred_poses, gt_poses, source_points, target_points,
    sample_step=1, max_iters=30, initial_corr_distance=0.5,
    final_corr_distance=0.02, sample_size=50000,
):
    """Robust Sim3: pose-based initialization + coarse-to-fine ICP refinement.

    When the camera barely moves (small ego motion), scale estimation from
    camera centers alone is unreliable — the center-derived scale may differ
    greatly from the actual depth scale, leaving the point clouds far apart.

    Approach:
      1. R from SO3 averaging (robust rotation from full pose matrices).
      2. Initial s and t from camera centers (Umeyama-style, same as
         pose_full_sim3_scale) — this gets the clouds *roughly* overlapping.
      3. Coarse-to-fine ICP on the pre-aligned cloud vs GT to refine s and t
         using actual 3D geometry, fixing any depth-scale mismatch.
    """
    from egosim_state.utils.geometry import align_points

    n = min(len(pred_poses), len(gt_poses))
    if n < 2:
        raise ValueError(f"Not enough frames for alignment: {n}")

    # Step 1: R from SO3 averaging
    pred = pred_poses[:n:sample_step].astype(np.float64)
    gt = gt_poses[:n:sample_step].astype(np.float64)
    m = len(pred)

    R_sum = np.zeros((3, 3), dtype=np.float64)
    for i in range(m):
        R_sum += gt[i, :3, :3] @ pred[i, :3, :3].T
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R_pose = (U @ D @ Vt).astype(np.float32)

    orient_errors = []
    for i in range(m):
        R_est = R_pose @ pred[i, :3, :3].astype(np.float32)
        R_gt = gt[i, :3, :3].astype(np.float32)
        cos_angle = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
        orient_errors.append(float(np.degrees(np.arccos(cos_angle))))
    mean_orient_err = float(np.mean(orient_errors))
    logger.info(f"  SO3 avg: mean_orient_err={mean_orient_err:.2f}deg, pairs={m}")

    # Step 2: Initial s and t from camera centers (brings clouds into rough overlap)
    pred_centers = pred[:, :3, 3].astype(np.float32)
    gt_centers = gt[:, :3, 3].astype(np.float32)
    rotated_pred = pred_centers @ R_pose.T

    rp_mean = rotated_pred.mean(axis=0)
    gc_mean = gt_centers.mean(axis=0)
    rp_centered = rotated_pred - rp_mean
    gc_centered = gt_centers - gc_mean

    denom = float(np.sum(rp_centered * rp_centered))
    if denom < 1e-12:
        s_init = 1.0
    else:
        s_init = float(np.sum(gc_centered * rp_centered) / denom)
    if s_init <= 0 or s_init > 100:
        s_init = 1.0
    t_init = (gc_mean - s_init * rp_mean).astype(np.float32)

    init_aligned = _apply_sim3(pred_centers, s_init, R_pose, t_init)
    init_center_rmse = float(np.sqrt(np.mean((init_aligned - gt_centers) ** 2)))
    logger.info(f"  Pose-based init: scale={s_init:.6f}, center_rmse={init_center_rmse:.4f}")

    if len(source_points) == 0 or len(target_points) == 0:
        logger.warning("  No point clouds for ICP, using pose-based init only")
        return s_init, R_pose, t_init, {
            "method": "pose_then_icp_fallback",
            "mean_orient_err_deg": mean_orient_err,
            "init_scale": float(s_init),
            "center_rmse": init_center_rmse,
        }

    # Step 3: Iterative median translation correction.
    # R and s are locked (from poses); only a translation offset is estimated
    # from point cloud geometry. This is safe even when GT is a partial view,
    # because median is robust to the many reconstructed points without GT counterparts.
    pre_aligned_src = _apply_sim3(source_points, s_init, R_pose, t_init)

    rng = np.random.default_rng(123)
    n_src = min(sample_size, len(pre_aligned_src))
    n_tgt = min(sample_size, len(target_points))
    src_sample = pre_aligned_src[rng.choice(len(pre_aligned_src), n_src, replace=False)].astype(np.float32)
    tgt_sample = target_points[rng.choice(len(target_points), n_tgt, replace=False)].astype(np.float32)
    tgt_tree = cKDTree(tgt_sample)

    t_correction = np.zeros(3, dtype=np.float32)
    final_inliers = 0
    final_rmse = float("inf")

    for it in range(max_iters):
        frac = it / max(max_iters - 1, 1)
        corr_dist = initial_corr_distance * (1 - frac) + final_corr_distance * frac

        src_shifted = src_sample + t_correction[None]
        dists, nn_idx = tgt_tree.query(src_shifted, k=1)
        inlier_mask = dists < corr_dist
        n_inliers = int(inlier_mask.sum())
        if n_inliers < 50:
            if it == 0:
                logger.warning(f"  Translation correction iter {it}: only {n_inliers} inliers "
                               f"at corr_dist={corr_dist:.3f}")
            break

        residuals = tgt_sample[nn_idx[inlier_mask]] - src_shifted[inlier_mask]
        final_inliers = n_inliers
        final_rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        delta_t = np.median(residuals, axis=0).astype(np.float32)
        t_correction = t_correction + delta_t

        if it % 5 == 0 or it == max_iters - 1:
            logger.info(f"  Translation iter {it}: inliers={n_inliers}, rmse={final_rmse:.4f}, "
                        f"corr_dist={corr_dist:.3f}, t_corr={t_correction}")

    t_total = (t_init + t_correction).astype(np.float32)

    aligned_centers = _apply_sim3(pred_centers, s_init, R_pose, t_total)
    center_rmse = float(np.sqrt(np.mean((aligned_centers - gt_centers) ** 2)))

    logger.info(f"  pose_then_icp final: scale={s_init:.6f}, inliers={final_inliers}, "
                f"rmse={final_rmse:.6f}, center_rmse={center_rmse:.4f}, "
                f"t_correction={t_correction}, init_center_rmse={init_center_rmse:.4f}")

    return s_init, R_pose, t_total, {
        "inliers": final_inliers,
        "rmse": final_rmse,
        "scale": float(s_init),
        "t_correction": t_correction.tolist(),
        "center_rmse": center_rmse,
        "init_center_rmse": init_center_rmse,
        "mean_orient_err_deg": mean_orient_err,
        "pose_pairs": int(m),
    }


def _compose_sim3(cs, cR, ct, ds, dR, dt):
    return cs * ds, dR @ cR, ds * (ct @ dR.T) + dt


def estimate_sim3_icp(source_points, target_points, max_iters=20,
                       corr_distance=0.08, sample_size=60000):
    from egosim_state.utils.geometry import align_points

    if len(source_points) == 0 or len(target_points) == 0:
        raise ValueError("Source/target point clouds must be non-empty.")

    rng = np.random.default_rng(123)
    src_sample = source_points[rng.choice(len(source_points), min(sample_size, len(source_points)), replace=False)].astype(np.float32)
    tgt_sample = target_points[rng.choice(len(target_points), min(sample_size, len(target_points)), replace=False)].astype(np.float32)
    tgt_tree = cKDTree(tgt_sample)

    s, R, t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    final_inliers = 0
    final_rmse = float("inf")

    for _ in range(max_iters):
        src_trans = _apply_sim3(src_sample, s, R, t)
        dists, nn_idx = tgt_tree.query(src_trans, k=1)
        inlier_mask = dists < corr_distance
        if int(inlier_mask.sum()) < 100:
            break

        src_in = src_trans[inlier_mask]
        tgt_in = tgt_sample[nn_idx[inlier_mask]]
        final_inliers = int(inlier_mask.sum())
        final_rmse = float(np.sqrt(np.mean((src_in - tgt_in) ** 2)))

        delta = align_points(torch.from_numpy(src_in), torch.from_numpy(tgt_in), scale=True)
        ds, dR, dt = _scaled_transform_to_components(delta)
        s, R, t = _compose_sim3(s, R, t, ds, dR, dt)

    return s, R, t, {"icp_inliers": final_inliers, "icp_rmse": final_rmse}


# ======================================================================
# Point cloud fusion
# ======================================================================

def _voxel_downsample(points, colors, voxel_size):
    if len(points) == 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_map = {}
    for idx, vi in enumerate(voxel_indices):
        key = (int(vi[0]), int(vi[1]), int(vi[2]))
        voxel_map.setdefault(key, []).append(idx)

    out_pts = np.empty((len(voxel_map), 3), dtype=np.float32)
    out_cols = np.empty((len(voxel_map), 3), dtype=np.uint8)
    for i, inliers in enumerate(voxel_map.values()):
        out_pts[i] = points[inliers].mean(axis=0)
        out_cols[i] = np.clip(colors[inliers].astype(float).mean(axis=0), 0, 255).astype(np.uint8)
    return out_pts, out_cols


def fuse_pointclouds(cumulative_points, cumulative_colors, new_points, new_colors,
                      overlap_radius=0.03):
    if len(new_points) == 0:
        return cumulative_points, cumulative_colors, {"new_points": 0, "replaced": 0}
    if len(cumulative_points) == 0:
        return new_points, new_colors, {"new_points": len(new_points), "replaced": 0}

    new_tree = cKDTree(new_points)
    dists, _ = new_tree.query(cumulative_points, k=1)
    survive_mask = dists > overlap_radius
    replaced = int((~survive_mask).sum())

    fused_points = np.concatenate([new_points, cumulative_points[survive_mask]], axis=0)
    fused_colors = np.concatenate([new_colors, cumulative_colors[survive_mask]], axis=0)

    stats = {
        "new_points": int(len(new_points)),
        "replaced": replaced,
        "kept": int(survive_mask.sum()),
        "total": int(len(fused_points)),
    }
    return fused_points.astype(np.float32), fused_colors.astype(np.uint8), stats


def fuse_gt_refresh(cumulative_points, cumulative_colors, gt_points, gt_colors,
                     overlap_radius=0.03):
    """GT-refresh fusion: replace overlapping cumulative points with GT points.
    GT points are treated as authoritative (corrects drift from reconstruction).
    """
    if len(gt_points) == 0:
        return cumulative_points, cumulative_colors, {
            "gt_points": 0, "cumulative_replaced": 0, "cumulative_kept": len(cumulative_points)
        }
    if len(cumulative_points) == 0:
        return gt_points, gt_colors, {
            "gt_points": len(gt_points), "cumulative_replaced": 0, "cumulative_kept": 0
        }

    gt_tree = cKDTree(gt_points)
    dists, _ = gt_tree.query(cumulative_points, k=1)
    survive_mask = dists > overlap_radius
    survived = int(survive_mask.sum())
    replaced = int((~survive_mask).sum())

    refreshed_points = np.concatenate([gt_points, cumulative_points[survive_mask]], axis=0)
    refreshed_colors = np.concatenate([gt_colors, cumulative_colors[survive_mask]], axis=0)

    stats = {
        "gt_points": int(len(gt_points)),
        "cumulative_replaced": replaced,
        "cumulative_kept": survived,
        "refreshed_total": int(len(refreshed_points)),
        "overlap_radius": overlap_radius,
    }
    return refreshed_points.astype(np.float32), refreshed_colors.astype(np.uint8), stats


# ======================================================================
# Render point cloud to video frames
# ======================================================================

def render_memory_to_frames(points_world, colors, camera_transforms, intrinsics,
                             original_size, target_num_frames=NUM_FRAMES,
                             point_size=2.0, point_world_size=0.002,
                             color_dilation=1,
                             zfar=100.0, opencv_to_opengl_points=False):
    """Render a world-frame point cloud from GT camera viewpoints.
    Returns (color_frames, mask_frames):
      - color_frames: list of uint8 [H, W, 3] RGB frames
      - mask_frames: list of uint8 [H, W, 3] mask frames
          white (255) = no point cloud coverage (model should generate)
          black (0)   = has point cloud coverage (keep rendered content)
    point_world_size: Viser UI point size in world units. When positive, it is
        projected into per-frame pixel sizes for pyrender so rendered_memory.mp4
        follows the same visual scale as the UI global point cloud.
    color_dilation: pixel radius used after rendering to fill tiny gaps in the
        video conditioning image without changing the underlying point cloud.
    opencv_to_opengl_points: if True, transform points (x,y,z)->(x,-y,-z) for OpenGL convention.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    pts = np.asarray(points_world, dtype=np.float32)
    if opencv_to_opengl_points:
        pts = pts * np.array([1.0, -1.0, -1.0], dtype=np.float32)

    src_width, src_height = int(original_size[0]), int(original_size[1])
    image_width, image_height = WIDTH, HEIGHT
    scale_x = image_width / max(src_width, 1)
    scale_y = image_height / max(src_height, 1)
    fx = float(intrinsics[0, 0]) * scale_x
    fy = float(intrinsics[1, 1]) * scale_y
    cx = float(intrinsics[0, 2]) * scale_x
    cy = float(intrinsics[1, 2]) * scale_y

    logger.info(
        f"Rendering {target_num_frames} frames from {len(pts)} points, "
        f"image={image_width}x{image_height} "
        f"(source={src_width}x{src_height}, scale=({scale_x:.4f},{scale_y:.4f}))"
    )
    if len(pts) > 0:
        bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
        logger.info(f"  Point cloud bbox: min={bbox_min}, max={bbox_max}, extent={bbox_max - bbox_min}")

    if colors.dtype != np.uint8:
        c = colors.astype(np.float32)
        if c.max() <= 1.0:
            c = c * 255.0
        colors_uint8 = np.clip(c, 0, 255).astype(np.uint8)
    else:
        colors_uint8 = colors

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
    mesh = pyrender.Mesh.from_points(pts, colors=colors_uint8)
    scene.add(mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=zfar
    )

    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    total_cam_frames = len(camera_transforms)
    stride = max(1, total_cam_frames // target_num_frames)
    frame_indices = list(range(0, total_cam_frames, stride))[:target_num_frames]
    if len(pts) > 0 and frame_indices:
        cam_pos = camera_transforms[frame_indices[0]][:3, 3]
        logger.info(f"  First camera position: {cam_pos}")

    def _ui_projected_point_size(c2w: np.ndarray) -> float:
        if point_world_size <= 0 or len(pts) == 0:
            return float(point_size)

        # Viser uses a world-space point size. Pyrender's GL point size is in
        # pixels, so project that world size using the visible points' depth.
        sample_pts = pts
        if len(sample_pts) > 200_000:
            sample_idx = np.linspace(0, len(sample_pts) - 1, 200_000, dtype=np.int64)
            sample_pts = sample_pts[sample_idx]

        rot = c2w[:3, :3]
        trans = c2w[:3, 3]
        pts_cam = (sample_pts - trans) @ rot
        z = pts_cam[:, 2]
        valid = (z > 0.01) & (z < zfar)
        if valid.any():
            x = pts_cam[:, 0]
            y = pts_cam[:, 1]
            u = fx * x / np.maximum(z, 1e-6) + cx
            v = fy * y / np.maximum(z, 1e-6) + cy
            in_frame = valid & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
            if in_frame.any():
                valid = in_frame

        if not valid.any():
            return float(point_size)

        depth = float(np.median(z[valid]))
        projected = max(fx, fy) * float(point_world_size) / max(depth, 1e-6)
        min_pixel_size = max(float(point_size), 1.0)
        return float(np.clip(projected, min_pixel_size, 12.0))

    rendered_frames = []
    mask_frames = []
    projected_point_sizes = []
    for idx in frame_indices:
        frame_point_size = _ui_projected_point_size(camera_transforms[idx])
        projected_point_sizes.append(frame_point_size)
        renderer = pyrender.OffscreenRenderer(
            image_width, image_height, point_size=frame_point_size
        )
        c2w_gl = camera_transforms[idx] @ opencv_to_opengl
        for node in list(scene.get_nodes()):
            if node.camera is not None:
                scene.remove_node(node)
        scene.add(camera, pose=c2w_gl)
        color_rgb, depth_buf = renderer.render(scene)
        renderer.delete()
        if color_dilation > 0:
            coverage = depth_buf > 0
            kernel = np.ones((color_dilation * 2 + 1, color_dilation * 2 + 1), np.uint8)
            dilated_coverage = cv2.dilate(coverage.astype(np.uint8), kernel, iterations=1).astype(bool)
            fill_mask = dilated_coverage & (~coverage)
            if fill_mask.any():
                color_rgb = cv2.dilate(color_rgb, kernel, iterations=1)
                depth_buf = depth_buf.copy()
                depth_buf[fill_mask] = 1.0
        rendered_frames.append(color_rgb.copy())

        has_depth = depth_buf > 0
        mask_frame = np.full((depth_buf.shape[0], depth_buf.shape[1], 3), 255, dtype=np.uint8)
        mask_frame[has_depth] = 0
        mask_frames.append(mask_frame)

    if projected_point_sizes:
        logger.info(
            "UI-projected render point_size(px): "
            f"mean={np.mean(projected_point_sizes):.2f}, "
            f"min={np.min(projected_point_sizes):.2f}, max={np.min(projected_point_sizes):.2f} "
            f"(world={point_world_size})"
        )

    while len(rendered_frames) < target_num_frames:
        rendered_frames.append(rendered_frames[-1].copy())
        mask_frames.append(mask_frames[-1].copy())

    coverage = np.mean([np.mean(m[:, :, 0] == 0) for m in mask_frames]) * 100
    logger.info(f"Rendered {len(rendered_frames)} frames, avg coverage={coverage:.1f}%")
    return rendered_frames, mask_frames


def render_pointcloud_turntable(points_world, colors, num_frames=60,
                                 image_width=640, image_height=480,
                                 point_size=2.0):
    """Render a 360-degree turntable video of a point cloud.

    The camera orbits around the point cloud center at a fixed elevation,
    producing a smooth rotation visualization.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    if len(points_world) == 0:
        return []

    if colors.dtype != np.uint8:
        c = colors.astype(np.float32)
        if c.max() <= 1.0:
            c = c * 255.0
        colors_uint8 = np.clip(c, 0, 255).astype(np.uint8)
    else:
        colors_uint8 = colors

    center = points_world.mean(axis=0)
    extent = points_world.max(axis=0) - points_world.min(axis=0)
    radius = float(np.linalg.norm(extent)) * 0.8
    if radius < 1e-6:
        radius = 1.0
    elevation = 0.3  # ~17 degrees above horizontal

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
    mesh = pyrender.Mesh.from_points(points_world, colors=colors_uint8)
    scene.add(mesh)

    fov = np.pi / 3.0
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=image_width / image_height,
                                         znear=0.01, zfar=radius * 10)
    renderer = pyrender.OffscreenRenderer(image_width, image_height, point_size=point_size)

    frames = []
    for i in range(num_frames):
        angle = 2.0 * np.pi * i / num_frames
        eye = center + radius * np.array([
            np.cos(angle) * np.cos(elevation),
            np.sin(elevation),
            np.sin(angle) * np.cos(elevation),
        ])

        forward = center - eye
        forward /= np.linalg.norm(forward) + 1e-8
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye

        for node in list(scene.get_nodes()):
            if node.camera is not None:
                scene.remove_node(node)
        scene.add(camera, pose=c2w)
        color_rgb, _ = renderer.render(scene)
        frames.append(color_rgb.copy())

    renderer.delete()
    logger.info(f"Turntable rendered {len(frames)} frames ({image_width}x{image_height})")
    return frames


def get_host_ip() -> str:
    """UDP trick for primary NIC address (typical Viser bind / URL hint)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def _resolve_viser_bind_host(host: str) -> str:
    """``auto`` → ``get_host_ip()``. Use ``0.0.0.0`` for all interfaces."""
    h = (host or "auto").strip().lower()
    if h in ("auto", "egosim_state", "default"):
        return get_host_ip()
    return host


def run_viser_interactive_memory(
    points_world: np.ndarray,
    colors: np.ndarray,
    *,
    host: str = "auto",
    port: int = 20540,
    point_size: float = 0.002,
    max_points: int = 800_000,
) -> None:
    """Minimal Viser point cloud (``points`` backend). Requires ``pip install viser``."""
    try:
        import viser
    except ImportError as e:
        raise ImportError(
            "Interactive Viser UI requires: pip install viser"
        ) from e

    pts = np.asarray(points_world, dtype=np.float32)
    cols = np.asarray(colors)
    if len(pts) == 0:
        logger.warning("Viser: empty point cloud, skipping")
        return

    if max_points > 0 and len(pts) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[sel]
        cols = cols[sel]
        logger.info(f"Viser: subsampled to {max_points} points for display")

    if cols.dtype != np.uint8:
        c = cols.astype(np.float32)
        if c.size and float(c.max()) <= 1.0:
            c = c * 255.0
        cols_u8 = np.clip(c, 0, 255).astype(np.uint8)
    else:
        cols_u8 = cols

    bind_host = _resolve_viser_bind_host(host)
    server = viser.ViserServer(host=bind_host, port=port, verbose=False)
    try:
        server.scene.add_point_cloud(
            "/global_point_cloud",
            points=pts,
            colors=cols_u8,
            point_size=point_size,
            point_shape="rounded",
        )
    except TypeError:
        server.scene.add_point_cloud(
            "/global_point_cloud",
            points=pts,
            colors=cols_u8,
            point_size=point_size,
        )

    actual_port = server.get_port()
    logger.info(
        f"Viser (points backend): http://{bind_host}:{actual_port} — Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Shutting down Viser server...")
    finally:
        server.stop()


def run_viser_egosim_state_cli(output_dir: Path, port: int = 20540) -> None:
    """Full scene-stack Viser UI (``egosim_state`` backend); wraps upstream ``run_viser``."""
    from egosim_state.utils.viser import run_viser

    run_viser(Path(output_dir), port=port)


# ======================================================================
# CLI + main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EgoSim state subprocess: infer + reconstruct + align + fuse + render"
    )
    parser.add_argument("--generated_video", type=str, required=True)
    parser.add_argument("--hdf5_path", type=str, default="",
                        help="HDF5 for GT camera transforms (optional; Sim3 alignment & rendering skipped if empty)")
    parser.add_argument("--process_result_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Intermediate artifacts output directory")
    parser.add_argument("--output_memory", type=str, required=True,
                        help="Path to save updated cumulative memory (.npz)")

    parser.add_argument("--cumulative_memory", type=str, default="",
                        help="Path to existing cumulative memory (.npz). Empty for first clip.")
    parser.add_argument("--phrases", nargs="*", default=None,
                        help="Inline phrase list for instance segmentation")
    parser.add_argument("--phrases_json", type=str, default="",
                        help="Path to phrases JSON for instance segmentation")

    parser.add_argument("--next_hdf5_path", type=str, default="",
                        help="HDF5 for next clip (to render memory for next clip)")
    parser.add_argument("--next_gt_process_result_dir", type=str, default="",
                        help="Next clip's GT process_result dir (for cross-HDF5 coordinate transform)")
    parser.add_argument("--clip_start_frame", type=int, default=-1,
                        help="Current clip start frame (for GT pose slice in alignment)")
    parser.add_argument("--clip_end_frame", type=int, default=-1,
                        help="Current clip end frame (for GT pose slice in alignment)")
    parser.add_argument("--next_clip_start_frame", type=int, default=-1,
                        help="Next clip start frame (slice cameras for rendering)")
    parser.add_argument("--next_clip_end_frame", type=int, default=-1,
                        help="Next clip end frame (slice cameras for rendering)")
    parser.add_argument("--rendered_video_out", type=str, default="",
                        help="Path to save rendered memory video for next clip")
    parser.add_argument("--mask_video_out", type=str, default="",
                        help="Path to save mask video for next clip (white=generate, black=keep)")
    parser.add_argument("--pointcloud_video_out", type=str, default="",
                        help="Path to save 360-degree turntable video of cumulative point cloud")
    parser.add_argument("--viser", action="store_true", default=False,
                        help="After saving memory JSON, start Viser (blocks until Ctrl+C)")
    parser.add_argument(
        "--viser_backend",
        type=str,
        choices=["egosim_state", "points"],
        default="egosim_state",
        help="egosim_state: full Viser UI via scene stack run_viser (default). "
             "points: minimal fused point cloud viewer (debug).",
    )
    parser.add_argument(
        "--viser_host",
        type=str,
        default="auto",
        help="Bind address: 'auto' uses get_host_ip() (primary NIC); 0.0.0.0 = all interfaces (points backend)",
    )
    parser.add_argument(
        "--viser_port",
        type=int,
        default=20540,
        help="Viser HTTP port (run_viser default 20540)",
    )
    parser.add_argument(
        "--viser_point_size",
        type=float,
        default=0.002,
        help="Point size (points backend; default 0.002)",
    )
    parser.add_argument("--viser_max_points", type=int, default=800_000,
                        help="Random subsample cap for Viser (0 = no subsampling)")

    parser.add_argument("--render_gt_pointcloud", action="store_true",
                        help="Debug: render GT pointcloud instead of reconstruction (requires GT pointcloud)")
    parser.add_argument("--opencv_to_opengl_points", action="store_true",
                        help="Debug: transform points [x,-y,-z] before render (test coordinate system)")
    parser.add_argument("--prefer_icp_alignment", action="store_true",
                        help="When GT pointcloud available, use ICP instead of pose-based Sim3")
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
    parser.add_argument("--icp_refine_iters", type=int, default=20,
                        help="Max ICP iterations for refinement")
    parser.add_argument("--icp_refine_corr_dist", type=float, default=0.5,
                        help="ICP correspondence distance threshold for refinement (start value)")
    parser.add_argument("--pose_center_sample_step", type=int, default=1,
                        help="Subsample step for pose-center correspondences in Sim3 alignment")

    parser.add_argument("--egosim_state_pipeline", type=str, default="dav3")
    parser.add_argument("--scene_save_viz", action="store_true", default=False,
                        help="Match `egosim_state infer --visualize` by saving scene-side visualization artifacts")
    parser.add_argument("--no_scene_save_viz", action="store_false", dest="scene_save_viz")
    parser.add_argument("--scene_slam_visualize", action="store_true", default=False,
                        help="Enable scene-side SLAM visualization during inference")
    parser.add_argument("--no_scene_slam_visualize", action="store_false", dest="scene_slam_visualize")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--spatial_subsample", type=int, default=1)
    parser.add_argument("--temporal_subsample", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--fuse_overlap_radius", type=float, default=0.005)
    parser.add_argument("--render_point_size", type=float, default=3.0,
                        help="Fallback fixed pyrender point size in pixels when render_point_world_size <= 0")
    parser.add_argument("--render_point_world_size", type=float, default=0.002,
                        help="World-space point size matching the Viser UI global point cloud; "
                             "projected per frame into pyrender pixels. Set <=0 to use render_point_size.")
    parser.add_argument("--render_color_dilation", type=int, default=1,
                        help="Pixel radius to dilate rendered memory colors/mask and fill small holes.")
    parser.add_argument("--fps", type=int, default=16)

    parser.add_argument("--filter_interactive", action="store_true", default=True)
    parser.add_argument("--no_filter_interactive", action="store_false", dest="filter_interactive")
    parser.add_argument("--mask_dilation", type=int, default=5)
    parser.add_argument("--filter_body_parts", action="store_true", default=True,
                        help="Filter body-part instances (hand/arm/person) from background by phrase")
    parser.add_argument("--no_filter_body_parts", action="store_false", dest="filter_body_parts")
    parser.add_argument("--body_part_dilation", type=int, default=10,
                        help="Dilation radius for body-part mask (larger catches more boundary pixels)")
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
    parser.add_argument("--remove_small_clusters", action="store_true", default=False,
                        help="Remove small disconnected point clusters (floating hand/arm fragments)")
    parser.add_argument("--no_remove_small_clusters", action="store_false", dest="remove_small_clusters")
    parser.add_argument("--cluster_distance_thresh", type=float, default=0.05,
                        help="Max distance between points in the same cluster")
    parser.add_argument("--min_cluster_fraction", type=float, default=0.02,
                        help="Clusters smaller than this fraction of total points are removed")
    parser.add_argument("--min_cluster_size", type=int, default=500,
                        help="Absolute minimum cluster size to keep")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    generated_video = Path(args.generated_video)
    hdf5_path = Path(args.hdf5_path) if args.hdf5_path else None
    process_result_dir = Path(args.process_result_dir) if args.process_result_dir else None
    output_dir = Path(args.output_dir)
    output_memory_path = Path(args.output_memory)

    import time as _time
    result = {"status": "success"}

    logger.info(f"=== egosim_state_subprocess START ===")
    logger.info(f"  generated_video: {generated_video}")
    logger.info(f"  hdf5_path: {hdf5_path}")
    logger.info(f"  process_result_dir: {process_result_dir}")
    logger.info(f"  pipeline: {args.egosim_state_pipeline}")
    logger.info(f"  egosim_state scene flags: filter_interactive={args.filter_interactive}, "
                f"filter_body_parts={args.filter_body_parts} (dilation={args.body_part_dilation}), "
                f"use_tsdf={args.use_tsdf}, use_color_depth_overlap={args.use_color_depth_overlap} (overlap will {'RUN' if args.use_color_depth_overlap else 'NOT run'}), "
                f"use_last_frame_objects={args.use_last_frame_objects}, "
                f"statistical_outlier_removal={args.statistical_outlier_removal}, "
                f"remove_small_clusters={args.remove_small_clusters} "
                f"(dist={args.cluster_distance_thresh}, frac={args.min_cluster_fraction}, min={args.min_cluster_size})")

    has_hdf5 = hdf5_path is not None and hdf5_path.exists()

    # Load phrases if provided
    phrases = None
    if args.phrases:
        phrases = normalize_scene_phrases(args.phrases)
        logger.info(f"Loaded inline phrases: {phrases}")
    elif args.phrases_json and Path(args.phrases_json).exists():
        with open(args.phrases_json) as f:
            phrases = normalize_scene_phrases(json.load(f))
        logger.info(f"Loaded phrases: {phrases}")

    # Load GT data (optional — requires HDF5)
    _t = _time.time()
    gt_camera_transforms = None
    gt_pts = np.zeros((0, 3), dtype=np.float32)
    gt_cols = np.zeros((0, 3), dtype=np.uint8)
    gt_intrinsics = None
    gt_original_size = None

    if has_hdf5:
        logger.info("Loading GT data from HDF5...")
        if args.clip_start_frame >= 0 and args.clip_end_frame >= 0:
            seg_len = args.clip_end_frame - args.clip_start_frame + 1
            gt_camera_transforms = _load_gt_cameras_range(
                hdf5_path, args.clip_start_frame, args.clip_end_frame
            )
            logger.info(f"  GT cameras: {len(gt_camera_transforms)} poses for "
                        f"frames {args.clip_start_frame}-{args.clip_end_frame} "
                        f"(expected {seg_len})")
        else:
            gt_camera_transforms = load_gt_camera_transforms(hdf5_path)
        try:
            gt_pts, gt_cols, gt_intrinsics, gt_original_size = load_gt_pointcloud(
                process_result_dir, hdf5_path
            )
        except (FileNotFoundError, Exception):
            logger.warning("No GT pointcloud available")
        logger.info(f"GT data loaded in {_time.time() - _t:.1f}s "
                    f"(gt_points={len(gt_pts)}, gt_cameras={len(gt_camera_transforms)})")
    else:
        logger.info("No HDF5 provided — skipping GT data loading, Sim3 alignment, and rendering")

    # Load cumulative memory if provided (background only; last_frame is per-segment).
    cumulative_background = np.zeros((0, 3), dtype=np.float32)
    cumulative_background_colors = np.zeros((0, 3), dtype=np.uint8)
    if args.cumulative_memory and Path(args.cumulative_memory).exists():
        mem = np.load(args.cumulative_memory)
        if "background_points" in mem:
            cumulative_background = mem["background_points"]
            cumulative_background_colors = mem["background_colors"]
        else:
            cumulative_background = mem["points"]
            cumulative_background_colors = mem["colors"]
        logger.info(f"Loaded cumulative memory: {len(cumulative_background)} background points")

    # Step 1: EgoSim state scene inference
    _t = _time.time()
    logger.info("=== Step 1: EgoSim state scene inference ===")
    artifact = run_egosim_state_infer(
        generated_video,
        output_dir,
        args.egosim_state_pipeline,
        phrases=phrases,
        save_viz=args.scene_save_viz,
        slam_visualize=args.scene_slam_visualize,
    )
    logger.info(f"=== Step 1 done in {_time.time() - _t:.1f}s ===")

    # Step 2: Reconstruct point cloud (background + last-frame objects separately)
    _t = _time.time()
    logger.info("=== Step 2: Reconstructing point cloud ===")
    recon_kwargs = dict(
        spatial_subsample=args.spatial_subsample,
        temporal_subsample=args.temporal_subsample,
        filter_interactive=args.filter_interactive,
        mask_dilation=args.mask_dilation,
        filter_body_parts=args.filter_body_parts,
        body_part_dilation=args.body_part_dilation,
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
        remove_small_clusters=args.remove_small_clusters,
        cluster_distance_thresh=args.cluster_distance_thresh,
        min_cluster_fraction=args.min_cluster_fraction,
        min_cluster_size=args.min_cluster_size,
    )
    bg_points, bg_colors, lf_points, lf_colors, _all_inst_points, pred_poses, recon_stats = reconstruct_from_artifacts(
        artifact, **recon_kwargs,
    )
    if (
        args.filter_interactive
        and recon_stats.get("interactive_filtered_points", 0) > recon_stats.get("raw_valid_points", 0)
    ):
        logger.warning(
            f"interactive_filtered_points ({recon_stats['interactive_filtered_points']}) > "
            f"raw_valid_points ({recon_stats['raw_valid_points']}); re-running scene infer (SAM segmentation)"
        )
        artifact = run_egosim_state_infer(
            generated_video,
            output_dir,
            args.egosim_state_pipeline,
            phrases=phrases,
            save_viz=args.scene_save_viz,
            slam_visualize=args.scene_slam_visualize,
            force_reinfer=True,
        )
        bg_points, bg_colors, lf_points, lf_colors, _all_inst_points, pred_poses, recon_stats = reconstruct_from_artifacts(
            artifact, **recon_kwargs,
        )
        result["reconstruction_retried"] = True
        logger.info(f"Reconstruct after scene re-run: background={len(bg_points)}, last_frame={len(lf_points)}")
    result["reconstruction"] = recon_stats
    canonical_ply = artifact.base_path / f"{artifact.artifact_name}_global_point_cloud.ply"
    if len(bg_points) > 0:
        _save_point_cloud_ply(bg_points, bg_colors, canonical_ply)
        result["reconstruction"]["global_point_cloud_ply"] = str(canonical_ply)
    logger.info(f"=== Step 2 done in {_time.time() - _t:.1f}s === "
                f"(background={len(bg_points)}, last_frame={len(lf_points)}, recon_stats={recon_stats})")

    if args.voxel_size > 0:
        if len(bg_points) > 0:
            bg_points, bg_colors = _voxel_downsample(bg_points, bg_colors, args.voxel_size)
            logger.info(f"  Background voxel downsampled to {len(bg_points)} points")
        if len(lf_points) > 0:
            lf_points, lf_colors = _voxel_downsample(lf_points, lf_colors, args.voxel_size)
            logger.info(f"  Last-frame objects voxel downsampled to {len(lf_points)} points")
    # Step 3: Sim3 alignment (apply to both background and last-frame)
    _t = _time.time()
    logger.info("=== Step 3: Sim3 alignment ===")
    aligned_bg = bg_points
    aligned_bg_colors = bg_colors
    aligned_lf = lf_points
    aligned_lf_colors = lf_colors
    sim3_scale = 1.0
    if gt_camera_transforms is None:
        logger.info("  Skipped (no HDF5 / GT cameras)")
    elif len(bg_points) > 0:
        local_range = bg_points.max(axis=0) - bg_points.min(axis=0)
        logger.info(f"  Pre-alignment: {len(bg_points)} bg pts, "
                    f"range={local_range}, mean={bg_points.mean(axis=0)}")
        method = args.align_method
        try:
            if method == "icp" and len(gt_pts) > 0:
                logger.info("  Using ICP alignment")
                s, R, t, align_stats = estimate_sim3_icp(bg_points, gt_pts)
                align_stats["method"] = "icp"
            elif method == "pose_then_icp" and len(gt_pts) > 0:
                logger.info("  Using pose_then_icp (SO3 avg for R, coarse-to-fine ICP for s+t)")
                s, R, t, align_stats = estimate_sim3_pose_then_icp(
                    pred_poses, gt_camera_transforms,
                    bg_points, gt_pts,
                    sample_step=args.pose_center_sample_step,
                )
                align_stats["method"] = "pose_then_icp"
            elif method == "pose_full_sim3":
                logger.info("  Using full-pose rigid transform (R, t from cameras, s=1.0)")
                s, R, t, align_stats = estimate_sim3_from_full_poses(
                    pred_poses, gt_camera_transforms,
                    sample_step=args.pose_center_sample_step,
                )
                align_stats["method"] = "pose_full_sim3"
            elif method == "pose_full_sim3_scale":
                logger.info("  Using full-pose Sim3 (SO3 avg for R + scale estimation)")
                s, R, t, align_stats = estimate_sim3_from_full_poses_with_scale(
                    pred_poses, gt_camera_transforms,
                    sample_step=args.pose_center_sample_step,
                )
                align_stats["method"] = "pose_full_sim3_scale"
            else:
                logger.info("  Using pose-center Sim3 (Umeyama on centers only)")
                s, R, t, align_stats = estimate_sim3_from_pose_centers(
                    pred_poses, gt_camera_transforms,
                    sample_step=args.pose_center_sample_step,
                )
                align_stats["method"] = "pose_sim3"
            sim3_scale = s
            aligned_bg = _apply_sim3(bg_points, s, R, t)
            aligned_bg_colors = bg_colors
            if len(lf_points) > 0:
                aligned_lf = _apply_sim3(lf_points, s, R, t)
                aligned_lf_colors = lf_colors
            result["alignment"] = align_stats
            result["alignment"]["sim3_scale"] = float(s)
            logger.info(f"  Sim3 ({align_stats['method']}): scale={s:.6f}")

            # Optional ICP refinement: use GT point cloud to fine-tune pose-based alignment
            if (args.icp_refine and len(gt_pts) > 0 and len(aligned_bg) > 0
                    and method not in ("icp", "pose_then_icp")):
                try:
                    ds, dR, dt, icp_stats = estimate_sim3_icp(
                        aligned_bg, gt_pts,
                        max_iters=args.icp_refine_iters,
                        corr_distance=args.icp_refine_corr_dist,
                        sample_size=30000,
                    )
                    aligned_bg = _apply_sim3(aligned_bg, ds, dR, dt)
                    if len(aligned_lf) > 0:
                        aligned_lf = _apply_sim3(aligned_lf, ds, dR, dt)
                    s_total, R_total, t_total = _compose_sim3(s, R, t, ds, dR, dt)
                    sim3_scale = s_total
                    result["alignment"]["icp_refine"] = icp_stats
                    result["alignment"]["icp_refine_delta_scale"] = float(ds)
                    result["alignment"]["sim3_scale"] = float(s_total)
                    logger.info(f"  ICP refinement: delta_s={ds:.6f}, inliers={icp_stats['icp_inliers']}, "
                                f"rmse={icp_stats['icp_rmse']:.6f}")
                except Exception as e_icp:
                    logger.warning(f"  ICP refinement failed (non-fatal): {e_icp}")

        except Exception as e:
            logger.warning(f"Alignment ({method}) failed: {e}, trying fallback")
            try:
                s, R, t, align_stats = estimate_sim3_from_full_poses(
                    pred_poses, gt_camera_transforms,
                    sample_step=args.pose_center_sample_step,
                )
                sim3_scale = s
                aligned_bg = _apply_sim3(bg_points, s, R, t)
                aligned_bg_colors = bg_colors
                if len(lf_points) > 0:
                    aligned_lf = _apply_sim3(lf_points, s, R, t)
                    aligned_lf_colors = lf_colors
                result["alignment"] = align_stats
                result["alignment"]["sim3_scale"] = float(s)
                result["alignment"]["method"] = "pose_full_sim3_fallback"
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}")

    logger.info(f"=== Step 3 done in {_time.time() - _t:.1f}s ===")

    # Step 4: Fuse backgrounds only, then append current segment's last-frame objects
    _t = _time.time()
    logger.info("=== Step 4: Fuse backgrounds, then append last-frame objects ===")
    if len(aligned_bg) > 0 or len(cumulative_background) > 0:
        fused_bg, fused_bg_colors, fuse_stats = fuse_pointclouds(
            cumulative_background, cumulative_background_colors,
            aligned_bg, aligned_bg_colors,
            overlap_radius=args.fuse_overlap_radius,
        )
        if args.voxel_size > 0 and len(fused_bg) > 0:
            pre_ds = len(fused_bg)
            fused_bg, fused_bg_colors = _voxel_downsample(
                fused_bg, fused_bg_colors, args.voxel_size
            )
            logger.info(f"  Fused background voxel downsample: {pre_ds} -> {len(fused_bg)}")

        cumulative_points = np.concatenate([fused_bg, aligned_lf], axis=0) if len(aligned_lf) > 0 else fused_bg
        cumulative_colors = np.concatenate([fused_bg_colors, aligned_lf_colors], axis=0) if len(aligned_lf) > 0 else fused_bg_colors

        result["fusion"] = fuse_stats
        result["fusion"]["last_frame_points"] = int(len(aligned_lf))
        result["cumulative_points"] = int(len(cumulative_points))
        logger.info(
            f"Fused background: new={fuse_stats['new_points']} replaced={fuse_stats['replaced']} "
            f"bg_total={len(fused_bg)}; appended last_frame={len(aligned_lf)}; "
            f"cumulative={len(cumulative_points)}"
        )
    else:
        fused_bg = np.zeros((0, 3), dtype=np.float32)
        fused_bg_colors = np.zeros((0, 3), dtype=np.uint8)
        cumulative_points = aligned_lf
        cumulative_colors = aligned_lf_colors
        if len(cumulative_points) == 0:
            logger.warning("Reconstruction produced empty point cloud")
            result["status"] = "empty_pointcloud"
        else:
            result["fusion"] = {"new_points": 0, "replaced": 0, "last_frame_points": len(aligned_lf)}
            result["cumulative_points"] = int(len(cumulative_points))
    logger.info(f"=== Step 4 done in {_time.time() - _t:.1f}s ===")

    # Save: background and last_frame separately for next clip's fusion
    output_memory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_memory_path,
        points=cumulative_points,
        colors=cumulative_colors,
        background_points=fused_bg,
        background_colors=fused_bg_colors,
        last_frame_points=aligned_lf,
        last_frame_colors=aligned_lf_colors,
    )
    logger.info(f"Saved memory: {output_memory_path} ({len(cumulative_points)} points, "
                f"bg={len(fused_bg)} lf={len(aligned_lf)})")

    # Optional: 360° turntable of cumulative memory (visualization)
    if args.pointcloud_video_out and len(cumulative_points) > 0:
        try:
            _tt_frames = render_pointcloud_turntable(
                cumulative_points,
                cumulative_colors,
                num_frames=60,
                point_size=args.render_point_size,
            )
            if _tt_frames:
                Path(args.pointcloud_video_out).parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(args.pointcloud_video_out), _tt_frames, fps=args.fps, quality=8)
                logger.info(f"Saved EgoSim memory turntable: {args.pointcloud_video_out}")
                result["pointcloud_video_out"] = str(args.pointcloud_video_out)
        except Exception as e:
            logger.warning(f"Turntable visualization failed: {e}")
            traceback.print_exc()

    # Step 5: Render for next clip if requested
    _t = _time.time()
    logger.info("=== Step 5: Render for next clip ===")
    if (args.next_hdf5_path and args.rendered_video_out
            and len(cumulative_points) > 0):
        try:
            next_hdf5 = Path(args.next_hdf5_path)
            if next_hdf5.exists():
                if args.next_clip_start_frame >= 0 and args.next_clip_end_frame >= 0:
                    seg_len = args.next_clip_end_frame - args.next_clip_start_frame + 1
                    next_camera_transforms = _load_gt_cameras_range(
                        next_hdf5, args.next_clip_start_frame, args.next_clip_end_frame
                    )
                    logger.info(f"  Render cameras: {len(next_camera_transforms)} poses for "
                                f"frames {args.next_clip_start_frame}-{args.next_clip_end_frame} "
                                f"(expected {seg_len})")
                else:
                    next_camera_transforms = load_gt_camera_transforms(next_hdf5)

                intrinsics, original_size = resolve_render_intrinsics(
                    gt_intrinsics=gt_intrinsics,
                    gt_original_size=gt_original_size,
                    hdf5_path=next_hdf5,
                    process_result_dir=(
                        Path(args.next_gt_process_result_dir)
                        if args.next_gt_process_result_dir
                        else Path(args.process_result_dir)
                    ),
                    scene_artifact=artifact,
                )
                if intrinsics is None:
                    logger.warning("Cannot render: no intrinsics available")

                if intrinsics is not None:
                    render_pts = gt_pts if (args.render_gt_pointcloud and len(gt_pts) > 0) else cumulative_points
                    render_cols = gt_cols if (args.render_gt_pointcloud and len(gt_pts) > 0) else cumulative_colors

                    hdf5_path_str = str(hdf5_path) if hdf5_path is not None else ""
                    # Cross-HDF5: only update intrinsics/original_size from next clip.
                    # ICP alignment of cumulative points to next_gt is DISABLED to avoid
                    # re-scaling memory across clips (causes recon/GT size drift in later segments).
                    if (not args.render_gt_pointcloud and len(render_pts) > 0 and args.next_gt_process_result_dir
                            and hdf5_path_str and args.next_hdf5_path
                            and Path(hdf5_path_str).resolve() != Path(args.next_hdf5_path).resolve()):
                        next_pr_dir = Path(args.next_gt_process_result_dir)
                        try:
                            _, _, next_intr, next_orig = load_gt_pointcloud(next_pr_dir, next_hdf5)
                            intrinsics, original_size = next_intr, next_orig
                            logger.info(
                                f"  Cross-HDF5: reuse cumulative points, only updated intrinsics/original_size"
                            )
                        except Exception as e:
                            logger.warning(f"  Cross-HDF5 intrinsics load failed: {e}, rendering may be wrong")

                    if args.render_gt_pointcloud:
                        logger.info(f"  DEBUG: rendering GT pointcloud ({len(render_pts)} pts) instead of reconstruction")
                    rendered_frames, mask_frames = render_memory_to_frames(
                        render_pts, render_cols,
                        next_camera_transforms, intrinsics, original_size,
                        target_num_frames=NUM_FRAMES,
                        point_size=args.render_point_size,
                        point_world_size=args.render_point_world_size,
                        color_dilation=args.render_color_dilation,
                        opencv_to_opengl_points=args.opencv_to_opengl_points,
                    )
                    rendered_out = Path(args.rendered_video_out)
                    rendered_out.parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(str(rendered_out), rendered_frames, fps=args.fps, quality=8)
                    logger.info(f"Saved rendered memory video: {rendered_out}")
                    result["rendered_video"] = str(rendered_out)

                    if args.mask_video_out:
                        mask_out = Path(args.mask_video_out)
                        mask_out.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(str(mask_out), mask_frames, fps=args.fps, quality=8)
                        logger.info(f"Saved mask video: {mask_out}")
                        result["mask_video"] = str(mask_out)
        except Exception as e:
            logger.warning(f"Rendering for next clip failed: {e}")
            traceback.print_exc()

    logger.info(f"=== Step 5 done in {_time.time() - _t:.1f}s ===")

    # Write result JSON
    result_json_path = output_memory_path.with_suffix(".json")
    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result: {result_json_path}")

    if args.viser:
        if args.viser_backend == "egosim_state":
            try:
                logger.info(
                    "Viser backend=egosim_state: starting scene stack run_viser "
                    "(needs importable scene stack / PYTHONPATH)."
                )
                run_viser_egosim_state_cli(output_dir, port=args.viser_port)
            except ImportError as e:
                logger.error(
                    "viser_backend=egosim_state requires the scene stack (``import egosim_state``). "
                    "Set EGOWM_EGOSIM_STATE_ROOT or install the package. "
                    f"Import error: {e}"
                )
                return 1
            except Exception as e:
                logger.warning(f"run_viser failed: {e}")
                traceback.print_exc()
                return 1
        else:
            if len(cumulative_points) == 0:
                logger.warning("viser_backend=points: empty cumulative point cloud, skipping Viser")
            else:
                try:
                    run_viser_interactive_memory(
                        cumulative_points,
                        cumulative_colors,
                        host=args.viser_host,
                        port=args.viser_port,
                        point_size=args.viser_point_size,
                        max_points=args.viser_max_points,
                    )
                except ImportError as e:
                    logger.error(str(e))
                    return 1
                except Exception as e:
                    logger.warning(f"Viser (points) failed: {e}")
                    traceback.print_exc()
                    return 1

    logger.info(f"=== egosim_state_subprocess DONE ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
