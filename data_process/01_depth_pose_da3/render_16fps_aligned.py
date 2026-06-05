#!/usr/bin/env python3
# Copyright (c) jiamingda (https://github.com/Luyitas)
"""
Render 16fps point-cloud video for a single clip (pipeline Steps 01d / 01d-mask).

Logic:
1. Load first-frame RGB/Depth + Intrinsics -> Reconstruct Point Cloud in Frame 0.
2. Load smoothed extrinsics (30fps) -> Compute map to 16fps -> Select poses.
3. Compute 16fps-effective Intrinsics (account for CenterCrop and Resize).
4. Render using PyRender with valid poses.
5. Overlay on 16fps video.
"""

import os
import sys
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ.get('RENDER_PLATFORM', 'osmesa'))
os.environ['PYOPENGL_ERROR_ON_COPY'] = '0'

import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import glob
import shutil
from fractions import Fraction
from pathlib import Path

# Add current dir to path to find sibling module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from camera_trajectory_render_pyrender import (
        render_video_with_pyrender,
        pointcloud_cam_to_world,
        render_overlay_video
    )
except ImportError as e:
    print(f"Error: Could not import camera_trajectory_render_pyrender.py! Details: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Transformation from OpenCV (X right, Y down, Z fwd) to OpenGL (X right, Y up, Z back)
OPENCV_TO_OPENGL = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float32)


def as_4x4(m):
    """Ensure matrix is 4x4"""
    if m.shape == (4, 4):
        return m
    ret = np.eye(4)
    if m.shape == (3, 4):
        ret[:3, :4] = m
    elif m.shape == (3, 3):
        ret[:3, :3] = m
    return ret


def backproject_depth(depth, K):
    """Backproject depth to 3D points in camera frame"""
    H, W = depth.shape
    # meshgrid 'xy' gives i (col, x), j (row, y)
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    Z = depth
    X = (i - cx) * Z / fx
    Y = (j - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return points


def load_depth_npy(depth_path: str) -> np.ndarray:
    depth = np.load(depth_path)
    # Handle (1, H, W) case if necessary
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D (H,W). Got shape={depth.shape}")
    return depth


def load_rgb_rgb(rgb_path: str) -> np.ndarray:
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB: {rgb_path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def load_first_frame_from_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open color video: {video_path}")
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame from color video: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def create_point_cloud_from_rgbd(depth: np.ndarray, rgb: np.ndarray, K_orig: np.ndarray):
    """Create point cloud from already-loaded Depth/RGB."""
    H, W = depth.shape
    H_rgb, W_rgb = rgb.shape[:2]

    if (H, W) != (H_rgb, W_rgb):
        print(f"Warning: Resize RGB ({W_rgb}x{H_rgb}) to Depth ({W}x{H})")
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    print("Backprojecting point cloud...")
    points_cam = backproject_depth(depth, K_orig)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return points_cam, colors


def get_aligned_intrinsics(K, orig_w, orig_h, target_w=1280, target_h=720):
    """
    Calculate new intrinsics after Center Crop (16:9) and Resize.
    Logic matches process_egovid_16fps_720p.py
    """
    # 1. Center Crop to 16:9
    target_h_crop = int(orig_w * 9 / 16)
    target_h_crop = (target_h_crop // 2) * 2  # Ensure even
    
    K_new = K.copy()
    cropped_w, cropped_h = 0, 0
    
    if target_h_crop >= orig_h:
        # Crop Width
        target_w_crop = int(orig_h * 16 / 9)
        target_w_crop = (target_w_crop // 2) * 2
        crop_x = (orig_w - target_w_crop) // 2
        K_new[0, 2] -= crop_x
        cropped_w, cropped_h = target_w_crop, orig_h
        print(f"Crop Mode: Width (Crop X={crop_x}, New W={cropped_w})")
    else:
        # Crop Height
        crop_y = (orig_h - target_h_crop) // 2
        K_new[1, 2] -= crop_y
        cropped_w, cropped_h = orig_w, target_h_crop
        print(f"Crop Mode: Height (Crop Y={crop_y}, New H={cropped_h})")
        
    # 2. Resize
    scale_x = target_w / float(cropped_w)
    scale_y = target_h / float(cropped_h)
    
    print(f"Resizing: {cropped_w}x{cropped_h} -> {target_w}x{target_h} (Scale X={scale_x:.4f}, Y={scale_y:.4f})")
    
    K_new[0, :] *= scale_x
    K_new[1, :] *= scale_y
    
    return K_new


def stitch_pose_jumps(T_wc_all, min_jump=0.03, factor=8.0):
    """
    Detect large discontinuities in camera centers and stitch later segments with a rigid transform.

    Args:
        T_wc_all: list[np.ndarray(4,4)] camera-to-world poses.
        min_jump: absolute minimum jump (meters) to consider as discontinuity.
        factor: threshold factor over median step.

    Returns:
        stitched_poses, jump_indices
        jump_indices are boundary indices i where jump is between i and i+1.
    """
    if len(T_wc_all) <= 2:
        return T_wc_all, []

    camera_centers = np.array([T[:3, 3] for T in T_wc_all], dtype=np.float64)
    steps = np.linalg.norm(camera_centers[1:] - camera_centers[:-1], axis=1)
    positive_steps = steps[steps > 1e-12]
    if positive_steps.size == 0:
        return T_wc_all, []

    median_step = float(np.median(positive_steps))
    threshold = max(min_jump, median_step * factor)

    stitched = [T.copy() for T in T_wc_all]
    jumps = []
    for i, step in enumerate(steps):
        if step <= threshold:
            continue

        # boundary between i and i+1 has discontinuity
        prev_pose = stitched[i]
        next_pose = stitched[i + 1]
        correction = prev_pose @ np.linalg.inv(next_pose)

        for j in range(i + 1, len(stitched)):
            stitched[j] = correction @ stitched[j]

        jumps.append(i)

        # refresh step sequence after this correction to avoid cascading false alarms
        camera_centers = np.array([T[:3, 3] for T in stitched], dtype=np.float64)
        steps = np.linalg.norm(camera_centers[1:] - camera_centers[:-1], axis=1)

    return stitched, jumps


def align_chunk_boundaries(T_wc_all, chunk_size_src=64):
    """
    Align each chunk boundary (e.g., batch=64) so later chunk poses share the same world frame.

    For boundary b, compute a rigid correction for [b:]
      correction = T_target_b @ inv(T_raw_b)

    where T_target_b is predicted from previous motion using robust fitting (N=8).
    """
    if chunk_size_src <= 0 or len(T_wc_all) <= chunk_size_src:
        return T_wc_all, []

    aligned = [T.copy() for T in T_wc_all]
    boundaries = list(range(chunk_size_src, len(aligned), chunk_size_src))

    for b in boundaries:
        if b >= len(aligned):
            continue
            
        raw_boundary_pose = aligned[b] # This is the start of the new chunk
        
        # Predict where we should be at index b based on 0..b-1
        # We use strict causal history from 'aligned'
        # User Feedback: Do not assume curvature/acceleration (degree=2) as it may overfit complex motion.
        # Trust that internal relative positions are accurate. 
        # Use linear velocity (degree=1) over a robust window (e.g. 12 frames) to stitch the coordinate systems.
        target_boundary_pose = predict_next_pose_robust(aligned[:b], window_size=12, degree=1)
        
        # Correction to bring raw_boundary_pose to target_boundary_pose
        # target = correction @ raw
        # correction = target @ raw_inv
        correction = target_boundary_pose @ np.linalg.inv(raw_boundary_pose)
        
        # Apply to this chunk AND all subsequent frames?
        # Yes, because 'aligned' list is being updated in-place, so next boundary depends on this one.
        for j in range(b, len(aligned)):
            aligned[j] = correction @ aligned[j]

    return aligned, boundaries


def predict_next_pose_robust(pose_history, window_size=12, degree=1):
    """
    Predict next pose based on history using linear extrapolation on translation.
    For rotation, assume constant angular velocity from last two frames.
    """
    n = len(pose_history)
    if n < 2:
        return pose_history[-1]
        
    start_idx = max(0, n - window_size)
    # Use most recent frames
    history = pose_history[start_idx:]
    times = np.arange(len(history))
    next_t = times[-1] + 1
    
    # 1. Translation: Polynomial fit
    trans = np.array([T[:3, 3] for T in history])
    # Prevent rank warning if history is too short for degree
    deg = min(degree, len(history) - 1)
    if deg < 1:
        coeffs = [trans[-1]]
        pred_trans = trans[-1]
    else:
        coeffs = np.polyfit(times, trans, deg=deg)
        pred_trans = np.polyval(coeffs, next_t)
    
    # 2. Rotation: Constant angular velocity
    # R_next = R_last * (R_last * R_prev_inv)
    # Equivalent to applying the last relative rotation again.
    R_last = R.from_matrix(history[-1][:3, :3])
    R_prev = R.from_matrix(history[-2][:3, :3])
    R_rel = R_last * R_prev.inv()
    pred_rot = R_rel * R_last
    
    pred_pose = np.eye(4)
    pred_pose[:3, :3] = pred_rot.as_matrix()
    pred_pose[:3, 3] = pred_trans
    
    return pred_pose


def _ffmpeg_round_near_inf_positive(numer: int, denom: int) -> int:
    """ffmpeg AV_ROUND_NEAR_INF style for non-negative values: floor((numer/denom)+0.5)."""
    return (numer + (denom // 2)) // denom


def load_smoothed_poses_for_16fps(poses_dir, num_frames_16fps, src_fps=30, target_fps=16):
    """
    Load smoothed extrinsics and align to 16fps video frame indices.
    """
    files = sorted(glob.glob(os.path.join(poses_dir, "extrinsics_*.npy")))
    num_frames_30fps = len(files)
    print(f"Found {num_frames_30fps} smoothed 30fps pose files.")
    
    if num_frames_30fps == 0:
        raise FileNotFoundError(f"No extrinsics found in {poses_dir}")

    if num_frames_16fps <= 0:
        raise ValueError(f"Invalid target frame count: {num_frames_16fps}")

    pose_list_gl = []

    print("Loading and converting poses...")
    # Load all poses as T_wc first, then stitch potential discontinuities.
    T_wc_all = []
    for f in files:
        ext = np.load(f)
        T_wc_all.append(np.linalg.inv(as_4x4(ext)))

    # 1) Explicitly align post-64 chunk(s) to previous chunk world frame.
    T_wc_all, chunk_boundaries = align_chunk_boundaries(T_wc_all, chunk_size_src=64)
    if chunk_boundaries:
        print(f"[Chunk Align] Applied boundary alignment at source indices: {chunk_boundaries}")

    # 2) Safety net: still stitch any residual large discontinuities.
    T_wc_all, jump_indices = stitch_pose_jumps(T_wc_all)
    if jump_indices:
        jump_pairs = [(i, i + 1) for i in jump_indices]
        print(f"[Pose Stitch] Residual discontinuity boundaries: {jump_pairs}")
    else:
        print("[Pose Stitch] No discontinuity detected.")

    # Keep original target frame count from video; only fix pose-frame alignment.
    src_ratio = Fraction(src_fps).limit_denominator()
    tgt_ratio = Fraction(target_fps).limit_denominator()
    if tgt_ratio <= 0:
        raise ValueError(f"Invalid target_fps: {target_fps}")

    num_src_effective = len(T_wc_all)

    # Index mapping (ffmpeg fps filter style timeline sampling)
    indices = []
    if num_frames_16fps == 1 or num_src_effective == 1:
        indices = [0] * num_frames_16fps
    else:
        # idx_src = round_near_inf(k * src_fps / target_fps)
        # Use integer math:
        #   k * (src_num/src_den) / (tgt_num/tgt_den)
        # = k * src_num * tgt_den / (src_den * tgt_num)
        ratio_num = src_ratio.numerator * tgt_ratio.denominator
        ratio_den = src_ratio.denominator * tgt_ratio.numerator
        for k in range(num_frames_16fps):
            idx30 = _ffmpeg_round_near_inf_positive(k * ratio_num, ratio_den)
            idx30 = min(max(idx30, 0), num_src_effective - 1)
            indices.append(idx30)

    # Preview mapping
    preview_n = min(10, len(indices))
    print(f"Pose index mapping preview (target->source): {list(enumerate(indices[:preview_n]))}")

    # Pose 0 defines world frame for first-frame point cloud
    T_wc_0 = T_wc_all[0]

    # Load selected poses from stitched trajectory
    for i in indices:
        T_wc_k = T_wc_all[i]

        # Convert to GL Camera Frame (-Z fwd, +Y up)
        T_gl = T_wc_k @ OPENCV_TO_OPENGL
        pose_list_gl.append(T_gl)
        
    return pose_list_gl, T_wc_0


def main():
    parser = argparse.ArgumentParser(description="Render 16fps aligned video with smoothed camera")
    parser.add_argument("--video_path", type=str, required=True, help="16fps processed video path")
    parser.add_argument("--pose_dir", type=str, required=True, help="Smoothed extrinsics directory")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Original intrinsics file (frame 0)")
    parser.add_argument("--rgb_path", type=str, required=True, help="Original RGB image (hand_inpaint.png)")
    parser.add_argument("--color_video_path", type=str, default=None, help="Original video path used for point-cloud colors (first frame)")
    parser.add_argument("--depth_path", type=str, required=True, help="Generated depth (depth_first_frame.npy)")
    parser.add_argument("--output_video", type=str, default="rendered_16fps.mp4", help="Output render path")
    parser.add_argument("--overlay_video", type=str, default="rendered_16fps_overlay.mp4", help="Output overlay path")
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--mask_mode", action="store_true", help="Render mask-style video (black points, white background)")
    
    args = parser.parse_args()
    
    # 1. Get Video Duration
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if video_fps <= 0:
        video_fps = float(args.fps)
    print(f"Video Info: {video_w}x{video_h}, {total_frames} frames, fps={video_fps:.4f}")
    
    # 2. Get Aligned Poses
    # Strictly follow the preprocessing script: source 30fps -> fps=16
    camera_poses_gl, T_wc_0 = load_smoothed_poses_for_16fps(
        args.pose_dir,
        total_frames,
        src_fps=30,
        target_fps=args.fps
    )
    
    # 3. Load Intrinsics and Adapt
    print(f"Loading intrinsics: {args.intrinsics_path}")
    K_orig = np.load(args.intrinsics_path)

    # 4. Load Depth/RGB and infer ORIGINAL resolution robustly
    # In predict_multi_gpu.py, depth is always resized back to orig_hw when saved.
    print(f"Loading depth: {args.depth_path}")
    depth = load_depth_npy(args.depth_path)
    orig_h, orig_w = int(depth.shape[0]), int(depth.shape[1])
    print(f"Inferred original resolution from depth: {orig_w}x{orig_h}")

    # Point colors prefer the first frame of the source video when available.
    if args.color_video_path is not None and os.path.exists(args.color_video_path):
        print(f"Loading color from original video first frame: {args.color_video_path}")
        rgb = load_first_frame_from_video(args.color_video_path)
    else:
        print(f"Loading color fallback RGB image: {args.rgb_path}")
        rgb = load_rgb_rgb(args.rgb_path)
    H_rgb, W_rgb = rgb.shape[:2]
    if (H_rgb, W_rgb) != (orig_h, orig_w):
        print(
            f"Warning: RGB resolution ({W_rgb}x{H_rgb}) != Depth resolution ({orig_w}x{orig_h}). "
            "Will resize RGB to match depth for point colors."
        )

    # 5. Adapt intrinsics to match the 16fps processed video (crop+resize)
    K_new = get_aligned_intrinsics(K_orig, orig_w, orig_h, target_w=video_w, target_h=video_h)

    # 6. Create Point Cloud in World Frame (Smoothed Frame 0)
    points_c0, colors = create_point_cloud_from_rgbd(depth, rgb, K_orig)
    
    # Transform to World
    # P_w = T_wc_0 @ P_c0
    # Note: T_wc_0 is OpenCV convention. P_c0 is OpenCV convention.
    points_c0_h = np.hstack([points_c0, np.ones((len(points_c0), 1))])
    points_world = (T_wc_0 @ points_c0_h.T).T[:, :3]
    
    # 7. Render
    temp_dir = Path(args.output_video).parent / "temp_frames_render"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting Optimized Render & Overlay...")
    render_video_with_pyrender(
        points_world=points_world,
        colors=colors,
        camera_transforms=camera_poses_gl,
        intrinsics=K_new,
        image_width=video_w,
        image_height=video_h,
        frame_stride=1,
        render_frames=total_frames,
        total_frames=total_frames,
        temp_dir=temp_dir,
        fps=args.fps,
        output_video=args.output_video,
        point_size=args.point_size,
        mask_mode=args.mask_mode,
        overlay_video_path=args.overlay_video,
        original_video_path=args.video_path
    )
    
    # 8. Overlay - Deprecated independent call
    # print("Generating Overlay...")
    # render_overlay_video(...)
    
    # Cleanup
    # if temp_dir.exists():
    #     shutil.rmtree(temp_dir)
        
    print("Done!")

if __name__ == "__main__":
    main()
