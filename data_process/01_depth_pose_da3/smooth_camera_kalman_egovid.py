#!/usr/bin/env python3
# Copyright (c) jiamingda (https://github.com/Luyitas)
"""
Kalman smoothing of DA3 camera parameters for a single clip (pipeline Step 01b).

Also supports legacy batch mode over many clip folders under --input_dir.
"""

import os
import re
import shutil
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import inv
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import gc
import time
from tqdm import tqdm

# ===================== Outlier frame repair =====================
def detect_and_fix_outliers(poses_raw, trans_thresh=0.1, rot_thresh=5.0):
    """
    Detect / fix outlier frames (works on older scipy).
    """
    poses_fixed = [p.copy() for p in poses_raw]
    n_frames = len(poses_raw)
    if n_frames <= 3:
        return poses_fixed

    for i in range(2, n_frames-2):
        t_curr = poses_raw[i][:3, 3]
        t_prev = poses_raw[i-1][:3, 3]
        trans_diff_prev = np.linalg.norm(t_curr - t_prev)

        rot_curr = Rot.from_matrix(poses_raw[i][:3, :3])
        rot_prev = Rot.from_matrix(poses_raw[i-1][:3, :3])
        rot_diff_prev = (rot_curr.inv() * rot_prev).magnitude() * 180 / np.pi

        t_next = poses_raw[i+1][:3, 3]
        trans_diff_next = np.linalg.norm(t_curr - t_next)
        rot_next = Rot.from_matrix(poses_raw[i+1][:3, :3])
        rot_diff_next = (rot_curr.inv() * rot_next).magnitude() * 180 / np.pi

        is_outlier = (trans_diff_prev > trans_thresh and trans_diff_next > trans_thresh) or \
                     (rot_diff_prev > rot_thresh and rot_diff_next > rot_thresh)
        
        if not is_outlier:
            continue

        # Interpolate outlier pose from neighboring frames.
        t_ref = [poses_raw[j][:3, 3] for j in range(i-1, i+2) if j != i]
        t_fixed = np.mean(t_ref, axis=0)
        
        rot_ref = [Rot.from_matrix(poses_raw[j][:3, :3]) for j in range(i-2, i+3) if j != i]
        rot_quats = [r.as_quat() for r in rot_ref]
        rot_quat_mean = np.mean(rot_quats, axis=0)
        rot_quat_mean = rot_quat_mean / np.linalg.norm(rot_quat_mean)
        rot_fixed = Rot.from_quat(rot_quat_mean)
        
        poses_fixed[i][:3, 3] = t_fixed
        poses_fixed[i][:3, :3] = rot_fixed.as_matrix()

    return poses_fixed


# ===================== Weak-prediction Kalman smoother =====================
class PoseKalmanFilter:
    """
    Trust measurements more than prediction to limit drift.
    """
    def __init__(self, dt=1.0, process_noise=1e-4, measurement_noise=1e-3):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.x_pos = np.zeros(6)
        self.A_pos = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H_pos = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.Q_pos = np.eye(6) * process_noise
        self.R_pos = np.eye(3) * measurement_noise
        self.P_pos = np.eye(6) * 0.1

        self.x_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.Q_rot = np.eye(4) * process_noise * 0.2
        self.R_rot = np.eye(4) * measurement_noise
        self.P_rot = np.eye(4) * 0.1

    def update_pos(self, z_pos):
        x_pred = self.A_pos @ self.x_pos
        P_pred = self.A_pos @ self.P_pos @ self.A_pos.T + self.Q_pos
        y = z_pos - self.H_pos @ x_pred
        S = self.H_pos @ P_pred @ self.H_pos.T + self.R_pos
        K = P_pred @ self.H_pos.T @ inv(S)
        self.x_pos = x_pred + K @ y
        self.P_pos = (np.eye(6) - K @ self.H_pos) @ P_pred
        return self.x_pos[:3]

    def update_rot(self, z_rot_quat):
        x_pred = self.x_rot.copy()
        P_pred = self.P_rot + self.Q_rot
        y = z_rot_quat - x_pred
        S = P_pred + self.R_rot
        K = P_pred @ inv(S)
        self.x_rot = x_pred + K @ y
        self.x_rot = self.x_rot / np.linalg.norm(self.x_rot)
        self.P_rot = (np.eye(4) - K) @ P_pred
        return self.x_rot

    def smooth_pose(self, pose_mat):
        t = pose_mat[:3, 3]
        rot_mat = pose_mat[:3, :3]
        rot_quat = Rot.from_matrix(rot_mat).as_quat()
        rot_quat = np.roll(rot_quat, 1)

        t_smoothed = self.update_pos(t)
        rot_quat_smoothed = self.update_rot(rot_quat)

        rot_mat_smoothed = Rot.from_quat(np.roll(rot_quat_smoothed, -1)).as_matrix()
        pose_smoothed = np.eye(4)
        pose_smoothed[:3, :3] = rot_mat_smoothed
        pose_smoothed[:3, 3] = t_smoothed
        return pose_smoothed


# ===================== EgoVid filenames (6-digit padding) =====================
def extract_extrinsics_idx_egovid(name):
    """
    EgoVid pattern: extrinsics_000000.npy (six-digit index).
    """
    match = re.match(r'^extrinsics_(\d{6})\.npy$', name)
    if match:
        return int(match.group(1))
    return None


def _is_target_file_egovid(name, prefix):
    """
    EgoVid pattern: prefix_000000.npy.
    """
    pattern = rf'^{prefix}_(\d{{6}})\.npy$'
    return bool(re.match(pattern, name))


# ===================== Single-clip processing =====================
def process_single_clip_egovid(clip_name, root_input_dir, root_output_dir, 
                               fps=30, process_noise=1e-4, measurement_noise=1e-3,
                               trans_thresh=0.1, rot_thresh=5.0,
                               skip_existing=True, copy_depth=False, copy_intrinsics=True):
    """
    Process one clip -> (clip_name, success, message).
    
    Args:
        skip_existing: skip if outputs already complete (unused shortcut disabled below).
        copy_depth: copy depth .npy files (large; optional).
        copy_intrinsics: copy intrinsics .npy files.
    """
    input_clip_dir = os.path.join(root_input_dir, clip_name)
    output_clip_dir = os.path.join(root_output_dir, clip_name)
    
    try:
        # (Disabled) resume shortcut was: output dir + extrinsics_000000.npy

        if not os.path.isdir(input_clip_dir):
            return (clip_name, False, "input dir not found")

        os.makedirs(output_clip_dir, exist_ok=True)

        # Prefer os.scandir over listdir.
        extrinsics_files = []
        intrinsics_files = []
        depth_files = []
        other_files = []
        
        with os.scandir(input_clip_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.endswith('.npy'):
                    # Track non-.npy sidecars (e.g. summary.txt).
                    other_files.append(name)
                    continue
                    
                idx = extract_extrinsics_idx_egovid(name)
                if idx is not None:
                    extrinsics_files.append((idx, name))
                elif _is_target_file_egovid(name, 'intrinsics'):
                    intrinsics_files.append(name)
                elif _is_target_file_egovid(name, 'depth'):
                    depth_files.append(name)

        if len(extrinsics_files) == 0:
            return (clip_name, False, "no extrinsics files found")

        # Sort frames by extrinsics index.
        extrinsics_files.sort(key=lambda x: x[0])
        file_names = [f for _, f in extrinsics_files]

        # Copy intrinsics (optional).
        if copy_intrinsics:
            for f in intrinsics_files:
                shutil.copy2(os.path.join(input_clip_dir, f), os.path.join(output_clip_dir, f))

        # Copy depth (optional, large).
        if copy_depth:
            for f in depth_files:
                shutil.copy2(os.path.join(input_clip_dir, f), os.path.join(output_clip_dir, f))
        
        # Copy remaining sidecars.
        for f in other_files:
            src = os.path.join(input_clip_dir, f)
            dst = os.path.join(output_clip_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Load raw extrinsics.
        poses_raw = []
        for f in file_names:
            pose = np.load(os.path.join(input_clip_dir, f))
            if pose.shape != (4, 4):
                poses_raw.append(np.eye(4))
            else:
                poses_raw.append(pose)

        # Repair temporal outliers.
        poses_fixed = detect_and_fix_outliers(poses_raw, trans_thresh, rot_thresh)

        # Kalman smoothing pass.
        dt = 1.0 / fps
        kf = PoseKalmanFilter(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
        poses_smoothed = []
        for pose_fixed in poses_fixed:
            pose_smoothed = kf.smooth_pose(pose_fixed)
            poses_smoothed.append(pose_smoothed)

        # Write smoothed poses.
        for i, f in enumerate(file_names):
            output_path = os.path.join(output_clip_dir, f)
            np.save(output_path, poses_smoothed[i])

        # Release arrays.
        del poses_raw, poses_fixed, poses_smoothed
        
        return (clip_name, True, "success")

    except Exception as e:
        return (clip_name, False, f"error: {str(e)}")


# ===================== Progress logger =====================
class ProgressLogger:
    def __init__(self, log_file, total_count=None):
        self.log_file = log_file
        self.total_count = total_count
        self.processed = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        
    def log(self, clip_name, success, message, pbar=None):
        self.processed += 1
        if success:
            if "skipped" in message:
                self.skipped += 1
            else:
                self.success += 1
        else:
            self.failed += 1
        
        # tqdm postfix.
        if pbar is not None:
            pbar.set_postfix({
                'success': self.success,
                'failed': self.failed,
                'skipped': self.skipped
            }, refresh=False)
            
        # Log failed clips.
        if not success and "skipped" not in message:
            with open(self.log_file, 'a') as f:
                f.write(f"{clip_name}\t{message}\n")


# ===================== Iterate clip subfolders =====================
def iter_clip_dirs(root_dir, start_idx=None, end_idx=None):
    """
    Yield subdirectory names with optional [start_idx, end_idx) window.
    
    Args:
        root_dir: dataset root.
        start_idx: first directory index (inclusive); None -> 0.
        end_idx: past-last index; None -> no upper bound.
    """
    with os.scandir(root_dir) as entries:
        idx = 0
        for entry in entries:
            if entry.is_dir():
                if start_idx is not None and idx < start_idx:
                    idx += 1
                    continue
                if end_idx is not None and idx >= end_idx:
                    break
                yield entry.name
                idx += 1


def get_sorted_clip_dirs(root_dir, start_idx=None, end_idx=None):
    """
    Sorted clip folder names (stable sharding across machines).
    
    Args:
        root_dir: dataset root.
        start_idx: first index (inclusive).
        end_idx: past-last index.
    
    Returns:
        (names_in_range, total_folder_count_before_slice).
    """
    print("Listing and sorting clip folders...")
    all_dirs = []
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                all_dirs.append(entry.name)
    
    # Deterministic sort for multi-node splits.
    all_dirs.sort()
    total = len(all_dirs)
    print(f"Found {total} clip folders")
    
    # Slice to requested window.
    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else total
    
    # Clamp indices.
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    
    selected = all_dirs[start:end]
    print(f"Range [{start}, {end}): {len(selected)} clips selected")
    
    # Drop large name list.
    del all_dirs
    gc.collect()
    
    return selected, total


def count_clip_dirs(root_dir):
    """
    Count child folders.
    """
    count = 0
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                count += 1
    return count


# ===================== Parallel batch processing =====================
def batch_process_clips_parallel(root_input_dir, root_output_dir, 
                                 fps=30, process_noise=1e-4, measurement_noise=1e-3,
                                 trans_thresh=0.1, rot_thresh=5.0,
                                 num_workers=4, skip_existing=True,
                                 copy_depth=False, copy_intrinsics=True,
                                 batch_size=1000,
                                 start_idx=None, end_idx=None):
    """
    Process clips with a thread pool.
    
    Args:
        num_workers: concurrent workers.
        batch_size: futures scheduled per round.
        start_idx / end_idx: clip index window for multi-node jobs.
    """
    if not os.path.isdir(root_input_dir):
        raise ValueError(f"Input root does not exist: {root_input_dir}")
    os.makedirs(root_output_dir, exist_ok=True)

    # Sorted list for index window.
    clip_dirs, total_all = get_sorted_clip_dirs(root_input_dir, start_idx, end_idx)
    total_count = len(clip_dirs)
    
    if total_count == 0:
        print("No clips to process")
        return

    # Error log encodes slice in filename.
    range_str = f"_{start_idx}_{end_idx}" if start_idx is not None or end_idx is not None else ""
    log_file = os.path.join(root_output_dir, f"process_errors{range_str}.log")
    logger = ProgressLogger(log_file, total_count)

    # Bind kwargs via partial.
    process_func = partial(
        process_single_clip_egovid,
        root_input_dir=root_input_dir,
        root_output_dir=root_output_dir,
        fps=fps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        trans_thresh=trans_thresh,
        rot_thresh=rot_thresh,
        skip_existing=skip_existing,
        copy_depth=copy_depth,
        copy_intrinsics=copy_intrinsics
    )

    print(f"\nBatch start ({num_workers} workers)...")
    print(f"Input root:  {root_input_dir}")
    print(f"Output root: {root_output_dir}")
    print(f"Window: [{start_idx if start_idx else 0}, {end_idx if end_idx else total_all}) / {total_all}")
    print(f"This run: {total_count} clips")
    print(f"skip_existing: {skip_existing}")
    print(f"copy_depth: {copy_depth}")
    print(f"copy_intrinsics: {copy_intrinsics}\n")

    # Heuristic Pool chunksize.
    # Larger chunksize => fewer IPC round-trips.
    chunksize = max(50, total_count // (num_workers * 20))
    chunksize = min(chunksize, 200)  # Cap so tqdm still updates.
    print(f"chunksize: {chunksize}")
    
    # ThreadPool for I/O-heavy per-clip work.
    pbar = tqdm(total=total_count, desc="Progress", unit="clip", 
                dynamic_ncols=True, smoothing=0.1)
    
    # Schedule futures in chunks.
    batch_size = num_workers * 50  # futures per chunk
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch_clips = clip_dirs[batch_start:batch_end]
            
            # Submit chunk.
            futures = {executor.submit(process_func, clip_name): clip_name 
                      for clip_name in batch_clips}
            
            # Drain futures as they finish.
            for future in as_completed(futures):
                clip_name, success, message = future.result()
                logger.log(clip_name, success, message, pbar=pbar)
                pbar.update(1)
            
            # GC between chunks.
            del futures
            gc.collect()
    
    pbar.close()

    print(f"\nBatch finished.")
    print(f"- Total clips:     {total_all}")
    print(f"- Window:          [{start_idx if start_idx else 0}, {end_idx if end_idx else total_all})")
    print(f"- This run:        {total_count}")
    print(f"- Succeeded:       {logger.success}")
    print(f"- Skipped:         {logger.skipped}")
    print(f"- Failed:          {logger.failed}")
    print(f"- Error log:       {log_file}")
    print(f"- Output root:     {root_output_dir}")


# ===================== Single-process (debug) =====================
def batch_process_clips_single(root_input_dir, root_output_dir, 
                               fps=30, process_noise=1e-4, measurement_noise=1e-3,
                               trans_thresh=0.1, rot_thresh=5.0,
                               skip_existing=True, copy_depth=False, copy_intrinsics=True,
                               start_idx=None, end_idx=None):
    """
    Single-process driver for debugging.
    
    Args:
        start_idx / end_idx: half-open clip index range.
    """
    if not os.path.isdir(root_input_dir):
        raise ValueError(f"Input root does not exist: {root_input_dir}")
    os.makedirs(root_output_dir, exist_ok=True)

    # Sorted clip list
    clip_dirs, total_all = get_sorted_clip_dirs(root_input_dir, start_idx, end_idx)
    total_count = len(clip_dirs)
    
    if total_count == 0:
        print("No clips to process")
        return

    range_str = f"_{start_idx}_{end_idx}" if start_idx is not None or end_idx is not None else ""
    log_file = os.path.join(root_output_dir, f"process_errors{range_str}.log")
    logger = ProgressLogger(log_file, total_count)

    print(f"\nSingle-process batch...")
    print(f"Window: [{start_idx if start_idx else 0}, {end_idx if end_idx else total_all}) / {total_all}")
    print(f"Clips this run: {total_count}\n")

    # tqdm over clip list.
    pbar = tqdm(clip_dirs, desc="Progress", unit="clip", 
                dynamic_ncols=True, smoothing=0.1)
    
    for clip_name in pbar:
        result_clip_name, success, message = process_single_clip_egovid(
            clip_name=clip_name,
            root_input_dir=root_input_dir,
            root_output_dir=root_output_dir,
            fps=fps,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            trans_thresh=trans_thresh,
            rot_thresh=rot_thresh,
            skip_existing=skip_existing,
            copy_depth=copy_depth,
            copy_intrinsics=copy_intrinsics
        )
        logger.log(result_clip_name, success, message, pbar=pbar)
        
        if logger.processed % 1000 == 0:
            gc.collect()
    
    pbar.close()

    print(f"\nBatch finished.")
    print(f"- Total clips:     {total_all}")
    print(f"- This run:        {total_count}")
    print(f"- Succeeded:       {logger.success}")
    print(f"- Skipped:         {logger.skipped}")
    print(f"- Failed:          {logger.failed}")


# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalman smoothing of DA3 camera trajectories")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input root (each clip subdir has extrinsics_*.npy)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root")
    parser.add_argument("--clip_name", type=str, default=None,
                        help="Process one clip subfolder (preferred for single-clip annotation)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--process_noise", type=float, default=1e-4, help="Kalman process noise")
    parser.add_argument("--measurement_noise", type=float, default=1e-3, help="Kalman measurement noise")
    parser.add_argument("--trans_thresh", type=float, default=0.1, help="Translation outlier threshold (m)")
    parser.add_argument("--rot_thresh", type=float, default=50.0, help="Rotation outlier threshold (deg)")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--batch_size", type=int, default=1000, help="Clip batch size (scheduling)")
    parser.add_argument("--skip_existing", action="store_true", default=True, 
                        help="Skip clips with existing outputs")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="Re-process even if output exists")
    parser.add_argument("--copy_depth", action="store_true", default=False,
                        help="Copy depth .npy (large; off by default)")
    parser.add_argument("--copy_intrinsics", action="store_true", default=True,
                        help="Copy intrinsics .npy")
    parser.add_argument("--single_process", action="store_true", default=False,
                        help="Single-process mode (debug)")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start clip index (inclusive); multi-node sharding")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End clip index (exclusive)")
    
    args = parser.parse_args()

    if args.clip_name:
        result_clip_name, success, message = process_single_clip_egovid(
            clip_name=args.clip_name,
            root_input_dir=args.input_dir,
            root_output_dir=args.output_dir,
            fps=args.fps,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            trans_thresh=args.trans_thresh,
            rot_thresh=args.rot_thresh,
            skip_existing=args.skip_existing,
            copy_depth=args.copy_depth,
            copy_intrinsics=args.copy_intrinsics,
        )
        if not success:
            raise SystemExit(f"[Step 01b] Failed ({result_clip_name}): {message}")
        print(f"[Step 01b] Done ({result_clip_name}): {message}")
    elif args.single_process:
        batch_process_clips_single(
            root_input_dir=args.input_dir,
            root_output_dir=args.output_dir,
            fps=args.fps,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            trans_thresh=args.trans_thresh,
            rot_thresh=args.rot_thresh,
            skip_existing=args.skip_existing,
            copy_depth=args.copy_depth,
            copy_intrinsics=args.copy_intrinsics,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
    else:
        batch_process_clips_parallel(
            root_input_dir=args.input_dir,
            root_output_dir=args.output_dir,
            fps=args.fps,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            trans_thresh=args.trans_thresh,
            rot_thresh=args.rot_thresh,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            copy_depth=args.copy_depth,
            copy_intrinsics=args.copy_intrinsics,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
