#!/usr/bin/env python3
# Copyright (c) jiamingda (https://github.com/Luyitas)
'''
Run DA3 on the inpainted first frame (hand_inpaint.png) for a single clip folder.
Writes depth / intrinsics / point cloud back into that folder.

Single-clip example (pipeline Step 01c):
CUDA_VISIBLE_DEVICES=0 python process_depth_inpainted.py \
    --clip_dir /path/to/_proc/inpainted/<clip_name> \
    --model_path /path/to/DA3NESTED-GIANT-LARGE-1.1
'''

import os
import sys
import argparse
import logging
import time
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# Allow importing src.* when run from Depth-Anything-3 subdirs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def list_video_folders(input_root: Path) -> list[str]:
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {input_root}")
    return sorted(str(p) for p in input_root.iterdir() if p.is_dir())


def process_single_folder(
    folder_path: Path,
    model,
    device: torch.device,
    temp_dir: Path,
    output_name: str,
    intrinsics_output_name: str,
    pointcloud_output_name: str,
    process_res: int,
) -> tuple[bool, str, float]:
    start_time = time.time()

    try:
        image_candidates = [
            folder_path / "hand_inpaint.png",
            folder_path / "robot_arm_inpaint.png",
        ]
        image_path = None
        for candidate in image_candidates:
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            return False, f"Missing inpainted image (hand_inpaint.png / robot_arm_inpaint.png): {folder_path}", 0.0

        output_path = folder_path / output_name
        intrinsics_output_path = folder_path / intrinsics_output_name
        pointcloud_output_path = folder_path / pointcloud_output_name

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return False, f"Cannot read image: {image_path.name}", 0.0

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        with torch.no_grad():
            pred = model.inference(
                image=[rgb],
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
            )

        depth = pred.depth[0].astype(np.float32)
        if pred.intrinsics is None:
            return False, "DA3 did not return intrinsics", 0.0
        intrinsics = pred.intrinsics[0].astype(np.float32)
        proc_h, proc_w = pred.processed_images.shape[1:3]

        # Scale intrinsics from process_res back to original image size (match resized depth).
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)
        intrinsics_orig = intrinsics.copy()
        intrinsics_orig[0, 0] *= scale_x
        intrinsics_orig[1, 1] *= scale_y
        intrinsics_orig[0, 2] *= scale_x
        intrinsics_orig[1, 2] *= scale_y

        if (proc_h, proc_w) != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Back-project depth at full resolution to camera-frame point cloud.
        yy, xx = np.meshgrid(np.arange(orig_h), np.arange(orig_w), indexing="ij")
        fx, fy = intrinsics_orig[0, 0], intrinsics_orig[1, 1]
        cx, cy = intrinsics_orig[0, 2], intrinsics_orig[1, 2]
        z = depth
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

        ts = int(time.time() * 1e6)
        tmp_depth_path = temp_dir / f"tmp_{folder_path.name}_{ts}_depth.npy"
        tmp_intr_path = temp_dir / f"tmp_{folder_path.name}_{ts}_intrinsics.npy"
        tmp_cloud_path = temp_dir / f"tmp_{folder_path.name}_{ts}_cloud.npy"

        np.save(str(tmp_depth_path), depth)
        np.save(str(tmp_intr_path), intrinsics_orig)
        np.save(str(tmp_cloud_path), points)

        shutil.move(str(tmp_depth_path), str(output_path))
        shutil.move(str(tmp_intr_path), str(intrinsics_output_path))
        shutil.move(str(tmp_cloud_path), str(pointcloud_output_path))

        elapsed = time.time() - start_time
        return True, f"ok ({depth.shape[1]}x{depth.shape[0]}) depth+intrinsics+cloud", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, str(e), elapsed


def main():
    parser = argparse.ArgumentParser(description="First-frame depth from one inpainted clip folder")
    parser.add_argument(
        "--clip_dir",
        type=str,
        default=None,
        help="Single clip folder containing hand_inpaint.png (preferred for Step 01c).",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="[Legacy] Root containing one subfolder per clip.",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default=None,
        help="[Legacy] Text file with one clip folder path per line.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="DA3 model directory (from_pretrained).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (0-based, inclusive).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (0-based, exclusive); default: all.",
    )
    parser.add_argument("--gpu_rank", type=int, default=0, help="[Legacy batch] GPU rank index")
    parser.add_argument("--world_size", type=int, default=1, help="[Legacy batch] Total GPU count")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Torch cuda:0 index inside this process (use with CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument("--log_dir", type=str, default="./depth_logs_resume", help="Log directory")
    parser.add_argument(
        "--output_name",
        type=str,
        default="depth_first_frame.npy",
        help="Output depth filename inside each clip folder.",
    )
    parser.add_argument(
        "--intrinsics_output_name",
        type=str,
        default="intrinsics_first_frame.npy",
        help="Output intrinsics filename inside each clip folder.",
    )
    parser.add_argument(
        "--pointcloud_output_name",
        type=str,
        default="cloud_first_frame.npy",
        help="Output point-cloud filename (N×3 float32, camera frame).",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="DA3 process_res (keep consistent with pred_multi_gpu_2.py).",
    )
    
    args = parser.parse_args()

    if args.clip_dir is None and args.task_file is None and args.input_root is None:
        raise SystemExit("Provide --clip_dir (single clip) or --task_file / --input_root (legacy batch)")

    # Suppress INFO logs.
    logging.getLogger().setLevel(logging.WARNING)

    rank = args.gpu_rank
    world_size = args.world_size

    # Device.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    print(f"[Step 01c] device: {device}", flush=True)
    if args.clip_dir is None:
        print(f"[Step 01c] index range: [{args.start_idx}:{args.end_idx}] (half-open)", flush=True)

    print(f"[Step 01c] loading DA3: {args.model_path}", flush=True)
    from src.depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(args.model_path).to(device)
    model.eval()
    print(f"[Step 01c] model ready.", flush=True)

    # Logs / temp dir.
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = log_dir / f"temp_gpu{rank}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_success_path = log_dir / f"success_gpu{rank}.txt"
    log_fail_path = log_dir / f"failed_gpu{rank}.txt"

    if args.clip_dir is not None:
        my_folders = [str(Path(args.clip_dir).resolve())]
        print(f"[Step 01c] clip_dir: {my_folders[0]}", flush=True)
    elif args.task_file is not None:
        print(f"[Step 01c] task file: {args.task_file}", flush=True)
        with open(args.task_file, "r") as f:
            all_tasks = [line.strip() for line in f if line.strip()]
        end_idx = args.end_idx if args.end_idx is not None else len(all_tasks)
        sub_tasks = all_tasks[args.start_idx:end_idx]
        my_folders = [p for i, p in enumerate(sub_tasks) if i % world_size == rank]
        print(f"[Step 01c] tasks: {len(sub_tasks)}, assigned: {len(my_folders)}", flush=True)
    else:
        input_root = Path(args.input_root)
        print(f"[Step 01c] scanning: {input_root}", flush=True)
        all_tasks = list_video_folders(input_root)
        end_idx = args.end_idx if args.end_idx is not None else len(all_tasks)
        sub_tasks = all_tasks[args.start_idx:end_idx]
        my_folders = [p for i, p in enumerate(sub_tasks) if i % world_size == rank]
        print(f"[Step 01c] tasks: {len(sub_tasks)}, assigned: {len(my_folders)}", flush=True)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_time = 0.0
    
    # Success/fail logs.
    with open(log_success_path, 'a') as log_success, \
         open(log_fail_path, 'a') as log_fail:
        
        with tqdm(
            total=len(my_folders),
            desc=f"GPU{rank}",
            position=rank,
            leave=True,
            ncols=100,
            file=sys.stderr,
        ) as pbar:
            for folder in my_folders:
                folder_path = Path(folder)
                success, msg, elapsed = process_single_folder(
                    folder_path=folder_path,
                    model=model,
                    device=device,
                    temp_dir=temp_dir,
                    output_name=args.output_name,
                    intrinsics_output_name=args.intrinsics_output_name,
                    pointcloud_output_name=args.pointcloud_output_name,
                    process_res=args.process_res,
                )

                if success:
                    if "already exists" in msg:
                        skip_count += 1
                    else:
                        success_count += 1
                        total_time += elapsed
                    log_success.write(
                        f"{folder_path.name}\t{folder}\t"
                        f"{folder_path / args.output_name}\t"
                        f"{folder_path / args.intrinsics_output_name}\t"
                        f"{folder_path / args.pointcloud_output_name}\n"
                    )
                    log_success.flush()
                else:
                    fail_count += 1
                    log_fail.write(f"{folder_path.name}\t{folder}\t{msg}\n")
                    log_fail.flush()

                avg_time = total_time / success_count if success_count > 0 else 0
                pbar.set_postfix(
                    {"✓": success_count, "⊙": skip_count, "✗": fail_count, "avg": f"{avg_time:.1f}s"},
                    refresh=True,
                )
                pbar.update(1)
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print(f"\n[Step 01c] done.", flush=True)
    print(f"[Step 01c] ok={success_count}, skip={skip_count}, fail={fail_count}", flush=True)
    if success_count > 0:
        print(f"[Step 01c] avg: {total_time/success_count:.2f}s/clip", flush=True)
    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

