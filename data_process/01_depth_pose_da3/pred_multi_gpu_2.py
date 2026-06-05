# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
DA3 depth + camera prediction for a single clip (pipeline Step 01a).

Usage:
  CUDA_VISIBLE_DEVICES=0 python pred_multi_gpu_2.py \\
      --video_path /path/to/video_16fps.mp4 \\
      --clip_name my_clip \\
      --output_root /path/to/poses_da3 \\
      --model_path /path/to/DA3NESTED-GIANT-LARGE-1.1
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
import cv2
import argparse
import os
import sys
import time
import logging

logging.getLogger().setLevel(logging.WARNING)
for name in list(logging.root.manager.loggerDict.keys()):
    logging.getLogger(name).setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="DA3 depth + camera prediction for one clip")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to the clip video (preferred for single-clip annotation)")
    parser.add_argument("--prepared_list", type=str, default=None,
                        help="Text file with one video path per line (legacy batch helper)")
    parser.add_argument("--clip_name", type=str, default=None,
                        help="Output subfolder name under output_root (default: video stem)")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_check", action="store_true")
    args = parser.parse_args()

    if args.video_path:
        all_videos = [args.video_path]
    elif args.prepared_list:
        with open(args.prepared_list) as f:
            all_videos = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit("Provide --video_path or --prepared_list")

    if len(all_videos) != 1:
        raise SystemExit(
            f"Single-clip mode expects exactly one video, got {len(all_videos)}. "
            "Pass --video_path for one clip."
        )

    print(f"[Step 01a] Started, PID={os.getpid()}", flush=True)
    
    
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    import numpy as np
    import torch
    
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES is set, so use device 0
    device = torch.device("cuda:0")
    
    print(f"[Step 01a] Device: {torch.cuda.get_device_name(0)}", flush=True)

    video_path = all_videos[0]
    clip_name = args.clip_name or os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(args.output_root, clip_name)

    if not args.skip_check and os.path.exists(os.path.join(output_dir, "summary.txt")):
        print(f"[Step 01a] Already done: {output_dir}", flush=True)
        return

    pending = [video_path]
    print(f"[Step 01a] Clip: {clip_name}", flush=True)
    print(f"[Step 01a] Video: {video_path}", flush=True)
    
    print(f"[Step 01a] Loading model...", flush=True)
    from src.depth_anything_3.api import DepthAnything3
    from src.depth_anything_3.utils.export.glb import _as_homogeneous44
    
    model = DepthAnything3.from_pretrained(args.model_path).to(device)
    model.eval()
    
    print(f"[Step 01a] Processing video...", flush=True)
    
    success, fail, total_frames = 0, 0, 0
    start_time = time.time()
    
    for idx, video_path in enumerate(pending):
        try:
            output_dir = os.path.join(args.output_root, clip_name)
            
            # Read video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if len(frames) == 0:
                raise RuntimeError(f"No frames: {video_path}")
            
            orig_h, orig_w = frames[0].shape[:2]
            num_frames = len(frames)
            
            # Inference
            all_intrinsics, all_extrinsics, all_depth = [], [], []
            proc_h, proc_w = 0, 0
            
            with torch.no_grad():
                for batch_start in range(0, len(frames), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(frames))
                    batch_frames = frames[batch_start:batch_end]
                    
                    pred = model.inference(
                        image=batch_frames,
                        process_res=504,
                        process_res_method="upper_bound_resize",
                        export_dir=None,
                        export_format="mini_npz",
                    )
                    
                    all_intrinsics.extend(pred.intrinsics)
                    all_extrinsics.extend(pred.extrinsics)
                    all_depth.extend(pred.depth)
                    
                    if proc_h == 0:
                        proc_h, proc_w = pred.processed_images.shape[1:3]
            
            del frames
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
            
            for i in range(len(all_intrinsics)):
                k = all_intrinsics[i].copy()
                k[0, 0] *= scale_x
                k[1, 1] *= scale_y
                k[0, 2] *= scale_x
                k[1, 2] *= scale_y
                
                ext = all_extrinsics[i]
                ext = _as_homogeneous44(ext) if ext.shape == (3, 4) else ext
                
                if i == 0:
                    d = cv2.resize(all_depth[i].astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    np.save(os.path.join(output_dir, f"depth_{i:06d}.npy"), d)
                
                np.save(os.path.join(output_dir, f"intrinsics_{i:06d}.npy"), k)
                np.save(os.path.join(output_dir, f"extrinsics_{i:06d}.npy"), ext)
            
            with open(os.path.join(output_dir, "summary.txt"), "w") as f:
                f.write(f"video: {video_path}\nframes: {len(all_intrinsics)}\nresolution: {orig_w}x{orig_h}\nmodel: da3-giant\n")
                f.write("depth_mode: first_frame_only\n")
            
            del all_intrinsics, all_extrinsics, all_depth
            
            success += 1
            total_frames += num_frames
            
            torch.cuda.empty_cache()

        except Exception as e:
            fail += 1
            print(f"[Step 01a] Failed: {os.path.basename(video_path)} - {e}", flush=True)
            torch.cuda.empty_cache()
            raise

    elapsed = time.time() - start_time
    print(
        f"[Step 01a] Done | Success: {success} Failed: {fail} | "
        f"{total_frames/elapsed:.1f} frames/s | {elapsed/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
