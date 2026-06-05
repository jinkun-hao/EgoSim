"""
Generate a synthetic test annotation JSON + dummy video for visualize_skeleton_3d.py.

Creates a realistic right-hand skeleton (21 MANO joints) positioned in front of
the camera, with slight per-frame motion to verify the 3D rendering pipeline.

Usage:
    cd /mnt/shared-storage-user/ailab-idc1-shared/haojinkun/private/WorldModel/EgoWM
    python egosim-opensource/data_process/02_mano_predict/generate_test_skeleton.py

Then run:
    python egosim-opensource/data_process/02_mano_predict/visualize_skeleton_3d.py \
        --video_dir /tmp/skeleton_3d_test/video \
        --annot_dir /tmp/skeleton_3d_test/annot \
        --output_dir /tmp/skeleton_3d_test/output
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np


OUTPUT_BASE = Path("/tmp/skeleton_3d_test")
VIDEO_DIR = OUTPUT_BASE / "video"
ANNOT_DIR = OUTPUT_BASE / "annot"

WIDTH, HEIGHT = 640, 480
FPS = 16
NUM_FRAMES = 32


def make_right_hand_keypoints() -> np.ndarray:
    """
    Create a plausible right-hand skeleton (21 joints) in MANO local space.
    Joint order: wrist(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), little(17-20).
    Units are roughly in meters (hand ~0.18m long).
    """
    kp = np.zeros((21, 3), dtype=np.float64)

    # Wrist at origin
    kp[0] = [0.0, 0.0, 0.0]

    # Thumb (slightly offset to the side)
    kp[1] = [0.025, -0.01, 0.01]
    kp[2] = [0.045, -0.025, 0.015]
    kp[3] = [0.060, -0.040, 0.012]
    kp[4] = [0.070, -0.055, 0.010]

    # Index finger
    kp[5] = [0.020, -0.05, 0.0]
    kp[6] = [0.022, -0.08, 0.0]
    kp[7] = [0.021, -0.10, 0.0]
    kp[8] = [0.020, -0.115, 0.0]

    # Middle finger
    kp[9]  = [0.0, -0.055, 0.0]
    kp[10] = [0.0, -0.090, 0.0]
    kp[11] = [0.0, -0.112, 0.0]
    kp[12] = [0.0, -0.128, 0.0]

    # Ring finger
    kp[13] = [-0.018, -0.050, 0.0]
    kp[14] = [-0.020, -0.082, 0.0]
    kp[15] = [-0.020, -0.102, 0.0]
    kp[16] = [-0.020, -0.116, 0.0]

    # Little finger
    kp[17] = [-0.035, -0.042, 0.005]
    kp[18] = [-0.038, -0.068, 0.005]
    kp[19] = [-0.039, -0.084, 0.005]
    kp[20] = [-0.040, -0.095, 0.005]

    return kp


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate dummy video (gray gradient background)
    video_path = VIDEO_DIR / "test_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(NUM_FRAMES):
        gray_val = int(40 + 3 * i)
        frame = np.full((HEIGHT, WIDTH, 3), gray_val, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        writer.write(frame)
    writer.release()
    print(f"Video saved: {video_path}")

    # Generate annotation JSON
    base_kp3d = make_right_hand_keypoints()
    cam_t = np.array([0.0, 0.05, 0.5])  # hand ~0.5m in front of camera

    frames = []
    for fidx in range(NUM_FRAMES):
        # Add slight waving motion (rotate around Y axis)
        angle = np.sin(fidx / NUM_FRAMES * 2 * np.pi) * 0.15
        rot = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        kp3d_frame = (rot @ base_kp3d.T).T

        frames.append({
            "frame_idx": fidx,
            "hands": [
                {
                    "is_right": 1,
                    "keypoints_3d": kp3d_frame.tolist(),
                    "cam_t_full": cam_t.tolist(),
                }
            ]
        })

    annot = {"frames": frames}
    annot_path = ANNOT_DIR / "test_clip.json"
    with open(annot_path, "w") as f:
        json.dump(annot, f, indent=2)
    print(f"Annotation saved: {annot_path}")

    print()
    print("=" * 60)
    print("Test data generated! Run 3D visualization with:")
    print()
    print("  python egosim-opensource/data_process/02_mano_predict/visualize_skeleton_3d.py \\")
    print(f"      --video_dir {VIDEO_DIR} \\")
    print(f"      --annot_dir {ANNOT_DIR} \\")
    print(f"      --output_dir {OUTPUT_BASE / 'output'}")
    print()
    print("Output will be at:")
    print(f"  {OUTPUT_BASE / 'output' / 'test_clip_overlay.mp4'}")
    print(f"  {OUTPUT_BASE / 'output' / 'test_clip_black.mp4'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
