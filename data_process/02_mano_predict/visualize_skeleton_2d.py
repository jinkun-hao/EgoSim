# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
Batch-render hand skeleton videos (ego_demo-style layout).
Projects existing keypoints_3d + cam_t_full to 2D; no MANO forward pass.

Inputs:
  annot_dir  : .../clips_16fps_mano_full/<uuid>/<clip>.json
  video_dir  : .../clips_16fps/<uuid>/<clip>.mp4
Outputs (per clip):
  output_dir/<uuid>/<clip>_overlay.mp4   skeleton on video
  output_dir/<uuid>/<clip>_black.mp4    skeleton on black background
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

# ─── Skeleton colors (BGR) ──────────────────────────────────────────────
FINGER_COLORS_BGR = {
    "thumb":  (238, 130, 238)[::-1],
    "index":  (255,  99,  71)[::-1],
    "middle": (250, 245, 230)[::-1],
    "ring":   ( 47, 255, 173)[::-1],
    "little": (191, 152,   0)[::-1],
}
FINGER_COLORS_BGR = {k: tuple(int(c) for c in v) for k, v in FINGER_COLORS_BGR.items()}

FINGER_CONNECTIONS = {
    "thumb":  [0,  1,  2,  3,  4],
    "index":  [0,  5,  6,  7,  8],
    "middle": [0,  9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "little": [0, 17, 18, 19, 20],
}
JOINT_TO_FINGER: Dict[int, str] = {}
for _finger, _indices in FINGER_CONNECTIONS.items():
    for _idx in _indices[1:]:
        JOINT_TO_FINGER[_idx] = _finger

HAMER_FOCAL = 5000.0
HAMER_IMG_SIZE = 256.0


def project_keypoints(kp3d: np.ndarray, cam_t: np.ndarray,
                      is_right: int, scaled_fl: float,
                      cx: float, cy: float):
    """Project (21,3) keypoints + cam_t to pixels; return (21,2) and per-joint depth."""
    kp = kp3d.copy()
    multiplier = 2 * is_right - 1
    kp[:, 0] *= multiplier
    kp_trans = kp + cam_t.reshape(1, 3)
    z = kp_trans[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = scaled_fl * (kp_trans[:, 0] / z) + cx
    v = scaled_fl * (kp_trans[:, 1] / z) + cy
    return np.stack([u, v], axis=-1), z


# Line / point thickness in 3D "scene" units; screen size scales with depth.
BASE_LINE_THICKNESS_3D = 0.004
BASE_POINT_RADIUS_3D   = 0.006


def draw_skeleton(frame: np.ndarray, kp2d: np.ndarray, depths: np.ndarray,
                  scaled_fl: float):
    """Depth-aware 2D hand skeleton (thicker when closer)."""
    for finger, indices in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS_BGR[finger]
        for i in range(len(indices) - 1):
            idx_a, idx_b = indices[i], indices[i + 1]
            pt1 = tuple(map(int, kp2d[idx_a]))
            pt2 = tuple(map(int, kp2d[idx_b]))
            avg_z = 0.5 * (depths[idx_a] + depths[idx_b])
            thick = max(1, int(round(BASE_LINE_THICKNESS_3D * scaled_fl / avg_z)))
            cv2.line(frame, pt1, pt2, color, thick, cv2.LINE_AA)
    for i in range(kp2d.shape[0]):
        pt = tuple(map(int, kp2d[i]))
        finger = JOINT_TO_FINGER.get(i, "middle")
        radius = max(1, int(round(BASE_POINT_RADIUS_3D * scaled_fl / depths[i])))
        cv2.circle(frame, pt, radius, FINGER_COLORS_BGR[finger], -1, cv2.LINE_AA)


def process_clip(video_path: Path, annot_path: Path, output_dir: Path):
    """One clip -> overlay + black-background mp4."""
    with open(annot_path, "r") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 16.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scaled_fl = HAMER_FOCAL / HAMER_IMG_SIZE * max(width, height)
    cx, cy = width / 2.0, height / 2.0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path = output_dir / f"{stem}_overlay.mp4"
    black_path   = output_dir / f"{stem}_black.mp4"

    w_overlay = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    w_black   = cv2.VideoWriter(str(black_path),   fourcc, fps, (width, height))

    frame_map = {f["frame_idx"]: f for f in data.get("frames", [])}

    for fidx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        overlay_canvas = frame.copy()
        black_canvas   = np.zeros((height, width, 3), dtype=np.uint8)

        frame_annot = frame_map.get(fidx, {})
        for hand in frame_annot.get("hands", []):
            kp3d = hand.get("keypoints_3d")
            if kp3d is None:
                continue
            kp3d = np.array(kp3d, dtype=np.float64)
            if kp3d.shape[0] < 21:
                continue

            cam_t_key = "cam_t_full" if "cam_t_full" in hand else "cam_t"
            cam_t = np.array(hand[cam_t_key], dtype=np.float64)
            is_right = int(hand["is_right"])

            kp2d, depths = project_keypoints(kp3d, cam_t, is_right, scaled_fl, cx, cy)
            draw_skeleton(overlay_canvas, kp2d, depths, scaled_fl)
            draw_skeleton(black_canvas, kp2d, depths, scaled_fl)

        w_overlay.write(overlay_canvas)
        w_black.write(black_canvas)

    cap.release()
    w_overlay.release()
    w_black.release()


def main():
    parser = argparse.ArgumentParser(
        description="Batch visualize hand skeleton videos for ego_demo dataset"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory tree with per-clip mp4 files")
    parser.add_argument("--annot_dir", type=str, required=True,
                        help="Directory tree with per-clip full MANO json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for overlay / black-bg videos")
    parser.add_argument("--no_resume", action="store_true",
        help="Re-process all clips even if output already exists")
    args = parser.parse_args()

    video_dir  = Path(args.video_dir)
    annot_dir  = Path(args.annot_dir)
    output_dir = Path(args.output_dir)

    annot_files = sorted(annot_dir.rglob("*.json"))
    annot_files = [a for a in annot_files if a.name != "source.json"]

    pairs = []
    for ap in annot_files:
        rel = ap.relative_to(annot_dir)
        vp = video_dir / rel.with_suffix(".mp4")
        if not vp.exists():
            print(f"[SKIP] Video not found: {vp}")
            continue
        out_sub = output_dir / rel.parent
        pairs.append((vp, ap, out_sub))

    print(f"Total annotation files: {len(annot_files)}")
    print(f"Clips to process: {len(pairs)}")

    for vp, ap, out_sub in tqdm(pairs, desc="Visualizing"):
        try:
            process_clip(vp, ap, out_sub)
        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}")


if __name__ == "__main__":
    main()
