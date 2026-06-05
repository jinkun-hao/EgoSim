# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
EgoVid-5M MANO Annotation Script
Batch MANO annotation for EgoVid-5M-style clips.
"""
# Mock pyrender / OpenGL before importing hamer (headless).
# Must run before any `hamer` import.
import sys
from unittest.mock import MagicMock

sys.modules['pyrender'] = MagicMock()
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.error'] = MagicMock()
sys.modules['OpenGL.platform'] = MagicMock()

import os
import csv
import json
import torch
import cv2
import numpy as np
import threading
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global lock for cv2 operations (cv2.VideoCapture is not thread-safe)
_cv2_lock = threading.Lock()

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel

import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed() -> None:
    if is_dist_avail_and_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


class EgoVidMANOAnnotator:
    """MANO annotator for EgoVid-style clips."""
    
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        clips_dir: Optional[str] = None,
        video_root: Optional[str] = None,
        device: str = "cuda",
        use_clips: bool = True,
        hand_batch_size: int = 8,
        body_batch_size: int = 8,
        dataloader_workers: int = 0,
        missing_clip_log: Optional[str] = None,
        output_mode: str = "light",
        rank: int = 0,
        world_size: int = 1,
        vitdet_init_checkpoint: Optional[str] = None,
    ):
        self.clips_dir = Path(clips_dir) if use_clips and clips_dir else None
        self.video_root = Path(video_root) if not use_clips and video_root else None
        self.use_clips = use_clips
        # Keep string device for detectors; torch.device for the model.
        self.device_str = device if isinstance(device, str) else str(device)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.hand_batch_size = hand_batch_size
        self.body_batch_size = body_batch_size
        self.dataloader_workers = dataloader_workers
        self.missing_clip_log = Path(missing_clip_log) if missing_clip_log else None
        self.output_mode = output_mode
        self.rank = rank
        self.world_size = world_size
        self.vitdet_init_checkpoint = vitdet_init_checkpoint
        
        if self.rank == 0:
            print("Loading HaMeR model...")
        self.model, self.model_cfg = load_hamer(
            checkpoint_path, 
            init_renderer=False, 
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load detectors.
        if self.rank == 0:
            print("Loading detectors...")
        self._init_detectors()
        
    def _init_detectors(self):
        """Init body and hand detectors."""
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hamer
        
        # ViTDet body detector.
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        if not self.vitdet_init_checkpoint:
            raise ValueError(
                "ViTDet checkpoint missing: pass vitdet_init_checkpoint or --vitdet_checkpoint "
                "(e.g. hamer/_DATA/model_final_f05665.pkl)"
            )
        detectron2_cfg.train.init_checkpoint = self.vitdet_init_checkpoint
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.body_detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        # ViTPose keypoint detector.
        self.pose_detector = ViTPoseModel(self.device_str)
    
    def _load_with_cv2(self, video_path: str, start_frame: int = 0, end_frame: int = -1) -> Optional[np.ndarray]:
        """Read video frames via OpenCV (ffmpeg backend).
        
        Args:
            video_path: Path to video file.
            start_frame: First frame index (0-based).
            end_frame: Last frame (-1 = read until EOF).
            
        Returns:
            frames: (N, H, W, 3) BGR uint8, or None.
        """
        # Use lock to serialize cv2 VideoCapture operations (prevents SIGSEGV)
        with _cv2_lock:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frames = []
            frame_idx = 0
            
            # Seek to start frame when requested.
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Stop once past end_frame.
                    if end_frame >= 0 and frame_idx > end_frame:
                        break
                    
                    frames.append(frame)  # cv2 yields BGR.
                    frame_idx += 1
            finally:
                cap.release()
        
        if len(frames) == 0:
            return None
        
        return np.stack(frames, axis=0)
    
    def load_clip_frames(self, video_id: str) -> Optional[np.ndarray]:
        """Load all frames for a clip (pre-extracted file or span in a source video).
        
        Uses OpenCV (ffmpeg backend).
        
        Args:
            video_id: Clip stem or filename, e.g. "uuid_start_end.mp4".
            
        Returns:
            frames: (N,H,W,3) BGR, or None on failure.
        """
        # Normalize video_id to .mp4.
        if not video_id.endswith('.mp4'):
            video_id = video_id + '.mp4'
        
        try:
            if self.use_clips:
                clip_path = str(self.clips_dir / video_id)
                frames = self._load_with_cv2(clip_path)
                return frames
            else:
                # Decode frame range from full source video.
                base_name = video_id.replace('.mp4', '')
                parts = base_name.rsplit('_', 2)
                uuid, start_frame, end_frame = parts
                start_frame, end_frame = int(start_frame), int(end_frame)
                
                source_video = str(self.video_root / f"{uuid}.mp4")
                frames = self._load_with_cv2(source_video, start_frame, end_frame)
                return frames
        except Exception as e:
            print(f"Warning: Failed to load video {video_id}: {e}")
            return None

    def _log_missing_clip(self, video_id: str):
        """Append missing clip id to log file."""
        if self.missing_clip_log is None:
            return
        try:
            self.missing_clip_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.missing_clip_log, 'a') as f:
                f.write(f"{video_id}\n")
        except Exception as e:
            print(f"Warning: Failed to write missing clip log {self.missing_clip_log}: {e}")
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in one frame.
        
        Returns:
            hand_detections: [{"bbox": [...], "is_right": bool, "confidence": float}]
        """
        # Body detection.
        det_out = self.body_detector(frame)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0:
            return []
        
        # Keypoints.
        img_rgb = frame[:, :, ::-1]
        vitposes_out = self.pose_detector.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        hand_detections = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            # Left hand.
            valid = left_hand_keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    left_hand_keyp[valid, 0].min(),
                    left_hand_keyp[valid, 1].min(),
                    left_hand_keyp[valid, 0].max(),
                    left_hand_keyp[valid, 1].max()
                ]
                confidence = left_hand_keyp[valid, 2].mean()
                hand_detections.append({
                    "bbox": bbox,
                    "is_right": False,
                    "confidence": float(confidence)
                })
            
            # Right hand.
            valid = right_hand_keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    right_hand_keyp[valid, 0].min(),
                    right_hand_keyp[valid, 1].min(),
                    right_hand_keyp[valid, 0].max(),
                    right_hand_keyp[valid, 1].max()
                ]
                confidence = right_hand_keyp[valid, 2].mean()
                hand_detections.append({
                    "bbox": bbox,
                    "is_right": True,
                    "confidence": float(confidence)
                })
        
        return hand_detections

    def _extract_hands_from_vitposes(self, vitposes_out: List[Dict]) -> List[Dict]:
        """Parse hand boxes from ViTPose output."""
        hand_detections = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            # Left hand.
            valid = left_hand_keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    left_hand_keyp[valid, 0].min(),
                    left_hand_keyp[valid, 1].min(),
                    left_hand_keyp[valid, 0].max(),
                    left_hand_keyp[valid, 1].max()
                ]
                confidence = left_hand_keyp[valid, 2].mean()
                hand_detections.append({
                    "bbox": bbox,
                    "is_right": False,
                    "confidence": float(confidence)
                })
            
            # Right hand.
            valid = right_hand_keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    right_hand_keyp[valid, 0].min(),
                    right_hand_keyp[valid, 1].min(),
                    right_hand_keyp[valid, 0].max(),
                    right_hand_keyp[valid, 1].max()
                ]
                confidence = right_hand_keyp[valid, 2].mean()
                hand_detections.append({
                    "bbox": bbox,
                    "is_right": True,
                    "confidence": float(confidence)
                })
        return hand_detections

    def detect_hands_batch(self, frames: np.ndarray, body_batch_size: int = 8) -> List[List[Dict]]:
        """Batch hand detection over many frames (optimized).
        
        Args:
            frames: (N, H, W, 3) BGR.
            body_batch_size: ViTDet batch size.
            
        Returns:
            Per-frame list of hand_detections.
        """
        num_frames = len(frames)
        all_hand_detections = [[] for _ in range(num_frames)]
        
        # Step 1: batched body detection.
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        
        frame_list = [frames[i] for i in range(num_frames)]
        all_det_results = self.body_detector.batch_predict(frame_list, batch_size=body_batch_size)
        
        mem_after_body = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [VRAM] Body Detection (bs={body_batch_size}, frames={num_frames}): peak {mem_after_body:.2f} GB, delta {mem_after_body - mem_before:.2f} GB")
        
        # Keep frames with valid person boxes.
        frame_body_results = []  # [(frame_idx, frame_rgb, bboxes_with_scores), ...]
        for fidx, det_out in enumerate(all_det_results):
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            
            if len(pred_bboxes) > 0:
                bboxes_with_scores = np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)
                img_rgb = frames[fidx][:, :, ::-1]
                frame_body_results.append((fidx, img_rgb, bboxes_with_scores))
        
        # Step 2: ViTPose (per-frame; simple loop).
        torch.cuda.reset_peak_memory_stats()
        mem_before_pose = torch.cuda.memory_allocated() / 1024**3
        
        # ViTPose is per-frame; loop stays simple.
        for fidx, img_rgb, bboxes_with_scores in frame_body_results:
            vitposes_out = self.pose_detector.predict_pose(
                img_rgb[:, :, ::-1],  # RGB -> BGR for predict_pose which converts back
                [bboxes_with_scores],
            )
            hand_detections = self._extract_hands_from_vitposes(vitposes_out)
            all_hand_detections[fidx] = hand_detections
        
        mem_after_pose = torch.cuda.max_memory_allocated() / 1024**3
        if len(frame_body_results) > 0:
            print(f"  [VRAM] ViTPose (frames={len(frame_body_results)}): peak {mem_after_pose:.2f} GB, delta {mem_after_pose - mem_before_pose:.2f} GB")
        
        return all_hand_detections

    def load_clips_parallel(self, video_ids: List[str], max_workers: int = 8) -> Dict[str, Optional[np.ndarray]]:
        """Load multiple clips in parallel.
        
        Args:
            video_ids: Clip ids / filenames.
            max_workers: Thread pool size.
            
        Returns:
            video_id -> frames or None.
        """
        results = {}
        
        def load_single(vid: str) -> Tuple[str, Optional[np.ndarray]]:
            try:
                frames = self.load_clip_frames(vid)
                return (vid, frames)
            except Exception as e:
                print(f"Warning: Exception loading {vid}: {e}")
                return (vid, None)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_single, vid): vid for vid in video_ids}
            for future in as_completed(futures):
                try:
                    vid, frames = future.result(timeout=60)
                    results[vid] = frames
                except Exception as e:
                    vid = futures.get(future, "unknown")
                    print(f"Warning: Future failed for {vid}: {e}")
                    results[vid] = None
        
        return results

    def run_hamer(self, frame: np.ndarray, hand_detections: List[Dict]) -> List[Dict]:
        """Run HaMeR on detected hands.
        
        Returns:
            mano_results: list of MANO parameter dicts.
        """
        if not hand_detections:
            return []
        
        # Bboxes / handedness.
        bboxes = np.array([det["bbox"] for det in hand_detections])
        is_right = np.array([det["is_right"] for det in hand_detections])
        
        # ViTDet crops dataset.
        dataset = ViTDetDataset(
            self.model_cfg,
            frame,
            bboxes,
            is_right,
            rescale_factor=2.0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hand_batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers
        )
        
        results = []
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model.forward_step_full(batch) if self.output_mode == "full" else self.model(batch)
            
            # full-image cam_t (same as demo.py).
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()
            
            # Pack outputs.
            for i in range(len(batch['img'])):
                result = {
                    "is_right": bool(batch['right'][i].cpu().numpy()),
                    "cam": out['pred_cam'][i].cpu().numpy().tolist(),  # (3,) weak-perspective camera.
                    "cam_t": out['pred_cam_t'][i].cpu().numpy().tolist(),  # (3,) translation in crop space.
                    "cam_t_full": pred_cam_t_full[i].tolist(),  # (3,) translation in full image.
                    "confidence": hand_detections[i]["confidence"]
                }
                if self.output_mode == "light":
                    result.update({
                        "global_orient": out['pred_mano_params']['global_orient'][i].cpu().numpy().tolist(),  # (1, 3, 3)
                        "hand_pose": out['pred_mano_params']['hand_pose'][i].cpu().numpy().tolist(),  # (15, 3, 3)
                        "betas": out['pred_mano_params']['betas'][i].cpu().numpy().tolist(),  # (10,)
                    })
                elif self.output_mode == "full":
                    result.update({
                        "pose": out['pred_mano_params']['hand_pose'][i].cpu().numpy().tolist(),
                        "shape": out['pred_mano_params']['betas'][i].cpu().numpy().tolist(),
                        "vertices": out['pred_vertices'][i].cpu().numpy().tolist(),
                        "pred_keypoints_3d": out['pred_keypoints_3d'][i].cpu().numpy().tolist(),
                    })
                results.append(result)
        
        return results
    
    def annotate_clip(self, video_id: str) -> Optional[Dict]:
        """Annotate one clip (per-frame HaMeR).
        
        Returns:
            Dict with video_id and frames list with per-frame hands.
        """
        # Load frames.
        frames = self.load_clip_frames(video_id)
        if frames is None:
            return None
        
        # Per-frame pipeline.
        frame_results = []
        for frame_idx, frame in enumerate(tqdm(frames, desc=f"Processing {video_id}")):
            # Detect hands.
            hand_detections = self.detect_hands(frame)
            
            # HaMeR forward.
            mano_results = self.run_hamer(frame, hand_detections)
            
            frame_results.append({
                "frame_idx": frame_idx,
                "hands": mano_results
            })
        
        return {
            "video_id": video_id,
            "frames": frame_results
        }
    
    def annotate_dataset(
        self,
        clip_list_path: Optional[str],
        output_dir: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
        batch_size_videos: int = 1,
        num_clips: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        """Batch annotation.
        
        Only clips from the list (or clips_dir) are processed—not whole source videos.
        
        Args:
            clip_list_path: CSV/TXT list path (optional).
            output_dir: JSON output directory.
            max_samples: Cap for testing.
            resume: Skip clips that already have outputs.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if clip_list_path:
            # Read clip list (CSV or TXT).
            if self.rank == 0:
                print(f"Loading clip list from {clip_list_path}...")
            
            if clip_list_path.endswith('.csv'):
                # CSV: video_id column.
                with open(clip_list_path, 'r') as f:
                    reader = csv.DictReader(f)
                    video_ids = [row['video_id'] for row in reader]
            else:
                # TXT: one clip id per line.
                with open(clip_list_path, 'r') as f:
                    video_ids = [line.strip() for line in f if line.strip()]
            
            if self.rank == 0:
                print(f"Found {len(video_ids)} clips in list")
            
            # Slice by start/end index.
            if start_index is not None or end_index is not None:
                original_len = len(video_ids)
                start_idx = start_index if start_index is not None else 0
                end_idx = end_index if end_index is not None else len(video_ids)
                video_ids = video_ids[start_idx:end_idx]
                if self.rank == 0:
                    print(f"Slicing clips [{start_idx}:{end_idx}] -> {len(video_ids)} clips (from {original_len} total)")
            
            if max_samples:
                video_ids = video_ids[:max_samples]
                if self.rank == 0:
                    print(f"Processing first {max_samples} clips for testing")
        else:
            # Or enumerate .mp4 under clips_dir.
            if not self.clips_dir:
                raise ValueError("clips_dir is required when csv_path is None")
            video_ids = sorted([p.name for p in self.clips_dir.iterdir() if p.suffix == '.mp4'])
            if self.rank == 0:
                print(f"Found {len(video_ids)} clips in {self.clips_dir}")
            if num_clips:
                video_ids = video_ids[:num_clips]
                if self.rank == 0:
                    print(f"Processing first {num_clips} clips from clips_dir")
            elif max_samples:
                video_ids = video_ids[:max_samples]
                if self.rank == 0:
                    print(f"Processing first {max_samples} clips for testing")

        if self.world_size > 1:
            video_ids = video_ids[self.rank::self.world_size]
            if self.rank == 0:
                print(f"DDP enabled: world_size={self.world_size}")
            print(f"[Rank {self.rank}] Assigned {len(video_ids)} clips", flush=True)
        
        # Pre-filter finished clips when resume.
        skipped = 0
        if resume:
            print(f"[Rank {self.rank}] Checking completed files in {output_dir}...", flush=True)
            
            # Parallel exists() checks (fast when output dir is huge).
            # Stat target path instead of listing everything.
            def check_exists(vid: str) -> Tuple[str, bool]:
                base_name = vid.replace('.mp4', '') if vid.endswith('.mp4') else vid
                target_json = os.path.join(output_dir, f"{base_name}.json")
                target_no_ext = os.path.join(output_dir, base_name)
                return (vid, os.path.exists(target_json) or os.path.exists(target_no_ext))
            
            original_count = len(video_ids)
            pending_ids = []
            
            # Thread pool for exists checks.
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = executor.map(check_exists, video_ids)
                for vid, exists in results:
                    if exists:
                        skipped += 1
                    else:
                        pending_ids.append(vid)
            video_ids = pending_ids
            print(f"[Rank {self.rank}] Skipped {skipped} completed clips, {len(video_ids)} remaining", flush=True)
        
        # Counters.
        processed = 0
        failed = 0
        
        # Group clips to build large hand batches.
        group_size = max(1, int(batch_size_videos))
        total = len(video_ids)
        idx0 = 0
        
        while idx0 < total:
            # Next group of pending clips.
            group_ids = video_ids[idx0:idx0 + group_size]
            idx0 += len(group_ids)

            if not group_ids:
                continue

            # Log group start per rank (multi-GPU visibility).
            print(f"[Rank {self.rank}] Processing group of {len(group_ids)} videos: {group_ids[:2]}{'...' if len(group_ids)>2 else ''}")

            # ConcatDataset + index_map for batched HaMeR.
            datasets = []
            index_map: List[Dict] = []
            per_video_frame_count: Dict[str, int] = {}
            # Output dicts per video.
            grouped_results: Dict[str, Dict] = {}

            # Step 1: decode videos.
            import time
            load_start = time.time()
            # Limit to 4 workers to prevent cv2 thread-safety issues (SIGSEGV)
            loaded_videos = self.load_clips_parallel(group_ids, max_workers=min(4, len(group_ids)))
            print(f"  [Rank {self.rank}] Loaded {len(group_ids)} videos in {time.time() - load_start:.2f}s")
            
            # Step 2: detect hands per frame.
            for vid in group_ids:
                frames = loaded_videos.get(vid)
                if frames is None:
                    failed += 1
                    print(f"✗ Failed to load frames: {vid}")
                    self._log_missing_clip(vid)  # log missing clip
                    continue
                per_video_frame_count[vid] = len(frames)
                grouped_results[vid] = {
                    "video_id": vid,
                    "frames": [ {"frame_idx": i, "hands": []} for i in range(len(frames)) ]
                }

                # Batched hand detection for all frames.
                all_hand_detections = self.detect_hands_batch(frames, body_batch_size=self.body_batch_size)
                
                for fidx, hand_detections in enumerate(all_hand_detections):
                    if not hand_detections:
                        continue
                    bboxes = np.array([det["bbox"] for det in hand_detections])
                    is_right = np.array([det["is_right"] for det in hand_detections])
                    confidences = [float(det.get("confidence", 1.0)) for det in hand_detections]

                    ds = ViTDetDataset(
                        self.model_cfg,
                        frames[fidx],
                        bboxes,
                        is_right,
                        rescale_factor=2.0
                    )
                    base_len = len(ds)
                    if base_len == 0:
                        continue
                    datasets.append(ds)
                    for j in range(base_len):
                        index_map.append({
                            "video_id": vid,
                            "frame_idx": fidx,
                            "confidence": confidences[j],
                        })

            if not datasets:
                # No hand crops: write empty JSON.
                for vid in group_ids:
                    if vid in grouped_results:
                        base_name = vid.replace('.mp4', '') if vid.endswith('.mp4') else vid
                        out_path = Path(output_dir) / f"{base_name}.json"
                        with open(out_path, 'w') as f:
                            json.dump(grouped_results[vid], f, indent=2)
                        processed += 1
                        if self.rank == 0:
                            print(f"✓ Saved (no hands): {out_path.name}")
                continue

            concat_ds = torch.utils.data.ConcatDataset(datasets)
            print(f"  [DEBUG] Total hand samples in concat_ds: {len(concat_ds)}, hand_batch_size: {self.hand_batch_size}")
            loader = torch.utils.data.DataLoader(
                concat_ds,
                batch_size=self.hand_batch_size,
                shuffle=False,
                num_workers=self.dataloader_workers,
            )

            # Batched HaMeR over concat dataset.
            seen = 0
            first_batch = True
            for batch in tqdm(loader, desc=f"HaMeR batched inference ({len(index_map)} hands)"):
                batch = recursive_to(batch, self.device)
                if first_batch:
                    print(f"  [DEBUG] First batch img shape: {batch['img'].shape}")
                    torch.cuda.reset_peak_memory_stats()
                    mem_before_hamer = torch.cuda.memory_allocated() / 1024**3
                
                with torch.no_grad():
                    out = self.model.forward_step_full(batch) if self.output_mode == "full" else self.model(batch)
                
                if first_batch:
                    mem_after_hamer = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"  [VRAM] HaMeR (bs={len(batch['img'])}): peak {mem_after_hamer:.2f} GB, delta {mem_after_hamer - mem_before_hamer:.2f} GB")
                    first_batch = False

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                ).detach().cpu().numpy()

                B = len(batch['img'])
                for i in range(B):
                    map_idx = seen + i
                    if map_idx >= len(index_map):
                        continue
                    meta = index_map[map_idx]
                    vid = meta['video_id']
                    fidx = int(meta['frame_idx'])
                    hand_item = {
                        "is_right": bool(batch['right'][i].cpu().numpy()),
                        "cam": out['pred_cam'][i].detach().cpu().numpy().tolist(),  # (3,)
                        "cam_t": out['pred_cam_t'][i].detach().cpu().numpy().tolist(),  # (3,)
                        "cam_t_full": pred_cam_t_full[i].tolist(),  # (3,)
                        "confidence": float(meta['confidence']),
                    }
                    if self.output_mode == "light":
                        hand_item.update({
                            "global_orient": out['pred_mano_params']['global_orient'][i].detach().cpu().numpy().tolist(),  # (1, 3, 3)
                            "hand_pose": out['pred_mano_params']['hand_pose'][i].detach().cpu().numpy().tolist(),  # (15, 3, 3)
                            "betas": out['pred_mano_params']['betas'][i].detach().cpu().numpy().tolist(),  # (10,)
                        })
                    elif self.output_mode == "full":
                        hand_item.update({
                            "pose": out['pred_mano_params']['hand_pose'][i].detach().cpu().numpy().tolist(),
                            "shape": out['pred_mano_params']['betas'][i].detach().cpu().numpy().tolist(),
                            "vertices": out['pred_vertices'][i].detach().cpu().numpy().tolist(),
                            "pred_keypoints_3d": out['pred_keypoints_3d'][i].detach().cpu().numpy().tolist(),
                        })
                    grouped_results[vid]['frames'][fidx]['hands'].append(hand_item)
                seen += B

            # Write JSON for each video in group.
            for vid, res in grouped_results.items():
                out_path = Path(output_dir) / f"{vid.replace('.mp4', '.json')}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(res, f, indent=2)
                processed += 1
                if self.rank == 0:
                    print(f"✓ Saved: {out_path.name}")
        
        # Summary.
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"FINAL STATISTICS")
            print(f"{'='*60}")
            print(f"Total clips in CSV: {len(video_ids)}")
            print(f"Successfully processed: {processed}")
            print(f"Skipped (already exists): {skipped}")
            print(f"Failed: {failed}")
            print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate MANO annotations for EgoVid-5M dataset clips'
    )
    parser.add_argument('--clip_list', type=str, default=None,
                       help='Path to clip list file (CSV or TXT format, optional). If not set, read clips from clips_dir')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for MANO annotations')
    parser.add_argument('--clips_dir', type=str, default=None,
                       help='Directory with pre-extracted clips (required unless --no_clips)')
    parser.add_argument('--video_root', type=str, default=None,
                       help='Root of original videos when using --no_clips')
    parser.add_argument('--no_clips', action='store_true',
                       help='Extract frames directly from original videos instead of using pre-extracted clips')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                       help='HaMeR checkpoint path')
    parser.add_argument('--vitdet_checkpoint', type=str, default=None,
                       help='ViTDet init checkpoint (.pkl); or set env VITDET_INIT_CHECKPOINT')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--no_resume', action='store_true',
                       help='Do not skip already processed clips')
    parser.add_argument('--batch_size_videos', type=int, default=80,
                       help='Number of clips to process concurrently (process-level)')
    parser.add_argument('--hand_batch_size', type=int, default=640,
                       help='Number of hand crops per forward pass')
    parser.add_argument('--body_batch_size', type=int, default=8,
                       help='Batch size for body detector (ViTDet)')
    parser.add_argument('--dataloader_workers', type=int, default=0,
                       help='CPU workers for preprocessing hand crops')
    parser.add_argument('--missing_log', type=str, default=None,
                       help='Path to log missing clips (default: <output_dir>/missing_clips.log)')
    parser.add_argument('--num_clips', type=int, default=None,
                       help='Number of clips to process from clips_dir (ordered by filename)')
    parser.add_argument('--output_mode', type=str, default='light', choices=['light', 'full'],
                       help='Output mode: light (global_orient/hand_pose/betas) or full (vertices/keypoints)')
    parser.add_argument('--ddp', action='store_true',
                       help='Enable DDP multi-GPU inference (use torchrun)')
    parser.add_argument('--start_index', type=int, default=None,
                       help='Start index of clips to process (0-based, inclusive)')
    parser.add_argument('--end_index', type=int, default=None,
                       help='End index of clips to process (0-based, exclusive)')
    
    args = parser.parse_args()

    vitdet_ckpt = args.vitdet_checkpoint or os.environ.get("VITDET_INIT_CHECKPOINT")
    if not vitdet_ckpt:
        parser.error(
            "Provide --vitdet_checkpoint or set VITDET_INIT_CHECKPOINT "
            "(ViTDet model_final_f05665.pkl path under hamer/_DATA)"
        )
    if not args.no_clips and not args.clips_dir:
        parser.error("--clips_dir is required unless --no_clips")
    if args.no_clips and not args.video_root:
        parser.error("--video_root is required with --no_clips")
    
    if args.ddp:
        init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    # DDP: one GPU per LOCAL_RANK.
    if args.ddp:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # Scale global batch sizes by world_size.
    # Videos / hands are global counts → divide by world_size.
    # body batch is already per-GPU.
    per_gpu_batch_size_videos = max(1, args.batch_size_videos // world_size)
    per_gpu_hand_batch_size = max(1, args.hand_batch_size // world_size)
    per_gpu_body_batch_size = args.body_batch_size  # per-GPU already

    if rank == 0:
        print("="*60)
        print("EgoVid-5M MANO Annotation Pipeline")
        print("="*60)
        print(f"Clip list file: {args.clip_list}")
        print(f"Output directory: {args.output_dir}")
        print(f"Clips directory: {args.clips_dir}")
        print(f"Video root: {args.video_root}")
        print(f"Device: {device} (local_rank={local_rank})")
        print(f"Use clips: {not args.no_clips}")
        print(f"Resume mode: {not args.no_resume}")
        print(f"DDP: {args.ddp} | world_size: {world_size}")
        print("-" * 40)
        print(f"[Global] batch_size_videos: {args.batch_size_videos}")
        print(f"[Global] hand_batch_size: {args.hand_batch_size}")
        print(f"[Per-GPU] batch_size_videos: {per_gpu_batch_size_videos}")
        print(f"[Per-GPU] hand_batch_size: {per_gpu_hand_batch_size}")
        print(f"[Per-GPU] body_batch_size: {per_gpu_body_batch_size}")
        print(f"Dataloader workers: {args.dataloader_workers}")
        print(f"Output mode: {args.output_mode}")
        if args.max_samples:
            print(f"Max samples (testing): {args.max_samples}")
        print("="*60)

    # Multi-GPU: per-rank batch sizes.
    if rank == 0:
        print("\nInitializing annotator...")
    # Default missing_log under output_dir.
    missing_log_path = args.missing_log or str(Path(args.output_dir) / 'missing_clips.log')
    annotator = EgoVidMANOAnnotator(
        checkpoint_path=args.checkpoint,
        clips_dir=args.clips_dir,
        video_root=args.video_root,
        device=device,
        use_clips=not args.no_clips,
        hand_batch_size=per_gpu_hand_batch_size,
        body_batch_size=per_gpu_body_batch_size,
        dataloader_workers=args.dataloader_workers,
        missing_clip_log=missing_log_path,
        output_mode=args.output_mode,
        rank=rank,
        world_size=world_size,
        vitdet_init_checkpoint=vitdet_ckpt,
    )
    
    if rank == 0:
        print("\nStarting batched annotation...")
        print("Note: Only processing clips listed in the clip list file")
        print("-" * 60)
    annotator.annotate_dataset(
        clip_list_path=args.clip_list,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        resume=not args.no_resume,
        batch_size_videos=per_gpu_batch_size_videos,
        num_clips=args.num_clips,
        start_index=args.start_index,
        end_index=args.end_index,
    )


if __name__ == '__main__':
    main()