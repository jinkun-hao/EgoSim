# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

from pathlib import Path

import gdown
import numpy as np
import torch

from egosim_state.streams.base import VideoFrame
from egosim_state.utils.model_paths import get_aot_checkpoint_path, get_sam3_checkpoint_path

from .seg_tracker import SegTracker


class TrackAnythingPipeline:
    def __init__(
        self,
        mask_phrases: list[str],
        sam_points_per_side: int = 30,
        sam_run_gap: int = 10,
        use_sam3: bool = True,
        continuous_tracking: bool = True,
        iou_thresh: float = 0.15,
        total_frames: int = 0,
        force_last_frame_keyframe: bool = True,
    ) -> None:
        """
        Initialize the TrackAnythingPipeline.
        
        Args:
            mask_phrases: List of phrases to detect and mask.
            sam_points_per_side: Number of points per side for SAM.
            sam_run_gap: Interval between keyframes for detection.
            use_sam3: Whether to use SAM3 instead of SAM.
            continuous_tracking: If True (default), objects are continuously tracked
                across all frames and new objects are accumulated over time.
                If False, tracker is reset at each keyframe - objects are detected
                fresh at keyframes and propagated to intermediate frames, but
                tracking history is not accumulated across keyframes.
            iou_thresh: IoU threshold for determining interactive objects (default 0.15).
            total_frames: Total number of frames in the video (used to ensure last frame is a keyframe).
            force_last_frame_keyframe: If True (default), always run SAM detection on the last frame.
        """
        self.use_sam3 = use_sam3
        self.continuous_tracking = continuous_tracking
        self.iou_thresh = iou_thresh
        self.total_frames = total_frames
        self.force_last_frame_keyframe = force_last_frame_keyframe
        
        # Prepare checkpoints.
        sam_ckpt_path = get_sam3_checkpoint_path()
        if not use_sam3 and not sam_ckpt_path:
            sam_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                dst=str(sam_ckpt_path),
            )

        aot_ckpt_path = get_aot_checkpoint_path()
        if not aot_ckpt_path.exists():
            aot_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view",
                output=str(aot_ckpt_path),
                fuzzy=True,
            )

        self.threshold_args = {
            "box_threshold": 0.35,
            "text_threshold": 0.5,  # Not useful now!
            "box_size_threshold": 1.0 ,
            "reset_image": True,
        }
        self.frame_idx = 0
        self.caption = "".join([m + "." for m in mask_phrases])
        self.sam_run_gap = sam_run_gap
        self.segtracker = SegTracker(
            segtracker_args={
                "sam_gap": sam_run_gap,  # the interval to run sam to segment new objects
                "min_area": 200,  # minimal mask area to add a new mask as a new object
                "max_obj_num": 255,  # maximal object number to track in a video
                "min_new_obj_iou": 0.8,  # the background area ratio of a new object should > 80%
                "iou_thresh": iou_thresh,  # IoU threshold for interactive objects
            },
            sam_args={
                "sam_checkpoint": str(sam_ckpt_path) if not use_sam3 else None,
                "model_type": "vit_b",
                "generator_args": {
                    "points_per_side": sam_points_per_side,
                    "pred_iou_thresh": 0.8,
                    "stability_score_thresh": 0.9,
                    "crop_n_layers": 1,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 200,
                },
                "gpu_id": 0,
            },
            aot_args={
                "phase": "PRE_YTB_DAV",
                "model": "r50_deaotl",
                "model_path": str(aot_ckpt_path),
                "long_term_mem_gap": 9999,
                "max_len_long_term": 9999,
                "gpu_id": 0,
            },
            use_sam3=use_sam3,
        )
        self.segtracker.restart_tracker()
        self.instance_phrase = {0: "background"}
        self.human_obj_ids = set()  # Track which object IDs correspond to hands/persons
        self.depth_distance_thresh = 0.15  # Depth distance threshold for interactive filtering (in meters)

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        return intersection / union if union > 0 else 0.0

    def _compute_mask_depth_distance(self, mask1: np.ndarray, mask2: np.ndarray, depth: np.ndarray) -> float:
        """
        Compute depth distance between two masks.
        Returns the absolute difference of median depths of the two masks.
        """
        if depth is None:
            return 0.0
        
        # Get depth values for each mask
        depth_np = depth.cpu().numpy() if hasattr(depth, 'cpu') else depth
        
        mask1_depths = depth_np[mask1 > 0]
        mask2_depths = depth_np[mask2 > 0]
        
        if len(mask1_depths) == 0 or len(mask2_depths) == 0:
            return float('inf')
        
        # Use median depth to be robust to outliers
        median1 = np.median(mask1_depths)
        median2 = np.median(mask2_depths)
        
        return abs(median1 - median2)

    def _compute_interactive_from_mask(self, pred_mask: np.ndarray, depth: np.ndarray = None) -> set[int]:
        """
        Compute interactive object IDs based on mask bounding box IOU with hand.
        - Background (ID=0) is always visible and never marked as interactive
        - Hand, arm, person, and object in hand are always marked as interactive (default not visible)
        - Objects with high IOU with hand mask bounding box AND close depth are considered interactive
        """
        interactive_obj_ids = set()
        
        # Get all unique object IDs in the mask
        unique_ids = np.unique(pred_mask)
        
        # Collect hand masks for interaction computation, and mark human-related as interactive
        hand_masks = []
        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background - it's always visible
                continue
            phrase = self.instance_phrase.get(obj_id, "")
            if isinstance(phrase, str):
                phrase_lower = phrase.lower()
                # Check if it's hand, arm, person (default not visible)
                is_hand = "hand" in phrase_lower and "in hand" not in phrase_lower
                if is_hand or "arm" in phrase_lower or "person" in phrase_lower or "in hand" in phrase_lower:
                    self.human_obj_ids.add(obj_id)
                    # Mark hand, arm, person, object in hand as interactive (default not visible)
                    interactive_obj_ids.add(obj_id)
                    # Use hand/arm for IOU computation
                    if is_hand or "arm" in phrase_lower:
                        hand_masks.append((obj_id, pred_mask == obj_id))
        
        if not hand_masks:
            return interactive_obj_ids
        
        # For each object, compute bounding box from mask and check IOU with hand boxes
        def _mask_to_bbox(mask):
            rows, cols = np.where(mask > 0)
            if len(rows) == 0 or len(cols) == 0:
                return None
            return [cols.min(), rows.min(), cols.max(), rows.max()]
        
        hand_bboxes = []
        for _, hmask in hand_masks:
            bbox = _mask_to_bbox(hmask)
            if bbox:
                hand_bboxes.append(bbox)
        
        def _iou_xyxy(a, c):
            ax1, ay1, ax2, ay2 = a
            cx1, cy1, cx2, cy2 = c
            inter_x1 = max(ax1, cx1)
            inter_y1 = max(ay1, cy1)
            inter_x2 = min(ax2, cx2)
            inter_y2 = min(ay2, cy2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            union = area_a + area_c - inter + 1e-6
            return inter / union
        
        for obj_id in unique_ids:
            if obj_id == 0 or obj_id in interactive_obj_ids:  # Skip background and already marked
                continue
            obj_mask = pred_mask == obj_id
            obj_bbox = _mask_to_bbox(obj_mask)
            if not obj_bbox:
                continue
            max_iou = 0.0
            for hbbox in hand_bboxes:
                max_iou = max(max_iou, _iou_xyxy(obj_bbox, hbbox))
            if max_iou > self.iou_thresh:
                # Check depth distance if depth is available
                if depth is not None:
                    min_depth_dist = float('inf')
                    for _, hmask in hand_masks:
                        depth_dist = self._compute_mask_depth_distance(obj_mask, hmask, depth)
                        min_depth_dist = min(min_depth_dist, depth_dist)
                    if min_depth_dist > self.depth_distance_thresh:
                        phrase = self.instance_phrase.get(obj_id, "unknown")
                        print(f"Object ID={obj_id} ({phrase}) has high IOU={max_iou:.3f} but depth distance={min_depth_dist:.3f}m > {self.depth_distance_thresh}m, skipping")
                        continue
                phrase = self.instance_phrase.get(obj_id, "unknown")
                print(f"Interactive object detected: ID={obj_id} ({phrase}), IOU={max_iou:.3f}")
                interactive_obj_ids.add(obj_id)
        
        return interactive_obj_ids

    def track(self, frame_data: VideoFrame) -> tuple[torch.Tensor, dict[int, str], set[int], dict]:
        """
        Detect new and track existing objects in the frame.

        Args:
            frame_data (VideoFrame): The frame data to track.

        Returns:
            torch.Tensor: The mask of the tracked objects (H, W) uint8 tensor.
                0 is background, >0 is object id.
            dict[int, str]: The phrases associated with each object id.
            set[int]: Set of object IDs that are interactive (high IoU with hand boxes).
                These objects are still tracked but should be filtered in visualization.
            dict: Boxes info for visualization {obj_id: {'box': [x1,y1,x2,y2], 'phrase': str, 'is_human': bool, 'is_interactive': bool}}.
        """

        # Convert to RGB numpy images
        rgb_frame = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
        boxes_info = {}  # Store boxes info for visualization
        
        # Keep last detection's interactive_obj_ids for non-keyframes
        if not hasattr(self, '_last_interactive_obj_ids'):
            self._last_interactive_obj_ids = set()
        interactive_obj_ids = self._last_interactive_obj_ids.copy()

        # Always run SAM detection on the last frame (ensure it is in the keyframe list)
        is_last_frame = (
            self.force_last_frame_keyframe
            and self.total_frames > 0
            and self.frame_idx == self.total_frames - 1
        )
        is_keyframe = (self.frame_idx % self.sam_run_gap == 0) or is_last_frame

        if self.frame_idx == 0:
            # First frame: always detect and segment
            pred_mask, _, pred_phrase, interactive_obj_ids, boxes_info = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            print(f"Frame {self.frame_idx}: First frame detected {len(boxes_info)} boxes, interactive_obj_ids={interactive_obj_ids}")
            self.segtracker.add_reference(rgb_frame, pred_mask)
            self.instance_phrase.update(pred_phrase)
            # Save for non-keyframes
            self._last_interactive_obj_ids = interactive_obj_ids.copy()

        elif is_keyframe:
            # Keyframe: detect objects
            seg_mask, _, pred_phrase, interactive_obj_ids_detected, boxes_info = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            
            print(f"Frame {self.frame_idx}: Keyframe detected {len(boxes_info)} boxes")
            
            if self.continuous_tracking:
                # Continuous tracking mode: track existing + detect new objects (accumulate)
                track_mask = self.segtracker.track(rgb_frame)
                new_obj_mask, seg_to_final_mapping = self.segtracker.find_new_objs(track_mask, seg_mask)
                if np.sum(new_obj_mask > 0) > rgb_frame.shape[0] * rgb_frame.shape[1] * 0.4:
                    new_obj_mask = np.zeros_like(new_obj_mask)
                    # Recompute mapping without new objects
                    seg_to_final_mapping = {k: v for k, v in seg_to_final_mapping.items() 
                                           if v in np.unique(track_mask)}
                pred_mask = track_mask + new_obj_mask
                
                # Update instance phrases for new objects
                for seg_id, final_id in seg_to_final_mapping.items():
                    if seg_id in pred_phrase and final_id not in self.instance_phrase:
                        self.instance_phrase[final_id] = pred_phrase[seg_id]
                pred_phrase = {seg_to_final_mapping[k]: v for k, v in pred_phrase.items() if k in seg_to_final_mapping}
                
                # Remap boxes_info and interactive_obj_ids using seg_to_final_mapping
                remapped_boxes_info = {}
                remapped_interactive_ids = set()
                for seg_id, box_data in boxes_info.items():
                    if seg_id in seg_to_final_mapping:
                        final_id = seg_to_final_mapping[seg_id]
                        remapped_boxes_info[final_id] = box_data
                        if seg_id in interactive_obj_ids_detected:
                            remapped_interactive_ids.add(final_id)
                boxes_info = remapped_boxes_info
                interactive_obj_ids = remapped_interactive_ids
                print(f"Frame {self.frame_idx}: seg_to_final_mapping = {seg_to_final_mapping}")
                print(f"Frame {self.frame_idx}: After remapping, interactive_obj_ids = {interactive_obj_ids}")
                self.segtracker.add_reference(rgb_frame, pred_mask)
            else:
                # Keyframe propagation mode: reset tracker and use fresh detection
                # This avoids tracking drift by starting fresh at each keyframe
                self.segtracker.restart_tracker()
                pred_mask = seg_mask
                interactive_obj_ids = interactive_obj_ids_detected
                self.segtracker.add_reference(rgb_frame, pred_mask)
            
            self.instance_phrase.update(pred_phrase)
            # Save for non-keyframes
            self._last_interactive_obj_ids = interactive_obj_ids.copy()

        else:
            # Non-keyframe: propagate mask using AOT tracker
            pred_mask = self.segtracker.track(rgb_frame, update_memory=True)
            # Keep interactive_obj_ids from last detection (no new detection on non-keyframes)
            # interactive_obj_ids remains unchanged from previous frame
            # No new boxes info for propagated frames

        self.frame_idx += 1

        pred_mask_unique = np.unique(pred_mask)
        pred_phrase = {k: self.instance_phrase[k] for k in pred_mask_unique}

        # Compute interactive objects based on mask (using bounding boxes from masks)
        # Depth filtering will be done later if depth is available
        interactive_obj_ids = self._compute_interactive_from_mask(pred_mask, None)

        # Debug: log interactive objects
        if len(interactive_obj_ids) > 0:
            print(f"Frame {self.frame_idx}: Interactive objects: {interactive_obj_ids}")

        return torch.from_numpy(pred_mask).cuda(), pred_phrase, interactive_obj_ids, boxes_info

    def filter_interactive_by_depth(
        self, 
        instance_mask: torch.Tensor, 
        depth: torch.Tensor, 
        interactive_obj_ids: set[int],
        depth_thresh: float = None,
        return_details: bool = False
    ) -> set[int] | tuple[set[int], dict[int, dict]]:
        """
        Filter interactive objects by depth distance to person.
        Objects that have depth distance > threshold to all person masks are removed from interactive set.
        
        Args:
            instance_mask: (H, W) tensor with object IDs
            depth: (H, W) tensor with depth values
            interactive_obj_ids: set of object IDs marked as interactive
            depth_thresh: depth distance threshold in meters (default uses self.depth_distance_thresh)
            return_details: if True, also return detailed interaction info for all objects
            
        Returns:
            If return_details=False: Filtered set of interactive object IDs
            If return_details=True: (filtered_ids, interaction_details)
                interaction_details: {obj_id: {'is_interacting': bool, 'interacting_with': [ids], 
                                                'depth_dist': float, 'phrase': str}}
        """
        if depth is None or len(interactive_obj_ids) == 0:
            if return_details:
                return interactive_obj_ids, {}
            return interactive_obj_ids
        
        depth_thresh = depth_thresh if depth_thresh is not None else self.depth_distance_thresh
        
        instance_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
        depth_np = depth.cpu().numpy() if hasattr(depth, 'cpu') else depth
        
        # Find human reference masks (hands/arms + persons)
        hand_ids = []
        person_ids = []
        hand_masks = []
        person_masks = []
        unique_ids = np.unique(instance_np)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            phrase = self.instance_phrase.get(int(obj_id), "")
            if isinstance(phrase, str):
                pl = phrase.lower()
                if (("hand" in pl and "in hand" not in pl) or ("arm" in pl)):
                    hand_ids.append(int(obj_id))
                    hand_masks.append(instance_np == obj_id)
                if ("person" in pl):
                    person_ids.append(int(obj_id))
                    person_masks.append(instance_np == obj_id)

        human_ids = hand_ids + person_ids
        human_masks = hand_masks + person_masks
        if not human_masks:
            # No human reference masks found, keep all interactive objects
            if return_details:
                return interactive_obj_ids, {}
            return interactive_obj_ids
        
        filtered_ids = set()
        interaction_details = {}
        
        for obj_id in interactive_obj_ids:
            phrase = self.instance_phrase.get(int(obj_id), "")
            phrase_lower = phrase.lower() if isinstance(phrase, str) else ""
            
            # Always keep hand, arm, person (human-related) - they are always hidden
            # "object in hand" will be checked by depth
            is_hand = "hand" in phrase_lower and "in hand" not in phrase_lower
            if is_hand or "arm" in phrase_lower or "person" in phrase_lower:
                filtered_ids.add(obj_id)
                if return_details:
                    interaction_details[int(obj_id)] = {
                        'is_interacting': True,
                        'interacting_with': [],  # Human parts interact with themselves
                        'depth_dist': 0.0,
                        'phrase': phrase
                    }
                continue
            
            # For other objects (including "object in hand"), check depth distance to human masks (hand/person)
            obj_mask = instance_np == obj_id
            if np.sum(obj_mask) == 0:
                continue
                
            # Check depth distance to any human mask
            min_depth_dist = float('inf')
            interacting_with = []
            for idx, (ref_id, ref_mask) in enumerate(zip(human_ids, human_masks)):
                depth_dist = self._compute_mask_depth_distance(obj_mask, ref_mask, depth_np)
                if depth_dist < min_depth_dist:
                    min_depth_dist = depth_dist
                if depth_dist <= depth_thresh:
                    interacting_with.append(ref_id)
            
            is_interacting = min_depth_dist <= depth_thresh
            if is_interacting:
                filtered_ids.add(obj_id)
                print(f"Object ID={obj_id} ({phrase}) passed depth filter: distance={min_depth_dist:.3f}m <= {depth_thresh}m")
            else:
                print(f"Object ID={obj_id} ({phrase}) filtered out by depth: distance={min_depth_dist:.3f}m > {depth_thresh}m")
            
            if return_details:
                interaction_details[int(obj_id)] = {
                    'is_interacting': is_interacting,
                    'interacting_with': interacting_with,
                    'depth_dist': float(min_depth_dist),
                    'phrase': phrase
                }
        
        if return_details:
            return filtered_ids, interaction_details
        return filtered_ids
