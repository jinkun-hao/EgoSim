# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np

from .aot_tracker import get_aot
from .detector import Detector


class SegTracker:
    def __init__(self, segtracker_args, sam_args, aot_args, use_sam3: bool = False) -> None:
        """
        Initialize SAM/SAM3 and AOT.
        
        Args:
            segtracker_args: arguments for segtracker
            sam_args: arguments for SAM/SAM3
            aot_args: arguments for AOT tracker
            use_sam3: if True, use SAM 3 instead of original SAM
        """
        if use_sam3:
            from .segmentor_sam3 import Segmentor
        else:
            from .segmentor import Segmentor
        
        self.sam = Segmentor(sam_args)
        self.tracker = get_aot(aot_args)
        self.detector = Detector(self.sam.device if hasattr(self.sam, 'device') else 0)
        self.sam_gap = segtracker_args["sam_gap"]
        self.min_area = segtracker_args["min_area"]
        self.max_obj_num = segtracker_args["max_obj_num"]
        self.min_new_obj_iou = segtracker_args["min_new_obj_iou"]
        self.iou_thresh = segtracker_args.get("iou_thresh", 0.10)  # IoU threshold for interactive objects
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None
        self.use_sam3 = use_sam3

        # debug
        self.everything_points = []
        self.everything_labels = []

    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        # obj_ids = np.unique(updated_merged_mask)
        # obj_ids = obj_ids[obj_ids!=0]
        # self.object_idx = int(max(obj_ids)) + 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self, frame, mask, frame_step=0):
        """
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        """
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
        self.tracker.add_reference_frame(frame, mask, self.curr_idx, frame_step)
        self.curr_idx += 1

    def track(self, frame, update_memory=False):
        """
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        """
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i != 0]
        return objs

    def get_obj_num(self):
        objs = self.get_tracking_objs()
        if len(objs) == 0:
            return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        """
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Also compute mapping from seg_mask IDs to track_mask IDs for already-tracked objects.
        
        Arguments:
            track_mask: numpy array (h,w) - mask from AOT tracker with tracked object IDs
            seg_mask: numpy array (h,w) - mask from detection with detection-time IDs
        Return:
            new_obj_mask: numpy array (h,w) - mask of new objects with assigned IDs
            seg_to_final_mapping: dict - mapping from seg_mask ID to final pred_mask ID
                For new objects: seg_id -> new_assigned_id
                For already-tracked objects: seg_id -> track_mask_id (matched by IoU)
        """
        new_obj_mask = (track_mask == 0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids != 0]
        seg_to_final_mapping = {}
        obj_num = self.curr_idx
        
        # Process new objects (not yet tracked)
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask == idx)
            obj_area = np.sum(seg_mask == idx)
            if (
                new_obj_area / obj_area < self.min_new_obj_iou
                or new_obj_area < self.min_area
                or obj_num > self.max_obj_num
            ):
                new_obj_mask[new_obj_mask == idx] = 0
            else:
                new_obj_mask[new_obj_mask == idx] = obj_num
                seg_to_final_mapping[idx] = obj_num
                obj_num += 1
        
        # Now find mapping for already-tracked objects (seg_mask objects that overlap with track_mask)
        all_seg_ids = np.unique(seg_mask)
        all_seg_ids = all_seg_ids[all_seg_ids != 0]
        track_ids = np.unique(track_mask)
        track_ids = track_ids[track_ids != 0]
        
        for seg_id in all_seg_ids:
            if seg_id in seg_to_final_mapping:
                # Already mapped as new object
                continue
            
            # Find the best matching track_mask ID by IoU
            seg_obj_mask = (seg_mask == seg_id)
            best_track_id = None
            best_iou = 0.0
            
            for track_id in track_ids:
                track_obj_mask = (track_mask == track_id)
                intersection = np.sum(seg_obj_mask & track_obj_mask)
                union = np.sum(seg_obj_mask | track_obj_mask)
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
            
            # If good IoU match found, map seg_id to track_id
            if best_track_id is not None and best_iou > 0.3:  # threshold for matching
                seg_to_final_mapping[seg_id] = best_track_id
        
        return new_obj_mask, seg_to_final_mapping

    def restart_tracker(self):
        self.tracker.restart()

    def add_mask(self, interactive_mask: np.ndarray):
        """
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        """
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape, dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask

    def detect_and_seg(
        self,
        origin_frame: np.ndarray,
        grounding_caption,
        box_threshold,
        text_threshold: float = 0.0,
        box_size_threshold=1,
        reset_image=False,
    ):
        """
        Using Grounding-DINO to detect objects according to text prompts, then
        segment with SAM/SAM3 (SAM3 uses box center by default).
        Return:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        """
        # backup id and origin-merged-mask
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask
        seg_phrase = {}

        annotated_frame_shape = origin_frame.shape[:2]
        refined_merged_mask = np.zeros(annotated_frame_shape, dtype=np.uint8)

        # Always use GroundingDINO + SAM/SAM3 segmentation (SAM3 via box center)
        annotated_frame_shape, boxes, phrases = self.detector.run_grounding(
            origin_frame, grounding_caption, box_threshold, text_threshold
        )
        refined_merged_mask = np.zeros(annotated_frame_shape, dtype=np.uint8)
        
        # --- Compute interactiveness with hands via IoU ---
        def _to_xyxy(b):
            # b: ((x1,y1),(x2,y2)) or [x1,y1,x2,y2]
            if isinstance(b, (list, tuple)) and len(b) == 4 and not isinstance(b[0], (list, tuple)):
                x1, y1, x2, y2 = b
            else:
                x1, y1 = b[0]
                x2, y2 = b[1]
            return float(x1), float(y1), float(x2), float(y2)

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

        # Interactive computation moved to mask-based in TrackAnythingPipeline
        interactive_indices = set()
        
        # Track which object IDs are interactive (for visualization filtering)
        interactive_obj_ids = set()
        
        # Store boxes info for visualization: {obj_id: {'box': [x1, y1, x2, y2], 'phrase': str, 'is_human': bool, 'is_interactive': bool}}
        boxes_info = {}
        
        for i in range(len(boxes)):
            bbox = boxes[i]
            phrase = phrases[i]
            if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame_shape[0] * annotated_frame_shape[
                1
            ] * box_size_threshold:
                continue
            # Use box center path for SAM3 when supported; otherwise fall back to box segmentation
            # Pass phrase as text condition for better segmentation
            if hasattr(self.sam, 'segment_with_dino_box_center'):
                interactive_mask = self.sam.segment_with_dino_box_center(
                    origin_frame, [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], 
                    reset_image=reset_image, 
                    phrase=phrase if isinstance(phrase, str) else None
                )
            else:
                interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            # Filter by size
            mask_area = np.sum(interactive_mask > 0)
            if mask_area > annotated_frame_shape[0] * annotated_frame_shape[1] * box_size_threshold:
                continue
            if mask_area < self.min_area:
                continue
            refined_merged_mask = self.add_mask(interactive_mask)
            seg_phrase[self.curr_idx] = phrase
            
            # Store box info for this object
            box_xyxy = [float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1])]
            # is_human: hand, arm, person, or any phrase containing 'in hand' (default not visualized)
            phrase_lower = phrase.lower() if isinstance(phrase, str) else ""
            is_human = ("hand" in phrase_lower) or ("arm" in phrase_lower) or ("person" in phrase_lower) or ("in hand" in phrase_lower)
            is_interactive = i in interactive_indices
            boxes_info[self.curr_idx] = {
                'box': box_xyxy,
                'phrase': phrase,
                'is_human': is_human,
                'is_interactive': is_interactive
            }
            
            # Mark this object ID as interactive if:
            # 1. It has high IoU with hand (is_interactive)
            # 2. OR it's hand/person/object in hand itself (is_human) - these should be hidden by default
            if is_interactive or is_human:
                interactive_obj_ids.add(self.curr_idx)
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame_shape, seg_phrase, interactive_obj_ids, boxes_info
