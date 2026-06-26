# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import Iterator

import numpy as np
import torch

from egosim_state.priors.depth import DepthEstimationInput, make_depth_model
from egosim_state.priors.depth.alignment import align_inv_depth_to_depth
from egosim_state.priors.depth.priorda import PriorDAModel
from egosim_state.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from egosim_state.priors.geocalib import GeoCalib
from egosim_state.priors.track_anything import TrackAnythingPipeline
from egosim_state.slam.interface import SLAMOutput
from egosim_state.streams.base import (CachedVideoStream, FrameAttribute,
                               StreamProcessor, VideoFrame, VideoStream)
from egosim_state.utils.cameras import CameraType
from egosim_state.utils.logging import pbar
from egosim_state.utils.misc import unpack_optional
from egosim_state.utils.model_paths import get_dav3_nested_model_path
from egosim_state.utils.morph import erode

logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
        use_sam3: bool = True,
        continuous_tracking: bool = True,
        depth_distance_thresh: float = 0.15,
        total_frames: int = 0,
        force_last_frame_keyframe: bool = True,
    ) -> None:
        """
        Initialize the TrackAnythingProcessor.
        
        Args:
            mask_phrases: List of phrases to detect and mask.
            add_sky: Whether to add sky detection to mask phrases.
            sam_run_gap: Interval between keyframes for detection.
            mask_expand: Number of pixels to erode the mask.
            use_sam3: Whether to use SAM3 instead of SAM.
            continuous_tracking: If True (default), objects are continuously tracked
                across all frames. If False, objects are only detected at keyframes
                and the mask is repeated for intermediate frames without tracking.
            total_frames: Total number of frames in the video (used to ensure last frame is a keyframe).
            force_last_frame_keyframe: If True (default), always run SAM detection on the last frame.
        """
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(
            self.mask_phrases, 
            sam_points_per_side=50, 
            sam_run_gap=self.sam_run_gap,
            use_sam3=use_sam3,
            continuous_tracking=continuous_tracking,
            total_frames=total_frames,
            force_last_frame_keyframe=force_last_frame_keyframe,
        )
        self.mask_expand = mask_expand
        self.depth_distance_thresh = depth_distance_thresh
        # Collect interactive object IDs for each frame
        self.all_interactive_obj_ids: dict[int, set[int]] = {}
        # Collect detailed interaction states for each object in each frame
        # Format: {frame_idx: {obj_id: {'is_interacting': bool, 'interacting_with': [hand_ids], 'iou': float, 'depth_diff': float}}}
        self.all_object_interaction_states: dict[int, dict[int, dict]] = {}
        # Track last appearance frame for each object
        self.object_last_frame: dict[int, int] = {}  # {obj_id: last_frame_idx}
        # Collect boxes info for each frame
        self.all_boxes_info: dict[int, dict] = {}

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases, frame.interactive_obj_ids, boxes_info = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        # Store interactive_obj_ids and boxes_info for stable saving
        self.all_interactive_obj_ids[frame_idx] = frame.interactive_obj_ids
        self.all_boxes_info[frame_idx] = boxes_info
        
        # Track last appearance frame for each object
        if frame.instance is not None:
            unique_obj_ids = torch.unique(frame.instance).cpu().numpy()
            for obj_id in unique_obj_ids:
                if obj_id > 0:  # Skip background
                    self.object_last_frame[int(obj_id)] = frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame
    
    def filter_interactive_by_depth(self, frame_idx: int, frame):
        """
        Post-process to filter interactive objects by depth distance to hand.
        Should be called after depth is available.
        
        Args:
            frame_idx: Frame index
            frame: Object with instance mask, depth, and interactive_obj_ids attributes
            
        Returns:
            Updated frame (or None if frame doesn't have required attributes)
        """
        if not hasattr(frame, 'depth') or frame.depth is None:
            return frame
        if not hasattr(frame, 'instance') or frame.instance is None:
            return frame
        if not hasattr(frame, 'interactive_obj_ids') or frame.interactive_obj_ids is None:
            return frame
        
        # Get detailed interaction info from tracker
        filtered_ids, interaction_details = self.tracker.filter_interactive_by_depth(
            frame.instance, frame.depth, frame.interactive_obj_ids, self.depth_distance_thresh,
            return_details=True
        )
        frame.interactive_obj_ids = filtered_ids
        # Update stored interactive_obj_ids
        self.all_interactive_obj_ids[frame_idx] = filtered_ids
        
        # Store detailed interaction states for all objects in this frame
        if interaction_details and frame_idx not in self.all_object_interaction_states:
            self.all_object_interaction_states[frame_idx] = interaction_details
        
        return frame
    
    def check_and_update_last_frame_interactions(self, slam_stream, output_stream, keyframe_indices):
        """
        Check objects whose last frame is not a keyframe and update their interaction states.
        This ensures all objects have interaction info for their final appearance.
        
        Args:
            slam_stream: CachedVideoStream with instance masks
            output_stream: CachedVideoStream with depth maps  
            keyframe_indices: List of keyframe indices
        """
        print(f"Checking last frame interactions for {len(self.object_last_frame)} objects...")
        frames_to_check = set()
        
        # Find objects whose last frame is not a keyframe
        for obj_id, last_frame_idx in self.object_last_frame.items():
            if last_frame_idx not in keyframe_indices:
                # This object's last frame is not a keyframe, need to check interaction
                if last_frame_idx not in self.all_object_interaction_states:
                    frames_to_check.add(last_frame_idx)
        
        if not frames_to_check:
            print("  All objects' last frames already have interaction states.")
            return
        
        print(f"  Need to check interaction for {len(frames_to_check)} non-keyframe last frames")
        
        # Process each frame that needs checking
        for frame_idx in sorted(frames_to_check):
            if frame_idx >= len(slam_stream) or frame_idx >= len(output_stream):
                continue
                
            slam_frame = slam_stream[frame_idx]
            out_frame = output_stream[frame_idx]
            
            if (hasattr(out_frame, 'depth') and out_frame.depth is not None and 
                hasattr(slam_frame, 'instance') and slam_frame.instance is not None):
                
                # Get all objects in this frame
                unique_obj_ids = torch.unique(slam_frame.instance).cpu().numpy()
                interactive_obj_ids = self.all_interactive_obj_ids.get(frame_idx, set())
                
                # Perform interaction detection
                _, interaction_details = self.tracker.filter_interactive_by_depth(
                    slam_frame.instance, out_frame.depth, interactive_obj_ids, 
                    self.depth_distance_thresh, return_details=True
                )
                
                if interaction_details:
                    self.all_object_interaction_states[frame_idx] = interaction_details
                    print(f"  Frame {frame_idx}: Updated interaction states for {len(interaction_details)} objects")
        
        print(f"  Completed checking {len(frames_to_check)} frames")
    
    def save_interactive_obj_ids(self, out_path):
        """Save collected interactive object IDs to npz file.
        
        Args:
            out_path: The base_path directory (e.g., egosim_state_results_xxx/)
        """
        import numpy as np
        from pathlib import Path
        
        if len(self.all_interactive_obj_ids) > 0:
            # out_path is base_path, so interactive.npz goes to base_path/mask/interactive.npz
            interactive_path = Path(out_path) / "mask" / "interactive.npz"
            interactive_path.parent.mkdir(exist_ok=True, parents=True)
            # Save as dict: {frame_idx: list of interactive obj ids}
            interactive_dict = {str(frame_idx): list(obj_ids) for frame_idx, obj_ids in self.all_interactive_obj_ids.items()}
            np.savez_compressed(interactive_path, **interactive_dict)
            print(f"Saved interactive object IDs to {interactive_path} for {len(self.all_interactive_obj_ids)} frames")
            
            # Also save detailed interaction states
            if len(self.all_object_interaction_states) > 0:
                interaction_states_path = Path(out_path) / "mask" / "interaction_states.npz"
                # Convert to serializable format
                states_dict = {}
                for frame_idx, obj_states in self.all_object_interaction_states.items():
                    states_dict[str(frame_idx)] = obj_states
                np.savez_compressed(interaction_states_path, **states_dict)
                print(f"Saved detailed interaction states to {interaction_states_path} for {len(self.all_object_interaction_states)} frames")
            
            return interactive_path
        else:
            print("WARNING: No interactive_obj_ids collected")
            return None
    
    def save_boxes_info(self, out_path):
        """Save collected boxes info to npz file.
        
        Args:
            out_path: The base_path directory (e.g., egosim_state_results_xxx/)
        """
        import numpy as np
        from pathlib import Path
        
        if len(self.all_boxes_info) > 0:
            # out_path is base_path, so boxes.npz goes to base_path/mask/boxes.npz
            boxes_path = Path(out_path) / "mask" / "boxes.npz"
            boxes_path.parent.mkdir(exist_ok=True, parents=True)
            # Save as dict: {frame_idx: boxes_info}
            # Each boxes_info is a dict {obj_id: {'box': [x1,y1,x2,y2], 'phrase': str, 'is_human': bool, 'is_interactive': bool}}
            np.savez_compressed(boxes_path, **{str(frame_idx): boxes_info for frame_idx, boxes_info in self.all_boxes_info.items()})
            print(f"Saved boxes info to {boxes_path} for {len(self.all_boxes_info)} frames")
            return boxes_path
        else:
            print("WARNING: No boxes_info collected")
            return None


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_result: torch.Tensor = unpack_optional(
            self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale, bias = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    scale, bias = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale, bias)
                scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                self.cache_scale_bias = (scale, bias)

                video_inv_depth = video_depth_inv_depth * scale + bias
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


class MultiviewDepthProcessor(StreamProcessor):
    """
    Use multi-view depth model (e.g. DAv3, MapAnything, CAPA) to estimate depth map for each frame.
    To ensure that the depth maps are consistent with the SLAM map/pose (metric), we condition the depth model either with
    (a) sparse points, or (b) camera poses & intrinsics.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    (Multi-view input video frames are currently not supported)
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,                  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,       # This is found to cause jittering for some scenes due to abrupt context changes.
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[VideoFrame] = []
        self.n_frames = 0

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.model == "mvd_dav3":
            try:
                from depth_anything_3.api import DepthAnything3
                from depth_anything_3.api import logger as dav3_logger
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "depth-anything-3 not found. Please reinstall egosim_state with `pip install --no-build-isolation -e .[dav3]`"
                )

            dav3_logger.level = 0  # Disable logging timing information
            self.dav3_api = DepthAnything3.from_pretrained(str(get_dav3_nested_model_path()))
            self.dav3_api = self.dav3_api.cuda().eval()

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("MultiviewDepthProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame)
            yield frame

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        current_sliding_window: list[VideoFrame] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        
        # Get poses for alignment
        try:
            all_poses = self.slam_output.get_view_trajectory(0)
        except Exception:
            all_poses = None

        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating multi-view depth"):
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                # Grab all neighboring keyframes to anchor the current sliding window.
                # Note that we remove redundant keyframes that already exist in the current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds])
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                # Perform inference
                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
                    process_res_method="lower_bound_resize",  # Keep aspect ratio
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]

                # --- Scale Alignment fix for "size flickering" issue ---
                # DAv3 metrics can drift across sliding windows. We force align to SLAM sparse depth.
                if all_poses is not None:
                    align_ratios = []
                    for i, idx in enumerate(current_sliding_window_idx):
                        curr_frame = current_sliding_window[i]
                        # Project sparse map points
                        sparse_depth = self.slam_output.slam_map.project_map(
                            idx, 0, curr_frame.size(),
                            unpack_optional(curr_frame.intrinsics),
                            all_poses[idx],
                            unpack_optional(curr_frame.camera_type),
                            infill=False
                        )
                        pred = sw_depth[i]
                        mask = (sparse_depth > 0) & (pred > 1e-3)
                        if mask.sum() > 50:
                            ratio = torch.median(sparse_depth[mask] / pred[mask])
                            align_ratios.append(ratio)
                    
                    if len(align_ratios) > 0:
                        window_scale = torch.median(torch.stack(align_ratios))
                        # Soft update if we had a previous scale? No, just fix this window.
                        sw_depth *= window_scale
                        # Store per-frame scale for later use (e.g., in visualization)
                        per_frame_scales = align_ratios
                    else:
                        window_scale = 1.0
                        per_frame_scales = [torch.tensor(1.0)] * len(current_sliding_window_idx)
                else:
                    window_scale = 1.0
                    per_frame_scales = [torch.tensor(1.0)] * len(current_sliding_window_idx)
                # -----------------------------------------------------

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                # Linearly interpolate the trailing depth with new depth
                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    # Store the depth scale for this frame
                    if sw_idx < len(per_frame_scales):
                        frame.depth_scale = per_frame_scales[sw_idx].item()
                    else:
                        frame.depth_scale = window_scale.item() if isinstance(window_scale, torch.Tensor) else window_scale
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")
