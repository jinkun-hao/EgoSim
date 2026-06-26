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
import pickle

from pathlib import Path

import torch

from omegaconf import DictConfig

from egosim_state.slam.system import SLAMOutput, SLAMSystem
from egosim_state.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from egosim_state.utils import io
from egosim_state.utils.cameras import CameraType
from egosim_state.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    AdaptiveDepthProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        self.track_anything_processor = None
        if self.init_cfg.instance is not None:
            total_frames = len(video_stream)
            if total_frames <= 0:
                raise ValueError("Video stream has no frames, cannot run TrackAnything.")
            self.track_anything_processor = TrackAnythingProcessor(
                self.init_cfg.instance.phrases,
                add_sky=self.init_cfg.instance.add_sky,
                sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                use_sam3=self.init_cfg.instance.get("use_sam3", True),
                continuous_tracking=self.init_cfg.instance.get("continuous_tracking", True),
                depth_distance_thresh=self.init_cfg.instance.get("depth_distance_thresh", 0.15),
                total_frames=total_frames,
                force_last_frame_keyframe=self.init_cfg.instance.get("force_last_frame_sam_keyframe", True),
            )
            self.depth_filter_enabled = self.init_cfg.instance.get("depth_filter", True)
            init_processors.append(self.track_anything_processor)
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for slam_stream, output_stream, artifact_path in zip(slam_streams, output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Apply depth filtering to interactive objects before saving
            # Use instance from slam_stream and depth from output_stream
            if self.track_anything_processor is not None and getattr(self, 'depth_filter_enabled', True):
                logger.info("Filtering interactive objects by depth distance...")
                for frame_idx, (slam_frame, out_frame) in enumerate(zip(slam_stream, output_stream)):
                    if (hasattr(out_frame, 'depth') and out_frame.depth is not None and 
                        hasattr(slam_frame, 'instance') and slam_frame.instance is not None):
                        # Create a temporary frame-like object with instance from slam_frame and depth from out_frame
                        class TempFrame:
                            pass
                        temp_frame = TempFrame()
                        temp_frame.instance = slam_frame.instance
                        temp_frame.depth = out_frame.depth
                        temp_frame.interactive_obj_ids = self.track_anything_processor.all_interactive_obj_ids.get(frame_idx, set())
                        self.track_anything_processor.filter_interactive_by_depth(frame_idx, temp_frame)
                
                # Check and update interaction states for objects' last non-keyframe appearances
                logger.info("Checking last frame interactions for objects...")
                keyframe_indices = slam_output.slam_map.dense_disp_frame_inds if slam_output.slam_map else []
                self.track_anything_processor.check_and_update_last_frame_interactions(
                    slam_stream, output_stream, keyframe_indices
                )
            elif self.track_anything_processor is not None:
                logger.info("Depth filtering disabled, skipping interactive object depth filtering.")
            
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                # Get boxes_info from TrackAnythingProcessor if available
                boxes_info = None
                if self.track_anything_processor is not None:
                    boxes_info = self.track_anything_processor.all_boxes_info
                # Save most artifacts from output_stream (has depth alignment), with boxes drawn on RGB
                io.save_artifacts(artifact_path, output_stream, boxes_info)
                # Save instance mask from slam_stream (has TrackAnything results)
                io.save_instance_masks(artifact_path, slam_stream)
                # Save interactive_obj_ids and boxes_info from TrackAnythingProcessor
                if self.track_anything_processor is not None:
                    self.track_anything_processor.save_interactive_obj_ids(artifact_path.base_path)
                    self.track_anything_processor.save_boxes_info(artifact_path.base_path)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
