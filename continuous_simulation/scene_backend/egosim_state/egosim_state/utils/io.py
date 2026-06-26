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
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import Imath
import numpy as np
import OpenEXR
import torch

from egosim_state.ext.lietorch import SE3
from egosim_state.streams.base import FrameAttribute, VideoFrame, VideoStream
from egosim_state.utils.cameras import CameraType
from egosim_state.utils.geometry import se3_matrix_to_se3
from egosim_state.utils.visualization import VideoWriter


logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}_camera.txt"

    @property
    def flow_path(self) -> Path:
        return self.base_path / "flow" / f"{self.artifact_name}.zip"

    @property
    def mask_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.zip"

    @property
    def mask_phrase_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.txt"

    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "egosim_state" / f"{self.artifact_name}_info.pkl"

    @classmethod
    def glob_artifacts(cls, base_path: Path, use_video: bool = False) -> Iterator["ArtifactPath"]:
        if use_video:
            for artifact_path in (base_path / "rgb").glob("*.mp4"):
                artifact_name = artifact_path.stem
                yield cls(base_path, artifact_name)
        else:
            for artifact_path in (base_path / "egosim_state").glob("*_info.pkl"):
                artifact_name = artifact_path.stem.replace("_info", "")
                yield cls(base_path, artifact_name)

    @property
    def meta_vis_path(self) -> Path:
        return self.base_path / "egosim_state" / f"{self.artifact_name}_vis.mp4"

    @property
    def slam_map_path(self) -> Path:
        return self.base_path / "egosim_state" / f"{self.artifact_name}_slam_map.pt"

    @property
    def depth_scale_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}_scale.npz"

    @property
    def interaction_states_path(self) -> Path:
        return self.base_path / "mask" / "interaction_states.npz"

    @property
    def essential_paths(self) -> list[Path]:
        return [
            self.rgb_path,
            self.pose_path,
            self.depth_path,
            self.intrinsics_path,
            self.flow_path,
            self.mask_path,
            self.mask_phrase_path,
            self.meta_info_path,
            self.meta_vis_path,
        ]

    @property
    def eval_metrics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_metrics.pkl"

    @property
    def eval_traj_vis_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_trajectory_vis.png"

    @property
    def eval_gt_pose_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_pose_gt.npz"

    @property
    def eval_gt_intrinsics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_intrinsics_gt.npz"

    @property
    def eval_gt_camera_type_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_camera_gt.txt"

    @property
    def eval_gt_depth_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_depth_gt.zip"

    @property
    def aux_vis_plot_path(self) -> Path:
        return self.base_path / "egosim_state_aux_vis" / f"{self.artifact_name}_plot.png"

    @property
    def aux_vis_traj_path(self) -> Path:
        return self.base_path / "egosim_state_aux_vis" / f"{self.artifact_name}_traj.mp4"


def save_pose_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save OpenCV cam2world matrices as 4x4 matrix in npz file
    if gt:
        pose_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.POSE)
        path = out_path.eval_gt_pose_path
    else:
        pose_list = cached_final_stream.get_stream_attribute(FrameAttribute.POSE)
        path = out_path.pose_path

    pose_list = [
        (frame_idx, pose_data.matrix().cpu().numpy())
        for frame_idx, pose_data in enumerate(pose_list)
        if pose_data is not None
    ]
    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(path, data=pose_data, inds=pose_inds)


def read_pose_artifacts(npz_file_path: Path) -> tuple[np.ndarray, SE3]:
    data = np.load(npz_file_path)
    return data["inds"], se3_matrix_to_se3(data["data"])


def read_pose_artifacts_benchmark(npz_file_path: Path) -> dict:
    data = np.load(npz_file_path)
    return dict(
        ids=data["inds"],
        trajectory=se3_matrix_to_se3(data["data"]),
        runtime=data.get("runtime", None),
        keyframe_ids=data.get("keyframe_ids", None),
        frame_num=len(data["inds"]),
    )


def save_intrinsics_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save intrinsics as [fx, fy, cx, cy] in npz file
    if gt:
        intrinsics_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.CAMERA_TYPE)
        intr_path = out_path.eval_gt_intrinsics_path
        camera_type_path = out_path.eval_gt_camera_type_path
    else:
        intrinsics_list = cached_final_stream.get_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_stream_attribute(FrameAttribute.CAMERA_TYPE)
        intr_path = out_path.intrinsics_path
        camera_type_path = out_path.camera_type_path

    intrinsics_list = [
        (frame_idx, intr_data.cpu().numpy())
        for frame_idx, intr_data in enumerate(intrinsics_list)
        if intr_data is not None
    ]
    if len(intrinsics_list) > 0:
        intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
        intr_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(intr_path, data=intrinsics_data, inds=intrinsics_inds)

    camera_type_list = [
        (frame_idx, camera_type_data)
        for frame_idx, camera_type_data in enumerate(camera_type_list)
        if camera_type_data is not None
    ]
    if len(camera_type_list) > 0:
        camera_type_path.parent.mkdir(exist_ok=True, parents=True)
        with camera_type_path.open("w") as f:
            for frame_idx, camera_type_data in camera_type_list:
                f.write(f"{frame_idx}: {camera_type_data.name}\n")


def read_intrinsics_artifacts(
    intr_file_path: Path, camera_file_path: Path | None = None
) -> tuple[np.ndarray, torch.Tensor, list[CameraType]]:
    data = np.load(intr_file_path)
    inds, intrinsics = data["inds"], torch.from_numpy(data["data"])
    if camera_file_path is None or not camera_file_path.exists():
        assert intrinsics.shape[1] == 4
        camera_types = [CameraType.PINHOLE] * intrinsics.shape[0]

    else:
        with camera_file_path.open("r") as f:
            camera_types = [CameraType[line.split(":")[1].strip()] for line in f.readlines()]

    return inds, intrinsics, camera_types


def save_rgb_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, boxes_info: dict[int, dict] | None = None) -> None:
    # Save original RGB as H264-encoded video.
    # If boxes_info is provided, draw bounding boxes on the RGB frames.
    with VideoWriter(out_path.rgb_path, cached_final_stream.fps()) as rgb_writer:
        for frame_idx, frame_data in enumerate(cached_final_stream):
            rgb_frame = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
            """
            # Draw boxes if available for this frame
            if boxes_info and frame_idx in boxes_info:
                frame_boxes = boxes_info[frame_idx]
                for obj_id, box_info in frame_boxes.items():
                    box = box_info['box']  # [x1, y1, x2, y2]
                    phrase = box_info.get('phrase', '')
                    is_human = box_info.get('is_human', False)
                    is_interactive = box_info.get('is_interactive', False)
                    
                    # Choose color: red for hand/person, green for interactive objects, blue for others
                    if is_human:
                        color = (255, 0, 0)  # Red for hand/person
                        label = f"[H] {phrase}"
                    elif is_interactive:
                        color = (0, 255, 0)  # Green for interactive objects
                        label = f"[I] {phrase}"
                    else:
                        color = (0, 100, 255)  # Orange/blue for non-interactive objects
                        label = f"{phrase}"
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(rgb_frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(rgb_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            """
            rgb_writer.write(rgb_frame)


def read_rgb_artifacts(rgb_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read RGB from H264-encoded video.
    """
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        yield frame_idx, rgb


def save_depth_scale_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    """Save per-frame depth scale factors used for SLAM alignment."""
    from egosim_state.streams.base import FrameAttribute
    
    # Collect scale factors from frames
    scale_dict = {}
    for frame_idx, frame in enumerate(cached_final_stream):
        if hasattr(frame, 'depth_scale') and frame.depth_scale is not None:
            scale_dict[frame_idx] = float(frame.depth_scale)
    
    if len(scale_dict) > 0:
        path = out_path.depth_scale_path
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(path, **{str(k): v for k, v in scale_dict.items()})
        logger.info(f"Saved depth scales for {len(scale_dict)} frames to {path}")


def read_depth_scale_artifacts(scale_path: Path) -> dict[int, float]:
    """Read per-frame depth scale factors."""
    if not scale_path.exists():
        return {}
    
    data = np.load(scale_path, allow_pickle=True)
    scale_dict = {int(k): float(v) for k, v in data.items()}
    return scale_dict


def read_interaction_states_artifacts(states_path: Path) -> dict[int, dict[int, dict]]:
    """Read per-frame per-object interaction states.
    
    Returns:
        Dict mapping frame_idx -> obj_id -> interaction_info
        interaction_info contains: {'is_interacting', 'interacting_with', 'iou', 'depth_diff'}
    """
    if not states_path.exists():
        return {}
    
    data = np.load(states_path, allow_pickle=True)
    states_dict = {}
    for k, v in data.items():
        frame_idx = int(k)
        # v is a dict of obj_id -> interaction_info
        obj_states = {}
        if isinstance(v, dict):
            for obj_id_str, info in v.items():
                obj_id = int(obj_id_str) if isinstance(obj_id_str, str) else obj_id_str
                obj_states[obj_id] = dict(info) if hasattr(info, 'item') else info
        states_dict[frame_idx] = obj_states
    
    return states_dict


def save_depth_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save metric depth as zipped exr files.
    if gt:
        metric_depth_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.eval_gt_depth_path
    else:
        metric_depth_list = cached_final_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.depth_path

    metric_depth_list = [
        (frame_idx, depth_data.cpu().numpy())
        for frame_idx, depth_data in enumerate(metric_depth_list)
        if depth_data is not None
    ]
    if len(metric_depth_list) > 0:
        path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, metric_depth in metric_depth_list:
                height, width = metric_depth.shape
                header = OpenEXR.Header(width, height)
                header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
                with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                    exr = OpenEXR.OutputFile(f.name, header)
                    # Clip limits must preserve 0 (invalid depth)
                    # We only want to clip limiting the far distance, not forcing 0s to 0.01
                    # So we use a mask for valid positive depths
                    clipped_depth = metric_depth.copy()
                    
                    # 1. Cap large depths to 300m (avoid float16 overflow and extreme outliers)
                    # Note: float16 max is 65504, but >300m is typically noise in this context
                    clipped_depth[clipped_depth > 300.0] = 300.0
                    
                    # 2. Preserve 0 as 0 (invalid), but ensure small positive values are not too small if needed
                    # However, usually we just want to avoid the upper bound overflow.
                    # np.clip approaches:
                    # If we used np.clip(depth, 0.01, 300), we destroyed 0s. 
                    
                    # Just ensure no overflow for float16:
                    clipped_depth = np.clip(clipped_depth, 0, 65504)
                    
                    exr.writePixels({"Z": clipped_depth.astype(np.float16).tobytes()})
                    exr.close()
                    z.write(f.name, f"{frame_idx:05d}.exr")


def read_depth_artifacts(zip_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read metric depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    # Sometimes EXR loader might fail, we return all nan maps.
                    logger.warning(f"Failed to load EXR file {zip_file_path}-{file_name}. Returning all nan maps.")
                    assert valid_width > 0 and valid_height > 0
                    yield (
                        frame_idx,
                        torch.full(
                            (valid_height, valid_width),
                            float("nan"),
                            dtype=torch.float32,
                        ),
                    )
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                yield frame_idx, torch.from_numpy(depth_data.copy()).float()


def read_instance_artifacts(
    zip_file_path: Path,
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read instance mask from zipped PNG files.
    """
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                yield frame_idx, torch.from_numpy(mask.copy()).byte()


def read_instance_phrases(instance_phrase_path: Path) -> dict[int, str]:
    """
    Read instance phrases from txt file.
    """
    instance_phrases = {}
    with instance_phrase_path.open("r") as f:
        for line in f.readlines():
            idx, phrase = line.split(":")
            instance_phrases[int(idx)] = phrase.strip()
    return instance_phrases


def read_boxes_artifacts(boxes_path: Path) -> dict[int, dict]:
    """
    Read boxes info from npz file.
    
    Returns:
        dict[int, dict]: {frame_idx: {obj_id: {'box': [x1,y1,x2,y2], 'phrase': str, 'is_human': bool, 'is_interactive': bool}}}
    """
    if not boxes_path.exists():
        return {}
    
    data = np.load(boxes_path, allow_pickle=True)
    boxes_info = {}
    for key in data.keys():
        frame_idx = int(key)
        boxes_data = data[key].item()  # Convert numpy array to dict
        boxes_info[frame_idx] = boxes_data
    return boxes_info


def save_instance_masks(out_path: ArtifactPath, cached_stream: VideoStream) -> None:
    """
    Save instance mask and phrases from the stream with TrackAnything results.
    """
    from egosim_state.streams.base import VideoFrame
    
    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")


def save_instance_and_interactive(out_path: ArtifactPath, cached_stream: VideoStream) -> None:
    """
    Save instance mask and interactive object IDs from the stream with TrackAnything results.
    This should be called with the slam_stream which has the instance and interactive_obj_ids.
    """
    from egosim_state.streams.base import VideoFrame
    
    print(f"DEBUG: save_instance_and_interactive called with stream type: {type(cached_stream)}")
    print(f"DEBUG: Stream length: {len(cached_stream)}")
    print(f"DEBUG: Stream attributes: {cached_stream.attributes()}")
    
    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")
    
    # Save interactive object IDs per frame as npz file.
    print("DEBUG: Checking interactive_obj_ids in slam_stream...")
    interactive_list = []
    for frame_idx, frame_data in enumerate(cached_stream):
        has_attr = hasattr(frame_data, 'interactive_obj_ids')
        if frame_idx < 5:
            print(f"  Frame {frame_idx}: hasattr(interactive_obj_ids)={has_attr}")
            if has_attr:
                print(f"    Value: {frame_data.interactive_obj_ids}")
        if has_attr and frame_data.interactive_obj_ids is not None:
            interactive_list.append((frame_idx, frame_data.interactive_obj_ids))
    
    print(f"DEBUG: Found {len(interactive_list)} frames with interactive_obj_ids")
    
    if len(interactive_list) > 0:
        interactive_path = out_path.mask_path.parent / "interactive.npz"
        interactive_path.parent.mkdir(exist_ok=True, parents=True)
        # Save as dict: {frame_idx: list of interactive obj ids}
        interactive_dict = {str(frame_idx): list(obj_ids) for frame_idx, obj_ids in interactive_list}
        np.savez_compressed(interactive_path, **interactive_dict)
        print(f"Saved interactive object IDs to {interactive_path} for {len(interactive_list)} frames")
    else:
        print("WARNING: No frames with interactive_obj_ids found in slam_stream, skipping interactive.npz")


def save_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, boxes_info: dict[int, dict] | None = None) -> None:
    """
    Save each attribute independently.
    Note: Instance mask and interactive_obj_ids are saved separately via save_instance_and_interactive()
    """

    # Save OpenCV cam2world matrices as 4x4 matrix in npz file
    save_pose_artifacts(out_path, cached_final_stream)

    # Save intrinsics as [fx, fy, cx, cy] in npz file
    save_intrinsics_artifacts(out_path, cached_final_stream)

    # Save original RGB as H264-encoded video (with boxes drawn if provided).
    save_rgb_artifacts(out_path, cached_final_stream, boxes_info)

    # Save metric depth as zipped exr files.
    save_depth_artifacts(out_path, cached_final_stream)

    # Save per-frame depth scale factors (for metric alignment; used by viser and downstream)
    save_depth_scale_artifacts(out_path, cached_final_stream)
