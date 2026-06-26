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

import argparse
import asyncio
import cv2
import logging
import socket
import time

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import viser.transforms as tf

from matplotlib import cm
from PIL import Image
from rich.logging import RichHandler

from egosim_state.slam.interface import SLAMMap
from egosim_state.utils.cameras import CameraType
from egosim_state.utils.depth import reliable_depth_mask_range
from egosim_state.utils.geometry import align_points
from egosim_state.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    """Save point cloud to PLY file. Uses Open3D if available, else writes ASCII PLY."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(out_path), pcd)
        logger.info(f"Saved global point cloud to {out_path} ({len(points)} points)")
    except ImportError:
        # Fallback: write ASCII PLY manually
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {r} {g} {b}\n")
        logger.info(f"Saved global point cloud to {out_path} ({len(points)} points, ASCII fallback)")


@dataclass
class GlobalContext:
    artifacts: list[ArtifactPath]


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None
    _show_camera: bool = True
    _frustum_scale: float = 0.15

    def __post_init__(self):
        self._frustum_scale = self.frustum_handle.scale
        self.visible = False

    @property
    def show_camera(self) -> bool:
        return self._show_camera

    @show_camera.setter
    def show_camera(self, value: bool):
        self._show_camera = value
        self.frustum_handle.visible = value
        if value:
            self.frustum_handle.scale = self._frustum_scale
        else:
            self.frustum_handle.scale = 0.0

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        if value and not self._show_camera:
            self.frustum_handle.visible = False
            self.frustum_handle.scale = 0.0
        else:
            self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value


class ClientClosures:
    """
    All class methods automatically capture 'self', ensuring proper locals.
    """

    

    def _integrate_frames_voxelblockgrid_tsdf(
            self, frames_data: list[dict], voxel_size: float, trunc_multiplier: float
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate multi-frame depth/color into a VoxelBlockGrid (TSDF) and extract a point cloud (Open3D 0.19.0).
        Args:
            frames_data: list of per-frame dicts with 'depth'/'mask'/'color'/'intrinsic'/'pose'.
            voxel_size: voxel size (meters)
            trunc_multiplier: TSDF truncation distance multiplier (truncation = voxel_size * trunc_multiplier)
        Returns:
            points: point coordinates (N,3)
            colors: point colors (N,3), uint8
        """
        import open3d as o3d
        import open3d.core as o3c
        import open3d.t.geometry as t_geometry
        import numpy as np

        # Empty input
        if len(frames_data) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        
        # Device: CUDA if available, else CPU
        device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
        sdf_trunc = voxel_size * trunc_multiplier  # TSDF truncation
        print(f"  Using device: {device}, voxel_size={voxel_size:.4f}m, sdf_trunc={sdf_trunc:.4f}m")

        try:
            # Initialize VoxelBlockGrid (recommended layout for Open3D 0.19.0)
            vbg = t_geometry.VoxelBlockGrid(
                attr_names=["tsdf", "weight", "color"],
                attr_dtypes=[o3c.float32, o3c.uint16, o3c.uint16],
                attr_channels=[1, 1, 3],  # Open3D 0.19.0 accepts plain integers for attr_channels
                voxel_size=voxel_size,
                block_resolution=16,
                block_count=len(frames_data) * 100,
                device=device,
            )

            for i, frame in enumerate(frames_data):
                if i % 10 == 0:
                    print(f"  Integrating frame {i+1}/{len(frames_data)} (VoxelBlockGrid)...")

                # 1. Depth: set depth outside mask to nan (invalid)
                depth = frame['depth'].astype(np.float32)
                mask = frame['mask'].astype(bool)
                depth[~mask] = np.nan

                # 2. Color: ensure (H,W,3), normalize to [0,1]
                color = frame['color'].astype(np.float32) / 255.0
                if color.ndim != 3 or color.shape[-1] != 3:
                    raise ValueError(f"Frame {i} color shape {color.shape} is not (H,W,3)")

                # 3. Convert to Open3D Tensor (float32 for depth and color)
                depth_o3d = o3c.Tensor(depth, dtype=o3c.float32, device=device)
                color_o3d = o3c.Tensor(color, dtype=o3c.float32, device=device)

                # 4. Intrinsics: must live on CPU
                fx, fy, cx, cy = frame['intrinsic']
                intrinsic = o3c.Tensor(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3c.float64, device=o3c.Device("CPU:0")
                )

                # 5. Extrinsics: c2w → w2c (camera-to-world to world-to-camera), CPU
                c2w = frame['pose']
                w2c = np.eye(4, dtype=np.float64)
                w2c[:3, :3] = c2w[:3, :3].T
                w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
                extrinsic = o3c.Tensor(w2c, dtype=o3c.float64, device=o3c.Device("CPU:0"))

                # 6. Build t.geometry.Image
                depth_image = t_geometry.Image(depth_o3d)
                color_image = t_geometry.Image(color_o3d)

                # 7. Active voxel block coordinates
                frustum_block_coords = vbg.compute_unique_block_coordinates(
                    depth=depth_image,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    depth_scale=1.0,
                    depth_max=3.0,
                )

                # 8. Fuse into VoxelBlockGrid
                vbg.integrate(
                    block_coords=frustum_block_coords,
                    depth=depth_image,
                    color=color_image,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    depth_scale=1.0,
                    depth_max=3.0,
                )

            # Extract point cloud
            print("  Extracting point cloud from VoxelBlockGrid...")
            pcd = vbg.extract_point_cloud(weight_threshold=3.0)
            
            if pcd.is_empty():
                print("  Warning: Extracted point cloud is empty!")
                return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
            
            # Convert to NumPy
            points = pcd.point.positions.cpu().numpy()
            colors = (pcd.point.colors.cpu().numpy() * 255).astype(np.uint8)
            
            print(f"  VoxelBlockGrid extraction: {len(points)} points")
            return points, colors

        except Exception as e:
            print(f"  Error during integration/extraction: {str(e)}")
            # Print full traceback
            import traceback
            traceback.print_exc()
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        # Don't await to not block the rest of the coroutine.
        self.task = asyncio.create_task(_run())

        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0
        self.global_pcd_handle: viser.PointCloudHandle | None = None
        self._pcd_center: np.ndarray | None = None
        self._pcd_extent: float = 1.0

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        all_artifacts = self.global_context().artifacts

        with self.client.gui.add_folder("Sample"):
            self.gui_id = self.client.gui.add_slider(
                "Artifact ID", min=0, max=len(all_artifacts) - 1, step=1, initial_value=0
            )
            gui_id_changer = self.client.gui.add_button_group(label="ID +/-", options=["Prev", "Next"])

            @gui_id_changer.on_click
            async def _(_) -> None:
                if gui_id_changer.value == "Prev":
                    self.gui_id.value = (self.gui_id.value - 1) % len(all_artifacts)
                else:
                    self.gui_id.value = (self.gui_id.value + 1) % len(all_artifacts)

            self.gui_name = self.client.gui.add_text("Artifact Name", "")
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_id.on_update(self.on_sample_update)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=0.01, step=0.001, initial_value=0.001
            )

            # Update point cloud size
            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value

            self.gui_show_cameras = self.client.gui.add_checkbox(
                "Show Cameras",
                initial_value=True,
                hint="Show/hide camera frustums",
            )

            @self.gui_show_cameras.on_update
            async def _(_) -> None:
                visible = self.gui_show_cameras.value
                for frame_node in self.scene_frame_handles:
                    frame_node.show_camera = visible

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node._frustum_scale = self.gui_frustum_size.value
                    if frame_node._show_camera:
                        frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox(
                "Colorful Frustum",
                initial_value=False,
            )

            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            # === Basic filtering ===
            self.gui_filter_interactive = self.client.gui.add_checkbox(
                "Filter Interactive Objects",
                initial_value=True,
                hint="Filter interactive objects (held items, hands, arms, etc.)",
            )
            self.gui_filter_interactive.on_update(self.on_sample_update)

            self.gui_vis_mode = self.client.gui.add_dropdown(
                "Visualization Mode",
                options=["All", "Background Only", "Interactive Objects Only", "Held Objects (Latest)", "Target Labels (Latest)"],
                initial_value="All",
                hint="All | Background Only | Interactive Objects Only | Held objects (latest) | Target labels (latest frame per label)",
            )
            self.gui_vis_mode.on_update(self.on_sample_update)

            self.gui_target_labels = self.client.gui.add_text(
                "Target Labels",
                initial_value="white cup in hand, white cup, white cup lid in hand, white cup lid",
                hint="Comma-separated labels for Target Labels (Latest); matched against instance_phrases",
            )
            self.gui_target_labels.on_update(self.on_sample_update)
            
            self.client.gui.add_text("Global Scene Reconstruction", "────────────────────", disabled=True)
            
            # === Global point cloud ===
            self.gui_show_global_pcd = self.client.gui.add_checkbox(
                "Enable Global Point Cloud",
                initial_value=True,
                hint="Show globally aligned scene (accumulated background + static objects)",
            )
            self.gui_show_global_pcd.on_update(self.on_sample_update)
            
            # Sub-options for global cloud (when enabled)
            self.gui_use_last_frame_objects = self.client.gui.add_checkbox(
                "  ↳ Add Last Frame Objects",
                initial_value=True,
                hint="Add state of non-interactive objects from the last frame",
            )
            self.gui_use_last_frame_objects.on_update(self.on_sample_update)
            
            self.gui_voxel_downsample = self.client.gui.add_checkbox(
                "  ↳ Incremental Fusion",
                initial_value=True,
                hint="Incremental fusion: add new background points not covered by the previous view (replaces voxel downsampling)",
            )
            self.gui_voxel_downsample.on_update(self.on_sample_update)
            
            self.gui_voxel_size = self.client.gui.add_slider(
                "    Overlap Threshold (m)",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=0.05,
                hint="Depth threshold for overlap: small depth delta and unoccluded → treat as overlap",
            )
            self.gui_voxel_size.on_update(self.on_sample_update)
            
            self.gui_statistical_outlier_removal = self.client.gui.add_checkbox(
                "  ↳ Statistical Outlier Removal",
                initial_value=True,
                hint="Statistical outlier removal for noise and flying pixels (e.g. dynamic edges)",
            )
            self.gui_statistical_outlier_removal.on_update(self.on_sample_update)
            
            self.gui_outlier_nb_neighbors = self.client.gui.add_slider(
                "    Neighbors",
                min=5,
                max=50,
                step=1,
                initial_value=20,
                hint="Number of neighbors for statistical analysis",
            )
            self.gui_outlier_nb_neighbors.on_update(self.on_sample_update)
            
            self.gui_outlier_std_ratio = self.client.gui.add_slider(
                "    Std Ratio",
                min=0.5,
                max=5.0,
                step=0.1,
                initial_value=2.0,
                hint="Std multiplier; lower = stricter (1.0–2.0 typical)",
            )
            self.gui_outlier_std_ratio.on_update(self.on_sample_update)
            
            self.gui_mask_dilation = self.client.gui.add_slider(
                "Mask Dilation (pixels)",
                min=0,
                max=20,
                step=1,
                initial_value=5,
                hint="Dilate interactive-object masks to reduce boundary noise",
            )
            self.gui_mask_dilation.on_update(self.on_sample_update)
            
            self.client.gui.add_text("Fusion Method", "────────────────────", disabled=True)
            
            self.gui_use_tsdf = self.client.gui.add_checkbox(
                "  ↳ Use TSDF Fusion",
                initial_value=True,
                hint="TSDF fusion for alignment errors and surface smoothing",
            )
            self.gui_use_tsdf.on_update(self.on_sample_update)
            
            self.gui_tsdf_implementation = self.client.gui.add_dropdown(
                "    TSDF Implementation",
                options=["Classic", "VoxelBlockGrid"],
                initial_value="Classic",
                hint="Classic: CPU ScalableTSDFVolume | VoxelBlockGrid: sparse GPU TSDF (faster)",
            )
            self.gui_tsdf_implementation.on_update(self.on_sample_update)
            
            self.gui_tsdf_voxel_size = self.client.gui.add_slider(
                "    TSDF Voxel Size (m)",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=0.0025,
                hint="TSDF voxel size; smaller = more detail, more VRAM",
            )
            self.gui_tsdf_voxel_size.on_update(self.on_sample_update)
            
            self.gui_tsdf_trunc_multiplier = self.client.gui.add_slider(
                "    TSDF Truncation",
                min=2.0,
                max=40.0,
                step=1.0,
                initial_value=20.0,
                hint="Truncation multiplier (vs voxel size) for surface smoothing",
            )
            self.gui_tsdf_trunc_multiplier.on_update(self.on_sample_update)

            gui_snapshot = self.client.gui.add_button(
                "Snapshot",
                hint="Take a snapshot of the current scene",
            )

            @gui_snapshot.on_click
            def _(_) -> None:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                file_name = f"{current_artifact.base_path.name}_{current_artifact.artifact_name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

            self.gui_orbit_num_views = self.client.gui.add_slider(
                "Orbit Views", min=4, max=36, step=1, initial_value=8,
                hint="Number of orbit views",
            )
            self.gui_orbit_radius_scale = self.client.gui.add_slider(
                "Orbit Radius Scale", min=0.5, max=3.0, step=0.1, initial_value=1.0,
                hint="Orbit radius scale vs mean camera–center distance",
            )

            gui_orbit_snapshot = self.client.gui.add_button(
                "Orbit Snapshot",
                hint="Save multiple orbit PNGs near the camera path",
            )

            @gui_orbit_snapshot.on_click
            def _(_) -> None:
                self._save_orbit_snapshots()

            self.gui_orbit_video_frames = self.client.gui.add_slider(
                "Orbit Video Frames", min=60, max=360, step=30, initial_value=120,
                hint="Frames for one full 360° orbit video",
            )
            self.gui_orbit_video_fps = self.client.gui.add_slider(
                "Orbit Video FPS", min=15, max=60, step=5, initial_value=30,
                hint="Output video FPS",
            )
            gui_orbit_video = self.client.gui.add_button(
                "Orbit Video",
                hint="Render MP4 of point cloud rotating around center",
            )

            @gui_orbit_video.on_click
            def _(_) -> None:
                self._save_orbit_video()

        await self.on_sample_update(None)

        while True:
            if self.gui_framerate is not None and self.gui_framerate.value > 0:
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(1.0)

    async def on_sample_update(self, _):
        with self.client.atomic():
            self._rebuild_scene()
        self._rebuild_playback_gui()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)
        show_cam = self.gui_show_cameras.value
        for frame_node in self.scene_frame_handles:
            frame_node.show_camera = show_cam

    def _save_orbit_snapshots(self):
        """Generate orbit views around the point cloud center, using camera trajectory as reference."""
        import os

        current_artifact = self.global_context().artifacts[self.gui_id.value]
        out_dir = current_artifact.base_path / current_artifact.artifact_name / "orbit_snapshots"
        os.makedirs(out_dir, exist_ok=True)

        cam_positions = []
        for fh in self.scene_frame_handles:
            pos = fh.frame_handle.position
            cam_positions.append(np.array(pos))

        if len(cam_positions) == 0:
            print("No camera frames available for orbit snapshot.")
            return

        cam_positions = np.stack(cam_positions, axis=0)
        center = cam_positions.mean(axis=0)
        avg_radius = np.linalg.norm(cam_positions - center, axis=1).mean()
        radius = avg_radius * self.gui_orbit_radius_scale.value

        up = np.array([0.0, 0.0, 1.0])
        cam_forward_avg = np.zeros(3)
        for fh in self.scene_frame_handles:
            wxyz = np.array(fh.frame_handle.wxyz)
            rot = tf.SO3(wxyz).as_matrix()
            cam_forward_avg += rot[:, 2]
        cam_forward_avg /= len(self.scene_frame_handles)

        cam_up = np.zeros(3)
        for fh in self.scene_frame_handles:
            wxyz = np.array(fh.frame_handle.wxyz)
            rot = tf.SO3(wxyz).as_matrix()
            cam_up += rot[:, 1]
        cam_up /= len(self.scene_frame_handles)
        if np.linalg.norm(cam_up) > 1e-6:
            up = -cam_up / np.linalg.norm(cam_up)

        num_views = int(self.gui_orbit_num_views.value)
        height_offset = center[2]

        original_wxyz = self.client.camera.wxyz
        original_pos = self.client.camera.position

        print(f"Saving {num_views} orbit snapshots to {out_dir}...")
        print(f"  Center: {center}, Radius: {radius:.4f}m")

        for i in range(num_views):
            angle = 2.0 * np.pi * i / num_views
            eye = center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])

            forward = center - eye
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            actual_up = np.cross(right, forward)

            R = np.stack([right, -actual_up, forward], axis=1)
            wxyz = tf.SO3.from_matrix(R).wxyz

            self.client.camera.wxyz = wxyz
            self.client.camera.position = eye
            time.sleep(0.3)

            img = self.client.get_render(height=1080, width=1920, transport_format="png")
            save_path = out_dir / f"orbit_{i:03d}.png"
            iio.imwrite(str(save_path), img)
            print(f"  Saved {save_path.name}")

        self.client.camera.wxyz = original_wxyz
        self.client.camera.position = original_pos

        print(f"Done! {num_views} orbit snapshots saved to {out_dir}")

    def _save_orbit_video(self):
        """Render an MP4 video of the point cloud rotating around its center."""
        import os

        current_artifact = self.global_context().artifacts[self.gui_id.value]
        out_dir = current_artifact.base_path / current_artifact.artifact_name / "orbit_video"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "orbit_pointcloud.mp4"

        # Get center and radius: prefer point cloud, fallback to camera trajectory
        if self._pcd_center is not None and self._pcd_extent > 0:
            center = self._pcd_center.copy()
            radius = self._pcd_extent * 1.8 * self.gui_orbit_radius_scale.value
            print(f"Using point cloud center: {center}, extent~{self._pcd_extent:.3f}m, orbit radius={radius:.3f}m")
        else:
            cam_positions = []
            for fh in self.scene_frame_handles:
                cam_positions.append(np.array(fh.frame_handle.position))
            if len(cam_positions) == 0:
                print("No point cloud or camera frames available. Enable Global Point Cloud and rebuild, or add frames.")
                return
            cam_positions = np.stack(cam_positions, axis=0)
            center = cam_positions.mean(axis=0)
            avg_radius = np.linalg.norm(cam_positions - center, axis=1).mean()
            radius = avg_radius * self.gui_orbit_radius_scale.value
            print(f"Using camera trajectory center: {center}, orbit radius={radius:.3f}m")

        up = np.array([0.0, 0.0, 1.0])
        if len(self.scene_frame_handles) > 0:
            cam_up = np.zeros(3)
            for fh in self.scene_frame_handles:
                wxyz = np.array(fh.frame_handle.wxyz)
                rot = tf.SO3(wxyz).as_matrix()
                cam_up += rot[:, 1]
            cam_up /= len(self.scene_frame_handles)
            if np.linalg.norm(cam_up) > 1e-6:
                up = -cam_up / np.linalg.norm(cam_up)

        num_frames = int(self.gui_orbit_video_frames.value)
        fps = int(self.gui_orbit_video_fps.value)

        original_wxyz = self.client.camera.wxyz
        original_pos = self.client.camera.position

        print(f"Rendering {num_frames} frames at {fps} FPS to {out_path}...")

        frames_list = []
        for i in range(num_frames):
            angle = 2.0 * np.pi * i / num_frames
            eye = center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])

            forward = center - eye
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            actual_up = np.cross(right, forward)

            R = np.stack([right, -actual_up, forward], axis=1)
            wxyz = tf.SO3.from_matrix(R).wxyz

            self.client.camera.wxyz = wxyz
            self.client.camera.position = eye
            time.sleep(0.05)

            img = self.client.get_render(height=1080, width=1920, transport_format="png")
            frames_list.append(np.asarray(img))
            if (i + 1) % 30 == 0:
                print(f"  Rendered frame {i + 1}/{num_frames}")

        self.client.camera.wxyz = original_wxyz
        self.client.camera.position = original_pos

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames_list[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for frame in frames_list:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Done! Video saved to {out_path}")

    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                # Use a rainbow color based on the frame index
                denom = len(self.scene_frame_handles) - 1
                if denom <= 0: denom = 1
                rainbow_value = cm.jet(1.0 - frame_idx / denom)[:3]
                rainbow_value = tuple((int(c * 255) for c in rainbow_value))
                frame_node.frustum_handle.color = rainbow_value

    def _rebuild_scene(self):
        self._pcd_center = None
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        instance_phrases = None
        spatial_subsample: int = self.gui_s_sub.value
        temporal_subsample: int = self.gui_t_sub.value
        filter_interactive: bool = self.gui_filter_interactive.value
        show_global_pcd: bool = self.gui_show_global_pcd.value
        use_last_frame_objects: bool = self.gui_use_last_frame_objects.value if show_global_pcd else False
        voxel_downsample: bool = self.gui_voxel_downsample.value if show_global_pcd else False
        voxel_size: float = self.gui_voxel_size.value
        statistical_outlier_removal: bool = self.gui_statistical_outlier_removal.value if show_global_pcd else False
        outlier_nb_neighbors: int = int(self.gui_outlier_nb_neighbors.value)
        outlier_std_ratio: float = self.gui_outlier_std_ratio.value
        mask_dilation: int = int(self.gui_mask_dilation.value)
        vis_mode: str = self.gui_vis_mode.value
        use_tsdf: bool = self.gui_use_tsdf.value if show_global_pcd else False
        tsdf_implementation: str = self.gui_tsdf_implementation.value if use_tsdf else "Classic"
        tsdf_voxel_size: float = self.gui_tsdf_voxel_size.value
        tsdf_trunc_multiplier: float = self.gui_tsdf_trunc_multiplier.value

        rays: np.ndarray | None = None
        first_frame_y: np.ndarray | None = None

        self.client.scene.reset()
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []
        self.global_pcd_handle = None
        
        # For global point cloud, accumulate manually
        global_pcd_points = []
        global_pcd_colors = []
        
        # For TSDF fusion, collect frame data
        tsdf_frames_data = []  # List of dicts with depth, color, intrinsic, pose, mask
        
        interactive_pcd_points = []
        interactive_pcd_colors = []
        
        held_objects_latest = {}  # {obj_id: {'points': np.ndarray, 'colors': np.ndarray}}
        
        target_labels_latest = {}  # {obj_id: {'points': np.ndarray, 'colors': np.ndarray}}
        target_labels_set = set()
        if vis_mode == "Target Labels (Latest)":
            target_labels_set = {l.strip().lower() for l in self.gui_target_labels.value.split(",") if l.strip()}
        
        # Load SLAM map to get keyframe information
        slam_map = None
        keyframe_indices = []
        if use_last_frame_objects:
            slam_map_path = current_artifact.slam_map_path
            if slam_map_path.exists():
                try:
                    print(f"Loading SLAM map from {slam_map_path}...")
                    slam_map = torch.load(slam_map_path, map_location="cuda")
                    keyframe_indices = slam_map.dense_disp_frame_inds.tolist()
                    print(f"SLAM map loaded: {len(keyframe_indices)} keyframes")
                    print(f"Keyframe indices: {keyframe_indices[:10]}...")
                except Exception as e:
                    print(f"Warning: Failed to load SLAM map: {e}")
                    keyframe_indices = []
            else:
                print(f"Warning: SLAM map not found at {slam_map_path}")
        
        # Load per-frame depth scales
        depth_scales = {}
        depth_scale_path = current_artifact.depth_scale_path
        if depth_scale_path.exists():
            try:
                from egosim_state.utils.io import read_depth_scale_artifacts
                depth_scales = read_depth_scale_artifacts(depth_scale_path)
                print(f"Loaded depth scales for {len(depth_scales)} frames")
            except Exception as e:
                print(f"Warning: Failed to load depth scales: {e}")
        
        # Load per-frame per-object interaction states
        interaction_states = {}
        interaction_states_path = current_artifact.interaction_states_path
        if interaction_states_path.exists():
            try:
                from egosim_state.utils.io import read_interaction_states_artifacts
                interaction_states = read_interaction_states_artifacts(interaction_states_path)
                print(f"Loaded interaction states for {len(interaction_states)} frames")
            except Exception as e:
                print(f"Warning: Failed to load interaction states: {e}")
        
        # Load instance phrases to get object descriptions
        instance_phrases = {}
        try:
            from egosim_state.utils.io import read_instance_phrases
            phrase_path = current_artifact.mask_phrase_path
            if phrase_path.exists():
                instance_phrases = read_instance_phrases(phrase_path)
                print(f"Loaded phrases for {len(instance_phrases)} objects")
        except Exception as e:
            print(f"Warning: Failed to load instance phrases: {e}")
        
        # For non-interactive objects: track multi-frame appearances
        # {obj_id: {'frames': [frame_idx, ...], 'pcd_data': [{'pcd_world', 'colors', 'depth', 'c2w', ...}], 
        #           'last_frame': int, 'prev_keyframe': int, 'interacted': bool}}
        object_tracking_data = {}
        
        # Track hand object IDs for interaction detection
        hand_related_phrases = ['hand', 'arm']

        def none_it(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None
        
        def get_prev_keyframe(frame_idx, keyframes):
            """Get the previous keyframe index before frame_idx"""
            if not keyframes:
                return 0
            prev_kfs = [kf for kf in keyframes if kf < frame_idx]
            return prev_kfs[-1] if prev_kfs else 0
        
        def check_hand_object_interaction(obj_mask, hand_mask, obj_depth, hand_depth, iou_threshold=0.1, depth_threshold=0.15):
            """
            Check if object is interacting with hand based on mask overlap and depth proximity
            Returns: True if interacting, False otherwise
            """
            if obj_mask is None or hand_mask is None:
                return False
            
            # Calculate mask IOU
            intersection = np.logical_and(obj_mask, hand_mask).sum()
            union = np.logical_or(obj_mask, hand_mask).sum()
            if union == 0:
                return False
            iou = intersection / union
            
            if iou < iou_threshold:
                return False
            
            # Check depth proximity in overlapping region
            overlap_region = np.logical_and(obj_mask, hand_mask)
            if overlap_region.sum() > 0:
                obj_depths_overlap = obj_depth[overlap_region]
                hand_depths_overlap = hand_depth[overlap_region]
                # Filter valid depths
                valid_mask = (obj_depths_overlap > 0) & (hand_depths_overlap > 0)
                if valid_mask.sum() > 10:  # At least 10 valid depth points
                    depth_diff = np.abs(obj_depths_overlap[valid_mask] - hand_depths_overlap[valid_mask])
                    median_depth_diff = np.median(depth_diff)
                    if median_depth_diff < depth_threshold:
                        return True
            
            return False

        # Prepare instance mask iterator if filtering interactive
        instance_iter = none_it(read_instance_artifacts(current_artifact.mask_path)) if filter_interactive else None
        
        # Load interactive object IDs if filtering interactive
        interactive_data = None
        if filter_interactive:
            # Use the mask directory from current artifact
            interactive_path = current_artifact.mask_path.parent / "interactive.npz"
            print(f"Looking for interactive.npz at: {interactive_path}")
            try:
                from pathlib import Path
                if interactive_path.exists():
                    interactive_data = dict(np.load(interactive_path, allow_pickle=True))
                    # Convert string keys to int and values to sets
                    interactive_data = {int(k): set(v) for k, v in interactive_data.items()}
                    print(f"Loaded interactive data for {len(interactive_data)} frames")
                    # Debug: print first few frames' interactive IDs
                    for i, (k, v) in enumerate(interactive_data.items()):
                        if i < 3:
                            print(f"  Frame {k}: interactive_obj_ids = {v}")
                else:
                    print(f"Warning: interactive.npz not found at {interactive_path}. Interactive filtering disabled.")
            except Exception as e:
                print(f"Warning: Failed to load interactive data: {e}. Interactive filtering disabled.")
        

        # Prepare for incremental fusion
        use_incremental = self.gui_voxel_downsample.value if show_global_pcd else False
        overlap_threshold = self.gui_voxel_size.value
        # Color difference threshold (0-255 RGB L2). Used when depths are close:
        # - If color similar (<= threshold): treat as overlapped (skip)
        # - If color different (> threshold): keep new point
        color_threshold = 40.0
        prev_frame_info = None

        last_frame_idx = -1
        last_frame_data = None
        for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth)) in enumerate(
            zip(
                read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy(),
                read_rgb_artifacts(current_artifact.rgb_path),
                *read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1:3],
                none_it(read_depth_artifacts(current_artifact.depth_path)),
            )
        ):
            # Read instance mask for current frame if filtering foreground
            instance_mask = None
            if instance_iter is not None:
                try:
                    _, instance_mask = next(instance_iter)
                except StopIteration:
                    instance_mask = None

            if frame_idx % temporal_subsample != 0:
                continue
            # Track last frame data
            last_frame_idx = frame_idx
            last_frame_data = {
                "c2w": c2w,
                "rgb": rgb,
                "intr": intr,
                "camera_type": camera_type,
                "depth": depth,
                "instance_mask": instance_mask,
            }

            # Build camera model to preserve distortion parameters (MEI, etc.)
            camera_model = camera_type.build_camera_model(intr)
            pinhole_intr = camera_model.pinhole().intrinsics
            frame_height, frame_width = rgb.shape[:2]
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())

            sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
            
            sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

            if first_frame_y is None:
                first_frame_y = c2w[:3, 1]
                self.client.scene.set_up_direction(-first_frame_y)

            if rays is None:
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                if camera_type == CameraType.PANORAMA:
                    disp_v = disp_v / (frame_height - 1)
                    disp_u = disp_u / (frame_width - 1)
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                if camera_type != CameraType.PANORAMA:
                    rays /= rays[..., 2:3]

            if depth is not None:
                pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
                depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
                depth_mask_original = depth_mask.copy()
                
                # Apply interactive filtering: filter out interactive objects
                if filter_interactive and instance_mask is not None and interactive_data is not None:
                    instance_mask_subsampled = instance_mask[::spatial_subsample, ::spatial_subsample].numpy()
                    interactive_obj_ids = interactive_data.get(frame_idx, set())
                    
                    # Convert all IDs to Python int for consistent set operations
                    interactive_obj_ids = {int(x) for x in interactive_obj_ids}
                    
                    # Debug: print unique IDs in instance mask and interactive IDs
                    unique_mask_ids = np.unique(instance_mask_subsampled)
                    print(f"Frame {frame_idx}: instance_mask unique IDs = {unique_mask_ids[:20]}... (total {len(unique_mask_ids)})")
                    print(f"Frame {frame_idx}: interactive_obj_ids from file = {interactive_obj_ids}")
                    
                    # Safety: ensure background (ID=0) is never filtered
                    interactive_obj_ids = interactive_obj_ids - {0}
                    
                    # Also filter human body parts not in interactive_ids (e.g. person)
                    for uid in unique_mask_ids:
                        uid = int(uid)
                        if uid == 0 or uid in interactive_obj_ids:
                            continue
                        phrase = instance_phrases.get(uid, "").lower()
                        is_human = any(kw in phrase for kw in hand_related_phrases)
                        is_in_hand = "in hand" in phrase
                        if is_human and not is_in_hand:
                            interactive_obj_ids.add(uid)
                    
                    print(f"Frame {frame_idx}: interactive_obj_ids after removing 0 and adding human body = {interactive_obj_ids}")
                    
                    # Mask out all pixels belonging to any interactive object
                    if interactive_obj_ids:
                        # Convert to list of ints explicitly
                        interactive_ids_list = list(interactive_obj_ids)
                        print(f"Frame {frame_idx}: filtering out IDs = {interactive_ids_list}")
                        
                        interactive_mask = np.isin(instance_mask_subsampled, interactive_ids_list)
                        
                        # Dilate mask to filter edges
                        if mask_dilation > 0:
                            kernel = np.ones((mask_dilation * 2 + 1, mask_dilation * 2 + 1), np.uint8)
                            interactive_mask = cv2.dilate(interactive_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                            print(f"Frame {frame_idx}: dilated mask with kernel size {mask_dilation * 2 + 1}x{mask_dilation * 2 + 1}")
                        
                        num_interactive_pixels = np.sum(interactive_mask)
                        print(f"Frame {frame_idx}: found {num_interactive_pixels} interactive pixels to filter")
                        
                        non_interactive_mask = ~interactive_mask
                        old_mask_count = np.sum(depth_mask)
                        depth_mask = depth_mask & non_interactive_mask
                        new_mask_count = np.sum(depth_mask)
                        print(f"Frame {frame_idx}: depth_mask points {old_mask_count} -> {new_mask_count} (filtered {old_mask_count - new_mask_count})")
            else:
                pcd, depth_mask = None, None

            # For global point cloud, accumulate data
            if show_global_pcd and pcd is not None:
                if instance_mask is not None:
                    instance_mask_subsampled = instance_mask[::spatial_subsample, ::spatial_subsample].numpy()
                    unique_obj_ids = np.unique(instance_mask_subsampled)
                    
                    # Get interactive IDs for this frame
                    frame_interactive_ids = interactive_data.get(frame_idx, set()) if interactive_data else set()
                    frame_interactive_ids = {int(x) for x in frame_interactive_ids}
                    
                    # Process background (obj_id == 0) — accumulate across frames
                    bg_mask = (instance_mask_subsampled == 0)
                    valid_bg_mask = depth_mask & bg_mask if depth_mask is not None else bg_mask

                    if np.sum(valid_bg_mask) > 0 and vis_mode not in ("Interactive Objects Only", "Held Objects (Latest)", "Target Labels (Latest)"):
                        if use_tsdf:
                            # Collect frame data for TSDF fusion
                            # Create binary mask for background (1=keep, 0=filter)
                            bg_depth_mask = valid_bg_mask.astype(np.uint8)
                            
                            # CRITICAL: Scale original camera intrinsics to match downsampled resolution
                            # This preserves camera type (MEI keeps distortion, etc.)
                            scaled_camera_model = camera_model.scaled(1.0 / spatial_subsample)
                            # For TSDF (Open3D), we need pinhole intrinsics
                            scaled_pinhole_intr = scaled_camera_model.pinhole().intrinsics.cpu().numpy()
                            
                            tsdf_frames_data.append({
                                'depth': depth.numpy()[::spatial_subsample, ::spatial_subsample],
                                'color': sampled_rgb,
                                'mask': bg_depth_mask,
                                'intrinsic': scaled_pinhole_intr,  # Use scaled pinhole for TSDF
                                'pose': c2w,
                                'frame_height': sampled_rgb.shape[0],
                                'frame_width': sampled_rgb.shape[1],
                            })
                        else:
                            # Original point cloud concatenation method
                            bg_pcd_flat = pcd.reshape(-1, 3)[valid_bg_mask.reshape(-1)]
                            bg_rgb_flat = sampled_rgb.reshape(-1, 3)[valid_bg_mask.reshape(-1)]
                            bg_pcd_world = (c2w[:3, :3] @ bg_pcd_flat.T).T + c2w[:3, 3]
                            
                            if use_incremental and prev_frame_info is not None:
                                # Apply incremental fusion: filter out points covered by previous view
                                overlap_mask = self._check_overlap(
                                    bg_pcd_world,
                                    prev_frame_info['c2w'],
                                    prev_frame_info['camera_model'],
                                    prev_frame_info['depth'],
                                    prev_frame_info['rgb'],
                                    bg_rgb_flat,
                                    overlap_threshold,
                                    color_threshold,
                                )
                                # Keep points that are NOT overlapped
                                keep_mask = ~overlap_mask
                                bg_pcd_world = bg_pcd_world[keep_mask]
                                bg_rgb_flat = bg_rgb_flat[keep_mask]
                            
                            # Append background points
                            global_pcd_points.append(bg_pcd_world)
                            global_pcd_colors.append(bg_rgb_flat)
                    
                    # Update previous frame info for incremental fusion
                    # We store the scaled camera model and subsampled depth
                    if use_incremental:
                        pass # handled below

                    # Process non-interactive objects - track multi-frame appearances
                    if use_last_frame_objects and vis_mode == "All":
                        # Get hand/person masks for interaction detection
                        hand_masks = {}
                        hand_depths = {}
                        for obj_id in unique_obj_ids:
                            # Check if this is a hand-related object (from phrases or interactive_data)
                            if obj_id in frame_interactive_ids:
                                obj_mask = (instance_mask_subsampled == obj_id)
                                hand_masks[obj_id] = obj_mask
                                hand_depths[obj_id] = depth.numpy()[::spatial_subsample, ::spatial_subsample] * obj_mask
                        
                        for obj_id in unique_obj_ids:
                            if obj_id == 0:  # Background already handled
                                continue
                            if obj_id in frame_interactive_ids:  # Skip interactive objects themselves
                                continue
                            phrase = instance_phrases.get(int(obj_id), "").lower()
                            is_human_body = any(kw in phrase for kw in hand_related_phrases)
                            is_in_hand = "in hand" in phrase
                            if is_human_body and not is_in_hand:
                                continue
                            
                            # Create mask for this specific object
                            obj_mask = (instance_mask_subsampled == obj_id)
                            valid_obj_mask = depth_mask & obj_mask if depth_mask is not None else obj_mask
                            
                            if np.sum(valid_obj_mask) > 0:
                                # Initialize tracking data if this is the first time seeing this object
                                if obj_id not in object_tracking_data:
                                    object_tracking_data[obj_id] = {
                                        'frames': [],
                                        'frame_data': [],
                                        'last_frame': frame_idx,
                                        'prev_keyframe': get_prev_keyframe(frame_idx, keyframe_indices),
                                        'interacted': False,
                                        'disappeared_after_interaction': False
                                    }
                                
                                # Update last seen frame
                                object_tracking_data[obj_id]['last_frame'] = frame_idx
                                
                                # Check for hand-object interaction using pre-computed states if available
                                is_interacting = False
                                if interaction_states and frame_idx in interaction_states:
                                    # Use pre-computed interaction state from inference
                                    obj_state = interaction_states[frame_idx].get(obj_id, {})
                                    is_interacting = obj_state.get('is_interacting', False)
                                    if is_interacting:
                                        interacting_with = obj_state.get('interacting_with', [])
                                        print(f"  Frame {frame_idx}: Object {obj_id} is interacting with {interacting_with} (from pre-computed states)")
                                        object_tracking_data[obj_id]['interacted'] = True
                                else:
                                    # Fallback: compute interaction on-the-fly (slower)
                                    for hand_id, hand_mask in hand_masks.items():
                                        hand_depth = hand_depths[hand_id]
                                        obj_depth_full = depth.numpy()[::spatial_subsample, ::spatial_subsample]
                                        if check_hand_object_interaction(obj_mask, hand_mask, obj_depth_full, hand_depth):
                                            is_interacting = True
                                            object_tracking_data[obj_id]['interacted'] = True
                                            print(f"  Frame {frame_idx}: Object {obj_id} is interacting with hand {hand_id} (computed)")
                                            break
                                
                                # Store frame data (we'll decide later which frames to keep)
                                obj_pcd_flat = pcd.reshape(-1, 3)[valid_obj_mask.reshape(-1)]
                                obj_rgb_flat = sampled_rgb.reshape(-1, 3)[valid_obj_mask.reshape(-1)]
                                obj_depth_flat = depth.numpy()[::spatial_subsample, ::spatial_subsample][valid_obj_mask]
                                
                                # Apply depth scale alignment (same as background)
                                frame_scale = depth_scales.get(frame_idx, 1.0)
                                obj_depth_flat_aligned = obj_depth_flat * frame_scale
                                obj_pcd_flat_aligned = obj_pcd_flat * frame_scale
                                
                                # Store in camera coordinates for later alignment
                                scaled_camera_model = camera_model.scaled(1.0 / spatial_subsample)
                                scaled_pinhole_intr = scaled_camera_model.pinhole().intrinsics.cpu().numpy()
                                
                                object_tracking_data[obj_id]['frames'].append(frame_idx)
                                object_tracking_data[obj_id]['frame_data'].append({
                                    'pcd_camera': obj_pcd_flat_aligned,  # In camera coordinates, scale-aligned
                                    'colors': obj_rgb_flat,
                                    'depth': obj_depth_flat_aligned,  # Scale-aligned depth
                                    'mask': valid_obj_mask.astype(np.uint8),
                                    'c2w': c2w,
                                    'intrinsic': scaled_pinhole_intr,
                                    'frame_idx': frame_idx,
                                    'is_interacting': is_interacting,
                                    'depth_scale': frame_scale,
                                })
                    
                    # Collect target label object points (for "Target Labels (Latest)" mode)
                    if vis_mode == "Target Labels (Latest)" and target_labels_set:
                        for obj_id in unique_obj_ids:
                            if obj_id == 0:
                                continue
                            phrase = instance_phrases.get(int(obj_id), "").lower()
                            if phrase not in target_labels_set:
                                continue
                            obj_mask = (instance_mask_subsampled == obj_id)
                            valid_obj_mask = depth_mask_original & obj_mask
                            if np.sum(valid_obj_mask) > 0:
                                obj_pcd_flat = pcd.reshape(-1, 3)[valid_obj_mask.reshape(-1)]
                                obj_rgb_flat = sampled_rgb.reshape(-1, 3)[valid_obj_mask.reshape(-1)]
                                obj_pcd_world = (c2w[:3, :3] @ obj_pcd_flat.T).T + c2w[:3, 3]
                                target_labels_latest[int(obj_id)] = {
                                    'points': obj_pcd_world,
                                    'colors': obj_rgb_flat,
                                    'frame_idx': frame_idx,
                                }

                    # Collect interactive object points
                    if vis_mode in ("Interactive Objects Only", "Held Objects (Latest)") and frame_interactive_ids:
                        for int_id in frame_interactive_ids:
                            if vis_mode == "Held Objects (Latest)":
                                phrase = instance_phrases.get(int(int_id), "").lower()
                                is_human_body = any(kw in phrase for kw in hand_related_phrases)
                                is_in_hand = "in hand" in phrase
                                if is_human_body and not is_in_hand:
                                    continue
                            int_obj_mask = (instance_mask_subsampled == int_id)
                            valid_int_mask = depth_mask_original & int_obj_mask
                            if np.sum(valid_int_mask) > 0:
                                int_pcd_flat = pcd.reshape(-1, 3)[valid_int_mask.reshape(-1)]
                                int_rgb_flat = sampled_rgb.reshape(-1, 3)[valid_int_mask.reshape(-1)]
                                int_pcd_world = (c2w[:3, :3] @ int_pcd_flat.T).T + c2w[:3, 3]
                                if vis_mode == "Interactive Objects Only":
                                    interactive_pcd_points.append(int_pcd_world)
                                    interactive_pcd_colors.append(int_rgb_flat)
                                else:
                                    held_objects_latest[int(int_id)] = {
                                        'points': int_pcd_world,
                                        'colors': int_rgb_flat,
                                    }
                else:
                    # No instance mask — accumulate all filtered points (skip BG-only modes)
                    if vis_mode == "All":
                        pcd_flat = pcd.reshape(-1, 3)
                        rgb_flat = sampled_rgb.reshape(-1, 3)
                        if depth_mask is not None:
                            mask_flat = depth_mask.reshape(-1)
                            pcd_flat = pcd_flat[mask_flat]
                            rgb_flat = rgb_flat[mask_flat]
                        pcd_world = (c2w[:3, :3] @ pcd_flat.T).T + c2w[:3, 3]
                        global_pcd_points.append(pcd_world)
                        global_pcd_colors.append(rgb_flat)

            frame_node = self._make_frame_nodes(
                frame_idx,
                c2w,
                sampled_rgb,
                fov,
                pcd if not show_global_pcd else None,  # Don't show per-frame pcd if showing global
                depth_mask,
            )
            self.scene_frame_handles.append(frame_node)

            # Update prev_frame_info for the next iteration (outside the object filtering loop)
            if show_global_pcd and use_incremental:
                # Ensure we have the scaled camera model corresponding to the subsampled depth
                scaled_cam = camera_model.scaled(1.0 / spatial_subsample)
                current_depth_sub = depth.numpy()[::spatial_subsample, ::spatial_subsample]
                
                prev_frame_info = {
                    'c2w': c2w,
                    'camera_model': scaled_cam,
                    'depth': current_depth_sub,
                    'rgb': sampled_rgb,
                }
        
        # Merge background and static objects (global cloud only)
        if show_global_pcd:
            if vis_mode == "Interactive Objects Only":
                if interactive_pcd_points:
                    all_points = np.concatenate(interactive_pcd_points, axis=0)
                    all_colors = np.concatenate(interactive_pcd_colors, axis=0)
                    print(f"Interactive objects: {len(all_points)} points from {len(interactive_pcd_points)} frames")
                else:
                    all_points = None
                    all_colors = None
                    print("Warning: No interactive object points found")
            elif vis_mode == "Held Objects (Latest)":
                if held_objects_latest:
                    all_obj_points = [v['points'] for v in held_objects_latest.values()]
                    all_obj_colors = [v['colors'] for v in held_objects_latest.values()]
                    all_points = np.concatenate(all_obj_points, axis=0)
                    all_colors = np.concatenate(all_obj_colors, axis=0)
                    obj_info = {oid: instance_phrases.get(oid, "?") for oid in held_objects_latest}
                    print(f"Held objects (latest state): {len(all_points)} points, {len(held_objects_latest)} objects: {obj_info}")
                else:
                    all_points = None
                    all_colors = None
                    print("Warning: No held objects found (all interactive objects are human body parts?)")
            elif vis_mode == "Target Labels (Latest)":
                if target_labels_latest:
                    all_obj_points = [v['points'] for v in target_labels_latest.values()]
                    all_obj_colors = [v['colors'] for v in target_labels_latest.values()]
                    all_points = np.concatenate(all_obj_points, axis=0)
                    all_colors = np.concatenate(all_obj_colors, axis=0)
                    for oid, v in target_labels_latest.items():
                        print(f"  Target object ID {oid} ({instance_phrases.get(oid, '?')}): {len(v['points'])} points, last frame = {v['frame_idx']}")
                    print(f"Target labels (latest state): {len(all_points)} points, {len(target_labels_latest)} objects")
                    print(f"  Requested labels: {target_labels_set}")
                    matched = {oid: instance_phrases.get(oid, "?") for oid in target_labels_latest}
                    unmatched = target_labels_set - {v.lower() for v in matched.values()}
                    if unmatched:
                        print(f"  Warning: labels not found in instance_phrases: {unmatched}")
                else:
                    all_points = None
                    all_colors = None
                    print(f"Warning: No objects found matching target labels: {target_labels_set}")
                    print(f"  Available phrases: {instance_phrases}")
            elif use_tsdf and tsdf_frames_data:
                # Use TSDF fusion for background
                print(f"Using {tsdf_implementation} TSDF fusion with {len(tsdf_frames_data)} frames...")
                #import pdb; pdb.set_trace()
                if tsdf_implementation == "VoxelBlockGrid":
                    all_points, all_colors = self._integrate_frames_voxelblockgrid_tsdf(
                        tsdf_frames_data, tsdf_voxel_size, tsdf_trunc_multiplier
                    )
                else:
                    all_points, all_colors = self._integrate_frames_tsdf(
                        tsdf_frames_data, tsdf_voxel_size, tsdf_trunc_multiplier
                    )
                print(f"TSDF fusion produced {len(all_points)} points")
            elif global_pcd_points:
                # Original concatenation method
                print(f"Accumulated background from all frames: {sum(len(p) for p in global_pcd_points)} points")
                all_points = np.concatenate(global_pcd_points, axis=0)
                all_colors = np.concatenate(global_pcd_colors, axis=0)
                print(f"Total points before post-processing: {len(all_points)}")
            else:
                print("Warning: No background points accumulated")
                all_points = None
                all_colors = None

            # 2. Process non-interactive objects with multi-frame fusion (only in "All" mode)
            if use_last_frame_objects and object_tracking_data and vis_mode == "All":
                print(f"\n=== Processing {len(object_tracking_data)} tracked objects ===")
                obj_points_list = []
                obj_colors_list = []
                
                for obj_id, obj_data in object_tracking_data.items():
                    last_frame = obj_data['last_frame']
                    prev_kf = obj_data['prev_keyframe']
                    frames = obj_data['frames']
                    frame_data_list = obj_data['frame_data']
                    
                    # Get object phrase to check for filtering
                    obj_phrase = instance_phrases.get(obj_id, "")
                    
                    print(f"\nObject {obj_id}: {obj_phrase}")
                    print(f"  Last seen: frame {last_frame}, prev keyframe: {prev_kf}")
                    print(f"  Total appearances: {len(frames)} frames")
                    print(f"  Interacted: {obj_data['interacted']}")
                    
                    # Filter out "in hand" objects (REMOVED: relying on inference interaction_states instead)
                    #if "in hand" in obj_phrase.lower():
                    #    print(f"  -> Object phrase contains 'in hand', SKIPPING")
                    #    continue
                    
                    
                    # Check if object disappeared after interaction
                    if obj_data['interacted']:
                        # Find the last interaction frame
                        last_interaction_frame = -1
                        for fd in reversed(frame_data_list):
                            if fd['is_interacting']:
                                last_interaction_frame = fd['frame_idx']
                                break
                        
                        # Check if object appeared after the interaction
                        frames_after_interaction = [f for f in frames if f > last_interaction_frame]
                        if len(frames_after_interaction) < 2:
                            print(f"  -> Object disappeared after interaction at frame {last_interaction_frame} (only {len(frames_after_interaction)} frames after), SKIPPING")
                            obj_data['disappeared_after_interaction'] = True
                            continue
                    
                    # Select frames: use last 5 frames before and including last appearance
                    max_frames_for_fusion = 1
                    start_frame = max(0, last_frame - max_frames_for_fusion + 1)
                    selected_frames = [f for f in frames if start_frame <= f <= last_frame]
                    selected_frame_data = [fd for fd in frame_data_list if start_frame <= fd['frame_idx'] <= last_frame]
                    
                    if len(selected_frame_data) == 0:
                        print(f"  -> No valid frames in range [{start_frame}, {last_frame}], SKIPPING")
                        continue
                    
                    print(f"  Selected last {len(selected_frame_data)} frames for fusion (from frame {start_frame} to {last_frame})")
                    
                    # Use TSDF fusion for multi-frame object reconstruction
                    if len(selected_frame_data) > 1 and use_tsdf:
                        print(f"  Using TSDF fusion for object {obj_id}...")
                        # Prepare data for TSDF
                        obj_tsdf_data = []
                        for fd in selected_frame_data:
                            # Create a full-size depth map with only this object
                            h, w = fd['mask'].shape
                            obj_depth_map = np.zeros((h, w), dtype=np.float32)
                            # Reconstruct depth from point cloud
                            # Note: This is approximate; ideally we'd use the original depth map
                            obj_depth_map[fd['mask'].astype(bool)] = fd['depth']
                            
                            obj_tsdf_data.append({
                                'depth': obj_depth_map,
                                'color': np.zeros((h, w, 3), dtype=np.uint8),  # Will be filled
                                'mask': fd['mask'],
                                'intrinsic': fd['intrinsic'],
                                'pose': fd['c2w'],
                                'frame_height': h,
                                'frame_width': w,
                            })
                            
                            # Fill color for valid pixels
                            valid_mask = fd['mask'].astype(bool)
                            n_valid = np.sum(valid_mask)
                            if n_valid == len(fd['colors']):
                                obj_tsdf_data[-1]['color'][valid_mask] = fd['colors']
                            else:
                                print(f"    Warning: color size mismatch for frame {fd['frame_idx']}")
                        
                        try:
                            if tsdf_implementation == "VoxelBlockGrid":
                                obj_fused_points, obj_fused_colors = self._integrate_frames_voxelblockgrid_tsdf(
                                    obj_tsdf_data, tsdf_voxel_size, tsdf_trunc_multiplier
                                )
                            else:
                                obj_fused_points, obj_fused_colors = self._integrate_frames_tsdf(
                                    obj_tsdf_data, tsdf_voxel_size, tsdf_trunc_multiplier
                                )
                            
                            if len(obj_fused_points) > 0:
                                print(f"  -> TSDF fusion produced {len(obj_fused_points)} points")
                                obj_points_list.append(obj_fused_points)
                                obj_colors_list.append(obj_fused_colors)
                            else:
                                print(f"  -> TSDF fusion produced 0 points, falling back to last frame")
                                # Fallback: use last frame only
                                last_fd = selected_frame_data[-1]
                                last_pcd_world = (last_fd['c2w'][:3, :3] @ last_fd['pcd_camera'].T).T + last_fd['c2w'][:3, 3]
                                obj_points_list.append(last_pcd_world)
                                obj_colors_list.append(last_fd['colors'])
                        except Exception as e:
                            print(f"  -> TSDF fusion failed: {e}, using last frame only")
                            last_fd = selected_frame_data[-1]
                            last_pcd_world = (last_fd['c2w'][:3, :3] @ last_fd['pcd_camera'].T).T + last_fd['c2w'][:3, 3]
                            obj_points_list.append(last_pcd_world)
                            obj_colors_list.append(last_fd['colors'])
                    else:
                        # Single frame or no TSDF: just use the last frame
                        print(f"  Using last frame only (single frame or TSDF disabled)")
                        last_fd = selected_frame_data[-1]
                        last_pcd_world = (last_fd['c2w'][:3, :3] @ last_fd['pcd_camera'].T).T + last_fd['c2w'][:3, 3]
                        obj_points_list.append(last_pcd_world)
                        obj_colors_list.append(last_fd['colors'])
                
                # Combine all object point clouds
                if obj_points_list:
                    print(f"\n=== Combining {len(obj_points_list)} object point clouds ===")
                    obj_points_concat = np.concatenate(obj_points_list, axis=0)
                    obj_colors_concat = np.concatenate(obj_colors_list, axis=0)
                    print(f"Total object points: {len(obj_points_concat)}")
                    
                    if all_points is not None and len(all_points) > 0:
                        all_points = np.concatenate([all_points, obj_points_concat], axis=0)
                        all_colors = np.concatenate([all_colors, obj_colors_concat], axis=0)
                    else:
                        all_points = obj_points_concat
                        all_colors = obj_colors_concat

            # 3. Post-process if we have a point cloud
            if all_points is not None and len(all_points) > 0:
                print(f"Total points: {len(all_points)}")
                
                # 4. [disabled] voxel-grid downsampling
                # if voxel_downsample:
                #     print(f"Applying voxel grid downsampling (voxel_size={voxel_size:.3f}m)...")
                #     all_points, all_colors = self._voxel_downsample(all_points, all_colors, voxel_size)
                #     print(f"Total points after downsampling: {len(all_points)}")
                
                # 5. Statistical outlier removal (if enabled)
                if statistical_outlier_removal:
                    print(f"Applying statistical outlier removal (nb_neighbors={outlier_nb_neighbors}, std_ratio={outlier_std_ratio:.1f})...")
                    all_points, all_colors = self._remove_statistical_outliers(
                        all_points, all_colors, outlier_nb_neighbors, outlier_std_ratio
                    )
                    print(f"Total points after outlier removal: {len(all_points)}")
                self.global_pcd_handle = self.client.scene.add_point_cloud(
                    name="/global_point_cloud",
                    points=all_points,
                    colors=all_colors.astype(np.uint8),
                    point_size=0.002,
                )
                self._pcd_center = all_points.mean(axis=0)
                dists = np.linalg.norm(all_points - self._pcd_center, axis=1)
                self._pcd_extent = float(np.percentile(dists, 95)) if len(dists) > 0 else 1.0
                print("Global point cloud created successfully!")
                # Save post-processed global point cloud to workspace directory
                out_ply = current_artifact.base_path / f"{current_artifact.artifact_name}_global_point_cloud.ply"
                _save_point_cloud_ply(all_points, all_colors.astype(np.uint8), out_ply)

    def _integrate_frames_tsdf(
        self, frames_data: list[dict], voxel_size: float, trunc_multiplier: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fuse multi-frame background with TSDF (truncated signed distance function).
        
        TSDF overview:
        - Voxelize physical space
        - Signed distance to surface per voxel
        - Weighted fusion across frames reduces misalignment and ghosting
        
        Args:
            frames_data: per-frame dicts with depth, color, mask, intrinsic, pose
            voxel_size: meters; smaller = finer detail, more memory
            trunc_multiplier: truncation multiplier for smoothing
            
        Returns:
            Fused point coordinates and colors
        """
        if len(frames_data) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        
        import open3d as o3d
        
        # 1. Init TSDF volume
        sdf_trunc = voxel_size * trunc_multiplier
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        print(f"  TSDF parameters: voxel_length={voxel_size:.4f}m, sdf_trunc={sdf_trunc:.4f}m")
        
        # 2. Integrate each frame
        for i, frame in enumerate(frames_data):
            if i % 10 == 0:
                print(f"  Integrating frame {i+1}/{len(frames_data)}...")
            
            # Depth: zero outside mask (invalid)
            depth_array = frame['depth'].copy().astype(np.float32)
            mask = frame['mask'].astype(bool)
            depth_array[~mask] = 0.0
            
            # Prepare color
            color_array = frame['color'].astype(np.uint8)
            
            # Open3D images
            color_o3d = o3d.geometry.Image(color_array)
            depth_o3d = o3d.geometry.Image(depth_array)
            
            # Build RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,  # depth in meters
                depth_trunc=3.0,  # clamp depth beyond 3 m
                convert_rgb_to_intensity=False
            )
            
            # Pinhole intrinsics
            fx, fy, cx, cy = frame['intrinsic']
            h, w = frame['frame_height'], frame['frame_width']
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy
            )
            
            # TSDF expects world-to-camera extrinsics
            # Invert camera-to-world (c2w)
            c2w = frame['pose']
            w2c = np.eye(4)
            w2c[:3, :3] = c2w[:3, :3].T
            w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
            
            # Integrate into volume
            volume.integrate(
                rgbd,
                intrinsic_o3d,
                w2c
            )
        
        # 3. Extract point cloud
        print("  Extracting point cloud from TSDF volume...")
        pcd_o3d = volume.extract_point_cloud()
        
        # To NumPy
        points = np.asarray(pcd_o3d.points)
        colors = (np.asarray(pcd_o3d.colors) * 255).astype(np.uint8)
        
        print(f"  TSDF extraction: {len(points)} points")
        
        return points, colors

    def _remove_statistical_outliers(
        self, points: np.ndarray, colors: np.ndarray, nb_neighbors: int, std_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Statistical outlier removal via mean distance to k neighbors.
        
        Steps:
        1. k nearest neighbors per point
        2. Mean distance to those neighbors
        3. Global mean and std of those means
        4. Drop points with mean distance > mean + std_ratio * std
        
        Args:
            points: (N, 3) coordinates
            colors: (N, 3) colors
            nb_neighbors: neighbor count
            std_ratio: std multiplier; lower = stricter
            
        Returns:
            Filtered points and colors
        """
        if len(points) == 0:
            return points, colors
        
        from scipy.spatial import KDTree
        
        # KD-tree for nearest neighbors
        tree = KDTree(points)
        
        # k+1 nearest neighbors (includes self)
        distances, indices = tree.query(points, k=nb_neighbors + 1)
        
        # Mean distance excluding self (first neighbor)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        # Global statistics
        mean_distance = np.mean(avg_distances)
        std_distance = np.std(avg_distances)
        
        # Inlier threshold
        threshold = mean_distance + std_ratio * std_distance
        
        # Keep inliers
        inlier_mask = avg_distances < threshold
        
        filtered_points = points[inlier_mask]
        filtered_colors = colors[inlier_mask]
        
        num_outliers = np.sum(~inlier_mask)
        outlier_ratio = num_outliers / len(points) * 100
        print(f"  Statistical outlier removal: removed {num_outliers}/{len(points)} points ({outlier_ratio:.1f}% outliers)")
        print(f"  Distance threshold: {threshold:.4f}m (mean={mean_distance:.4f}m, std={std_distance:.4f}m)")
        
        return filtered_points, filtered_colors

    def _voxel_downsample(
        self, points: np.ndarray, colors: np.ndarray, voxel_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Voxel-grid downsampling: average points inside each voxel.
        
        Args:
            points: (N, 3) coordinates
            colors: (N, 3) colors
            voxel_size: voxel size (meters)
            
        Returns:
            Downsampled points and colors
        """
        if len(points) == 0:
            return points, colors
        
        # Voxel index per point
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Group point indices by voxel
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(i)
        
        # Average each voxel
        downsampled_points = []
        downsampled_colors = []
        
        for voxel_key, point_indices in voxel_dict.items():
            # Mean position and color per voxel
            voxel_points = points[point_indices]
            voxel_colors = colors[point_indices]
            
            avg_point = np.mean(voxel_points, axis=0)
            avg_color = np.mean(voxel_colors, axis=0)
            
            downsampled_points.append(avg_point)
            downsampled_colors.append(avg_color)
        
        downsampled_points = np.array(downsampled_points)
        downsampled_colors = np.array(downsampled_colors)
        
        reduction_ratio = (1 - len(downsampled_points) / len(points)) * 100
        print(f"  Voxel downsampling: {len(points)} -> {len(downsampled_points)} points ({reduction_ratio:.1f}% reduction)")
        
        return downsampled_points, downsampled_colors

    def _align_accumulated_point_clouds(
        self, point_lists: list[np.ndarray], color_lists: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Geometric alignment (Umeyama) to refine accumulated clouds.
        Align each frame to the first.
        """
        if len(point_lists) < 2:
            return point_lists, color_lists
        
        # First frame is reference
        reference_points = point_lists[0]
        aligned_points = [reference_points]
        aligned_colors = [color_lists[0]]
        
        print(f"  Aligning {len(point_lists)} point clouds to reference frame...")
        
        for i in range(1, len(point_lists)):
            source_points = point_lists[i]
            source_colors = color_lists[i]
            
            # Subsample for faster alignment
            sample_stride = max(1, len(source_points) // 10000)
            source_sample = source_points[::sample_stride]
            reference_sample = reference_points[::min(sample_stride, len(reference_points) // 10000)]
            
            # Match sample counts (min)
            n_samples = min(len(source_sample), len(reference_sample))
            if n_samples < 10:
                # Too few samples; skip alignment
                aligned_points.append(source_points)
                aligned_colors.append(source_colors)
                continue
            
            source_sample = source_sample[:n_samples]
            reference_sample = reference_sample[:n_samples]
            
            try:
                # Torch tensors for alignment
                source_torch = torch.from_numpy(source_sample).float()
                reference_torch = torch.from_numpy(reference_sample).float()
                
                # Umeyama similarity transform
                from egosim_state.utils.geometry import align_points
                transform = align_points(source_torch, reference_torch, scale=True)
                
                # Extract transform params (batched or unbatched)
                R_mat = transform.rotation.matrix()
                if R_mat.ndim == 3:  # Batched: shape [1, 3, 3]
                    R = R_mat[0].numpy()
                else:  # Unbatched: shape [3, 3]
                    R = R_mat.numpy()
                
                # Extract only 3x3 rotation (in case it's 3x4 or 4x4)
                R = R[:3, :3]
                
                t_vec = transform.translation
                if t_vec.ndim > 1:  # Batched: shape [1, 3]
                    t = t_vec[0].numpy()
                else:  # Unbatched: shape [3]
                    t = t_vec.numpy()
                
                s = transform.scale.item() if hasattr(transform.scale, 'item') else float(transform.scale)
                
                # Apply T(x) = s R x + t
                aligned = s * (source_points @ R.T) + t
                aligned_points.append(aligned)
                aligned_colors.append(source_colors)
                
                print(f"    Frame {i}: scale={s:.4f}, translation_norm={np.linalg.norm(t):.4f}")
            except Exception as e:
                print(f"    Frame {i}: alignment failed ({e}), using original points")
                aligned_points.append(source_points)
                aligned_colors.append(source_colors)
        
        return aligned_points, aligned_colors

    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd: np.ndarray | None,
        pcd_mask: np.ndarray | None = None,
    ) -> SceneFrameHandle:
        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=0.0,
            axes_radius=0.0,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )
        frame_height, frame_width = rgb.shape[:2]

        frame_thumbnail = Image.fromarray(rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=frame_width / frame_height,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        if pcd is not None:
            pcd = pcd.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            if pcd_mask is not None:
                pcd_mask = pcd_mask.reshape(-1)
                pcd = pcd[pcd_mask]
                rgb = rgb[pcd_mask]
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud",
                points=pcd,
                colors=rgb,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        else:
            pcd_handle = None

        return SceneFrameHandle(
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
        )

    def _check_overlap(
        self,
        points_world: np.ndarray,
        prev_c2w: np.ndarray,
        prev_camera_model,
        prev_depth_map: np.ndarray,
        prev_rgb_map: np.ndarray,
        curr_rgb_points: np.ndarray,
        depth_threshold: float,
        rgb_threshold: float,
    ) -> np.ndarray:
        """
        Check if points in world frame are covered by the previous frame view.
        Returns a boolean mask where True means redundant (overlapped).
        """
        # 1. World -> Camera
        # w2c = inv(prev_c2w)
        R = prev_c2w[:3, :3].T
        t = -R @ prev_c2w[:3, 3]
        points_cam = (points_world @ R.T) + t # (N, 3)

        # 2. Project using previous camera model
        points_cam_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=-1)
        points_cam_homo_torch = torch.from_numpy(points_cam_homo).float()
        
        # camera_model.proj_points usually returns (coords, Jp, Jf)
        # We assume prev_camera_model is on CPU
        coords, _, _ = prev_camera_model.proj_points(points_cam_homo_torch, compute_jp=False, compute_jf=False, limit_min_depth=False)
        coords = coords.numpy()

        # 3. Check bounds
        u, v = coords[:, 0], coords[:, 1]
        z_curr = points_cam[:, 2] # Use Z in camera frame
        
        H, W = prev_depth_map.shape
        valid_proj_mask = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z_curr > 0)
        
        # 4. Compare Depth for valid projections
        is_overlapped = np.zeros(len(points_world), dtype=bool)
        
        if valid_proj_mask.any():
            # Nearest neighbor sampling
            u_int = np.clip(np.round(u[valid_proj_mask]).astype(int), 0, W - 1)
            v_int = np.clip(np.round(v[valid_proj_mask]).astype(int), 0, H - 1)
            
            d_prev = prev_depth_map[v_int, u_int]
            z_curr_valid = z_curr[valid_proj_mask]
            
            # Check if prev depth is valid
            valid_depth_mask = (d_prev > 1e-3) # Assuming 0 is invalid
            
            # Compute depth difference
            abs_diff = np.abs(z_curr_valid - d_prev)

            # Compute color difference between current point and previous frame pixel
            # Ensure float for distance computation
            prev_colors = prev_rgb_map[v_int, u_int].astype(np.float32)
            curr_colors_valid = curr_rgb_points[valid_proj_mask].astype(np.float32)
            color_diff = np.linalg.norm(prev_colors - curr_colors_valid, axis=1)
            rgb_similar = color_diff <= rgb_threshold

            # Overlap rule:
            # - If depth difference is small AND rgb is similar -> overlapped (redundant)
            # - If depth difference is small BUT rgb differs a lot -> not overlapped (keep)
            # - If depth difference is large -> not overlapped (keep regardless of rgb)
            is_depth_close = (abs_diff < depth_threshold)
            overlap_indices = valid_depth_mask & is_depth_close & rgb_similar
            
            # Map back to full mask
            temp_mask = np.zeros(np.sum(valid_proj_mask), dtype=bool)
            temp_mask[overlap_indices] = True
            is_overlapped[valid_proj_mask] = temp_mask
            
        return is_overlapped

    def _incr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _rebuild_playback_gui(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        self.gui_name.value = current_artifact.artifact_name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=len(self.scene_frame_handles) - 1, step=1, initial_value=0
            )
            gui_frame_control = self.client.gui.add_button_group("Control", options=["Prev", "Next"])
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=15)

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "Prev":
                    self._decr_timestep()
                else:
                    self._incr_timestep()

            self.current_displayed_timestep = self.gui_timestep.value

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_timestep = self.gui_timestep.value
                prev_timestep = self.current_displayed_timestep
                with self.client.atomic():
                    self.scene_frame_handles[current_timestep].visible = True
                    self.scene_frame_handles[prev_timestep].visible = False
                self.current_displayed_timestep = current_timestep

    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def get_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def run_viser(base_path: Path, port: int = 20540):
    # Get list of artifacts.
    logger.info(f"Loading artifacts from {base_path}")
    artifacts: list[ArtifactPath] = list(ArtifactPath.glob_artifacts(base_path, use_video=True))
    if len(artifacts) == 0:
        logger.error("No artifacts found. Exiting.")
        return

    global _global_context
    _global_context = GlobalContext(artifacts=sorted(artifacts, key=lambda x: x.artifact_name))

    server = viser.ViserServer(host=get_host_ip(), port=port, verbose=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        # wait synchronously in this function for task to be finished.
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Visualizer")
    parser.add_argument("base_path", type=Path, help="Base path for the visualizer")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=20540,
        help="Port number for the viser server.",
    )
    args = parser.parse_args()

    run_viser(args.base_path, args.port)


if __name__ == "__main__":
    main()
