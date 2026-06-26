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

import os
import torch

from einops import rearrange

from egosim_state.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .priorda import PriorDepthAnything


class PriorDAModel(DepthEstimationModel):
    """
    https://github.com/SpatialVision/Prior-Depth-Anything
    """

    def __init__(
        self,
        weights_dir: str | None = None,
        frozen_model_size: str | None = None,
        conditioned_model_size: str | None = None,
    ) -> None:
        super().__init__()
        
        # Support environment variable for weights directory
        if weights_dir is None:
            weights_dir = os.environ.get("PRIORDA_WEIGHTS_DIR")
        
        # Fallback to a project-local weights directory if one was prepared.
        if weights_dir is None:
            import pathlib
            project_root = pathlib.Path(__file__).parent.parent.parent.parent.parent
            weights_dir = str(project_root / "weights" / "priorda")
            if not os.path.exists(weights_dir):
                weights_dir = None  # Let it download from HuggingFace
        
        self.model = PriorDepthAnything(
            device="cuda",
            fmde_dir=weights_dir,
            cmde_dir=weights_dir,
            ckpt_dir=weights_dir,
            frozen_model_size=frozen_model_size,
            conditioned_model_size=conditioned_model_size,
        )

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        prompt_metric_depth: torch.Tensor = unpack_optional(src.prompt_metric_depth)

        assert rgb.dim() == 3 and prompt_metric_depth.dim() == 2, "Single batch only"
        final_depth = self.model.infer_one_sample(
            image=rgb * 255.0,
            prior=prompt_metric_depth,
            geometric=None,
        )

        return DepthEstimationResult(metric_depth=final_depth)
