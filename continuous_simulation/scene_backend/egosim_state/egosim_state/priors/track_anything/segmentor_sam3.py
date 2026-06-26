# SAM 3 Segmentor for Track Anything Pipeline
# Replaces the original SAM segmentor with SAM 3

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from egosim_state.utils.model_paths import get_sam3_checkpoint_path

SAM3_LOCAL_CHECKPOINT = get_sam3_checkpoint_path()


class Segmentor:

    def segment_with_dino_box_center(self, origin_frame, bbox, label=1, reset_image=False, phrase=None):
        """
        Use GroundingDINO detected box center as click point for SAM3 point prompt,
        optionally with text phrase as additional condition.
        Args:
            origin_frame: numpy array (H, W, 3) RGB image
            bbox: [x1, y1, x2, y2] (pixel coordinates, float or int)
            label: 1 for foreground, 0 for background
            reset_image: whether to reset image embedding
            phrase: optional text description of the object (e.g., "hand", "cup")
        Returns:
            mask: binary mask (H, W) uint8
        """
        x1, y1, x2, y2 = map(float, bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.segment_with_point(origin_frame, (cx, cy), label=label, reset_image=reset_image, phrase=phrase)

    @torch.no_grad()
    def segment_with_point(self, origin_frame, point, label=1, radius=0, reset_image=False, phrase=None):
        """
        Segment with a single point prompt (native SAM3 geometric prompt),
        optionally with text phrase as additional condition.
        Args:
            origin_frame: numpy array (H, W, 3) RGB image
            point: (x, y) pixel coordinates (float or int)
            label: 1 for foreground, 0 for background
            radius: ignored (for API compatibility)
            reset_image: whether to reset image embedding
            phrase: optional text description of the object (e.g., "hand", "cup")
        Returns:
            mask: binary mask (H, W) uint8
        """
        if reset_image:
            self.reset_image()
        self.set_image(origin_frame)

        H, W = origin_frame.shape[:2]
        # Normalize point to [0,1]
        x, y = float(point[0]), float(point[1])
        norm_point = [x / W, y / H]

        # Prepare tensor for SAM3 geometric_prompt.append_points
        pts = torch.tensor(norm_point, dtype=torch.float32, device=self.device).view(1, 1, 2)  # [seq=1, bs=1, 2]
        lbl = torch.tensor([[label]], dtype=torch.long, device=self.device)  # [seq=1, bs=1]


        # Ensure geometric_prompt exists
        if "geometric_prompt" not in self.inference_state:
            self.inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
        self.inference_state["geometric_prompt"].append_points(pts, lbl)

        # Set language features - use phrase if provided, otherwise use dummy
        if "backbone_out" in self.inference_state:
            text_prompt = phrase if phrase else "visual"
            text_outputs = self.model.backbone.forward_text([text_prompt], device=self.device)
            self.inference_state["backbone_out"].update(text_outputs)

        # Run forward
        output = self.processor._forward_grounding(self.inference_state)

        masks = output.get("masks", None)
        scores = output.get("scores", None)
        if masks is None or (isinstance(masks, torch.Tensor) and masks.numel() == 0):
            return np.zeros(origin_frame.shape[:2], dtype=np.uint8)
        if isinstance(scores, torch.Tensor):
            if scores.numel() == 0:
                return np.zeros(origin_frame.shape[:2], dtype=np.uint8)
            best_idx = int(torch.argmax(scores).item())
        else:
            best_idx = int(np.argmax(scores)) if scores is not None else 0
        if isinstance(masks, torch.Tensor):
            mask_t = masks[best_idx]
            if mask_t.dim() == 3 and mask_t.size(0) == 1:
                mask_t = mask_t.squeeze(0)
            mask_np = mask_t.detach().to("cpu").numpy().astype(np.uint8)
        else:
            mask_np = np.asarray(masks[best_idx], dtype=np.uint8)
        return mask_np
    """
    SAM 3 based segmentor that provides the same interface as the original SAM segmentor.
    """

    def __init__(self, sam_args):
        """
        sam_args:
            gpu_id: device
            sam3_checkpoint: local path to SAM3 checkpoint (optional)
            Other args are kept for compatibility but not used with SAM 3
        """
        self.device = sam_args["gpu_id"]
        
        # Get checkpoint path from args or use default
        checkpoint_path = sam_args.get("sam3_checkpoint", str(SAM3_LOCAL_CHECKPOINT))
        
        # Build SAM 3 model from local checkpoint (disable HF download)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(self.model)
        
        self.have_embedded = False
        self.inference_state = None
        self.current_image = None

    @torch.no_grad()
    def set_image(self, image):
        """
        Set the image for segmentation.
        
        Args:
            image: numpy array (H, W, 3) RGB image, uint8
        """
        if not self.have_embedded:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            self.inference_state = self.processor.set_image(pil_image)
            self.current_image = image
            self.have_embedded = True

    def reset_image(self):
        """Reset the image embedding state."""
        self.have_embedded = False
        self.inference_state = None
        self.current_image = None

    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):
        """
        Predict masks using prompts.
        This is a compatibility layer - SAM 3 has different API.
        """
        assert self.have_embedded, "Image embedding for SAM 3 needs to be set before predict."

        if mode == "point":
            # SAM 3 point prompting
            point_coords = prompts["point_coords"]
            point_labels = prompts["point_modes"]
            
            output = self.processor.set_point_prompt(
                state=self.inference_state,
                points=point_coords,
                labels=point_labels,
            )
            
        elif mode == "mask":
            # SAM 3 doesn't directly support mask prompts in the same way
            # Return empty results as fallback
            return np.zeros((1, *self.current_image.shape[:2])), np.array([0.0]), np.zeros((1, 256, 256))
            
        elif mode == "point_mask":
            # Combine point and mask - use points only for SAM 3
            point_coords = prompts["point_coords"]
            point_labels = prompts["point_modes"]
            
            output = self.processor.set_point_prompt(
                state=self.inference_state,
                points=point_coords,
                labels=point_labels,
            )

        masks = output.get("masks", np.zeros((1, *self.current_image.shape[:2])))
        scores = output.get("scores", np.array([0.0]))
        
        # SAM 3 doesn't return logits in the same way, create dummy
        logits = np.zeros((len(masks), 256, 256))
        
        return masks, scores, logits

    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        """
        Segment with point clicks.

        Args:
            origin_frame: numpy array (H, W, 3) RGB image
            coords: point coordinates
            modes: point labels (1 for foreground, 0 for background)
            multimask: whether to return multiple masks
            
        Returns:
            mask: binary mask (H, W) uint8
        """
        self.set_image(origin_frame)

        output = self.processor.set_point_prompt(
            state=self.inference_state,
            points=coords,
            labels=modes,
        )
        
        masks = output.get("masks", [])
        scores = output.get("scores", [])
        
        if len(masks) == 0:
            return np.zeros(origin_frame.shape[:2], dtype=np.uint8)
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        return mask.astype(np.uint8)

    @torch.no_grad()
    def segment_with_box(self, origin_frame, bbox, reset_image=False):
        """
        Segment with bounding box prompt for SAM3.

        Args:
            origin_frame: numpy array (H, W, 3) RGB image
            bbox: [[x1, y1], [x2, y2]] bounding box coordinates (pixel)
            reset_image: whether to reset image embedding
        Returns:
            list containing the mask
        """
        if reset_image:
            self.reset_image()
        self.set_image(origin_frame)

        # Convert bbox [[x1, y1], [x2, y2]] -> [cx, cy, w, h] in pixel
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        H, W = origin_frame.shape[:2]
        # Normalize to [0,1]
        norm_box = [cx / W, cy / H, w / W, h / H]

        # Add geometric prompt and run inference
        output = self.processor.add_geometric_prompt(
            box=norm_box,
            label=True,
            state=self.inference_state,
        )
        
        masks = output.get("masks", None)
        scores = output.get("scores", None)

        # Handle empty results
        if masks is None or (isinstance(masks, torch.Tensor) and masks.numel() == 0):
            return [np.zeros(origin_frame.shape[:2], dtype=np.uint8)]

        # Determine best index using torch to avoid numpy conversion on CUDA tensors
        if isinstance(scores, torch.Tensor):
            if scores.numel() == 0:
                return [np.zeros(origin_frame.shape[:2], dtype=np.uint8)]
            best_idx = int(torch.argmax(scores).item())
        else:
            best_idx = int(np.argmax(scores)) if scores is not None else 0

        # Extract mask and convert to numpy uint8 [H, W]
        if isinstance(masks, torch.Tensor):
            # SAM3 stores masks as [N, 1, H, W] bool
            mask_t = masks[best_idx]
            if mask_t.dim() == 3 and mask_t.size(0) == 1:
                mask_t = mask_t.squeeze(0)
            mask_np = mask_t.detach().to("cpu").numpy().astype(np.uint8)
        else:
            # Fallback for non-tensor mask
            mask_np = np.asarray(masks[best_idx], dtype=np.uint8)

        return [mask_np]

    @torch.no_grad()
    def segment_with_text(self, origin_frame, text_prompt, reset_image=False):
        """
        Segment with text prompt - SAM 3's new capability!

        Args:
            origin_frame: numpy array (H, W, 3) RGB image
            text_prompt: text description of object to segment
            reset_image: whether to reset image embedding
            
        Returns:
            masks: list of masks
            boxes: list of bounding boxes
            scores: confidence scores
        """
        if reset_image:
            self.reset_image()
            
        self.set_image(origin_frame)
        
        output = self.processor.set_text_prompt(
            state=self.inference_state,
            prompt=text_prompt,
        )
        
        masks_t = output.get("masks", None)
        boxes_t = output.get("boxes", None)
        scores_t = output.get("scores", None)

        masks = []
        if isinstance(masks_t, torch.Tensor) and masks_t.numel() > 0:
            # [N, 1, H, W] -> list of [H, W] uint8
            m = masks_t.detach().to("cpu").squeeze(1).numpy().astype(np.uint8)
            masks = [m[i] for i in range(m.shape[0])]

        boxes = []
        if isinstance(boxes_t, torch.Tensor) and boxes_t.numel() > 0:
            boxes = boxes_t.detach().to("cpu").numpy().tolist()

        scores = []
        if isinstance(scores_t, torch.Tensor) and scores_t.numel() > 0:
            scores = scores_t.detach().to("cpu").numpy().tolist()

        return masks, boxes, scores
