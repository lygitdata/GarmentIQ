"""Matting Anything (MAM) wired to the Hugging Face SAM implementation.

MAM refines a coarse SAM mask into a soft alpha matte. Upstream ships its own fork of
segment-anything exposing a custom `forward_m2m`; GarmentIQ reproduces that behaviour on
top of the standard Hugging Face `SamModel` so a single SAM checkpoint serves both the
segmentation and matting modules.

The decoder architecture and weights come from Matting Anything
(https://github.com/SHI-Labs/Matting-Anything), Copyright (c) 2023 SHI Labs, MIT License.
MAM keeps SAM frozen during training, so any standard SAM checkpoint of the matching
variant is equivalent to the one embedded in the published MAM checkpoint.
"""

import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from garmentiq.matting.model_definition.mam.m2m import sam_decoder_deep
from garmentiq.utils.device import resolve_device

# Elliptical structuring elements indexed by width, matching upstream MAM.
_KERNELS = [None] + [
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)
]


def _unknown_region_from_mask(mask: torch.Tensor, rand_width: int = 30):
    """
    Finds the uncertain band just outside a hard guidance mask.

    Everything inside the mask, and everything comfortably outside it, is treated as
    confident; only the ring between them is left for the matting decoder to refine.

    Args:
        mask (torch.Tensor): Binary mask of shape `(N, 1, H, W)`.
        rand_width (int, optional): Erosion width controlling band thickness. Default is 30.

    Returns:
        torch.Tensor: Binary weight map of shape `(N, 1, H, W)` on the mask's device.
    """
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    weight = np.ones_like(mask_np, dtype=np.uint8)

    width = max(1, min(rand_width // 2, len(_KERNELS) - 1))
    for i in range(mask_np.shape[0]):
        fg = mask_np[i, 0]
        bg = cv2.erode(1 - mask_np[i, 0], _KERNELS[width])
        weight[i, 0][fg == 1] = 0
        weight[i, 0][bg == 1] = 0

    return torch.from_numpy(weight).to(mask.device)


def _unknown_region_from_pred(pred: torch.Tensor, rand_width: int = 30):
    """
    Finds the uncertain (soft) band of an alpha prediction.

    Progressive fusion replaces only the uncertain band of a coarse prediction with a
    finer one, which is how MAM keeps confident interior/exterior regions stable while
    sharpening edges.

    Args:
        pred (torch.Tensor): Alpha prediction of shape `(N, 1, H, W)` in `[0, 1]`.
        rand_width (int, optional): Dilation width controlling band thickness. Default is 30.

    Returns:
        torch.Tensor: Binary weight map of shape `(N, 1, H, W)` on the same device as `pred`.
    """
    n = pred.shape[0]
    pred_np = pred.detach().cpu().numpy()
    uncertain = np.ones_like(pred_np, dtype=np.uint8)
    uncertain[pred_np < 1.0 / 255.0] = 0

    width = max(1, min(rand_width // 2, len(_KERNELS) - 1))
    for i in range(n):
        uncertain[i, 0] = cv2.dilate(uncertain[i, 0], _KERNELS[width])

    uncertain[pred_np > 1 - 1.0 / 255.0] = 0
    # Upstream hardcodes .cuda(); keep the weight on the caller's device instead.
    return torch.from_numpy(uncertain).to(pred.device)


class MattingAnything(nn.Module):
    """
    Couples a frozen Hugging Face SAM with the Matting Anything M2M decoder.

    SAM supplies both the image embedding and a coarse mask for whatever prompt is given;
    the M2M decoder turns that pair into a soft alpha matte. Unlike ViTMatte, no trimap is
    required, because the SAM mask itself acts as the guidance signal.

    Attributes:
        sam_model (nn.Module): The frozen SAM model providing embeddings and coarse masks.
        sam_processor: The processor matching `sam_model`.
        m2m (nn.Module): The mask-to-matte decoder.
    """

    def __init__(self, sam_model, sam_processor, m2m):
        """
        Args:
            sam_model (nn.Module): A loaded Hugging Face `SamModel`.
            sam_processor: The matching `SamProcessor`.
            m2m (nn.Module): The `SAM_Decoder_Deep` instance holding MAM weights.
        """
        super().__init__()
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        self.m2m = m2m

    def forward(self, pixel_values, reshaped_size=None, original_size=None, **sam_prompt_kwargs):
        """
        Produces multi-scale alpha predictions plus the full-resolution SAM mask.

        Args:
            pixel_values (torch.Tensor): SAM-preprocessed image, shape `(N, 3, 1024, 1024)`.
            reshaped_size (tuple[int, int], optional): Image size after SAM resizing but before
                                                       padding, as `(height, width)`. Required to
                                                       return a full-resolution mask.
            original_size (tuple[int, int], optional): Original image size as `(height, width)`.
            **sam_prompt_kwargs: Prompt tensors accepted by `SamModel`, such as
                                 `input_points`, `input_labels`, or `input_boxes`.

        Returns:
            tuple[dict, torch.Tensor | None]: The decoder output (`alpha_os1`, `alpha_os4`,
                                              `alpha_os8`, `mask`) and the binarised SAM mask
                                              at original resolution, or None when sizes are
                                              not supplied.
        """
        self.sam_model.eval()
        with torch.no_grad():
            image_embeddings = self.sam_model.get_image_embeddings(pixel_values)
            outputs = self.sam_model(
                image_embeddings=image_embeddings,
                multimask_output=True,
                **sam_prompt_kwargs,
            )
            # pred_masks: (batch, point_batch, num_masks, 256, 256)
            low_res = outputs.pred_masks[:, 0]
            iou = outputs.iou_scores[:, 0]
            best = int(torch.argmax(iou[0]))
            selected = low_res[:, best : best + 1]
            guide_mask = (selected > 0.0).float()

            post_mask = None
            if reshaped_size is not None and original_size is not None:
                # Mirror SAM's postprocess_masks: upsample to the padded canvas, drop the
                # padding, then resize to the true image size.
                upscaled = F.interpolate(
                    selected,
                    (pixel_values.shape[-2], pixel_values.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                rh, rw = int(reshaped_size[0]), int(reshaped_size[1])
                upscaled = upscaled[..., :rh, :rw]
                upscaled = F.interpolate(
                    upscaled,
                    (int(original_size[0]), int(original_size[1])),
                    mode="bilinear",
                    align_corners=False,
                )
                post_mask = (upscaled > 0.0).float()

        return self.m2m(image_embeddings, pixel_values, guide_mask), post_mask


def load_mam(
    checkpoint_path: str,
    sam_model,
    sam_processor,
    device: Union[str, torch.device] = "cpu",
):
    """
    Loads Matting Anything decoder weights and pairs them with a SAM model.

    The published MAM checkpoint stores both the frozen SAM (`module.seg_model.*`) and the
    matting decoder (`module.m2m.*`). Only the decoder weights are read here; SAM comes from
    the model you pass in, so the same SAM checkpoint can be shared with the segmentation
    module. Loading is strict, so a mismatched checkpoint fails loudly instead of silently
    producing a meaningless matte.

    Args:
        checkpoint_path (str): Local path to a MAM checkpoint, e.g. `mam_sam_vitb.pth`.
        sam_model (nn.Module): A loaded Hugging Face `SamModel` of the matching variant.
        sam_processor: The matching `SamProcessor`.
        device (Union[str, torch.device], optional): Device to place the model on, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`.
                                                     Default is `"cpu"`.

    Raises:
        FileNotFoundError: If `checkpoint_path` does not exist.
        ValueError: If the checkpoint contains no `module.m2m.*` weights, or if the decoder
                    weights do not match the expected architecture.

    Returns:
        MattingAnything: The assembled model in evaluation mode, placed on `device`.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MAM checkpoint not found at {checkpoint_path}.")

    device = resolve_device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    m2m_state = {
        k[len("module.m2m.") :]: v
        for k, v in state_dict.items()
        if k.startswith("module.m2m.")
    }
    if not m2m_state:
        # Fall back to a checkpoint holding only the decoder.
        m2m_state = {
            k[len("m2m.") :]: v for k, v in state_dict.items() if k.startswith("m2m.")
        }
    if not m2m_state:
        raise ValueError(
            f"No Matting Anything decoder weights found in {checkpoint_path!r}. "
            f"Expected keys prefixed with 'module.m2m.' or 'm2m.'."
        )

    m2m = sam_decoder_deep(nc=256)
    missing, unexpected = m2m.load_state_dict(m2m_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Matting Anything decoder weights do not match the expected architecture. "
            f"{len(missing)} missing and {len(unexpected)} unexpected tensors "
            f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]})."
        )

    model = MattingAnything(sam_model=sam_model, sam_processor=sam_processor, m2m=m2m)
    model = model.to(device)
    model.eval()
    return model


def fuse_alpha(
    pred: dict,
    reshaped_size,
    original_size,
    post_mask: Optional[torch.Tensor] = None,
):
    """
    Fuses MAM's three alpha predictions into a single full-resolution matte.

    When the hard SAM mask is supplied, fusion follows upstream's mask-guidance mode: the
    matte starts from the confident mask and only the uncertain ring around its border is
    replaced by predicted alpha, progressively from the coarsest scale to the finest. This
    is what keeps the garment interior fully opaque while edges stay soft. Without the mask
    the matte starts from the coarsest prediction instead, which leaves interiors dim.

    Args:
        pred (dict): Decoder output containing `alpha_os1`, `alpha_os4`, and `alpha_os8`.
        reshaped_size (tuple[int, int]): Image size after SAM resizing but before padding,
                                         as `(height, width)`.
        original_size (tuple[int, int]): Original image size as `(height, width)`.
        post_mask (torch.Tensor, optional): Binarised SAM mask at original resolution,
                                            shape `(1, 1, H, W)`. Default is None.

    Returns:
        torch.Tensor: Fused alpha of shape `(1, 1, height, width)` in `[0, 1]`.
    """
    h, w = int(reshaped_size[0]), int(reshaped_size[1])
    os1 = pred["alpha_os1"][..., :h, :w]
    os4 = pred["alpha_os4"][..., :h, :w]
    os8 = pred["alpha_os8"][..., :h, :w]

    target = (int(original_size[0]), int(original_size[1]))
    os1 = F.interpolate(os1, target, mode="bilinear", align_corners=False)
    os4 = F.interpolate(os4, target, mode="bilinear", align_corners=False)
    os8 = F.interpolate(os8, target, mode="bilinear", align_corners=False)

    if post_mask is not None:
        alpha = post_mask.clone()
        weight_os8 = _unknown_region_from_mask(post_mask, rand_width=10)
        alpha[weight_os8 > 0] = os8[weight_os8 > 0]
    else:
        alpha = os8.clone()

    weight_os4 = _unknown_region_from_pred(alpha, rand_width=20)
    alpha[weight_os4 > 0] = os4[weight_os4 > 0]
    weight_os1 = _unknown_region_from_pred(alpha, rand_width=10)
    alpha[weight_os1 > 0] = os1[weight_os1 > 0]

    return alpha.clamp(0.0, 1.0)


__all__ = ["MattingAnything", "load_mam", "fuse_alpha"]
