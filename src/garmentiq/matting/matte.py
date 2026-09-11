"""Producing an alpha matte and compositing with it.

Dispatches on the model family: ViTMatte is guided by a trimap, which can be supplied
directly or derived from a segmentation mask, while Matting Anything is guided by a SAM
prompt and needs no trimap at all.
"""
from PIL import Image
import numpy as np
import torch
import warnings
from typing import Optional, Union

from garmentiq.utils.device import resolve_device, empty_cache, inputs_to_device
from garmentiq.matting.trimap import generate_trimap

# Above roughly this many pixels the MPS backend has been observed to diverge from CPU
# for ViTMatte, silently producing a degenerate matte. Verified identical up to
# 2048x1536 (3.15 MP) and divergent at 2400x1800 (4.32 MP) on Apple Silicon.
_MPS_PIXEL_WARN_THRESHOLD = 3_500_000


def _detect_matting_family(model):
    """
    Identifies which matting approach a loaded model implements.

    Detection walks the class hierarchy by name rather than importing the transformers
    classes, so a model is recognised without requiring every backend to be installed.
    """
    names = {cls.__name__ for cls in type(model).__mro__}
    if "MattingAnything" in names:
        return "mam"
    if {"VitMatteForImageMatting", "VitMattePreTrainedModel"} & names:
        return "vitmatte"
    return None


def matte(
    model,
    image_path: Union[str, np.ndarray, Image.Image],
    processor=None,
    mask: Optional[np.ndarray] = None,
    trimap: Optional[np.ndarray] = None,
    prompt: Optional[dict] = None,
    trimap_args: Optional[dict] = None,
    device: Union[str, torch.device] = "cpu",
):
    """
    Extracts a soft alpha matte for an image, refining a hard segmentation into soft edges.

    Segmentation answers "which pixels belong to the garment" with a yes/no decision, which
    leaves stair-stepped borders and loses semi-transparent detail such as loose fabric,
    lace, and stray fibres. Matting instead predicts a continuous alpha value per pixel, so
    compositing onto a new background looks natural.

    The two supported approaches need different guidance:
        - **ViTMatte** requires a trimap marking definite foreground, definite background,
          and the uncertain band between them. Pass `trimap` directly, or pass `mask` and a
          trimap will be derived from it via `generate_trimap`.
        - **Matting Anything (MAM)** needs no trimap. It prompts a frozen SAM with `prompt`
          and refines the resulting coarse mask into an alpha matte.

    Args:
        model: A loaded matting model, either `VitMatteForImageMatting` or `MattingAnything`.
        image_path (Union[str, numpy.ndarray, PIL.Image.Image]): The image to matte, given as
                                                                 a file path, RGB array, or
                                                                 PIL image.
        processor (VitMatteImageProcessor, optional): Required for ViTMatte, which uses it to
                                                      stack the image and trimap into a
                                                      four-channel input. Default is None.
        mask (numpy.ndarray, optional): A binary segmentation mask used to derive a trimap
                                        when `trimap` is not supplied. Default is None.
        trimap (numpy.ndarray, optional): An explicit trimap with values `0`, `128`, and
                                          `255`. Takes precedence over `mask`. Default is None.
        prompt (dict, optional): Prompt for MAM's internal SAM, accepting `"points"`,
                                 `"labels"`, and/or `"boxes"` exactly as
                                 `garmentiq.segmentation.extract` does. Default is None.
        trimap_args (dict, optional): Options forwarded to `generate_trimap`, such as
                                      `"erode_size"` and `"dilate_size"`. Default is None.
        device (Union[str, torch.device], optional): The device to run inference on, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`. Hardware
                                                     acceleration is opt-in. Default is `"cpu"`.

    Raises:
        ValueError: If the model is not a recognised matting model, if ViTMatte is used
                    without a processor or without either `trimap` or `mask`, if MAM is used
                    without a prompt, or if the requested `device` is unavailable.
        FileNotFoundError: If `image_path` points to a file that does not exist.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray): The original image as an RGB array, and the
                                              alpha matte as a `uint8` array in `[0, 255]`
                                              matching the image's height and width.
    """
    device = resolve_device(device)
    family = _detect_matting_family(model)
    if family is None:
        raise ValueError(
            "Unrecognised matting model. Expected a ViTMatte model "
            "(VitMatteForImageMatting) or a Matting Anything model (MattingAnything)."
        )

    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path.astype(np.uint8)).convert("RGB")
    else:
        image = image_path.convert("RGB")

    image_np = np.array(image)
    model = model.to(device)

    if (
        device.type == "mps"
        and image_np.shape[0] * image_np.shape[1] > _MPS_PIXEL_WARN_THRESHOLD
    ):
        warnings.warn(
            f"Matting a {image_np.shape[1]}x{image_np.shape[0]} image on the MPS backend. "
            f"Above roughly {_MPS_PIXEL_WARN_THRESHOLD / 1e6:.1f} megapixels MPS has been "
            f"observed to return a degenerate alpha matte that differs substantially from "
            f"CPU. Consider device='cpu' for images this large, or downscale first.",
            RuntimeWarning,
            stacklevel=2,
        )

    if family == "vitmatte":
        if processor is None:
            raise ValueError(
                "ViTMatte requires a processor. Pass processor=load_vitmatte_processor(...)."
            )
        if trimap is None:
            if mask is None:
                raise ValueError(
                    "ViTMatte is a trimap-based model, so it needs either an explicit "
                    "trimap= or a segmentation mask= to derive one from. Run segmentation "
                    "first, then pass its mask here."
                )
            trimap = generate_trimap(mask, **(trimap_args or {}))

        trimap_arr = np.asarray(trimap)
        if trimap_arr.shape[:2] != image_np.shape[:2]:
            raise ValueError(
                f"trimap shape {trimap_arr.shape[:2]} does not match image shape "
                f"{image_np.shape[:2]}."
            )

        # The processor rescales trimaps by its own rescale_factor (1/255), so hand it
        # the raw 0-255 trimap. Pre-normalising here would rescale twice and collapse
        # the trimap to near-zero, making the model see everything as background.
        inputs = inputs_to_device(
            processor(
                images=image, trimaps=trimap_arr.astype(np.uint8), return_tensors="pt"
            ),
            device,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        alpha = outputs.alphas
        # ViTMatte pads to a multiple of its patch size; crop back to the true size.
        alpha = alpha[..., : image_np.shape[0], : image_np.shape[1]]
        alpha_np = alpha[0, 0].detach().cpu().numpy()
        del inputs, outputs

    else:
        from garmentiq.matting.model_definition.mam.mam import fuse_alpha

        if not prompt:
            raise ValueError(
                "Matting Anything prompts a frozen SAM, so it needs a prompt. Pass "
                "prompt={'points': [[[x, y]]]} or prompt={'boxes': [[[x0, y0, x1, y1]]]}."
            )

        sam_processor = model.sam_processor
        processor_kwargs = {}
        if prompt.get("points") is not None:
            processor_kwargs["input_points"] = prompt["points"]
        if prompt.get("labels") is not None:
            processor_kwargs["input_labels"] = prompt["labels"]
        if prompt.get("boxes") is not None:
            processor_kwargs["input_boxes"] = prompt["boxes"]

        inputs = inputs_to_device(
            sam_processor(images=image, return_tensors="pt", **processor_kwargs),
            device,
        )

        pixel_values = inputs.pop("pixel_values")
        original_sizes = inputs.pop("original_sizes")
        reshaped_sizes = inputs.pop("reshaped_input_sizes")

        with torch.no_grad():
            pred, post_mask = model(
                pixel_values=pixel_values,
                reshaped_size=reshaped_sizes[0].tolist(),
                original_size=original_sizes[0].tolist(),
                **inputs,
            )
            alpha = fuse_alpha(
                pred,
                reshaped_size=reshaped_sizes[0].tolist(),
                original_size=original_sizes[0].tolist(),
                post_mask=post_mask,
            )

        alpha_np = alpha[0, 0].detach().cpu().numpy()
        del inputs, pred

    alpha_np = np.clip(alpha_np, 0.0, 1.0)
    alpha_u8 = (alpha_np * 255).astype(np.uint8)

    empty_cache(device)
    return image_np, alpha_u8


def composite(
    image_np: np.ndarray,
    alpha_np: np.ndarray,
    background_color=(255, 255, 255),
):
    """
    Composites a matted foreground onto a solid background colour.

    Unlike hard mask replacement, this blends each pixel by its alpha value, so soft edges
    stay soft instead of showing a cut-out outline.

    Args:
        image_np (numpy.ndarray): The original RGB image, shape `(H, W, 3)`.
        alpha_np (numpy.ndarray): The alpha matte, shape `(H, W)`, `uint8` in `[0, 255]`
                                  or float in `[0, 1]`.
        background_color (tuple[int, int, int], optional): RGB colour to composite onto.
                                                           Default is white.

    Raises:
        ValueError: If the image and alpha shapes do not match.

    Returns:
        numpy.ndarray: The composited RGB image as `uint8`, shape `(H, W, 3)`.
    """
    image_np = np.asarray(image_np)
    alpha_np = np.asarray(alpha_np)

    if image_np.shape[:2] != alpha_np.shape[:2]:
        raise ValueError(
            f"image shape {image_np.shape[:2]} does not match alpha shape "
            f"{alpha_np.shape[:2]}."
        )

    alpha = alpha_np.astype(np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    alpha = alpha[..., None]

    background = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
    out = alpha * image_np.astype(np.float32) + (1.0 - alpha) * background
    return np.clip(out, 0, 255).astype(np.uint8)


__all__ = ["matte", "composite"]
