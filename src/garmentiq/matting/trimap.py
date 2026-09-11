"""Building a trimap from a segmentation mask.

A trimap marks pixels as definite foreground, definite background, or unknown. The
unknown band is the only region where a matting model may predict soft alpha, so its
width controls how much of the edge can be refined.
"""
import cv2
import numpy as np


def generate_trimap(
    mask: np.ndarray,
    erode_size: int = 10,
    dilate_size: int = 10,
    threshold: int = 127,
):
    """
    Builds a trimap from a binary segmentation mask.

    Trimap-based matting models such as ViTMatte need to be told which pixels are
    definitely foreground, definitely background, and uncertain. This function derives
    that from a segmentation mask by eroding it to obtain confident foreground, dilating
    it to obtain confident background, and marking the band in between as unknown, which
    is where the matting model is free to predict soft alpha values.

    The size of the unknown band matters: too narrow and hair or fabric edges fall outside
    it and can never become soft, too wide and the model has to guess over large areas.

    Args:
        mask (numpy.ndarray): Segmentation mask as a 2D array. Values may be binary
                              (`0`/`1`) or 8-bit (`0`-`255`).
        erode_size (int, optional): Erosion kernel size controlling how far the confident
                                    foreground is pulled inward. Default is 10.
        dilate_size (int, optional): Dilation kernel size controlling how far the unknown
                                     band extends outward. Default is 10.
        threshold (int, optional): Cutoff used to binarise an 8-bit mask. Default is 127.

    Raises:
        ValueError: If `mask` is not 2D, or if `erode_size` or `dilate_size` is negative.

    Returns:
        numpy.ndarray: Trimap as `uint8` with values `0` (background), `128` (unknown),
                       and `255` (foreground), matching `mask` in shape.
    """
    if mask is None or not hasattr(mask, "ndim"):
        raise ValueError("mask must be a numpy array.")

    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be a 2D array, got shape {mask.shape}. Pass a single-channel "
            f"segmentation mask."
        )
    if erode_size < 0 or dilate_size < 0:
        raise ValueError(
            f"erode_size and dilate_size must be non-negative, got "
            f"erode_size={erode_size}, dilate_size={dilate_size}."
        )

    if mask.dtype == bool:
        binary = mask.astype(np.uint8)
    elif mask.max() <= 1:
        binary = (mask > 0).astype(np.uint8)
    else:
        binary = (mask > threshold).astype(np.uint8)

    foreground = binary
    if erode_size > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size)
        )
        foreground = cv2.erode(binary, erode_kernel, iterations=1)

    background = binary
    if dilate_size > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
        )
        background = cv2.dilate(binary, dilate_kernel, iterations=1)

    trimap = np.full(binary.shape, 128, dtype=np.uint8)
    trimap[background == 0] = 0
    trimap[foreground == 1] = 255
    return trimap


__all__ = ["generate_trimap"]
