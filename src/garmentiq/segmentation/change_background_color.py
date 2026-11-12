"""Replacing the background of an image using a segmentation mask."""
import numpy as np


def change_background_color(
    image_np: np.ndarray, mask_np: np.ndarray, background_color: tuple[int, int, int]
):
    """
    Changes the background color of an image based on a given mask.

    This function modifies the background of an image by replacing the pixels
    that correspond to the background (as identified by the mask) with a specified color.

    Args:
        image_np (numpy.ndarray): The original image as a numpy array (RGB).
        mask_np (numpy.ndarray): A binary, normalized soft, or 8-bit mask. Soft
                                 masks are thresholded at 0.5 (127 for 8-bit masks).
        background_color (tuple[int, int, int]): A tuple representing the RGB color to replace the background with,
                                                 e.g., (255, 255, 255) for white.

    Returns:
        numpy.ndarray: A new image with the background color changed.
    """
    if mask_np.ndim != 2 or mask_np.shape != image_np.shape[:2]:
        raise ValueError("The mask must be 2D and match the image dimensions.")
    # BiRefNet includes near-zero background probabilities, not just exact zeros.
    threshold = 0.5 if mask_np.max() <= 1 else 127
    background_mask = mask_np <= threshold

    # Create a copy of the original image to modify
    modified_image = image_np.copy()

    # Change the background areas to the specified background color
    modified_image[background_mask] = background_color

    return modified_image
