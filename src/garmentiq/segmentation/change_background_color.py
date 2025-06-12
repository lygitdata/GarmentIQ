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
        mask_np (numpy.ndarray): A binary mask where background pixels are labeled as 0 and foreground as 1.
        background_color (tuple[int, int, int]): A tuple representing the RGB color to replace the background with,
                                                 e.g., (255, 255, 255) for white.

    Returns:
        numpy.ndarray: A new image with the background color changed.
    """
    # Create a mask where the background (black) is 1, and the foreground is 0
    background_mask = mask_np == 0  # Mask values of 0 represent background

    # Create a copy of the original image to modify
    modified_image = image_np.copy()

    # Change the background areas to the specified background color
    modified_image[background_mask] = background_color

    return modified_image
