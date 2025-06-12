import cv2


def refine_landmark_with_blur(x, y, blurred_mask, window_size=5):
    """
    Refine the landmark location using a Gaussian-blurred version of the segmentation mask.
    The function searches a small window around (x, y) in the blurred mask and returns
    the coordinates of the pixel with the maximum value, which is assumed to be a more
    reliable location for the landmark.

    Args:
        x (float): The initial x-coordinate of the landmark.
        y (float): The initial y-coordinate of the landmark.
        blurred_mask (np.ndarray): The Gaussian-blurred segmentation mask.
        window_size (int, optional): The size of the square window (in pixels) around (x, y)
                                    to search for the maximum value. Defaults to 5.

    Returns:
        tuple: A tuple (refined_x, refined_y) representing the refined coordinates.
    """
    height, width = blurred_mask.shape
    x_min = int(max(0, x - window_size))
    x_max = int(min(width, x + window_size + 1))
    y_min = int(max(0, y - window_size))
    y_max = int(min(height, y + window_size + 1))

    # Extract the local region from the blurred mask.
    local_region = blurred_mask[y_min:y_max, x_min:x_max]
    # Find the location of the maximum value within the local region.
    _, max_val, _, max_loc = cv2.minMaxLoc(local_region)

    # Compute refined coordinates relative to the entire image.
    refined_x = x_min + max_loc[0]
    refined_y = y_min + max_loc[1]
    return refined_x, refined_y
