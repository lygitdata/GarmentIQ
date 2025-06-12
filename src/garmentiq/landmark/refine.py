import numpy as np
import cv2
import copy
from garmentiq.landmark.refinement import refine_landmark_with_blur
from garmentiq.landmark.utils import (
    find_instruction_landmark_index,
    fill_instruction_landmark_coordinate,
)


def refine(
    class_name: str,
    detection_np: np.array,
    detection_conf: np.array,
    detection_dict: dict,
    mask: np.array,
    window_size: int = 5,
    ksize: tuple = (11, 11),
    sigmaX: float = 0.0,
):
    """
    Refines detected landmarks using a blurred mask and updates the detection dictionary.

    This function applies Gaussian blur to the given mask, then refines landmark coordinates
    based on their confidence scores and local intensity structure. Only landmarks with a
    confidence score greater than 0 are refined. The refined coordinates are used to update
    predefined landmarks in the detection dictionary.

    Args:
        class_name (str): The name of the class to access in the detection dictionary.
        detection_np (np.array): The initial landmark predictions. Shape: (1, N, 2).
        detection_conf (np.array): Confidence scores for each predicted landmark. Shape: (1, N, 1).
        detection_dict (dict): Dictionary containing landmark data for each class.
        mask (np.array): Grayscale mask image used to guide refinement.
        window_size (int, optional): Size of the window used in the refinement algorithm. Defaults to 5.
        ksize (tuple, optional): Kernel size for Gaussian blur. Must be odd integers. Defaults to (11, 11).
        sigmaX (float, optional): Gaussian kernel standard deviation in the X direction. Defaults to 0.0.

    Returns:
        tuple:
            - refined_detection_np (np.array): Array of the same shape as `detection_np` with refined coordinates.
            - detection_dict (dict): Updated detection dictionary with refined landmark coordinates.
    """
    blurred_mask = cv2.GaussianBlur(mask, ksize, sigmaX)

    refined_detection_np = np.zeros(detection_np.shape)

    for i, coord in enumerate(detection_np[0]):
        if detection_conf[0, i, 0] > 0:
            refined_x, refined_y = refine_landmark_with_blur(
                coord[0], coord[1], blurred_mask, window_size
            )
            refined_detection_np[0, i] = [refined_x, refined_y]

    predefined_index = find_instruction_landmark_index(
        detection_dict[class_name]["landmarks"], predefined=True
    )

    preds = refined_detection_np[:, predefined_index, :]

    detection_dict_copy = copy.deepcopy(detection_dict)

    detection_dict_copy[class_name]["landmarks"] = fill_instruction_landmark_coordinate(
        instruction_landmarks=detection_dict_copy[class_name]["landmarks"],
        index=predefined_index,
        fill_in_value=preds,
    )

    return refined_detection_np, detection_dict_copy
