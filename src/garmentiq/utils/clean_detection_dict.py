import os
from garmentiq.utils import compute_measurement_distances


def clean_detection_dict(class_name: str, image_name: str, detection_dict: dict):
    """
    Cleans and reformats a detection dictionary, computes measurement distances,
    and nests the cleaned data under the image name, with class name stored inside.

    Args:
        class_name (str): The name of the garment class.
        image_name (str): The original filename of the image.
        detection_dict (dict): The raw detection dictionary containing landmark and measurement data.

    Returns:
        dict: A dictionary structured as {image_name: {..., "class": class_name}}.
    """

    transformed_name = os.path.splitext(image_name)[0]

    # Compute distances and get a fresh copy of detection_dict
    _, clean_dict = compute_measurement_distances(detection_dict)

    # Safely extract and clean the content under the class_name
    original_data = clean_dict.get(class_name, {})

    # Clean landmarks
    if "landmarks" in original_data:
        for lm_id in list(original_data["landmarks"].keys()):
            lm = original_data["landmarks"][lm_id]
            original_data["landmarks"][lm_id] = {
                k: lm[k] for k in ("x", "y", "conf") if k in lm
            }

    # Clean measurements
    if "measurements" in original_data:
        for m_id in list(original_data["measurements"].keys()):
            m = original_data["measurements"][m_id]
            original_data["measurements"][m_id] = {
                k: m[k] for k in ("landmarks", "distance") if k in m
            }

    # Insert the class name as metadata
    original_data["class"] = class_name

    # Return dict keyed by image name
    final_dict = {image_name: original_data}
    return final_dict
