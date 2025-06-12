import copy
from garmentiq.landmark.derivation import (
    prepare_args,
    process,
)


def derive(
    class_name: str, detection_dict: dict, derivation_dict: dict, **extra_args
) -> dict:
    """
    Derives non-predefined landmark coordinates based on predefined landmarks and a mask.

    This function identifies landmarks marked for derivation within the `detection_dict`,
    prepares arguments for the derivation function using `prepare_args`,
    processes the derivation using `process`, and updates the `detection_dict`
    with the newly derived coordinates.

    Args:
        class_name (str): The name of the garment class.
        detection_dict (dict): The dictionary containing detected landmarks, including predefined ones.
        derivation_dict (dict): The dictionary defining derivation rules and available functions.
        **extra_args: Additional keyword arguments required by the derivation functions,
                      such as `landmark_coords` (NumPy array of landmark coordinates)
                      and `np_mask` (NumPy array of the segmentation mask).

    Returns:
        dict: A tuple containing:
            - derived_coords (dict): A dictionary mapping the derived landmark IDs to their new (x, y) coordinates.
            - detection_dict (dict): The original `detection_dict` updated with the derived landmark coordinates.
    """
    non_predefined_landmark = {
        k: detection_dict[class_name]["landmarks"][k]["derivation"]
        for k, v in detection_dict[class_name]["landmarks"].items()
        if v.get("predefined") is False
    }
    derived_coords = {}
    detection_dict_copy = copy.deepcopy(detection_dict)
    for k, v in non_predefined_landmark.items():
        args = prepare_args(non_predefined_landmark[k], derivation_dict, **extra_args)
        derived_coord = tuple(float(x) for x in process(**args))
        derived_coords[k] = derived_coord
        detection_dict_copy[class_name]["landmarks"][k]["x"] = derived_coord[0]
        detection_dict_copy[class_name]["landmarks"][k]["y"] = derived_coord[1]
    return derived_coords, detection_dict_copy
