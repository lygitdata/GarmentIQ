import numpy as np
from typing import Tuple, Optional, List


def _calculate_line1_vector(
    p2_coord: Tuple[float, float], p3_coord: Tuple[float, float], direction: str
) -> Optional[Tuple[float, float]]:
    """
    Calculates the direction vector for Line 1 based on p2, p3, and direction.

    Args:
        p2_coord (Tuple[float, float]): The (x, y) coordinates of the second point.
        p3_coord (Tuple[float, float]): The (x, y) coordinates of the third point.
        direction (str): The desired direction of Line 1 relative to the vector
            from `p2_coord` to `p3_coord`. Must be "parallel" or "perpendicular".

    Returns:
        Optional[Tuple[float, float]]: The calculated direction vector (dx, dy)
            as a tuple of floats, or `None` if the direction is invalid or the
            vector (p3-p2) is a zero vector.
    """
    ref_dx = p3_coord[0] - p2_coord[0]
    ref_dy = p3_coord[1] - p2_coord[1]

    if direction == "parallel":
        v1 = (ref_dx, ref_dy)
    elif direction == "perpendicular":
        v1 = (-ref_dy, ref_dx)
    else:
        print(
            f"Error: Invalid direction '{direction}'. Use 'parallel' or 'perpendicular'."
        )
        return None

    # Check for zero vector
    if np.isclose(v1[0], 0) and np.isclose(v1[1], 0):
        print(
            f"Warning: Direction vector for Line 1 is zero (p2 and p3 likely coincide)."
        )
        # Decide if this should be a fatal error or handled downstream
        # Returning None signals an issue.
        return None

    return v1


def _find_closest_point(
    points_list: List[Tuple[float, float]], target_point: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Finds the point in `points_list` that is closest (Euclidean distance) to `target_point`.

    Args:
        points_list (List[Tuple[float, float]]): A list of 2D points (x, y) to search within.
        target_point (Tuple[float, float]): The reference point (x, y) to find the closest point to.

    Returns:
        Optional[Tuple[float, float]]: The (x, y) coordinates of the closest point from
            `points_list` as a tuple of floats, or `None` if `points_list` is empty.
    """
    if not points_list:
        return None

    points_np = np.array(points_list)
    target_np = np.array(target_point)

    distances = np.linalg.norm(points_np - target_np, axis=1)
    closest_index = np.argmin(distances)

    return tuple(points_np[closest_index])


def parse_derivation_args(deriv_dict, json_path, mask_path):
    """
    Parses a derivation dictionary to extract arguments for a derivation function.

    This function is a helper for preparing arguments required by specific derivation
    functions (e.g., `derive_keypoint_coord`). It extracts parameters and adds fixed
    inputs like `json_path` and `mask_path`.

    Args:
        deriv_dict (dict): A dictionary containing derivation parameters for a specific landmark.
        json_path (str): Path to the JSON file related to the image.
        mask_path (str): Path to the mask file related to the image.

    Returns:
        dict: A dictionary of parsed arguments ready to be passed to a derivation function.
    """
    args = {}
    for k, v in deriv_dict.items():
        if k == "function":
            continue
        # p*_id should be ints, everything else leave as‐is
        if k.endswith("_id"):
            try:
                args[k] = int(v)
            except ValueError:
                # in case someone uses numbers not strictly digits
                args[k] = int(float(v))
        else:
            args[k] = v
    args["json_path"] = json_path
    args["mask_path"] = mask_path
    return args
    return args
