import numpy as np
from typing import Tuple, Optional, List

def _calculate_line1_vector(
    p2_coord: Tuple[float, float],
    p3_coord: Tuple[float, float],
    direction: str
    ) -> Optional[Tuple[float, float]]:
    """Calculates the direction vector for Line 1 based on p2, p3, and direction."""
    ref_dx = p3_coord[0] - p2_coord[0]
    ref_dy = p3_coord[1] - p2_coord[1]

    if direction == "parallel":
        v1 = (ref_dx, ref_dy)
    elif direction == "perpendicular":
        v1 = (-ref_dy, ref_dx)
    else:
        print(f"Error: Invalid direction '{direction}'. Use 'parallel' or 'perpendicular'.")
        return None

    # Check for zero vector
    if np.isclose(v1[0], 0) and np.isclose(v1[1], 0):
         print(f"Warning: Direction vector for Line 1 is zero (p2 and p3 likely coincide).")
         # Decide if this should be a fatal error or handled downstream
         # Returning None signals an issue.
         return None

    return v1

def _find_closest_point(
    points_list: List[Tuple[float, float]],
    target_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
    """Finds the point in points_list closest to target_point."""
    if not points_list:
        return None

    points_np = np.array(points_list)
    target_np = np.array(target_point)

    distances = np.linalg.norm(points_np - target_np, axis=1)
    closest_index = np.argmin(distances)

    return tuple(points_np[closest_index])
    
def parse_derivation_args(deriv_dict, json_path, mask_path):
    # args = {}
    # for key, val in deriv_dict.items():
    #     if key.startswith("input"):
    #         exec(f"{val}", {}, args)  # Parses 'p1_id=2' → args['p1_id'] = 2
    # # Add fixed inputs
    # args["json_path"] = json_path
    # args["mask_path"] = mask_path
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
