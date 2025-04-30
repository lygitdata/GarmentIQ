from typing import Tuple, Optional
import numpy as np

from .load_and_extract_coords import _get_coordinates_from_json, _get_required_coordinates
from .line_intersect import _find_line_line_intersection
from .mask_intersect import _get_mask_boundary, _find_line_mask_intersections
from .utils import _calculate_line1_vector, _find_closest_point

def derive_keypoint_coord(
    p1_id: int,
    p2_id: int,
    p3_id: int,
    p4_id: int,
    p5_id: int,
    json_path: str,
    mask_path: str,
    direction: str,
    line_length_factor: float = 10000.0
    ) -> Optional[Tuple[float, float]]:
    
    """
    Calculates a target coordinate based on keypoint geometry and mask intersection.

    The target coordinate is found by:
    1. Defining Line 1 passing through p1. Its direction is either parallel or
       perpendicular to the line segment p2-p3.
    2. Defining Line 2 passing through p4 and p5.
    3. Calculating the intersection (ix, iy) of Line 1 and Line 2.
    4. Finding the intersection(s) of Line 1 with the boundary of a segmentation mask.
    5. If Line 1 intersects the mask boundary, the function returns the mask
       intersection point closest to (ix, iy).
    6. If Line 1 does not intersect the mask boundary, the function returns (ix, iy).

    Args:
        p1_id: The index (keypoint_id) of the starting point for Line 1.
        p2_id: The index of the first point defining the reference direction.
        p3_id: The index of the second point defining the reference direction.
        p4_id: The index of the first point defining Line 2.
        p5_id: The index of the second point defining Line 2.
        json_path: Path to the JSON file containing keypoint coordinates.
                   Expected format: {"coordinates": [{"keypoint_id": id, "x": x, "y": y}, ...]}
        mask_path: Path to the segmentation mask image file (grayscale).
        direction: "parallel" or "perpendicular". Determines Line 1's direction
                   relative to the p2-p3 vector.
        line_length_factor: A large multiplier to create effectively infinite lines
                            for Shapely intersection tests.

    Returns:
        A tuple (x, y) representing the calculated coordinate, or None if an error
        occurs (e.g., file not found, keypoint missing, lines parallel, no mask boundary).
    """

    # 1. Load All Coordinates
    all_coords = _get_coordinates_from_json(json_path)
    if all_coords is None:
        return None

    # 2. Get Specific Coordinates Needed
    required_ids = {p1_id, p2_id, p3_id, p4_id, p5_id}
    key_coords = _get_required_coordinates(required_ids, all_coords)
    if key_coords is None:
        return None

    p1_coord = key_coords[p1_id]
    p2_coord = key_coords[p2_id]
    p3_coord = key_coords[p3_id]
    p4_coord = key_coords[p4_id]
    p5_coord = key_coords[p5_id]

    # 3. Define Line 1 Vector (v1)
    v1 = _calculate_line1_vector(p2_coord, p3_coord, direction)
    if v1 is None:
        return None # Error or zero vector detected

    # 4. Define Line 2 Vector (v2)
    v2 = (p5_coord[0] - p4_coord[0], p5_coord[1] - p4_coord[1])
    if np.isclose(v2[0], 0) and np.isclose(v2[1], 0):
         print(f"Warning: Direction vector for Line 2 is zero (p4_id={p4_id} and p5_id={p5_id} likely coincide).")
         return None # Cannot define Line 2

    # 5. Calculate Line-Line Intersection
    line_intersection_point = _find_line_line_intersection(p1_coord, v1, p4_coord, v2)
    if line_intersection_point is None:
        print("Info: Line 1 and Line 2 are parallel or collinear. No unique intersection.")
        return None

    # 6. Load Mask Boundary
    mask_boundary_geom = _get_mask_boundary(mask_path)
    if mask_boundary_geom is None or mask_boundary_geom.is_empty:
        print("Warning: No valid mask boundary found or mask is empty. Returning line-line intersection.")
        return line_intersection_point

    # 7. Find Intersection(s) between Line 1 and Mask Boundary
    mask_intersection_points = _find_line_mask_intersections(
        p1_coord, v1, mask_boundary_geom, line_length_factor
    )

    # 8. Determine Final Point
    if mask_intersection_points is None:
        # An error occurred during intersection calculation
        print("Warning: Error finding mask intersections. Returning line-line intersection as fallback.")
        return line_intersection_point
    elif not mask_intersection_points:
        # No intersection found between line and mask boundary
        # print("Info: Line 1 does not intersect the mask boundary. Returning line-line intersection.")
        return line_intersection_point
    else:
        # Found intersection(s), find the one closest to the line-line intersection
        closest_mask_point = _find_closest_point(mask_intersection_points, line_intersection_point)
        # _find_closest_point should always return a point if the list is not empty
        return closest_mask_point
