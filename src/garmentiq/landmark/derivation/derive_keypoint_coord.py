from typing import Tuple, Optional
import numpy as np
from .line_intersect import _find_line_line_intersection
from .mask_intersect import _get_mask_boundary, _find_line_mask_intersections
from .utils import _calculate_line1_vector, _find_closest_point


def derive_keypoint_coord(
    p1_id: int,
    p2_id: int,
    p3_id: int,
    p4_id: int,
    p5_id: int,
    direction: str,
    landmark_coords: np.array,
    np_mask: np.array,
    line_length_factor: float = 10000.0,
) -> Optional[Tuple[float, float]]:
    p1_coord = landmark_coords[:, p1_id - 1, :].reshape(2)
    p2_coord = landmark_coords[:, p2_id - 1, :].reshape(2)
    p3_coord = landmark_coords[:, p3_id - 1, :].reshape(2)
    p4_coord = landmark_coords[:, p4_id - 1, :].reshape(2)
    p5_coord = landmark_coords[:, p5_id - 1, :].reshape(2)

    # 3. Define Line 1 Vector (v1)
    v1 = _calculate_line1_vector(p2_coord, p3_coord, direction)
    if v1 is None:
        return None  # Error or zero vector detected

    # 4. Define Line 2 Vector (v2)
    v2 = (p5_coord[0] - p4_coord[0], p5_coord[1] - p4_coord[1])
    if np.isclose(v2[0], 0) and np.isclose(v2[1], 0):
        print(
            f"Warning: Direction vector for Line 2 is zero (p4_id={p4_id} and p5_id={p5_id} likely coincide)."
        )
        return None  # Cannot define Line 2

    # 5. Calculate Line-Line Intersection
    line_intersection_point = _find_line_line_intersection(p1_coord, v1, p4_coord, v2)
    if line_intersection_point is None:
        print(
            "Info: Line 1 and Line 2 are parallel or collinear. No unique intersection."
        )
        return None

    # 6. Load Mask Boundary
    mask_boundary_geom = _get_mask_boundary(np_mask)
    if mask_boundary_geom is None or mask_boundary_geom.is_empty:
        print(
            "Warning: No valid mask boundary found or mask is empty. Returning line-line intersection."
        )
        return line_intersection_point

    # 7. Find Intersection(s) between Line 1 and Mask Boundary
    mask_intersection_points = _find_line_mask_intersections(
        p1_coord, v1, mask_boundary_geom, line_length_factor
    )

    # 8. Determine Final Point
    if mask_intersection_points is None:
        # An error occurred during intersection calculation
        print(
            "Warning: Error finding mask intersections. Returning line-line intersection as fallback."
        )
        return line_intersection_point
    elif not mask_intersection_points:
        # No intersection found between line and mask boundary
        # print("Info: Line 1 does not intersect the mask boundary. Returning line-line intersection.")
        return line_intersection_point
    else:
        # Found intersection(s), find the one closest to the line-line intersection
        closest_mask_point = _find_closest_point(
            mask_intersection_points, line_intersection_point
        )
        # _find_closest_point should always return a point if the list is not empty
        return closest_mask_point
