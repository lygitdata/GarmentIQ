import numpy as np
from typing import Tuple, Optional


def _find_line_line_intersection(
    p1: Tuple[float, float],
    v1: Tuple[float, float],
    p2: Tuple[float, float],
    v2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Calculates the intersection point of two lines, each defined by a point and a direction vector.

    Args:
        p1 (Tuple[float, float]): A point (x1, y1) on the first line.
        v1 (Tuple[float, float]): The direction vector (dx1, dy1) of the first line.
        p2 (Tuple[float, float]): A point (x2, y2) on the second line.
        v2 (Tuple[float, float]): The direction vector (dx2, dy2) of the second line.

    Returns:
        Optional[Tuple[float, float]]: The (x, y) coordinates of the intersection point,
                                       or None if the lines are parallel or collinear (no unique intersection).
    """
    x1, y1 = p1
    dx1, dy1 = v1
    x2, y2 = p2
    dx2, dy2 = v2

    denominator = dx2 * dy1 - dy2 * dx1
    if np.isclose(denominator, 0):  # Lines are parallel or collinear
        return None

    qp_x = x1 - x2
    qp_y = y1 - y2
    t = (qp_x * dy1 - qp_y * dx1) / denominator
    ix = x2 + t * dx2
    iy = y2 + t * dy2

    return ix, iy
