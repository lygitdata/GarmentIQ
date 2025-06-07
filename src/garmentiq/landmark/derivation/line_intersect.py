import numpy as np
from typing import Tuple, Optional

def _find_line_line_intersection(p1: Tuple[float, float],
                                 v1: Tuple[float, float],
                                 p2: Tuple[float, float],
                                 v2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Calculates the intersection point of two lines defined by point and vector."""
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
