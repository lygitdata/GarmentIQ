import numpy as np
import cv2
from typing import Optional, List, Tuple, Any
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString
from shapely.ops import unary_union


def _get_mask_boundary(mask: np.ndarray):
    """
    Processes a binary or grayscale mask array and returns the primary boundary as Shapely geometry.

    This function finds contours in the mask, converts them into Shapely Polygon boundaries
    or LineStrings, and then unions them into a single (potentially Multi-) geometry.

    Args:
        mask (np.ndarray): The input binary or grayscale mask array.

    Returns:
        Optional[Any]: A Shapely geometry (Polygon.boundary, LineString, MultiPoint, or MultiLineString)
                       representing the mask boundary, or None if the mask is invalid or no contours are found.
    """
    try:
        if mask is None or not isinstance(mask, np.ndarray):
            print("Error: Provided mask is not a valid NumPy array.")
            return None

        # Ensure binary mask (values 0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("Warning: No contours found in the mask array.")
            return None

        geometries = []
        for contour in contours:
            points = [tuple(p[0]) for p in contour]
            if len(points) >= 3:
                geometries.append(Polygon(points).boundary)
            elif len(points) == 2:
                geometries.append(LineString(points))

        if not geometries:
            print("Warning: No valid boundary geometries found in the mask.")
            return None

        return unary_union(geometries)

    except Exception as e:
        print(f"Error processing mask array: {str(e)}")
        return None


def _find_line_mask_intersections(
    line_point: Tuple[float, float],
    line_vector: Tuple[float, float],
    mask_boundary: Any,  # Shapely Geometry
    line_length_factor: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    Finds intersection points between a line (defined by a point and vector) and a Shapely mask boundary.

    Constructs a long Shapely LineString from the input line definition and
    computes its intersection with the provided `mask_boundary` geometry.

    Args:
        line_point (Tuple[float, float]): A point (x, y) on the line.
        line_vector (Tuple[float, float]): The direction vector (dx, dy) of the line.
        mask_boundary (Any): A Shapely Geometry object representing the mask boundary (e.g., from `_get_mask_boundary`).
        line_length_factor (float): A factor to extend the line segment for intersection testing,
                                    ensuring it crosses the entire mask if needed.

    Returns:
        Optional[List[Tuple[float, float]]]: A list of (x, y) tuples for all unique intersection points,
                                            an empty list if no intersections, or None if an error occurs.
    """
    try:
        # Create a long Shapely line representing the mathematical line
        norm_v = np.linalg.norm(line_vector)
        if np.isclose(norm_v, 0):
            print("Error: Line vector is zero during Shapely line creation.")
            return None  # Cannot create line

        unit_v = (line_vector[0] / norm_v, line_vector[1] / norm_v)

        pt_a = (
            line_point[0] - line_length_factor * unit_v[0],
            line_point[1] - line_length_factor * unit_v[1],
        )
        pt_b = (
            line_point[0] + line_length_factor * unit_v[0],
            line_point[1] + line_length_factor * unit_v[1],
        )
        shapely_line = LineString([pt_a, pt_b])

        # Calculate intersection
        intersection = mask_boundary.intersection(shapely_line)

        # Process intersection results
        if intersection.is_empty:
            return []  # Return empty list for no intersection

        intersection_points = []
        geoms_to_process = []

        if isinstance(intersection, Point):
            geoms_to_process.append(intersection)
        elif isinstance(intersection, (MultiPoint, LineString, MultiLineString)):
            # Use .geoms for MultiPoint/MultiLineString, .coords for LineString
            if hasattr(intersection, "geoms"):
                geoms_to_process.extend(list(intersection.geoms))
            elif hasattr(intersection, "coords"):  # LineString (boundary coincidence)
                # Extract points from the LineString coords
                coords = list(intersection.coords)
                for coord in coords:
                    intersection_points.append(coord)  # Add individual vertices
                # Avoid processing LineString further as Point below

        # Extract coordinates from Point geometries
        for geom in geoms_to_process:
            if isinstance(geom, Point):
                intersection_points.append((geom.x, geom.y))

        # Remove duplicates if necessary (e.g., from LineString endpoints)
        # Using list(dict.fromkeys(intersection_points)) preserves order unlike set
        unique_intersection_points = list(dict.fromkeys(intersection_points))

        return unique_intersection_points

    except Exception as e:
        print(f"Error during Shapely intersection calculation: {str(e)}")
        return None  # Indicate an error occurred
