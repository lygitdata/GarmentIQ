import numpy as np
import cv2
from typing import Optional, List, Tuple, Any
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString
from shapely.ops import unary_union

def _get_mask_boundary(mask_path: str) -> Optional[Any]: # Returns Shapely Geometry
    """Loads a mask, finds contours, and returns the primary boundary as Shapely geometry."""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not read mask file '{mask_path}'")
            return None

        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found in mask '{mask_path}'")
            return None

        geometries = []
        for contour in contours:
            points = [tuple(p[0]) for p in contour]
            if len(points) >= 3:
                geometries.append(Polygon(points).boundary)
            elif len(points) == 2:
                geometries.append(LineString(points))

        if not geometries:
             print(f"Warning: No valid boundary geometries found from contours in '{mask_path}'")
             return None

        primary_boundary = unary_union(geometries)
        return primary_boundary

    except FileNotFoundError:
        print(f"Error: Mask file not found at '{mask_path}'")
        return None
    except Exception as e:
        print(f"Error processing mask '{mask_path}': {str(e)}")
        return None

def _find_line_mask_intersections(
    line_point: Tuple[float, float],
    line_vector: Tuple[float, float],
    mask_boundary: Any, # Shapely Geometry
    line_length_factor: float
    ) -> Optional[List[Tuple[float, float]]]:
    """Finds intersection points between a line and the mask boundary using Shapely."""
    try:
        # Create a long Shapely line representing the mathematical line
        norm_v = np.linalg.norm(line_vector)
        if np.isclose(norm_v, 0):
             print("Error: Line vector is zero during Shapely line creation.")
             return None # Cannot create line

        unit_v = (line_vector[0] / norm_v, line_vector[1] / norm_v)

        pt_a = (line_point[0] - line_length_factor * unit_v[0],
                line_point[1] - line_length_factor * unit_v[1])
        pt_b = (line_point[0] + line_length_factor * unit_v[0],
                line_point[1] + line_length_factor * unit_v[1])
        shapely_line = LineString([pt_a, pt_b])

        # Calculate intersection
        intersection = mask_boundary.intersection(shapely_line)

        # Process intersection results
        if intersection.is_empty:
            return [] # Return empty list for no intersection

        intersection_points = []
        geoms_to_process = []

        if isinstance(intersection, Point):
            geoms_to_process.append(intersection)
        elif isinstance(intersection, (MultiPoint, LineString, MultiLineString)):
             # Use .geoms for MultiPoint/MultiLineString, .coords for LineString
             if hasattr(intersection, 'geoms'):
                 geoms_to_process.extend(list(intersection.geoms))
             elif hasattr(intersection, 'coords'): # LineString (boundary coincidence)
                 # Extract points from the LineString coords
                 coords = list(intersection.coords)
                 for coord in coords:
                     intersection_points.append(coord) # Add individual vertices
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
        return None # Indicate an error occurred
