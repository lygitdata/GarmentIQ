import json
import numpy as np
from typing import Optional, Dict, Tuple, Set

def _get_coordinates_from_json(json_path: str) -> Optional[Dict[int, Tuple[float, float]]]:
    """Loads keypoint coordinates from a JSON file into a dictionary."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'coordinates' not in data or not isinstance(data['coordinates'], list):
            print(f"Error: JSON file '{json_path}' missing 'coordinates' list.")
            return None

        coords_dict = {
            kp['keypoint_id']: (kp['x'], kp['y'])
            for kp in data['coordinates']
            if 'keypoint_id' in kp and 'x' in kp and 'y' in kp
        }
        return coords_dict

    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'")
        return None
    except Exception as e:
        print(f"Error loading or parsing JSON '{json_path}': {str(e)}")
        return None

def _get_required_coordinates(
    required_ids: Set[int],
    all_coords: Dict[int, Tuple[float, float]]
    ) -> Optional[Dict[int, Tuple[float, float]]]:
    """Extracts specific coordinates and checks if all required IDs are present."""
    if not required_ids.issubset(all_coords.keys()):
        missing_ids = required_ids - all_coords.keys()
        print(f"Error: Missing coordinates for keypoint IDs: {missing_ids}")
        return None

    return {id_: all_coords[id_] for id_ in required_ids}
