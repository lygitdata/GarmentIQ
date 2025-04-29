import math
import json

def calculate_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Args:
        p1 (tuple): Coordinates of the first point (x1, y1).
        p2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def load_json(file_path: str) -> dict:
    """
    Load a JSON file into a Python dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
