"""The predefined garment classes and their measurement instructions.

`garment_classes` maps each supported garment to the number of landmarks the detection
model predicts for it, the slice of the model output those landmarks occupy, and the
path to the JSON instruction defining its measurements. Instruction files are bundled
with the package and resolved to absolute paths at import time.

Pass a modified copy as `class_dict` to use your own measurement instructions.
"""
from importlib.resources import files

package_name = __name__.split(".")[0]


def _instruction_path(filename: str) -> str:
    """Resolve a bundled instruction JSON to an absolute filesystem path."""
    return str(files(package_name) / "instruction" / filename)


garment_classes = {
    "long sleeve dress": {
        "num_predefined_points": 37,
        "index_range": (219, 256),
        "instruction": _instruction_path("long sleeve dress.json")
    }, 
    "long sleeve top": {
        "num_predefined_points": 33,
        "index_range": (25, 58),
        "instruction": _instruction_path("long sleeve top.json")
    }, 
    "short sleeve dress": {
        "num_predefined_points": 29,
        "index_range": (190, 219),
        "instruction": _instruction_path("short sleeve dress.json")
    }, 
    "short sleeve top": {
        "num_predefined_points": 25,
        "index_range": (0, 25),
        "instruction": _instruction_path("short sleeve top.json")
    }, 
    "shorts": {
        "num_predefined_points": 10,
        "index_range": (158, 168),
        "instruction": _instruction_path("shorts.json")
    }, 
    "skirt": {
        "num_predefined_points": 8,
        "index_range": (182, 190),
        "instruction": _instruction_path("skirt.json")
    }, 
    "trousers": {
        "num_predefined_points": 14,
        "index_range": (168, 182),
        "instruction": _instruction_path("trousers.json")
    }, 
    "vest": {
        "num_predefined_points": 15,
        "index_range": (128, 143),
        "instruction": _instruction_path("vest.json")
    }, 
    "vest dress": {
        "num_predefined_points": 19,
        "index_range": (256, 275),
        "instruction": _instruction_path("vest dress.json")
    }
}
"""dict: A comprehensive dictionary defining various garment classes and their properties for GarmentIQ.

Each key in this dictionary represents a specific garment type (e.g., "long sleeve dress").
The value associated with each garment type is a dictionary containing:
    - "num_predefined_points" (int): The total number of predefined (detected)
      keypoints for this garment class.
    - "index_range" (tuple): A tuple (start_index, end_index) specifying the
      slice of the model's output that corresponds to the predefined keypoints
      for this garment. These indices are 0-based and exclusive of the end_index.
    - "instruction" (str): The file path to a JSON schema that details the
      specific landmarks (predefined and derivable) and measurements for this
      garment type. These paths are resolved using `importlib.resources.files`
      to ensure they are correctly located within the installed package.

This dictionary serves as a central registry for the GarmentIQ system to
understand and process different types of clothing.
"""
