import pkg_resources
import os

package_name = __name__.split(".")[0]
garment_classes = {
    "long sleeve dress": {
        "num_predefined_points": 37,
        "index_range": (219, 256),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/long sleeve dress.json")
    }, 
    "long sleeve top": {
        "num_predefined_points": 33,
        "index_range": (25, 58),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/long sleeve top.json")
    }, 
    "short sleeve dress": {
        "num_predefined_points": 29,
        "index_range": (190, 219),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/short sleeve dress.json")
    }, 
    "short sleeve top": {
        "num_predefined_points": 25,
        "index_range": (0, 25),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/short sleeve top.json")
    }, 
    "shorts": {
        "num_predefined_points": 10,
        "index_range": (158, 168),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/shorts.json")
    }, 
    "skirt": {
        "num_predefined_points": 8,
        "index_range": (182, 190),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/skirt.json")
    }, 
    "trousers": {
        "num_predefined_points": 14,
        "index_range": (168, 182),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/trousers.json")
    }, 
    "vest": {
        "num_predefined_points": 15,
        "index_range": (128, 143),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/vest.json")
    }, 
    "vest dress": {
        "num_predefined_points": 19,
        "index_range": (256, 275),
        "instruction": pkg_resources.resource_filename(package_name, "instruction/vest dress.json")
    }
}
