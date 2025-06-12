derivation_dict = {
    "derive_keypoint_coord": {
        "p1_id": None,
        "p2_id": None,
        "p3_id": None,
        "p4_id": None,
        "p5_id": None,
        "direction": ["parallel", "perpendicular"],
    }
}
"""dict: A dictionary defining the schemas for various landmark derivation functions.

Each key in this dictionary represents a derivation function, and its value is
another dictionary specifying the expected parameters for that function.
`None` as a parameter value indicates that the parameter must be provided
when calling the function, and it is typically an integer ID corresponding to a
predefined landmark.

Example structure for "derive_keypoint_coord":
    "derive_keypoint_coord": {
        "p1_id": None,  # Required: Integer ID of the first point.
        "p2_id": None,  # Required: Integer ID of the second point.
        "p3_id": None,  # Required: Integer ID of the third point.
        "p4_id": None,  # Required: Integer ID of the fourth point.
        "p5_id": None,  # Required: Integer ID of the fifth point.
        "direction": ["parallel", "perpendicular"], # Required: How to derive line 1's direction.
    }
"""