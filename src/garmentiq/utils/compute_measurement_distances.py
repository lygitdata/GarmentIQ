import math


def compute_measurement_distances(garment_dict):
    """
    Computes Euclidean distances between specified landmark points for each garment
    in a dictionary and updates the dictionary with these distances.

    This function iterates through the `garment_dict`, accesses the 'landmarks'
    and 'measurements' for each garment, calculates the Euclidean distance
    between the 'start' and 'end' landmarks defined for each measurement,
    and stores the rounded distance in both a new dictionary and within the
    original `garment_dict`.

    Args:
        garment_dict (dict): A dictionary where keys are garment names and values
                             contain 'landmarks' (with 'x' and 'y' coordinates)
                             and 'measurements' (with 'start' and 'end' landmark IDs).

    Returns:
        tuple:
            - distances (dict): A dictionary mapping measurement names to their computed distances.
            - garment_dict (dict): The original `garment_dict` updated with a 'distance' field
                                   for each measurement.
    """

    def euclidean(p1, p2):
        return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)

    distances = {}
    for garment_name, garment_data in garment_dict.items():
        landmarks = garment_data["landmarks"]
        for measurement_name, measurement_data in garment_data["measurements"].items():
            start_id = measurement_data["landmarks"]["start"]
            end_id = measurement_data["landmarks"]["end"]

            point1 = landmarks[start_id]
            point2 = landmarks[end_id]
            distance = euclidean(point1, point2)

            # Store the distance in the return dict
            distances[measurement_name] = round(distance, 16)

            # Update the original dictionary
            garment_dict[garment_name]["measurements"][measurement_name][
                "distance"
            ] = round(distance, 16)

    return distances, garment_dict
