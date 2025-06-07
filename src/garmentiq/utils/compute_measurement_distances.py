import math

def compute_measurement_distances(garment_dict):
    def euclidean(p1, p2):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

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
            garment_dict[garment_name]["measurements"][measurement_name]["distance"] = round(distance, 16)

    return distances, garment_dict
