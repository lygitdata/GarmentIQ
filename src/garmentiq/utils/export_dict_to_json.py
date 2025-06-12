import json


def export_dict_to_json(data: dict, filename: str):
    """
    Export a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to export.
        filename (str): The path to the JSON file (e.g., "output.json").
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to export dictionary: {e}")
