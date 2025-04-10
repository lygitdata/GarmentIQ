import os

def check_unzipped_dir(folder):
    image_dir = os.path.join(folder, "images")
    metadata_path = os.path.join(folder, "metadata.csv")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing 'images' folder in: {folder}")

    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg files found in: {image_dir}")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing 'metadata.csv' in: {folder}")