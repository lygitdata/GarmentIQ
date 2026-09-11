# garmentiq/utils/__init__.py
"""Shared helpers used across the GarmentIQ modules.

Small utilities that several modules depend on: archive handling and dataset
validation, device resolution and accelerator cache management, checkpoint loading
with a mismatch guard, measurement distance computation, and JSON export.
"""
from .unzip import unzip
from .check_unzipped_dir import check_unzipped_dir
from .check_filenames_metadata import check_filenames_metadata
from .validate_garment_class_dict import validate_garment_class_dict
from .compute_measurement_distances import compute_measurement_distances
from .export_dict_to_json import export_dict_to_json
from .clean_detection_dict import clean_detection_dict
from .device import resolve_device, empty_cache, inputs_to_device
from .checkpoint import load_state_dict_checked
