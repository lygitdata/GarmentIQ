# garmentiq/landmark/derivation/__init__.py
"""Derivation of landmarks the detection model does not predict.

Some measurement points are defined geometrically rather than learned, and are marked
`predefined: False` in a garment class definition. They are computed by intersecting a
constructed line with the segmentation mask, following the rules in a derivation
dictionary.
"""
from .derive_keypoint_coord import derive_keypoint_coord
from .prepare_args import prepare_args
from .process import process