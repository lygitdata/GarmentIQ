# garmentiq/landmark/__init__.py
"""Garment landmark detection, refinement, and derivation.

Landmarks are the key points a measurement runs between, such as shoulders, sleeve
ends, and hems. This module produces them in up to three steps:

- `detect` predicts the landmarks the pose model was trained on.
- `refine` snaps those predictions onto the true garment boundary using a
  segmentation mask, correcting points that sit slightly off the edge.
- `derive` computes additional landmarks that the model does not predict at all,
  using geometric rules evaluated against the mask.

Refinement and derivation both require a segmentation mask.
"""
from .detect import detect
from .detection import *
from .derive import derive
from .derivation import *
from .refine import refine
from .refinement import *
from .plot import plot
from .utils import (
	find_instruction_landmark_index,
	fill_instruction_landmark_coordinate,
)
