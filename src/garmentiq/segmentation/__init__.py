# garmentiq/segmentation/__init__.py
"""Garment segmentation.

Separates the garment from its background and produces the mask the later stages rely
on. Landmark refinement and derivation both need this mask, and matting uses it as
guidance.

Four backends are supported through one interface: BiRefNet, which needs no prompt at
all, and SAM 1, SAM 2, and SAM 3, which are guided by a unified `prompt` dictionary
accepting points, labels, boxes, and text.
"""
from .load_model import load_model
from .extract import extract
from .plot import plot
from .change_background_color import change_background_color
from .process_and_save_images import process_and_save_images
