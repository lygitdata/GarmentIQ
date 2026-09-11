# garmentiq/matting/__init__.py
"""Alpha matting.

Segmentation produces a hard yes/no mask; matting refines it into a continuous alpha
matte so edges and semi-transparent detail composite naturally. Two approaches are
supported: ViTMatte, which needs a trimap, and Matting Anything, which refines a SAM
mask directly.
"""
from .trimap import generate_trimap
from .matte import matte, composite
from .load_model import load_model

__all__ = [
    "generate_trimap",
    "matte",
    "composite",
    "load_model",
]
