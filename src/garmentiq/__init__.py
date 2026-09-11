# garmentiq/__init__.py
"""Automated garment measurement from images.

GarmentIQ turns a photograph of a garment into a set of tape-measure-style
measurements. The pipeline runs in four stages: **classification** identifies the
garment type, **segmentation** separates it from the background, **landmark**
detection locates its key points, and the measurements are computed between those
points using a per-garment instruction schema.

Each stage is a module that can be used on its own, or the whole sequence can be run
by the `tailor` agent. Two optional modules refine the result: `matting` turns a hard
mask into a soft alpha matte, and `grounding` turns a text phrase into boxes so SAM 1
and SAM 2 can be prompted with natural language.

Every model-backed module follows the same two-step shape::

    model = giq.<module>.load_model(...)
    result = giq.<module>.<run>(model=model, image_path=..., device="cpu")

Hardware acceleration is opt-in everywhere: `device` defaults to `"cpu"` and is never
auto-detected. Pass `"cuda"` or `"mps"` to use an accelerator.
"""
__version__ = "0.0.4.11"

from .tailor import tailor
from . import utils
from . import classification
from . import segmentation
from . import landmark
from . import matting


def __getattr__(name):
    """Expose optional subpackages without importing their heavy dependencies eagerly."""
    if name == "grounding":
        import importlib

        # import_module avoids re-entering this __getattr__ (which `from . import
        # grounding` would do, recursing infinitely); caching stops repeat lookups.
        module = importlib.import_module(f"{__name__}.grounding")
        globals()["grounding"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
