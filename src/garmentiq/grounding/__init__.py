# garmentiq/grounding/__init__.py
"""Text-to-region grounding.

Grounding models turn a natural-language phrase into bounding boxes. This is what
gives SAM 1 and SAM 2 text-prompted segmentation, since neither contains a text
encoder. The boxes are model-agnostic, so this module is deliberately independent
of the segmentation package.

The model classes are re-exported from `transformers` and documented upstream, so they
are not repeated here. `GroundingDinoForObjectDetection`, `GroundingDinoConfig`, and
`GroundingDinoProcessor` can all be imported from this module directly.
"""
from .grounding_dino import (
    load_grounding_config,
    load_grounding_processor,
    load_grounding_model,
    ground_text_to_boxes,
)

_LAZY = {
    "GroundingDinoForObjectDetection",
    "GroundingDinoConfig",
    "GroundingDinoProcessor",
}


def __getattr__(name):
    """Defer transformers imports until a grounding class is actually requested."""
    if name in _LAZY:
        from . import grounding_dino

        return getattr(grounding_dino, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_grounding_config",
    "load_grounding_processor",
    "load_grounding_model",
    "ground_text_to_boxes",
]


def __dir__():
    """List the lazily resolved model classes alongside the eager exports."""
    return sorted(set(__all__) | _LAZY)
