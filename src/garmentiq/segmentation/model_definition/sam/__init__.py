# garmentiq/segmentation/model_definition/sam/__init__.py
"""The Segment Anything model family.

Covers SAM 1, SAM 2.1, and SAM 3 behind one registry, together with their bundled
configurations and processors. The model classes are imported lazily, so GarmentIQ
still imports on a `transformers` release that predates a given generation, and asking
for a model your release does not carry raises a clear `ImportError`.

The model classes themselves are re-exported from `transformers` and documented
upstream, so they are not repeated here. They can be imported from this module
directly:

    from garmentiq.segmentation.model_definition.sam import (
        SamModel, Sam2Model, Sam3Model,
    )

The full set is `SamModel`, `SamConfig`, `SamProcessor`, `Sam2Model`, `Sam2Config`,
`Sam2Processor`, `Sam2ImageProcessor`, `Sam3Model`, `Sam3Config`, `Sam3Processor`,
`Sam3ImageProcessor`, `Sam3TrackerModel`, `Sam3TrackerConfig`, and
`Sam3TrackerProcessor`.
"""
from .sam import (
    load_sam_config,
    load_sam_processor,
    load_sam3_tracker,
    load_sam3_tracker_config,
    load_sam3_tracker_processor,
    sam_family,
    sam_capabilities,
    sam_model_class,
    SAM1_MODELS,
    SAM2_MODELS,
    SAM3_MODELS,
    VALID_SAM_MODELS,
    ALL_SAM_MODELS,
    SAM_CAPABILITIES,
    SAM_PROMPT_DEPTHS,
)

# Model classes come from transformers and are resolved on first access, so that
# importing GarmentIQ does not require a transformers release new enough to
# provide every SAM generation.
_LAZY = {
    "Sam3TrackerModel",
    "Sam3TrackerConfig",
    "Sam3TrackerProcessor",
    "Sam3ImageProcessor",
    "SamModel",
    "SamConfig",
    "SamProcessor",
    "Sam2Model",
    "Sam2Config",
    "Sam2Processor",
    "Sam2ImageProcessor",
    "Sam3Model",
    "Sam3Config",
    "Sam3Processor",
}


def __getattr__(name):
    """Defer transformers imports until a SAM class is actually requested."""
    if name in _LAZY:
        from . import sam

        return getattr(sam, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_sam_config",
    "load_sam_processor",
    "load_sam3_tracker",
    "load_sam3_tracker_config",
    "load_sam3_tracker_processor",
    "sam_family",
    "sam_capabilities",
    "sam_model_class",
    "SAM1_MODELS",
    "SAM2_MODELS",
    "SAM3_MODELS",
    "VALID_SAM_MODELS",
    "ALL_SAM_MODELS",
    "SAM_CAPABILITIES",
    "SAM_PROMPT_DEPTHS",
]


def __dir__():
    """List the lazily resolved model classes alongside the eager exports."""
    return sorted(set(__all__) | _LAZY)
