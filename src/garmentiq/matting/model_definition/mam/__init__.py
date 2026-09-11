# garmentiq/matting/model_definition/mam/__init__.py
"""The Matting Anything model.

Matting Anything refines a SAM mask into an alpha matte. It consumes SAM's own image
embeddings rather than just its mask, so it is structurally tied to SAM and cannot run
from another segmentation model. `load_mam` reads only the matting decoder from the
checkpoint and pairs it with a SAM model you already loaded.
"""
from .m2m import SAM_Decoder_Deep, sam_decoder_deep
from .mam import MattingAnything, load_mam, fuse_alpha

__all__ = [
    "SAM_Decoder_Deep",
    "sam_decoder_deep",
    "MattingAnything",
    "load_mam",
    "fuse_alpha",
]
