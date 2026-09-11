# garmentiq/landmark/refinement/__init__.py
"""Refinement of detected landmarks against a segmentation mask.

The pose model predicts landmarks to within a few pixels, which can leave them just
inside or outside the garment edge. Refinement searches a small window around each
point and moves it onto the mask boundary, after blurring the mask to smooth away
jagged edges.
"""
from .refine_landmark_with_blur import refine_landmark_with_blur
