# garmentiq/classification/__init__.py
"""Garment type classification.

Identifies which category a garment image belongs to, such as a short sleeve top, a
vest dress, or a skirt. This is the first pipeline stage, because the garment type
decides which landmarks and which measurement instructions apply downstream.

Beyond inference, this module covers the full model lifecycle: splitting a dataset,
caching it in memory, training a model from scratch, fine-tuning a pretrained one, and
evaluating the result.
"""
from .train_test_split import train_test_split
from .load_data import load_data
from .load_model import load_model
from .train_pytorch_nn import train_pytorch_nn
from .fine_tune_pytorch_nn import fine_tune_pytorch_nn
from .test_pytorch_nn import test_pytorch_nn
from .predict import predict
from .utils import (
    CachedDataset,
    seed_worker,
    train_epoch,
    validate_epoch,
    save_best_model,
    validate_train_param,
    validate_test_param,
)
from .model_definition import (
    CNN3, 
    CNN4, 
    tinyViT,
)
