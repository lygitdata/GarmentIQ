import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Type
from tqdm.notebook import tqdm
import os
from sklearn.model_selection import StratifiedKFold
from garmentiq.classification.utils import (
    CachedDataset,
    seed_worker,
    train_epoch,
    validate_epoch,
    save_best_model,
    validate_train_param,
    validate_test_param,
)

def fine_tune_pytorch_nn(
    model_class: Type[torch.nn.Module],
    model_args: dict,
    dataset_class: Callable,
    dataset_args: dict,
    param: dict,
):
    """
    Fine-tunes a pretrained PyTorch model using k-fold cross-validation, early stopping, and checkpointing.

    This function loads pretrained weights, optionally freezes specified layers, and trains the model on a new dataset
    while preserving original learned features. It performs stratified k-fold CV, monitors validation loss, and saves
    the best performing model.

    Args:
        model_class (Type[torch.nn.Module]): Class of the PyTorch model (inherits from `torch.nn.Module`).
        model_args (dict): Arguments for model initialization.
        dataset_class (Callable): Callable that returns a Dataset given indices and cached tensors.
        dataset_args (dict): Dict containing:
            - 'metadata_df': DataFrame for stratification
            - 'raw_labels': Labels array for KFold
            - 'cached_images': Tensor of images
            - 'cached_labels': Tensor of labels
        param (dict): Training configuration dict. Must include:
            - 'pretrained_path' (str): Path to pretrained weights (.pt)
            - 'freeze_layers' (bool): Whether to freeze base layers
            - 'optimizer_class', 'optimizer_args'
            - optional: 'device', 'n_fold', 'n_epoch', 'patience',
                        'batch_size', 'model_save_dir', 'seed',
                        'seed_worker', 'max_workers', 'pin_memory',
                        'persistent_workers', 'best_model_name'

    Raises:
        ValueError: If required keys are missing.
        Returns: None
    """
    # Validate parameters
    validate_train_param(param)
    os.makedirs(param.get("model_save_dir", "./models"), exist_ok=True)
    overall_best_loss = float("inf")
    best_model_path = os.path.join(param["model_save_dir"], param["best_model_name"])

    # Stratified KFold
    kfold = StratifiedKFold(
        n_splits=param.get("n_fold", 5), shuffle=True, random_state=param.get("seed", 88)
    )

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(dataset_args["metadata_df"], dataset_args["raw_labels"])
    ):
        print(f"\nFold {fold+1}/{param.get('n_fold',5)}")

        # Prepare data loaders
        train_dataset = dataset_class(
            train_idx, dataset_args["cached_images"], dataset_args["cached_labels"]
        )
        val_dataset = dataset_class(
            val_idx, dataset_args["cached_images"], dataset_args["cached_labels"]
        )

        g = torch.Generator()
        g.manual_seed(param.get("seed", 88))

        train_loader = DataLoader(
            train_dataset,
            batch_size=param.get("batch_size", 64),
            shuffle=True,
            num_workers=param.get("max_workers", 1),
            worker_init_fn=param.get("seed_worker", seed_worker),
            generator=g,
            pin_memory=param.get("pin_memory", True),
            persistent_workers=param.get("persistent_workers", False),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=param.get("batch_size", 64),
            shuffle=False,
            num_workers=param.get("max_workers", 1),
            worker_init_fn=param.get("seed_worker", seed_worker),
            generator=g,
            pin_memory=param.get("pin_memory", True),
            persistent_workers=param.get("persistent_workers", False),
        )

        # Initialize model and load pretrained weights
        device = param.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model = model_class(**model_args).to(device)

        # Load pretrained weights
        state_dict = torch.load(param["pretrained_path"], map_location=device)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)

        # Freeze base layers if requested
        if param.get("freeze_layers", False):
            for name, p in model.named_parameters():
                if not any(x in name for x in param.get("unfreeze_patterns", [])):
                    p.requires_grad = False

        # DataParallel if multiple GPUs
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = param["optimizer_class"](
            filter(lambda p: p.requires_grad, model.parameters()),
            **param["optimizer_args"]
        )
        torch.cuda.empty_cache()

        best_fold_loss = float("inf")
        patience_counter = 0
        epoch_pbar = tqdm(range(param.get("n_epoch", 100)), desc="Epoch", leave=False)

        # Training loop
        for epoch in epoch_pbar:
            train_loss = train_epoch(model, train_loader, optimizer, param)
            val_loss, val_f1, val_acc = validate_epoch(model, val_loader, param)

            best_fold_loss, patience_counter, overall_best_loss = save_best_model(
                model, val_loss, best_fold_loss, patience_counter,
                overall_best_loss, param, fold, best_model_path
            )

            epoch_pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'val_acc': f"{val_acc:.4f}",
                'val_f1': f"{val_f1:.4f}",
                'patience': patience_counter,
            })

            print(f"Fold {fold+1} | Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            if patience_counter >= param.get("patience", 5):
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.cuda.empty_cache()
    print(f"\nFine-tuning completed. Best model saved at: {best_model_path}")
