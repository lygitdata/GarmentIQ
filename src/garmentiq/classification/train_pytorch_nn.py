import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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


def train_pytorch_nn(
    model_class: Type[torch.nn.Module],
    model_args: dict,
    dataset_class: Callable,
    dataset_args: dict,
    param: dict,
):
    """
    Trains a PyTorch neural network using k-fold cross-validation with early stopping and model checkpointing.

    This function performs training and validation across multiple folds using stratified sampling.
    It manages model instantiation, training loops, early stopping, and saves the best model based on validation loss.

    Args:
        model_class (Type[torch.nn.Module]): The class of the PyTorch model to instantiate.
                                            Must inherit from `torch.nn.Module`.
        model_args (dict): Dictionary of arguments used to initialize `model_class`.
        dataset_class (Callable): A callable class or function that returns a `torch.utils.data.Dataset`-compatible dataset.
        dataset_args (dict): Dictionary with dataset components:
            - 'metadata_df' (pandas.DataFrame): Metadata with labels, used for stratification.
            - 'raw_labels' (array-like): Raw class labels used by StratifiedKFold.
            - 'cached_images' (torch.Tensor): Preprocessed image tensor.
            - 'cached_labels' (torch.Tensor): Corresponding labels.
        param (dict): Dictionary of training hyperparameters and configuration values.
                      Required Keys:
                          - `optimizer_class` (type): PyTorch optimizer class (e.g., `torch.optim.Adam`).
                          - `optimizer_args` (dict): Arguments passed to the optimizer.
                      Optional Keys (with defaults and types):
                          - `device` (torch.device): Training device. Default is `"cuda"` if available, else `"cpu"`.
                          - `n_fold` (int): Number of stratified folds for cross-validation. Default: 5.
                          - `n_epoch` (int): Number of training epochs per fold. Default: 100.
                          - `patience` (int): Epochs to wait before early stopping. Default: 5.
                          - `batch_size` (int): Batch size for training and validation. Default: 64.
                          - `model_save_dir` (str): Directory to save model checkpoints. Default: `"./models"`.
                          - `seed` (int): Random seed for reproducibility. Default: 88.
                          - `seed_worker` (Callable): Function to seed workers in the DataLoader. Default: `seed_worker`.
                          - `max_workers` (int): Number of subprocesses for data loading. Default: `os.cpu_count()`.
                          - `best_model_name` (str): Filename for saving the best model. Default: `"best_model.pt"`.

    Raises:
        ValueError: If any required key is missing from `param`.
        TypeError: If any parameter is of the wrong type.
        FileNotFoundError: If the model directory cannot be created or accessed.

    Returns:
        None
    """
    # Validate and catch parameters
    validate_train_param(param)

    # Prepare save directories
    os.makedirs(param["model_save_dir"], exist_ok=True)
    overall_best_loss = float("inf")
    best_model_path = os.path.join(param["model_save_dir"], param["best_model_name"])

    kfold = StratifiedKFold(
        n_splits=param["n_fold"], shuffle=True, random_state=param["seed"]
    )

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(dataset_args["metadata_df"], dataset_args["raw_labels"])
    ):
        print(f"\nFold {fold + 1}/{param['n_fold']}")

        # Prepare datasets and dataloaders
        train_dataset = dataset_class(
            train_idx, dataset_args["cached_images"], dataset_args["cached_labels"]
        )
        val_dataset = dataset_class(
            val_idx, dataset_args["cached_images"], dataset_args["cached_labels"]
        )

        g = torch.Generator()
        g.manual_seed(param["seed"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=param["batch_size"],
            shuffle=True,
            num_workers=param["max_workers"],
            worker_init_fn=param["seed_worker"],
            generator=g,
            pin_memory=param["pin_memory"],
            persistent_workers=param["persistent_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=param["batch_size"],
            shuffle=False,
            num_workers=param["max_workers"],
            worker_init_fn=param["seed_worker"],
            generator=g,
            pin_memory=param["pin_memory"],
            persistent_workers=param["persistent_workers"],
        )

        # Initialize model and optimizer
        model = model_class(**model_args).to(param["device"])
        if param["device"].type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        optimizer = param["optimizer_class"](
            model.parameters(), **param["optimizer_args"]
        )
        torch.cuda.empty_cache()

        best_fold_loss = float("inf")
        patience_counter = 0
        epoch_pbar = tqdm(range(param["n_epoch"]), desc="Total Progress", leave=False)

        # Training and Validation Loop
        for epoch in epoch_pbar:
            # Training phase
            epoch_loss = train_epoch(model, train_loader, optimizer, param)
            # Validation phase
            val_loss, f1, acc = validate_epoch(model, val_loader, param)

            # Save the best model and check for early stopping
            best_fold_loss, patience_counter, overall_best_loss = save_best_model(
                model,
                val_loss,
                best_fold_loss,
                patience_counter,
                overall_best_loss,
                param,
                fold,
                best_model_path,
            )
            # Early stopping
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{epoch_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{acc:.4f}",
                    "val_f1": f"{f1:.4f}",
                    "patience": patience_counter,
                }
            )

            print(
                f"Fold {fold+1} | Epoch {epoch+1} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}"
            )

            if patience_counter >= param["patience"]:
                print(f"Early stopping at epoch {epoch+1} (fold {fold + 1})")
                break

    del model
    torch.cuda.empty_cache()

    print(f"\nTraining completed. Best model saved at: {best_model_path}")
