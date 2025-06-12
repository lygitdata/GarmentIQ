import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from tqdm.notebook import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score
import random
import numpy as np


class CachedDataset(Dataset):
    """
    A PyTorch Dataset that wraps pre-loaded data (images and labels) in memory.

    This dataset is designed to be used when images and labels have already been
    loaded and preprocessed into PyTorch tensors or NumPy arrays, avoiding
    repeated disk I/O during training/validation.

    Attributes:
        indices (list or numpy.ndarray): A list or array of indices that map
            to specific items in `cached_images` and `cached_labels`. This
            allows for flexible subsetting (e.g., for train/validation splits).
        cached_images (torch.Tensor): A tensor containing the pre-loaded images.
        cached_labels (torch.Tensor): A tensor containing the pre-loaded labels.
    """

    def __init__(self, indices, cached_images, cached_labels):
        """
        Initializes the CachedDataset.

        Args:
            indices (list or numpy.ndarray): Indices to select from the cached data.
            cached_images (torch.Tensor): Pre-loaded image tensor.
            cached_labels (torch.Tensor): Pre-loaded label tensor.
        """
        self.indices = indices
        self.cached_images = cached_images
        self.cached_labels = cached_labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        actual_idx = self.indices[idx]
        return self.cached_images[actual_idx], self.cached_labels[actual_idx]


def seed_worker(worker_id, SEED=88):
    """
    Seeds the random number generators for a DataLoader worker.

    This function is intended to be passed as `worker_init_fn` to a PyTorch
    DataLoader to ensure reproducibility across different worker processes.
    It seeds Python's `random` module, NumPy, and PyTorch for each worker.

    Args:
        worker_id (int): The ID of the current worker process.
        SEED (int, optional): The base seed value. The worker's seed will be
            `SEED + worker_id`. Defaults to 88.
    """
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_epoch(model, train_loader, optimizer, param):
    """
    Performs a single training epoch for a PyTorch model.

    Sets the model to training mode, iterates through the `train_loader`,
    performs forward and backward passes, and updates model weights using
    the provided optimizer. A progress bar is displayed, showing the batch loss.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        param (dict): A dictionary containing training parameters, including:
            - "device" (torch.device): The device (e.g., 'cuda' or 'cpu') to use for training.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Training", leave=False)

    for images, labels in train_pbar:
        images, labels = images.to(param["device"], non_blocking=True), labels.to(
            param["device"], non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate the loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate_epoch(model, val_loader, param):
    """
    Performs a single validation epoch for a PyTorch model.

    Sets the model to evaluation mode, iterates through the `val_loader`
    without gradient calculations, and computes the validation loss, F1 score,
    and accuracy. A progress bar is displayed, showing the batch validation loss.

    Args:
        model (torch.nn.Module): The PyTorch model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        param (dict): A dictionary containing training parameters, including:
            - "device" (torch.device): The device (e.g., 'cuda' or 'cpu') to use for validation.

    Returns:
        tuple: A tuple containing:
            - val_loss (float): The average validation loss for the epoch.
            - f1 (float): The weighted average F1 score on the validation set.
            - acc (float): The accuracy score on the validation set.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)

    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(param["device"]), labels.to(param["device"])
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # Collect predictions and true labels for F1 score calculation
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())  # Move to CPU and convert to numpy
            all_labels.extend(labels.cpu().numpy())  # Move to CPU and convert to numpy

            val_pbar.set_postfix({"val_batch_loss": f"{loss.item():.4f}"})

    # Calculate the average validation loss
    val_loss /= len(val_loader.dataset)

    # Calculate F1 Score (if needed)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)

    return val_loss, f1, acc


def save_best_model(
    model,
    val_loss,
    best_fold_loss,
    patience_counter,
    overall_best_loss,
    param,
    fold,
    best_model_path,
):
    """
    Saves the best model checkpoints based on validation loss and manages early stopping.

    This function updates the `best_fold_loss` and `patience_counter` for the current
    cross-validation fold. It also saves the model's state dictionary if it's the best
    performing model for the current fold or the overall best model across all folds.

    Args:
        model (torch.nn.Module): The current PyTorch model being trained.
        val_loss (float): The validation loss from the current epoch.
        best_fold_loss (float): The best validation loss recorded so far for the current fold.
        patience_counter (int): The number of epochs since the last improvement for the current fold.
        overall_best_loss (float): The best validation loss recorded so far across all folds.
        param (dict): A dictionary containing training parameters, including:
            - "model_save_dir" (str): Directory where model checkpoints will be saved.
        fold (int): The current fold number (0-indexed).
        best_model_path (str): The full path where the overall best model will be saved.

    Returns:
        tuple: A tuple containing:
            - best_fold_loss (float): The updated best validation loss for the current fold.
            - patience_counter (int): The updated patience counter for the current fold.
            - overall_best_loss (float): The updated overall best validation loss.
    """
    # Save the best model for this fold
    if val_loss < best_fold_loss:
        best_fold_loss = val_loss
        patience_counter = 0
        torch.save(
            model.state_dict(),
            os.path.join(param["model_save_dir"], f"fold_{fold + 1}_best.pt"),
        )
    else:
        patience_counter += 1

    # Save the overall best model
    if val_loss < overall_best_loss:
        overall_best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    return best_fold_loss, patience_counter, overall_best_loss


def validate_train_param(param: dict):
    """
    Validates the parameter dictionary for training configuration.

    This function checks for the presence and correct types of required
    parameters for training, and applies default values for optional parameters
    if they are not provided.

    Args:
        param (dict): The dictionary of training parameters to validate.

    Raises:
        ValueError: If a required parameter is missing.
        TypeError: If a parameter has an incorrect type.
    """
    # --- Required fields and types
    required_keys = {"optimizer_class": type, "optimizer_args": dict}

    # --- Optional fields with default values and expected types
    optional_keys = {
        "device": (
            torch.device,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        "n_fold": (int, 5),
        "n_epoch": (int, 100),
        "patience": (int, 5),
        "batch_size": (int, 64),
        "model_save_dir": (str, "./models"),
        "seed": (int, 88),
        "seed_worker": (Callable, seed_worker),
        "max_workers": (int, 0),
        "best_model_name": (str, "best_model.pt"),
        "pin_memory": (bool, False),
        "persistent_workers": (bool, False),
    }

    # --- Validate required keys
    for key, expected_types in required_keys.items():
        if key not in param:
            raise ValueError(f"Missing required param key: '{key}'")
        if not isinstance(param[key], expected_types):
            raise TypeError(
                f"param['{key}'] must be of type {expected_types}, got {type(param[key])}"
            )

    # --- Apply defaults and type-check optional keys
    for key, (expected_type, default) in optional_keys.items():
        if key not in param:
            param[key] = default
        elif expected_type is not None and not isinstance(param[key], expected_type):
            raise TypeError(
                f"param['{key}'] must be of type {expected_type}, got {type(param[key])}"
            )


def validate_test_param(param: dict):
    """
    Validates the parameter dictionary for testing configuration.

    This function checks for the presence and correct types of optional
    parameters for testing, and applies default values if they are not provided.

    Args:
        param (dict): The dictionary of testing parameters to validate.

    Raises:
        TypeError: If a parameter has an incorrect type.
    """
    # --- Optional fields with default values and expected types
    optional_keys = {
        "device": (
            torch.device,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        "batch_size": (int, 64),
    }

    # --- Apply defaults and type-check optional keys
    for key, (expected_type, default) in optional_keys.items():
        if key not in param:
            param[key] = default
        elif expected_type is not None and not isinstance(param[key], expected_type):
            raise TypeError(
                f"param['{key}'] must be of type {expected_type}, got {type(param[key])}"
            )


def validate_pred_param(param: dict):
    """
    Validates the parameter dictionary for prediction configuration.

    This function checks for the presence and correct types of optional
    parameters for prediction, and applies default values if they are not provided.

    Args:
        param (dict): The dictionary of prediction parameters to validate.

    Raises:
        TypeError: If a parameter has an incorrect type.
    """
    # --- Optional fields with default values and expected types
    optional_keys = {
        "device": (
            torch.device,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        "batch_size": (int, 64),
    }

    # --- Apply defaults and type-check optional keys
    for key, (expected_type, default) in optional_keys.items():
        if key not in param:
            param[key] = default
        elif expected_type is not None and not isinstance(param[key], expected_type):
            raise TypeError(
                f"param['{key}'] must be of type {expected_type}, got {type(param[key])}"
            )
