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
    def __init__(self, indices, cached_images, cached_labels):
        self.indices = indices
        self.cached_images = cached_images
        self.cached_labels = cached_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.cached_images[actual_idx], self.cached_labels[actual_idx]


def seed_worker(worker_id, SEED=88):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_epoch(model, train_loader, optimizer, param):
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
    Validate the parameter dictionary for required and optional keys with type checks.
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
