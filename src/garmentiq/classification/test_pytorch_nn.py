import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Type
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from garmentiq.classification.utils import (
    CachedDataset,
    seed_worker,
    train_epoch,
    validate_epoch,
    save_best_model,
    validate_train_param,
    validate_test_param,
)


def test_pytorch_nn(
    model_path: str,
    model_class: Type[torch.nn.Module],
    model_args: dict,
    dataset_class: Callable,
    dataset_args: dict,
    param: dict,
):
    """
    Evaluates a trained PyTorch model on a test dataset.

    Loads the model from disk, prepares the test dataset, and computes loss, accuracy,
    F1 score, and prints a full classification report.

    :param model_path: Path to the saved model checkpoint file.
    :type model_path: str
    :param model_class: The class of the PyTorch model to instantiate. Must inherit from `torch.nn.Module`.
    :type model_class: Type[torch.nn.Module]
    :param model_args: Dictionary of arguments used to initialize the model.
    :type model_args: dict
    :param dataset_class: A callable class or function that returns a `torch.utils.data.Dataset`-compatible dataset.
                          (Note: Not directly used, but included for consistency with training pipeline.)
    :type dataset_class: Callable
    :param dataset_args: Dictionary with dataset components:
        - ``cached_images`` (torch.Tensor): Preprocessed test image tensors.
        - ``cached_labels`` (torch.Tensor): Corresponding test labels.
        - ``raw_labels`` (pandas.Series or array-like): Original labels for report generation.
    :type dataset_args: dict
    :param param: Dictionary of optional configuration parameters.

        **Optional Keys**:
            - ``device`` (torch.device): Device for computation. Defaults to `"cuda"` if available, else `"cpu"`.
            - ``batch_size`` (int): Batch size used for testing. Default is 64.

    :type param: dict

    :raises FileNotFoundError: If the model checkpoint cannot be loaded.
    :raises TypeError: If any parameter is of an incorrect type.

    :returns: None — prints test loss, accuracy, F1 score, and a classification report.
    :rtype: None
    """
    validate_test_param(param)
    model = model_class(**model_args).to(param["device"])
    state_dict = torch.load(model_path, map_location=param["device"], weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    test_dataset = TensorDataset(
        dataset_args["cached_images"], dataset_args["cached_labels"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=param["batch_size"], shuffle=False
    )

    # Evaluation
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(param["device"])
            labels = labels.to(param["device"])

            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_loss = total_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=sorted(dataset_args["raw_labels"].unique()),
        )
    )
