import os
import pandas as pd
import shutil
import random
from typing import Optional
from garmentiq.utils.check_unzipped_dir import check_unzipped_dir
from garmentiq.utils.unzip import unzip
from garmentiq.utils.check_filenames_metadata import check_filenames_metadata


def train_test_split(
    output_dir: str,
    train_zip_dir: str,
    test_zip_dir: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 88,
    verbose: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    # Unzip train data
    train_out = os.path.join(output_dir, "train")
    unzip(train_zip_dir, train_out)
    print("\n")
    check_unzipped_dir(train_out)

    # If a test zip is provided, unzip and process it
    if test_zip_dir:
        test_out = os.path.join(output_dir, "test")
        unzip(test_zip_dir, test_out)
        check_unzipped_dir(test_out)

        # Load train metadata and check filenames
        train_metadata_path = os.path.join(train_out, "metadata.csv")
        df_train = pd.read_csv(train_metadata_path)
        if "filename" not in df_train.columns:
            raise ValueError("Train metadata must contain a 'filename' column.")
        check_filenames_metadata(
            output_dir, os.path.join(train_out, "images"), df_train
        )

        # Load test metadata and check filenames
        test_metadata_path = os.path.join(test_out, "metadata.csv")
        df_test = pd.read_csv(test_metadata_path)
        if "filename" not in df_test.columns:
            raise ValueError("Test metadata must contain a 'filename' column.")
        check_filenames_metadata(output_dir, os.path.join(test_out, "images"), df_test)

        # Summary information
        if verbose:
            print(f"\n\nTrain set summary (sample size: {len(df_train)}):\n")
            print(f"{df_train['garment'].value_counts()}\n")

            print(f"Test set summary (sample size: {len(df_test)}):\n")
            print(f"{df_test['garment'].value_counts()}\n")

        return {
            "train_images": f"{train_out}/images",
            "train_metadata": pd.read_csv(f"{train_out}/metadata.csv"),
            "test_images": f"{test_out}/images",
            "test_metadata": pd.read_csv(f"{test_out}/metadata.csv"),
        }

    # If no test zip is provided, split from train data
    print("Splitting train data into train/test sets...")

    # Load train metadata
    metadata_path = os.path.join(train_out, "metadata.csv")
    df = pd.read_csv(metadata_path)

    if "filename" not in df.columns:
        raise ValueError("metadata.csv must contain a 'filename' column.")

    # Select test split
    random.seed(seed)
    filenames = df["filename"].tolist()
    test_filenames = set(random.sample(filenames, int(len(filenames) * test_size)))

    # Prepare test folder
    test_out = os.path.join(output_dir, "test")
    test_images_dir = os.path.join(test_out, "images")
    os.makedirs(test_images_dir, exist_ok=True)

    train_images_dir = os.path.join(train_out, "images")

    # Move test files from train to test folder
    for fname in test_filenames:
        src = os.path.join(train_images_dir, fname)
        dst = os.path.join(test_images_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"File listed in metadata not found: {fname}")
        shutil.move(src, dst)

    # Save updated CSVs
    df_test = df[df["filename"].isin(test_filenames)]
    df_train = df[~df["filename"].isin(test_filenames)]

    df_test.to_csv(os.path.join(test_out, "metadata.csv"), index=False)
    df_train.to_csv(metadata_path, index=False)

    # Check if output images match the metadata records
    check_filenames_metadata(output_dir, os.path.join(train_out, "images"), df_train)
    check_filenames_metadata(output_dir, os.path.join(test_out, "images"), df_test)

    # Summary information
    if verbose:
        print(f"\n\nTrain set summary (sample size: {len(df_train)}):\n")
        print(f"{df_train['garment'].value_counts()}\n")

        print(f"Test set summary (sample size: {len(df_test)}):\n")
        print(f"{df_test['garment'].value_counts()}\n")

    return {
        "train_images": f"{train_out}/images",
        "train_metadata": pd.read_csv(f"{train_out}/metadata.csv"),
        "test_images": f"{test_out}/images",
        "test_metadata": pd.read_csv(f"{test_out}/metadata.csv"),
    }

    # If any mismatch found, remove the output directory and raise an error
    try:
        check_filenames_metadata(
            output_dir, os.path.join(train_out, "images"), df_train
        )
        check_filenames_metadata(output_dir, os.path.join(test_out, "images"), df_test)
    except ValueError as e:
        shutil.rmtree(output_dir)  # Clean up the output directory in case of error
        raise e
