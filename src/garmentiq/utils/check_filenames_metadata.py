import os
import shutil


def check_filenames_metadata(output_dir, file_dir, metadata_df):
    """
    Validates that the filenames in a directory match those listed in a metadata DataFrame.

    This function compares the filenames found in the specified directory with the filenames
    listed in the provided metadata DataFrame. If the filenames do not match exactly (after sorting),
    the output directory is deleted and a `ValueError` is raised. If they match, a confirmation
    message is printed.

    :param output_dir: Path to the output directory that will be deleted if filenames do not match.
    :type output_dir: str
    :param file_dir: Path to the directory containing the files to check.
    :type file_dir: str
    :param metadata_df: A pandas DataFrame containing a 'filename' column with expected filenames.
    :type metadata_df: pandas.DataFrame

    :raises ValueError: If the filenames in the directory do not match those in the metadata.

    :returns: None
    :rtype: None
    """
    file_list = os.listdir(file_dir)
    metadata_filenames = metadata_df["filename"].tolist()
    file_list.sort()
    metadata_filenames.sort()

    if file_list != metadata_filenames:
        shutil.rmtree(output_dir)
        raise ValueError(
            f"Mismatch between directory filenames and metadata filenames. Maybe try again."
        )
    else:
        print(f"\nAll filenames in {file_dir} match the metadata.\n")
