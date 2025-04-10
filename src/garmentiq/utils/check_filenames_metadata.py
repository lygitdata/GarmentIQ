import os
import shutil

def check_filenames_metadata(output_dir, file_dir, metadata_df):
    """Checks if the filenames in the directory match the metadata's 'filename' column."""
    file_list = os.listdir(file_dir)
    metadata_filenames = metadata_df['filename'].tolist()
    file_list.sort()
    metadata_filenames.sort()

    if file_list != metadata_filenames:
        shutil.rmtree(output_dir)
        raise ValueError(f"Mismatch between directory filenames and metadata filenames. Maybe try again.")
    else:
        print(f"\nAll filenames in {file_dir} match the metadata.\n")