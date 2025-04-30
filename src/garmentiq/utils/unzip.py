import zipfile
from tqdm.notebook import tqdm


def unzip(zip_path, extract_to="."):
    """
    Extracts the contents of a ZIP file with a progress bar.

    This function unzips the contents of the specified ZIP file to the given directory.
    A progress bar is displayed during the extraction using `tqdm`.

    :param zip_path: Path to the ZIP file to be extracted.
    :type zip_path: str
    :param extract_to: Destination directory where the contents will be extracted. Defaults to the current directory.
    :type extract_to: str

    :returns: None
    :rtype: None
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)
