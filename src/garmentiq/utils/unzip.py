import zipfile
from tqdm import tqdm

def unzip(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)