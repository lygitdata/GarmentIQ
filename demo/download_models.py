"""Download the three published checkpoints used by the local demo."""
import argparse
from pathlib import Path
import shutil
import urllib.request

from .pipeline import MODEL_URLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".local-demo" / "models")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = args.model_dir / name
        if path.is_file() and path.stat().st_size:
            print(f"Already present: {name}", flush=True)
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name}…", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
            if not temporary.stat().st_size or (expected and temporary.stat().st_size != int(expected)):
                raise OSError(f"Incomplete download: {name}")
            temporary.replace(path)
            print(f"Saved {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
