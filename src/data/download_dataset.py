"""Download the public Medical Appointment No Shows dataset."""
from __future__ import annotations

from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

URLS = [
    "https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/master/noshowappointments-kagglev2-may-2016.csv",
    "https://d17h27t6h515a.cloudfront.net/topher/2017/October/59dd2e9a_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv",
]
URL = URLS[0]
OUTPUT = Path("data/raw/KaggleV2-May-2016.csv")


def download(url: str | None = None, output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = [url] if url else URLS
    last_error = None
    for candidate in candidates:
        try:
            print(f"Downloading dataset from {candidate}")
            with urlopen(candidate, timeout=90) as response, output.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if output.stat().st_size < 1_000_000:
                raise ValueError("Downloaded file is unexpectedly small.")
            print(f"Saved {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
            return output
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
            print(f"Download attempt failed: {exc}")
            if output.exists():
                output.unlink()

    raise RuntimeError(
        "Could not download the dataset automatically. "
        "Download the public 'Medical Appointment No Shows' CSV from Kaggle and "
        "save it as data/raw/KaggleV2-May-2016.csv, then run the pipeline again. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    download()
