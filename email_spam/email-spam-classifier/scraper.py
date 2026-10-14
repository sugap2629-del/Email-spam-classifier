"""Download the Kaggle SMS Spam Collection into the project's data directory."""

import argparse
import csv
from io import BytesIO
from io import StringIO
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset"


def download_dataset(output_path: Path, url: str = DEFAULT_URL, force: bool = False) -> int:
    """Fetch, validate, and save a Kaggle CSV or UCI TSV; return its message count."""
    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it.")

    request = Request(url, headers={"User-Agent": "email-spam-classifier/1.0"})
    with urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        dataset_info = next(
            (
                info
                for info in archive.infolist()
                if info.filename.rsplit("/", 1)[-1].casefold() == "spam.csv"
            ),
            None,
        )
        is_kaggle_csv = dataset_info is not None
        if dataset_info is None:
            dataset_info = next(
                (
                    info
                    for info in archive.infolist()
                    if info.filename.rsplit("/", 1)[-1].casefold() == "smsspamcollection"
                ),
                None,
            )
        if dataset_info is None:
            raise ValueError("The archive contains neither spam.csv nor SMSSpamCollection.")
        dataset_bytes = archive.read(dataset_info)

    try:
        decoded = dataset_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        if not is_kaggle_csv:
            raise
        decoded = dataset_bytes.decode("cp1252")
        dataset_bytes = decoded.encode("utf-8")
    if is_kaggle_csv:
        reader = csv.reader(StringIO(decoded))
        header = next(reader, None)
        if header is None:
            raise ValueError("Downloaded spam.csv is empty.")
        normalized_header = [column.strip().lower() for column in header]
        label_index = next(
            (normalized_header.index(column) for column in ("v1", "label", "category") if column in normalized_header),
            None,
        )
        message_index = next(
            (normalized_header.index(column) for column in ("v2", "message", "text") if column in normalized_header),
            None,
        )
        if label_index is None or message_index is None:
            raise ValueError("Kaggle CSV must contain label and message columns such as v1 and v2.")
        row_count = 0
        labels = set()
        for row in reader:
            if len(row) > max(label_index, message_index) and row[message_index].strip():
                row_count += 1
                labels.add(row[label_index].strip().casefold())
    else:
        rows = [line for line in decoded.splitlines() if line.strip()]
        row_count = len(rows)
        labels = {line.split("\t", 1)[0].casefold() for line in rows if "\t" in line}

    if not row_count or not {"ham", "spam"}.issubset(labels):
        raise ValueError("Downloaded file does not contain valid ham and spam examples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as temporary_file:
            temporary_file.write(dataset_bytes)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "scam.csv")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL of a ZIP containing spam.csv or SMSSpamCollection")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()

    try:
        count = download_dataset(args.output, args.url, args.force)
    except (OSError, BadZipFile, UnicodeDecodeError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {count:,} messages to {args.output.resolve()}")


if __name__ == "__main__":
    main()