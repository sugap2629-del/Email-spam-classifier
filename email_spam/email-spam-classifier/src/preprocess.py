"""Dataset loading and lightweight SMS text normalization."""

import re
from pathlib import Path

import pandas as pd


_LABEL_COLUMNS = {"label", "v1", "category", "class", "target"}
_TEXT_COLUMNS = {"message", "v2", "text", "sms", "content"}
_LABEL_MAP = {
    "ham": "ham",
    "not spam": "ham",
    "legitimate": "ham",
    "legit": "ham",
    "0": "ham",
    "spam": "spam",
    "junk": "spam",
    "phishing": "spam",
    "1": "spam",
}


def clean_text(text: str) -> str:
    """Normalize URLs, email addresses, numbers, and whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-z_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _column_key(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load common Kaggle CSV or UCI TSV layouts into label/message columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for download instructions."
        )

    frame = pd.read_csv(
        path,
        sep=None,
        engine="python",
        header=None,
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    if frame.shape[1] < 2:
        raise ValueError("Expected a dataset with at least label and message columns.")

    first_row = [_column_key(value) for value in frame.iloc[0].tolist()]
    if any(value in _LABEL_COLUMNS | _TEXT_COLUMNS for value in first_row):
        frame.columns = frame.iloc[0].tolist()
        frame = frame.iloc[1:].reset_index(drop=True)

    columns = {_column_key(column): column for column in frame.columns}
    label_column = next(
        (columns[key] for key in _LABEL_COLUMNS if key in columns), frame.columns[0]
    )
    text_column = next(
        (columns[key] for key in _TEXT_COLUMNS if key in columns),
        next((column for column in frame.columns if column != label_column), None),
    )
    if text_column is None:
        raise ValueError("Could not identify the message column in the dataset.")

    result = frame[[label_column, text_column]].copy()
    result.columns = ["label", "message"]
    result = result.dropna(subset=["label", "message"])
    result["label"] = result["label"].astype(str).str.strip().str.lower().map(_LABEL_MAP)
    result["message"] = result["message"].astype(str).str.strip()
    result = result.dropna(subset=["label"])
    result = result[result["message"] != ""].reset_index(drop=True)

    if result.empty or result["label"].nunique() < 2:
        raise ValueError("Dataset must contain non-empty examples of both ham and spam.")
    return result
