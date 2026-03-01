"""
Dataset loading and preparation for news credibility classification.

Loads the Kaggle Fake and Real News dataset (Fake.csv, True.csv) from the dataset/
folder. Merges both files, adds binary labels (Fake=1, Real=0), combines title+text,
and returns a cleaned DataFrame ready for feature extraction.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DATASET_DIR_NAME = "dataset"
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"
MIN_TEXT_LENGTH = 20  # Drop rows with combined text shorter than this (after cleaning)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _find_dataset_dir() -> Path:
    """Resolve dataset directory: repo/dataset, cwd/dataset, or repo root."""
    root = _repo_root()
    for base in [root / DATASET_DIR_NAME, Path.cwd() / DATASET_DIR_NAME, root]:
        fake_path = base / FAKE_CSV
        true_path = base / TRUE_CSV
        if fake_path.exists() and true_path.exists():
            return base
    return root / DATASET_DIR_NAME


def load_dataset(
    dataset_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the Fake and Real News dataset from Fake.csv and True.csv.

    - Loads Fake.csv and True.csv
    - Adds label column: Fake=0, Real (True)=1
    - Concatenates datasets, shuffles (random_state=42), removes duplicates
    - Drops missing rows, drops very short texts
    - Creates combined text column: title + " " + text

    Args:
        dataset_dir: Optional path to folder containing Fake.csv and True.csv.
                     If None, uses dataset/ under repo root or cwd.

    Returns:
        DataFrame with columns: title, text, subject, date, label, combined_text.
        label: 0 = Fake, 1 = Real (True).

    Raises:
        FileNotFoundError: If Fake.csv or True.csv are not found.
    """
    if dataset_dir is None:
        base = _find_dataset_dir()
    else:
        base = Path(dataset_dir)

    fake_path = base / FAKE_CSV
    true_path = base / TRUE_CSV
    if not fake_path.exists():
        raise FileNotFoundError(
            f"Fake.csv not found. Place {FAKE_CSV} in {base} or pass dataset_dir."
        )
    if not true_path.exists():
        raise FileNotFoundError(
            f"True.csv not found. Place {TRUE_CSV} in {base} or pass dataset_dir."
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Label: 0 = Fake, 1 = Real (True news) — per dataset convention
    fake["label"] = 0
    true["label"] = 1

    # Concatenate and shuffle
    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove duplicates (on title+text to avoid duplicate articles)
    if "title" in df.columns and "text" in df.columns:
        df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    # Drop rows with missing title or text
    for col in ["title", "text"]:
        if col in df.columns:
            df = df.dropna(subset=[col])
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)

    # Combined text: title + space + text
    df["combined_text"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()

    # Drop rows with extremely short combined text
    df = df[df["combined_text"].str.len() >= MIN_TEXT_LENGTH].reset_index(drop=True)

    return df


def get_feature_target(
    df: pd.DataFrame,
    text_column: str = "cleaned_text",
    label_column: str = "label",
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract feature (text) and target (label) series from a prepared DataFrame.

    Args:
        df: DataFrame with text_column and label_column.
        text_column: Name of the column containing cleaned text.
        label_column: Name of the binary label column (0=Real, 1=Fake).

    Returns:
        Tuple of (X, y) as pandas Series.
    """
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' not in DataFrame")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not in DataFrame")
    return df[text_column], df[label_column]
