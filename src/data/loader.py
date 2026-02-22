"""
Dataset loading and preparation for news credibility classification.

Loads BharatFakeNewsKosh Excel data and provides feature/target arrays
for training and evaluation.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_dataset(
    path: str = "bharatfakenewskosh.xlsx",
) -> pd.DataFrame:
    """
    Load the BharatFakeNewsKosh dataset from Excel.

    Args:
        path: Path to the .xlsx file.

    Returns:
        DataFrame with raw columns including Statement, News Body, Label.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_excel(path)


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
