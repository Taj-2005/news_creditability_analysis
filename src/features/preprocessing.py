"""
Text preprocessing for news credibility classification.

Provides a reproducible cleaning pipeline: lowercase, URL/mention removal,
tokenization, stopword removal, and lemmatization. Must match exactly
between training and inference (e.g. Streamlit app).
"""

import re
from typing import Optional, Set

import pandas as pd


def _get_stopwords_and_lemmatizer() -> tuple:
    """Lazy-load NLTK resources to avoid import-time downloads."""
    import nltk

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(
    text: Optional[str],
    stop_words: Optional[Set[str]] = None,
) -> str:
    """
    Full text preprocessing pipeline for news text.

    Steps: lowercase, remove URLs and @mentions, keep only letters,
    normalize whitespace, tokenize, remove stopwords and short tokens,
    lemmatize.

    Args:
        text: Raw input text (e.g. headline or article body).
        stop_words: Optional set of stopwords; if None, NLTK English is used.

    Returns:
        Cleaned tokenized string (space-joined).
    """
    if not isinstance(text, str):
        return ""
    if stop_words is None:
        stop_words, lemmatizer = _get_stopwords_and_lemmatizer()
    else:
        _, lemmatizer = _get_stopwords_and_lemmatizer()

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    min_len = 2
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > min_len
    ]
    return " ".join(tokens)


def prepare_text_column(
    df: pd.DataFrame,
    statement_col: str = "Eng_Trans_Statement",
    body_col: str = "Eng_Trans_News_Body",
    label_col: str = "Label",
    combined_col: str = "combined_text",
    cleaned_col: str = "cleaned_text",
    output_label_col: str = "label",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Add combined text, cleaned text, and numeric label to the DataFrame.

    Args:
        df: Raw DataFrame with statement and body columns.
        statement_col: Column for headline/statement.
        body_col: Column for news body.
        label_col: Boolean label column (True=Fake, False=Real).
        combined_col: Name for combined text column.
        cleaned_col: Name for cleaned text column.
        output_label_col: Name for numeric label (0=Real, 1=Fake).
        drop_empty: If True, drop rows where cleaned text is empty.

    Returns:
        DataFrame with new columns; optionally with empty-text rows removed.
    """
    df = df.copy()
    df[combined_col] = (
        df[statement_col].fillna("") + " " + df[body_col].fillna("")
    )
    df[cleaned_col] = df[combined_col].apply(clean_text)
    df[output_label_col] = df[label_col].astype(int)

    if drop_empty:
        df = df[df[cleaned_col].str.strip() != ""].reset_index(drop=True)
    return df
