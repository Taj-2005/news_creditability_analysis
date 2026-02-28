"""
Shared core for News Credibility dashboard: model loading, prediction, validation.
No UI; used by all pages. Deployment-safe paths, cached model, error handling.
Metrics and dataset stats are loaded from model/evaluation_results.json when available.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

from src.evaluation.results_loader import (
    artifact_available,
    get_dataset_stats,
    get_metrics,
)
from src.features.preprocessing import clean_text

logger = logging.getLogger("news_credibility_app")

# -----------------------------------------------------------------------------
# Constants (artifact paths + content)
# -----------------------------------------------------------------------------
MODEL_DIR_NAME = "model"
MODEL_FILENAME = "pipeline.pkl"
PAGE_TITLE = "News Credibility AI"
MODEL_ALGORITHM = "Logistic Regression"
MODEL_FEATURES = "TF-IDF (unigrams + bigrams)"
DATASET_NAME = "BharatFakeNewsKosh"
# Dataset size and metrics come from evaluation_results.json when available
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 50_000


def get_dataset_size_str() -> str:
    """Return dataset size string from evaluation artifact, or placeholder if missing."""
    stats = get_dataset_stats()
    if stats and "dataset_size_str" in stats:
        return stats["dataset_size_str"]
    if stats and "total_samples" in stats:
        return f"{stats['total_samples']:,}"
    if stats and "after_drop_empty" in stats:
        return f"{stats['after_drop_empty']:,}"
    return "—"  # Placeholder when artifact not generated yet


def get_expected_metrics() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Return per-model metrics from evaluation_results.json.
    None if artifact is missing (caller should show "Run notebook/script to generate results").
    """
    return get_metrics()

EXAMPLE_TEXTS = {
    "Real (credible)": (
        "Prime Minister Modi received an honorary doctorate from Oxford University "
        "in recognition of India's digital transformation initiatives and governance reforms."
    ),
    "Fake (misinformation)": (
        "Viral video claims to show a digitally edited flood in Rajasthan designed "
        "to mislead viewers about the actual situation. Fact-checkers have debunked the clip."
    ),
    "Short headline": (
        "Breaking: Scientists announce breakthrough in renewable energy storage technology."
    ),
}


def _model_base_dir() -> Path:
    file_based = repo_root / MODEL_DIR_NAME
    cwd_based = Path.cwd() / MODEL_DIR_NAME
    if (file_based / MODEL_FILENAME).exists():
        return file_based
    if (cwd_based / MODEL_FILENAME).exists():
        return cwd_based
    return file_based


@st.cache_resource
def load_model():
    import joblib

    base = _model_base_dir()
    model_path = base / MODEL_FILENAME
    if not model_path.exists():
        logger.error("Model file missing: %s", model_path)
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Ensure {MODEL_DIR_NAME}/{MODEL_FILENAME} exists. See README."
        )
    try:
        pipeline = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
        return pipeline
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


def run_prediction(pipeline, raw_text: str) -> Tuple[int, float, float]:
    cleaned = clean_text(raw_text)
    prediction = int(pipeline.predict([cleaned])[0])
    proba = pipeline.predict_proba([cleaned])[0]
    return prediction, float(proba[1]), float(proba[0])


def validate_input(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "Please enter some text."
    t = text.strip()
    if len(t) < MIN_INPUT_LENGTH:
        return False, f"Text is too short (minimum {MIN_INPUT_LENGTH} characters)."
    if len(t) > MAX_INPUT_LENGTH:
        return False, f"Text exceeds maximum length ({MAX_INPUT_LENGTH:,} characters)."
    return True, ""
