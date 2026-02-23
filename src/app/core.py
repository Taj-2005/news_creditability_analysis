"""
Shared core for News Credibility dashboard: model loading, prediction, validation.
No UI; used by all pages. Deployment-safe paths, cached model, error handling.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

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
DATASET_SIZE = "26,000+"
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 50_000

# Expected metrics (from README / training) for dashboard when no eval artifacts
EXPECTED_METRICS = {
    "Logistic Regression": {
        "Accuracy": 0.87,
        "Precision": 0.88,
        "Recall": 0.90,
        "F1 Score": 0.89,
        "ROC-AUC": 0.93,
        "CV F1": (0.88, 0.01),
    },
    "Decision Tree": {
        "Accuracy": 0.80,
        "Precision": 0.81,
        "Recall": 0.84,
        "F1 Score": 0.82,
        "ROC-AUC": 0.80,
        "CV F1": (0.81, 0.02),
    },
}

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
