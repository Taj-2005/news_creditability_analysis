"""
Load evaluation results from the artifact produced by the notebook or run_evaluation script.
Single source of truth for dataset stats and model metrics in the dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("news_credibility_app")

# Artifact path relative to repo root
EVALUATION_ARTIFACT_NAME = "evaluation_results.json"
MODEL_DIR_NAME = "model"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _artifact_path() -> Path:
    """Resolve path to evaluation_results.json (model/ or repo root)."""
    root = _repo_root()
    for base in [root / MODEL_DIR_NAME, root, Path.cwd() / MODEL_DIR_NAME, Path.cwd()]:
        p = base / EVALUATION_ARTIFACT_NAME
        if p.exists():
            return p
    return root / MODEL_DIR_NAME / EVALUATION_ARTIFACT_NAME


def load_evaluation_artifact() -> Optional[Dict[str, Any]]:
    """
    Load evaluation_results.json if it exists.
    Returns None if file is missing or invalid.
    """
    path = _artifact_path()
    if not path.exists():
        logger.debug("Evaluation artifact not found at %s", path)
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load evaluation artifact: %s", e)
        return None


def get_dataset_stats() -> Optional[Dict[str, Any]]:
    """
    Return dataset statistics from the evaluation artifact.
    Keys: total_samples, after_drop_empty, train_size, test_size,
          class_counts (dict Fake/Real), class_pct (dict), dataset_size_str.
    """
    data = load_evaluation_artifact()
    if not data:
        return None
    stats = data.get("dataset_stats")
    if not stats or not isinstance(stats, dict):
        return None
    total = stats.get("total_samples") or stats.get("after_drop_empty")
    if total is not None:
        stats = dict(stats)
        stats["dataset_size_str"] = f"{total:,}"
    return stats


def get_metrics() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Return per-model metrics from the evaluation artifact.
    Structure: { "Logistic Regression": { "Accuracy": 0.87, ... }, ... }
    """
    data = load_evaluation_artifact()
    if not data:
        return None
    models = data.get("models")
    if not models or not isinstance(models, dict):
        return None
    return models


def get_best_model_name() -> Optional[str]:
    """Return the best model name from the artifact, or None."""
    data = load_evaluation_artifact()
    if not data:
        return None
    return data.get("best_model")


def get_confusion_matrices() -> Optional[Dict[str, List[List[int]]]]:
    """Return per-model confusion matrices (2x2). Keys: model names."""
    data = load_evaluation_artifact()
    if not data:
        return None
    models = data.get("models")
    if not models:
        return None
    out = {}
    for name, m in models.items():
        cm = m.get("confusion_matrix")
        if cm is not None and isinstance(cm, (list, tuple)) and len(cm) == 2:
            out[name] = [list(row) for row in cm]
    return out if out else None


def artifact_available() -> bool:
    """True if evaluation_results.json exists and is loadable."""
    return load_evaluation_artifact() is not None
