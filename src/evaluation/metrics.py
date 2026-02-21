"""
Evaluation metrics and model comparison for news credibility.

Provides classification report, ROC-AUC, precision/recall/F1,
and cross-validation scoring.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    y_pred: List[int],
    y_proba: List[float],
    target_names: Tuple[str, str] = ("Real (0)", "Fake (1)"),
) -> dict:
    """
    Compute classification report and ROC-AUC for a single model.

    Args:
        pipeline: Fitted pipeline (unused; for API consistency).
        X_test: Test features (unused; predictions passed directly).
        y_test: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probability for positive class (Fake).
        target_names: Names for the two classes in the report.

    Returns:
        Dict with keys: report (str), roc_auc (float), precision, recall, f1, accuracy.
    """
    report = classification_report(
        y_test, y_pred, target_names=list(target_names)
    )
    roc_auc = roc_auc_score(y_test, y_proba)
    return {
        "report": report,
        "roc_auc": roc_auc,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def model_comparison_table(
    model_names: List[str],
    y_test: pd.Series,
    y_preds: List[List[int]],
    y_probas: List[List[float]],
) -> pd.DataFrame:
    """
    Build a comparison table of Accuracy, Precision, Recall, F1, ROC-AUC.

    Args:
        model_names: List of model display names.
        y_test: True labels.
        y_preds: List of predicted label arrays.
        y_probas: List of positive-class probability arrays.

    Returns:
        DataFrame with columns Model, Accuracy, Precision, Recall, F1 Score, ROC-AUC.
    """
    results = []
    for name, y_pred, y_proba in zip(model_names, y_preds, y_probas):
        results.append(
            {
                "Model": name,
                "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
                "Precision": f"{precision_score(y_test, y_pred, zero_division=0):.4f}",
                "Recall": f"{recall_score(y_test, y_pred, zero_division=0):.4f}",
                "F1 Score": f"{f1_score(y_test, y_pred, zero_division=0):.4f}",
                "ROC-AUC": f"{roc_auc_score(y_test, y_proba):.4f}",
            }
        )
    return pd.DataFrame(results)


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
) -> Tuple[float, float]:
    """
    Run k-fold cross-validation and return mean and std of the score.

    Args:
        pipeline: Unfitted pipeline (will be cloned per fold).
        X: Full feature series.
        y: Full target series.
        cv: Number of folds.
        scoring: Metric name (e.g. 'f1', 'accuracy').
        n_jobs: Number of parallel jobs.

    Returns:
        (mean_score, std_score).
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std())
