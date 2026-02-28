"""
Run full evaluation pipeline and save evaluation_results.json.
Use this or the notebook to generate the single source of truth for metrics and dataset stats.

Usage (from repo root):
    python scripts/run_evaluation.py

Expects bharatfakenewskosh.xlsx in one of: data/, notebooks/, current directory.
Saves model/evaluation_results.json.
"""

import json
import sys
from pathlib import Path

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score

from src.data.loader import get_feature_target, load_dataset
from src.features.preprocessing import prepare_text_column
from src.models.pipelines import build_dt_pipeline, build_lr_pipeline


def find_dataset_path() -> Path:
    """Find bharatfakenewskosh.xlsx in data/, notebooks/, or cwd."""
    candidates = [
        REPO_ROOT / "data" / "bharatfakenewskosh.xlsx",
        REPO_ROOT / "notebooks" / "bharatfakenewskosh.xlsx",
        REPO_ROOT / "bharatfakenewskosh.xlsx",
        Path.cwd() / "bharatfakenewskosh.xlsx",
        Path.cwd() / "data" / "bharatfakenewskosh.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset not found. Place bharatfakenewskosh.xlsx in data/, notebooks/, or repo root."
    )


def main():
    dataset_path = find_dataset_path()
    print(f"Loading dataset: {dataset_path}")

    df_raw = load_dataset(str(dataset_path))
    total_raw = len(df_raw)
    print(f"Raw rows: {total_raw}")

    df = prepare_text_column(df_raw, drop_empty=True)
    after_drop = len(df)
    print(f"After dropping empty text: {after_drop}")

    # Class counts (label 1 = Fake, 0 = Real)
    y_all = df["label"]
    fake_count = int((y_all == 1).sum())
    real_count = int((y_all == 0).sum())
    assert fake_count + real_count == after_drop, "Class counts must sum to total"
    fake_pct = fake_count / after_drop
    real_pct = real_count / after_drop
    assert abs(fake_pct + real_pct - 1.0) < 1e-6, "Class percentages must sum to 100%"

    print(f"Fake: {fake_count} ({fake_pct:.2%})")
    print(f"Real: {real_count} ({real_pct:.2%})")

    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_size = len(X_train)
    test_size = len(X_test)
    print(f"Train: {train_size}, Test: {test_size}")

    # Sanity: test set size
    expected_test = int(round(0.2 * after_drop))
    assert abs(test_size - expected_test) < 2, (
        f"Test set size {test_size} inconsistent with 20% of {after_drop}"
    )

    lr_pipeline = build_lr_pipeline(random_state=42)
    dt_pipeline = build_dt_pipeline(random_state=42)

    print("Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    print("Training Decision Tree...")
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    y_proba_dt = dt_pipeline.predict_proba(X_test)[:, 1]

    def metrics_dict(y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "confusion_matrix": cm,
        }

    lr_metrics = metrics_dict(y_test, y_pred_lr, y_proba_lr)
    dt_metrics = metrics_dict(y_test, y_pred_dt, y_proba_dt)

    # Cross-validation (5-fold F1)
    cv_lr = cross_val_score(lr_pipeline, X, y, cv=5, scoring="f1", n_jobs=-1)
    cv_dt = cross_val_score(dt_pipeline, X, y, cv=5, scoring="f1", n_jobs=-1)
    lr_metrics["CV F1"] = [round(float(cv_lr.mean()), 4), round(float(cv_lr.std()), 4)]
    dt_metrics["CV F1"] = [round(float(cv_dt.mean()), 4), round(float(cv_dt.std()), 4)]

    # Sanity: accuracy = correct / total (tolerance allows for round(acc, 4))
    lr_correct = (np.array(y_pred_lr) == np.array(y_test)).sum()
    assert abs(lr_metrics["Accuracy"] - lr_correct / test_size) < 5e-4, (
        f"Accuracy sanity check failed: {lr_metrics['Accuracy']} vs {lr_correct}/{test_size}"
    )

    # Sanity: confusion matrix totals
    cm_lr = lr_metrics["confusion_matrix"]
    assert cm_lr[0][0] + cm_lr[0][1] + cm_lr[1][0] + cm_lr[1][1] == test_size
    cm_dt = dt_metrics["confusion_matrix"]
    assert cm_dt[0][0] + cm_dt[0][1] + cm_dt[1][0] + cm_dt[1][1] == test_size

    dataset_stats = {
        "total_samples": total_raw,
        "after_drop_empty": after_drop,
        "train_size": train_size,
        "test_size": test_size,
        "class_counts": {"Fake": fake_count, "Real": real_count},
        "class_pct": {"Fake": round(fake_pct, 4), "Real": round(real_pct, 4)},
    }

    artifact = {
        "dataset_stats": dataset_stats,
        "models": {
            "Logistic Regression": lr_metrics,
            "Decision Tree": dt_metrics,
        },
        "split": {"test_size": test_size, "random_state": 42, "stratify": True},
    }

    out_path = REPO_ROOT / "model" / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved {out_path}")
    print("Dataset stats:", dataset_stats)
    print("LR metrics:", lr_metrics)
    print("DT metrics:", dt_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
