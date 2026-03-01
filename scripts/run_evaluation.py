"""
Run full evaluation pipeline and save evaluation_results.json.
Use this or the notebook to generate the single source of truth for metrics and dataset stats.

Usage (from repo root):
    python scripts/run_evaluation.py

Expects dataset/Fake.csv and dataset/True.csv (Kaggle Fake and Real News dataset).
Trains Logistic Regression, Naive Bayes, Random Forest, SVM; picks best by F1; saves to model/pipeline.pkl.
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
from src.models.pipelines import (
    build_lr_pipeline,
    build_nb_pipeline,
    build_rf_pipeline,
    build_svm_pipeline,
)


def _get_proba_for_auc(pipe, X_test):
    """Return class 1 probabilities for ROC-AUC (SVM uses decision_function)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_test)[:, 1]
    scores = pipe.decision_function(X_test)
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores) + 0.5
    return (scores - min_s) / (max_s - min_s)


def main():
    dataset_dir = REPO_ROOT / "dataset"
    if not (dataset_dir / "Fake.csv").exists() or not (dataset_dir / "True.csv").exists():
        raise FileNotFoundError(
            "Dataset not found. Place Fake.csv and True.csv in the dataset/ folder. "
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    print(f"Loading dataset from: {dataset_dir}")
    df_raw = load_dataset(str(dataset_dir))
    total_raw = len(df_raw)
    print(f"Raw rows (after merge, shuffle, dedup, drop short): {total_raw}")

    df = prepare_text_column(
        df_raw,
        text_column="combined_text",
        label_column="label",
        cleaned_col="cleaned_text",
        drop_empty=True,
    )
    after_drop = len(df)
    print(f"After dropping empty cleaned text: {after_drop}")

    # Label: 0 = Fake, 1 = Real
    y_all = df["label"]
    fake_count = int((y_all == 0).sum())
    real_count = int((y_all == 1).sum())
    assert fake_count + real_count == after_drop, "Class counts must sum to total"
    fake_pct = fake_count / after_drop
    real_pct = real_count / after_drop
    assert abs(fake_pct + real_pct - 1.0) < 1e-6, "Class percentages must sum to 100%"

    print(f"Fake: {fake_count} ({fake_pct:.2%})")
    print(f"Real: {real_count} ({real_pct:.2%})")

    X, y = get_feature_target(df, text_column="cleaned_text", label_column="label")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_size = len(X_train)
    test_size = len(X_test)
    print(f"Train: {train_size}, Test: {test_size}")

    pipelines = {
        "Logistic Regression": build_lr_pipeline(random_state=42),
        "Naive Bayes": build_nb_pipeline(),
        "Random Forest": build_rf_pipeline(random_state=42),
        "SVM": build_svm_pipeline(random_state=42),
    }

    def metrics_dict(y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "confusion_matrix": cm,
        }

    all_metrics = {}
    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = _get_proba_for_auc(pipe, X_test)
        all_metrics[name] = metrics_dict(y_test, y_pred, y_proba)
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1", n_jobs=-1)
        all_metrics[name]["CV F1"] = [
            round(float(cv_scores.mean()), 4),
            round(float(cv_scores.std()), 4),
        ]

    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["F1 Score"])
    best_pipeline = pipelines[best_model_name]
    print(f"Best model (by F1): {best_model_name}")

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
        "models": all_metrics,
        "best_model": best_model_name,
        "split": {"test_size": test_size, "random_state": 42, "stratify": True},
    }

    out_path = REPO_ROOT / "model" / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    import joblib
    pipeline_path = REPO_ROOT / "model" / "pipeline.pkl"
    joblib.dump(best_pipeline, pipeline_path)
    print(f"Saved best pipeline ({best_model_name}) to {pipeline_path}")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
