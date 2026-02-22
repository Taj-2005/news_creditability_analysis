"""
Visualization utilities for model evaluation.

Confusion matrices, ROC curves, metrics comparison bar chart,
and top TF-IDF feature coefficients.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def plot_confusion_matrices(
    y_test: pd.Series,
    y_preds: List[List[int]],
    titles: List[str],
    display_labels: List[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = "confusion_matrices.png",
) -> None:
    """
    Plot side-by-side confusion matrices for multiple models.

    Args:
        y_test: True labels.
        y_preds: List of predicted label arrays.
        titles: Title for each subplot.
        display_labels: Class labels for axes (e.g. ['Real', 'Fake']).
        figsize: Figure size.
        save_path: If set, save figure to this path.
    """
    if display_labels is None:
        display_labels = ["Real", "Fake"]
    n = len(y_preds)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, y_pred, title in zip(axes, y_preds, titles):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{title}\nConfusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(
    y_test: pd.Series,
    model_probas: List[List[float]],
    model_names: List[str],
    colors: Optional[List[str]] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = "roc_curves.png",
) -> None:
    """
    Plot ROC curves for multiple models (one subplot per model).

    Args:
        y_test: True labels.
        model_probas: List of positive-class probability arrays.
        model_names: Label for each model.
        colors: Optional list of colors; default uses royalblue, darkorange.
        figsize: Figure size.
        save_path: If set, save figure.
    """
    if colors is None:
        colors = ["royalblue", "darkorange", "green", "purple"][: len(model_names)]
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, y_proba, name, color in zip(axes, model_probas, model_names, colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}\nROC Curve", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = "model_comparison.png",
) -> None:
    """
    Bar chart comparing models on Accuracy, Precision, Recall, F1, ROC-AUC.

    Args:
        results_df: DataFrame from model_comparison_table (Model + metric columns).
        metrics: Column names to plot; default ['Accuracy','Precision','Recall','F1 Score','ROC-AUC'].
        figsize: Figure size.
        save_path: If set, save figure.
    """
    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    # Ensure we only use columns that exist
    metrics = [m for m in metrics if m in results_df.columns]
    if not metrics:
        return
    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(results_df)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["royalblue", "darkorange", "green", "purple"]
    for i, (_, row) in enumerate(results_df.iterrows()):
        scores = [float(row[m]) for m in metrics]
        offset = width * (i - (n_models - 1) / 2)
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=row["Model"],
            color=colors[i % len(colors)],
            edgecolor="black",
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_features(
    pipeline: Pipeline,
    top_n: int = 20,
    figsize: tuple = (16, 6),
    save_path: Optional[str] = "top_features.png",
) -> None:
    """
    Plot top coefficients for Logistic Regression (words associated with Fake vs Real).

    Only applicable for pipelines whose classifier has .coef_ (e.g. LogisticRegression).

    Args:
        pipeline: Fitted pipeline with 'tfidf' and 'clf' steps.
        top_n: Number of top words per class.
        figsize: Figure size.
        save_path: If set, save figure.
    """
    try:
        feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
        coefs = pipeline.named_steps["clf"].coef_[0]
    except (KeyError, AttributeError):
        return
    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].barh(
        [feature_names[i] for i in top_fake_idx],
        [coefs[i] for i in top_fake_idx],
        color="#e74c3c",
        edgecolor="black",
    )
    axes[0].set_title("Top 20 Words → FAKE News", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("LR Coefficient")
    axes[0].invert_yaxis()
    axes[1].barh(
        [feature_names[i] for i in top_real_idx],
        [abs(coefs[i]) for i in top_real_idx],
        color="#2ecc71",
        edgecolor="black",
    )
    axes[1].set_title("Top 20 Words → REAL News", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("|LR Coefficient|")
    axes[1].invert_yaxis()
    plt.suptitle(
        "Logistic Regression — Most Informative Features",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
