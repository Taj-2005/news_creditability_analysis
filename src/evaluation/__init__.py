"""Evaluation metrics and visualizations."""

from src.evaluation.metrics import (
    evaluate_model,
    model_comparison_table,
    run_cross_validation,
)

# Matplotlib-based plots (plot_confusion_matrices, plot_roc_curves, etc.) are in
# src.evaluation.visualization — import from there if needed (e.g. notebook).
# Not imported here to avoid pulling matplotlib on Streamlit Cloud where only
# plotly_viz is used.

__all__ = [
    "evaluate_model",
    "model_comparison_table",
    "run_cross_validation",
]
