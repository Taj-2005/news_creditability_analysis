"""Dataset Intelligence — class distribution, text length, top TF-IDF. Plotly-only charts."""

import numpy as np
import streamlit as st

from src.app.components.ui import page_header
from src.app.core import DATASET_NAME, get_dataset_size_str, load_model
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_donut_class_distribution,
    plotly_histogram_text_length,
    plotly_top_tfidf_features,
)
from src.evaluation.results_loader import get_dataset_stats


def _sample_text_lengths(seed: int = 42) -> tuple:
    """Illustrative text length samples per class (when no dataset is loaded)."""
    rng = np.random.default_rng(seed)
    fake_len = rng.lognormal(5, 1, 500)
    real_len = rng.lognormal(5.2, 1, 500)
    return list(fake_len), list(real_len)


def render():
    dataset_size_str = get_dataset_size_str()
    page_header("Dataset Intelligence", f"Summary for **{DATASET_NAME}** ({dataset_size_str} articles)")

    stats = get_dataset_stats()
    if not stats:
        st.warning(
            "Dataset statistics not found. Run the notebook (with the dataset) or "
            "`python scripts/run_evaluation.py` to generate evaluation results."
        )
        st.stop()

    total = stats.get("after_drop_empty") or stats.get("total_samples")
    class_counts = stats.get("class_counts") or {}
    class_pct = stats.get("class_pct") or {}
    # Support both "Fake"/"Real" and numeric keys (0/1)
    fake_count = class_counts.get("Fake") or class_counts.get(0) or 0
    real_count = class_counts.get("Real") or class_counts.get(1) or 0
    fake_count = int(fake_count) if fake_count is not None else 0
    real_count = int(real_count) if real_count is not None else 0
    fake_pct = class_pct.get("Fake") or class_pct.get(0) or (fake_count / total if total else 0)
    real_pct = class_pct.get("Real") or class_pct.get(1) or (real_count / total if total else 0)
    if total and (fake_count + real_count) > 0 and abs((fake_pct or 0) + (real_pct or 0) - 1.0) > 0.01:
        fake_pct = fake_count / (fake_count + real_count)
        real_pct = real_count / (fake_count + real_count)

    # Top metrics row — all from artifact
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{total:,}" if total is not None else "—")
    col2.metric("Fake", f"{fake_pct:.1%}" if fake_pct else "—", help=f"Count: {fake_count:,}" if fake_count else None)
    col3.metric("Real", f"{real_pct:.1%}" if real_pct else "—", help=f"Count: {real_count:,}" if real_count else None)

    # Class distribution donut — actual counts
    st.markdown("#### Class distribution")
    if fake_count > 0 or real_count > 0:
        fig_donut = plotly_donut_class_distribution(
            counts=[fake_count, real_count],
            labels=["Fake", "Real"],
            colors=["#f87171", "#4ade80"],
        )
        st.plotly_chart(fig_donut, width="stretch")
    else:
        st.info("No class counts in evaluation results.")

    # Text length distribution (illustrative when we don't load raw data)
    st.markdown("#### Text length distribution")
    st.caption("Illustrative; actual distributions from training notebook.")
    lengths_fake, lengths_real = _sample_text_lengths()
    fig_hist = plotly_histogram_text_length(lengths_fake, lengths_real)
    st.plotly_chart(fig_hist, width="stretch")

    # Top TF-IDF features (from pipeline if available)
    st.markdown("---")
    st.markdown("#### Top TF-IDF features (Logistic Regression)")
    try:
        pipeline = load_model()
        names_fake, coefs_fake, names_real, coefs_real = get_lr_feature_importance(pipeline, top_n=15)
        if names_fake and names_real:
            left, right = st.columns(2)
            with left:
                fig_fake = plotly_top_tfidf_features(
                    names_fake, coefs_fake, "Words → Fake", color="#f87171", top_n=15
                )
                st.plotly_chart(fig_fake, width="stretch")
            with right:
                fig_real = plotly_top_tfidf_features(
                    names_real, coefs_real, "Words → Real", color="#22c55e", top_n=15
                )
                st.plotly_chart(fig_real, width="stretch")
        else:
            st.info("Pipeline has no LR coefficients. Train and add `model/pipeline.pkl`.")
    except FileNotFoundError:
        st.info("Load the trained model (`model/pipeline.pkl`) to show feature importance here.")
