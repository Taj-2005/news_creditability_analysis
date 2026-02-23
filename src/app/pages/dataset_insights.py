"""Dataset Intelligence — class distribution, text length, top TF-IDF. Plotly-only charts."""

import numpy as np
import streamlit as st

from src.app.components.ui import page_header
from src.app.core import DATASET_NAME, DATASET_SIZE, load_model
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_donut_class_distribution,
    plotly_histogram_text_length,
    plotly_top_tfidf_features,
)

# Approximate from README: 60.6% Fake, 39.4% Real
FAKE_COUNT = 15_800
REAL_COUNT = 10_400


def _sample_text_lengths(seed: int = 42) -> tuple:
    """Illustrative text length samples per class (no dataset load)."""
    rng = np.random.default_rng(seed)
    fake_len = rng.lognormal(5, 1, 500)
    real_len = rng.lognormal(5.2, 1, 500)
    return list(fake_len), list(real_len)


def render():
    page_header("Dataset Intelligence", f"Summary for **{DATASET_NAME}** ({DATASET_SIZE} articles)")

    # Top metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", "26,232")
    col2.metric("Fake (approx.)", "60.6%")
    col3.metric("Real (approx.)", "39.4%")

    # Class distribution donut
    st.markdown("#### Class distribution")
    fig_donut = plotly_donut_class_distribution(
        counts=[FAKE_COUNT, REAL_COUNT],
        labels=["Fake", "Real"],
        colors=["#f87171", "#4ade80"],
    )
    st.plotly_chart(fig_donut, width="stretch")

    # Text length distribution
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
