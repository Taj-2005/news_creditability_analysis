"""Model Comparison — ROC, PR, confusion matrix, metric bars, CV boxplot, feature importance. Plotly only."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import EXPECTED_METRICS, load_model
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_confusion_heatmap_reference,
    plotly_cv_boxplot_reference,
    plotly_metric_comparison_bar,
    plotly_pr_reference,
    plotly_roc_reference,
    plotly_top_tfidf_features,
)


def render():
    page_header(
        "Model comparison",
        "Logistic Regression vs Decision Tree on the same TF-IDF features and stratified split.",
    )

    # Model selector for single-model charts (ROC, PR, Confusion)
    model_names = list(EXPECTED_METRICS.keys())
    selected = st.radio("Show curves for", model_names, horizontal=True, key="model_select")

    # Row 1: ROC + PR
    c1, c2 = st.columns(2)
    with c1:
        m = EXPECTED_METRICS[selected]
        fig_roc = plotly_roc_reference(selected, m["ROC-AUC"])
        st.plotly_chart(fig_roc, width="stretch")
    with c2:
        fig_pr = plotly_pr_reference(
            selected, m["Precision"], m["Recall"], m["ROC-AUC"]
        )
        st.plotly_chart(fig_pr, width="stretch")

    # Row 2: Confusion matrix + Metric comparison bar
    c3, c4 = st.columns(2)
    with c3:
        fig_cm = plotly_confusion_heatmap_reference(
            m["Accuracy"], m["Precision"], m["Recall"], selected
        )
        st.plotly_chart(fig_cm, width="stretch")
    with c4:
        fig_bar = plotly_metric_comparison_bar(EXPECTED_METRICS)
        st.plotly_chart(fig_bar, width="stretch")

    # CV distribution boxplot
    st.markdown("#### Cross-validation F1 distribution")
    fig_cv = plotly_cv_boxplot_reference(EXPECTED_METRICS)
    st.plotly_chart(fig_cv, width="stretch")

    # Commentary
    st.markdown("---")
    st.markdown("#### Why Logistic Regression outperforms Decision Tree")
    st.markdown(
        "TF-IDF produces high-dimensional sparse features where linear separability is strong. "
        "Logistic Regression exploits this with a single linear boundary and benefits from L2 regularization. "
        "Decision Trees must build many axis-aligned splits to approximate the same boundary, which leads to overfitting and higher variance. "
        "For this task, LR is also more interpretable (coefficients per token) and is the model used in **Live Prediction Lab**."
    )

    # Feature importance (LR only)
    st.markdown("---")
    st.markdown("#### Feature importance (Logistic Regression)")
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
            st.info("Pipeline has no LR coefficients.")
    except FileNotFoundError:
        st.info("Load the trained model (`model/pipeline.pkl`) to show feature importance.")
