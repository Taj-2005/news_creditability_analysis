"""Model Comparison — ROC, PR, confusion matrix, metric bars, CV boxplot, feature importance. Plotly only."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import get_expected_metrics, load_model
from src.evaluation.results_loader import get_confusion_matrices
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_confusion_heatmap_from_matrix,
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
        "Compare all trained models (Logistic Regression, Naive Bayes, Random Forest, SVM) on the same TF-IDF features and stratified split.",
    )

    metrics = get_expected_metrics()
    if not metrics:
        st.warning(
            "Evaluation results not found. Run the notebook (with the dataset) or "
            "`python scripts/run_evaluation.py` to generate metrics."
        )
        st.stop()

    model_names = list(metrics.keys())
    selected = st.radio("Show curves for", model_names, horizontal=True, key="model_select")

    m = metrics.get(selected)
    if not m:
        st.stop()

    # Row 1: ROC + PR
    c1, c2 = st.columns(2)
    with c1:
        fig_roc = plotly_roc_reference(selected, m["ROC-AUC"])
        st.plotly_chart(fig_roc, width="stretch")
    with c2:
        fig_pr = plotly_pr_reference(
            selected, m["Precision"], m["Recall"], m["ROC-AUC"]
        )
        st.plotly_chart(fig_pr, width="stretch")

    # Row 2: Confusion matrix (real from artifact when available) + Metric comparison bar
    c3, c4 = st.columns(2)
    confusion_matrices = get_confusion_matrices()
    with c3:
        cm = confusion_matrices.get(selected) if confusion_matrices else None
        if cm and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
            fig_cm = plotly_confusion_heatmap_from_matrix(cm, selected)
        else:
            fig_cm = plotly_confusion_heatmap_reference(
                m["Accuracy"], m["Precision"], m["Recall"], selected
            )
        st.plotly_chart(fig_cm, width="stretch")
    with c4:
        fig_bar = plotly_metric_comparison_bar(metrics)
        st.plotly_chart(fig_bar, width="stretch")

    # CV distribution boxplot
    st.markdown("#### Cross-validation F1 distribution")
    fig_cv = plotly_cv_boxplot_reference(metrics)
    st.plotly_chart(fig_cv, width="stretch")

    # Commentary
    st.markdown("---")
    st.markdown("#### Model notes")
    st.markdown(
        "TF-IDF produces high-dimensional sparse features. Logistic Regression often performs well with a linear boundary and L2 regularization. "
        "Naive Bayes is fast and interpretable; Random Forest and SVM offer alternative decision boundaries. "
        "The **best model by F1** is saved and used in **Live Prediction Lab**."
    )

    # Feature importance (LR or linear models with coef_)
    st.markdown("---")
    st.markdown("#### Feature importance (linear models: LR, SVM)")
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
