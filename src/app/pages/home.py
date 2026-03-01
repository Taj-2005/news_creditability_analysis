"""Overview — Executive summary, big KPI cards, product pitch. Production-grade layout."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import (
    DATASET_NAME,
    get_dataset_size_str,
    get_expected_metrics,
    get_model_algorithm_display,
    PAGE_TITLE,
)


def render():
    page_header(
        "News Credibility Analyzer",
        "This system detects whether a news article is Fake or Real using machine learning. Trained on the Kaggle Fake and Real News dataset with classical NLP (TF-IDF) and interpretable classifiers."
    )

    # Problem statement
    st.markdown("#### What it does")
    st.markdown(
        "Misinformation spreads faster than manual fact-checking can scale. This app provides **automated binary classification** of news as **Fake** or **Real**, using a reproducible ML pipeline — interpretable models, no LLMs, no GPU."
    )
    st.info(
        "**Trained on:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). "
        "For the most accurate credibility results, use **US-based articles or news** that align with this dataset (English, similar style and topics)."
    )

    # KPI section: use best model metrics when available
    metrics = get_expected_metrics()
    if not metrics:
        st.warning(
            "Evaluation results not found. Run the notebook (with the dataset) or "
            "`python scripts/run_evaluation.py` to generate metrics, then restart the app."
        )
        st.stop()

    best_name = get_model_algorithm_display()
    lr = metrics.get(best_name) or metrics.get("Logistic Regression") or next(iter(metrics.values()), None)
    if not lr:
        st.warning("No model metrics found in evaluation results.")
        st.stop()

    st.markdown("#### Key performance metrics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Accuracy", f"{lr['Accuracy']:.2%}", help="Overall correct predictions")
    with k2:
        st.metric("F1 Score", f"{lr['F1 Score']:.2%}", help="Fake-class F1")
    with k3:
        st.metric("ROC-AUC", f"{lr['ROC-AUC']:.2%}", help="Ranking / discrimination")
    with k4:
        # CV F1 is added by scripts/run_evaluation.py; notebook may not include it
        cv = lr.get("CV F1")
        if isinstance(cv, (list, tuple)) and len(cv) >= 2:
            cv_mean, cv_std = float(cv[0]), float(cv[1])
            st.metric("5-Fold CV F1", f"{cv_mean:.2%} ± {cv_std:.2%}", help="Cross-validation stability")
        else:
            # Fallback: show from any model that has CV F1 (e.g. from run_evaluation.py)
            cv_fallback = None
            for m in (metrics or {}).values():
                if isinstance(m.get("CV F1"), (list, tuple)) and len(m["CV F1"]) >= 2:
                    cv_fallback = (float(m["CV F1"][0]), float(m["CV F1"][1]))
                    break
            if cv_fallback:
                st.metric("5-Fold CV F1", f"{cv_fallback[0]:.2%} ± {cv_fallback[1]:.2%}", help="Cross-validation (from evaluation script)")
            else:
                st.metric("5-Fold CV F1", "—", help="Run scripts/run_evaluation.py to compute")

    # Animated-style progress bars (visual only; Streamlit progress is 0–1)
    st.markdown("")
    st.markdown("**Metric levels**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.progress(float(lr["Accuracy"]), text="Accuracy")
    with c2:
        st.progress(float(lr["F1 Score"]), text="F1 Score")
    with c3:
        st.progress(float(lr["ROC-AUC"]), text="ROC-AUC")

    # Dataset summary
    st.markdown("---")
    st.markdown("#### Dataset summary")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Dataset", DATASET_NAME)
    with d2:
        st.metric("Articles", get_dataset_size_str())
    with d3:
        st.metric("Task", "Fake vs Real (binary)")
    st.caption(
        "Dataset: [Fake and Real News (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). "
        "Best results with US-based, English news that matches the training distribution."
    )

    # Product pitch
    st.markdown("---")
    st.markdown("#### Product pitch")
    st.markdown(
        "**Problem → Data → Model → Evaluation → Deployment → Live Testing**  \n"
        "This milestone delivers an end-to-end pipeline: from raw text to a credibility verdict, with stratified splits, class balancing, four compared models (Logistic Regression, Naive Bayes, Random Forest, SVM), and a Streamlit app suitable for demos and further agentic AI integration."
    )
