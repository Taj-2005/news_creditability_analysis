"""Architecture — system diagram, train vs inference, repo mapping. Card-based layout."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import MODEL_ALGORITHM, MODEL_FILENAME, MODEL_DIR_NAME


def render():
    page_header("System architecture", "End-to-end flow from data to deployment.")

    st.markdown("#### High-level workflow")
    st.markdown(
        """
        ```
        DATA SOURCE (Kaggle Fake and Real News)
            → dataset/Fake.csv + True.csv → merge, label (Fake=1, Real=0), title + text
            → Preprocess (lowercase, URLs/mentions removed, stopwords, lemmatize)
            → TF-IDF (unigrams + bigrams)
            → Train: Logistic Regression / Decision Tree (sklearn Pipeline)
            → Evaluate (stratified split, CV, ROC, confusion matrix)
            → Persist: model/pipeline.pkl

        INFERENCE (this app)
            → User input → same clean_text() → pipeline.predict / predict_proba
            → Verdict + probability
        ```
        """
    )

    st.markdown("#### Training vs inference")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training** (notebook / scripts)")
        st.markdown(
            """
            - `src/data/loader.py` — load dataset  
            - `src/features/preprocessing.py` — clean_text, prepare_text_column  
            - `src/models/pipelines.py` — build_lr_pipeline, build_dt_pipeline  
            - `src/evaluation/` — metrics, confusion matrix, ROC, comparison table  
            - Output: `model/pipeline.pkl` (single artifact)
            """
        )
    with col2:
        st.markdown("**Inference** (this dashboard)")
        st.markdown(
            f"""
            - Load `{MODEL_DIR_NAME}/{MODEL_FILENAME}` (cached)  
            - `src/features/preprocessing.py` — clean_text only (same contract as training)  
            - Pipeline: TF-IDF transform → {MODEL_ALGORITHM} predict/proba  
            - No training code at runtime
            """
        )

    st.markdown("#### Repository mapping (`src/`)")
    st.markdown(
        """
        | Folder | Role |
        |--------|------|
        | `src/data/` | Dataset loading, feature/target extraction |
        | `src/features/` | Text preprocessing (clean_text); shared train & inference |
        | `src/models/` | Pipeline definitions (LR, DT); used only at train time |
        | `src/evaluation/` | Metrics, ROC, confusion matrix, comparison table |
        | `src/app/` | Streamlit dashboard and pages |
        """
    )
