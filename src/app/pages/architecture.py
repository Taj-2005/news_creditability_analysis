"""Architecture — system diagram (Mermaid), train vs inference, repo mapping. Card-based layout."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import get_model_algorithm_display, MODEL_FILENAME, MODEL_DIR_NAME


def _mermaid_html(diagram: str, height: int = 280) -> str:
    """Render a Mermaid diagram via Mermaid.js (loaded from CDN)."""
    return f"""
    <div class="mermaid" style="text-align: center; min-height: {height}px;">
{diagram}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'base', themeVariables: {{ primaryColor: '#2563eb', lineColor: '#64748b' }} }});</script>
    """


def render():
    page_header("System architecture", "End-to-end flow from data to deployment.")

    st.markdown("#### High-level workflow (Mermaid)")

    training_mermaid = """
flowchart LR
    subgraph DATA["Data"]
        A[Fake.csv] --> M[Merge]
        B[True.csv] --> M
        M --> L[Label 0/1]
    end
    subgraph PREP["Preprocess"]
        L --> C[clean_text]
        C --> D[TF-IDF]
    end
    subgraph MODELS["Models"]
        D --> E[LR]
        D --> F[NB]
        D --> G[RF]
        D --> H[SVM]
        E --> I[Best F1]
        F --> I
        G --> I
        H --> I
    end
    I --> O[pipeline.pkl]
    """
    inference_mermaid = """
flowchart LR
    U[User Input] --> C[clean_text]
    C --> P[Pipeline]
    P --> V[Verdict]
    """

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Training pipeline**")
        st.components.v1.html(_mermaid_html(training_mermaid, height=320), height=350, scrolling=False)
    with col_m2:
        st.markdown("**Inference pipeline**")
        st.components.v1.html(_mermaid_html(inference_mermaid, height=180), height=220, scrolling=False)

    st.markdown("---")
    st.markdown("#### End-to-end flow (text)")
    st.markdown(
        """
        **DATA SOURCE** — [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
        `dataset/Fake.csv` + `True.csv` → merge, label (Fake=0, Real=1) → **Preprocess** (lowercase, stopwords, lemmatize) → **TF-IDF** → **Train**: LR, Naive Bayes, Random Forest, SVM → **Persist**: `model/pipeline.pkl`.

        **INFERENCE**: User input → same `clean_text()` → pipeline → Verdict + probability.
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
            - `src/models/pipelines.py` — build_lr_pipeline, build_nb_pipeline, build_rf_pipeline, build_svm_pipeline  
            - `src/evaluation/` — metrics, confusion matrix, ROC, comparison table  
            - Output: `model/pipeline.pkl` (best model), `model/evaluation_results.json`
            """
        )
    with col2:
        st.markdown("**Inference** (this dashboard)")
        best_algo = get_model_algorithm_display()
        st.markdown(
            f"""
            - Load `{MODEL_DIR_NAME}/{MODEL_FILENAME}` (cached)  
            - `src/features/preprocessing.py` — clean_text only (same contract as training)  
            - Pipeline: TF-IDF transform → {best_algo} predict/proba  
            - No training code at runtime
            """
        )

    st.markdown("#### Repository mapping (`src/`)")
    st.markdown(
        """
        | Folder | Role |
        |--------|------|
        | `src/data/` | Dataset loading (Fake.csv, True.csv), feature/target extraction |
        | `src/features/` | Text preprocessing (clean_text); shared train & inference |
        | `src/models/` | Pipeline definitions (LR, NB, RF, SVM); used at train time |
        | `src/evaluation/` | Metrics, ROC, confusion matrix, results_loader, plotly_viz |
        | `src/app/` | Streamlit dashboard, pages, core (load_model, prediction) |
        """
    )

    st.markdown("---")
    st.markdown("#### Repo structure (Mermaid)")
    repo_mermaid = """
flowchart TB
    subgraph ROOT["Repo root"]
        APP[app.py]
        NB[notebook]
        SCRIPT[run_evaluation.py]
        DS[dataset/]
        MD[model/pipeline.pkl]
    end
    subgraph SRC["src/"]
        DATA[data/loader]
        FEAT[features/preprocessing]
        MOD[models/pipelines]
        EVAL[evaluation/]
        APP_P[app/]
    end
    DS --> DATA
    DATA --> FEAT
    FEAT --> MOD
    MOD --> EVAL
    EVAL --> MD
    MD --> APP_P
    NB --> DATA
    SCRIPT --> DATA
    APP --> APP_P
    """
    st.components.v1.html(_mermaid_html(repo_mermaid, height=320), height=360, scrolling=False)
