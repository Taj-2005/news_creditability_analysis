"""
Streamlit UI for News Credibility Analyzer.

Inference is fully independent from training: at runtime we only need
- model/pipeline.pkl (serialized sklearn Pipeline from training)
- clean_text() (same preprocessing contract as training)

Run: streamlit run app.py  or  streamlit run src/app/main.py
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is on path for "src" imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

from src.features.preprocessing import clean_text

# -----------------------------------------------------------------------------
# Logging (production reliability)
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("news_credibility_app")

# -----------------------------------------------------------------------------
# Constants (UI + model artifact paths)
# -----------------------------------------------------------------------------

# Model artifacts: store pipeline.pkl under <repo_root>/model/ or <cwd>/model/
# Inference depends only on this artifact and clean_text(); no training code at runtime.
MODEL_DIR_NAME = "model"
MODEL_FILENAME = "pipeline.pkl"

PAGE_TITLE = "News Credibility Analyzer"
PAGE_SUBTITLE = "Intelligent misinformation detection using classical NLP & machine learning"
MODEL_ALGORITHM = "Logistic Regression"
MODEL_FEATURES = "TF-IDF (unigrams + bigrams)"
DATASET_NAME = "BharatFakeNewsKosh"
DATASET_SIZE = "26,000+"

EXAMPLE_TEXTS = {
    "Real (credible)": (
        "Prime Minister Modi received an honorary doctorate from Oxford University "
        "in recognition of India's digital transformation initiatives and governance reforms."
    ),
    "Fake (misinformation)": (
        "Viral video claims to show a digitally edited flood in Rajasthan designed "
        "to mislead viewers about the actual situation. Fact-checkers have debunked the clip."
    ),
    "Short headline": (
        "Breaking: Scientists announce breakthrough in renewable energy storage technology."
    ),
}

MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 50_000

# -----------------------------------------------------------------------------
# Model logic (no UI)
# -----------------------------------------------------------------------------


def _model_base_dir() -> Path:
    """Resolve model directory in a deployment-safe way (no hardcoded local paths)."""
    file_based = repo_root / MODEL_DIR_NAME
    cwd_based = Path.cwd() / MODEL_DIR_NAME
    if (file_based / MODEL_FILENAME).exists():
        return file_based
    if (cwd_based / MODEL_FILENAME).exists():
        return cwd_based
    return file_based  # Expected location for error message


@st.cache_resource
def load_model():
    """
    Load the trained pipeline from model/pipeline.pkl (cached, environment-safe).
    Raises FileNotFoundError if the model artifact is missing.
    """
    import joblib

    base = _model_base_dir()
    model_path = base / MODEL_FILENAME
    if not model_path.exists():
        logger.error("Model file missing: %s", model_path)
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Ensure {MODEL_DIR_NAME}/{MODEL_FILENAME} exists (run training or add artifact to repo). See README."
        )
    try:
        pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return pipeline
    except Exception as e:
        logger.exception("Failed to load model from %s: %s", model_path, e)
        raise


def run_prediction(pipeline, raw_text: str) -> Tuple[int, float, float]:
    """
    Run model prediction on raw text. Core logic only; no UI.
    Caller should handle exceptions for production reliability.

    Returns:
        (prediction, fake_probability, real_probability)
        prediction: 0 = Real, 1 = Fake
    """
    cleaned = clean_text(raw_text)
    prediction = int(pipeline.predict([cleaned])[0])
    proba = pipeline.predict_proba([cleaned])[0]
    real_prob = float(proba[0])
    fake_prob = float(proba[1])
    return prediction, fake_prob, real_prob


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate user input. Returns (is_valid, error_message).
    Empty error_message means valid. Logs validation failures.
    """
    if not text or not text.strip():
        logger.debug("Validation failed: empty input")
        return False, "Please enter some text."
    t = text.strip()
    if len(t) < MIN_INPUT_LENGTH:
        logger.debug("Validation failed: input too short (len=%d)", len(t))
        return False, f"Text is too short (minimum {MIN_INPUT_LENGTH} characters)."
    if len(t) > MAX_INPUT_LENGTH:
        logger.debug("Validation failed: input too long (len=%d)", len(t))
        return False, f"Text exceeds maximum length ({MAX_INPUT_LENGTH:,} characters)."
    return True, ""


# -----------------------------------------------------------------------------
# UI components (no model logic)
# -----------------------------------------------------------------------------


def render_header() -> None:
    """Styled header section with title and subtitle."""
    st.markdown(
        """
        <div style="
            padding: 1rem 0 1.5rem 0;
            border-bottom: 1px solid rgba(49, 51, 63, 0.2);
            margin-bottom: 1.5rem;
        ">
            <h1 style="margin: 0; font-size: 2rem;">📰 News Credibility Analyzer</h1>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 1rem;">
                """ + PAGE_SUBTITLE + """
            </p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #9ca3af;">
                Project 11 · Milestone 1 · BharatFakeNewsKosh Dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_section() -> Tuple[str, bool]:
    """
    Render input area, example buttons, and action buttons.
    Returns (input_text, should_analyze).
    """
    # Pre-fill from example if a fill was requested (from session_state)
    if st.session_state.get("example_fill_key") and st.session_state["example_fill_key"] in EXAMPLE_TEXTS:
        st.session_state["main_input"] = EXAMPLE_TEXTS[st.session_state["example_fill_key"]]
        del st.session_state["example_fill_key"]

    st.markdown("### 📝 Input")
    input_text = st.text_area(
        "Paste a news article, headline, or claim to analyze",
        height=200,
        placeholder="Paste your text here... (e.g. headline, article excerpt, or social media post)",
        key="main_input",
        label_visibility="collapsed",
    )

    st.markdown("**Load example:**")
    ex1, ex2, ex3, _ = st.columns(4)
    with ex1:
        if st.button("Example: Real", key="ex_real", use_container_width=True):
            st.session_state["example_fill_key"] = "Real (credible)"
            st.rerun()
    with ex2:
        if st.button("Example: Fake", key="ex_fake", use_container_width=True):
            st.session_state["example_fill_key"] = "Fake (misinformation)"
            st.rerun()
    with ex3:
        if st.button("Example: Headline", key="ex_short", use_container_width=True):
            st.session_state["example_fill_key"] = "Short headline"
            st.rerun()

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Credibility", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["main_input"] = ""
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()

    return input_text or "", analyze_clicked


def render_output_section(prediction: int, fake_prob: float, real_prob: float) -> None:
    """Render verdict, metrics, and confidence bar. Color-coded: Green = Real, Red = Fake."""
    st.markdown("### 📊 Result")

    # Color-coded verdict
    if prediction == 1:
        verdict_text = "Likely **FAKE** / Misinformation"
        verdict_icon = "🔴"
        st.error(f"{verdict_icon} Verdict: {verdict_text}")
    else:
        verdict_text = "Likely **CREDIBLE** / Real"
        verdict_icon = "🟢"
        st.success(f"{verdict_icon} Verdict: {verdict_text}")

    # Metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fake probability", f"{fake_prob:.1%}")
    with col2:
        st.metric("Real probability", f"{real_prob:.1%}")
    with col3:
        confidence = fake_prob if prediction == 1 else real_prob
        st.metric("Model confidence", f"{confidence:.1%}")

    st.markdown("**Credibility risk score** (higher = more likely fake)")
    st.progress(fake_prob)

    # Model metadata inline
    st.caption(
        f"Algorithm: {MODEL_ALGORITHM} · Features: {MODEL_FEATURES} · "
        f"Trained on {DATASET_NAME} ({DATASET_SIZE} articles)"
    )


def render_about_model() -> None:
    """Expandable 'About Model' section."""
    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown(
            f"""
            - **Algorithm:** {MODEL_ALGORITHM} with **{MODEL_FEATURES}**
            - **Dataset:** {DATASET_NAME} — {DATASET_SIZE} fact-checked Indian news articles
            - **Task:** Binary classification (Fake vs Real)
            - **Preprocessing:** Lowercasing, URL/mention removal, stopword removal, WordNet lemmatization
            - **Output:** Probability score for "Fake"; higher value = higher risk of misinformation

            ⚠️ **Disclaimer:** This is an AI-assisted tool. Always verify important news with trusted sources.
            """
        )


def render_footer() -> None:
    """Footer caption."""
    st.divider()
    st.caption("Project 11 · Intelligent News Credibility Analysis · Milestone 1")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    render_header()

    # Layout: main content in a container; optional sidebar later
    main_container = st.container()
    with main_container:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Input section")
            input_text, analyze_clicked = render_input_section()

        with col_right:
            st.markdown("#### Output")
            if analyze_clicked:
                is_valid, err = validate_input(input_text)
                if not is_valid:
                    st.warning(f"⚠️ {err}")
                else:
                    try:
                        with st.spinner("Analyzing..."):
                            pipeline = load_model()
                            prediction, fake_prob, real_prob = run_prediction(
                                pipeline, input_text
                            )
                        st.session_state["last_result"] = (prediction, fake_prob, real_prob)
                        render_output_section(prediction, fake_prob, real_prob)
                    except FileNotFoundError as e:
                        logger.error("Model not found: %s", e)
                        st.error(
                            f"**Model not available.** {e}. "
                            f"Add `{MODEL_DIR_NAME}/{MODEL_FILENAME}` to the project or run training (see README)."
                        )
                    except Exception as e:
                        logger.exception("Prediction failed: %s", e)
                        st.error(
                            "**Prediction failed.** Something went wrong while analyzing the text. "
                            "Please try again or use different input."
                        )
            elif "last_result" in st.session_state:
                p, fp, rp = st.session_state["last_result"]
                render_output_section(p, fp, rp)
            else:
                st.info("👈 Enter text and click **Analyze Credibility** to see the result.")

    render_about_model()
    render_footer()


if __name__ == "__main__":
    main()
