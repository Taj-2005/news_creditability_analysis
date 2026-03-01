"""Live Prediction Lab — large input, Predict button, result card with gauge and probability. Production UX."""

import logging
import streamlit as st

from src.app.components.ui import page_header
from src.app.core import (
    EXAMPLE_TEXTS,
    get_model_algorithm_display,
    MODEL_DIR_NAME,
    MODEL_FILENAME,
    load_model,
    run_prediction,
    validate_input,
)
from src.evaluation.plotly_viz import plotly_confidence_gauge

logger = logging.getLogger("news_credibility_app")


def render():
    # Apply pending example/clear before widget is created (Streamlit forbids setting widget key after creation)
    if "live_pending_example" in st.session_state:
        st.session_state["live_input"] = st.session_state.pop("live_pending_example")
        if "live_result" in st.session_state:
            del st.session_state["live_result"]

    page_header(
        "Live prediction lab",
        "Paste a headline or article excerpt. The system detects whether the news is Fake or Real using machine learning. Get a verdict and confidence score.",
    )

    # Large text input
    input_text = st.text_area(
        "Text to analyze",
        height=220,
        placeholder="Paste news headline or article text here, then click Analyze...",
        key="live_input",
        label_visibility="collapsed",
    )

    # Sample news: click to auto-fill input
    st.markdown("**Sample news** (click to try)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Fake example", key="ex_fake"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Fake example"]
            st.rerun()
    with c2:
        if st.button("Real example", key="ex_real"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Real example"]
            st.rerun()
    with c3:
        if st.button("Government policy...", key="ex_r"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Government policy"]
            st.rerun()
    with c4:
        if st.button("Scientists breakthrough...", key="ex_f"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Scientists breakthrough"]
            st.rerun()
    with c5:
        if st.button("Conspiracy / weather...", key="ex_h"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Conspiracy sample"]
            st.rerun()

    # Analyze / Clear — force primary button text white (override theme)
    st.markdown(
        "<style>"
        ".block-container button[kind=\"primary\"], .block-container button[kind=\"primary\"] *, "
        "div[data-testid=\"column\"] button[kind=\"primary\"], div[data-testid=\"column\"] button[kind=\"primary\"] * "
        "{ color: #ffffff !important; fill: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        predict_clicked = st.button("Analyze credibility", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear", use_container_width=True):
            st.session_state["live_pending_example"] = ""
            st.rerun()

    if predict_clicked:
        is_valid, err = validate_input(input_text)
        if not is_valid:
            st.warning(err)
        else:
            try:
                with st.spinner("Analyzing..."):
                    pipeline = load_model()
                    prediction, fake_prob, real_prob = run_prediction(pipeline, input_text)
                st.session_state["live_result"] = (prediction, fake_prob, real_prob)
            except FileNotFoundError:
                logger.error("Model not found")
                st.error(
                    f"Model not available. Add `{MODEL_DIR_NAME}/{MODEL_FILENAME}` or run training (see README)."
                )
            except Exception as e:
                logger.exception("Prediction failed: %s", e)
                st.error("Prediction failed. Please try again or use different input.")

    # Result card: verdict, gauge, probability breakdown, explanation
    if "live_result" in st.session_state:
        prediction, fake_prob, real_prob = st.session_state["live_result"]
        confidence = fake_prob if prediction == 0 else real_prob

        st.markdown("---")
        st.markdown("#### Result")

        # Color-coded verdict (0 = Fake, 1 = Real)
        if prediction == 0:
            st.markdown(
                '<p style="font-size: 1.1rem; padding: 0.75rem 1rem; border-radius: 8px; background: #fef2f2; color: #b91c1c; font-weight: 600;">Verdict: Likely FAKE / Misinformation</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="font-size: 1.1rem; padding: 0.75rem 1rem; border-radius: 8px; background: #f0fdf4; color: #15803d; font-weight: 600;">Verdict: Likely CREDIBLE / Real</p>',
                unsafe_allow_html=True,
            )

        # Gauge + probability breakdown in one row
        col_gauge, col_probs = st.columns([1, 2])
        with col_gauge:
            fig_gauge = plotly_confidence_gauge(confidence, "Model confidence", height=280)
            st.plotly_chart(fig_gauge, width="stretch")
        with col_probs:
            st.markdown("**Probability breakdown**")
            p1, p2 = st.columns(2)
            p1.metric("Fake (misinformation)", f"{fake_prob:.1%}")
            p2.metric("Real (credible)", f"{real_prob:.1%}")
            st.markdown("**Credibility risk** (higher = more likely fake)")
            st.progress(float(fake_prob))

        with st.expander("Model explanation"):
            st.markdown(
                f"""
                - **Algorithm:** {get_model_algorithm_display()} with TF-IDF (unigrams + bigrams).
                - Text is preprocessed (lowercase, stopwords, lemmatization) then vectorized and classified.
                - This is an AI-assisted tool; verify important news with trusted sources.
                """
            )
