"""Live Prediction Lab — large input, Predict button, result card with gauge and probability. Production UX."""

import logging
import streamlit as st

from src.app.components.ui import page_header
from src.app.core import (
    EXAMPLE_TEXTS,
    MODEL_ALGORITHM,
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
        "Paste a headline or article excerpt for a **Fake** / **Real** verdict and confidence score.",
    )

    # Large text input
    input_text = st.text_area(
        "Text to analyze",
        height=220,
        placeholder="Paste news headline or article text here...",
        key="live_input",
        label_visibility="collapsed",
    )

    # Example buttons: set pending example and rerun so next run sets live_input before widget
    st.markdown("**Load example**")
    c1, c2, c3, _ = st.columns(4)
    with c1:
        if st.button("Example: Real", key="ex_r"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Real (credible)"]
            st.rerun()
    with c2:
        if st.button("Example: Fake", key="ex_f"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Fake (misinformation)"]
            st.rerun()
    with c3:
        if st.button("Example: Headline", key="ex_h"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Short headline"]
            st.rerun()

    # Predict / Clear (Clear uses same pending pattern to avoid modifying live_input after widget)
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
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
        confidence = fake_prob if prediction == 1 else real_prob

        st.markdown("---")
        st.markdown("#### Result")

        # Color-coded verdict
        if prediction == 1:
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
                - **Algorithm:** {MODEL_ALGORITHM} with TF-IDF (unigrams + bigrams).
                - Text is preprocessed (lowercase, stopwords, lemmatization) then vectorized and classified.
                - This is an AI-assisted tool; verify important news with trusted sources.
                """
            )
