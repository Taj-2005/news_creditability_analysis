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


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        .stApp { background: #ffffff; }

        .lp-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .lp-heading {
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 20px 0;
            line-height: 1.3;
        }
        .lp-rule {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 36px 0;
        }

        /* ── Textarea override ── */
        .stTextArea textarea {
            font-family: 'DM Mono', monospace !important;
            font-size: 13px !important;
            font-weight: 300 !important;
            color: #0f172a !important;
            background: #fafafa !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
            padding: 16px !important;
            line-height: 1.7 !important;
            box-shadow: none !important;
            resize: vertical !important;
        }
        .stTextArea textarea:focus {
            border-color: #0f172a !important;
            box-shadow: none !important;
            outline: none !important;
        }
        .stTextArea textarea::placeholder { color: #cbd5e1 !important; }
        .stTextArea label { display: none !important; }

        /* ── Example chips ── */
        .example-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 10px;
        }

        /* Streamlit button resets for example chips */
        div[data-testid="stHorizontalBlock"] .stButton button {
            font-family: 'DM Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 400 !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 20px !important;
            color: #475569 !important;
            padding: 5px 14px !important;
            transition: all 0.15s !important;
            white-space: nowrap !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton button:hover {
            background: #f1f5f9 !important;
            border-color: #cbd5e1 !important;
            color: #0f172a !important;
        }

        /* ── Primary analyze button ── */
        .stButton button[kind="primary"],
        .stButton button[kind="primary"] p,
        .stButton button[kind="primary"] div,
        .stButton button[kind="primary"] span {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em !important;
            background: #f1f5f9 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            padding: 10px 20px !important;
        }
        .stButton button[kind="primary"]:hover,
        .stButton button[kind="primary"]:hover p,
        .stButton button[kind="primary"]:hover div,
        .stButton button[kind="primary"]:hover span {
            background: #e2e8f0 !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }

        /* ── Secondary clear button ── */
        .stButton button[kind="secondary"] {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #64748b !important;
        }

        /* ── Verdict banners ── */
        .verdict-fake {
            display: flex;
            align-items: center;
            gap: 14px;
            background: #fff5f5;
            border: 1px solid #fecaca;
            border-left: 4px solid #f87171;
            border-radius: 0 10px 10px 0;
            padding: 18px 22px;
            margin: 8px 0 20px;
        }
        .verdict-real {
            display: flex;
            align-items: center;
            gap: 14px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 4px solid #4ade80;
            border-radius: 0 10px 10px 0;
            padding: 18px 22px;
            margin: 8px 0 20px;
        }
        .verdict-icon {
            font-size: 22px;
            flex-shrink: 0;
        }
        .verdict-text-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .verdict-fake .verdict-text-label { color: #f87171; }
        .verdict-real .verdict-text-label { color: #4ade80; }
        .verdict-text-main {
            font-family: 'Fraunces', serif;
            font-size: 20px;
            font-weight: 300;
            line-height: 1.2;
        }
        .verdict-fake .verdict-text-main { color: #b91c1c; }
        .verdict-real .verdict-text-main { color: #15803d; }

        /* ── Probability breakdown ── */
        .prob-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        .prob-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 10px;
            padding: 16px 18px;
        }
        .prob-card-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .prob-card-value {
            font-family: 'Fraunces', serif;
            font-size: 28px;
            font-weight: 300;
            color: #0f172a;
            line-height: 1;
        }

        /* ── Risk bar ── */
        .risk-section {
            margin-top: 4px;
        }
        .risk-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .risk-track {
            height: 6px;
            background: #f1f5f9;
            border-radius: 6px;
            overflow: hidden;
        }
        .risk-fill-fake {
            height: 100%;
            background: #f87171;
            border-radius: 6px;
            transition: width 0.5s ease;
        }
        .risk-fill-real {
            height: 100%;
            background: #4ade80;
            border-radius: 6px;
            transition: width 0.5s ease;
        }

        /* ── Expander override ── */
        .streamlit-expanderHeader {
            font-family: 'DM Mono', monospace !important;
            font-size: 11px !important;
            color: #64748b !important;
            letter-spacing: 0.08em !important;
        }
        .streamlit-expanderContent {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            color: #475569 !important;
            line-height: 1.7 !important;
        }

        /* ── Info / warn notices ── */
        .info-notice {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #475569;
            line-height: 1.6;
        }
        .warn-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 10px;
            padding: 14px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #92400e;
            line-height: 1.6;
        }
        .error-box {
            background: #fff5f5;
            border: 1px solid #fecaca;
            border-radius: 10px;
            padding: 14px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #b91c1c;
            line-height: 1.6;
        }

        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render():
    _inject_styles()

    # Apply pending example/clear before widget is created
    if "live_pending_example" in st.session_state:
        st.session_state["live_input"] = st.session_state.pop("live_pending_example")
        if "live_result" in st.session_state:
            del st.session_state["live_result"]

    page_header(
        "Live prediction lab",
        "Paste a headline or article excerpt — get a verdict and confidence score instantly. "
        "For the full agent (ML + RAG + Groq report), open Deep Analysis in the sidebar.",
    )

    # ── Section 1: Input ──
    st.markdown('<p class="lp-eyebrow">01 / Input</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="lp-heading">Text to analyze</h2>', unsafe_allow_html=True)

    input_text = st.text_area(
        "Text to analyze",
        height=200,
        placeholder="Paste a news headline or article excerpt here…",
        key="live_input",
        label_visibility="collapsed",
    )

    # ── Example chips ──
    st.markdown('<p class="example-label">Try a sample</p>', unsafe_allow_html=True)

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
        if st.button("Gov. policy", key="ex_r"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Government policy"]
            st.rerun()
    with c4:
        if st.button("Science story", key="ex_f"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Scientists breakthrough"]
            st.rerun()
    with c5:
        if st.button("Conspiracy", key="ex_h"):
            st.session_state["live_pending_example"] = EXAMPLE_TEXTS["Conspiracy sample"]
            st.rerun()

    # ── Action buttons ──
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    # Style the Analyze credibility button with black text
    st.markdown(
        """<style>
        div[data-testid="stButton"] > button[kind="secondary"] {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em !important;
            background: #f1f5f9 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: #e2e8f0 !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        predict_clicked = st.button("Analyze credibility", use_container_width=True)
    with col_btn2:
        if st.button("Clear", use_container_width=True):
            st.session_state["live_pending_example"] = ""
            st.rerun()

    # ── Run prediction ──
    if predict_clicked:
        is_valid, err = validate_input(input_text)
        if not is_valid:
            st.markdown(f'<div class="warn-box">⚠ {err}</div>', unsafe_allow_html=True)
        else:
            try:
                with st.spinner("Analyzing…"):
                    pipeline = load_model()
                    prediction, fake_prob, real_prob = run_prediction(pipeline, input_text)
                st.session_state["live_result"] = (prediction, fake_prob, real_prob)
            except FileNotFoundError:
                logger.error("Model not found")
                st.markdown(
                    f'<div class="error-box">✕ Model not available. Add '
                    f'<code>{MODEL_DIR_NAME}/{MODEL_FILENAME}</code> or run training (see README).</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                logger.exception("Prediction failed: %s", e)
                st.markdown(
                    '<div class="error-box">✕ Prediction failed. Please try again or use different input.</div>',
                    unsafe_allow_html=True,
                )

    # ── Result card ──
    if "live_result" in st.session_state:
        prediction, fake_prob, real_prob = st.session_state["live_result"]
        confidence = fake_prob if prediction == 0 else real_prob
        is_fake = prediction == 0

        st.markdown('<hr class="lp-rule">', unsafe_allow_html=True)
        st.markdown('<p class="lp-eyebrow">02 / Result</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="lp-heading">Verdict</h2>', unsafe_allow_html=True)

        # Verdict banner
        if is_fake:
            st.markdown(
                '<div class="verdict-fake">'
                '<span class="verdict-icon">✕</span>'
                '<div>'
                '<div class="verdict-text-label">Classification</div>'
                '<div class="verdict-text-main">Likely Fake · Misinformation</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-real">'
                '<span class="verdict-icon">✓</span>'
                '<div>'
                '<div class="verdict-text-label">Classification</div>'
                '<div class="verdict-text-main">Likely Credible · Real News</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        # Gauge + breakdown side by side
        col_gauge, col_probs = st.columns([1, 2], gap="large")

        with col_gauge:
            fig_gauge = plotly_confidence_gauge(confidence, "Model confidence", height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_probs:
            bar_width = fake_prob if is_fake else real_prob
            risk_label = "Credibility risk · higher = more likely fake" if is_fake else "Credibility score · higher = more likely real"
            bar_class = "risk-fill-fake" if is_fake else "risk-fill-real"

            st.markdown(
                f"""
                <div class="prob-grid">
                <div class="prob-card">
                    <div class="prob-card-label">Fake probability</div>
                    <div class="prob-card-value">{fake_prob:.1%}</div>
                </div>
                <div class="prob-card">
                    <div class="prob-card-label">Real probability</div>
                    <div class="prob-card-value">{real_prob:.1%}</div>
                </div>
                </div>
                <div class="risk-section">
                <div class="risk-label">{risk_label}</div>
                <div class="risk-track">
                    <div class="{bar_class}"
                        style="width:{bar_width*100:.1f}%"></div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Explanation expander
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.expander("Model explanation"):
            algo = get_model_algorithm_display()
            st.markdown(
                f"""
                <div style="font-family:'DM Mono',monospace;font-size:12px;color:#475569;line-height:1.8;">
                  <span style="color:#0f172a;font-weight:500;">Algorithm</span> &nbsp;—&nbsp;
                  {algo} with TF-IDF (unigrams + bigrams)<br>
                  <span style="color:#0f172a;font-weight:500;">Preprocessing</span> &nbsp;—&nbsp;
                  lowercase · stopwords removed · lemmatized · vectorized<br>
                  <span style="color:#0f172a;font-weight:500;">Caveat</span> &nbsp;—&nbsp;
                  AI-assisted tool. Verify important news with trusted primary sources.
                </div>
                """,
                unsafe_allow_html=True,
            )