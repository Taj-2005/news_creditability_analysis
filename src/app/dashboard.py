"""
News Credibility AI — production-grade multi-page dashboard.
Pure white SaaS theme, wide layout, sidebar navigation, Plotly-only charts.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

from src.app.components.styles import inject_app_css
from src.app.core import get_model_algorithm_display, PAGE_TITLE
from src.app.pages import (
    architecture,
    dataset_insights,
    home,
    live_prediction,
    model_compare,
)

PAGES = {
    "Overview":              ("Executive summary & KPIs",        home.render),
    "Dataset Intelligence":  ("Class distribution & stats",      dataset_insights.render),
    "Model Comparison":      ("LR vs DT metrics & charts",       model_compare.render),
    "Live Prediction Lab":   ("Try the model",                   live_prediction.render),
    "Architecture":          ("Pipeline & repo mapping",         architecture.render),
}

APP_VERSION = "1.0.0"

# Icon map for each page
PAGE_ICONS = {
    "Overview":             "○",
    "Dataset Intelligence": "○",
    "Model Comparison":     "○",
    "Live Prediction Lab":  "○",
    "Architecture":         "○",
}


def _inject_sidebar_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        /* ── Sidebar background ── */
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] > div {
            padding: 28px 20px 20px !important;
        }

        /* ── Brand block ── */
        .sb-brand-title {
            font-family: 'Fraunces', serif;
            font-size: 15px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 3px 0;
            line-height: 1.3;
        }
        .sb-brand-sub {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 300;
            color: #94a3b8;
            letter-spacing: 0.06em;
            margin: 0;
        }

        /* ── Divider ── */
        .sb-rule {
            border: none;
            border-top: 1px solid #f1f5f9;
            margin: 16px 0;
        }

        /* ── Nav label ── */
        .sb-nav-label {
            font-family: 'DM Mono', monospace;
            font-size: 9px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #cbd5e1;
            margin-bottom: 8px;
        }

        /* ── Radio nav items ── */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            font-weight: 400 !important;
            color: #64748b !important;
            padding: 6px 8px !important;
            border-radius: 6px !important;
            transition: background 0.15s !important;
            display: flex !important;
            align-items: center !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: #f8fafc !important;
            color: #0f172a !important;
        }
        /* Hide default radio circle */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }
        /* Active nav item */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
            background: #f1f5f9 !important;
            color: #0f172a !important;
            font-weight: 500 !important;
        }

        /* ── Meta block ── */
        .sb-meta {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 300;
            color: #94a3b8;
            line-height: 1.8;
        }
        .sb-meta strong {
            color: #64748b;
            font-weight: 400;
        }
        .sb-meta a {
            color: #64748b;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        /* ── Version badge ── */
        .sb-version {
            display: inline-block;
            font-family: 'DM Mono', monospace;
            font-size: 9px;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #94a3b8;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 2px 10px;
            margin-top: 4px;
        }

        /* ── Global app font override ── */
        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run():
    st.set_page_config(
        layout="wide",
        page_title=PAGE_TITLE,
        page_icon="📰",
        initial_sidebar_state="expanded",
    )
    inject_app_css()
    _inject_sidebar_styles()

    with st.sidebar:
        # Brand
        st.markdown(
            """
            <p class="sb-brand-title">News Credibility Analyzer</p>
            <p class="sb-brand-sub">Fake vs Real news detection</p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)

        # Nav
        st.markdown('<p class="sb-nav-label">Navigate</p>', unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            list(PAGES.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)

        # Meta
        model_algo = get_model_algorithm_display()
        st.markdown(
            f"""
            <div class="sb-meta">
                <strong>Model</strong> &nbsp;·&nbsp; {model_algo}<br>
                <strong>Dataset</strong> &nbsp;·&nbsp;
                <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
                   target="_blank">Fake and Real News</a>
            </div>
            <span class="sb-version">v{APP_VERSION}</span>
            """,
            unsafe_allow_html=True,
        )

    # Render selected page
    PAGES[page][1]()


if __name__ == "__main__":
    run()